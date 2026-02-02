# -*- coding: utf-8 -*-
"""
AlphaCFG 独立训练模块 - 完整版（不依赖main01.py）
包含所有必要的算法实现

用法:
    python train_alpha_cfg_standalone.py --data data.pkl.gz --output results --episodes 100
    继承于main01.py
"""
import numpy as np
import pandas as pd
import pickle
import gzip
import warnings
import traceback
import argparse
import os
import json
import random
import multiprocessing as mp
import atexit
from collections import deque, defaultdict, OrderedDict
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools

warnings.filterwarnings('ignore')


def _is_main_process():
    try:
        return mp.current_process().name == "MainProcess"
    except Exception:
        return True

# PyTorch相关库（可选）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
    if _is_main_process():
        print(f"✅ PyTorch版本: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    # 为了让“无 torch 环境”也能 import 本文件（即使不启用 PPO），提供最小占位符
    torch = None  # type: ignore
    class _NNStub:  # noqa: N801
        Module = object
    nn = _NNStub()  # type: ignore
    F = None  # type: ignore
    optim = None  # type: ignore
    Categorical = None  # type: ignore
    if _is_main_process():
        print("⚠️ PyTorch不可用，将使用简化模式")

if _is_main_process():
    print("=" * 80)
    print("AlphaCFG独立训练模块 v2.0")
    print("=" * 80)


# ==================== 0. 多进程辅助模块 (新增) ====================
# 全局变量，用于在Worker进程中持有只读数据，避免每次调用都序列化传输
_worker_data = None
_worker_grammar = None
_worker_cfg = None


def _init_worker(data_pool, grammar_args, runtime_cfg=None):
    """Worker进程初始化函数"""
    global _worker_data, _worker_grammar, _worker_cfg
    import os
    pid = os.getpid()
    # 添加调试信息
    print(f"[Worker-{pid}] 初始化开始...")
    _worker_data = data_pool
    print(f"[Worker-{pid}] 数据加载完成，keys: {list(data_pool.keys()) if isinstance(data_pool, dict) else 'N/A'}")
    # 重建轻量级语法对象
    _worker_grammar = AlphaGrammar(**grammar_args)
    _worker_cfg = runtime_cfg or {}
    print(f"[Worker-{pid}] 初始化完成！")


def _eval_task(expr):
    """Worker进程执行函数"""
    global _worker_data, _worker_grammar
    import os
    pid = os.getpid()
    try:
        # 添加调试：确认 Worker 在执行
        # print(f"[Worker-{pid}] 执行任务: {expr}")
        
        # 检查全局变量是否已初始化
        if _worker_data is None or _worker_grammar is None:
            raise RuntimeError(f"Worker-{pid} 未正确初始化: data={_worker_data is not None}, grammar={_worker_grammar is not None}")
        
        # 直接调用语法对象的内部评估
        res = _worker_grammar.evaluate_expression(expr, _worker_data)
        return res
    except Exception as e:
        import traceback
        print(f"[Worker-{pid}] 任务失败: {e}")
        print(f"[Worker-{pid}] 表达式: {expr}")
        traceback.print_exc()
        return None


def _standardize_factor_df(df):
    """Worker 侧/主进程侧通用：对因子做横截面标准化，尽量保持 float32。"""
    try:
        if df is None or getattr(df, "empty", True):
            return df
        if isinstance(df, pd.DataFrame):
            df = df.replace([np.inf, -np.inf], np.nan)
            std = df.std(axis=1).replace(0, 1)
            mean = df.mean(axis=1)
            df = df.sub(mean, axis=0).div(std, axis=0)
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            df = df.clip(-3, 3)
            try:
                df = df.astype(np.float32, copy=False)
            except Exception:
                pass
        return df
    except Exception:
        try:
            return df.fillna(0) if df is not None else None
        except Exception:
            return None


def _calc_ic_worker(factor_df):
    """
    Worker 侧计算 IC/ICIR，避免把巨大 factor_df 通过 IPC 传回主进程。
    返回: (ic_mean, icir, effective_days)
    """
    global _worker_data, _worker_cfg
    if factor_df is None or getattr(factor_df, "empty", True):
        return 0.0, 0.0, 0

    period = int((_worker_cfg or {}).get('ic_period', 5) or 5)
    min_dates = int((_worker_cfg or {}).get('min_dates', 20) or 20)
    min_stocks = int((_worker_cfg or {}).get('min_stocks', 10) or 10)

    # 获取收益率数据（优先用预先计算好的 return_xxd）
    return_key = f"return_{period}d"
    returns = None
    try:
        if isinstance(_worker_data, dict) and return_key in _worker_data:
            returns = _worker_data[return_key]
    except Exception:
        returns = None

    if returns is None or getattr(returns, "empty", True):
        # fallback: close 计算
        try:
            close_df = _worker_data.get('close', None) if isinstance(_worker_data, dict) else None
            if close_df is None or getattr(close_df, "empty", True):
                return 0.0, 0.0, 0
            returns = close_df.pct_change(period).shift(-period)
        except Exception:
            return 0.0, 0.0, 0

    try:
        common_dates = factor_df.index.intersection(returns.index)
        if len(common_dates) < min_dates:
            return 0.0, 0.0, 0
        common_stocks = factor_df.columns.intersection(returns.columns)
        if len(common_stocks) < min_stocks:
            return 0.0, 0.0, 0

        factor_aligned = factor_df.loc[common_dates, common_stocks]
        returns_aligned = returns.loc[common_dates, common_stocks]
    except Exception:
        return 0.0, 0.0, 0

    daily_ics = []
    # 注意：这里仍是逐日计算 Spearman（与主进程逻辑一致），但跑在多进程里可并行提速
    for date in common_dates:
        fc = None
        rc = None
        try:
            factor_vals = factor_aligned.loc[date]
            return_vals = returns_aligned.loc[date]
            mask = factor_vals.notna() & return_vals.notna()
            if mask.sum() < max(5, min_stocks // 2):
                continue
            fc = factor_vals[mask]
            rc = return_vals[mask]
            ic = fc.corr(rc, method='spearman')
            if ic is not None and (not np.isnan(ic)):
                daily_ics.append(float(ic))
        except Exception:
            # fallback Pearson（尽量兼容）
            try:
                if fc is not None and rc is not None:
                    ic = fc.corr(rc)
                    if ic is not None and (not np.isnan(ic)):
                        daily_ics.append(float(ic))
            except Exception:
                continue

    effective_days = len(daily_ics)
    if effective_days < 10:
        return 0.0, 0.0, effective_days

    ic_mean = float(np.mean(daily_ics))
    ic_std = float(np.std(daily_ics))
    icir = float(ic_mean / (ic_std + 1e-10))
    return ic_mean, icir, effective_days


def _eval_task_ic(expr):
    """Worker 侧：计算 (expr -> ic/icir/days)，只回传小结果。"""
    global _worker_data, _worker_grammar
    import os
    pid = os.getpid()
    try:
        # 检查全局变量是否已初始化
        if _worker_data is None or _worker_grammar is None:
            raise RuntimeError(f"Worker-{pid} 未正确初始化")
        
        factor_df = _worker_grammar.evaluate_expression(expr, _worker_data, verbose=False)
        factor_df = _standardize_factor_df(factor_df)
        ic, icir, days = _calc_ic_worker(factor_df)
        return (expr, ic, icir, days)
    except Exception as e:
        import traceback
        print(f"[Worker-{pid}] IC计算失败: {e}")
        print(f"[Worker-{pid}] 表达式: {expr}")
        traceback.print_exc()
        return (expr, 0.0, 0.0, 0)


def _test_worker_pid():
    """测试函数：返回 Worker 进程的 PID（用于验证进程池是否就绪）"""
    import os
    return os.getpid()


# ==================== 1. 数据加载和预处理 ====================

def load_data(input_path):
    """从文件加载数据"""
    print(f"\n📂 加载数据从: {input_path}")
    
    try:
        with gzip.open(input_path, 'rb') as f:
            data_package = pickle.load(f)
        
        print(f"✅ 数据加载成功！")
        if 'metadata' in data_package:
            meta = data_package['metadata']
            print(f"   股票数: {meta['n_stocks']}")
            print(f"   交易日数: {meta['n_days']}")
            print(f"   日期范围: {meta['date_range']}")
            print(f"   特征数: {len(meta['features'])}")
        
        return data_package['data_pool'], data_package['common_dates']
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        traceback.print_exc()
        return None, None


def load_and_merge_chunks(data_dir):
    """
    从目录中加载并合并所有分块文件
    
    参数:
        data_dir: str, 包含分块文件的目录路径
    
    返回:
        data_pool: dict, 合并后的数据池
        common_dates: list, 合并后的日期列表
    """
    print(f"\n📂 从目录加载并合并分块数据...")
    print(f"   目录: {data_dir}")
    
    try:
        # 查找所有匹配的分块文件
        import glob
        chunk_pattern = os.path.join(data_dir, "preprocessed_*.pkl.gz")
        chunk_files = sorted(glob.glob(chunk_pattern))
        
        if not chunk_files:
            print(f"❌ 未找到分块文件: {chunk_pattern}")
            return None, None
        
        print(f"   找到 {len(chunk_files)} 个分块文件")
        
        # 初始化合并容器
        merged_data_pool = {}
        all_dates = []
        
        # 逐个加载并合并
        for i, chunk_file in enumerate(chunk_files, 1):
            print(f"   [{i}/{len(chunk_files)}] 加载: {os.path.basename(chunk_file)}", end='')
            
            with gzip.open(chunk_file, 'rb') as f:
                chunk_package = pickle.load(f)
            
            chunk_data = chunk_package['data_pool']
            chunk_dates = chunk_package['common_dates']
            
            print(f" - {len(chunk_dates)} 天")
            
            # 合并日期
            all_dates.extend(chunk_dates)
            
            # 合并每个特征的数据
            for feature_name, feature_df in chunk_data.items():
                if feature_df is None or feature_df.empty:
                    continue

                # ✅ [优化] 统一转 float32，降低内存与加速后续计算
                try:
                    if isinstance(feature_df, pd.DataFrame) and feature_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
                        feature_df = feature_df.astype(np.float32, copy=False)
                except Exception:
                    pass
                
                if feature_name not in merged_data_pool:
                    merged_data_pool[feature_name] = feature_df
                else:
                    # 拼接 DataFrame（按日期索引）
                    merged_data_pool[feature_name] = pd.concat(
                        [merged_data_pool[feature_name], feature_df],
                        axis=0
                    )
            
            # 清理
            del chunk_package, chunk_data
            import gc
            gc.collect()
        
        # 去重并排序日期
        common_dates = sorted(list(set(all_dates)))
        
        # 确保所有特征的索引对齐
        print(f"\n   整理数据索引...")
        for feature_name in merged_data_pool.keys():
            merged_data_pool[feature_name] = merged_data_pool[feature_name].sort_index()
            # 去重（如果有重复日期，保留最后一个）
            merged_data_pool[feature_name] = merged_data_pool[feature_name][~merged_data_pool[feature_name].index.duplicated(keep='last')]
        
        print(f"\n✅ 数据合并成功！")
        print(f"   股票数: {len(merged_data_pool.get('close', pd.DataFrame()).columns)}")
        print(f"   交易日数: {len(common_dates)}")
        print(f"   日期范围: {common_dates[0]} ~ {common_dates[-1]}")
        print(f"   特征数: {len(merged_data_pool)}")
        
        return merged_data_pool, common_dates
        
    except Exception as e:
        print(f"❌ 合并数据失败: {e}")
        traceback.print_exc()
        return None, None


def preprocess_data(data_pool, verbose=True):
    """对原始数据进行预处理和标准化"""
    if verbose:
        print(f"\n🔧 数据预处理和标准化...")
    
    processed_data = {}

    # 0. 补齐缺失的衍生特征（不依赖外部数据）
    # turnover: volume / circulating_shares ≈ volume / (circulating_mcap * 1e8 / close)
    if 'turnover' not in data_pool:
        if 'circulating_market_cap' in data_pool and 'close' in data_pool and 'volume' in data_pool:
            try:
                cap = data_pool['circulating_market_cap'].replace(0, np.nan)
                close_px = data_pool['close'].replace(0, np.nan)
                circ_shares = cap * 1e8 / close_px
                turnover = data_pool['volume'] / circ_shares
                turnover = turnover.replace([np.inf, -np.inf], np.nan)
                data_pool['turnover'] = turnover
                if verbose:
                    print("  ✓ 已补齐 turnover (由 circulating_market_cap 计算)")
            except Exception:
                if verbose:
                    print("  ⚠️ turnover 补齐失败")
        else:
            if verbose:
                print("  ⚠️ 缺少 turnover 且无法计算（缺 circulating_market_cap/close/volume）")

    # pb_ratio 需要外部估值数据，离线无法推导，缺失时仅提示
    if 'pb_ratio' not in data_pool and verbose:
        print("  ⚠️ 缺少 pb_ratio（Value因子），请确保数据构造阶段已补齐")
    
    # 1. 量价特征：横截面标准化
    price_features = ['open', 'close', 'high', 'low', 'vwap']
    for feat in price_features:
        if feat in data_pool and data_pool[feat] is not None and not data_pool[feat].empty:
            # ✅ [优化] 显式 float32
            df = data_pool[feat].astype(np.float32, copy=False)
            mean = df.mean(axis=1)
            std = df.std(axis=1).replace(0, 1)
            processed_data[feat] = df.sub(mean, axis=0).div(std, axis=0).fillna(0).astype(np.float32, copy=False)
            if verbose:
                print(f"  ✓ {feat} 标准化完成")
    
    # 2. 成交量：取对数再标准化
    if 'volume' in data_pool and data_pool['volume'] is not None and not data_pool['volume'].empty:
        # ✅ [优化] 显式 float32
        vol = data_pool['volume'].astype(np.float32, copy=False)
        log_vol = np.log(vol.replace(0, 1))
        mean_vol = log_vol.mean(axis=1)
        std_vol = log_vol.std(axis=1).replace(0, 1)
        processed_data['volume'] = log_vol.sub(mean_vol, axis=0).div(std_vol, axis=0).fillna(0).astype(np.float32, copy=False)
        if verbose:
            print(f"  ✓ volume 对数标准化完成")
    
    # 3. 成交额
    if 'money' in data_pool:
        try:
            processed_data['money'] = data_pool['money'].astype(np.float32, copy=False)
        except Exception:
            processed_data['money'] = data_pool['money'].copy()
    
    # 4. 基本面因子
    fundamental_features = {
        'market_cap': 'log', 'circulating_market_cap': 'log',
        'net_profit_ttm': 'signed_log', 'operating_revenue_ttm': 'signed_log',
        'roe_ttm': 'clip', 'roa_ttm': 'clip',
        'gross_income_ratio': 'clip', 'net_profit_ratio': 'clip',
        'debt_to_asset_ratio': 'clip_0_2', 'current_ratio': 'clip_0_10',
        'eps_ttm': 'signed_log', 'net_asset_per_share': 'signed_log',
        'turnover': 'clip_0_1', 'pb_ratio': 'none',
    }
    
    for feat, transform_type in fundamental_features.items():
        if feat in data_pool and data_pool[feat] is not None and not data_pool[feat].empty:
            # ✅ [优化] 显式 float32
            df = data_pool[feat].astype(np.float32, copy=False)
            
            if transform_type == 'log':
                df = np.log(df.replace(0, np.nan))
            elif transform_type == 'signed_log':
                df = np.sign(df) * np.log(np.abs(df).replace(0, 1) + 1)
            elif transform_type == 'clip':
                df = df.clip(-1, 1)
            elif transform_type == 'clip_0_2':
                df = df.clip(0, 2)
            elif transform_type == 'clip_0_10':
                df = df.clip(0, 10)
            elif transform_type == 'clip_0_1':
                df = df.clip(0, 1)
            
            mean = df.mean(axis=1)
            std = df.std(axis=1).replace(0, 1)
            processed_data[feat] = df.sub(mean, axis=0).div(std, axis=0).fillna(0).astype(np.float32, copy=False)
            if verbose:
                print(f"  ✓ {feat} 标准化完成")
    
    # 5. 收益率：直接复制
    for ret_key in ['return_5d', 'return_10d', 'return_20d']:
        if ret_key in data_pool:
            try:
                processed_data[ret_key] = data_pool[ret_key].astype(np.float32, copy=False)
            except Exception:
                processed_data[ret_key] = data_pool[ret_key].copy()
    
    if verbose:
        print(f"  ✅ 预处理完成，共 {len(processed_data)} 个特征")
    
    return processed_data


def split_train_valid(data_pool, common_dates, split_ratio=0.7, preprocess=True):
    """拆分训练集和验证集"""
    print(f"\n✂️ 拆分数据集 (训练集比例: {split_ratio*100:.0f}%)")
    
    if preprocess:
        data_pool = preprocess_data(data_pool, verbose=True)
    
    split_idx = int(len(common_dates) * split_ratio)
    train_dates = common_dates[:split_idx]
    valid_dates = common_dates[split_idx:]
    
    print(f"\n  训练集: {len(train_dates)} 天 ({train_dates[0]} ~ {train_dates[-1]})")
    print(f"  验证集: {len(valid_dates)} 天 ({valid_dates[0]} ~ {valid_dates[-1]})")
    
    train_data = {}
    valid_data = {}
    
    for feature, df in data_pool.items():
        if df is not None and not df.empty:
            available_train_dates = [d for d in train_dates if d in df.index]
            available_valid_dates = [d for d in valid_dates if d in df.index]
            
            train_data[feature] = df.loc[available_train_dates]
            valid_data[feature] = df.loc[available_valid_dates]
    
    return train_data, valid_data, train_dates, valid_dates


# ==================== 2. Alpha语法定义 / 3. 训练内核（全部照搬 v4/main01.py）====================

# 注意：此段为 v4/main01.py 的训练核心“原样搬运”，仅用于离线数据（不含 jqdata 数据获取与 main01 的 __main__ 入口）。

# ==================== 3. Alpha语法定义（完整修复版）====================
class AlphaGrammar:
    """定义Alpha因子的上下文无关文法（修复版）"""
    
    def __init__(self, max_length=10):
        self.max_length = max_length
        
        # 特征集合（添加基本面因子）
        self.features = [
            # 量价特征
            'open', 'close', 'high', 'low', 'volume', 'vwap',
            # 基本面因子
            'market_cap',   # 市值因子 (Size)
            'pb_ratio',     # 估值因子 (Value, 已转为BP即Book-to-Price)
            'turnover',     # 换手率 (流动性)
            'roe_ttm'       # 质量因子 (Quality)
        ]
        
        # 常数集合（包含整数和浮点数）
        self.constants = [-1.0, -0.5, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10, 20, 30,60]
        
        # 窗口参数
        self.windows = [5, 10, 20, 30,60]
        
        # 运算符定义
        # 一元运算符
        self.unary_ops = {
            'abs': lambda x: np.abs(x),
            'sign': lambda x: np.sign(x),
            'log': lambda x: np.log(np.abs(x) + 1e-10),
            'neg': lambda x: -x,
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            # 添加在 unary_ops 字典中
            'rank': lambda x: x.rank(axis=1, pct=True),  # 横截面Rank，归一化到 0-1
            'ts_rank': lambda x: x.rolling(10).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1]), # 时序Rank (较慢，慎用)
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),   # Sigmoid非线性变换

        }
        
        self.ternary_ops = {
            'if_else': lambda c, t, f: t.where(c > 0, f)
        }
        
        # 二元运算符（对称）
        self.binary_sym_ops = {
            'add': lambda x, y: x + y,
            'mul': lambda x, y: x * y,
            'max': lambda x, y: np.maximum(x, y),
            'min': lambda x, y: np.minimum(x, y)
        }
        
        # 二元运算符（非对称）
        self.binary_asym_ops = {
            'sub': lambda x, y: x - y,
            'div': lambda x, y: x / y.replace(0, np.nan)
        }
        
        # 滚动运算符（修复：每个都接受2个参数）
        self.rolling_ops = {
            'mean': lambda x, w: x.rolling(w, min_periods=max(1, int(w*0.5))).mean(),
            'std': lambda x, w: x.rolling(w, min_periods=max(1, int(w*0.5))).std(),
            'ts_max': lambda x, w: x.rolling(w, min_periods=max(1, int(w*0.5))).max(),
            'ts_min': lambda x, w: x.rolling(w, min_periods=max(1, int(w*0.5))).min(),
            'delta': lambda x, w: x - x.shift(w),
            'ts_delay': lambda x, w: x.shift(w), # 滞后算子
            # 线性衰减加权平均
            'decay_linear': lambda x, w: x.rolling(w).apply(
                lambda z: np.average(z, weights=np.arange(1, len(z)+1))
            )
        }
        
        # 配对滚动运算符（修复：每个都接受3个参数）
        self.paired_rolling_ops = {
            'corr': lambda x, y, w: x.rolling(w, min_periods=max(1, int(w*0.5))).corr(y),
            'cov': lambda x, y, w: x.rolling(w, min_periods=max(1, int(w*0.5))).cov(y)
        }
        
        # 操作符参数数量映射（新增）
        self.op_arg_counts = {
            # 一元操作符：1个参数
            'abs': 1, 'sign': 1, 'log': 1, 'neg': 1, 'sqrt': 1,'rank':1, 'ts_rank':1,'sigmoid':1,
            # 二元操作符：2个参数
            'add': 2, 'mul': 2, 'max': 2, 'min': 2, 'sub': 2, 'div': 2,
            # 滚动操作符：2个参数
            'mean': 2, 'std': 2, 'ts_max': 2, 'ts_min': 2, 'delta': 2,'ts_delay':2,'decay_linear':2,
            # 配对滚动操作符：3个参数
            'corr': 3, 'cov': 3,'if_else':3
        }
    
    def _is_number_string(self, s):
        """检查字符串是否为有效数字"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def validate_expression(self, expr, depth=0, max_depth=10):
        """验证表达式语法是否正确（修复版）"""
        if depth > max_depth:
            return False, "表达式深度过大"
        
        # 字符串节点（特征或常数）
        if isinstance(expr, str):
            if expr in self.features:
                return True, f"有效特征: {expr}"
            # 检查是否为数字字符串
            elif self._is_number_string(expr):
                return True, f"有效常数: {expr}"
            else:
                return False, f"无效叶子节点: {expr}"
        
        # 整数或浮点数常数
        elif isinstance(expr, (int, float)):
            return True, f"有效常数: {expr}"
        
        # 元组表达式（操作符 + 操作数）
        elif isinstance(expr, tuple) and len(expr) > 0:
            op = expr[0]
            
            # 检查操作符是否有效
            if op not in self.op_arg_counts:
                return False, f"未知操作符: {op}"
            
            expected_args = self.op_arg_counts[op]
            actual_args = len(expr) - 1
            
            if actual_args != expected_args:
                return False, f"操作符 {op} 需要 {expected_args} 个参数，但得到 {actual_args} 个"
            
            # 递归验证子表达式
            for i in range(1, len(expr)):
                valid, msg = self.validate_expression(expr[i], depth + 1, max_depth)
                if not valid:
                    return False, f"子表达式 {i} 无效: {msg}"
            
            return True, f"有效表达式，操作符: {op}"
        
        # 其他类型
        return False, f"无效表达式类型: {type(expr)}"
    
    def evaluate_expression(self, expr, data_pool, verbose=False):
        """评估表达式，返回因子值（修复版）"""
        # 先验证表达式语法
        valid, msg = self.validate_expression(expr)
        if not valid:
            if verbose:
                print(f"  ❌ 表达式语法错误: {msg}")
            return None
        
        # 评估表达式
        return self._evaluate_expression_internal(expr, data_pool, verbose)
    
    def _evaluate_expression_internal(self, expr, data_pool, verbose=False):
        """内部评估函数（修复版）"""
        if isinstance(expr, str):
            # 单个特征
            if expr in data_pool:
                return data_pool[expr]
            # 常数字符串
            elif self._is_number_string(expr):
                try:
                    const_val = float(expr)
                    # 创建常数矩阵
                    if 'close' in data_pool:
                        const_df = pd.DataFrame(
                            const_val,
                            index=data_pool['close'].index,
                            columns=data_pool['close'].columns
                        )
                        return const_df
                except:
                    pass
            return None
        
        # 处理整数或浮点数常数
        elif isinstance(expr, (int, float)):
            if 'close' in data_pool:
                const_df = pd.DataFrame(
                    float(expr),
                    index=data_pool['close'].index,
                    columns=data_pool['close'].columns
                )
                return const_df
            return None
        
        # 元组表达式 (op, arg1, arg2, ...)
        if not isinstance(expr, tuple) or len(expr) == 0:
            return None
        
        op = expr[0]
        
        try:
            # 一元操作符
            if op in self.unary_ops:
                child_val = self._evaluate_expression_internal(expr[1], data_pool, verbose)
                if child_val is None:
                    return None
                result = self.unary_ops[op](child_val)
                return self._clean_nan_inf(result)
            
            # 二元操作符（对称）
            elif op in self.binary_sym_ops:
                left_val = self._evaluate_expression_internal(expr[1], data_pool, verbose)
                right_val = self._evaluate_expression_internal(expr[2], data_pool, verbose)
                if left_val is None or right_val is None:
                    return None
                result = self.binary_sym_ops[op](left_val, right_val)
                return self._clean_nan_inf(result)
            
            # 二元操作符（非对称）
            elif op in self.binary_asym_ops:
                left_val = self._evaluate_expression_internal(expr[1], data_pool, verbose)
                right_val = self._evaluate_expression_internal(expr[2], data_pool, verbose)
                if left_val is None or right_val is None:
                    return None
                if op == 'div':
                    # 避免除零
                    right_val = right_val.replace(0, np.nan)
                result = self.binary_asym_ops[op](left_val, right_val)
                return self._clean_nan_inf(result)
            
            # 滚动操作符
            elif op in self.rolling_ops:
                source_val = self._evaluate_expression_internal(expr[1], data_pool, verbose)
                if source_val is None or source_val.empty:
                    return None
                
                # 获取窗口参数
                window_val = expr[2]
                
                # 处理窗口参数
                if isinstance(window_val, (int, float)):
                    window_val = int(window_val)
                elif isinstance(window_val, str) and self._is_number_string(window_val):
                    window_val = int(float(window_val))
                else:
                    # 尝试评估窗口参数表达式
                    window_val_eval = self._evaluate_expression_internal(window_val, data_pool, verbose)
                    if window_val_eval is None:
                        window_val = 20  # 默认值
                    else:
                        # 尝试获取标量值
                        try:
                            if isinstance(window_val_eval, pd.DataFrame) and not window_val_eval.empty:
                                window_val = float(window_val_eval.iloc[0, 0])
                            elif isinstance(window_val_eval, (int, float)):
                                window_val = int(window_val_eval)
                            else:
                                window_val = 20
                        except:
                            window_val = 20
                
                # 确保窗口是整数且在合理范围内
                try:
                    window_val = int(window_val)
                    if window_val <= 0 or window_val > 100:
                        window_val = 20
                except:
                    window_val = 20
                
                try:
                    result = self.rolling_ops[op](source_val, window_val)
                    return self._clean_nan_inf(result)
                except Exception as e:
                    if verbose:
                        print(f"  滚动操作符 {op} 执行错误: {e}")
                    return None
            
            # 配对滚动操作符
            elif op in self.paired_rolling_ops:
                source1_val = self._evaluate_expression_internal(expr[1], data_pool, verbose)
                source2_val = self._evaluate_expression_internal(expr[2], data_pool, verbose)
                if source1_val is None or source2_val is None:
                    return None
                
                # 获取窗口参数
                window_val = expr[3]
                
                # 处理窗口参数
                if isinstance(window_val, (int, float)):
                    window_val = int(window_val)
                elif isinstance(window_val, str) and self._is_number_string(window_val):
                    window_val = int(float(window_val))
                else:
                    # 尝试评估窗口参数表达式
                    window_val_eval = self._evaluate_expression_internal(window_val, data_pool, verbose)
                    if window_val_eval is None:
                        window_val = 20  # 默认值
                    else:
                        # 尝试获取标量值
                        try:
                            if isinstance(window_val_eval, pd.DataFrame) and not window_val_eval.empty:
                                window_val = float(window_val_eval.iloc[0, 0])
                            elif isinstance(window_val_eval, (int, float)):
                                window_val = int(window_val_eval)
                            else:
                                window_val = 20
                        except:
                            window_val = 20
                
                # 确保窗口是整数且在合理范围内
                try:
                    window_val = int(window_val)
                    if window_val <= 0 or window_val > 100:
                        window_val = 20
                except:
                    window_val = 20
                
                try:
                    result = self.paired_rolling_ops[op](source1_val, source2_val, window_val)
                    return self._clean_nan_inf(result)
                except Exception as e:
                    if verbose:
                        print(f"  配对滚动操作符 {op} 执行错误: {e}")
                    return None

            elif op in self.ternary_ops:
                cond_val = self._evaluate_expression_internal(expr[1], data_pool, verbose)
                true_val = self._evaluate_expression_internal(expr[2], data_pool, verbose)
                false_val = self._evaluate_expression_internal(expr[3], data_pool, verbose)

                if cond_val is None or true_val is None or false_val is None:
                    return None

                try:
                    result = self.ternary_ops[op](cond_val, true_val, false_val)
                    return self._clean_nan_inf(result)
                except Exception as e:
                    if verbose:
                        print(f"  三元操作符 {op} 执行错误: {e}")
                    return None
            
        except Exception as e:
            if verbose:
                print(f"  表达式评估错误: {e}, 表达式: {expr}")
                traceback.print_exc()
            return None
        
        return None
    
    def _clean_nan_inf(self, df):
        """清理NaN和无穷大值"""
        if df is None:
            return None
        df = df.replace([np.inf, -np.inf], np.nan)
        # 填充NaN值（照搬 main01）
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
    
    def generate_random_expr(self, max_depth=3, current_depth=0, enforce_complexity=False):
        """随机生成表达式（修复版）"""
        if current_depth >= max_depth or (not enforce_complexity and random.random() < 0.3):
            # 叶子节点：特征或常数
            if random.random() < 0.7:
                return random.choice(self.features)
            else:
                # 随机选择常数（可能是整数或浮点数）
                const = random.choice(self.constants)
                # 50%概率返回字符串，50%概率返回数值
                return str(const) if random.random() < 0.5 else const
        
        # 选择算子类型
        weights = [0.1, 0.35, 0.35, 0.15, 0.05]   # [unary, binary, rolling, paired, ternary]
        op_type = random.choices(['unary', 'binary', 'rolling', 'paired', 'ternary'], weights=weights)[0]
        
        if op_type == 'unary':
            op = random.choice(list(self.unary_ops.keys()))
            child = self.generate_random_expr(max_depth, current_depth + 1)
            return (op, child)
        
        elif op_type == 'binary':
            op = random.choice(list(self.binary_sym_ops.keys()) + 
                              list(self.binary_asym_ops.keys()))
            left = self.generate_random_expr(max_depth, current_depth + 1)
            right = self.generate_random_expr(max_depth, current_depth + 1)
            return (op, left, right)
        
        elif op_type == 'rolling':
            op = random.choice(list(self.rolling_ops.keys()))
            source = self.generate_random_expr(max_depth, current_depth + 1)
            window = random.choice(self.windows)
            return (op, source, window)

        elif op_type == 'ternary':
            op = random.choice(list(self.ternary_ops.keys()))
            cond = self.generate_random_expr(max_depth, current_depth + 1)
            true_expr = self.generate_random_expr(max_depth, current_depth + 1)
            false_expr = self.generate_random_expr(max_depth, current_depth + 1)
            return (op, cond, true_expr, false_expr)
        else:  # paired
            op = random.choice(list(self.paired_rolling_ops.keys()))
            source1 = self.generate_random_expr(max_depth, current_depth + 1)
            source2 = self.generate_random_expr(max_depth, current_depth + 1)
            window = random.choice(self.windows)
            return (op, source1, source2, window)


# ==================== 4. 表达式ID转换器 ====================
class ExpressionEncoder:
    """将表达式转换为ID序列"""
    
    def __init__(self):
        # 完整词汇表映射
        self.vocab = {
            # 特征 (10个: 6个量价 + 4个基本面)
            'open': 0, 'close': 1, 'high': 2, 'low': 3, 'volume': 4, 'vwap': 5,
            'market_cap': 6, 'pb_ratio': 7, 'turnover': 8, 'roe_ttm': 9,
            
            # 一元运算符 (8个)
            'abs': 10, 'sign': 11, 'log': 12, 'neg': 13, 'sqrt': 14, 'rank':15, 'ts_rank':16, 'sigmoid':17,
            
            # 二元运算符 - 对称 (4个)
            'add': 18, 'mul': 19, 'max': 20, 'min': 21,
            
            # 二元运算符 - 非对称 (2个)
            'sub': 22, 'div': 23,
            
            # 滚动运算符 (7个)
            'mean': 24, 'std': 25, 'ts_max': 26, 'ts_min': 27, 'delta': 28, 'ts_delay':29, 'decay_linear':30,
            
            # 配对滚动运算符 (2个)
            'corr': 31, 'cov': 32,
            
            # 窗口参数 (5个)
            'window_5': 33, 'window_10': 34, 'window_20': 35, 'window_30': 36,
            'window_60': 37,
            # 常数 (17个)
            'const_-1.0': 38, 'const_-0.5': 39, 'const_-0.1': 40,
            'const_-0.05': 41, 'const_-0.01': 42, 'const_0.0': 43,
            'const_0.01': 44, 'const_0.05': 45, 'const_0.1': 46,
            'const_0.5': 47, 'const_1.0': 48, 'const_2.0': 49, 
            'const_5.0': 50, 'const_10': 51, 'const_20': 52, 'const_30': 53,
            'const_60': 54,
            
            # 特殊标记 (3个)
            'start': 55, 'end': 56, 'pad': 57,'if_else':58

        }
        
        # 反向词汇表
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # 常数映射
        self.constant_mapping = {
            -1.0: 'const_-1.0', -0.5: 'const_-0.5', -0.1: 'const_-0.1',
            -0.05: 'const_-0.05', -0.01: 'const_-0.01', 0.0: 'const_0.0',
            0.01: 'const_0.01', 0.05: 'const_0.05', 0.1: 'const_0.1',
            0.5: 'const_0.5', 1.0: 'const_1.0', 2.0: 'const_2.0', 
            5.0: 'const_5.0', 10: 'const_10', 20: 'const_20', 30: 'const_30',60: 'const_60'
        }
        
        # 窗口映射
        self.window_mapping = {
            5: 'window_5', 10: 'window_10', 20: 'window_20', 30: 'window_30',60: 'window_60'
        }
        
        self.vocab_size = len(self.vocab)
    
    def expr_to_ids(self, expr):
        """将表达式转换为ID序列（完整版本）"""
        def recursive_convert(node, depth=0):
            if depth > 10:  # 防止无限递归
                return [self.vocab['end']]
            
            # 字符串节点
            if isinstance(node, str):
                # 特征
                if node in self.vocab:
                    return [self.vocab[node]]
                # 常数（字符串形式）
                elif node.replace('.', '').replace('-', '').isdigit():
                    try:
                        const_val = float(node)
                        const_token = self._constant_to_token(const_val)
                        if const_token in self.vocab:
                            return [self.vocab[const_token]]
                    except:
                        pass
                # 未知标记
                return [self.vocab['pad']]
            
            # 数值节点
            elif isinstance(node, (int, float)):
                const_token = self._constant_to_token(float(node))
                if const_token in self.vocab:
                    return [self.vocab[const_token]]
                return [self.vocab['pad']]
            
            # 元组节点
            elif isinstance(node, tuple) and len(node) > 0:
                ids = []
                
                # 处理操作符
                op = node[0]
                if op in self.vocab:
                    ids.append(self.vocab[op])
                else:
                    ids.append(self.vocab['pad'])
                
                # 递归处理操作数
                for i in range(1, len(node)):
                    child_ids = recursive_convert(node[i], depth + 1)
                    ids.extend(child_ids)
                
                return ids
            
            # 其他类型
            else:
                return [self.vocab['pad']]
        
        # 执行转换并添加开始结束标记
        id_sequence = [self.vocab['start']]
        id_sequence.extend(recursive_convert(expr))
        id_sequence.append(self.vocab['end'])
        
        return id_sequence
    
    def _constant_to_token(self, value):
        """将常数映射到token"""
        # 四舍五入到最近的已知常数
        if value in self.constant_mapping:
            return self.constant_mapping[value]
        
        # 查找最接近的已知常数
        known_constants = list(self.constant_mapping.keys())
        closest = min(known_constants, key=lambda x: abs(x - value))
        return self.constant_mapping[closest]
    
    def ids_to_expr(self, ids):
        """将ID序列转换回表达式（用于调试）"""
        tokens = []
        for id_val in ids:
            if id_val in self.id_to_token:
                token = self.id_to_token[id_val]
                if token not in ['start', 'end', 'pad']:
                    tokens.append(token)
        
        # 简化版本：返回token列表
        return tokens
    
    def batch_encode(self, exprs, max_len=20):
        """批量编码表达式为填充后的张量"""
        if not TORCH_AVAILABLE:
            return None
            
        batch_ids = []
        
        for expr in exprs:
            ids = self.expr_to_ids(expr)
            
            # 截断或填充
            if len(ids) > max_len:
                ids = ids[:max_len-1] + [self.vocab['end']]
            else:
                ids = ids + [self.vocab['pad']] * (max_len - len(ids))
            
            batch_ids.append(ids)
        
        return torch.tensor(batch_ids, dtype=torch.long)
    
    def single_encode(self, expr, max_len=20):
        """编码单个表达式"""
        batch_result = self.batch_encode([expr], max_len)
        return batch_result[0] if batch_result is not None else None


# ==================== 5. 神经网络定义 (AlphaNetwork) ====================
class AlphaNetwork(nn.Module):
    """
    Actor-Critic网络，用于处理Alpha因子表达式序列
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1, action_dim=4):
        super(AlphaNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        embedded = self.embedding(x)
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(embedded)
        expr_feature = h_n[-1]
        value = self.critic_head(expr_feature)
        action_logits = self.actor_head(expr_feature)
        return action_logits, value


# ==================== 6. PPO 代理 (PPOAgent) ====================
class PPOAgent:
    def __init__(self, network, encoder, lr=1e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.02):
        self.network = network
        self.encoder = encoder
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.device = next(network.parameters()).device
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = defaultdict(list)

    def act(self, expr):
        self.network.eval()
        state_ids = self.encoder.single_encode(expr)
        state_tensor = state_ids.unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, value = self.network(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, expr, action, log_prob, value, reward, done):
        state_id = self.encoder.single_encode(expr)
        self.memory['states'].append(state_id)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)

    def update(self, next_value, num_epochs=4, batch_size=16):
        if len(self.memory['states']) < batch_size:
            return 0.0
        self.network.train()
        states = nn.utils.rnn.pad_sequence(self.memory['states'], batch_first=True, padding_value=57).to(self.device)
        actions = torch.tensor(self.memory['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float).to(self.device)
        rewards = self.memory['rewards']
        values = self.memory['values'] + [next_value]
        dones = self.memory['dones']

        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        indices = np.arange(dataset_size)
        total_loss = 0

        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                new_logits, new_values = self.network(states[idx])
                new_values = new_values.squeeze(1)
                dist = Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns[idx])

                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.memory = defaultdict(list)
        return total_loss / num_epochs


# ==================== 7. AlphaCFG主框架（深度修复版）====================
# ✅ 全部照搬 v4/main01.py（离线版不含 jqdata 数据获取与 main01 的 __main__ 入口）
class AlphaCFG:
    """AlphaCFG主框架（深度修复版）"""
    
    def __init__(self, train_data, valid_data=None, config=None):
        self.train_data = train_data  # 训练数据
        self.valid_data = valid_data  # 验证数据
        self.config = config or {
            'max_depth': 3,
            'pool_size': 15,
            'gamma': 0.95,
            'clip_epsilon': 0.1,
            'learning_rate': 1e-4,
            'embedding_dim': 32,
            'hidden_dim': 64,
            'num_layers': 1,
            'overfit_penalty': 1.2,  # 增加过拟合惩罚
            'complexity_reward': 0.05,
            'use_ppo': False,  # 暂时禁用PPO，先确保基础搜索正常
            'ic_period': 5,
            'min_train_ic': 0.02,  # 新增：最小训练IC阈值
            'min_valid_ic': 0.02,  # 提高最小验证IC阈值
            'min_stocks': 10,
            'min_dates': 20,
            'ic_consistency_weight': 2.0,  # IC一致性权重
            # ========= 性能优化（默认不改变行为）=========
            'enable_expr_cache': True,      # 表达式评估缓存（不影响结果，仅加速）
            'factor_cache_size': 256,       # 每个数据池（train/valid）缓存的 factor_df 数量（LRU）
            'parallel_eval': False,         # 是否启用“批量候选并行评估”（默认关闭，保持原串行链式搜索）
            'candidate_batch_size': 1,      # 每一步生成的候选表达式数量；>1 时才有并行价值
            'parallel_backend': 'auto',     # auto|thread|process（auto 会根据任务量/平台选择）
            'parallel_workers': None,       # None=自动；否则显式指定 worker 数
            'process_workers_cap_windows': None,  # Windows 下 process worker 上限；None=不额外限制（更吃内存）
            'accept_all_candidates': False, # 是否将批量里所有合格候选都加入因子池（默认仅选中最佳一个，行为更接近原版）
        }
        
        # 初始化语法
        self.grammar = AlphaGrammar(max_length=self.config['max_depth'])
        
        # 初始化表达式编码器
        self.encoder = ExpressionEncoder()
        
        # 初始化神经网络和PPO智能体（如果可用）
        if TORCH_AVAILABLE and self.config.get('use_ppo', False):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.network = AlphaNetwork(
                vocab_size=self.encoder.vocab_size,
                embedding_dim=self.config['embedding_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers']
            ).to(self.device)
            
            self.agent = PPOAgent(
                network=self.network,
                encoder=self.encoder,
                lr=self.config['learning_rate'],
                gamma=self.config['gamma'],
                clip_epsilon=self.config['clip_epsilon']
            )
            print(f"✅ PPO智能体初始化完成 (设备: {self.device})")
        else:
            self.agent = None
            print("⚠️  PyTorch不可用或已禁用，使用简化搜索模式")
        
        # 搜索状态
        self.best_train_ic = -999
        self.best_valid_ic = -999
        self.best_expr = None
        self.best_factor = None
        self.factor_pool = []
        self.search_history = []
        
        # 训练统计
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'ic_history': [],
            'valid_ic_history': [],
            'overfit_ratios': [],
            'expr_quality': []  # 新增：表达式质量统计
        }

        # ========= 性能优化：表达式去重/缓存 =========
        self._commutative_ops = frozenset(['add', 'mul', 'max', 'min'])
        self._enable_expr_cache = bool(self.config.get('enable_expr_cache', True))
        self._factor_cache_maxsize = int(self.config.get('factor_cache_size', 256))
        # LRU 缓存：key=(pool_name, expr_key) -> factor_df
        self._factor_cache = OrderedDict()
        self._cache_stats = defaultdict(int)  # hits/misses
        self._debug_parallel = bool(self.config.get('debug_parallel', False))
        self._debug_counter = 0
        self._run_log_dir = self.config.get('run_log_dir', None)
        self._run_log_path = None
        if self._run_log_dir:
            try:
                os.makedirs(self._run_log_dir, exist_ok=True)
                self._run_log_path = os.path.join(self._run_log_dir, 'log.txt')
            except Exception:
                self._run_log_path = None

        # ========= 进程池复用（避免频繁 spawn 卡死）=========
        self._process_pools = {}
        self._process_pool_data_ids = {}
        atexit.register(self._shutdown_process_pools)
        
        print(f"✅ AlphaCFG深度修复版初始化完成 (词汇表大小: {self.encoder.vocab_size})")
        if self.valid_data is not None:
            print(f"   训练数据: {len(self.train_data.get('close', pd.DataFrame()).index)} 天")
            print(f"   验证数据: {len(self.valid_data.get('close', pd.DataFrame()).index)} 天")
        
        # 初始化种子表达式
        self._init_seed_exprs()

    def _append_run_log(self, line):
        if not self._run_log_path or not _is_main_process():
            return
        try:
            with open(self._run_log_path, 'a', encoding='utf-8') as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            pass

    def _shutdown_process_pools(self):
        """优雅地关闭所有进程池"""
        for pool_name, pool in list(self._process_pools.items()):
            try:
                print(f"[进程池] 正在关闭 {pool_name}...")
                # 先尝试优雅关闭(等待最多5秒)
                pool.shutdown(wait=True, cancel_futures=False)
                print(f"[进程池] ✅ {pool_name} 已关闭")
            except TypeError:
                # 旧版本Python不支持cancel_futures参数
                try:
                    pool.shutdown(wait=True)
                    print(f"[进程池] ✅ {pool_name} 已关闭")
                except Exception as e:
                    print(f"[进程池] ⚠️ {pool_name} 关闭失败: {e}")
            except Exception as e:
                print(f"[进程池] ⚠️ {pool_name} 关闭失败: {e}")
        self._process_pools.clear()
        self._process_pool_data_ids.clear()

    def _get_process_pool(self, pool_name, data_pool, max_workers):
        """获取或创建进程池（带数据一致性检查和错误处理增强）"""
        data_id = id(data_pool)
        pool = self._process_pools.get(pool_name)
        
        # 检查是否需要重建进程池
        if pool is not None and self._process_pool_data_ids.get(pool_name) != data_id:
            print(f"[进程池] 检测到数据变化，需要重建 {pool_name} 进程池")
            try:
                pool.shutdown(wait=True, cancel_futures=False)
                print(f"[进程池] 旧进程池已关闭")
            except TypeError:
                try:
                    pool.shutdown(wait=True)
                except Exception as e:
                    print(f"[进程池] ⚠️ 关闭旧进程池时出错: {e}")
            except Exception as e:
                print(f"[进程池] ⚠️ 关闭旧进程池时出错: {e}")
            pool = None

        if pool is None:
            print(f"[进程池] 正在创建新的进程池: {pool_name}, workers={max_workers}")
            
            grammar_args = {'max_length': self.grammar.max_length}
            # Worker 侧需要的运行时参数（只传小配置，避免 initargs 变大）
            runtime_cfg = {
                'ic_period': int(self.config.get('ic_period', 5) or 5),
                'min_dates': int(self.config.get('min_dates', 20) or 20),
                'min_stocks': int(self.config.get('min_stocks', 10) or 10),
            }
            
            # 估算数据大小并警告
            import sys
            data_size_mb = sys.getsizeof(data_pool) / (1024 * 1024)
            print(f"[进程池] 数据池大小估算: {data_size_mb:.2f} MB")
            if data_size_mb > 100:
                print(f"[进程池] ⚠️ 警告: 数据池很大({data_size_mb:.2f} MB)，进程初始化可能需要较长时间")
            print(f"[进程池] 开始初始化 {max_workers} 个Worker进程...")
            
            try:
                pool = ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker,
                    initargs=(data_pool, grammar_args, runtime_cfg)
                )
            except Exception as e:
                print(f"[进程池] ❌ 创建进程池失败: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            print(f"[进程池] ProcessPoolExecutor 创建完成，等待 Worker 初始化...")
            # 提交测试任务，确保所有 Worker 都已初始化（增加超时和重试）
            try:
                test_futures = [pool.submit(_test_worker_pid) for _ in range(min(max_workers, 4))]
                test_pids = []
                for f in test_futures:
                    try:
                        pid = f.result(timeout=30)  # 增加超时时间到30秒
                        test_pids.append(pid)
                    except Exception as e:
                        print(f"[进程池] ⚠️ Worker 测试超时或失败: {e}")
                if test_pids:
                    print(f"[进程池] Worker 进程已就绪，PID 示例: {test_pids[:4]}")
                else:
                    print(f"[进程池] ⚠️ 所有Worker测试都失败，但进程池已创建")
            except Exception as e:
                print(f"[进程池] ⚠️ Worker 测试失败（不影响使用）: {e}")
            
            self._process_pools[pool_name] = pool
            self._process_pool_data_ids[pool_name] = data_id
            print(f"[进程池] ✅ {pool_name} 进程池就绪！")
        return pool

    def _canonical_expr(self, expr):
        """
        将表达式规范化（canonicalization），用于去重与稳定哈希。
        - 对满足交换律的算子（add/mul/max/min），对子表达式做排序
        """
        if isinstance(expr, tuple) and len(expr) > 0:
            op = expr[0]
            children = [self._canonical_expr(c) for c in expr[1:]]
            if op in self._commutative_ops:
                children = sorted(children, key=lambda x: repr(x))
            return tuple([op] + children)
        return expr

    def _expr_key(self, expr):
        """生成稳定的表达式 key（字符串），用于缓存/去重。"""
        try:
            return repr(self._canonical_expr(expr))
        except Exception:
            return repr(expr)

    def _infer_pool_name(self, data_pool):
        if data_pool is self.train_data:
            return 'train'
        if self.valid_data is not None and data_pool is self.valid_data:
            return 'valid'
        return 'custom'

    def _cache_get(self, pool_name, expr_key):
        if not self._enable_expr_cache or self._factor_cache_maxsize <= 0:
            return None
        k = (pool_name, expr_key)
        if k in self._factor_cache:
            self._cache_stats['hits'] += 1
            # LRU: move to end
            self._factor_cache.move_to_end(k)
            return self._factor_cache[k]
        self._cache_stats['misses'] += 1
        return None

    def _cache_set(self, pool_name, expr_key, factor_df):
        if (not self._enable_expr_cache) or self._factor_cache_maxsize <= 0:
            return
        k = (pool_name, expr_key)
        self._factor_cache[k] = factor_df
        self._factor_cache.move_to_end(k)
        # LRU eviction
        while len(self._factor_cache) > self._factor_cache_maxsize:
            self._factor_cache.popitem(last=False)

    def evaluate_expr_batch(self, expr_list, data_pool=None, pool_name=None, backend=None, max_workers=None):
        """
        批量评估表达式（高性能多进程版）
        """
        if not expr_list:
            return []

        if data_pool is None:
            data_pool = self.train_data

        # 1. 预处理参数
        if pool_name is None:
            pool_name = self._infer_pool_name(data_pool)

        # 强制从配置读取并行参数
        if backend is None:
            backend = self.config.get('parallel_backend', 'process')
        backend = str(backend).lower()
        if backend not in ('serial', 'thread', 'process'):
            backend = 'process'
        if backend == 'process' and (not _is_main_process()):
            backend = 'serial'
        if max_workers is None:
            # 预留1-2个核给系统，其余全用
            default_workers = max(1, (os.cpu_count() or 1) - 1)
            max_workers = self.config.get('parallel_workers', default_workers)

        max_workers = int(max_workers)
        if max_workers < 1:
            max_workers = 1

        # 2. 缓存检查 (去重)
        results = [None] * len(expr_list)

        # 建立 key -> original_indices 的映射，处理列表中重复的表达式
        key_map = defaultdict(list)

        for i, expr in enumerate(expr_list):
            key = self._expr_key(expr)
            cached = self._cache_get(pool_name, key)
            if cached is not None:
                results[i] = cached
            else:
                key_map[key].append(i)

        # 提取需要计算的唯一表达式
        unique_pending_exprs = []
        unique_pending_keys = []

        for key, indices in key_map.items():
            # 检查是否所有都在缓存里（上面可能漏网）
            if results[indices[0]] is None:
                unique_pending_exprs.append(expr_list[indices[0]])
                unique_pending_keys.append(key)

        if not unique_pending_exprs:
            return list(zip(expr_list, results))

        # 避免 worker 数远大于任务数（你日志里 unique=16 却 workers=31 会极度浪费并产生额外开销）
        if backend in ('thread', 'process'):
            task_n = len(unique_pending_exprs)
            if task_n > 0:
                max_workers = min(max_workers, task_n)
            if backend == 'process' and os.name == 'nt':
                cap_cfg = self.config.get('process_workers_cap_windows', None)
                if cap_cfg is not None:
                    try:
                        cap = int(cap_cfg)
                        if cap > 0:
                            max_workers = min(max_workers, max(1, cap))
                    except Exception:
                        pass

        if self._debug_parallel and _is_main_process():
            self._debug_counter += 1
            if self._debug_counter <= 8:
                print(
                    f"[并行调试] pool={pool_name} expr={len(expr_list)} "
                    f"unique={len(unique_pending_exprs)} backend={backend} "
                    f"workers={max_workers}"
                )

        # 3. 并行计算
        computed_factors = []

        # 如果任务少或者配置为串行
        if len(unique_pending_exprs) < 5 or max_workers <= 1 or backend == 'serial':
            if self._debug_parallel and _is_main_process() and self._debug_counter <= 8:
                print("[并行调试] 分支=串行")
            # 小批量任务不值得启动线程/进程池
            for expr in unique_pending_exprs:
                try:
                    res = self.evaluate_expr(expr, data_pool, verbose=False)
                except Exception:
                    res = None
                computed_factors.append(res)
        elif backend == 'thread':
            if self._debug_parallel and _is_main_process() and self._debug_counter <= 8:
                print("[并行调试] 分支=线程池")
            # 线程池：避免 Windows spawn 导致 data_pool 大拷贝；注意这里不要调用 self.evaluate_expr（会并发写 LRU cache）
            # 仅做纯计算，标准化与缓存写入统一在主线程处理
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    fn = functools.partial(self.grammar.evaluate_expression, data_pool=data_pool, verbose=False)
                    computed_factors = list(ex.map(fn, unique_pending_exprs))
            except Exception as e:
                print(f"  ⚠️ 线程池计算失败，回退到串行: {e}")
                computed_factors = []
                for expr in unique_pending_exprs:
                    try:
                        computed_factors.append(self.grammar.evaluate_expression(expr, data_pool, False))
                    except Exception:
                        computed_factors.append(None)
        else:
            if self._debug_parallel and _is_main_process() and self._debug_counter <= 8:
                print("[并行调试] 分支=进程池")
            # === 启动多进程计算 ===
            # 注意：这里利用全局变量 _worker_data 避免通过 pickle 传输巨大的 data_pool
            try:
                # 复用进程池，避免频繁 spawn
                print(f"[并行] 准备提交 {len(unique_pending_exprs)} 个任务到进程池...")
                executor = self._get_process_pool(pool_name, data_pool, max_workers)
                print(f"[并行] 进程池获取成功，开始执行 map()...")
                # 使用 map 保持顺序，chunksize 适当调大减少通讯开销
                # 经验：尽量让每个 worker 拿到 1-3 个任务，避免 chunksize=1 的 IPC/调度开销
                chunk = max(1, len(unique_pending_exprs) // max(1, (max_workers * 2)))
                print(f"[并行] chunksize={chunk}, max_workers={max_workers}")
                
                import time
                start_time = time.time()
                
                # 添加超时保护：为整个批次设置合理超时
                timeout_per_expr = 30  # 每个表达式最多30秒
                total_timeout = timeout_per_expr * len(unique_pending_exprs) / max_workers
                total_timeout = max(60, min(total_timeout, 600))  # 最少60秒，最多600秒
                print(f"[并行] 设置批次超时: {total_timeout:.0f}秒")
                
                try:
                    # 使用submit()代替map()实现动态负载均衡
                    from concurrent.futures import as_completed
                    future_to_idx = {executor.submit(_eval_task, expr): i for i, expr in enumerate(unique_pending_exprs)}
                    computed_factors = [None] * len(unique_pending_exprs)  # 预分配，保持顺序
                    completed_count = 0
                    
                    for future in as_completed(future_to_idx.keys(), timeout=total_timeout):
                        try:
                            result = future.result(timeout=5)  # 单个结果获取超时5秒
                            idx = future_to_idx[future]
                            computed_factors[idx] = result
                            completed_count += 1
                            if completed_count % 50 == 0:
                                elapsed_so_far = time.time() - start_time
                                speed = completed_count / elapsed_so_far if elapsed_so_far > 0 else 0
                                print(f"[并行] 进度: {completed_count}/{len(unique_pending_exprs)} ({completed_count*100//len(unique_pending_exprs)}%), 速度: {speed:.1f} expr/s")
                        except Exception as e:
                            idx = future_to_idx[future]
                            print(f"[并行] ⚠️ 任务 {idx} 失败: {e}")
                            computed_factors[idx] = None
                    
                    elapsed = time.time() - start_time
                    print(f"[并行] ✅ 并行计算完成！耗时 {elapsed:.2f}s, 平均速度: {len(unique_pending_exprs)/elapsed:.1f} expr/s")
                except Exception as timeout_e:
                    print(f"[并行] ⚠️ 批次超时或失败: {timeout_e}")
                    # 取消所有未完成的任务
                    for f in future_to_idx.keys():
                        f.cancel()
                    raise
                    
            except Exception as e:
                print(f"  ⚠️ 多进程计算失败，回退到串行: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                computed_factors = []
                for expr in unique_pending_exprs:
                    computed_factors.append(self.evaluate_expr(expr, data_pool, verbose=False))

        # 4. 结果回填与缓存更新
        for key, factor_df in zip(unique_pending_keys, computed_factors):
            # 标准化 (Worker里只做了计算，这里做标准化和缓存)
            if factor_df is not None and not factor_df.empty:
                # 在主进程做标准化，避免传输数据膨胀，或者也可以移到worker里
                try:
                    # 简单的标准化处理，确保数据合规
                    if isinstance(factor_df, pd.DataFrame):
                        factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
                        # 这里复用 evaluate_expr 里的逻辑
                        mean = factor_df.mean(axis=1)
                        std = factor_df.std(axis=1).replace(0, 1)
                        factor_df = factor_df.sub(mean, axis=0).div(std, axis=0).fillna(0)
                except Exception:
                    pass

                self._cache_set(pool_name, key, factor_df)

            # 回填到所有对应的索引位置
            if key in key_map:
                for idx in key_map[key]:
                    results[idx] = factor_df

        return list(zip(expr_list, results))

    def evaluate_expr_batch_metrics(self, expr_list, data_pool=None, pool_name=None, backend=None, max_workers=None):
        """
        批量评估表达式，但只返回轻量指标（ic/icir/days），用于 Windows 多进程提速。
        - process: Worker 内计算因子 + IC，只回传小结果，避免 IPC 传输巨大 DataFrame
        - thread/serial: 退化为 evaluate_expr_batch + 主进程 calculate_ic
        返回: List[dict]，每个 dict 至少含 expr/ic/icir/days
        """
        if not expr_list:
            return []

        if data_pool is None:
            data_pool = self.train_data

        if pool_name is None:
            pool_name = self._infer_pool_name(data_pool)

        if backend is None:
            backend = self.config.get('parallel_backend', 'process')
        backend = str(backend).lower()
        if backend not in ('serial', 'thread', 'process'):
            backend = 'process'
        if backend == 'process' and (not _is_main_process()):
            backend = 'serial'

        if max_workers is None:
            default_workers = max(1, (os.cpu_count() or 1) - 1)
            max_workers = self.config.get('parallel_workers', default_workers)
        max_workers = int(max_workers) if max_workers is not None else 1
        if max_workers < 1:
            max_workers = 1

        # 去重：key -> indices
        key_map = defaultdict(list)
        for i, expr in enumerate(expr_list):
            key_map[self._expr_key(expr)].append(i)

        unique_exprs = [expr_list[idxs[0]] for idxs in key_map.values()]
        task_n = len(unique_exprs)
        
        # ⚠️ 修复：合理限制worker数量，避免过度创建进程
        if backend in ('thread', 'process'):
            # 对于少量任务，worker不要超过任务数
            max_workers = min(max_workers, task_n) if task_n > 0 else 1
            
            # Windows进程池：建议worker数不超过CPU核心数的75%（避免调度开销）
            if backend == 'process' and os.name == 'nt':
                cpu_count = os.cpu_count() or 4
                # 优先使用用户配置的上限
                cap_cfg = self.config.get('process_workers_cap_windows', None)
                if cap_cfg is not None:
                    try:
                        cap = int(cap_cfg)
                        if cap > 0:
                            max_workers = min(max_workers, max(1, cap))
                    except Exception:
                        pass
                else:
                    # 默认上限：不超过CPU核心数的75%，且不超过8个
                    # 对于16核CPU，最多用12个worker；但考虑到Windows调度，建议更保守
                    recommended_max = min(8, max(2, int(cpu_count * 0.5)))
                    max_workers = min(max_workers, recommended_max)
                    if self._debug_parallel and _is_main_process():
                        print(f"[并行优化] Windows进程池自动限制: {max_workers} workers (CPU={cpu_count})")

        # 小任务直接串行，避免调度开销
        if task_n < 5 or max_workers <= 1 or backend == 'serial':
            out_unique = []
            for expr in unique_exprs:
                try:
                    factor_df = self.evaluate_expr(expr, data_pool, verbose=False)
                    ic, icir, days = self.calculate_ic(factor_df, data_pool, verbose=False) if (factor_df is not None and not factor_df.empty) else (0, 0, 0)
                except Exception:
                    ic, icir, days = 0, 0, 0
                out_unique.append({'expr': expr, 'ic': float(ic), 'icir': float(icir), 'days': int(days)})
        elif backend == 'process':
            if self._debug_parallel and _is_main_process():
                self._debug_counter += 1
                if self._debug_counter <= 8:
                    print(f"[并行调试] metrics pool={pool_name} expr={len(expr_list)} unique={task_n} backend=process workers={max_workers}")
                    print("[并行调试] metrics 分支=进程池(IC-only)")
            try:
                print(f"[并行-IC] 准备提交 {task_n} 个任务...")
                executor = self._get_process_pool(pool_name, data_pool, max_workers)
                
                # ⚠️ 使用submit()代替map()，实现动态负载均衡
                # 优点：worker完成任务后立即获取新任务，不会空闲等待
                print(f"[并行-IC] 使用动态调度模式, workers={max_workers}, 总任务={task_n}")
                
                import time
                from concurrent.futures import as_completed
                start_time = time.time()
                
                # 提交所有任务
                future_to_expr = {executor.submit(_eval_task_ic, expr): expr for expr in unique_exprs}
                
                # 动态收集结果
                raw = []
                completed_count = 0
                last_report_time = start_time
                last_report_count = 0
                
                for future in as_completed(future_to_expr, timeout=max(300, task_n * 30)):
                    try:
                        result = future.result(timeout=5)
                        raw.append(result)
                        completed_count += 1
                        
                        # 每完成25%或每5秒显示进度
                        elapsed_so_far = time.time() - start_time
                        time_since_report = elapsed_so_far - (last_report_time - start_time)
                        
                        should_report = (
                            completed_count % max(1, task_n // 4) == 0 or 
                            completed_count == task_n or
                            time_since_report >= 5.0
                        )
                        
                        if should_report:
                            # 计算增量速度（最近一段时间的速度）
                            tasks_since_report = completed_count - last_report_count
                            if time_since_report > 0:
                                recent_speed = tasks_since_report / time_since_report
                            else:
                                recent_speed = 0
                            avg_speed = completed_count / elapsed_so_far if elapsed_so_far > 0 else 0
                            
                            print(f"[并行-IC] 进度: {completed_count}/{task_n} ({completed_count*100//task_n}%), "
                                  f"最近速度: {recent_speed:.1f} expr/s, 平均: {avg_speed:.1f} expr/s")
                            
                            last_report_time = time.time()
                            last_report_count = completed_count
                            
                    except Exception as e:
                        # 单个任务失败，记录但继续
                        expr = future_to_expr.get(future, "unknown")
                        print(f"[并行-IC] ⚠️ 单个任务失败: {e}")
                        raw.append((expr, 0.0, 0.0, 0))
                        completed_count += 1
                
                elapsed = time.time() - start_time
                print(f"[并行-IC] ✅ 完成！耗时 {elapsed:.2f}s, 平均速度: {task_n/elapsed:.1f} expr/s")
                
                out_unique = [{'expr': e, 'ic': float(ic), 'icir': float(icir), 'days': int(days)} for (e, ic, icir, days) in raw]
            except Exception as e:
                print(f"  ⚠️ 多进程(IC-only)失败，回退串行: {e}")
                import traceback
                traceback.print_exc()
                out_unique = []
                for expr in unique_exprs:
                    try:
                        factor_df = self.evaluate_expr(expr, data_pool, verbose=False)
                        ic, icir, days = self.calculate_ic(factor_df, data_pool, verbose=False) if (factor_df is not None and not factor_df.empty) else (0, 0, 0)
                    except Exception:
                        ic, icir, days = 0, 0, 0
                    out_unique.append({'expr': expr, 'ic': float(ic), 'icir': float(icir), 'days': int(days)})
        else:
            # thread：仍在主进程算 IC（线程对这段帮助有限），但保持行为一致
            eval_pairs = self.evaluate_expr_batch(expr_list, data_pool=data_pool, pool_name=pool_name, backend='thread', max_workers=max_workers)
            # 注意：eval_pairs 已是原顺序，这里不复用 unique
            out = []
            for expr, factor_df in eval_pairs:
                try:
                    ic, icir, days = self.calculate_ic(factor_df, data_pool, verbose=False) if (factor_df is not None and not factor_df.empty) else (0, 0, 0)
                except Exception:
                    ic, icir, days = 0, 0, 0
                out.append({'expr': expr, 'ic': float(ic), 'icir': float(icir), 'days': int(days)})
            return out

        # 将 unique 结果回填为原顺序
        unique_by_key = {}
        for d in out_unique:
            unique_by_key[self._expr_key(d['expr'])] = d

        out = []
        for expr in expr_list:
            d = unique_by_key.get(self._expr_key(expr), None)
            if d is None:
                out.append({'expr': expr, 'ic': 0.0, 'icir': 0.0, 'days': 0})
            else:
                out.append({'expr': expr, 'ic': float(d['ic']), 'icir': float(d['icir']), 'days': int(d['days'])})
        return out
    
    def _crossover_expr(self, expr1, expr2):
        """
        子树交叉：交换两个表达式的随机子树
        """
        if not isinstance(expr1, tuple) or not isinstance(expr2, tuple):
            return expr1 if random.random() < 0.5 else expr2 # 如果是叶子节点，不做交叉

        # 辅助函数：获取所有子节点的路径
        def get_subtrees(expr, path=()):
            paths = [(path, expr)]
            if isinstance(expr, tuple):
                # 从1开始，跳过操作符
                for i in range(1, len(expr)):
                    paths.extend(get_subtrees(expr[i], path + (i,)))
            return paths

        # 辅助函数：替换指定路径的子树
        def replace_at_path(expr, path, new_subtree):
            if not path:
                return new_subtree
            idx = path[0]
            # 递归构建新元组
            child = replace_at_path(expr[idx], path[1:], new_subtree)
            new_list = list(expr)
            new_list[idx] = child
            return tuple(new_list)

        try:
            # 获取两个表达式的所有子树路径
            subtrees1 = get_subtrees(expr1)
            subtrees2 = get_subtrees(expr2)

            # 随机选择交叉点 (避开根节点，根节点直接交换没意义)
            if len(subtrees1) > 1:
                path1, _ = random.choice(subtrees1[1:])
            else:
                path1 = ()
            
            if len(subtrees2) > 1:
                path2, subtree2 = random.choice(subtrees2[1:])
            else:
                path2, subtree2 = (), expr2

            # 执行交叉：把 expr2 的某部分 放到 expr1 的某部分
            new_expr = replace_at_path(expr1, path1, subtree2)
            
            # 深度检查，如果交叉后太深，放弃
            if self._get_expr_depth(new_expr) > self.config['max_depth'] + 2:
                return expr1
                
            return new_expr
        except:
            return expr1
    
    def _init_seed_exprs(self):
        """初始化种子表达式池（包含基本面因子）"""
        self.seed_exprs = [
            # 原有量价因子表达式
            ('mean', 'close', 10),
            ('std', 'close', 20),
            ('delta', 'close', 5),
            ('corr', 'close', 'volume', 20),
            ('add', 'close', 'open'),
            ('sub', 'close', 'open'),
            ('ts_max', 'close', 20),
            ('ts_min', 'close', 20),
            ('add', ('mean', 'close', 10), ('std', 'volume', 20)),
            ('sub', ('ts_max', 'high', 10), ('ts_min', 'low', 10)),
            ('mul', ('sub', 'close', 'open'), 'volume'),
            ('div', ('mean', 'volume', 10), ('std', 'volume', 20)),
            ('corr', 'high', 'low', 10),
            ('add', 'vwap', ('neg', 'close')),
            ('mul', ('sign', ('sub', 'close', 'open')), 'volume'),
            ('add', ('mean', 'close', 20), ('neg', ('mean', 'open', 20))),
            ('mul', ('delta', 'volume', 5), ('sub', 'close', 'open')),
            ('div', ('std', 'close', 10), ('std', 'volume', 10)),
            ('corr', ('mean', 'close', 10), ('mean', 'volume', 10), 20),
            ('add', ('ts_max', 'high', 10), ('neg', ('ts_min', 'low', 10))),
            
            # 基本面因子单独表达式
            'market_cap',  # 市值因子
            'pb_ratio',    # 估值因子 (已转为BP)
            'turnover',    # 流动性因子
            'roe_ttm',     # 质量因子
            
            # 基本面因子组合
            ('neg', 'market_cap'),  # 反向市值（小市值）
            ('mul', 'pb_ratio', 'roe_ttm'),  # 估值×质量
            ('sub', 'roe_ttm', ('mean', 'roe_ttm', 60)),  # ROE动量
            ('div', 'turnover', ('std', 'turnover', 20)),  # 换手率异常
            
            # 量价与基本面交叉
            ('mul', 'market_cap', ('delta', 'close', 5)),  # 市值×价格动量
            ('mul', 'pb_ratio', ('mean', 'volume', 10)),   # 估值×成交量
            ('corr', 'close', 'market_cap', 20),  # 价格与市值相关性
            ('corr', 'turnover', 'roe_ttm', 30),  # 流动性与质量相关性
            ('add', ('neg', 'market_cap'), ('mean', 'close', 20)),  # 小市值+价格均线
            ('mul', 'roe_ttm', ('sub', 'close', 'open')),  # 质量×日内收益
        ]
        
        # 添加更多简单有效的表达式
        for period in [5, 10, 20]:
            self.seed_exprs.append(('mean', 'close', period))
            self.seed_exprs.append(('std', 'close', period))
            self.seed_exprs.append(('delta', 'close', period))
            self.seed_exprs.append(('ts_max', 'close', period))
            self.seed_exprs.append(('ts_min', 'close', period))
            
            # 基本面因子的时序特征
            if period >= 10:
                self.seed_exprs.append(('mean', 'turnover', period))
                self.seed_exprs.append(('delta', 'roe_ttm', period))
        
        print(f"✅ 初始化种子表达式池: {len(self.seed_exprs)} 个表达式（含基本面因子）")

    # 后续方法（calculate_ic/calculate_portfolio_metrics/evaluate_expr/calculate_reward/evaluate_factor_quality/search/_mutate_expr/analyze_results）
    # 由于篇幅原因继续照搬 main01 的其余部分（保持与 main01 一致）

    # NOTE: 这里为了保证文件可用，我们将剩余方法保留为从 main01 直接复制的版本
    # ——它们与上面已粘贴的方法属于同一个类体。

    def calculate_ic(self, factor_df, data_pool=None, period=None, verbose=False):
        """计算正确的时序IC（深度修复版）"""
        if factor_df is None or factor_df.empty:
            if verbose:
                print("  ❌ 因子数据为空")
            return 0, 0, 0  # 返回IC, ICIR, 有效天数
        
        if data_pool is None:
            data_pool = self.train_data
        
        if period is None:
            period = self.config.get('ic_period', 5)
        
        return_key = f'return_{period}d'
        if return_key not in data_pool:
            if 'close' not in data_pool or data_pool['close'].empty:
                if verbose:
                    print("  ❌ 缺少收盘价数据")
                return 0, 0, 0
            returns = data_pool['close'].pct_change(period).shift(-period)
        else:
            returns = data_pool[return_key]
        
        if returns is None or returns.empty:
            if verbose:
                print("  ❌ 收益率数据为空")
            return 0, 0, 0
        
        common_dates = factor_df.index.intersection(returns.index)
        if len(common_dates) < self.config.get('min_dates', 20):
            if verbose:
                print(f"  ❌ 共同日期不足: {len(common_dates)} < {self.config.get('min_dates', 20)}")
            return 0, 0, 0
        
        common_stocks = factor_df.columns.intersection(returns.columns)
        if len(common_stocks) < self.config.get('min_stocks', 10):
            if verbose:
                print(f"  ❌ 共同股票不足: {len(common_stocks)} < {self.config.get('min_stocks', 10)}")
            return 0, 0, 0
        
        factor_aligned = factor_df.loc[common_dates, common_stocks]
        returns_aligned = returns.loc[common_dates, common_stocks]
        
        daily_ics = []
        for date in common_dates:
            factor_vals = factor_aligned.loc[date]
            return_vals = returns_aligned.loc[date]
            
            mask = factor_vals.notna() & return_vals.notna()
            if mask.sum() < max(5, self.config.get('min_stocks', 10) // 2):
                continue
            
            factor_clean = factor_vals[mask]
            return_clean = return_vals[mask]
            
            try:
                ic = factor_clean.corr(return_clean, method='spearman')
                if not np.isnan(ic):
                    daily_ics.append(ic)
            except Exception as e:
                if verbose and len(daily_ics) == 0:
                    print(f"  ⚠️ {date} Spearman失败: {e}")
                try:
                    ic = factor_clean.corr(return_clean)
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                except Exception as e2:
                    if verbose and len(daily_ics) == 0:
                        print(f"  ⚠️ {date} Pearson也失败: {e2}")
                    continue
        
        effective_days = len(daily_ics)
        if effective_days < 10:
            if verbose:
                print(f"  ❌ 有效IC天数不足: {effective_days} < 10")
            return 0, 0, effective_days
        
        ic_mean = np.mean(daily_ics)
        ic_std = np.std(daily_ics)
        icir = ic_mean / (ic_std + 1e-10)
        
        if verbose:
            print(f"  ✓ IC统计: 均值={ic_mean:.4f}, 标准差={ic_std:.4f}, ICIR={icir:.4f}, 天数={effective_days}")
        
        return ic_mean, icir, effective_days

    def calculate_portfolio_metrics(self, factor_df, data_pool=None, period=None, top_pct=0.2, verbose=False):
        """
        计算基于因子的投资组合收益率和波动率指标
        """
        if factor_df is None or factor_df.empty:
            if verbose:
                print("  ❌ 因子数据为空")
            return {}
        
        if data_pool is None:
            data_pool = self.train_data
        
        if period is None:
            period = self.config.get('ic_period', 5)
        
        # 获取收益率数据
        return_key = f'return_{period}d'
        if return_key not in data_pool:
            if 'close' not in data_pool or data_pool['close'].empty:
                return {}
            returns = data_pool['close'].pct_change(period).shift(-period)
        else:
            returns = data_pool[return_key]
        
        if returns is None or returns.empty:
            return {}
        
        # 对齐数据
        common_dates = factor_df.index.intersection(returns.index)
        common_stocks = factor_df.columns.intersection(returns.columns)
        
        if len(common_dates) < 10 or len(common_stocks) < 5:
            return {}
        
        factor_aligned = factor_df.loc[common_dates, common_stocks]
        returns_aligned = returns.loc[common_dates, common_stocks]
        
        # 构建多头组合（选取因子值最高的股票）
        portfolio_returns = []
        
        for date in common_dates:
            factor_vals = factor_aligned.loc[date]
            return_vals = returns_aligned.loc[date]
            
            # 删除NaN
            valid_mask = factor_vals.notna() & return_vals.notna()
            if valid_mask.sum() < 3:
                continue
            
            factor_clean = factor_vals[valid_mask]
            return_clean = return_vals[valid_mask]
            
            # 选取因子值最高的前top_pct只股票
            n_select = max(1, int(len(factor_clean) * top_pct))
            top_stocks = factor_clean.nlargest(n_select).index
            
            # 等权组合收益
            portfolio_ret = return_clean[top_stocks].mean()
            portfolio_returns.append(portfolio_ret)
        
        if len(portfolio_returns) < 10:
            return {}
        
        portfolio_returns = np.array(portfolio_returns)
        
        mean_return = np.mean(portfolio_returns)
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        n_periods = len(portfolio_returns)
        # 用复利累计收益率反推年化，使两者口径一致（避免“对不上”）
        if n_periods > 0 and period > 0:
            annualized_return = (1 + cumulative_return) ** (252 / (n_periods * period)) - 1
        else:
            annualized_return = 0.0
        
        volatility = np.std(portfolio_returns)
        annualized_volatility = volatility * np.sqrt(252 / period)
        downside_volatility = np.std(portfolio_returns[portfolio_returns < 0])
        
        sharpe_ratio = mean_return / (volatility + 1e-10)
        annualized_sharpe = sharpe_ratio * np.sqrt(252 / period)
        sortino_ratio = mean_return / (downside_volatility + 1e-10)
        
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns)
        avg_win = np.mean(portfolio_returns[portfolio_returns > 0]) if np.any(portfolio_returns > 0) else 0
        avg_loss = np.mean(portfolio_returns[portfolio_returns < 0]) if np.any(portfolio_returns < 0) else 0
        profit_loss_ratio = abs(avg_win / (avg_loss + 1e-10))
        
        calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-10)
        
        metrics = {
            'mean_return': mean_return,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'downside_volatility': downside_volatility,
            'sharpe_ratio': sharpe_ratio,
            'annualized_sharpe': annualized_sharpe,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'calmar_ratio': calmar_ratio,
            'n_periods': n_periods
        }
        
        if verbose:
            print(f"  📈 组合表现指标 (持仓周期{period}天, 选股比例{top_pct*100:.0f}%):")
            print(f"     平均收益率: {mean_return:.4f} ({mean_return*100:.2f}%)")
            print(f"     累计收益率: {cumulative_return:.4f} ({cumulative_return*100:.2f}%)")
            print(f"     年化收益率: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
            print(f"     年化波动率: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")
            print(f"     年化夏普比率: {annualized_sharpe:.4f}")
            print(f"     最大回撤: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
            print(f"     胜率: {win_rate:.4f} ({win_rate*100:.2f}%)")
            print(f"     盈亏比: {profit_loss_ratio:.4f}")
            print(f"     卡玛比率: {calmar_ratio:.4f}")
        
        return metrics
    
    def evaluate_expr(self, expr, data_pool=None, verbose=False):
        """评估表达式，返回因子值（修复版）"""
        if data_pool is None:
            data_pool = self.train_data
        
        # 先验证表达式语法
        valid, msg = self.grammar.validate_expression(expr)
        if not valid:
            if verbose:
                print(f"  ❌ 表达式语法错误: {msg}")
            return None
        
        pool_name = self._infer_pool_name(data_pool)
        expr_key = self._expr_key(expr)
        cached = self._cache_get(pool_name, expr_key)
        if cached is not None:
            return cached

        factor_df = self.grammar.evaluate_expression(expr, data_pool, verbose)
        
        if factor_df is not None and not factor_df.empty:
            try:
                factor_std = factor_df.std(axis=1)
                factor_std = factor_std.replace(0, 1)
                
                factor_mean = factor_df.mean(axis=1)
                factor_df = factor_df.sub(factor_mean, axis=0).div(factor_std, axis=0)
                
                factor_df = factor_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                factor_df = factor_df.clip(-3, 3)
                
                if verbose:
                    print(f"  ✓ 因子标准化完成: {factor_df.shape}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ 因子标准化失败: {e}")
                factor_df = factor_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        if factor_df is not None and hasattr(factor_df, 'empty') and (not factor_df.empty):
            self._cache_set(pool_name, expr_key, factor_df)

        return factor_df
    
    def calculate_reward(self, ic, expr, icir=None, valid_ic=None, depth=None, 
                        train_days=None, valid_days=None, verbose=False):
        """计算奖励函数（深度修复版 - 添加valid惩罚机制）"""
        if depth is None:
            depth = self._get_expr_depth(expr)
        
        if abs(ic) < 0.005:
            reward = -3.0
        elif ic > 0:
            reward = ic * 8
        else:
            reward = ic * 12
        
        if icir is not None:
            if icir > 0 and ic > 0.01:
                reward += min(icir * 2, 2.0)
            elif icir < -1.0:
                reward -= 1.0
        
        if valid_ic is not None and ic != 0:
            if ic * valid_ic < 0:
                consistency_penalty = self.config.get('ic_consistency_weight', 2.0) * abs(ic)
                reward -= consistency_penalty
                if verbose:
                    print(f"    IC符号不一致惩罚: -{consistency_penalty:.4f}")
            else:
                consistency_bonus = min(abs(ic) * 0.5, 1.0)
                reward += consistency_bonus
        
        if valid_ic is not None and abs(ic) > 0.01:
            try:
                overfit_ratio = 1.0 - min(abs(valid_ic) / (abs(ic) + 1e-10), 2.0)
                overfit_ratio = max(-1.0, min(1.0, overfit_ratio))
                
                if overfit_ratio > 0:
                    overfit_penalty = overfit_ratio * self.config.get('overfit_penalty', 1.2)
                    reward -= overfit_penalty
                    
                    if verbose:
                        print(f"    过拟合比例: {overfit_ratio:.2%}, 惩罚: {overfit_penalty:.4f}")
                elif overfit_ratio < -0.5:
                    reward -= 1.0
                    if verbose:
                        print(f"    ⚠️ 验证IC远大于训练IC，可疑")
            except Exception as e:
                if verbose:
                    print(f"    过拟合计算错误: {e}")
        
        if train_days is not None and train_days > 0:
            days_reward = min(train_days / 100.0, 0.5)
            reward += days_reward
        
        if 2 <= depth <= 4:
            reward += self.config.get('complexity_reward', 0.05)
        elif depth == 1:
            reward -= 0.3
        elif depth > 5:
            reward -= 0.1 * (depth - 5)
        
        if self.factor_pool and len(self.factor_pool) > 1:
            similarities = []
            for factor in self.factor_pool:
                sim = self._calculate_similarity(expr, factor['expr'])
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            diversity_bonus = 1.0 - avg_similarity
            reward += diversity_bonus * 0.2
        
        if abs(ic) < 0.005 and abs(valid_ic or 0) > 0.02:
            reward -= 5.0
        
        if verbose:
            print(f"    最终奖励: {reward:.4f}")
        
        return reward
    
    def _get_expr_depth(self, expr):
        """计算表达式深度"""
        if isinstance(expr, str):
            return 1
        elif isinstance(expr, (int, float)):
            return 1
        elif isinstance(expr, tuple):
            child_depths = [self._get_expr_depth(child) for child in expr[1:]]
            return 1 + max(child_depths) if child_depths else 1
        return 1
    
    def _calculate_similarity(self, expr1, expr2):
        """计算表达式相似度"""
        if expr1 == expr2:
            return 1.0
        
        def flatten(e):
            if isinstance(e, str):
                return [e]
            elif isinstance(e, (int, float)):
                return [str(e)]
            elif isinstance(e, tuple):
                tokens = [e[0]]
                for child in e[1:]:
                    tokens.extend(flatten(child))
                return tokens
            return []
        
        tokens1 = set(flatten(expr1))
        tokens2 = set(flatten(expr2))
        
        if not tokens1 or not tokens2:
            return 0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0
    
    def evaluate_factor_quality(self, expr, train_ic, valid_ic, train_days=0, valid_days=0):
        """评估因子质量（深度修复版）"""
        if expr is None:
            return "INVALID", "无效表达式"
        
        valid, msg = self.grammar.validate_expression(expr)
        if not valid:
            return "INVALID", f"语法错误: {msg}"
        
        if train_days < 10 or valid_days < 5:
            return "INSUFFICIENT_DATA", f"数据不足: 训练{train_days}天, 验证{valid_days}天"
        
        if train_ic * valid_ic < 0:
            return "SUSPICIOUS", f"IC符号不一致: 训练={train_ic:.4f}, 验证={valid_ic:.4f}"
        
        min_train_ic = self.config.get('min_train_ic', 0.02)
        if abs(train_ic) < min_train_ic:
            return "WEAK_TRAIN", f"训练信号微弱: IC={train_ic:.4f} < {min_train_ic}"
        
        min_valid_ic = self.config.get('min_valid_ic', 0.02)
        if abs(valid_ic) < min_valid_ic:
            return "WEAK_VALID", f"验证信号微弱: IC={valid_ic:.4f} < {min_valid_ic}"
        
        if abs(train_ic) > 0.001:
            overfit_ratio = 1.0 - abs(valid_ic) / (abs(train_ic) + 1e-10)
        else:
            overfit_ratio = 1.0
        
        if valid_ic > 0:
            if valid_ic > 0.03 and overfit_ratio < 0.3:
                return "EXCELLENT", "强且稳健的正Alpha信号"
            elif valid_ic > 0.025 and overfit_ratio < 0.4:
                return "GOOD", "有效的正Alpha信号"
            elif valid_ic > 0.02 and overfit_ratio < 0.5:
                return "FAIR", "一般的正Alpha信号"
            elif overfit_ratio > 0.7:
                return "OVERFITTED", f"过拟合严重: 比例={overfit_ratio:.2%}"
            else:
                return "MARGINAL", "边际正Alpha信号"
        else:
            if valid_ic < -0.03 and overfit_ratio < 0.3:
                return "EXCELLENT_NEG", "强且稳健的负Alpha信号"
            elif valid_ic < -0.025 and overfit_ratio < 0.4:
                return "GOOD_NEG", "有效的负Alpha信号"
            elif valid_ic < -0.02 and overfit_ratio < 0.5:
                return "FAIR_NEG", "一般的负Alpha信号"
            elif overfit_ratio > 0.7:
                return "OVERFITTED_NEG", f"过拟合严重: 比例={overfit_ratio:.2%}"
            else:
                return "MARGINAL_NEG", "边际负Alpha信号"
    
    def search(self, num_episodes=50, steps_per_episode=6, verbose=True):
        """主搜索循环（深度修复版）"""
        print(f"\n🔍 开始AlphaCFG搜索 (共 {num_episodes} 轮)")
        
        for episode in range(num_episodes):
            # 显示episode进度
            if episode % 10 == 0 or episode < 5:
                print(f"\n{'='*60}")
                print(f"📍 开始轮次 {episode+1}/{num_episodes}")
                print(f"{'='*60}")
            
            if episode < 10:
                current_expr = random.choice(self.seed_exprs)
            elif episode < 30 or random.random() < 0.4:
                current_expr = self.grammar.generate_random_expr(
                    max_depth=min(3, self.config['max_depth']),
                    enforce_complexity=(episode > num_episodes//2)
                )
            else:
                if self.factor_pool and random.random() < 0.7:
                    current_expr = random.choice(self.factor_pool)['expr']
                else:
                    current_expr = random.choice(self.seed_exprs)
            
            if not isinstance(current_expr, tuple) and current_expr not in self.grammar.features:
                current_expr = random.choice(self.seed_exprs)
            
            episode_rewards = []
            episode_ics = []
            episode_valid_ics = []
            
            for step in range(steps_per_episode):
                try:
                    batch_size = int(self.config.get('candidate_batch_size', 1) or 1)
                    accept_all = bool(self.config.get('accept_all_candidates', False))
                    min_train_ic = self.config.get('min_train_ic', 0.02)
                    min_valid_ic = self.config.get('min_valid_ic', 0.02)

                    if batch_size <= 1:
                        valid, msg = self.grammar.validate_expression(current_expr)
                        if not valid:
                            if verbose and episode % 20 == 0 and step == 0:
                                print(f"  ⚠️ 无效表达式: {msg}, 重新生成")
                            current_expr = random.choice(self.seed_exprs)
                            reward = -1.0
                            ic = 0
                            icir = 0
                            valid_ic = 0
                            train_days = 0
                            valid_days = 0
                        else:
                            factor_df = self.evaluate_expr(current_expr, self.train_data, verbose=False)

                            if factor_df is None or factor_df.empty:
                                reward = -1.0
                                ic = 0
                                icir = 0
                                valid_ic = 0
                                train_days = 0
                                valid_days = 0
                            else:
                                ic, icir, train_days = self.calculate_ic(factor_df, self.train_data, verbose=False)

                                valid_ic = 0
                                valid_icir = 0
                                valid_days = 0
                                if self.valid_data is not None:
                                    valid_factor = self.evaluate_expr(current_expr, self.valid_data, verbose=False)
                                    if valid_factor is not None and not valid_factor.empty:
                                        valid_ic, valid_icir, valid_days = self.calculate_ic(valid_factor, self.valid_data, verbose=False)

                                depth = self._get_expr_depth(current_expr)
                                reward = self.calculate_reward(
                                    ic, current_expr, icir, valid_ic, depth,
                                    train_days, valid_days, verbose=False
                                )

                                episode_ics.append(ic)
                                episode_valid_ics.append(valid_ic)

                                if (valid_ic > self.best_valid_ic and
                                    ic > min_train_ic and
                                    train_days >= 10 and valid_days >= 5 and
                                    ic * valid_ic > 0):

                                    self.best_valid_ic = valid_ic
                                    self.best_train_ic = ic
                                    self.best_expr = current_expr
                                    self.best_factor = factor_df

                                    if valid_ic > 0.02 and verbose:
                                        quality, quality_msg = self.evaluate_factor_quality(
                                            current_expr, ic, valid_ic, train_days, valid_days
                                        )

                                        if abs(ic) > 0.001:
                                            overfit_ratio = 1.0 - abs(valid_ic) / (abs(ic) + 1e-10)
                                        else:
                                            overfit_ratio = 1.0

                                        print(f"\n🎉 [轮次 {episode+1}] 新最佳验证IC: {valid_ic:.4f}")
                                        print(f"    训练IC: {ic:.4f}, ICIR: {icir:.4f}")
                                        print(f"    验证ICIR: {valid_icir:.4f}")
                                        print(f"    有效天数: 训练{train_days}, 验证{valid_days}")
                                        print(f"    过拟合比例: {overfit_ratio:.2%}")
                                        print(f"    质量评估: {quality} - {quality_msg}")
                                        print(f"    表达式: {current_expr}")
                                    self._append_run_log(
                                        f"[BEST][E{episode+1} S{step+1}] "
                                        f"train_ic={ic:.4f} icir={icir:.4f} "
                                        f"valid_ic={valid_ic:.4f} valid_icir={valid_icir:.4f} "
                                        f"train_days={train_days} valid_days={valid_days} "
                                        f"expr={current_expr}"
                                    )

                                if (ic > min_train_ic and
                                    valid_ic > min_valid_ic and
                                    train_days >= 10 and valid_days >= 5 and
                                    ic * valid_ic > 0 and
                                    len(self.factor_pool) < self.config['pool_size']):

                                    expr_exists = False
                                    for factor in self.factor_pool:
                                        if str(factor['expr']) == str(current_expr):
                                            expr_exists = True
                                            break

                                    if not expr_exists:
                                        quality, quality_msg = self.evaluate_factor_quality(
                                            current_expr, ic, valid_ic, train_days, valid_days
                                        )

                                        self.factor_pool.append({
                                            'expr': current_expr,
                                            'ic': ic,
                                            'icir': icir,
                                            'valid_ic': valid_ic,
                                            'valid_icir': valid_icir,
                                            'factor': factor_df,
                                            'depth': depth,
                                            'quality': quality,
                                            'quality_msg': quality_msg,
                                            'train_days': train_days,
                                            'valid_days': valid_days
                                        })

                        episode_rewards.append(reward)

                        forced_action = None
                        if self.agent is not None:
                            try:
                                action, log_prob, value = self.agent.act(current_expr)
                                forced_action = action
                                done = (step == steps_per_episode - 1)
                                self.agent.store_transition(current_expr, action, log_prob, value, reward, done)
                            except Exception as e:
                                if verbose and episode % 20 == 0:
                                    print(f"  PPO动作选择失败: {e}")

                        if step < steps_per_episode - 1:
                            new_expr = self._mutate_expr(current_expr, mutation_rate=0.4, forced_action=forced_action)
                            valid, _ = self.grammar.validate_expression(new_expr)
                            if valid:
                                current_expr = new_expr
                            else:
                                current_expr = random.choice(self.seed_exprs)
                    else:
                        prev_expr = current_expr
                        forced_action = None
                        log_prob = None
                        value = None
                        if self.agent is not None:
                            try:
                                action, log_prob, value = self.agent.act(current_expr)
                                forced_action = action
                            except Exception as e:
                                if verbose and episode % 20 == 0:
                                    print(f"  PPO动作选择失败: {e}")

                        candidate_exprs = []
                        for i in range(batch_size):
                            cand = self._mutate_expr(
                                current_expr,
                                mutation_rate=0.4,
                                forced_action=forced_action if i == 0 else None
                            )
                            valid, _ = self.grammar.validate_expression(cand)
                            if not valid:
                                cand = random.choice(self.seed_exprs)
                            candidate_exprs.append(cand)

                        backend_cfg = str(self.config.get('parallel_backend', 'process')).lower()
                        # auto：在当前实现里等价于 process（且会走 IC-only 以避免 IPC 传输巨大 DataFrame）
                        backend = 'process' if backend_cfg in ('auto', 'process') else backend_cfg
                        # process: 用 IC-only 多进程（避免 IPC 传输巨大 factor_df）
                        if backend == 'process':
                            train_metrics = self.evaluate_expr_batch_metrics(
                                candidate_exprs,
                                self.train_data,
                                pool_name='train',
                                backend='process'
                            )
                            stats = []
                            for m in train_metrics:
                                stats.append({
                                    'expr': m['expr'],
                                    'factor_df': None,  # 需要时再在主进程 evaluate_expr
                                    'ic': float(m.get('ic', 0.0)),
                                    'icir': float(m.get('icir', 0.0)),
                                    'train_days': int(m.get('days', 0)),
                                    'valid_ic': 0.0,
                                    'valid_icir': 0.0,
                                    'valid_days': 0
                                })
                        else:
                            eval_pairs = self.evaluate_expr_batch(
                                candidate_exprs,
                                self.train_data,
                                pool_name='train',
                                backend=backend
                            )
                            stats = []
                            for expr, factor_df in eval_pairs:
                                if factor_df is None or factor_df.empty:
                                    ic = 0
                                    icir = 0
                                    train_days = 0
                                else:
                                    ic, icir, train_days = self.calculate_ic(
                                        factor_df, self.train_data, verbose=False
                                    )
                                stats.append({
                                    'expr': expr,
                                    'factor_df': factor_df,
                                    'ic': ic,
                                    'icir': icir,
                                    'train_days': train_days,
                                    'valid_ic': 0,
                                    'valid_icir': 0,
                                    'valid_days': 0
                                })

                        if self.valid_data is not None:
                            valid_indices = [
                                i for i, s in enumerate(stats)
                                if s['ic'] > min_train_ic
                                and s['train_days'] >= 10
                            ]
                            if valid_indices:
                                valid_exprs = [candidate_exprs[i] for i in valid_indices]
                                if backend == 'process':
                                    valid_metrics = self.evaluate_expr_batch_metrics(
                                        valid_exprs,
                                        self.valid_data,
                                        pool_name='valid',
                                        backend='process'
                                    )
                                    for idx, m in zip(valid_indices, valid_metrics):
                                        stats[idx]['valid_ic'] = float(m.get('ic', 0.0))
                                        stats[idx]['valid_icir'] = float(m.get('icir', 0.0))
                                        stats[idx]['valid_days'] = int(m.get('days', 0))
                                else:
                                    valid_pairs = self.evaluate_expr_batch(
                                        valid_exprs,
                                        self.valid_data,
                                        pool_name='valid',
                                        backend=backend
                                    )
                                    for idx, (_, valid_factor) in zip(valid_indices, valid_pairs):
                                        if valid_factor is not None and not valid_factor.empty:
                                            v_ic, v_icir, v_days = self.calculate_ic(
                                                valid_factor, self.valid_data, verbose=False
                                            )
                                        else:
                                            v_ic, v_icir, v_days = 0, 0, 0
                                        stats[idx]['valid_ic'] = v_ic
                                        stats[idx]['valid_icir'] = v_icir
                                        stats[idx]['valid_days'] = v_days

                        rewards = []
                        for s in stats:
                            depth = self._get_expr_depth(s['expr'])
                            reward = self.calculate_reward(
                                s['ic'], s['expr'], s['icir'], s['valid_ic'], depth,
                                s['train_days'], s['valid_days'], verbose=False
                            )
                            s['depth'] = depth
                            s['reward'] = reward
                            rewards.append(reward)

                        best_idx = int(np.argmax(rewards)) if rewards else 0
                        best_stat = stats[best_idx]
                        current_expr = best_stat['expr']
                        reward = best_stat['reward']

                        episode_rewards.append(reward)
                        episode_ics.append(best_stat['ic'])
                        episode_valid_ics.append(best_stat['valid_ic'])

                        if self.agent is not None and log_prob is not None and value is not None:
                            done = (step == steps_per_episode - 1)
                            self.agent.store_transition(prev_expr, forced_action, log_prob, value, reward, done)

                        for s in stats:
                            ic = s['ic']
                            valid_ic = s['valid_ic']
                            train_days = s['train_days']
                            valid_days = s['valid_days']
                            factor_df = s['factor_df']
                            depth = s.get('depth', None)
                            icir = s['icir']
                            valid_icir = s['valid_icir']
                            expr = s['expr']

                            if (valid_ic > self.best_valid_ic and
                                ic > min_train_ic and
                                train_days >= 10 and valid_days >= 5 and
                                ic * valid_ic > 0):

                                if factor_df is None and backend == 'process':
                                    # 仅当需要落盘/入池时，才在主进程生成 factor_df（可复用主进程 LRU cache）
                                    factor_df = self.evaluate_expr(expr, self.train_data, verbose=False)
                                    s['factor_df'] = factor_df

                                self.best_valid_ic = valid_ic
                                self.best_train_ic = ic
                                self.best_expr = expr
                                self.best_factor = factor_df

                                if valid_ic > 0.02 and verbose:
                                    quality, quality_msg = self.evaluate_factor_quality(
                                        expr, ic, valid_ic, train_days, valid_days
                                    )
                                    if abs(ic) > 0.001:
                                        overfit_ratio = 1.0 - abs(valid_ic) / (abs(ic) + 1e-10)
                                    else:
                                        overfit_ratio = 1.0
                                    print(f"\n🎉 [轮次 {episode+1}] 新最佳验证IC: {valid_ic:.4f}")
                                    print(f"    训练IC: {ic:.4f}, ICIR: {icir:.4f}")
                                    print(f"    验证ICIR: {valid_icir:.4f}")
                                    print(f"    有效天数: 训练{train_days}, 验证{valid_days}")
                                    print(f"    过拟合比例: {overfit_ratio:.2%}")
                                    print(f"    质量评估: {quality} - {quality_msg}")
                                    print(f"    表达式: {expr}")
                                    self._append_run_log(
                                        f"[BEST][E{episode+1} S{step+1}] "
                                        f"train_ic={ic:.4f} icir={icir:.4f} "
                                        f"valid_ic={valid_ic:.4f} valid_icir={valid_icir:.4f} "
                                        f"train_days={train_days} valid_days={valid_days} "
                                        f"expr={expr}"
                                    )

                            if (ic > min_train_ic and
                                valid_ic > min_valid_ic and
                                train_days >= 10 and valid_days >= 5 and
                                ic * valid_ic > 0 and
                                len(self.factor_pool) < self.config['pool_size']):

                                if (not accept_all) and (expr != current_expr):
                                    continue

                                expr_exists = False
                                for factor in self.factor_pool:
                                    if str(factor['expr']) == str(expr):
                                        expr_exists = True
                                        break

                                if not expr_exists:
                                    if factor_df is None and backend == 'process':
                                        factor_df = self.evaluate_expr(expr, self.train_data, verbose=False)
                                        s['factor_df'] = factor_df
                                    quality, quality_msg = self.evaluate_factor_quality(
                                        expr, ic, valid_ic, train_days, valid_days
                                    )

                                    self.factor_pool.append({
                                        'expr': expr,
                                        'ic': ic,
                                        'icir': icir,
                                        'valid_ic': valid_ic,
                                        'valid_icir': valid_icir,
                                        'factor': factor_df,
                                        'depth': depth,
                                        'quality': quality,
                                        'quality_msg': quality_msg,
                                        'train_days': train_days,
                                        'valid_days': valid_days
                                    })
                                    self._append_run_log(
                                        f"[POOL][E{episode+1} S{step+1}] "
                                        f"train_ic={ic:.4f} icir={icir:.4f} "
                                        f"valid_ic={valid_ic:.4f} valid_icir={valid_icir:.4f} "
                                        f"train_days={train_days} valid_days={valid_days} "
                                        f"expr={expr}"
                                    )
                                    self._append_run_log(
                                        f"[POOL][E{episode+1} S{step+1}] "
                                        f"train_ic={ic:.4f} icir={icir:.4f} "
                                        f"valid_ic={valid_ic:.4f} valid_icir={valid_icir:.4f} "
                                        f"train_days={train_days} valid_days={valid_days} "
                                        f"expr={current_expr}"
                                    )
                
                except Exception as e:
                    if verbose and episode % 20 == 0 and step == 0:
                        print(f"  步骤错误: {e}")
                    current_expr = random.choice(self.seed_exprs)
                    reward = -1.0
                    episode_rewards.append(reward)
            
            if self.agent is not None and episode % 5 == 0 and episode > 4:
                try:
                    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                    next_value_est = avg_reward * 3
                    loss = self.agent.update(next_value_est, num_epochs=2, batch_size=16)
                    
                    if loss > 0:
                        self.training_stats['losses'].append(loss)
                        print(f"🧠 PPO实时损失: {loss:.6f}")
                        self._append_run_log(
                            f"[LOSS][E{episode+1}] loss={loss:.6f}"
                        )
                except Exception as e:
                    if verbose and episode % 20 == 0:
                        print(f"  PPO更新失败: {e}")
            
            if episode % 10 == 0 and episode > 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                avg_ic = np.mean([ic for ic in episode_ics if ic > -1]) if episode_ics else 0
                avg_valid_ic = np.mean([ic for ic in episode_valid_ics if ic > -1]) if episode_valid_ics else 0
                
                self.training_stats['rewards'].append(avg_reward)
                
                if self.best_train_ic > -999:
                    self.training_stats['ic_history'].append(self.best_train_ic)
                    if self.best_valid_ic > -999:
                        self.training_stats['valid_ic_history'].append(self.best_valid_ic)
                        if self.best_train_ic > 0.001:
                            overfit_ratio = 1.0 - abs(self.best_valid_ic) / (abs(self.best_train_ic) + 1e-10)
                            self.training_stats['overfit_ratios'].append(overfit_ratio)
                
                if self.best_expr is not None:
                    quality, _ = self.evaluate_factor_quality(
                        self.best_expr, self.best_train_ic, self.best_valid_ic, 0, 0
                    )
                    self.training_stats['expr_quality'].append(quality)
                
                if episode % 20 == 0 and verbose:
                    print(f"\n📊 [轮次 {episode}] 诊断指标:")
                    print(f"   平均奖励: {avg_reward:.4f}")
                    print(f"   平均训练IC: {avg_ic:.4f}")
                    print(f"   平均验证IC: {avg_valid_ic:.4f}")
                    
                    if self.agent is not None and self.training_stats['losses']:
                        print(f"   PPO损失: {self.training_stats['losses'][-1]:.4f}")
                    
                    print(f"   最佳训练IC: {self.best_train_ic:.4f}")
                    print(f"   最佳验证IC: {self.best_valid_ic:.4f}")
                    print(f"   因子池大小: {len(self.factor_pool)}")
                    
                    if self.factor_pool:
                        quality_counts = {}
                        for factor in self.factor_pool:
                            quality = factor.get('quality', 'UNKNOWN')
                            quality_counts[quality] = quality_counts.get(quality, 0) + 1
                        
                        print(f"   因子质量分布: {quality_counts}")
                        
                        consistent_count = sum(1 for f in self.factor_pool 
                                              if f.get('ic', 0) * f.get('valid_ic', 0) > 0)
                        total_count = len(self.factor_pool)
                        if total_count > 0:
                            print(f"   IC符号一致性: {consistent_count}/{total_count} ({consistent_count/total_count*100:.1f}%)")
        
        print(f"\n✅ 搜索完成，最佳验证IC: {self.best_valid_ic:.4f}")
        print(f"    对应训练IC: {self.best_train_ic:.4f}")
        
        if self.best_expr is not None:
            factor_df = self.evaluate_expr(self.best_expr, self.train_data, verbose=False)
            if factor_df is not None:
                ic, icir, train_days = self.calculate_ic(factor_df, self.train_data, verbose=False)
                valid_ic = 0
                valid_days = 0
                if self.valid_data is not None:
                    valid_factor = self.evaluate_expr(self.best_expr, self.valid_data, verbose=False)
                    if valid_factor is not None:
                        valid_ic, _, valid_days = self.calculate_ic(valid_factor, self.valid_data, verbose=False)
                
                quality, quality_msg = self.evaluate_factor_quality(
                    self.best_expr, ic, valid_ic, train_days, valid_days
                )
                print(f"    最佳因子质量: {quality} - {quality_msg}")
        
        return self.best_expr, self.best_valid_ic
    
    def _mutate_expr(self, expr, mutation_rate=0.3, forced_action=None):
        """优化版变异逻辑"""
        if forced_action is None and random.random() > mutation_rate:
            return expr

        action_map = {0: 'replace', 1: 'add', 2: 'remove', 3: 'swap'}
        
        if forced_action is not None:
            m_type = action_map.get(forced_action, 'replace')
        else:
            m_type = random.choices(['replace', 'add', 'remove', 'swap'], weights=[0.4, 0.3, 0.1, 0.2])[0]

        if m_type == 'replace':
            if isinstance(expr, tuple):
                idx = random.randint(0, len(expr) - 1)
                new_expr = list(expr)
                
                if idx == 0:
                    op = expr[0]
                    if op in self.grammar.unary_ops:
                        new_expr[0] = random.choice(list(self.grammar.unary_ops.keys()))
                    elif op in self.grammar.binary_sym_ops or op in self.grammar.binary_asym_ops:
                        ops = list(self.grammar.binary_sym_ops.keys()) + list(self.grammar.binary_asym_ops.keys())
                        new_expr[0] = random.choice(ops)
                    elif op in self.grammar.rolling_ops:
                        new_expr[0] = random.choice(list(self.grammar.rolling_ops.keys()))
                    elif op in self.grammar.ternary_ops:
                        new_expr[0] = random.choice(list(self.grammar.ternary_ops.keys()))
                else:
                    op = expr[0]
                    is_window = False
                    if (op in self.grammar.rolling_ops and idx == 2) or \
                       (op in self.grammar.paired_rolling_ops and idx == 3):
                        is_window = True
                    
                    if is_window:
                        new_expr[idx] = random.choice(self.grammar.windows)
                    else:
                        new_expr[idx] = self.grammar.generate_random_expr(max_depth=1)
                return tuple(new_expr)
            else:
                return self.grammar.generate_random_expr(max_depth=0)

        elif m_type == 'add':
            op = random.choice(['add', 'sub', 'mul', 'max', 'min'])
            new_node = self.grammar.generate_random_expr(max_depth=1)
            if random.random() < 0.5:
                return (op, expr, new_node)
            else:
                return (op, new_node, expr)

        elif m_type == 'remove':
            if isinstance(expr, tuple):
                if len(expr) >= 3:
                    return expr[1]
            return expr

        elif m_type == 'swap':
            if isinstance(expr, tuple) and len(expr) == 3:
                return (expr[0], expr[2], expr[1])
        
        return expr

    def analyze_results(self):
        """分析搜索结果（深度修复版）"""
        print("\n" + "=" * 60)
        print("🔍 搜索结果深度分析")
        print("=" * 60)
        
        if not self.factor_pool:
            print("❌ 因子池为空，没有找到有效因子")
            return
        
        sorted_factors = sorted(self.factor_pool, key=lambda x: x.get('valid_ic', -999), reverse=True)
        
        print(f"📊 共发现 {len(sorted_factors)} 个有效因子")
        
        print("\n🏆 最佳因子排名:")
        for i, factor in enumerate(sorted_factors[:5]):
            print(f"\n  #{i+1}:")
            print(f"    验证IC: {factor['valid_ic']:.4f}")
            print(f"    训练IC: {factor['ic']:.4f}")
            print(f"    ICIR: {factor['icir']:.4f}")
            print(f"    验证ICIR: {factor.get('valid_icir', 0):.4f}")
            print(f"    深度: {factor['depth']}")
            print(f"    有效天数: 训练{factor.get('train_days', 0)}, 验证{factor.get('valid_days', 0)}")
            print(f"    质量: {factor.get('quality', 'UNKNOWN')} - {factor.get('quality_msg', '')}")
            print(f"    表达式: {factor['expr']}")
        
        print("\n📈 统计分析:")
        valid_ics = [f['valid_ic'] for f in sorted_factors if f['valid_ic'] > -999]
        train_ics = [f['ic'] for f in sorted_factors]
        
        if valid_ics:
            print(f"    验证IC均值: {np.mean(valid_ics):.4f}")
            print(f"    验证IC标准差: {np.std(valid_ics):.4f}")
            print(f"    验证IC最大值: {max(valid_ics):.4f}")
            print(f"    验证IC最小值: {min(valid_ics):.4f}")
        
        print(f"    训练IC均值: {np.mean(train_ics):.4f}")
        print(f"    训练IC标准差: {np.std(train_ics):.4f}")
        
        consistent_count = sum(1 for f in sorted_factors if f.get('ic', 0) * f.get('valid_ic', 0) > 0)
        total_count = len(sorted_factors)
        print(f"    IC符号一致性: {consistent_count}/{total_count} ({consistent_count/total_count*100:.1f}%)")
        
        overfit_ratios = []
        for factor in sorted_factors:
            if abs(factor['ic']) > 0.001:
                overfit_ratio = 1.0 - abs(factor['valid_ic']) / (abs(factor['ic']) + 1e-10)
                overfit_ratios.append(overfit_ratio)
        
        if overfit_ratios:
            print(f"    平均过拟合比例: {np.mean(overfit_ratios):.2%}")
            print(f"    过拟合比例>50%: {sum(1 for r in overfit_ratios if r > 0.5)} 个")
        
        print("\n📊 因子质量分布:")
        quality_counts = {}
        for factor in sorted_factors:
            quality = factor.get('quality', 'UNKNOWN')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        for quality, count in quality_counts.items():
            print(f"    {quality}: {count} 个 ({count/len(sorted_factors)*100:.1f}%)")
        
        print("\n📏 表达式深度分布:")
        depth_counts = {}
        for factor in sorted_factors:
            depth = factor['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        for depth, count in sorted(depth_counts.items()):
            print(f"    深度{depth}: {count} 个 ({count/len(sorted_factors)*100:.1f}%)")


# ==================== 4. 结果保存 ====================

def save_results(results, output_dir):
    """保存训练结果"""
    print(f"\n💾 保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1) 文本摘要
        with open(os.path.join(output_dir, 'best_expr.txt'), 'w', encoding='utf-8') as f:
            f.write(f"最佳表达式:\n{results.get('best_expr')}\n\n")
            f.write(f"训练IC: {results.get('train_ic', 0.0):.4f}\n")
            f.write(f"验证IC: {results.get('valid_ic', 0.0):.4f}\n")
            if 'train_icir' in results:
                f.write(f"训练ICIR: {results.get('train_icir', 0.0):.4f}\n")
            if 'valid_icir' in results:
                f.write(f"验证ICIR: {results.get('valid_icir', 0.0):.4f}\n")
            if 'train_days' in results:
                f.write(f"训练有效天数: {int(results.get('train_days', 0))}\n")
            if 'valid_days' in results:
                f.write(f"验证有效天数: {int(results.get('valid_days', 0))}\n")
            if 'overfit_ratio' in results and results.get('overfit_ratio') is not None:
                f.write(f"过拟合比例: {float(results.get('overfit_ratio', 0.0)):.4%}\n")
            if results.get('quality'):
                f.write(f"质量评估: {results.get('quality')} - {results.get('quality_msg','')}\n")
        
        # 2) 因子池
        if 'factor_pool' in results and results['factor_pool']:
            with open(os.path.join(output_dir, 'factor_pool.txt'), 'w', encoding='utf-8') as f:
                f.write(f"因子池大小: {len(results['factor_pool'])}\n\n")
                for i, factor in enumerate(results['factor_pool'], 1):
                    f.write(f"因子 {i}:\n")
                    f.write(f"  表达式: {factor['expr']}\n")
                    f.write(f"  训练IC: {factor['ic']:.4f}\n")
                    f.write(f"  验证IC: {factor['valid_ic']:.4f}\n\n")
        
        # 3) 最优因子值（训练/验证）
        if 'best_factor' in results and results['best_factor'] is not None:
            results['best_factor'].to_csv(os.path.join(output_dir, 'best_factor_values.csv'))

        if results.get('best_factor_valid') is not None:
            results['best_factor_valid'].to_csv(os.path.join(output_dir, 'best_factor_values_valid.csv'))

        # 4) 结构化报告（便于你后处理/画图）
        report = {
            k: v for k, v in results.items()
            if k not in {'best_factor', 'best_factor_valid', 'factor_pool'}  # 大对象不进 json
        }
        with open(os.path.join(output_dir, 'best_validation_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 结果保存成功！")
    except Exception as e:
        print(f"❌ 保存失败: {e}")


# ==================== 5. 主函数 ====================

def main():
    # Windows 多进程必须调用 freeze_support
    mp.freeze_support()
    
    # 设置启动方法为 spawn（Windows 默认，但显式设置更保险）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    parser = argparse.ArgumentParser(description='AlphaCFG 独立训练模块 - 高性能PC版')
    parser.add_argument('--data', type=str, help='单个数据文件路径')
    parser.add_argument('--data-dir', type=str, default="xsz2", help='包含多个分块文件的目录路径')
    parser.add_argument('--output', type=str, default='results_pro', help='输出目录')

    # === 升级的默认参数 ===
    parser.add_argument('--split', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--episodes', type=int, default=500, help='训练轮数 (建议 500+)')
    parser.add_argument('--steps', type=int, default=10, help='每轮步数 (建议 10+)')
    parser.add_argument('--pool_size', type=int, default=100, help='因子池大小 (建议 100+)')
    parser.add_argument('--max_depth', type=int, default=6, help='最大公式深度 (建议 6-8)')
    parser.add_argument('--batch-size', type=int, default=64, help='并行评估的批量大小 (越大越快)')
    parser.add_argument('--parallel-backend', type=str, default='auto', choices=['auto', 'serial', 'thread', 'process'],
                        help='候选表达式批量评估并行后端：auto/serial/thread/process（Windows 想吃满 8 核通常用 process）')
    parser.add_argument('--parallel-workers', type=int, default=None,
                        help='并行 worker 数；None=自动（推荐）。')
    parser.add_argument('--process-workers-cap-windows', type=int, default=None,
                        help='Windows 下 process 的 worker 上限；None=不额外限制（更吃内存）。')

    parser.add_argument('--verbose', action='store_true', default=True, help='详细输出')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # PPO/网络参数升级
    parser.add_argument('--use-ppo', action='store_true', default=True, help='启用PPO')
    parser.add_argument('--ic-period', type=int, default=5, help='IC周期')
    parser.add_argument('--ppo-update-every', type=int, default=10, help='PPO更新频率')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO更新轮数')
    parser.add_argument('--ppo-batch-size', type=int, default=64, help='PPO Batch Size')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM隐藏层维度')

    args = parser.parse_args()

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch:
            torch.manual_seed(args.seed)
    except Exception:
        pass

    # 运行日志目录: result-YYYYMMDD-尝试次数
    date_tag = time.strftime("%Y%m%d")
    base_prefix = f"result-{date_tag}-"
    try:
        existing = [
            d for d in os.listdir(".")
            if os.path.isdir(d) and d.startswith(base_prefix)
        ]
    except Exception:
        existing = []
    next_idx = 1
    for d in existing:
        try:
            suffix = d.split(base_prefix, 1)[1]
            next_idx = max(next_idx, int(suffix) + 1)
        except Exception:
            continue
    run_log_dir = f"{base_prefix}{next_idx}"
    try:
        os.makedirs(run_log_dir, exist_ok=True)
    except Exception:
        run_log_dir = None
    
    if not args.data and not args.data_dir:
        print("❌ 必须指定 --data 或 --data-dir")
        return
    
    if args.data and args.data_dir:
        print("❌ 不能同时指定 --data 和 --data-dir")
        return
    
    print(f"\n🚀 启动高性能训练模式:")
    print(f"  CPU核数: {os.cpu_count()}")
    print(f"  训练轮数: {args.episodes}")
    print(f"  种群大小: {args.pool_size}")
    print(f"  最大深度: {args.max_depth}")
    print(f"  并行批量: {args.batch_size}")
    if run_log_dir:
        print(f"  运行日志: {run_log_dir}")
    
    # 加载数据
    if args.data_dir:
        data_pool, common_dates = load_and_merge_chunks(args.data_dir)
    else:
        data_pool, common_dates = load_data(args.data)
    
    if data_pool is None:
        return
    
    # 拆分数据
    train_data, valid_data, _, _ = split_train_valid(data_pool, common_dates, args.split, preprocess=True)
    
    # 增强配置
    config = {
        'max_depth': args.max_depth,
        'pool_size': args.pool_size,
        'embedding_dim': 48,
        'hidden_dim': args.hidden_dim,
        'num_layers': 2,
        'overfit_penalty': 1.5,
        'complexity_reward': 0.1,
        'ic_consistency_weight': 2.5,
        'ic_period': args.ic_period,
        'min_train_ic': 0.025,
        'min_valid_ic': 0.02,
        'min_stocks': 10,
        'min_dates': 20,

        # PPO 配置
        'use_ppo': bool(args.use_ppo),
        'learning_rate': float(args.lr),
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'ppo_update_every': args.ppo_update_every,
        'ppo_epochs': args.ppo_epochs,
        'ppo_batch_size': args.ppo_batch_size,

        # === 高性能并行配置 ===
        'enable_expr_cache': True,
        'factor_cache_size': 1024,
        'parallel_eval': True,
        'candidate_batch_size': args.batch_size,
        'parallel_backend': str(args.parallel_backend).lower(),
        'parallel_workers': None,  # 下面统一计算默认值/安全上限
        'process_workers_cap_windows': args.process_workers_cap_windows,
        'accept_all_candidates': True,
        'debug_parallel': True,
        'run_log_dir': run_log_dir,
    }

    # === 并行 worker 默认值 ===
    cpu_n = (os.cpu_count() or 1)
    if args.parallel_workers is not None:
        config['parallel_workers'] = max(1, int(args.parallel_workers))
    else:
        # ⚠️ 修复：Windows多进程不要开太多worker，避免调度开销
        # 推荐：4-8核CPU用4个worker，16核CPU用6-8个worker
        if os.name == 'nt':  # Windows
            # 保守策略：不超过CPU核心数的50%，且上限为8
            config['parallel_workers'] = min(8, max(2, cpu_n // 2))
            print(f"💡 [配置优化] Windows平台，自动设置 parallel_workers={config['parallel_workers']} (CPU核心数: {cpu_n})")
            print(f"   理由：Windows进程创建开销大，适度并行即可达到最佳性能")
        else:  # Linux/Mac
            # Linux/Mac 可以更激进，fork模式开销小
            config['parallel_workers'] = max(1, cpu_n - 1)
            print(f"💡 [配置优化] Linux/Mac平台，自动设置 parallel_workers={config['parallel_workers']} (CPU核心数: {cpu_n})")
    
    # 初始化并搜索
    agent = AlphaCFG(train_data=train_data, valid_data=valid_data, config=config)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        print(f"  当前内存占用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    except Exception:
        pass

    best_expr, best_valid_ic = agent.search(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        verbose=args.verbose
    )

    # ====== 最优因子验证（尽量 1:1 对齐 v4/main01.py 的“最终评估最佳因子”输出风格）======
    best_train_factor = None
    best_valid_factor = None
    train_ic = float(agent.best_train_ic) if agent.best_train_ic is not None else 0.0
    train_icir = 0.0
    train_days = 0
    valid_ic = float(best_valid_ic) if best_valid_ic is not None else 0.0
    valid_icir = 0.0
    valid_days = 0
    overfit_ratio = None
    quality = None
    quality_msg = None
    train_metrics = None
    valid_metrics = None
    portfolio_period = 10   # ✅ main01 固定用 10 天做组合评估
    portfolio_top_pct = 0.2 # ✅ main01 固定用 20% 做多头组合

    if best_expr is not None:
        try:
            best_train_factor = agent.evaluate_expr(best_expr, agent.train_data, verbose=False)
            if best_train_factor is not None and not best_train_factor.empty:
                train_ic, train_icir, train_days = agent.calculate_ic(
                    best_train_factor, agent.train_data, period=args.ic_period, verbose=False
                )
                train_metrics = agent.calculate_portfolio_metrics(
                    best_train_factor, agent.train_data, period=portfolio_period, top_pct=portfolio_top_pct, verbose=False
                )
        except Exception:
            best_train_factor = None

        if agent.valid_data is not None:
            try:
                best_valid_factor = agent.evaluate_expr(best_expr, agent.valid_data, verbose=False)
                if best_valid_factor is not None and not best_valid_factor.empty:
                    valid_ic, valid_icir, valid_days = agent.calculate_ic(
                        best_valid_factor, agent.valid_data, period=args.ic_period, verbose=False
                    )
                    valid_metrics = agent.calculate_portfolio_metrics(
                        best_valid_factor, agent.valid_data, period=portfolio_period, top_pct=portfolio_top_pct, verbose=False
                    )
            except Exception:
                best_valid_factor = None

        try:
            if abs(train_ic) > 0.001:
                overfit_ratio = 1.0 - abs(valid_ic) / (abs(train_ic) + 1e-10)
            else:
                overfit_ratio = 1.0
            quality, quality_msg = agent.evaluate_factor_quality(best_expr, train_ic, valid_ic, train_days, valid_days)
        except Exception:
            overfit_ratio = None
            quality, quality_msg = None, None

        # ===== main01 风格输出（含分段评估与组合对比）=====
        print("\n" + "=" * 60)
        print("🏆 最佳Alpha因子详细信息:")
        print(f"   验证IC: {valid_ic:.4f}")
        print(f"   训练IC: {train_ic:.4f}")

        if abs(train_ic) > 0.001:
            _ofr = 1.0 - abs(valid_ic) / (abs(train_ic) + 1e-10)
        else:
            _ofr = 1.0
        print(f"   过拟合比例: {_ofr:.2%}")

        if quality:
            print(f"   质量评估: {quality} - {quality_msg}")

        try:
            expr_depth = agent._get_expr_depth(best_expr)
        except Exception:
            expr_depth = None
        if expr_depth is not None:
            print(f"   表达式深度: {expr_depth}")

        print(f"   有效天数: 训练{train_days}, 验证{valid_days}")
        print(f"   表达式: {best_expr}")

        print("\n🔍 因子详细评估:")

        # 评估信号强度（按 main01 的阈值分桶）
        print("\n  信号强度评估:")
        if valid_ic > 0.03:
            print("    ✅ 强信号 (验证IC > 0.03)")
        elif valid_ic > 0.025:
            print("    👍 中等偏强信号 (验证IC > 0.025)")
        elif valid_ic > 0.02:
            print("    📈 中等信号 (验证IC > 0.02)")
        elif valid_ic > 0.015:
            print("    📊 弱信号 (验证IC > 0.015)")
        else:
            print("    ⚠️ 微弱信号 (验证IC <= 0.015)")

        # 评估稳健性（按 main01 的 overfit_ratio 阈值分桶）
        print("\n  稳健性评估:")
        if _ofr < 0.3:
            print("    ✅ 高稳健性 (过拟合比例 < 30%)")
        elif _ofr < 0.4:
            print("    👍 中等稳健性 (过拟合比例 < 40%)")
        elif _ofr < 0.5:
            print("    📊 一般稳健性 (过拟合比例 < 50%)")
        elif _ofr < 0.7:
            print("    ⚠️ 低稳健性 (过拟合比例 < 70%)")
        else:
            print("    ❌ 严重过拟合 (过拟合比例 >= 70%)")

        # 评估IC一致性
        print("\n  IC一致性评估:")
        if train_ic * valid_ic > 0:
            print("    ✅ IC符号一致")
        else:
            print("    ❌ IC符号不一致 (严重警告)")

        # ===== 组合表现（main01 风格：period 固定 10 天）=====
        print("\n" + "=" * 60)
        print("💰 组合表现评估（基于最佳因子）")
        print("=" * 60)

        if best_train_factor is not None and train_metrics is None:
            train_metrics = agent.calculate_portfolio_metrics(
                best_train_factor, agent.train_data, period=portfolio_period, top_pct=portfolio_top_pct, verbose=False
            )
        if best_valid_factor is not None and valid_metrics is None:
            valid_metrics = agent.calculate_portfolio_metrics(
                best_valid_factor, agent.valid_data, period=portfolio_period, top_pct=portfolio_top_pct, verbose=False
            )

        print("\n📊 训练集组合表现:")
        if train_metrics:
            print(f"     平均收益率: {train_metrics.get('mean_return', 0.0):.4f} ({train_metrics.get('mean_return', 0.0)*100:.2f}%)")
            # 复合收益率=累计收益率（复利乘积）
            print(f"     复合/累计收益率: {train_metrics.get('cumulative_return', 0.0):.4f} ({train_metrics.get('cumulative_return', 0.0)*100:.2f}%)")
            print(f"     年化收益率: {train_metrics.get('annualized_return', 0.0):.4f} ({train_metrics.get('annualized_return', 0.0)*100:.2f}%)")
            print(f"     年化波动率: {train_metrics.get('annualized_volatility', 0.0):.4f} ({train_metrics.get('annualized_volatility', 0.0)*100:.2f}%)")
            print(f"     年化夏普比率: {train_metrics.get('annualized_sharpe', 0.0):.4f}")
            print(f"     最大回撤: {train_metrics.get('max_drawdown', 0.0):.4f} ({train_metrics.get('max_drawdown', 0.0)*100:.2f}%)")
            print(f"     胜率: {train_metrics.get('win_rate', 0.0):.4f} ({train_metrics.get('win_rate', 0.0)*100:.2f}%)")
            print(f"     盈亏比: {train_metrics.get('profit_loss_ratio', 0.0):.4f}")
            print(f"     卡玛比率: {train_metrics.get('calmar_ratio', 0.0):.4f}")
        else:
            print("     ⚠️ 组合指标不可用（可能是：有效交易日/有效股票不足，或当日可交易股票过少导致组合无法构建）")

        print("\n📊 验证集组合表现:")
        if valid_metrics:
            print(f"     平均收益率: {valid_metrics.get('mean_return', 0.0):.4f} ({valid_metrics.get('mean_return', 0.0)*100:.2f}%)")
            print(f"     复合/累计收益率: {valid_metrics.get('cumulative_return', 0.0):.4f} ({valid_metrics.get('cumulative_return', 0.0)*100:.2f}%)")
            print(f"     年化收益率: {valid_metrics.get('annualized_return', 0.0):.4f} ({valid_metrics.get('annualized_return', 0.0)*100:.2f}%)")
            print(f"     年化波动率: {valid_metrics.get('annualized_volatility', 0.0):.4f} ({valid_metrics.get('annualized_volatility', 0.0)*100:.2f}%)")
            print(f"     年化夏普比率: {valid_metrics.get('annualized_sharpe', 0.0):.4f}")
            print(f"     最大回撤: {valid_metrics.get('max_drawdown', 0.0):.4f} ({valid_metrics.get('max_drawdown', 0.0)*100:.2f}%)")
            print(f"     胜率: {valid_metrics.get('win_rate', 0.0):.4f} ({valid_metrics.get('win_rate', 0.0)*100:.2f}%)")
            print(f"     盈亏比: {valid_metrics.get('profit_loss_ratio', 0.0):.4f}")
            print(f"     卡玛比率: {valid_metrics.get('calmar_ratio', 0.0):.4f}")
        else:
            print("     ⚠️ 组合指标不可用（可能是：有效交易日/有效股票不足，或当日可交易股票过少导致组合无法构建）")

        if train_metrics and valid_metrics:
            print("\n📊 样本内外表现对比:")
            print(f"   年化收益率: 训练 {train_metrics.get('annualized_return', 0.0)*100:.2f}% vs 验证 {valid_metrics.get('annualized_return', 0.0)*100:.2f}%")
            print(f"   年化夏普: 训练 {train_metrics.get('annualized_sharpe', 0.0):.2f} vs 验证 {valid_metrics.get('annualized_sharpe', 0.0):.2f}")
            print(f"   最大回撤: 训练 {train_metrics.get('max_drawdown', 0.0)*100:.2f}% vs 验证 {valid_metrics.get('max_drawdown', 0.0)*100:.2f}%")
            print(f"   胜率: 训练 {train_metrics.get('win_rate', 0.0)*100:.1f}% vs 验证 {valid_metrics.get('win_rate', 0.0)*100:.1f}%")

            if train_metrics.get('annualized_return', 0.0) != 0:
                return_consistency = valid_metrics.get('annualized_return', 0.0) / train_metrics.get('annualized_return', 1e-10)
                print(f"   收益率一致性: {return_consistency:.2%}")

            if train_metrics.get('annualized_sharpe', 0.0) != 0:
                sharpe_consistency = valid_metrics.get('annualized_sharpe', 0.0) / train_metrics.get('annualized_sharpe', 1e-10)
                print(f"   夏普比率一致性: {sharpe_consistency:.2%}")
    
    # 保存结果
    results = {
        'best_expr': best_expr,
        # 用“复算”的结果作为最终口径（更像 main01 的最终评估）
        'train_ic': float(train_ic),
        'train_icir': float(train_icir),
        'train_days': int(train_days),
        'valid_ic': float(valid_ic),
        'valid_icir': float(valid_icir),
        'valid_days': int(valid_days),
        'overfit_ratio': overfit_ratio,
        'quality': quality,
        'quality_msg': quality_msg,
        'ic_period': int(args.ic_period),
        'portfolio_period': int(portfolio_period),
        'portfolio_top_pct': float(portfolio_top_pct),
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        # 原始对象（用于落盘 CSV）
        'best_factor': best_train_factor if best_train_factor is not None else agent.best_factor,
        'best_factor_valid': best_valid_factor,
        'factor_pool': agent.factor_pool,
        'training_stats': agent.training_stats,
    }
    
    save_results(results, args.output)
    
    print(f"\n{'='*80}")
    print("✅ 训练完成！")
    print(f"{'='*80}")
    if results.get('best_expr'):
        print(f"\n最佳表达式: {results['best_expr']}")
        print(f"验证IC: {results.get('valid_ic', 0):.4f}")
        print(f"训练IC: {results.get('train_ic', 0):.4f}")
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
