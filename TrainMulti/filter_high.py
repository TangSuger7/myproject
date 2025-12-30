import os
import sys
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import shutil
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count, Process, Queue, Manager
import multiprocessing as mp
import signal
import atexit

from config import Config
from SAM2UNeXT import SAM2UNeXT
from loss import PixLoss
from utils import path_to_image, check_state_dict

# 确保print输出不被缓冲，能够被tee捕获
def print_flush(*args, **kwargs):
    """带flush的print，确保输出立即写入，能被tee捕获"""
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

# 配置tqdm默认参数，确保输出能被tee捕获
def get_tqdm_kwargs(desc=""):
    """获取tqdm的默认参数，确保输出到stdout并能被tee捕获"""
    return {
        'file': sys.stdout,
        'ncols': 100,
        'mininterval': 0.5,  # 更频繁更新
        'maxinterval': 1.0,
        'dynamic_ncols': False,  # 固定宽度，避免频繁重绘
    }


def parse_args():
    parser = argparse.ArgumentParser(description='过滤loss较大的图片')
    
    # parser.add_argument('--input_txt', type=str, nargs='+', default=["/home/notebook/data/group/00group/users/myjyf/HFHOME2/trainall.txt","/home/notebook/data/group/00group/users/myjyf/Dataset/MYMatting/combineall1223.txt"], 
    parser.add_argument('--input_txt', type=str, nargs='+', default=["/home/notebook/data/group/00group/users/myjyf/Dataset/MYMattingADD/train_all1230.txt"], 
                        help='输入txt文件路径（支持多个，可以用列表形式或逗号分隔），每行格式：原图路径   mask路径。例如：--input_txt file1.txt file2.txt 或 --input_txt "file1.txt,file2.txt"')
    parser.add_argument('--checkpoint', type=str, default="/home/notebook/data/group/00group/users/myjyf/MultiSeg/lab3829/ckpt/tmp/epoch_30.pth", help='模型checkpoint路径')
    parser.add_argument('--loss_threshold', type=float, default=8, help='loss阈值，大于此值的图片将被过滤（默认0.5）')
    parser.add_argument('--batch_size', type=int, default=24, help='批处理大小（默认8）')
    parser.add_argument('--device', type=str, default='auto', help='设备（默认auto自动检测所有可用GPU，也可手动指定如cuda:0或cuda:0,1,2,3）')
    parser.add_argument('--output_dir', type=str, default='masks2', help='输出目录（默认masks2）')
    parser.add_argument('--save_filtered_txt', type=str, default=None, help='保存过滤后的图片路径到txt文件（可选）')
    parser.add_argument('--num_workers', type=int, default=16, help='数据加载的worker数量（默认4）')
    parser.add_argument('--merged_txt', type=str, default=None, help='融合后的txt文件保存路径（默认：脚本目录/merged_input.txt）')
    parser.add_argument('--use_fp16', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help='使用FP16半精度推理（加快速度，节省显存，默认True）')
    args = parser.parse_args()
    
    # 确保use_fp16是布尔值
    if isinstance(args.use_fp16, bool):
        pass  # 已经是布尔值
    elif isinstance(args.use_fp16, str):
        args.use_fp16 = args.use_fp16.lower() in ['true', '1', 'yes']
    else:
        args.use_fp16 = bool(args.use_fp16)
    
    # 处理input_txt：支持列表和逗号分隔两种格式
    if isinstance(args.input_txt, list):
        # 如果已经是列表，展开可能包含逗号分隔的项
        expanded_paths = []
        for path in args.input_txt:
            if ',' in path:
                # 如果路径中包含逗号，按逗号分割
                expanded_paths.extend([p.strip() for p in path.split(',')])
            else:
                expanded_paths.append(path.strip())
        args.input_txt = expanded_paths
    else:
        # 如果是单个字符串，尝试按逗号分割
        if ',' in args.input_txt:
            args.input_txt = [p.strip() for p in args.input_txt.split(',')]
        else:
            args.input_txt = [args.input_txt]
    
    return args


def load_model(checkpoint_path, device, config, use_multi_gpu=False, device_ids=None, use_fp16=False):
    """加载模型"""
    print(f"加载模型checkpoint: {checkpoint_path}")
    num_outputs = getattr(config, 'num_outputs', 1)
    model = SAM2UNeXT("sam2_hiera_large.pt", "model.safetensors", num_outputs=num_outputs)
    
    # 加载checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 多GPU支持
    if use_multi_gpu and device_ids is not None and len(device_ids) > 1:
        print(f"使用 {len(device_ids)} 个GPU进行并行计算: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    elif use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行并行计算")
        model = nn.DataParallel(model)
    
    # FP16支持
    if use_fp16 and device.type == 'cuda':
        # 检查CUDA是否支持FP16
        if torch.cuda.is_available():
            try:
                # 使用half()将模型转换为FP16
                model = model.half()
                print("已启用FP16半精度推理（加快速度，节省显存）")
            except Exception as e:
                print(f"警告: 模型转换为FP16失败: {e}，将使用FP32")
        else:
            print("警告: CUDA不可用，将使用FP32")
    
    print("模型加载完成")
    return model


class ImageMaskDataset(Dataset):
    """图片和mask数据集"""
    def __init__(self, image_mask_pairs, size=None):
        self.image_mask_pairs = image_mask_pairs
        self.size = size
        
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # 加载图片
        image = path_to_image(img_path, size=self.size, color_type='rgb')
        mask = path_to_image(mask_path, size=self.size, color_type='gray')
        
        # 转换为tensor
        image_tensor = self.transform_image(image)  # [3, H, W]
        mask_tensor = self.transform_label(mask)    # [1, H, W]
        
        return {
            'image_tensor': image_tensor,
            'mask_tensor': mask_tensor,
            'img_path': img_path,
            'mask_path': mask_path
        }


def load_image_pair(img_path, mask_path, size=None):
    """加载图片和mask对（保留用于兼容）"""
    # 加载图片
    image = path_to_image(img_path, size=size, color_type='rgb')
    mask = path_to_image(mask_path, size=size, color_type='gray')
    
    # 转换为tensor
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_label = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform_image(image).unsqueeze(0)  # [1, 3, H, W]
    mask_tensor = transform_label(mask).unsqueeze(0)    # [1, 1, H, W]
    
    return image_tensor, mask_tensor, image, mask


def compute_loss_batch(model, images, masks, loss_fn, device, epoch=0, use_fp16=False, return_preds=False):
    """批量计算loss，返回每个样本的loss值，可选返回预测结果"""
    batch_size = images.shape[0]
    images = images.to(device)
    masks = masks.to(device)
    
    # 检查模型是否使用FP16 - 更可靠的检测方法
    model_is_half = False
    if device.type == 'cuda':
        # 检查模型参数是否是half类型
        try:
            # 处理DataParallel的情况
            if isinstance(model, nn.DataParallel):
                actual_model = model.module
            else:
                actual_model = model
            
            # 检查多个参数以确保准确性（包括所有参数，不仅仅是requires_grad的）
            param_dtypes = []
            for param in actual_model.parameters():
                param_dtypes.append(param.dtype)
                if len(param_dtypes) >= 5:  # 检查前5个参数
                    break
            
            # 如果参数列表为空，尝试检查buffer
            if not param_dtypes:
                for buffer in actual_model.buffers():
                    param_dtypes.append(buffer.dtype)
                    if len(param_dtypes) >= 3:
                        break
            
            if param_dtypes:
                # 如果所有参数都是float16，则认为模型是half
                model_is_half = all(dtype == torch.float16 for dtype in param_dtypes)
            elif use_fp16:
                # 如果无法检测参数类型，但use_fp16为True，假设模型是half
                model_is_half = True
        except Exception as e:
            # 如果检测失败，根据use_fp16标志推断
            if use_fp16:
                model_is_half = True
                print(f"警告: 无法直接检测模型类型，根据use_fp16={use_fp16}推断为half类型: {e}")
    
    # 如果模型是half类型，确保输入也是half
    if model_is_half:
        # 确保输入是float32后再转换为half（避免类型不匹配）
        if images.dtype != torch.float16:
            images = images.float().half()
        if masks.dtype != torch.float16:
            masks = masks.float().half()
    
    # 逐个样本计算loss（更准确）
    per_sample_losses = []
    per_sample_preds = [] if return_preds else None
    
    with torch.no_grad():
        # 如果模型是half，直接推理；否则使用autocast
        if model_is_half:
            # 模型已经是half，直接推理
            # 确保输入类型与模型匹配
            try:
                outputs = model(images)
            except RuntimeError as e:
                if "type" in str(e).lower() and "should be the same" in str(e).lower():
                    # 类型不匹配错误，强制转换输入
                    print(f"警告: 检测到类型不匹配，强制转换输入为half类型")
                    images = images.float().half()
                    outputs = model(images)
                else:
                    raise
        elif use_fp16 and device.type == 'cuda':
            # 使用autocast进行混合精度推理
            with torch.cuda.amp.autocast():
                outputs = model(images)
        else:
            # 普通FP32推理
            outputs = model(images)
        
        # 处理模型输出（DataParallel会返回tuple或dict）
        if isinstance(outputs, dict):
            scaled_preds = outputs
        elif isinstance(outputs, (list, tuple)):
            scaled_preds = outputs
        else:
            scaled_preds = [outputs]
        
        # 逐个样本计算loss
        for i in range(batch_size):
            # 提取单个样本
            if isinstance(scaled_preds, dict):
                single_preds = {}
                for k, v in scaled_preds.items():
                    if isinstance(v, torch.Tensor):
                        single_preds[k] = v[i:i+1]
                    else:
                        single_preds[k] = v
            elif isinstance(scaled_preds, (list, tuple)):
                single_preds = [pred[i:i+1] if isinstance(pred, torch.Tensor) else pred for pred in scaled_preds]
            else:
                single_preds = [scaled_preds[i:i+1]]
            
            single_mask = masks[i:i+1]
            
            # 保存预测结果（如果需要）
            if return_preds:
                # 提取预测mask（参考infer.py的逻辑，使用pred_iou选择最优mask）
                if isinstance(single_preds, dict):
                    pred_mask = single_preds.get('pred_masks', None)
                    if pred_mask is not None:
                        # 如果是多mask输出，根据pred_iou选择最优的mask
                        if pred_mask.dim() == 4 and pred_mask.shape[1] > 1:
                            # 检查是否有pred_iou
                            pred_iou = single_preds.get('pred_iou', None)
                            if pred_iou is not None and pred_iou.dim() >= 2:
                                # 使用IoU最高的mask（参考infer.py行194）
                                best_idx = pred_iou.argmax(dim=1)  # [B]
                                best_idx_val = best_idx[0].item() if best_idx.numel() > 0 else 0
                                pred_mask = pred_mask[0, best_idx_val]  # [B, num_masks, H, W] -> [H, W]
                            else:
                                # 如果没有pred_iou，使用第一个mask（向后兼容）
                                pred_mask = pred_mask[0, 0]  # [B, num_masks, H, W] -> [H, W]
                        else:
                            pred_mask = pred_mask[0, 0] if pred_mask.dim() == 4 else pred_mask[0]
                    else:
                        # 尝试从其他键获取
                        for key in ['mask', 'output', 'pred']:
                            if key in single_preds:
                                pred_mask = single_preds[key]
                                if isinstance(pred_mask, torch.Tensor):
                                    pred_mask = pred_mask[0, 0] if pred_mask.dim() == 4 else pred_mask[0]
                                break
                        else:
                            pred_mask = None
                elif isinstance(single_preds, (list, tuple)) and len(single_preds) > 0:
                    # 取最后一个预测（通常是最终输出）
                    pred_mask = single_preds[-1]
                    if isinstance(pred_mask, torch.Tensor):
                        pred_mask = pred_mask[0, 0] if pred_mask.dim() == 4 else pred_mask[0]
                else:
                    pred_mask = None
                
                # 转换为numpy并保存
                if pred_mask is not None:
                    # 转换为FP32并sigmoid（如果是logits）
                    pred_mask = pred_mask.float()
                    if pred_mask.min() < 0 or pred_mask.max() > 1:
                        pred_mask = torch.sigmoid(pred_mask)
                    # 转换为numpy并缩放到0-255
                    pred_mask_np = (pred_mask.cpu().numpy() * 255).astype(np.uint8)
                    per_sample_preds.append(pred_mask_np)
                else:
                    per_sample_preds.append(None)
            
            # 计算单个样本的loss（loss计算使用FP32以保证精度）
            # 无论模型是FP16还是FP32，loss计算都使用FP32以保证精度
            if model_is_half or (use_fp16 and device.type == 'cuda'):
                # 将mask转换为FP32进行loss计算（保证精度）
                single_mask_fp32 = single_mask.float()
                # 将pred转换为FP32
                if isinstance(single_preds, dict):
                    single_preds_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in single_preds.items()}
                elif isinstance(single_preds, (list, tuple)):
                    single_preds_fp32 = [p.float() if isinstance(p, torch.Tensor) else p for p in single_preds]
                else:
                    single_preds_fp32 = [single_preds[0].float()] if isinstance(single_preds[0], torch.Tensor) else single_preds
                
                total_loss, loss_dict = loss_fn(
                    single_preds_fp32,
                    torch.clamp(single_mask_fp32, 0, 1),
                    pix_loss_lambda=1.0,
                    epoch=epoch
                )
            else:
                total_loss, loss_dict = loss_fn(
                    single_preds,
                    torch.clamp(single_mask, 0, 1),
                    pix_loss_lambda=1.0,
                    epoch=epoch
                )
            
            # 确保是标量
            if isinstance(total_loss, torch.Tensor):
                if total_loss.dim() > 0:
                    total_loss = total_loss.mean()
                per_sample_losses.append(total_loss.item())
            else:
                per_sample_losses.append(float(total_loss))
    
    if return_preds:
        return np.array(per_sample_losses), loss_dict, per_sample_preds
    else:
        return np.array(per_sample_losses), loss_dict


def apply_mask_to_greenscreen(original_img, mask_img, green_color=(127, 255, 0)):
    """
    将mask应用到原图，生成绿幕背景的主体图片
    mask的白色区域保留原图，黑色区域替换为绿色背景
    
    Args:
        original_img: 原图PIL Image
        mask_img: 灰度mask PIL Image
        green_color: RGB颜色值，默认为(127, 255, 0)
    
    Returns:
        PIL Image: 绿幕背景的主体图片
    """
    try:
        # 确保原图是RGB
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # 确保mask是灰度
        if mask_img.mode != 'L':
            mask_img = mask_img.convert('L')
        
        # 调整mask大小以匹配原图
        if mask_img.size != original_img.size:
            mask_img = mask_img.resize(original_img.size, Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        original_array = np.array(original_img)
        mask_array = np.array(mask_img)
        
        # 归一化mask (0-1)
        mask_normalized = mask_array.astype(np.float32) / 255.0
        
        # 创建结果数组
        result_array = np.zeros_like(original_array, dtype=np.float32)
        
        # 混合：mask值 * 原图 + (1 - mask值) * 绿色背景
        green_color_array = np.array(green_color, dtype=np.float32)
        
        for c in range(3):
            result_array[:, :, c] = (
                mask_normalized * original_array[:, :, c] + 
                (1 - mask_normalized) * green_color_array[c]
            )
        
        result_array = result_array.astype(np.uint8)
        result = Image.fromarray(result_array)
        
        return result
        
    except Exception as e:
        print(f"应用mask时出错: {e}")
        return None


def create_combined_result(original_img, greenscreen_img):
    """
    创建原图+绿幕图的拼接图（左右拼接）
    
    Args:
        original_img: 原图PIL Image
        greenscreen_img: 绿幕图PIL Image
    
    Returns:
        PIL Image: 拼接后的图片
    """
    try:
        # 确保两张图片高度一致
        h1, w1 = original_img.size[1], original_img.size[0]
        h2, w2 = greenscreen_img.size[1], greenscreen_img.size[0]
        
        # 统一高度
        target_height = max(h1, h2)
        if h1 != target_height:
            original_img = original_img.resize((int(w1 * target_height / h1), target_height), Image.Resampling.LANCZOS)
        if h2 != target_height:
            greenscreen_img = greenscreen_img.resize((int(w2 * target_height / h2), target_height), Image.Resampling.LANCZOS)
        
        # 左右拼接
        combined = Image.new('RGB', (original_img.size[0] + greenscreen_img.size[0], target_height))
        combined.paste(original_img, (0, 0))
        combined.paste(greenscreen_img, (original_img.size[0], 0))
        
        return combined
    except Exception as e:
        print(f"创建拼接图时出错: {e}")
        return None


def create_full_combined_result(original_img, gt_mask, pred_mask_np, greenscreen_gt, greenscreen_pred):
    """
    创建拼接图：原图 | GT绿幕图 | 预测绿幕图（已删除灰度图）
    
    Args:
        original_img: 原图PIL Image
        gt_mask: GT mask PIL Image（不再使用，保留参数以兼容）
        pred_mask_np: 预测mask numpy数组 (H, W) 0-255（不再使用，保留参数以兼容）
        greenscreen_gt: GT绿幕图PIL Image
        greenscreen_pred: 预测绿幕图PIL Image
    
    Returns:
        PIL Image: 拼接后的图片
    """
    try:
        # 确保所有图片高度一致
        target_height = original_img.size[1]
        images_to_combine = []
        
        # 1. 原图
        if original_img.size[1] != target_height:
            original_img = original_img.resize((int(original_img.size[0] * target_height / original_img.size[1]), target_height), Image.Resampling.LANCZOS)
        images_to_combine.append(('原图', original_img))
        
        # 2. GT绿幕图
        if greenscreen_gt is not None:
            if greenscreen_gt.size[1] != target_height:
                greenscreen_gt = greenscreen_gt.resize((int(greenscreen_gt.size[0] * target_height / greenscreen_gt.size[1]), target_height), Image.Resampling.LANCZOS)
            images_to_combine.append(('GT绿幕', greenscreen_gt))
        
        # 3. 预测绿幕图
        if greenscreen_pred is not None:
            if greenscreen_pred.size[1] != target_height:
                greenscreen_pred = greenscreen_pred.resize((int(greenscreen_pred.size[0] * target_height / greenscreen_pred.size[1]), target_height), Image.Resampling.LANCZOS)
            images_to_combine.append(('预测绿幕', greenscreen_pred))
        
        # 计算总宽度
        total_width = sum(img.size[0] for _, img in images_to_combine)
        
        # 创建拼接图
        combined = Image.new('RGB', (total_width, target_height))
        x_offset = 0
        for name, img in images_to_combine:
            combined.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        return combined
    except Exception as e:
        print(f"创建完整拼接图时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_file_pair_exists(pair):
    """检查文件对是否存在（用于多进程）"""
    img_path, mask_path = pair
    if os.path.exists(img_path) and os.path.exists(mask_path):
        return (img_path, mask_path), None
    else:
        return None, (img_path, mask_path)


def read_txt_files(txt_paths, num_workers=None):
    """
    读取多个txt文件，返回所有图片对
    使用多进程并行检查文件是否存在，提高速度
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 默认使用最多8个进程
    
    all_pairs = []
    all_candidates = []  # 所有待检查的文件对
    
    # 第一步：读取所有txt文件，收集所有文件对
    for txt_path in txt_paths:
        txt_path = txt_path.strip()
        if not os.path.exists(txt_path):
            print(f"警告: txt文件不存在，跳过: {txt_path}")
            continue
        print(f"读取txt文件: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # 空格分隔
            if len(parts) >= 2:
                img_path, mask_path = parts[0].strip(), parts[1].strip()
                all_candidates.append((img_path, mask_path))
    
    if not all_candidates:
        return all_pairs
    
    print(f"共找到 {len(all_candidates)} 对候选文件，使用 {num_workers} 个进程并行检查文件是否存在...")
    
    # 第二步：使用多进程并行检查文件是否存在
    valid_pairs = []
    invalid_pairs = []
    
    with Pool(processes=num_workers) as pool:
        # 使用imap_unordered提高效率，配合tqdm显示进度
        results = pool.imap_unordered(check_file_pair_exists, all_candidates)
        for result in tqdm(results, total=len(all_candidates), desc="检查文件存在性", file=sys.stdout, ncols=100, mininterval=1.0):
            valid_pair, invalid_pair = result
            if valid_pair:
                valid_pairs.append(valid_pair)
            else:
                invalid_pairs.append(invalid_pair)
    
    all_pairs = valid_pairs
    
    if invalid_pairs:
        print(f"警告: 发现 {len(invalid_pairs)} 对文件不存在（前10个）:")
        for img_path, mask_path in invalid_pairs[:10]:
            print(f"  图片: {img_path}, Mask: {mask_path}")
        if len(invalid_pairs) > 10:
            print(f"  ... 还有 {len(invalid_pairs) - 10} 对文件不存在")
    
    print(f"有效文件对: {len(all_pairs)} / {len(all_candidates)}")
    return all_pairs


def process_filter_worker(gpu_id, checkpoint_path, config_dict, data_subset, batch_size, num_workers, use_fp16, 
                          loss_threshold, pro_imgs_dir, pro_results_dir, result_queue):
    """
    多进程worker函数：在指定GPU上处理数据子集并保存结果
    每个进程独立加载模型，真正并行处理，不排队
    """
    try:
        # 确保输出不被缓冲，能被tee捕获
        import sys
        sys.stdout = sys.__stdout__  # 确保使用标准输出
        sys.stderr = sys.__stderr__  # 确保使用标准错误输出
        sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
        sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
        
        print(f"[GPU {gpu_id}] Worker进程开始，数据量: {len(data_subset)}")
        sys.stdout.flush()  # 确保输出立即刷新
        
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[GPU {gpu_id}] 已设置设备: {device}")
        sys.stdout.flush()
        
        # 重新创建config对象
        from config import Config
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # 加载模型（每个进程独立加载）
        num_outputs = getattr(config, 'num_outputs', 1)
        model = SAM2UNeXT("sam2_hiera_large.pt", "model.safetensors", num_outputs=num_outputs)
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}', weights_only=True)
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # FP16支持
        if use_fp16:
            try:
                model = model.half()
            except Exception as e:
                print(f"GPU {gpu_id}: 模型转换为FP16失败: {e}，将使用FP32")
        
        # 初始化loss函数
        full_mask_lambda = getattr(config, 'full_mask_lambda', 0.01)
        decay_rate = getattr(config, 'decay_rate', 0.2)
        loss_fn = PixLoss(full_mask_lambda=full_mask_lambda, decay_rate=decay_rate)
        loss_fn = loss_fn.to(device)
        
        # 创建数据集和数据加载器
        # 注意：在多进程模式下，每个进程已经独立运行，可以使用少量workers加速数据加载
        image_size = config.size if config.size else (1024, 1024)
        dataset = ImageMaskDataset(data_subset, size=image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(2, num_workers),  # 使用少量workers加速数据加载，避免过多子进程
            pin_memory=True,
            prefetch_factor=2 if min(2, num_workers) > 0 else None  # 预取数据加速
        )
        
        # 处理数据
        all_losses = []
        filtered_pairs = []
        saved_imgs_count = 0
        saved_results_count = 0
        
        # 批量收集需要保存的图片，避免在GPU计算时进行I/O操作
        to_save_queue = []  # [(img_path, mask_path, loss_value, pred_mask_np), ...]
        
        import time
        last_save_time = time.time()
        save_interval = 5.0  # 每5秒批量保存一次，或者累积到一定数量
        
        # 性能统计
        total_inference_time = 0.0
        total_save_time = 0.0
        total_samples = 0
        
        print(f"[GPU {gpu_id}] 开始处理，共 {len(data_subset)} 个样本，batch_size={batch_size}")
        
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                images = batch_data['image_tensor']
                masks = batch_data['mask_tensor']
                img_paths = batch_data['img_path']
                mask_paths = batch_data['mask_path']
                
                # 批量计算loss并获取预测结果
                inference_start = time.time()
                batch_loss, _, batch_preds = compute_loss_batch(model, images, masks, loss_fn, device, use_fp16=use_fp16, return_preds=True)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                total_samples += len(img_paths)
                
                batch_size_actual = len(img_paths)
                for i in range(batch_size_actual):
                    img_path = img_paths[i] if isinstance(img_paths, list) else img_paths[i].item() if hasattr(img_paths[i], 'item') else str(img_paths[i])
                    mask_path = mask_paths[i] if isinstance(mask_paths, list) else mask_paths[i].item() if hasattr(mask_paths[i], 'item') else str(mask_paths[i])
                    loss_value = float(batch_loss[i])
                    pred_mask_np = batch_preds[i] if batch_preds else None
                    
                    all_losses.append((img_path, mask_path, loss_value))
                    
                    # 如果loss大于阈值，加入保存队列（批量保存，避免阻塞GPU）
                    if loss_value > loss_threshold:
                        filtered_pairs.append((img_path, mask_path, loss_value))
                        to_save_queue.append((img_path, mask_path, loss_value, pred_mask_np))
                
                # 批量保存：每处理完一个batch，或者累积到一定数量/时间后批量保存
                current_time = time.time()
                should_save = (
                    len(to_save_queue) >= 10 or  # 累积10个以上
                    (current_time - last_save_time) >= save_interval or  # 超过5秒
                    batch_idx == len(dataloader) - 1  # 最后一个batch
                )
                
                if should_save and len(to_save_queue) > 0:
                    # 批量保存，避免阻塞GPU计算
                    save_start = time.time()
                    save_count = len(to_save_queue)
                    for save_item in to_save_queue:
                        img_path, mask_path, loss_value, pred_mask_np = save_item
                        try:
                            # 获取文件名（不含扩展名）
                            img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                            img_ext = os.path.splitext(os.path.basename(img_path))[1]
                            
                            # 1. 保存原图到pro_imgs文件夹
                            output_img_path = os.path.join(pro_imgs_dir, os.path.basename(img_path))
                            shutil.copy2(img_path, output_img_path)
                            saved_imgs_count += 1
                            
                            # 2. 生成完整的拼接结果并保存到pro_results文件夹
                            try:
                                # 加载原始尺寸的图片
                                orig_img_full = Image.open(img_path).convert('RGB')
                                orig_mask_full = Image.open(mask_path).convert('L')
                                
                                # 生成GT绿幕背景图
                                greenscreen_gt = apply_mask_to_greenscreen(orig_img_full, orig_mask_full, green_color=(127, 255, 0))
                                
                                # 生成预测绿幕背景图（如果有预测结果）
                                greenscreen_pred = None
                                if pred_mask_np is not None:
                                    try:
                                        # 将预测mask转换为PIL Image并调整大小
                                        pred_mask_img = Image.fromarray(pred_mask_np, mode='L')
                                        if pred_mask_img.size != orig_img_full.size:
                                            pred_mask_img = pred_mask_img.resize(orig_img_full.size, Image.Resampling.LANCZOS)
                                        greenscreen_pred = apply_mask_to_greenscreen(orig_img_full, pred_mask_img, green_color=(127, 255, 0))
                                    except Exception as e:
                                        pass  # 静默失败，避免输出过多
                                
                                # 创建完整拼接图
                                combined_img = create_full_combined_result(
                                    orig_img_full, 
                                    orig_mask_full, 
                                    pred_mask_np, 
                                    greenscreen_gt, 
                                    greenscreen_pred
                                )
                                
                                if combined_img is not None:
                                    # 保存拼接结果
                                    output_result_path = os.path.join(pro_results_dir, f"{img_name_base}_result{img_ext}")
                                    combined_img.save(output_result_path)
                                    saved_results_count += 1
                                else:
                                    # 如果完整拼接失败，尝试简单拼接（原图+GT绿幕）
                                    if greenscreen_gt is not None:
                                        simple_combined = create_combined_result(orig_img_full, greenscreen_gt)
                                        if simple_combined is not None:
                                            output_result_path = os.path.join(pro_results_dir, f"{img_name_base}_result{img_ext}")
                                            simple_combined.save(output_result_path)
                                            saved_results_count += 1
                            except Exception as e:
                                pass  # 静默失败，避免输出过多
                                
                        except Exception as e:
                            pass  # 静默失败，避免输出过多
                    
                    to_save_queue.clear()
                    last_save_time = current_time
                    save_time = time.time() - save_start
                    total_save_time += save_time
                    if save_count > 0:  # 如果刚才保存了
                        print(f"[GPU {gpu_id}] 批量保存 {save_count} 个结果，耗时 {save_time:.2f}s")
                            
            except Exception as e:
                print(f"GPU {gpu_id}: 处理批次失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 将结果放入队列
        result_queue.put((gpu_id, all_losses, filtered_pairs, saved_imgs_count, saved_results_count))
        
        # 输出性能统计
        avg_inference_time = total_inference_time / max(total_samples, 1)
        print(f"[GPU {gpu_id}] 处理完成:")
        print(f"  总样本数: {len(all_losses)}")
        print(f"  过滤结果: {len(filtered_pairs)}")
        print(f"  总推理时间: {total_inference_time:.2f}s (平均 {avg_inference_time*1000:.2f}ms/样本)")
        print(f"  总保存时间: {total_save_time:.2f}s")
        print(f"  推理占比: {total_inference_time/(total_inference_time+total_save_time)*100:.1f}%")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Worker进程出错: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((gpu_id, [], [], 0, 0))


def process_batch_worker(gpu_id, checkpoint_path, config_dict, data_subset, batch_size, num_workers, use_fp16, result_queue):
    """
    多进程worker函数：在指定GPU上处理数据子集
    每个进程独立加载模型，真正并行处理，不排队
    """
    try:
        # 确保输出不被缓冲，能被tee捕获
        import sys
        sys.stdout = sys.__stdout__  # 确保使用标准输出
        sys.stderr = sys.__stderr__  # 确保使用标准错误输出
        sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
        sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
        
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # 重新创建config对象（因为不能直接传递对象）
        from config import Config
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # 加载模型（每个进程独立加载）
        num_outputs = getattr(config, 'num_outputs', 1)
        model = SAM2UNeXT("sam2_hiera_large.pt", "model.safetensors", num_outputs=num_outputs)
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}', weights_only=True)
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # FP16支持
        if use_fp16:
            try:
                model = model.half()
            except Exception as e:
                print(f"GPU {gpu_id}: 模型转换为FP16失败: {e}，将使用FP32")
        
        # 初始化loss函数
        full_mask_lambda = getattr(config, 'full_mask_lambda', 0.01)
        decay_rate = getattr(config, 'decay_rate', 0.2)
        loss_fn = PixLoss(full_mask_lambda=full_mask_lambda, decay_rate=decay_rate)
        loss_fn = loss_fn.to(device)
        
        # 创建数据集和数据加载器
        # 注意：在多进程模式下，每个进程已经独立运行，所以num_workers设为0避免创建过多子进程
        image_size = config.size if config.size else (1024, 1024)
        dataset = ImageMaskDataset(data_subset, size=image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 多进程模式下设为0，避免创建过多子进程
            pin_memory=True
        )
        
        # 处理数据
        pair_losses = {}
        for batch_data in dataloader:
            try:
                images = batch_data['image_tensor']
                masks = batch_data['mask_tensor']
                img_paths = batch_data['img_path']
                mask_paths = batch_data['mask_path']
                
                batch_loss, _ = compute_loss_batch(model, images, masks, loss_fn, device, use_fp16=use_fp16, return_preds=False)
                
                batch_size_actual = len(img_paths)
                for i in range(batch_size_actual):
                    img_path = img_paths[i] if isinstance(img_paths, list) else img_paths[i].item() if hasattr(img_paths[i], 'item') else str(img_paths[i])
                    mask_path = mask_paths[i] if isinstance(mask_paths, list) else mask_paths[i].item() if hasattr(mask_paths[i], 'item') else str(mask_paths[i])
                    loss_value = float(batch_loss[i])
                    pair_losses[(img_path, mask_path)] = loss_value
            except Exception as e:
                print(f"GPU {gpu_id}: 处理批次失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 将结果放入队列
        result_queue.put((gpu_id, pair_losses))
        print(f"GPU {gpu_id}: 处理完成，共 {len(pair_losses)} 个结果")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Worker进程出错: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((gpu_id, {}))


def merge_duplicate_files_by_loss(image_mask_pairs, model, loss_fn, device, config, batch_size=8, num_workers=4, use_fp16=False, device_ids=None, checkpoint_path=None):
    """
    对于同名文件（基于图片文件名），计算每个pair的loss，只保留loss最小的
    返回融合后的图片对列表
    """
    print(f"\n开始处理同名文件，共 {len(image_mask_pairs)} 对图片...")
    
    # 按图片文件名分组
    img_name_to_pairs = {}
    for img_path, mask_path in image_mask_pairs:
        img_name = os.path.basename(img_path)
        if img_name not in img_name_to_pairs:
            img_name_to_pairs[img_name] = []
        img_name_to_pairs[img_name].append((img_path, mask_path))
    
    # 找出有重复的文件
    duplicate_groups = {name: pairs for name, pairs in img_name_to_pairs.items() if len(pairs) > 1}
    unique_pairs = {name: pairs[0] for name, pairs in img_name_to_pairs.items() if len(pairs) == 1}
    
    print(f"发现 {len(duplicate_groups)} 个同名文件组，{len(unique_pairs)} 个唯一文件")
    
    if len(duplicate_groups) == 0:
        print("没有同名文件，直接返回原始列表")
        return image_mask_pairs
    
    # 统计重复文件组的详细信息
    group_sizes = [len(pairs) for pairs in duplicate_groups.values()]
    total_duplicate_pairs = sum(group_sizes)
    avg_group_size = np.mean(group_sizes) if group_sizes else 0
    max_group_size = max(group_sizes) if group_sizes else 0
    min_group_size = min(group_sizes) if group_sizes else 0
    
    print(f"\n重复文件组统计:")
    print(f"  总文件对: {len(image_mask_pairs)}")
    print(f"  唯一文件对: {len(unique_pairs)} (不需要计算loss，直接保留)")
    print(f"  重复文件组: {len(duplicate_groups)} 个组")
    print(f"  重复文件对总数: {total_duplicate_pairs} (需要计算loss)")
    print(f"  每个组平均文件对数: {avg_group_size:.2f}")
    print(f"  最大组文件对数: {max_group_size}")
    print(f"  最小组文件对数: {min_group_size}")
    print(f"\n说明: 对于每个同名文件组，需要计算组内所有文件对的loss，然后选择loss最小的那个保留。")
    print(f"      因此需要计算 {total_duplicate_pairs} 个文件对的loss（而不是 {len(duplicate_groups)} 个）。")
    
    # 对于有重复的文件，计算loss并选择最小的
    image_size = config.size if config.size else (1024, 1024)
    merged_pairs = []
    
    # 先处理唯一文件
    merged_pairs.extend(unique_pairs.values())
    
    # 处理重复文件：计算每个pair的loss
    duplicate_pairs_to_process = []
    duplicate_group_info = []  # 记录每个组的信息，用于后续选择
    
    for img_name, pairs in duplicate_groups.items():
        duplicate_pairs_to_process.extend(pairs)
        # 记录每个pair属于哪个组
        for pair in pairs:
            duplicate_group_info.append((img_name, pair))
    
    print(f"\n需要计算loss的重复文件对: {len(duplicate_pairs_to_process)}")
    
    # 判断是否使用多进程并行（真正的并行，不排队）
    use_multi_process = device_ids is not None and len(device_ids) > 1
    
    if use_multi_process:
        # 多进程并行：每个进程使用不同的GPU，真正并行处理
        print(f"使用多进程并行处理，GPU列表: {device_ids}")
        
        # 将数据分割成多个子集
        num_gpus = len(device_ids)
        chunk_size = (len(duplicate_pairs_to_process) + num_gpus - 1) // num_gpus
        data_chunks = [duplicate_pairs_to_process[i:i+chunk_size] 
                      for i in range(0, len(duplicate_pairs_to_process), chunk_size)]
        
        # 确保chunk数量不超过GPU数量
        while len(data_chunks) < num_gpus:
            data_chunks.append([])
        data_chunks = data_chunks[:num_gpus]
        
        print(f"数据分割: {[len(chunk) for chunk in data_chunks]} (总计: {sum(len(chunk) for chunk in data_chunks)})")
        
        # 准备config字典（因为不能直接传递对象）
        config_dict = {
            'size': config.size,
            'num_outputs': getattr(config, 'num_outputs', 1),
            'full_mask_lambda': getattr(config, 'full_mask_lambda', 0.01),
            'decay_rate': getattr(config, 'decay_rate', 0.2),
        }
        
        # 检查checkpoint_path是否提供
        if checkpoint_path is None:
            # 如果无法获取，回退到单进程模式
            print("警告: 未提供checkpoint_path，回退到单进程模式")
            use_multi_process = False
    
    if not use_multi_process:
        # 单进程模式（原来的方式）
        dataset = ImageMaskDataset(duplicate_pairs_to_process, size=image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        pair_losses = {}  # {(img_path, mask_path): loss}
        
        for batch_data in tqdm(dataloader, desc="计算重复文件的loss", file=sys.stdout, ncols=100, mininterval=1.0):
            try:
                images = batch_data['image_tensor']
                masks = batch_data['mask_tensor']
                img_paths = batch_data['img_path']
                mask_paths = batch_data['mask_path']
                
                batch_loss, _ = compute_loss_batch(model, images, masks, loss_fn, device, use_fp16=use_fp16, return_preds=False)
                
                batch_size_actual = len(img_paths)
                for i in range(batch_size_actual):
                    img_path = img_paths[i] if isinstance(img_paths, list) else img_paths[i].item() if hasattr(img_paths[i], 'item') else str(img_paths[i])
                    mask_path = mask_paths[i] if isinstance(mask_paths, list) else mask_paths[i].item() if hasattr(mask_paths[i], 'item') else str(mask_paths[i])
                    loss_value = float(batch_loss[i])
                    pair_losses[(img_path, mask_path)] = loss_value
            except Exception as e:
                print(f"错误: 处理批次失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        # 多进程并行模式
        mp.set_start_method('spawn', force=True)  # 确保使用spawn方式
        
        # 使用Manager().Queue()避免资源泄漏
        manager = Manager()
        result_queue = manager.Queue()
        processes = []
        
        def cleanup_processes_merge():
            """清理所有进程"""
            for p in processes:
                if p.is_alive():
                    try:
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=2)
                    except:
                        pass
        
        def signal_handler_merge(signum, frame):
            """信号处理函数"""
            print(f"\n收到信号 {signum}，正在清理进程...")
            cleanup_processes_merge()
            try:
                result_queue.close()
                result_queue.join_thread()
                manager.shutdown()
            except:
                pass
            exit(1)
        
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler_merge)
        signal.signal(signal.SIGTERM, signal_handler_merge)
        atexit.register(cleanup_processes_merge)
        
        # 启动多个进程
        for idx, gpu_id in enumerate(device_ids):
            if idx < len(data_chunks) and len(data_chunks[idx]) > 0:
                p = Process(
                    target=process_batch_worker,
                    args=(gpu_id, checkpoint_path, config_dict, data_chunks[idx], batch_size, num_workers, use_fp16, result_queue)
                )
                p.daemon = False  # 非daemon进程，确保能正确清理
                p.start()
                processes.append(p)
                print(f"启动进程 {idx}，使用 GPU {gpu_id}，处理 {len(data_chunks[idx])} 个样本")
                sys.stdout.flush()
        
        # 收集结果
        pair_losses = {}
        try:
            # 添加进度条显示收集结果的进度
            # 使用file=sys.stdout确保输出能被tee捕获
            print(f"[Rank0] 开始收集 {len(processes)} 个GPU的结果...")
            sys.stdout.flush()
            with tqdm(total=len(processes), desc="[Rank0] 收集重复文件loss计算结果", unit="GPU", 
                     file=sys.stdout, ncols=100, mininterval=0.5, dynamic_ncols=False) as pbar:
                for _ in range(len(processes)):
                    gpu_id, chunk_losses = result_queue.get()
                    pair_losses.update(chunk_losses)
                    pbar.set_postfix({"GPU": gpu_id, "结果数": len(chunk_losses), "总计": len(pair_losses)})
                    pbar.update(1)
                    pbar.refresh()  # 强制刷新
                    sys.stdout.flush()
                    sys.stderr.flush()  # tqdm也可能使用stderr
                    print(f"✓ 收到 GPU {gpu_id} 的结果，共 {len(chunk_losses)} 个，累计 {len(pair_losses)} 个")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("收到中断信号，清理进程...")
            cleanup_processes_merge()
            raise
        finally:
            # 等待所有进程完成
            for p in processes:
                try:
                    p.join(timeout=60)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=10)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=5)
                    elif p.exitcode != 0:
                        print(f"警告: 进程退出码为 {p.exitcode}")
                except Exception as e:
                    print(f"等待进程时出错: {e}")
            
            # 清理资源
            cleanup_processes_merge()
            try:
                result_queue.close()
                result_queue.join_thread()
                manager.shutdown()
            except:
                pass
        
        print(f"多进程并行处理完成，共收集到 {len(pair_losses)} 个结果")
    
    # 对于每个同名文件组，选择loss最小的pair
    print(f"\n[Rank0] 开始选择每个文件组的最佳pair...")
    sys.stdout.flush()
    with tqdm(total=len(duplicate_groups), desc="[Rank0] 融合重复文件", unit="组", 
             file=sys.stdout, ncols=100, mininterval=1.0) as pbar:
        for img_name, pairs in duplicate_groups.items():
            best_pair = None
            best_loss = float('inf')
            for pair in pairs:
                if pair in pair_losses:
                    loss = pair_losses[pair]
                    if loss < best_loss:
                        best_loss = loss
                        best_pair = pair
                else:
                    # 如果计算失败，使用第一个pair（作为fallback）
                    if best_pair is None:
                        best_pair = pair
                        best_loss = float('inf')  # 标记为未计算
            
            if best_pair:
                merged_pairs.append(best_pair)
                if best_loss != float('inf'):
                    pbar.set_postfix({"文件": img_name[:30], "loss": f"{best_loss:.6f}"})
                else:
                    pbar.set_postfix({"文件": img_name[:30], "状态": "使用第一个pair"})
            pbar.update(1)
    
    print(f"融合完成: 原始 {len(image_mask_pairs)} 对 -> 融合后 {len(merged_pairs)} 对")
    return merged_pairs


def filter_high_loss_images(args, config):
    """过滤loss较大的图片"""
    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pro_imgs_dir = os.path.join(script_dir, 'pro_imgs')
    pro_results_dir = os.path.join(script_dir, 'pro_results')
    
    # 清空并创建输出文件夹
    if os.path.exists(pro_imgs_dir):
        shutil.rmtree(pro_imgs_dir)
    if os.path.exists(pro_results_dir):
        shutil.rmtree(pro_results_dir)
    os.makedirs(pro_imgs_dir, exist_ok=True)
    os.makedirs(pro_results_dir, exist_ok=True)
    print(f"已清空并创建输出文件夹: {pro_imgs_dir}, {pro_results_dir}")
    
    # 自动检测GPU或使用指定的设备
    device_str = args.device.lower().strip() if args.device else 'auto'
    use_multi_gpu = False
    device_ids = None
    
    if device_str == 'auto' or device_str == '':
        # 自动检测所有可用的GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                device_ids = list(range(num_gpus))
                device = torch.device('cuda:0')
                use_multi_gpu = True
                print(f"自动检测到 {num_gpus} 个GPU，使用所有GPU: {device_ids}")
            elif num_gpus == 1:
                device = torch.device('cuda:0')
                print(f"自动检测到 1 个GPU，使用: cuda:0")
            else:
                device = torch.device('cpu')
                print("未检测到GPU，使用CPU")
        else:
            device = torch.device('cpu')
            print("CUDA不可用，使用CPU")
    elif ',' in device_str:
        # 手动指定多GPU模式
        device_ids = [int(d.split(':')[-1]) for d in device_str.split(',')]
        device = torch.device(f'cuda:{device_ids[0]}')
        use_multi_gpu = True
        print(f"手动指定多GPU模式: {device_ids}")
    else:
        # 手动指定单GPU或CPU
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        print(f"手动指定设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device, config, use_multi_gpu=use_multi_gpu, device_ids=device_ids, use_fp16=args.use_fp16)
    
    # 自动调整batch_size以充分利用多GPU
    # DataParallel工作原理：
    # 1. DataLoader在主进程（单进程）中加载数据，batch_size=8
    # 2. 数据被送到主GPU（cuda:0），然后DataParallel自动将batch分割到所有GPU
    # 3. 如果batch_size=8，8个GPU，每个GPU处理1个样本（8/8=1）
    # 4. 如果batch_size=64，8个GPU，每个GPU处理8个样本（64/8=8）
    # 5. 所有GPU并行计算，结果在主GPU上聚合
    # 建议：batch_size应该是GPU数量的倍数，以充分利用所有GPU
    num_gpus_used = len(device_ids) if device_ids and use_multi_gpu else (1 if device.type == 'cuda' else 0)
    if num_gpus_used > 1:
        # 如果batch_size不是GPU数量的倍数，建议调整
        if args.batch_size % num_gpus_used != 0:
            suggested_batch_size = ((args.batch_size // num_gpus_used) + 1) * num_gpus_used
            print(f"提示: 当前batch_size={args.batch_size}，使用{num_gpus_used}个GPU")
            print(f"      建议将batch_size调整为{num_gpus_used}的倍数（如{suggested_batch_size}）以充分利用所有GPU")
        else:
            print(f"✓ batch_size={args.batch_size}是{num_gpus_used}的倍数，可以充分利用所有GPU")
            print(f"  每个GPU将处理 {args.batch_size // num_gpus_used} 个样本")
    
    # 初始化loss函数
    full_mask_lambda = getattr(config, 'full_mask_lambda', 0.01)
    decay_rate = getattr(config, 'decay_rate', 0.2)
    loss_fn = PixLoss(full_mask_lambda=full_mask_lambda, decay_rate=decay_rate)
    loss_fn = loss_fn.to(device)
    
    # 解析多个txt文件路径（已经是列表格式）
    txt_paths = args.input_txt
    print(f"输入txt文件数量: {len(txt_paths)}")
    print(f"输入txt文件列表: {txt_paths}")
    
    # 读取所有txt文件（使用多进程检查文件存在性）
    all_image_mask_pairs = read_txt_files(txt_paths, num_workers=args.num_workers)
    print(f"共读取到 {len(all_image_mask_pairs)} 对有效图片")
    
    # 第一步：处理同名文件，融合成唯一的pair列表
    merged_pairs = merge_duplicate_files_by_loss(
        all_image_mask_pairs, 
        model, 
        loss_fn, 
        device, 
        config, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fp16=args.use_fp16,
        device_ids=device_ids,
        checkpoint_path=args.checkpoint
    )
    
    # 验证融合后的pair中文件名是否唯一
    img_names_in_merged = [os.path.basename(img_path) for img_path, _ in merged_pairs]
    if len(img_names_in_merged) != len(set(img_names_in_merged)):
        print("警告: 融合后的pair中仍有重复的文件名！")
        # 统计重复
        from collections import Counter
        name_counts = Counter(img_names_in_merged)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        print(f"重复的文件名: {duplicates}")
    else:
        print(f"✓ 验证通过: 融合后的txt中每个文件名都是唯一的（共 {len(merged_pairs)} 个文件）")
    
    # 保存融合后的txt文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.merged_txt:
        merged_txt_path = args.merged_txt
    else:
        merged_txt_path = os.path.join(script_dir, 'merged_input.txt')
    
    print(f"\n保存融合后的txt文件: {merged_txt_path}")
    with open(merged_txt_path, 'w', encoding='utf-8') as f:
        for img_path, mask_path in merged_pairs:
            f.write(f"{img_path}   {mask_path}\n")
    print(f"融合后的txt文件已保存，共 {len(merged_pairs)} 对图片")
    
    # 第二步：使用融合后的pair列表进行loss计算和过滤
    image_mask_pairs = merged_pairs
    
    # 判断是否使用多进程并行（真正的并行，不排队）
    use_multi_process_filter = device_ids is not None and len(device_ids) > 1
    
    if use_multi_process_filter:
        # 多进程并行：每个进程使用不同的GPU，真正并行处理
        print(f"使用多进程并行处理主循环，GPU列表: {device_ids}")
        
        # 将数据分割成多个子集
        num_gpus = len(device_ids)
        chunk_size = (len(image_mask_pairs) + num_gpus - 1) // num_gpus
        data_chunks = [image_mask_pairs[i:i+chunk_size] 
                      for i in range(0, len(image_mask_pairs), chunk_size)]
        
        # 确保chunk数量不超过GPU数量
        while len(data_chunks) < num_gpus:
            data_chunks.append([])
        data_chunks = data_chunks[:num_gpus]
        
        print(f"数据分割: {[len(chunk) for chunk in data_chunks]} (总计: {sum(len(chunk) for chunk in data_chunks)})")
        
        # 准备config字典
        config_dict = {
            'size': config.size,
            'num_outputs': getattr(config, 'num_outputs', 1),
            'full_mask_lambda': getattr(config, 'full_mask_lambda', 0.01),
            'decay_rate': getattr(config, 'decay_rate', 0.2),
        }
        
        # 多进程并行模式
        mp.set_start_method('spawn', force=True)  # 确保使用spawn方式
        
        # 使用Manager().Queue()避免资源泄漏
        manager = Manager()
        result_queue = manager.Queue()
        processes = []
        
        def cleanup_processes_filter():
            """清理所有进程"""
            for idx, gpu_id, p in processes:
                if p.is_alive():
                    print(f"清理进程 {idx} (GPU {gpu_id})...")
                    try:
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=2)
                    except Exception as e:
                        print(f"清理进程 {idx} (GPU {gpu_id}) 时出错: {e}")
        
        def signal_handler_filter(signum, frame):
            """信号处理函数"""
            print(f"\n收到信号 {signum}，正在清理进程...")
            cleanup_processes_filter()
            try:
                result_queue.close()
                result_queue.join_thread()
                manager.shutdown()
            except:
                pass
            exit(1)
        
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler_filter)
        signal.signal(signal.SIGTERM, signal_handler_filter)
        atexit.register(cleanup_processes_filter)
        
        # 启动多个进程
        for idx, gpu_id in enumerate(device_ids):
            if idx < len(data_chunks) and len(data_chunks[idx]) > 0:
                print(f"准备启动进程 {idx}，使用 GPU {gpu_id}，处理 {len(data_chunks[idx])} 个样本")
                try:
                    p = Process(
                        target=process_filter_worker,
                        args=(gpu_id, args.checkpoint, config_dict, data_chunks[idx], args.batch_size, 
                              args.num_workers, args.use_fp16, args.loss_threshold, 
                              pro_imgs_dir, pro_results_dir, result_queue)
                    )
                    p.daemon = False  # 非daemon进程，确保能正确清理
                    p.start()
                    processes.append((idx, gpu_id, p))
                    print(f"✓ 成功启动进程 {idx}，使用 GPU {gpu_id}，处理 {len(data_chunks[idx])} 个样本")
                except Exception as e:
                    print(f"✗ 启动进程 {idx} (GPU {gpu_id}) 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"跳过进程 {idx} (GPU {gpu_id}): chunk不存在或为空")
        
        print(f"共启动了 {len(processes)} 个进程")
        
        # 收集结果
        all_losses = []
        filtered_pairs = []
        saved_imgs_count = 0
        saved_results_count = 0
        
        print(f"开始批量计算loss并实时保存，阈值: {args.loss_threshold}, batch_size: {args.batch_size}")
        print(f"等待 {len(processes)} 个进程完成...")
        
        # 使用超时机制收集结果
        import time
        start_time = time.time()
        timeout = 3600 * 24  # 24小时超时
        
        completed_gpus = set()
        total_samples = len(image_mask_pairs)
        
        # 添加详细的进度条显示主循环进度
        with tqdm(total=len(processes), desc="[Rank0] 主循环-等待所有GPU完成", unit="GPU", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                  file=sys.stdout, ncols=120, mininterval=1.0) as pbar:
            for _ in range(len(processes)):
                try:
                    # 设置超时，避免无限等待
                    remaining_timeout = timeout - (time.time() - start_time)
                    if remaining_timeout <= 0:
                        print(f"警告: 等待超时，剩余进程可能卡住")
                        break
                    
                    # 尝试从队列获取结果（带超时）
                    try:
                        result = result_queue.get(timeout=min(remaining_timeout, 300))  # 最多等待5分钟
                        gpu_id, chunk_losses, chunk_filtered, chunk_imgs, chunk_results = result
                        all_losses.extend(chunk_losses)
                        filtered_pairs.extend(chunk_filtered)
                        saved_imgs_count += chunk_imgs
                        saved_results_count += chunk_results
                        completed_gpus.add(gpu_id)
                        
                        # 更新进度条
                        processed_samples = len(all_losses)
                        pbar.set_postfix({
                            "GPU": gpu_id,
                            "已处理": f"{processed_samples}/{total_samples}",
                            "过滤": len(filtered_pairs),
                            "已保存图片": saved_imgs_count,
                            "已保存结果": saved_results_count
                        })
                        pbar.update(1)
                        pbar.refresh()  # 强制刷新
                        sys.stdout.flush()
                        print(f"✓ 收到 GPU {gpu_id} 的结果: {len(chunk_losses)} 个样本, {len(chunk_filtered)} 个过滤结果 | 累计: {processed_samples}/{total_samples} 样本, {len(filtered_pairs)} 个过滤")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"警告: 从队列获取结果失败: {e}")
                        break
                except Exception as e:
                    print(f"错误: 收集结果时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        # 检查哪些进程没有完成
        for idx, gpu_id, p in processes:
            if gpu_id not in completed_gpus:
                print(f"警告: GPU {gpu_id} (进程 {idx}) 未返回结果")
                if p.is_alive():
                    print(f"  GPU {gpu_id} 进程仍在运行，等待中...")
                else:
                    print(f"  GPU {gpu_id} 进程已结束，退出码: {p.exitcode}")
        
        # 等待所有进程完成
        print("等待所有进程结束...")
        try:
            for idx, gpu_id, p in processes:
                try:
                    p.join(timeout=60)  # 最多等待60秒
                    if p.is_alive():
                        print(f"警告: GPU {gpu_id} (进程 {idx}) 仍在运行，强制终止")
                        p.terminate()
                        p.join(timeout=10)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=5)
                    elif p.exitcode != 0:
                        print(f"警告: GPU {gpu_id} (进程 {idx}) 退出码为 {p.exitcode}")
                except Exception as e:
                    print(f"错误: 等待进程 {idx} (GPU {gpu_id}) 时出错: {e}")
        finally:
            # 确保清理所有进程
            cleanup_processes_filter()
            # 关闭队列和manager
            try:
                result_queue.close()
                result_queue.join_thread()
            except:
                pass
            try:
                manager.shutdown()
            except:
                pass
        
        print(f"多进程并行处理完成，共处理 {len(all_losses)} 个样本，{len(filtered_pairs)} 个过滤结果")
    else:
        # 单进程模式（原来的方式）
        # 使用config.size，如果为None则使用默认值
        image_size = config.size if config.size else (1024, 1024)
        
        # 创建数据集和数据加载器
        dataset = ImageMaskDataset(image_mask_pairs, size=image_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 批量计算loss并实时保存
        filtered_pairs = []
        all_losses = []
        saved_imgs_count = 0
        saved_results_count = 0
        
        print(f"开始批量计算loss并实时保存，阈值: {args.loss_threshold}, batch_size: {args.batch_size}")
        
        for batch_data in tqdm(dataloader, desc="计算loss并保存", file=sys.stdout, ncols=100, mininterval=1.0):
            try:
                images = batch_data['image_tensor']  # [B, 3, H, W]
                masks = batch_data['mask_tensor']    # [B, 1, H, W]
                img_paths = batch_data['img_path']
                mask_paths = batch_data['mask_path']
                
                # 批量计算loss并获取预测结果
                batch_loss, _, batch_preds = compute_loss_batch(model, images, masks, loss_fn, device, use_fp16=args.use_fp16, return_preds=True)
                
                # 处理每个样本的结果并实时保存
                batch_size = len(img_paths)
                for i in range(batch_size):
                    img_path = img_paths[i] if isinstance(img_paths, list) else img_paths[i].item() if hasattr(img_paths[i], 'item') else str(img_paths[i])
                    mask_path = mask_paths[i] if isinstance(mask_paths, list) else mask_paths[i].item() if hasattr(mask_paths[i], 'item') else str(mask_paths[i])
                    loss_value = float(batch_loss[i])
                    pred_mask_np = batch_preds[i] if batch_preds else None
                    
                    all_losses.append((img_path, mask_path, loss_value))
                    
                    # 如果loss大于阈值，立即保存
                    if loss_value > args.loss_threshold:
                        filtered_pairs.append((img_path, mask_path, loss_value))
                        
                        # 实时保存
                        try:
                            # 获取文件名（不含扩展名）
                            img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                            img_ext = os.path.splitext(os.path.basename(img_path))[1]
                            
                            # 1. 保存原图到pro_imgs文件夹
                            output_img_path = os.path.join(pro_imgs_dir, os.path.basename(img_path))
                            shutil.copy2(img_path, output_img_path)
                            saved_imgs_count += 1
                            
                            # 2. 生成完整的拼接结果并保存到pro_results文件夹
                            try:
                                # 加载原始尺寸的图片
                                orig_img_full = Image.open(img_path).convert('RGB')
                                orig_mask_full = Image.open(mask_path).convert('L')
                                
                                # 生成GT绿幕背景图
                                greenscreen_gt = apply_mask_to_greenscreen(orig_img_full, orig_mask_full, green_color=(127, 255, 0))
                                
                                # 生成预测绿幕背景图（如果有预测结果）
                                greenscreen_pred = None
                                if pred_mask_np is not None:
                                    try:
                                        # 将预测mask转换为PIL Image并调整大小
                                        pred_mask_img = Image.fromarray(pred_mask_np, mode='L')
                                        if pred_mask_img.size != orig_img_full.size:
                                            pred_mask_img = pred_mask_img.resize(orig_img_full.size, Image.Resampling.LANCZOS)
                                        greenscreen_pred = apply_mask_to_greenscreen(orig_img_full, pred_mask_img, green_color=(127, 255, 0))
                                    except Exception as e:
                                        print(f"\n警告: 生成预测绿幕图失败 - {img_path}: {e}")
                                
                                # 创建完整拼接图
                                combined_img = create_full_combined_result(
                                    orig_img_full, 
                                    orig_mask_full, 
                                    pred_mask_np, 
                                    greenscreen_gt, 
                                    greenscreen_pred
                                )
                                
                                if combined_img is not None:
                                    # 保存拼接结果
                                    output_result_path = os.path.join(pro_results_dir, f"{img_name_base}_result{img_ext}")
                                    combined_img.save(output_result_path)
                                    saved_results_count += 1
                                else:
                                    # 如果完整拼接失败，尝试简单拼接（原图+GT绿幕）
                                    if greenscreen_gt is not None:
                                        simple_combined = create_combined_result(orig_img_full, greenscreen_gt)
                                        if simple_combined is not None:
                                            output_result_path = os.path.join(pro_results_dir, f"{img_name_base}_result{img_ext}")
                                            simple_combined.save(output_result_path)
                                            saved_results_count += 1
                            except Exception as e:
                                print(f"\n警告: 生成拼接图失败 - {img_path}: {e}")
                                import traceback
                                traceback.print_exc()
                                
                            # 实时显示保存进度（每10张或第1张时显示）
                            if (saved_imgs_count + saved_results_count) % 10 == 0 or saved_imgs_count == 1:
                                print(f"\n[实时保存] loss={loss_value:.6f} > {args.loss_threshold}, 已保存: {saved_imgs_count}张原图, {saved_results_count}张拼接图")
                                
                        except Exception as e:
                            print(f"\n错误: 实时保存失败 - {img_path}: {e}")
                            import traceback
                            traceback.print_exc()
                        
            except Exception as e:
                print(f"\n错误: 处理批次失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n统计结果:")
        print(f"总图片数: {len(image_mask_pairs)}")
        print(f"过滤的图片数 (loss > {args.loss_threshold}): {len(filtered_pairs)}")
        print(f"已保存原图到pro_imgs: {saved_imgs_count} 张")
        print(f"已保存拼接图到pro_results: {saved_results_count} 张")
    
    # 打印loss分布信息，帮助用户了解情况
    if all_losses:
        losses = [loss for _, _, loss in all_losses]
        print(f"\nLoss分布统计:")
        print(f"  最小loss: {min(losses):.6f}")
        print(f"  最大loss: {max(losses):.6f}")
        print(f"  平均loss: {np.mean(losses):.6f}")
        print(f"  中位数loss: {np.median(losses):.6f}")
        print(f"  阈值: {args.loss_threshold}")
        print(f"  loss > 阈值的数量: {len(filtered_pairs)}")
        print(f"  loss <= 阈值的数量: {len(all_losses) - len(filtered_pairs)}")
        
        if len(filtered_pairs) == 0:
            print(f"\n⚠️  警告: 没有图片的loss大于阈值 {args.loss_threshold}")
            print(f"   建议: 可以尝试降低阈值，例如 --loss_threshold {max(losses) * 0.8:.6f}")
        
        # 保存过滤后的路径到txt文件
        if args.save_filtered_txt:
            with open(args.save_filtered_txt, 'w', encoding='utf-8') as f:
                for item in filtered_pairs:
                    img_path, mask_path, loss_value = item[0], item[1], item[2]
                    f.write(f"{img_path}   {mask_path}\n")
            print(f"过滤后的路径已保存到: {args.save_filtered_txt}")
        
        # 保存loss统计信息
        loss_stats_path = os.path.join(pro_imgs_dir, 'loss_statistics.txt')
        with open(loss_stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Loss阈值: {args.loss_threshold}\n")
            f.write(f"总图片数: {len(image_mask_pairs)}\n")
            f.write(f"过滤的图片数: {len(filtered_pairs)}\n\n")
            f.write("过滤的图片列表 (按loss降序):\n")
            sorted_pairs = sorted(filtered_pairs, key=lambda x: x[2], reverse=True)
            for item in sorted_pairs:
                img_path, mask_path, loss_value = item[0], item[1], item[2]
                f.write(f"Loss: {loss_value:.6f} - {img_path}   {mask_path}\n")
        print(f"Loss统计信息已保存到: {loss_stats_path}")
    else:
        print("没有图片需要过滤")


if __name__ == '__main__':
    # 确保输出不被缓冲，能被tee捕获（2>&1 | tee logfile）
    # 设置环境变量强制Python使用无缓冲输出
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 确保stdout和stderr使用行缓冲
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    
    # 设置tqdm默认输出到stdout（而不是stderr）
    # 导入tqdm模块以访问write函数（使用别名避免冲突）
    import tqdm as tqdm_module
    # 保存原始的write方法
    _tqdm_write = tqdm_module.tqdm.write
    def tqdm_write(s, file=None, end="\n"):
        """重写tqdm.write，确保输出到stdout"""
        if file is None:
            file = sys.stdout
        file.write(s + end)
        file.flush()
    tqdm_module.tqdm.write = tqdm_write
    
    args = parse_args()
    config = Config()
    
    # 确保输出目录是相对于脚本目录
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    filter_high_loss_images(args, config)

