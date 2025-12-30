# sample  默认就行 cma-es比较适合处理连续的参数空间
# 剪枝策略 最好使用Hyperband 

import argparse
from copy import deepcopy
import logging
import os
import pprint
from optuna.trial import TrialState
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time

from metrix import s2fb_iou

import numpy as np

cnt = 0

from pathlib import Path

from dataset.semi2 import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
# from util.dist_helper import setup_distributed
import torch.multiprocessing as mp
import torch.distributed as dist

import optuna
from functools import partial
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets
from torchvision import transforms
import torch.utils.data
import torch.nn.functional as F
import subprocess

RANDOM_SEED = 42
N_TRIALS = 200
EPOCHS = 15
storage_name = "sqlite:///optuna.db"
ignore_index = 255
batch_size = 6
data_root = '/home/notebook/code/personal/80410839/Dataset3'
crop_size = 518
labeled_id_path = '/home/notebook/code/personal/80410839/Dataset3/labeled0324.txt' # 这里实际训练的样本数量受超参数控制
unlabeled_id_path = '/home/notebook/code/personal/80410839/Dataset3/unlabeled0324.txt'
nclass = 2

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed) # Python随机数生成器
    np.random.seed(seed) # Numpy随机数生成器
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # 保证实验结果可复现

    """
    但有时候这样还不够，比如PyTorch中的一些计算，即使设置好了随机种子，在进行浮点数计算的时候，浮点数的运算顺序还是不确定的，而且不同的运算顺序可能造成精度上的差异，比如：3 + 2 + 1 = 5.000001， 而 1 + 2 + 3 = 4.9999999。具体可以参考：Felix Zhang：从加法结合律谈浮点数计算的可重复性

    这种不确定性带来的精度差异可能会造成最后模型的准确率有一定的变化。如果有这种情况，最好设置torch使用确定性的算法，即

    torch.use_deterministic_algorithms(True)
    """

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)
    cudnn.enabled = True

# 模型

import torch.nn as nn
from myloss2 import CompoundLoss
# 切分成两个
from myloss import ContourLoss,IoULoss,StructureLoss,PatchIoULoss,ThrReg_loss,SSIMLoss

MYLOCK = False

import gc
def model_init():
    gc.collect()
    torch.cuda.empty_cache()

def one_hot_encode(label, num_classes):
    # 创建全零张量，增加通道维度
    one_hot = torch.zeros(
        (label.shape[0], num_classes, *label.shape[1:]),
        device=label.device
    )
    # 使用scatter_填充one-hot编码（含背景类）
    return one_hot.scatter_(1, label.unsqueeze(1), 1)

def objective(single_trial, device_id):
    model_init()

    trial = optuna.integration.TorchDistributedTrial(single_trial)
    if device_id == 0:
        logger.info(f"trial.number:{trial.number}")

    # 把随机种子也考虑进去 4个
    # myseed = trial.suggest_int("seed", 0, 3)
    myseed = 1
    if myseed == 0:
        myseed = 1
    elif myseed == 1:
        myseed = 42
    elif myseed == 2:
        myseed = 3407
    elif myseed == 3:
        myseed = 114514
    if MYLOCK:
        myseed = params1["seed"]
    set_seed(myseed)
    model = DPT(encoder_size="base", features=128, out_channels = [96, 192, 384, 768], nclass = nclass)

    # state_dict = torch.load('./pretrained/dinov2_base.pth')
    # model.backbone.load_state_dict(state_dict)


    state_dict = torch.load('/home/notebook/code/personal/80410839/Uni/FIND_FINAL/epoch4_ema_82.87.pth')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    del new_state_dict, state_dict
    model_init()

    # 对于较大变化的值，使用对数学习率
    # lr0 = trial.suggest_float("lr0", 1e-6, 5e-5, log=True) #1e-5  #5e-6 这个是之前的
    lr0 = trial.suggest_float("lr0", 1e-7, 5e-6, log=True) #1e-5  #5e-6 这个是之前的
    lr_expand = trial.suggest_int("lr_expand", 5, 100, log=True) #1e-5  #5e-6 这个是之前的

    if MYLOCK:
        lr0 = params1["lr"]

    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': lr0},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': lr0 * lr_expand}
        ], 
        lr=lr0, betas=(0.9, 0.999), weight_decay=0.01
    )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(device_id)

    model = DDP(
        model, 
        device_ids=[device_id],
        broadcast_buffers=False,
        find_unused_parameters=True,
        output_device=device_id
        )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    # loss需要修整一下
    # ce regloss iou_loss ssim_loss dice_loss 梯度裁剪到10就行了  `conf_thresh` 0.8-0.99
    # 还需要注意训练要分成两部分
    # r_ce = trial.suggest_int("r_ce", 1, 100, log=True)
    r_ce = trial.suggest_float("r_ce", 0, 5)
    r_con = trial.suggest_float("r_con", 0, 5)
    r_com = trial.suggest_float("r_com", 0, 5)
    r_ssm = trial.suggest_float("r_ssm", 0, 5)
    r_iou = trial.suggest_float("r_iou", 0, 5)
    r_str = trial.suggest_float("r_str", 0, 5)
    r_pat = trial.suggest_float("r_pat", 0, 5)
    r_thr = trial.suggest_float("r_thr", 0, 5)
    r_mae = trial.suggest_float("r_mae", 0, 5)
    r_mse = trial.suggest_float("r_mse", 0, 5)

    r_conf = trial.suggest_int("r_conf", 85, 98)
    r_conf = r_conf / 100.
    # r_conf = 0.9
    rlu = trial.suggest_float("rlu", 0, 5)
    if MYLOCK:
        r_reg = params1["r_reg"]
        r_iou = params1["r_iou"]
        r_ssim = params1["r_ssim"]
        r_dice = params1["r_dice"]
        rlu = params1["rlu"]
        r_conf = params1["r_conf"]

    class_weights = torch.tensor([1., r_ce]).cuda(device_id)
    ce = nn.CrossEntropyLoss(weight=class_weights).cuda(device_id)
    com = CompoundLoss(mode="multiclass").cuda(device_id)
    con = ContourLoss().cuda(device_id)
    iou = IoULoss().cuda(device_id)
    strc = StructureLoss().cuda(device_id)
    thrreg = ThrReg_loss().cuda(device_id)
    ssim = SSIMLoss().cuda(device_id)
    patch = PatchIoULoss().cuda(device_id)
    mae = nn.L1Loss().cuda(device_id)
    mse = nn.MSELoss()

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(device_id)

    # dataset
    # un_labelnum = trial.suggest_int("un_labelnum", 4000, 8000)
    un_labelnum = 55000

    shadow_ratio = trial.suggest_float("shadow_ratio", 0., 0.9)#, log=True)
    noise_ratio = trial.suggest_float("noise_ratio", 0.2, 0.5)
    flitet_pixel_num = trial.suggest_int("flitet_pixel_num", 1, 100, log=True)

    if MYLOCK:
        un_labelnum = params1["un_labelnum"]
        shadow_ratio = params1["shadow_ratio"]
        rotate_ratio = params1["rotate_ratio"]
        noise_ratio = params1["noise_ratio"]

    ratio_list = [
        shadow_ratio, 15, noise_ratio, flitet_pixel_num
    ]

    trainset_u = SemiDataset(
        data_root, 'train_u', crop_size, unlabeled_id_path,un_labelnum = un_labelnum,seed=myseed,ratio_list=ratio_list
    )
    if device_id == 0:
        logger.info(f"len(trainset_u):{len(trainset_u)}")
    # 有标注的数量，目的是跟未标注的数量持平，所以最好在未标注数量处限制图片数目
    trainset_l = SemiDataset(
        data_root, 'train_l', crop_size, labeled_id_path, nsample=len(trainset_u),ratio_list=ratio_list
    )
    valset = SemiDataset(
        data_root, 'val', crop_size
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=batch_size, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )

    total_iters = len(trainloader_u) * EPOCHS # * 2 #EPOCHS // 2

    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    m1 = 14 # trial.suggest_int("grid_max", 5, 20)
    # if MYLOCK:
    #     m1 = params1["grid_max"]
    n1 = 3

    ber = 0
    ber_ema = 0

    for epoch in range(epoch + 1, EPOCHS):
        if device_id == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u)
        model.train()

        import time
        time1 = time.time()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):

            # if device_id == 0:
            #     logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
            #                 'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
            img_x, mask_x = img_x.cuda(device_id), mask_x.cuda(device_id)
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(device_id), img_u_s1.cuda(device_id), img_u_s2.cuda(device_id)
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(device_id), cutmix_box1.cuda(device_id), cutmix_box2.cuda(device_id)
            
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]


            def convert_label_to_one_hot(label, num_classes, cal_back):
                # 过滤背景类（标签0），生成 [b, c-1, w, h] 的二值标签
                batch_size, h, w = label.shape
                if cal_back:
                    one_hot = torch.zeros((batch_size, num_classes, h, w), device=label.device)
                    one_hot.scatter_(1, label.unsqueeze(1), 1)
                    # one_hot = F.one_hot(label, num_classes).to(label.device)
                else:
                    one_hot = torch.zeros((batch_size, num_classes-1, h, w), device=label.device)
                    valid_mask = (label > 0)  # 背景类为0，不参与填充
                    valid_labels = label[valid_mask] - 1  # 调整类别索引从0开始
                    # one_hot.scatter(0, valid_labels, 1)
                    one_hot = F.one_hot(valid_labels, num_classes-1).to(label.device).float().view(-1, num_classes-1)
                # 将非背景类别的标签转换为one-hot（注意标签减1以匹配通道索引）
                return one_hot

            # loss
            mask_x[mask_x>nclass] = 0
            mask_onehot = convert_label_to_one_hot(mask_x, nclass, True)


            # mask_x = F.one_hot(mask_x, num_classes=nclass)
            # mask_x = mask_x.permute(0,3,1,2).contiguous().float().cuda(device_id)

            # wbce和wiou不需要经过sigmoid
            # print(pred_x[:,i,...].max())
            # print(mask_onehot.max())
            # bce_loss = None
            # for i in range(1, nclass):
            #     # print(pred_x[:,i,...].sigmoid().shape)
            #     # print(mask_onehot[:,i,...].shape)
            #     bce_loss = bce(pred_x[:,i,...].sigmoid(), mask_onehot[:,i,...]) if bce_loss is None else bce_loss + bce(pred_x[:,i,...].sigmoid(), mask_onehot[:,i,...])

            # structure_loss = structure(pred_x, mask_onehot)

            # pred_sig = pred_x.sigmoid()
            if trial.suggest_int("sigmoid", 0, 1):
                pred_sig = pred_x.sigmoid()
            else:
                pred_sig = torch.softmax(pred_x, dim=1)
            gt_proportion, valid_mask = com.get_gt_proportion("multiclass", mask_x, pred_x.shape)
            pred_proportion = com.get_pred_proportion("multiclass", pred_x, temp=10, valid_mask=valid_mask)

            loss_x = (  ce(pred_x, mask_x) + 
                        (pred_proportion - gt_proportion).abs().mean() * r_com + 
                        con(pred_sig, mask_onehot) * r_con + 
                        iou(pred_sig, mask_onehot) * r_iou + 
                        strc(pred_x, mask_onehot) * r_str + 
                        thrreg(pred_sig, mask_onehot) * r_thr + 
                        ssim(pred_sig, mask_onehot) * r_ssm + 
                        patch(pred_sig, mask_onehot) * r_pat + 
                        mae(pred_sig, mask_onehot) * r_mae + 
                        mse(pred_sig, mask_onehot) * r_mse 
                        
                        )/(
                            1 + r_com + r_con + r_iou + r_str + r_thr + r_ssm + r_pat + r_mae + r_mse
                        )

            if torch.isinf(loss_x):
                raise optuna.exceptions.OptunaError(f"Trial was pruned at epoch {epoch}.")

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= r_conf) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= r_conf) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + rlu * loss_u_s) / (1 + rlu)
            
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),  max(n1, m1-(m1-n1)*(epoch+1)/EPOCHS)) # max_norm =float("inf")
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= r_conf) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i

            lr = lr0 * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * lr_expand
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if (i % 10 == 0) and (device_id == 0):
                cur_cost = time.time() - time1
                # time1
                time1 = time.time()
                logger.info('Iters: {:}/cost:{:.1f}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, cur_cost,optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))

        # 计算,最大iou

        model.eval()

        total_iou = 0.
        cnt = 0

        with torch.no_grad():
            for img, mask, id in valloader:
                img = img.cuda(device_id)
                pred = model(img)
                pred = pred.argmax(dim=1)

                mask_ = mask.cpu().numpy()[0,...].astype(np.uint8)
                pred_ = pred.cpu().numpy()[0,...].astype(np.uint8)
                for i in range(1, nclass):
                    # 如果画面中没有对应索引的真值，就跳过
                    if np.sum(mask_==i)==0:continue
                    total_iou += s2fb_iou((mask_==i).astype(np.uint8), (pred_==i).astype(np.uint8))
                    cnt += 1

        total_iou_tensor = torch.tensor([total_iou], dtype=torch.float).to(device_id)
        cnt_tensor = torch.tensor([cnt], dtype=torch.int).to(device_id)

        dist.all_reduce(total_iou_tensor)
        dist.all_reduce(cnt_tensor)

        ber = total_iou_tensor.item() / cnt_tensor.item()

        total_iou = 0.

        with torch.no_grad():
            for img, mask, id in valloader:
                img = img.cuda(device_id)
                pred = model_ema(img)
                pred = pred.argmax(dim=1)

                mask_ = mask.cpu().numpy()[0,...].astype(np.uint8)
                pred_ = pred.cpu().numpy()[0,...].astype(np.uint8)
                for i in range(1, nclass):
                    # 如果画面中没有对应索引的真值，就跳过
                    if np.sum(mask_==i)==0:continue
                    total_iou += s2fb_iou((mask_==i).astype(np.uint8), (pred_==i).astype(np.uint8))
                    cnt += 1

        total_iou_tensor = torch.tensor([total_iou], dtype=torch.float).to(device_id)

        dist.all_reduce(total_iou_tensor)

        ber_ema = total_iou_tensor.item() / cnt_tensor.item()

        if device_id == 0:
            logger.info(f"epoch:{epoch} ber:{ber} ber_ema:{ber_ema}")
        
        trial.report(max(ber, ber_ema), epoch)

        is_best = ber > previous_best
        is_best_ema = ber_ema > previous_best_ema
        previous_best = max(ber, previous_best)
        previous_best_ema = max(ber_ema, previous_best_ema)
        if ber == previous_best:
            best_epoch = epoch
        
        if ber_ema == previous_best_ema:
            best_epoch_ema = epoch

        if device_id == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'best_epoch': best_epoch,
            }
            
            checkpoint_ema = {
                'model': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best_ema,
                'best_epoch': best_epoch_ema
            }

            if is_best or (epoch + 1 == EPOCHS):
                Path(f"./EXP/{trial.number}").mkdir(parents=True,exist_ok=True)
                torch.save(checkpoint, os.path.join(f"./EXP/{trial.number}/epoch{epoch}_{100*previous_best:.2f}.pth"))
            if is_best_ema or (epoch + 1 == EPOCHS):
                Path(f"./EXP/{trial.number}").mkdir(parents=True,exist_ok=True)
                torch.save(checkpoint_ema, os.path.join(f"./EXP/{trial.number}/epoch{epoch}_ema_{100*previous_best_ema:.2f}.pth"))
            
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")

    torch.distributed.barrier()
    return max(previous_best,previous_best_ema)

# 初始化
optuna.logging.set_verbosity(optuna.logging.DEBUG) # DEBUG WARNING
import shutil
try:
    shutil.rmtree("log")
except:
    pass

logger = init_log('global', logging.INFO)
logger.propagate = 0

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str,default='configs/coco.yaml')
# parser.add_argument('--gpus', type=int, default=2)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
# parser.add_argument('--master_port', type=str, default='12345')

import datetime
def setup(backend, rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size,
    # timeout=datetime.timedelta(seconds=100)
    )

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % world_size)

    dist.init_process_group(
        backend="nccl", # nccl
        world_size=world_size,
        rank=rank,
        # init_method='env://'
    )

    rank = dist.get_rank()
    study = None

    dist.barrier()

    if rank == 0:
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
            direction="maximize", # maximize minimize
            study_name="unimatch2",
            storage=storage_name,load_if_exists=True
            )
        study.enqueue_trial( # 13
            {
                "lr0": 2.2756262235044394e-07,
                "lr_expand":18,
                "r_ce":0.12045148708003106,
                "r_con": 4.037333913961869,
                "r_com": 1.2735196687279189,
                "r_ssm": 1.9915494164252385,
                "r_iou": 2.7712353147147777,
                "r_str": 3.303578906512341,
                "r_pat": 4.943927557025036,
                "r_thr": 2.9322656106131904,
                "r_mae": 3.6891568933263215,
                "r_mse": 0.6594356662237959,
                "r_conf":91,
                "rlu":0.9487338113110795,
                "shadow_ratio":0.24161266633985756,
                "noise_ratio": 0.33859248006729503,
                "flitet_pixel_num": 6,
                "sigmoid":1,
            }
        )

        study.enqueue_trial( # 19
            {
                "lr0": 1.7385385440442103e-07,
                "lr_expand":63,
                "r_ce":1.4084455525171637,
                "r_con": 4.2872443735061765,
                "r_com": 0.14619242615867112,
                "r_ssm": 2.3178239314977227,
                "r_iou": 1.7878710218179483,
                "r_str": 3.6431744894762073,
                "r_pat": 1.3030892296333718,
                "r_thr": 2.193634402437535,
                "r_mae": 3.993322339942765,
                "r_mse": 2.162013368610162,
                "r_conf":90,
                "rlu":0.5348295181379762,
                "shadow_ratio":0.32988421155683917,
                "noise_ratio": 0.32479435054463984,
                "flitet_pixel_num": 6,
                "sigmoid":1,
            }
        )

        study.enqueue_trial( # 7
            {
                "lr0": 1.1868129608093652e-07,
                "lr_expand":15,
                "r_ce": 1.538134229295061,
                "r_con": 4.811071278563332,
                "r_com": 1.799703885892761,
                "r_ssm": 1.4405331208337924,
                "r_iou": 2.412723891083339,
                "r_str": 2.490831839940551,
                "r_pat": 1.7561831573942395,
                "r_thr": 3.0953146289146987,
                "r_mae": 4.136084430082587,
                "r_mse": 1.8444717418779217,
                "r_conf":90,
                "rlu":0.7846427203103212,
                "shadow_ratio":0.15059984356869419,
                "noise_ratio": 0.37685551777093473,
                "flitet_pixel_num": 42,
                "sigmoid":1,
            }
        )

        study.enqueue_trial( # 23
            {
                "lr0": 2.670310356195988e-07,
                "lr_expand":28,
                "r_ce": 0.6592567282244022,
                "r_con": 3.257472402997473,
                "r_com": 2.6834008977653037,
                "r_ssm": 1.268689294711708,
                "r_iou": 0.7581028429919345,
                "r_str": 2.3900063402908462,
                "r_pat": 4.403367833845323,
                "r_thr": 4.045521842970018,
                "r_mae": 4.329740467736126,
                "r_mse": 1.1368356872941778,
                "r_conf":90,
                "rlu":1.406971799178066,
                "shadow_ratio":0.44827888584355613,
                "noise_ratio": 0.34562078110091377,
                "flitet_pixel_num": 2,
                "sigmoid":1,
            }
        )

        study.optimize(
            partial(objective, device_id=rank),
            n_trials=N_TRIALS,
            timeout=7200000,
        )
        # return_dict["study"] = study
    else:
        for j in range(N_TRIALS):
            try:
                cnt = j
                objective(None, rank)
            except optuna.TrialPruned:
                pass

    if rank == 0:
        assert study is not None
        
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))