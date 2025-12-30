import torch
import torch.nn as nn

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
# from nnunet.training.loss_functions.focal_loss import FocalLoss
# from nnunet.utilities.nd_softmax import softmax_helper

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"
EPS: float = 1e-10


# SegLossBias
def expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        elif labels.dim() == 4:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2], inds[3]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels, valid_mask


def get_region_proportion(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        # x = torch.einsum("bcxyz,bxyz->bcxyz", x, valid_mask)
        # cardinality = torch.einsum("bxyz->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
        if valid_mask.dim() == 4:
            x = torch.einsum("bcwh, bcwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bcwh->bc", valid_mask)
        else:
            x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3] * x.shape[4]

    # region_proportion = (torch.einsum("bcxyz->bc", x) + EPS) / (cardinality + EPS)
    # region_proportion = (torch.sum(x, dim=(2, 3)) + EPS) / (cardinality + EPS)
    region_proportion = (torch.einsum("bcwh->bc", x) + EPS) / (cardinality + EPS)
    # region_proportion = (torch.sum(x, dim=(2, 3, 4)) + EPS) / (cardinality + EPS)

    return region_proportion


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn
    


def _iou2(pred, target, num_classes, size_average=True, ignore_index=255):
    """
    多分类IoU损失函数

    参数:
    pred (Tensor): 预测结果，形状为(batch_size, H, W)，每个元素为类别索引
    target (Tensor): 目标标签，形状为(batch_size, H, W)，每个元素为类别索引
    num_classes (int): 类别总数（包括背景）
    size_average (bool): 是否对类别的损失取平均，否则返回总和
    ignore_index (int): 需要忽略的类别索引

    返回:
    Tensor: 计算得到的IoU损失
    """
    batch_size = pred.shape[0]
    total_intersection = torch.zeros(num_classes, device=pred.device)
    total_union = torch.zeros(num_classes, device=pred.device)

    for i in range(batch_size):
        # 创建掩码排除忽略的像素
        mask = (target[i] != ignore_index)
        pred_i = pred[i]
        target_i = target[i]

        # 转换为one-hot编码（形状: C×H×W）
        pred_oh = torch.nn.functional.one_hot(pred_i, num_classes=num_classes).permute(2, 0, 1).float()
        target_oh = torch.nn.functional.one_hot(target_i, num_classes=num_classes).permute(2, 0, 1).float()

        # 应用掩码
        mask_expanded = mask.unsqueeze(0).float()  # 扩展为(1, H, W)
        pred_oh_masked = pred_oh * mask_expanded
        target_oh_masked = target_oh * mask_expanded

        # 计算当前样本的交集和并集
        intersection = (pred_oh_masked * target_oh_masked).sum(dim=(1, 2))
        sum_pred = pred_oh_masked.sum(dim=(1, 2))
        sum_target = target_oh_masked.sum(dim=(1, 2))
        union = sum_pred + sum_target - intersection

        # 累加到总计
        total_intersection += intersection
        total_union += union

    # 计算每个类别的IoU
    iou_per_class = torch.zeros(num_classes, device=pred.device)
    valid_mask = total_union > 0
    iou_per_class[valid_mask] = total_intersection[valid_mask] / (total_union[valid_mask] + 1e-8)
    # 处理并集为零且交集也为零的情况（视为正确预测）
    iou_per_class[total_union == 0] = 1.0

    # 计算损失（1 - IoU）
    loss_per_class = 1.0 - iou_per_class

    # 根据size_average决定返回平均损失还是总和
    if size_average:
        return loss_per_class.mean()
    else:
        return loss_per_class.sum()
def _iou2(pred, target, size_average = True, ignore_index=255):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(b):
        #compute the IoU of the foreground
        mask = (target[i] != ignore_index).float()

        pred_masked = pred[i] * mask
        target_masked = target[i] * mask

        Iand1 = torch.sum(pred_masked*target_masked)
        Ior1 = torch.sum(target_masked) + torch.sum(pred_masked)-Iand1

        if Ior1 > 0:  # 避免除零错误
            IoU1 = Iand1 / Ior1
        else:
            IoU1 = torch.tensor(0.0, device=pred.device)
        # IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    if size_average:
        return IoU / b
    else:
        return IoU

class IOU(torch.nn.Module):
    def __init__(self, size_average = True, ignore_index=None):
        super(IOU, self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average,self.ignore_index)

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, pred, target):
        """
        pred:   [b, 2, 512, 512] 模型输出(未归一化)
        target: [b, 512, 512]    真实标签(0/1/ignore_index)
        """

        pred = torch.softmax(pred, dim=1)  # [b, 2, 512, 512]

        # 确保target中的值在合法范围内
        valid_mask = (target != self.ignore_index)
        target_clone = target.clone()
        
        # 将ignore_index位置临时填充为0（避免one_hot报错）
        target_clone[~valid_mask] = 0
        
        # 生成one-hot编码 (自动过滤非法值)
        target_one_hot = F.one_hot(
            target_clone.long(), 
            num_classes=2
        ).permute(0, 3, 1, 2).float()  # [b, 2, 512, 512]

        # 创建三维掩码 [b, 1, 512, 512]
        valid_mask = valid_mask.unsqueeze(1)

        # 应用掩码（忽略无效区域）
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask

        # 计算Dice系数
        intersection = (pred * target_one_hot).sum(dim=(2, 3))        # [b, 2]
        cardinality = (pred + target_one_hot).sum(dim=(2, 3))         # [b, 2]
        dice_coeff = (2. * intersection + self.eps) / (cardinality + self.eps)  # [b, 2]
        
        return 1.0 - dice_coeff.mean()  # 对多通道取平均

class CompoundLoss(nn.Module):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """
    def __init__(self, mode: str,
                 alpha: float = 1., # ?1.2 | 5
                 factor: float = 1.,
                 step_size: int = 0, # 6
                 max_alpha: float = 100.,
                 temp: float = 10., # ? # 温度
                 ignore_index: int = 255,
                 background_index: int = -1,
                 weight: Optional[torch.Tensor] = None) -> None:
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight
        self.iouloss = IOU(size_average=True,ignore_index=ignore_index)
        self.diceloss = DiceLoss(ignore_index=ignore_index)

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor):
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
        if self.mode == MULTICLASS_MODE:
            loss = F.cross_entropy(
                inputs, labels.long(), weight=self.weight, ignore_index=self.ignore_index)
        else:
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            print(
                "CompoundLoss : Adjust the tradoff param alpha : {:.3g} -> {:.3g}".format(curr_alpha, self.alpha)
            )

    def get_gt_proportion(self, mode: str,
                          labels: torch.Tensor,
                          target_shape,
                          ignore_index: int = 255):
        if mode == MULTICLASS_MODE:
            bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            bin_labels = labels
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(self, mode: str,
                            logits: torch.Tensor,
                            temp: float = 1.0,
                            valid_mask=None):
        if mode == MULTICLASS_MODE:
            preds = F.log_softmax(temp * logits, dim=1).exp()
        else:
            preds = F.logsigmoid(temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion


# 这种loss不适合使用空mask训练
class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        # ce term
        labels2 = labels.clone() # 
        labels2[labels2>2] = 0
        one_hot = F.one_hot(labels2, num_classes=2)
        labels2 = one_hot.permute(0,3,1,2).float().to(labels.device)

        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
        labels = labels.long()

        
        loss_ce = self.cross_entropy(inputs, labels)
        # print("loss_ce:",loss_ce)
        # regularization
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        
        loss_reg = (pred_proportion - gt_proportion).abs().mean()
        # print("loss_reg:",loss_reg)

        labels2[:,0,...][labels>2] = 255
        labels2[:,1,...][labels>2] = 255
        pred = torch.softmax(inputs, dim=1)

        # 再加上一个iou loss
        
        iou_loss = self.iouloss(pred,labels2)
        # print(pred.shape)
        # print(labels.shape)
        # print(labels2.shape)
        dice_loss = self.diceloss(pred,labels2)
        print(dice_loss)
        # 4 : 1 : 4
        # 设置三种比例 2 1 2 
        loss = loss_ce + dice_loss # loss_reg*4 + iou_loss + 
        # print("loss_ce:",loss_ce)
        # print("loss_reg:",loss_reg)
        # print("iou_loss:",iou_loss)

        # return loss, loss_ce, loss_reg
        return loss / 3.


