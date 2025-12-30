import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from config import Config


class ContourLoss(torch.nn.Module):
    def __init__(self):
        super(ContourLoss, self).__init__()

    def forward(self, pred, target, weight=10):
        '''
        target, pred: tensor of shape (B, C, H, W), where target[:,:,region_in_contour] == 1,
                        target[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = pred[:,:,1:,:] - pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
        delta_c = pred[:,:,:,1:] - pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)

        delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c) 

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.

        c_in  = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in  = torch.mean( pred     * (target - c_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean( (1-pred) * (target - c_out)**2 ) 
        region = region_in + region_out

        loss =  weight * length + region

        return loss

class IoULoss2(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        # pred: [B, C, H, W] (0-1 probabilities)
        # target: [B, C, H, W] (0 or 1)
        b = pred.shape[0]
        IoU = 0.0
        smooth = 1e-6 # 关键：防止除以零
        
        # 向量化计算，比 for 循环更快且更稳定
        # Flatten: [B, -1]
        pred_flat = pred.view(b, -1)
        target_flat = target.view(b, -1)
        
        intersection = (pred_flat * target_flat).sum(1)
        total = (pred_flat + target_flat).sum(1)
        union = total - intersection
        
        IoU = (intersection + smooth) / (union + smooth)
        
        # IoU Loss = 1 - Mean IoU
        return 1 - IoU.mean()

class IoULoss(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] (0-1 probabilities)
        target: [B, C, H, W] (0 or 1)
        """
        # 处理维度：如果是 [B, C, H, W]，需要处理每个通道
        if pred.dim() == 4:
            B, C, H, W = pred.shape
            # 将 [B, C, H, W] 转换为 [B*C, H, W] 或直接处理
            pred_reshaped = pred.view(B * C, H, W)
            target_reshaped = target.view(B * C, H, W)
        else:
            # 已经是 [B, H, W] 格式
            pred_reshaped = pred
            target_reshaped = target
        
        # 【关键】强制二值化！
        # 只有足够确信(>threshold)的像素才算数。半透明的像素直接被切掉。
        pred_hard = (pred_reshaped > self.threshold).float()
        
        intersection = (pred_hard * target_reshaped).sum(dim=(1, 2))
        union = pred_hard.sum(dim=(1, 2)) + target_reshaped.sum(dim=(1, 2)) - intersection
        
        # 加上 eps 防止除零
        iou = intersection / (union + 1e-6)
        
        # IoU Loss = 1 - Mean IoU
        return 1 - iou.mean()

class StructureLoss(torch.nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, target):
        weit  = 1+5*torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15)-target)
        wbce  = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

        pred  = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou  = 1-(inter+1)/(union-inter+1)

        return (wbce+wiou).mean()


class PatchIoULoss(torch.nn.Module):
    def __init__(self):
        super(PatchIoULoss, self).__init__()
        self.iou_loss = IoULoss() # 使用上面修正后的 IoULoss

    def forward(self, pred, target):
        win_y, win_x = 64, 64
        iou_loss = 0.
        count = 0
        H, W = target.shape[2], target.shape[3]
        
        for anchor_y in range(0, H, win_y):
            for anchor_x in range(0, W, win_x):
                # 确保 Patch 不越界
                end_y = min(anchor_y+win_y, H)
                end_x = min(anchor_x+win_x, W)
                
                # 如果 Patch 太小，可以跳过
                if end_y - anchor_y < 16 or end_x - anchor_x < 16:
                    continue

                patch_pred = pred[:, :, anchor_y:end_y, anchor_x:end_x]
                patch_target = target[:, :, anchor_y:end_y, anchor_x:end_x]
                
                patch_iou_loss = self.iou_loss(patch_pred, patch_target)
                iou_loss += patch_iou_loss
                count += 1
        
        if count > 0:
            return iou_loss / count
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)



class ThrReg_loss(torch.nn.Module):
    def __init__(self):
        super(ThrReg_loss, self).__init__()

    def forward(self, pred, gt=None):
        return torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))


class AlphaConsistencyLoss(torch.nn.Module):
    """
    Alpha Consistency Loss: 使用软阈值方案惩罚预测值接近 0.5 的情况，减少脏预测（半透明区域）
    核心思想：
    - 预测的不确定性：alpha_pred * (1 - alpha_pred) (越接近0.5越大)
    - GT 的确定性作为权重：(alpha_gt - 0.5)^2 * 4 (GT越接近0或1，权重越大；GT越接近0.5，权重越小)
    - 加权计算：torch.mean(consistency_loss * weight)
    这样就不需要手动设定阈值了，GT 越接近 0 或 1，惩罚得越重；GT 越接近 0.5，惩罚得越轻。
    """
    def __init__(self, eps=1e-6, use_mask=True):
        super(AlphaConsistencyLoss, self).__init__()
        self.eps = eps
        self.use_mask = use_mask  # 保留参数以保持向后兼容性，但软阈值方案会自动处理

    def forward(self, alpha_pred, alpha_gt=None):
        """
        Args:
            alpha_pred: 模型预测的 alpha，范围 [0, 1]，形状 [B, C, H, W] 或 [B, H, W]
            alpha_gt: 真实的 alpha 标签，形状与 alpha_pred 相同（用于软阈值权重计算）
        """
        # 确保 alpha_pred 在 [0, 1] 范围内（如果输入是 logits，需要先 sigmoid）
        if alpha_pred.min() < 0 or alpha_pred.max() > 1:
            alpha_pred = torch.sigmoid(alpha_pred)
        
        # 1. 预测的不确定性 (越接近0.5越大)
        consistency_loss = alpha_pred * (1 - alpha_pred)
        
        if alpha_gt is not None:
            # 2. GT 的确定性 (作为权重)
            # 当 alpha_gt 为 0 或 1 时，weight 为 1 (完全惩罚)
            # 当 alpha_gt 为 0.5 时，weight 为 0 (完全不惩罚，保护细节)
            # 公式：(alpha_gt - 0.5)^2 * 4
            # alpha_gt=0 -> weight=1
            # alpha_gt=1 -> weight=1
            # alpha_gt=0.5 -> weight=0
            weight = torch.pow(alpha_gt - 0.5, 2) * 4
            
            # 3. 加权计算
            loss = torch.mean(consistency_loss * weight)
        else:
            # 如果没有提供 alpha_gt，回退到全局版本（对所有区域施加惩罚）
            loss = torch.mean(consistency_loss)
        
        return loss


class ClsLoss(nn.Module):
    """
    Auxiliary classification loss for each refined class output.
    """
    def __init__(self):
        super(ClsLoss, self).__init__()
        self.config = Config()
        self.lambdas_cls = self.config.lambdas_cls

        self.criterions_last = {
            'ce': nn.CrossEntropyLoss()
        }

    def forward(self, preds, gt):
        loss = 0.
        for _, pred_lvl in enumerate(preds):
            if pred_lvl is None:
                continue
            for criterion_name, criterion in self.criterions_last.items():
                loss += criterion(pred_lvl, gt) * self.lambdas_cls[criterion_name]
        return loss


class PixLoss(nn.Module):
    """
    Pixel loss for each refined map output.
    Supports both single mask and multi-mask outputs.
    """
    def __init__(self, full_mask_lambda: float = 0.01, decay_rate: float = 0.2):
        super(PixLoss, self).__init__()
        self.config = Config()
        self.lambdas_pix_last = self.config.lambdas_pix_last
        self.full_mask_lambda = full_mask_lambda
        self.decay_rate = decay_rate

        self.criterions_last = {}
        # 对于多 mask 训练，使用 reduction='none' 以便计算每个样本的损失
        # 在单 mask 训练中，会在 forward 中手动计算均值
        if 'bce' in self.lambdas_pix_last and self.lambdas_pix_last['bce']:
            # self.criterions_last['bce'] = nn.BCELoss(reduction='none')
            self.criterions_last['bce'] = nn.BCEWithLogitsLoss(reduction='none')
        if 'iou' in self.lambdas_pix_last and self.lambdas_pix_last['iou']:
            self.criterions_last['iou'] = IoULoss()
        if 'iou_patch' in self.lambdas_pix_last and self.lambdas_pix_last['iou_patch']:
            self.criterions_last['iou_patch'] = PatchIoULoss()
        if 'ssim' in self.lambdas_pix_last and self.lambdas_pix_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mae' in self.lambdas_pix_last and self.lambdas_pix_last['mae']:
            self.criterions_last['mae'] = nn.L1Loss(reduction='none')
        if 'mse' in self.lambdas_pix_last and self.lambdas_pix_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss(reduction='none')
        if 'reg' in self.lambdas_pix_last and self.lambdas_pix_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()
        if 'cnt' in self.lambdas_pix_last and self.lambdas_pix_last['cnt']:
            self.criterions_last['cnt'] = ContourLoss()
        if 'structure' in self.lambdas_pix_last and self.lambdas_pix_last['structure']:
            self.criterions_last['structure'] = StructureLoss()
        if 'alpha_cons' in self.lambdas_pix_last and self.lambdas_pix_last['alpha_cons']:
            # 使用软阈值方案，不需要手动设定阈值
            self.criterions_last['alpha_cons'] = AlphaConsistencyLoss(use_mask=True)
        if 'aleatoric' in self.lambdas_pix_last and self.lambdas_pix_last['aleatoric']:
            # 不确定性感知的 Matting Loss
            self.criterions_last['aleatoric'] = AleatoricMattingLoss()

    @staticmethod
    def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute IoU between predictions and targets.
        pred: [B, C, H, W] or [B, 1, H, W] or [B, H, W] (Probabilities 0~1)
        target: [B, C, H, W] or [B, 1, H, W] or [B, H, W] (0 or 1)
        """
        # 处理维度：统一转换为 [B, H, W] 格式
        if pred.dim() == 4:
            B, C, H, W = pred.shape
            if C == 1:
                # [B, 1, H, W] -> [B, H, W]
                pred_reshaped = pred.squeeze(1)
                target_reshaped = target.squeeze(1)
            else:
                # [B, C, H, W] -> [B*C, H, W]
                pred_reshaped = pred.view(B * C, H, W)
                target_reshaped = target.view(B * C, H, W)
        else:
            # [B, H, W]
            pred_reshaped = pred
            target_reshaped = target
        
        # 【关键】强制二值化！
        # 只有足够确信(>threshold)的像素才算数。半透明的像素直接被切掉。
        pred_hard = (pred_reshaped > threshold).float()
        
        intersection = (pred_hard * target_reshaped).sum(dim=(1, 2))
        union = pred_hard.sum(dim=(1, 2)) + target_reshaped.sum(dim=(1, 2)) - intersection
        
        # 加上 eps 防止除零
        iou = intersection / (union + 1e-6)
        
        # 如果原来是 [B, C, H, W] 且 C > 1，需要恢复形状并取平均
        if pred.dim() == 4 and pred.shape[1] > 1:
            iou = iou.view(B, C).mean(dim=1)
        
        return iou

    def compute_single_mask_loss(self, scaled_preds, gt, pix_loss_lambda=1.0, epoch=0):
        """Compute loss for single mask prediction (original behavior)."""
        loss = 0.
        loss_dict = {}
        
        # 处理 scaled_preds：可能是 list 或 dict
        if isinstance(scaled_preds, dict):
            pred_list = scaled_preds.get('pred_masks', [])
            if isinstance(pred_list, torch.Tensor):
                pred_list = [pred_list]
        else:
            pred_list = scaled_preds
        
        for idx, pred_lvl in enumerate(pred_list):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.functional.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            for criterion_name, criterion in self.criterions_last.items():
                if 'aleatoric' in criterion_name:
                    # AleatoricMattingLoss 需要 pred_alpha, log_var, gt_alpha
                    # 由于不再使用不确定性loss，跳过这个loss
                    continue
                elif 'structure' in criterion_name or 'bce' in criterion_name:
                    _loss_raw = criterion(pred_lvl, gt)
                elif 'alpha_cons' in criterion_name:
                    # AlphaConsistencyLoss 需要 sigmoid 后的预测值和 GT（用于软阈值权重计算）
                    pred_sigmoid = pred_lvl.sigmoid()
                    _loss_raw = criterion(pred_sigmoid, gt)
                else:
                    _loss_raw = criterion(pred_lvl.sigmoid(), gt)
                # 处理 reduction='none' 的情况
                if isinstance(_loss_raw, torch.Tensor) and _loss_raw.dim() > 0:
                    _loss = _loss_raw.mean()
                else:
                    _loss = _loss_raw
                # 使用配置中的权重（在train_epoch中动态调整）
                _loss = _loss * self.lambdas_pix_last[criterion_name] * pix_loss_lambda
                loss += _loss
                loss_dict[criterion_name] = loss_dict.get(criterion_name, 0.) + _loss.item() / len(pred_list)
        return loss, loss_dict

    def compute_multi_mask_losses(self, pred_masks: torch.Tensor, target_masks: torch.Tensor, 
                                   epoch: int, pix_loss_lambda=1.0, pred_iou: torch.Tensor = None):
        """Compute loss for multi-mask prediction with best mask selection and 10% perturbation."""
        import math
        batch_size, num_masks = pred_masks.shape[:2]
        target_expanded = target_masks.unsqueeze(1).expand(-1, num_masks, -1, -1)
        exp_decay = self.full_mask_lambda * math.exp(-self.decay_rate * epoch)

        # Compute IoU once for mask selection
        pred_sigmoid = torch.sigmoid(pred_masks)
        pred_masks_flat = pred_sigmoid.contiguous().reshape(batch_size * num_masks, 1, *pred_masks.shape[2:])
        gt_masks_flat = target_expanded.contiguous().reshape(batch_size * num_masks, 1, *target_masks.shape[1:])
        # 计算真实IoU（需要梯度用于训练IoU预测头）
        ious = self.compute_iou(
            pred_masks_flat,
            gt_masks_flat,
        ).reshape(batch_size, num_masks)
        # 使用detach的ious来选择最优mask（避免影响梯度），但保留原始ious用于训练
        best_indices = ious.detach().argmax(dim=1)
        
        prob = max(0, (5.-epoch)/5 * 0.3)

        if prob > 0:
            # 生成扰动掩码 [B]
            perturbation_mask = torch.rand(batch_size, device=pred_masks.device) < prob
            
            # 生成随机索引 [B]
            random_indices = torch.randint(0, num_masks, (batch_size,), device=pred_masks.device)
            
            # 融合索引：如果是 True 则用随机索引，否则用最佳索引
            selected_indices = torch.where(perturbation_mask, random_indices, best_indices)
        else:
            # 超过预热 Epoch 后，直接使用最佳索引，节省计算资源
            selected_indices = best_indices

        total_loss = torch.tensor(0.0, device=pred_masks.device)
        # 确保 loss_dict 中的值都是标量（float），而不是 Tensor
        # 使用detach的ious来记录，避免影响梯度流
        # 记录selected_indices对应的IoU（考虑10%扰动）
        selected_ious = ious.detach().gather(1, selected_indices.unsqueeze(1)).squeeze(1)  # [B]
        loss_dict = {
            'best_iou': selected_ious.mean().item(),  # 实际参与训练的mask的IoU
            'true_best_iou': ious.detach().max(dim=1)[0].mean().item(),  # 真实的max IoU（用于监控）
            # 'gt_ious' 是 Tensor，不放入 loss_dict，避免格式化错误
        }

        for criterion_name, criterion in self.criterions_last.items():
            if 'aleatoric' in criterion_name:
                # AleatoricMattingLoss 需要 pred_alpha, log_var, gt_alpha
                # 由于不再使用不确定性loss，跳过这个loss
                continue
            elif 'structure' in criterion_name or 'bce' in criterion_name:
                pred = pred_masks
            elif 'alpha_cons' in criterion_name:
                # AlphaConsistencyLoss 需要 sigmoid 后的预测值
                pred = pred_sigmoid
            else:
                pred = pred_sigmoid
            
            pred_flat = pred.reshape(batch_size * num_masks, 1, *pred.shape[2:])
            target_flat = target_expanded.reshape(batch_size * num_masks, 1, *target_expanded.shape[2:])

            # 对于返回标量的损失函数，需要逐个样本计算
            # 检查损失函数类型，决定计算方式
            is_scalar_loss = criterion_name in ['iou', 'iou_patch', 'reg', 'structure', 'cnt', 'alpha_cons']
            
            if is_scalar_loss:
                # 对于返回标量的损失函数，逐个样本计算
                all_losses_per_sample = []
                for i in range(batch_size * num_masks):
                    sample_pred = pred_flat[i:i+1]
                    sample_target = target_flat[i:i+1]
                    # 所有标量 loss（包括 alpha_cons）都接受 (pred, target) 两个参数
                    # 对于 alpha_cons，pred 已经在前面被设置为 pred_sigmoid
                    sample_loss = criterion(sample_pred, sample_target)
                    # 确保是标量
                    if sample_loss.dim() > 0:
                        sample_loss = sample_loss.mean()
                    all_losses_per_sample.append(sample_loss)
                all_losses = torch.stack(all_losses_per_sample)  # [batch_size * num_masks]
                all_losses = all_losses.reshape(batch_size, num_masks)
            else:
                # 对于返回逐元素损失的函数（如 BCE, MAE, MSE, SSIM）
                all_losses = criterion(pred_flat, target_flat)
                
                # 处理不同形状的损失输出
                if all_losses.dim() == 4:
                    # [B*num_masks, 1, H, W] -> [B*num_masks]
                    all_losses = all_losses.mean(dim=(1, 2, 3))
                elif all_losses.dim() == 2:
                    # [B*num_masks, ...] -> [B*num_masks]
                    all_losses = all_losses.mean(dim=1)
                elif all_losses.dim() == 0:
                    # 标量：扩展到所有样本（不应该发生，但作为fallback）
                    scalar_loss = all_losses.item()
                    all_losses = torch.full((batch_size * num_masks,), scalar_loss, 
                                           device=pred_masks.device, dtype=pred_masks.dtype)
                elif all_losses.dim() == 1:
                    # 已经是 [B*num_masks] 形状
                    if all_losses.shape[0] != batch_size * num_masks:
                        # 形状不匹配，使用均值填充
                        scalar_loss = all_losses.mean().item()
                        all_losses = torch.full((batch_size * num_masks,), scalar_loss, 
                                               device=pred_masks.device, dtype=pred_masks.dtype)
                
                all_losses = all_losses.reshape(batch_size, num_masks)
            
            best_loss = all_losses.gather(1, selected_indices.unsqueeze(1)).mean()
            component_loss = best_loss + all_losses.mean() * exp_decay
            total_loss += self.lambdas_pix_last[criterion_name] * component_loss * pix_loss_lambda
            # 确保所有值都转换为标量（float）
            loss_dict.update({
                f"{criterion_name}_best": best_loss.item(),
                f"{criterion_name}_full": all_losses.mean().item(),
            })
        
        # 训练IoU预测头：使用真实IoU作为标签
        if pred_iou is not None:
            # pred_iou: [B, num_masks], ious: [B, num_masks]
            # 使用MSE损失训练IoU预测头
            iou_pred_loss = nn.functional.mse_loss(pred_iou, ious)
            # IoU预测损失的权重，可以根据需要调整
            iou_pred_weight = getattr(self.config, 'iou_pred_loss_weight', 0.05)
            total_loss += iou_pred_weight * iou_pred_loss
            loss_dict['iou_pred_loss'] = iou_pred_loss.item()
            loss_dict['iou_pred_mean'] = pred_iou.mean().item()
            loss_dict['gt_iou_mean'] = ious.mean().item()

        return total_loss, loss_dict

    def forward(self, scaled_preds, gt, pix_loss_lambda=1.0, epoch=0):
        """
        Forward pass supporting both single and multi-mask outputs.
        
        Args:
            scaled_preds: Can be:
                - List of tensors [B, 1, H, W] for single mask (original format)
                - Dict with 'pred_masks' [B, num_masks, H, W] and optionally 'pred_iou' [B, num_masks] for multi-mask
            gt: Ground truth mask [B, 1, H, W] or [B, H, W]
            pix_loss_lambda: Loss scaling factor
            epoch: Current epoch for decay calculation
        """
        # Handle multi-mask output (dict format)
        if isinstance(scaled_preds, dict) and 'pred_masks' in scaled_preds:
            pred_masks = scaled_preds['pred_masks']
            pred_iou = scaled_preds.get('pred_iou', None)  # 获取IoU预测，如果存在
            # Ensure gt is [B, H, W] format
            if gt.dim() == 4:
                gt = gt.squeeze(1)
            # Resize if needed
            if pred_masks.shape[2:] != gt.shape[1:]:
                pred_masks = nn.functional.interpolate(
                    pred_masks, size=gt.shape[1:], mode='bilinear', align_corners=True
                )
            return self.compute_multi_mask_losses(pred_masks, gt, epoch, pix_loss_lambda, pred_iou)
        
        # Handle single mask output (original format)
        return self.compute_single_mask_loss(scaled_preds, gt, pix_loss_lambda, epoch)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - (1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x, y):
    ssim = torch.mean(SSIM(x,y))
    return ssim


class AleatoricMattingLoss(nn.Module):
    """
    不确定性感知的 Matting Loss (Aleatoric Uncertainty Loss)
    核心思想：模型不仅预测 alpha，还预测每个像素的不确定性（方差）
    不确定性大的区域（如边界）会被自动降低权重，不确定性小的区域（如主体）会被加强
    
    公式: loss = (diff / variance) + 0.5 * log(variance)
    其中 diff = |pred_alpha - gt_alpha|, variance = exp(log_var)
    """
    def __init__(self):
        super(AleatoricMattingLoss, self).__init__()

    def forward(self, pred_alpha, log_var, gt_alpha):
        """
        Args:
            pred_alpha: [B, H, W] 或 [B, 1, H, W]  预测的 Mask (0~1)
            log_var:    [B, H, W] 或 [B, 1, H, W]  预测的不确定性 (实数域，无Sigmoid)
            gt_alpha:   [B, H, W] 或 [B, 1, H, W]  真值 Mask
        """
        # 统一处理维度：确保都是 [B, H, W]
        if pred_alpha.dim() == 4:
            pred_alpha = pred_alpha.squeeze(1)
        if log_var.dim() == 4:
            log_var = log_var.squeeze(1)
        if gt_alpha.dim() == 4:
            gt_alpha = gt_alpha.squeeze(1)
        
        # 1. 计算像素级 L1 差异
        diff = torch.abs(pred_alpha - gt_alpha)
        
        # 2. 不确定性加权 Loss
        # 公式: loss = (diff / variance) + 0.5 * log(variance)
        # log_var 也就是 log(sigma^2)
        # torch.exp(-log_var) 等于 1/sigma^2 (precision)
        
        # 为了数值稳定，可以对 log_var 做一个 clamp，防止溢出
        log_var = torch.clamp(log_var, min=-10, max=10)
        precision = torch.exp(-log_var)
        
        # 计算加权损失
        loss = (diff * precision) + (0.5 * log_var)
        
        return loss.mean()


def get_gradient(alpha):
    """
    计算 alpha 的梯度，用于识别边界区域
    
    Args:
        alpha: [H, W] 或 [B, H, W] 的 alpha 值
    
    Returns:
        gradient: 相同形状的梯度值
    """
    if alpha.dim() == 2:
        alpha = alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)  # [B, 1, H, W]
    
    # 使用 Sobel 算子计算梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=alpha.dtype, device=alpha.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=alpha.dtype, device=alpha.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(alpha, sobel_x, padding=1)
    grad_y = F.conv2d(alpha, sobel_y, padding=1)
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    
    # 恢复原始维度
    if gradient.shape[0] == 1 and gradient.shape[1] == 1:
        gradient = gradient.squeeze(0).squeeze(0)  # [H, W]
    elif gradient.shape[1] == 1:
        gradient = gradient.squeeze(1)  # [B, H, W]
    
    return gradient


def inference_ranking(pred_alphas):
    """
    推理时的 Mask 排序函数，用于选择最佳 Mask
    基于覆盖面积进行排序
    
    Args:
        pred_alphas: [K, H, W] 或 [B, K, H, W] 模型的 K 个预测
    
    Returns:
        best_mask: [H, W] 或 [B, H, W] 选出的最佳 Mask
        best_idx: 最佳 Mask 的索引
    """
    # 处理 batch 维度
    has_batch = pred_alphas.dim() == 4
    if not has_batch:
        pred_alphas = pred_alphas.unsqueeze(0)  # [1, K, H, W]
    
    B, K, H, W = pred_alphas.shape
    scores = []
    
    for k in range(K):
        alpha = pred_alphas[:, k]  # [B, H, W]
        
        # 基础分: 覆盖面积
        base_score = alpha.sum(dim=(1, 2))  # [B]
        
        scores.append(base_score)
    
    # scores: list of [B] tensors
    scores_tensor = torch.stack(scores, dim=1)  # [B, K]
    
    # 选分最高的
    best_indices = torch.argmax(scores_tensor, dim=1)  # [B]
    
    # 收集最佳 mask
    best_masks = []
    for b in range(B):
        best_idx = best_indices[b].item()
        best_masks.append(pred_alphas[b, best_idx])
    best_mask = torch.stack(best_masks, dim=0)  # [B, H, W]
    
    if not has_batch:
        best_mask = best_mask.squeeze(0)  # [H, W]
        best_indices = best_indices.squeeze(0)  # scalar
    
    return best_mask, best_indices
