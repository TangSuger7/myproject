import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sam2.build_sam import build_sam2
from timm.models.layers import trunc_normal_   


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMaskHeadNoLogVar(nn.Module):
    def __init__(self, in_channels, num_outputs, inter_features=32):
        super().__init__()
        self.num_outputs = num_outputs
        self.in_channels = in_channels
        self.inter_features = inter_features
        
        # 1. 静态特征提取 (不变)
        self.conv_feature = nn.Sequential(
            nn.Conv2d(in_channels, inter_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_features),
            nn.ReLU(inplace=True)
        )
        
        # 2. Token (不变)
        self.task_tokens = nn.Parameter(torch.randn(num_outputs, in_channels))
        
        # 3. 动态权重生成器 (变简单了)
        # 只需要生成 1 个 1x1 卷积核 (只负责 Mask)
        # 参数量减半: 2*inter -> 1*inter
        self.weight_generator = nn.Linear(in_channels, inter_features) 
        self.bias_generator = nn.Linear(in_channels, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.task_tokens, std=0.02)
        nn.init.constant_(self.weight_generator.weight, 0)
        nn.init.constant_(self.bias_generator.weight, 0)
        # Bias 初始化为 0 即可，不需要特殊的偏移，因为后面直接接 Sigmoid
        nn.init.constant_(self.bias_generator.bias, 0)

    def forward(self, x, target_size):
        B, C, H, W = x.shape
        feat = self.conv_feature(x) # [B, inter, H, W]
        
        # 生成权重: [N, inter, 1, 1]
        kernel_weights = self.weight_generator(self.task_tokens).view(self.num_outputs, self.inter_features, 1, 1)
        # 生成偏置: [N, 1]
        biases = self.bias_generator(self.task_tokens).view(self.num_outputs)
        
        outputs = []
        for i in range(self.num_outputs):
            # 取出第 i 个任务的卷积核
            w = kernel_weights[i].unsqueeze(0) # [1, inter, 1, 1]
            b = biases[i].view(1)             # [1]
            
            # 动态卷积: 同样的特征图，被不同的卷积核处理
            # out: [B, 1, H, W]
            out = F.conv2d(feat, w, bias=b)
            outputs.append(out)
            
        # 堆叠结果: [B, N, 1, H, W] -> [B, N, H, W]
        out_tensor = torch.stack(outputs, dim=1).squeeze(2)
        
        # 上采样并归一化
        pred_masks = F.interpolate(out_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        return pred_masks # .sigmoid()

class RefinementModule(nn.Module):
    def __init__(self, in_ch=4, mid_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层：融合 RGB 和 粗糙 Mask
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            
            # 第二层：提取边缘特征
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            
            # 第三层：输出残差 (Residual)
            nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1, bias=True) 
        )

    def forward(self, img, coarse_logits):
        # img: [B, 3, H, W] 归一化后的原图
        # coarse_mask: [B, 1, H, W] 上采样后的粗糙 Mask
        coarse_prob = torch.sigmoid(coarse_logits)
        x = torch.cat([img, coarse_prob], dim=1)
        delta = self.net(x)
        return coarse_logits + delta  # 原始结果 + 修正值


class BetterIoUHead(nn.Module):
    def __init__(self, feature_dim=128, num_outputs=3):
        """
        Args:
            feature_dim: 输入特征图 x 的通道数 (例如 128)
            num_outputs: 需要预测的 IoU 数量 (对应 Mask 的数量)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_outputs = num_outputs

        # ----------------------------------------------------------------
        # 1. 融合层: 将 (Image Feature) + (Mask) 融合
        # 输入通道 = feature_dim + 1 (因为 concat 了一个 mask 通道)
        # ----------------------------------------------------------------
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim + 1, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # 2. 降维/特征提取层 (Downscaling / Feature Extraction)
        # 进一步压缩空间特征，为 MLP 做准备
        # ----------------------------------------------------------------
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1), # H,W -> H/2, W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1), # H/2, W/2 -> H/4, W/4
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # 3. 评分 MLP (Scoring MLP)
        # ----------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # 每个 Mask 输出一个标量分数
        )

    def forward(self, x, mask_logits):
        """
        Args:
            x: [B, C, H, W]  - 解码器的特征图
            masks: [B, N, H_m, W_m] - 预测出的 N 个 Mask (通常尺寸比 x 大)
        Returns:
            iou_pred: [B, N] - 预测的 IoU 分数
        """
        B, C, H, W = x.shape
        num_masks = mask_logits.shape[1]

        # 1. 对齐尺寸: 将 Mask 下采样到特征图 x 的尺寸
        # align_corners=False 也就是默认的 bilinear
        masks_resized_logits = F.interpolate(mask_logits, size=(H, W), mode='bilinear', align_corners=False)
        masks_prob = torch.sigmoid(masks_resized_logits)


        x_expanded = x.unsqueeze(1).expand(-1, num_masks, -1, -1, -1).reshape(-1, C, H, W)
        masks_flat = masks_prob.reshape(-1, 1, H, W) 

        # 拼接: [B*N, C+1, H, W]
        cat_input = torch.cat([x_expanded, masks_flat], dim=1)

        # 卷积特征提取
        feat = self.fusion_conv(cat_input)
        feat = self.dim_reduce(feat) # 空间尺寸变小

        # 全局池化: [B*N, C, h, w] -> [B*N, C, 1, 1] -> [B*N, C]
        # 同时使用 Max 和 Avg 池化，捕获更多信息
        feat_avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        feat_max = F.adaptive_max_pool2d(feat, 1).flatten(1)
        feat_fused = feat_avg + feat_max

        # MLP 预测
        scores = self.mlp(feat_fused) # [B*N, 1]
        
        # 恢复形状: [B*N, 1] -> [B, N]
        iou_pred = scores.reshape(B, num_masks)
        
        # 关键：IoU 必须在 [0, 1] 之间
        return torch.sigmoid(iou_pred)

    
class SAM2UNeXT(nn.Module):
    def __init__(self, checkpoint_path=None, dinov2_path=None, num_outputs: int = 1) -> None:
        super(SAM2UNeXT, self).__init__()
        self.num_outputs = num_outputs

        # ===== SAM2 Encoder =====    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.sam = model.image_encoder.trunk
        for param in self.sam.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.sam.blocks:
            blocks.append(
                Adapter(block)
            )
        self.sam.blocks = nn.Sequential(
            *blocks
        )

        # ===== DINOv2 Encoder =====
        if dinov2_path:
            self.dino = timm.create_model('vit_large_patch14_dinov2',
                                        features_only=True,
                                        img_size=(448, 448),
                                        pretrained=True,
                                        pretrained_cfg_overlay=dict(file=dinov2_path))
        else:
            self.dino = timm.create_model('vit_large_patch14_dinov2',
                                        features_only=True,
                                        img_size=(448, 448))

        self.align1 = nn.Conv2d(1024, 144, 1)
        self.align2 = nn.Conv2d(1024, 288, 1)
        self.align3 = nn.Conv2d(1024, 576, 1)
        self.align4 = nn.Conv2d(1024, 1152, 1)

        self.reduce1 = nn.Conv2d(144+144, 128, 1)
        self.reduce2 = nn.Conv2d(288+288, 128, 1)
        self.reduce3 = nn.Conv2d(576+576, 128, 1)
        self.reduce4 = nn.Conv2d(1152+1152, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        
        
        # 在 SAM2UNeXT 的 __init__ 中：

        if num_outputs == 1:
            self.head = nn.Conv2d(128, 1, 1)
        else:
            # 替换为带 Token 的新 Head
            # self.mask_head = TokenBasedMaskHead(in_channels=128, num_outputs=num_outputs) 
            self.mask_head = DynamicMaskHeadNoLogVar(in_channels=128, num_outputs=num_outputs) 
            
            
            # IoU Head 保持我上一条回复给你的 BetterIoUHead 不变
            # 记住：IoU Head 负责评判，Mask Head 负责产生差异
            self.iou_head = BetterIoUHead(feature_dim=128, num_outputs=num_outputs)
            # 添加refine模块
            self.refiner = RefinementModule(in_ch=4, mid_ch=16)
    def forward(self, x):
        original_img = x  # 保存原始输入图像用于refine
        x1_s, x2_s, x3_s, x4_s = self.sam(x)
        x_d = self.dino(F.interpolate(x, size=(448, 448), mode='bilinear'))[-1]

        x1_d = F.interpolate(self.align1(x_d), size=x1_s.shape[-2:], mode='bilinear')
        x2_d = F.interpolate(self.align2(x_d), size=x2_s.shape[-2:], mode='bilinear')
        x3_d = F.interpolate(self.align3(x_d), size=x3_s.shape[-2:], mode='bilinear')
        x4_d = F.interpolate(self.align4(x_d), size=x4_s.shape[-2:], mode='bilinear')

        x1, x2, x3, x4 = torch.cat([x1_s,x1_d], dim=1), torch.cat([x2_s,x2_d], dim=1), torch.cat([x3_s,x3_d], dim=1), torch.cat([x4_s,x4_d], dim=1)
        x1, x2, x3, x4 = self.reduce1(x1), self.reduce2(x2), self.reduce3(x3), self.reduce4(x4)
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        # 根据 num_outputs 返回不同格式的输出
        if self.num_outputs == 1:
            out = self.head(x)
            out = F.interpolate(out, scale_factor=2, mode='bilinear')
            return out
        else:
            # 多 mask 输出
            h, w = x.shape[-2:]
            target_size = (h * 2, w * 2)  # 上采样 2 倍
            pred_masks = self.mask_head(x, target_size)  # [B, num_outputs, H*2, W*2]
            pred_iou = self.iou_head(x, pred_masks)  # [B, num_outputs]
            
            # 应用refine结构
            refined_masks_list = []
            for i in range(self.num_outputs):
                single_mask = pred_masks[:, i:i+1, :, :]  # [B, 1, H*2, W*2]
                # 将原始图像上采样到mask的尺寸
                img_resized = F.interpolate(original_img, size=single_mask.shape[-2:], mode='bilinear', align_corners=False)  # [B, 3, H*2, W*2]
                refined_single = self.refiner(img_resized, single_mask)
                refined_masks_list.append(refined_single)
            refined_masks = torch.cat(refined_masks_list, dim=1)
            
            return {
                'pred_masks': refined_masks,
                'pred_iou': pred_iou
            }
