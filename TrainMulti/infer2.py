from SAM2UNeXT import SAM2UNeXT
import argparse
import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", type=str, 
    default="/home/notebook/data/group/00group/users/myjyf/MultiSeg/lab3829/ckpt/tmp/epoch_30.pth",
                    help="path to save the predicted masks")
parser.add_argument("--input_path", type=str, default="/home/notebook/data/group/00group/users/myjyf/MultiSeg/lab3829/Data",
                    help="path to input images directory (will search recursively)")
parser.add_argument("--save_path", type=str, default="output",
                    help="path to save the predicted masks")
args = parser.parse_args()
import copy
import cv2
class TestDataset:
    def __init__(self, image_root, size, save_path):
        # 递归查找所有图片文件
        self.images = []
        self.image_relative_paths = []  # 保存相对路径，用于输出时保持目录结构
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        # 递归遍历所有子目录
        for root, dirs, files in os.walk(image_root):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    # 计算相对于image_root的相对路径
                    rel_path = os.path.relpath(full_path, image_root)
                    self.images.append(full_path)
                    self.image_relative_paths.append(rel_path)
        
        self.images = sorted(self.images)
        self.image_relative_paths = sorted(self.image_relative_paths)
        self.save_path = save_path
        self.image_root = image_root
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_ori = copy.deepcopy(image)
        image = self.transform(image).unsqueeze(0)

        # 获取文件名和相对路径
        rel_path = self.image_relative_paths[self.index]
        name = os.path.basename(rel_path)  # 文件名
        rel_dir = os.path.dirname(rel_path)  # 相对目录路径

        self.index += 1
        return image, name, image_ori, rel_dir

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用命令行参数指定的输入路径
input_path = os.path.abspath(args.input_path)
test_loader = TestDataset(input_path, 1024, args.save_path)

model = SAM2UNeXT(num_outputs=3).to(device)

model_dict = torch.load(args.checkpoint,map_location='cuda:0')

new_model_dict ={}
for k,v in model_dict.items():
    new_model_dict[k[17:]] = v
model.load_state_dict(new_model_dict, strict=True)
model.eval()
model.cuda()
import shutil
try:
    shutil.rmtree(args.save_path)
except:
    pass
os.makedirs(args.save_path, exist_ok=True)
import time

def Checkerboard(h, w, size=100): # 棋盘格背景图
    bg = np.ones((h, w, 3), dtype="uint8")
    bg = bg *np.array([ 176., 230., 224.])
    # bg = bg *np.array([ 128., 0., 128.])
    for i in range(0, h, size*2):
        for j in range(0, w, size*2):
            bg[i:i+size, j:j+size] = [128, 0, 128]
            if j + size < w and i + size < h:
                bg[i+size:i+size*2, j+size:j+size*2] = [128, 0, 128]
    return bg

def log_var_to_heatmap(log_var, colormap='jet', vmin=None, vmax=None):
    """
    将log_var转换为热力图
    
    Args:
        log_var: numpy array, shape [H, W]
        colormap: matplotlib colormap名称，默认'jet'
        vmin: 最小值，如果为None则使用log_var的最小值
        vmax: 最大值，如果为None则使用log_var的最大值
        
    Returns:
        heatmap: RGB图像，shape [H, W, 3]，值范围[0, 255]
    """
    # 确保是numpy数组
    if isinstance(log_var, torch.Tensor):
        log_var = log_var.cpu().numpy()
    
    # 获取colormap
    cmap = cm.get_cmap(colormap)
    
    # 归一化到[0, 1]
    if vmin is None:
        vmin = log_var.min()
    if vmax is None:
        vmax = log_var.max()
    
    # 避免除零
    if vmax == vmin:
        normalized = np.zeros_like(log_var)
    else:
        normalized = (log_var - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
    
    # 应用colormap，得到RGBA图像
    heatmap_rgba = cmap(normalized)
    
    # 转换为RGB并缩放到[0, 255]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
    return heatmap_rgb

t1 = 0
for i in range(test_loader.size):
    with torch.no_grad():
        image, name, image_ori, rel_dir = test_loader.load_data()
        # gt = np.asarray(gt, np.float32)
        image = image.to(device)

        torch.cuda.synchronize()
        s = time.time()
        res = model(image)
        torch.cuda.synchronize()
        t1 += time.time() - s

        # 标记是否为多输出
        is_multi_output = isinstance(res, dict)
        
        # 处理多输出情况：保存所有mask并打印IoU值
        if is_multi_output:
            # 多mask输出，保存所有mask的预测结果
            pred_masks = res['pred_masks']  # [B, num_outputs, H, W]
            pred_iou = res['pred_iou']  # [B, num_outputs]
            pred_log_vars = res.get('pred_log_vars', None)  # [B, num_outputs, H, W] 或 None
            
            # 将IoU值移到CPU并转换为numpy
            iou_values = pred_iou.cpu().numpy().squeeze()  # [num_outputs] 或 [B, num_outputs]
            if iou_values.ndim == 0:
                iou_values = iou_values.reshape(1)
            elif iou_values.ndim == 2:
                iou_values = iou_values[0]  # 取第一个batch
            
            # 选择IoU最高的mask索引
            best_idx = pred_iou.argmax(dim=1)  # [B]
            best_idx_val = best_idx[0].item() if best_idx.numel() > 0 else 0
            
            # 打印所有mask的IoU值
            print(f"\n{'='*60}")
            print(f"处理图像: {name}")
            print(f"总共有 {pred_masks.shape[1]} 个mask输出")
            print(f"所有mask的IoU值:")
            for idx in range(pred_masks.shape[1]):
                iou_val = iou_values[idx] if idx < len(iou_values) else 0.0
                marker = " <-- 最优 (已选择)" if idx == best_idx_val else ""
                print(f"  Mask {idx}: IoU = {iou_val:.4f}{marker}")
            print(f"{'='*60}")
            
            # 只保存最优mask的combined图像
            num_outputs = pred_masks.shape[1]
            image_ori_np = np.array(image_ori)
            base_name = os.path.splitext(name)[0]  # 去掉扩展名
            
            # 创建输出目录（保持相对路径结构）
            if rel_dir:
                output_dir = os.path.join(args.save_path, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = args.save_path
            
            # 只处理最优mask
            mask_idx = best_idx_val
            
            # 获取最优mask
            mask = pred_masks[0, mask_idx, :, :]  # [H, W]
            mask = mask.sigmoid().data.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            
            # 调整mask尺寸以匹配原始图像
            mask_resized = np.array(Image.fromarray(mask).resize((image_ori_np.shape[1], image_ori_np.shape[0])))
            
            # 如果有log_var，生成并保存组合图像（原图 + mask + 热力图）
            if pred_log_vars is not None:
                # 计算全局的min和max用于归一化
                log_var_global_min = pred_log_vars.cpu().numpy().min()
                log_var_global_max = pred_log_vars.cpu().numpy().max()
                print(f"Log_Var 范围: [{log_var_global_min:.4f}, {log_var_global_max:.4f}]")
                
                # 获取最优mask对应的log_var
                log_var = pred_log_vars[0, mask_idx, :, :]  # [H, W]
                
                # 调整log_var尺寸以匹配原始图像（使用torch的interpolate）
                log_var_tensor = log_var.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                log_var_resized_tensor = torch.nn.functional.interpolate(
                    log_var_tensor,
                    size=(image_ori_np.shape[0], image_ori_np.shape[1]),
                    mode='bilinear',
                    align_corners=False
                )
                log_var_resized = log_var_resized_tensor.squeeze().cpu().numpy()
                
                # 转换为热力图
                heatmap = log_var_to_heatmap(
                    log_var_resized, 
                    colormap='jet',
                    vmin=log_var_global_min,
                    vmax=log_var_global_max
                )
                
                # 保存组合图像：原图 + mask + 热力图
                # 确保所有图像高度一致
                h = image_ori_np.shape[0]
                mask_3channel = np.stack([mask_resized] * 3, axis=-1) if mask_resized.ndim == 2 else mask_resized
                combined_three = np.concatenate([image_ori_np, mask_3channel, heatmap], axis=1)
                combined_name = f"{base_name}_mask{mask_idx}_combined_best.png"
                combined_path = os.path.join(output_dir, combined_name)
                imageio.imsave(combined_path, combined_three)
                print(f"  已保存组合图(原图+mask+热力图): {combined_path}")
            else:
                # 如果没有log_var，只保存原图+mask的组合
                mask_3channel = np.stack([mask_resized] * 3, axis=-1) if mask_resized.ndim == 2 else mask_resized
                combined_two = np.concatenate([image_ori_np, mask_3channel], axis=1)
                combined_name = f"{base_name}_mask{mask_idx}_combined_best.png"
                combined_path = os.path.join(output_dir, combined_name)
                imageio.imsave(combined_path, combined_two)
                print(f"  已保存组合图(原图+mask): {combined_path}")
            
            # 使用最优mask作为主要输出（保持向后兼容）
            batch_indices = torch.arange(pred_masks.shape[0], device=pred_masks.device)
            res = pred_masks[batch_indices, best_idx, :, :]  # [B, H, W]
            res = res.unsqueeze(1)  # [B, 1, H, W]
        else:
            # 单输出情况，正常处理
            pass
        
        # 单输出情况不保存（只保留多输出情况下的combine_best）
        if not is_multi_output:
            pass
print(f"avg time:{t1 / test_loader.size}")
