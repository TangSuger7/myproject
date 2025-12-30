from SAM2UNeXT import SAM2UNeXT
import argparse
import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint", type=str, 
    default="/home/notebook/data/group/00group/users/myjyf/MultiSeg/lab3829/ckpt/tmp/epoch_30.pth",
                    help="path to save the predicted masks")
parser.add_argument("--input_path", type=str, default="/home/notebook/data/group/00group/users/myjyf/Dataset/MYMattingADD/TMP",
                    help="path to input images directory (will search recursively)")
parser.add_argument("--save_path", type=str, default="output",
                    help="path to save the predicted masks")
parser.add_argument("--output_mode", type=str, default="composite",
                    choices=["default", "composite"],
                    help="输出模式: 'default'=默认模式, 'composite'=拼接模式(原图+绿幕+mask)")
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

def apply_mask_to_greenscreen(original, mask, green_color=(127, 255, 0)):
    """
    将mask应用到原图，生成绿幕背景的主体图片
    mask的白色区域保留原图，黑色区域替换为绿色背景
    
    Args:
        original: 原图numpy数组 (H, W, 3)
        mask: 灰度mask numpy数组 (H, W) 值范围0-255
        green_color: RGB颜色值，默认为(127, 255, 0)
    
    Returns:
        numpy数组: 绿幕背景的主体图片 (H, W, 3)
    """
    # 归一化mask (0-1)
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 创建结果数组
    result = np.zeros_like(original, dtype=np.float32)
    
    # 混合：mask值 * 原图 + (1 - mask值) * 绿色背景
    green_color_array = np.array(green_color, dtype=np.float32)
    
    for c in range(3):
        result[:, :, c] = (
            mask_normalized * original[:, :, c] + 
            (1 - mask_normalized) * green_color_array[c]
        )
    
    return result.astype(np.uint8)

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
            
            # 保存所有mask的预测结果
            num_outputs = pred_masks.shape[1]
            image_ori_np = np.array(image_ori)
            base_name = os.path.splitext(name)[0]  # 去掉扩展名
            
            # 创建输出目录（保持相对路径结构）
            if rel_dir:
                output_dir = os.path.join(args.save_path, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = args.save_path
            
            # 获取最优mask
            best_mask = pred_masks[0, best_idx_val, :, :]  # [H, W]
            best_mask = best_mask.sigmoid().data.cpu().numpy()
            best_mask = (best_mask * 255).astype(np.uint8)
            # 调整mask尺寸以匹配原始图像
            best_mask_resized = np.array(Image.fromarray(best_mask).resize((image_ori_np.shape[1], image_ori_np.shape[0])))
            
            # 如果是composite模式，生成拼接结果
            if args.output_mode == "composite":
                # 1. 原图
                original_img = image_ori_np.copy()
                
                # 2. 原图+mask作为透明度的绿幕背景图
                greenscreen_img = apply_mask_to_greenscreen(image_ori_np, best_mask_resized, green_color=(127, 255, 0))
                
                # 3. mask灰度图（转换为RGB以便拼接）
                mask_gray_rgb = np.stack([best_mask_resized] * 3, axis=-1)
                
                # 4. 水平拼接三张图
                composite_result = np.concatenate([original_img, greenscreen_img, mask_gray_rgb], axis=1)
                
                # 保存拼接结果
                save_name = f"{base_name}_composite_best.png"
                save_path_full = os.path.join(output_dir, save_name)
                imageio.imsave(save_path_full, composite_result)
                print(f"  已保存拼接结果: {save_path_full}")
            
            # 保存所有mask的预测结果（默认模式）
            if args.output_mode == "default":
                for mask_idx in range(num_outputs):
                    # 获取当前mask
                    mask = pred_masks[0, mask_idx, :, :]  # [H, W]
                    mask = mask.sigmoid().data.cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    
                    # 调整mask尺寸以匹配原始图像
                    mask_resized = np.array(Image.fromarray(mask).resize((image_ori_np.shape[1], image_ori_np.shape[0])))
                    
                    # 拼接图像和mask
                    res_combined = np.concatenate([image_ori_np, mask_resized[...,None]], axis=-1)
                    
                    # 保存文件，文件名包含mask索引和IoU值
                    iou_val = iou_values[mask_idx] if mask_idx < len(iou_values) else 0.0
                    suffix = "_best" if mask_idx == best_idx_val else ""
                    save_name = f"{base_name}_mask{mask_idx}_iou{iou_val:.4f}{suffix}.png"
                    save_path_full = os.path.join(output_dir, save_name)
                    imageio.imsave(save_path_full, res_combined)
                    print(f"  已保存: {save_path_full}")
            
            # 使用最优mask作为主要输出（保持向后兼容）
            batch_indices = torch.arange(pred_masks.shape[0], device=pred_masks.device)
            res = pred_masks[batch_indices, best_idx, :, :]  # [B, H, W]
            res = res.unsqueeze(1)  # [B, 1, H, W]
        else:
            # 单输出情况，正常处理
            pass
        
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)

        # 保存原始图像尺寸的numpy数组（用于composite模式）
        image_ori_np_single = np.array(image_ori)
        
        # 调整mask尺寸以匹配原始图像
        mask_resized_single = np.array(Image.fromarray(res).resize((image_ori_np_single.shape[1], image_ori_np_single.shape[0])))
        
        image_ori_resized = np.array(image_ori.resize(res.shape))
        h,w = image_ori_resized.shape[:2][::-1]
        # bg = Checkerboard(h,w)
        # res2 = (image_ori_resized * res[...,None] + bg * (1- res[...,None])).astype(np.uint8)

        res_combined = np.concatenate([image_ori_resized, res[...,None]],axis=-1)
        # res = (image_ori_resized * res[...,None]).astype(np.uint8)
        
        # np.stack([res] * 3, axis=-1)
        # res = cv2.hconcat([image_ori_resized, res2])
        # If you want to binarize the prediction results, please uncomment the following three lines. 
        # Note that this action will affect the calculation of evaluation metrics.
        # lambda = 0.5
        # res[res >= int(255 * lambda)] = 255
        # res[res < int(255 * lambda)] = 0
        
        # 只在单输出情况下保存（多输出已经在上面保存了所有mask）
        if not is_multi_output:
            # 创建输出目录（保持相对路径结构）
            if rel_dir:
                output_dir = os.path.join(args.save_path, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = args.save_path
            
            base_name = os.path.splitext(name)[0]
            
            # 如果是composite模式，生成拼接结果
            if args.output_mode == "composite":
                # 1. 原图
                original_img = image_ori_np_single.copy()
                
                # 2. 原图+mask作为透明度的绿幕背景图
                greenscreen_img = apply_mask_to_greenscreen(image_ori_np_single, mask_resized_single, green_color=(127, 255, 0))
                
                # 3. mask灰度图（转换为RGB以便拼接）
                mask_gray_rgb = np.stack([mask_resized_single] * 3, axis=-1)
                
                # 4. 水平拼接三张图
                composite_result = np.concatenate([original_img, greenscreen_img, mask_gray_rgb], axis=1)
                
                # 保存拼接结果
                save_path_full = os.path.join(output_dir, base_name + "_composite.png")
                print("Saving " + save_path_full)
                imageio.imsave(save_path_full, composite_result)
            else:
                # 默认模式：保存原图+mask拼接
                save_path_full = os.path.join(output_dir, base_name + ".png")
                print("Saving " + save_path_full)
                imageio.imsave(save_path_full, res_combined)
print(f"avg time:{t1 / test_loader.size}")
