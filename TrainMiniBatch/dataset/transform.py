import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import math

def rotate(img, mask, p=0.5, angle=0, fill_index=255, target_size=518):
    """旋转图像和掩码，并用指定值填充空白区域"""
    if random.random() < p:
        angle = random.randint(-angle, angle)
        # angle = random.choice([-angle, angle])  # 随机选择顺时针或逆时针旋转
        img = img.rotate(angle,expand=True)#,fillcolor=fill_index)# expand=True)#, fillcolor=fill_index)  # 用 fill_index 填充空白区域
        mask = mask.rotate(angle, expand=True,fillcolor=fill_index)#, fillcolor=fill_index)

        img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    return img, mask

def random_pepper(img, N=0.0015):
    img = np.array(img)
    noiseNum = int(N * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        img[randX, randY] = random.randint(0, 1) * 255

    return Image.fromarray(img)


def vflip(img, mask, p=0.5):
    """垂直翻转"""
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def crop2(img, mask, size, ignore_value=255,shadow_ratio = 0):

    def is_valid_mask(cropped_mask_np, orig_valid_pixel_count, shadow_ratio):
        valid_pixels = np.logical_and(cropped_mask_np > 0, cropped_mask_np < 255)
        a1_cnt = np.sum(valid_pixels)
        # print("orig_valid_pixel_count:",orig_valid_pixel_count, flush=True)
        valid_pixel_count = np.sum(valid_pixels)
        # print("valid_pixel_count:",valid_pixel_count, flush=True)
        ratio = (valid_pixel_count + 1e-6) / (orig_valid_pixel_count + 1e-6)
        return ratio > shadow_ratio # or valid_pixel_count > shadow_ratio

    w, h = img.size
    padw = max(0, size - w)
    padh = max(0, size - h)

    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    padded_mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    # 计算原始mask中符合条件的像素数量
    orig_mask_np = np.array(padded_mask)
    orig_valid_pixel_count = np.sum(np.logical_and(orig_mask_np > 0, orig_mask_np < 255))
    # print("kk:",np.sum(orig_mask_np > 0), flush=True)
    # print("orig_valid_pixel_count:",orig_valid_pixel_count, flush=True)

    w_, h_ = img.size

    cnt = 0
    while True:
        x = max(0, random.randint(0, w_ - size))
        y = max(0, random.randint(0, h_ - size))
            
        cropped_mask = padded_mask.crop((x, y, x + size, y + size))
        cropped_mask_np = np.array(cropped_mask)
        
        # 检查裁剪后的mask是否符合要求
        if is_valid_mask(cropped_mask_np, orig_valid_pixel_count, shadow_ratio):
            break
        cnt += 1
        if cnt > 10:
            img = img.resize((size, size), Image.BILINEAR)
            mask = mask.resize((size, size), Image.NEAREST)
            return img, mask

    # 现在我们知道了一个满足条件的裁剪位置，对原图进行同样的裁剪
    cropped_img = img.crop((x, y, x + size, y + size))

    return cropped_img, cropped_mask

    # w, h = img.size
    # padw = size - w if w < size else 0
    # padh = size - h if h < size else 0
    # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    # w, h = img.size
    # x = random.randint(0, w - size)
    # y = random.randint(0, h - size)
    # img = img.crop((x, y, x + size, y + size))
    # mask = mask.crop((x, y, x + size, y + size))

    # # if (img.size[0] %14==0) and (img.size[1] % 14 == 0):
    # #     pass
    # # else:
    # #     ow = math.ceil(img.size[0] / 14) * 14
    # #     oh = math.ceil(img.size[1] / 14) * 14
    # #     img = img.resize((ow, oh), Image.BILINEAR)
    # #     mask = mask.resize((ow, oh), Image.NEAREST)

    # return img, mask

def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    # if (img.size[0] %14==0) and (img.size[1] % 14 == 0):
    #     pass
    # else:
    #     ow = math.ceil(img.size[0] / 14) * 14
    #     oh = math.ceil(img.size[1] / 14) * 14
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    # 14位对齐
    # ow = math.ceil(ow / 14) * 14
    # oh = math.ceil(oh / 14) * 14

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
