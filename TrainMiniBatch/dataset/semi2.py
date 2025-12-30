from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random


import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import cv2

from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def filter_small_contours(mask, thd):
    # 转换为OpenCV二值格式（0和255）[2][3]
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 8联通轮廓检测（RETR_TREE获取层级关系）[2][4]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建待清除区域的掩膜
    remove_mask = np.zeros_like(binary)
    
    valid_contours = []
    # 遍历所有轮廓（含层级信息）[3]
    for i in range(len(contours)):
        # 跳过父轮廓已被清除的子轮廓[3]
        if hierarchy[0][i][3] != -1 and hierarchy[0][i][3] not in valid_contours:
            continue
            
        area = cv2.contourArea(contours[i])
        if area < thd:
            # 在掩膜上标记待清除区域（填充轮廓内部）[3]
            cv2.drawContours(remove_mask, [contours[i]], -1, 255, -1)
            # 记录有效父轮廓[3]
            valid_contours.append(i)
    
    # 通过位运算清除小轮廓区域[3]
    return remove_mask


class SemiDataset(Dataset):
    def __init__(self, root, mode, size=None, id_path=None, nsample=None,un_labelnum=None,seed=42,ratio_list=None):
        self.root = root
        self.mode = mode
        self.size = size
        self.ratio_list = ratio_list
        self.CLASSES = 1

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
                
                if un_labelnum:
                    current_seed = random.getstate()
                    random.seed(seed)
                    random.shuffle(self.ids)
                    random.setstate(current_seed)
                    self.ids = self.ids[:un_labelnum]
                # 选取前把ids给弄成我们想要的，哈哈哈
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                tmp_ids = []
                for id in self.ids:
                    mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert("L")
                    if mask.histogram()[1] >= self.ratio_list[3]:
                        tmp_ids.append(id)
                self.ids = tmp_ids
                print("cur_train_len:",len(self.ids))
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(os.path.join(root, "val0320.txt"), "r") as f:
            # with open("/home/notebook/code/personal/80410839/Dataset/SPLIT/val.txt", 'r') as f:
                self.ids = f.read().splitlines()
            # with open('splits/%s/val.txt' % name, 'r') as f:
            #     self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        # with open("tmp.txt","a+") as f:
        #     f.write(f"item-{id}\n")
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert("L")#Image.fromarray(np.array())) 
        
        # mask /= 255
        if self.mode == 'val':
            # resize到确定大小
            # ow = math.ceil(img.size[0] / 14) * 14
            # oh = math.ceil(img.size[1] / 14) * 14
            img = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)

            img, mask = normalize(img, mask)
            return img, mask, id


        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop2(img, mask, self.size, ignore_value, self.ratio_list[0])

        img, mask = rotate(img, mask, p=0.8, angle=15, fill_index=ignore_value,target_size = self.size)

        img, mask = hflip(img, mask, p=0.5)
        img, mask = vflip(img, mask, p=0.5)

        img = random_pepper(img)
        # 在这里控制mask膨胀，以及小目标过滤（先膨胀再过滤）
        if self.mode == 'train_l':
            # 直接把无效像素清空
            # 先保存无效像素
            mask_invaild = np.array(mask)
            mask_invaild[mask_invaild==ignore_value] = 0
            
            mask_cur = copy.deepcopy(mask_invaild)

            # 这就比较复杂了，分成多步
            for i in range(1, 1 + self.CLASSES):
                mask_ori = copy.deepcopy(mask_cur)
                mask_ori[mask_ori!=i]=0
                mask_ori[mask_ori==i]=255

                # mask_ori = cv2.dilate(mask_ori, np.ones((3,3)), iterations = self.ratio_list[4])
                # 这个是对应元素从255变成0
                mask2 = filter_small_contours(mask_ori,self.ratio_list[3]-1)

                mask_cur[mask2] = 0

            # mask_cur[mask_invaild==255]=255
            mask_cur[mask_cur>self.CLASSES] = 0
            # mask_cur[mask_cur==1] = 3
            # mask_cur[mask_cur==2] = 1
            # mask_cur[mask_cur==3] = 2
            mask = Image.fromarray(mask_cur)
            # img.save(f"./tmp2/img{item}.png")
            img = transforms.ColorJitter(self.ratio_list[2],self.ratio_list[2],self.ratio_list[2],self.ratio_list[2]/2.)(img)

            # mask_cur_show = copy.deepcopy(mask_cur)
            # if np.sum(mask_cur_show==1) > 0:
            #     mask_cur_show[mask_cur_show==1] = 255
            #     img.save(f"./tmp/img{item}.png")
            #     Image.fromarray(mask_cur_show).save(f"./tmp/gt{item}.png")
            #     mask_cur_show[mask_cur_show==255] = 0
            
            # if np.sum(mask_cur_show==2) > 0:
            #     mask_cur_show[mask_cur_show==2] = 255
            #     img.save(f"./tmp2/img{item}.png")
            #     Image.fromarray(mask_cur_show).save(f"./tmp2/gt{item}.png")
            #     mask_cur_show[mask_cur_show==255] = 0
            return normalize(img, mask)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(self.ratio_list[2]*2,self.ratio_list[2]*2,self.ratio_list[2]*2,self.ratio_list[2])(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(self.ratio_list[2]*2,self.ratio_list[2]*2,self.ratio_list[2]*2,self.ratio_list[2])(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255 # ?

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        # 在这儿控制数量
        return len(self.ids) # if len(self.ids) < 200 else 200
