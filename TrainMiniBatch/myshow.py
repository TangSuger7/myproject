import argparse
from copy import deepcopy
import logging
import os
import pprint
from dataset.transform import *

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
# from util.dist_helper import setup_distributed

os.environ["LOCAL_RANK"] = '0'
parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str,default='configs/coco.yaml')
parser.add_argument('--labeled-id-path', type=str, default='/home/notebook/code/personal/80410839/Dataset/SPLIT/labeled.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='/home/notebook/code/personal/80410839/Dataset/SPLIT/unlabeled.txt')
parser.add_argument('--save-path', type=str, default='exp/cityscapes/unimatch_v2/dinov2_small/366')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

import torch.distributed as dist
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

Clss = {
    0: (0, 255, 0),
    1: (255, 255, 0),
    # 0: (128, 64,128),
    # 1: (244, 35,232),
    2: ( 70, 70, 70),
    3: (102,102,156),
    4: (190,153,153),
    5: (153,153,153),
    6: (250,170, 30),
    7: (220,220,  0),
    8: (107,142, 35),
    9: (152,251,152),
    10: ( 70,130,180),
    11: (220, 20, 60),
    12: (255,  0,  0),
    13: (  0,  0,142),
    14: (  0,  0, 70),
    15: (  0, 60,100),
    16: (  0, 80,100),
    17: ( 0, 0,230),
    18: (119, 11, 32)

}

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # # rank, world_size = setup_distributed(port=args.port)
    # rank = 0
    # world_size = 1
    # dist.init_process_group(
    #     backend="nccl",
    #     world_size=world_size,
    #     rank=rank,
    # )

    # if rank == 0:
    #     all_args = {**cfg, **vars(args), 'ngpus': world_size}
    #     logger.info('{}\n'.format(pprint.pformat(all_args)))
        
    #     writer = SummaryWriter(args.save_path)
        
    #     os.makedirs(args.save_path, exist_ok=True)

    # cudnn.enabled = True
    # cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    nclass = 2
    model = DPT(encoder_size="base", features=128, out_channels = [96, 192, 384, 768], nclass = nclass)
    model.cuda()
    state_dict = torch.load("/home/notebook/code/personal/80410839/Uni/Test3/EXP/0/epoch6_26.02.pth")
    out_path = "show2"
    threshold_for_result = 0.0001
    # state_dict = torch.load('/home/notebook/code/personal/80410839/LAB/UN3/exp/coco/unimatch_v2/dinov2_small/366/best.pth')
    # state_dict = torch.load('/home/notebook/code/personal/80410839/LAB/UNI_TEST/exp/coco2/unimatch_v2/dinov2_small/366/latest.pth')

    new_state_dict = {}
    for k, v in state_dict['model'].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)


    model.eval()

    PPP = "/home/notebook/code/personal/80410839/Dataset/BDD"
    PPP = "/home/notebook/code/personal/80410839/Shadow4"
    # PPP = "/home/notebook/code/personal/80410839/ADD_MY"
    # PPP = "/home/notebook/code/personal/80410839/S2"

    imgs = os.listdir(PPP)
    from os.path import join as pjoin
    import numpy as np
    import cv2
    from pathlib import Path
    from PIL import Image
    import math
    import tqdm
    import shutil
    try:
        shutil.rmtree(out_path)
    except:
        pass

    cnt = 0
    time_sum = 0
    import time
    import torch.nn.functional as F
    for img in tqdm.tqdm(imgs):
        if not( img.endswith("png") or img.endswith("jpg") or img.endswith("jpeg")):
            continue
        # if not ("webp" in img  or "water" in img): continue

        imgt_ = Image.open(pjoin(PPP, img)).convert("RGB")

        with torch.no_grad():
            A = 0
            if A == 0:
                img_ = imgt_.resize((cfg['crop_size'], cfg['crop_size']), Image.BILINEAR)

                img_2 = normalize(img_).cuda()
                
                time_start = time.time()
                out = model(img_2.unsqueeze(0))
                torch.cuda.synchronize()
                time_end = time.time()
                time_sum += time_end - time_start
                cnt += 1
                # out[:,0,:,:] *= threshold_for_result
                # out[:,1,:,:] /= max(threshold_for_result, 0.00001)
                print(out.shape)
                # out[:,2,...] *= 100
                pred = out.argmax(dim=1)
                # mm1 = torch.max(out, dim=1, keepdim=True)[0]
                # mm2 = out.argmax(dim=1) * mm1
                # pred = mm2.squeeze(0) / torch.max(mm2)
                # pred = out[:,1,:,:]  - out[:,0,:,:]
                # pred = (pred-torch.min(pred)) / torch.max(pred)
                a = 1
            elif A == 1:
                # 这种方法实际上还是会输入不同分辨率的图像，这是不科学的
                grid = 518
                ow = math.ceil(img_.size[0] / 14) * 14
                oh = math.ceil(img_.size[1] / 14) * 14
                # if ow > 2000 or oh > 2000:
                #     ow , oh = 518, 518
                #     print("error")
                img_ = img_.resize((ow, oh), Image.BILINEAR)
                # img_2 = normalize(img_).cuda()

                # time_start = time.time()
                # out = model(img_2.unsqueeze(0))
                # torch.cuda.synchronize()
                # time_end = time.time()
                # time_sum += time_end - time_start
                # cnt += 1

                # pred = out.argmax(dim=1)

                img_2 = normalize(img_).cuda()
                img_2 = img_2.unsqueeze(0)

                b, _, h, w = img_2.shape
                final = torch.zeros(b, 2, h, w).cuda()
                row = 0
                print(f"w:{w},h:{h}")
                while row < h:
                    col = 0
                    while col < w:
                        img_t = img_2[:, :, row: row + grid, col: col + grid]
                        print(f"x1:{col}, x2:{col+grid}, y1:{row}, y2:{row+grid}")
                        # if list(img_2[:, :, row: row + grid, col: col + grid].shape) != [1,3,518,518]:
                        #     img_t = F.interpolate(img_2, (518, 518), mode='bilinear', align_corners=True)
                        # else:
                            
                        # print(img_t.shape)
                        time_start = time.time()
                        pred = model(img_t)
                        torch.cuda.synchronize()
                        time_end = time.time()
                        time_sum += time_end - time_start
                        cnt += 1
                        # if list(img_2[:, :, row: row + grid, col: col + grid].shape) != [1,3,518,518]:
                        #     pred = F.interpolate(pred, img_2[:, :, row: row + grid, col: col + grid].shape[2:], mode='bilinear', align_corners=True)
                        #     # print(pred.shape)

                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                pred = final
                pred = pred.argmax(dim=1)
            elif A == 2:
                resolution = 518
                threshold_for_result = 0.5
                filter_threshold = 0.1
                w, h = img_.size
                crop_num = 4

                transform = transforms.Compose([
                    transforms.Resize([resolution,resolution]),
                    transforms.ToTensor()
                ])

                # 先resize成518*2,
                img_ = img_.resize((resolution*2, resolution*2), Image.BILINEAR)
                img2_ = img_.resize((resolution, resolution), Image.BILINEAR)
                # 然后有一个原图大小
                img_p = normalize(img_).cuda()
                img2_p = normalize(img2_).cuda()

                img_sep = torch.zeros((crop_num + 1, 3, resolution, resolution)).cuda()

                img_sep[-1] = img2_p.clone()
                for i in range(2):
                    for j in range(2):
                        img_sep[i*2+j] = img_p[:, \
                                            resolution*i:(i+1)*resolution, \
                                            resolution*j:(j+1)*resolution]
                length = resolution // 2
                res_ini = model(img_sep)
                res_global = res_ini[-1].argmax(dim=0).unsqueeze(0).unsqueeze(0).clone()
                res_ini = transforms.Resize(length)(res_ini)

                res_local = torch.zeros((1, 1, resolution, resolution)).cuda()
                for i in range(2):
                    for j in range(2):
                        res_local[:, :, i * length:(i + 1) * length, j * length:(j + 1) * length] = \
                                res_ini[i * 2 + j:i * 2 + j + 1].argmax(dim=1)
                # res_local = res_local # * (res_global > filter_threshold)
                res_global[res_global>0]=0
                pred = torch.maximum(res_global, res_local)

                pred = transforms.Resize((h, w))(pred.squeeze(0))
                pred = (pred > threshold_for_result).float()


                


        print(pred.max())

        img2 = np.array(imgt_)

        pred = transforms.Resize(imgt_.size[::-1], Image.NEAREST)(pred)
        
        m1 = pred.permute(1,2,0).repeat(1,1,3).cpu()

        # img2 = np.array(img_)
        # img2 = cv2.resize(img2, m1.shape[:2][::-1])
        alpha = 0.5
        gt = img2
        for i in range(1, nclass):
            color = np.array(Clss[i-1]).reshape((1,1,3)).astype(np.uint8)
            color = color.repeat(img2.shape[0], 0)
            color = color.repeat(img2.shape[1], 1)
            gt = np.where(m1 == i, gt*(1-alpha)+color*alpha, gt)

        img_res = np.concatenate(((m1*255)//(nclass-1), gt),axis=1).astype('uint8') # img2,
        # 直接保存mask
        # img_res = (m1.cpu()*255).numpy().astype("uint8")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_res).save(pjoin(out_path, img))

    print("time_all:", time_sum)
    print("avg_time:", time_sum / cnt)
# Image.fromarray((out2.squeeze(0).cpu().detach().numpy()[0,...]*255).astype(np.uint8)).save("hahah.png")
if __name__ == "__main__":
    main()