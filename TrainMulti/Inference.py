from SAM2UNeXT import SAM2UNeXT
import argparse
import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from PIL import Image


import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t',       type=str,            default='green')
    parser.add_argument('--gpu', '-g',        action='store_true', default=True)
    parser.add_argument('--jit', '-j',        action='store_true', default=False)
    parser.add_argument('--verbose', '-v',    action='store_true', default=False)
    return parser.parse_args()

def get_format(source):
    img_count = len([i for i in source if i.lower().endswith(('.jpg', '.png', '.jpeg'))])
    vid_count = len([i for i in source if i.lower().endswith(('.mp4', '.avi', '.mov' ))])
    
    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''

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

def inference(args):

    model = SAM2UNeXT().to(device)

    model_dict = torch.load(args.checkpoint,map_location='cuda:0')

    new_model_dict ={}
    for k,v in model_dict.items():
        new_model_dict[k[17:]] = v
    model.load_state_dict(new_model_dict, strict=True)
    model.eval()
    model.cuda()
    model = model.half()
        

    _format = 'Image'

    ipath = "/home/notebook/code/personal/80410839/sources/JYF/DATASETS/VideoMatting"
    ipath = "/home/notebook/code/personal/80410839/Matting/os15-image-matting-training/Test0"
    ipath = "/home/notebook/code/personal/80410839/Video/os16-video-matting-training/Test0"
    ipath = "/home/notebook/code/personal/80410839/Video/Dataset/Tmp4/花壁纸调整7.21"
    ipath = "/home/notebook/code/personal/80410839/Video/os16-video-matting-training/Test"
    ipath = "/home/notebook/code/personal/80410839/Video/os16-video-matting-training/Test1/破框Live拼图极限场景或元素测试图片"
    ipath = "/home/notebook/code/personal/80410839/sources/JYF/DATASETS/0818_dataset/背景正例/云2_原图"
    ipath = "/home/notebook/code/personal/80410839/Matting/LAB/BirefNet_SOD_Mat/"
    ipath = "/home/notebook/code/personal/80410839/sources/JYF/"

    paths = os.listdir(ipath)
    from os.path import abspath, dirname, join as pjoin

    inc_dirs = [
        "动物",
        "人像",
        "山景",
        "平板",
        "折叠",
        "直板",
        "奔跑跳跃",
        "宠物动",
        "剪影 夕阳或逆光或影子",
        "类人玩偶",
        "连接物",
        "masks",
        "YUY",
        "生日"
    ]

    # save_prefix = "灵感壁纸20250610_"
    save_prefix = "灵感壁纸20250717_"

    idx = 1

    shold_rename = False
    args.type = 'green'

    # 以目录名+idx的形式结尾
    for path in paths:
    # if 1:
        # path = ""
        if not os.path.isdir(pjoin(ipath, path)): continue
        if path not in inc_dirs : continue
        # 删除result文件夹
        os.system(f"rm -rf {pjoin(ipath, path, 'result')}")
        imgs = os.listdir(pjoin(ipath, path))
        imgs = [pjoin(ipath, path, img) for img in imgs]
        

        # # 先rename
        # for mp4f in imgs:
        #     base = os.path.basename(mp4f)
        #     prefix = base[base.find("."):]
        #     os.rename(mp4f, pjoin(ipath, path, save_prefix+path+f"{idx:0>6}.jpg"))
        #     idx += 1

        # imgs = os.listdir(pjoin(ipath, path))
        # imgs = [pjoin(ipath, path, img) for img in imgs]

        if shold_rename:
            # 分离png、jpg和mp4
            mp4_list = [i for i in imgs if i.endswith(("mp4", 'mov'))]
            img_list = [i for i in imgs if i.endswith(("jpg","png","JPG","PNG","jpeg","JPEG","BMP",'bmp',"webp")) and "mask" not in i]

            # 先rename
            for mp4f in mp4_list:
                prefix = mp4f[mp4f.rfind("."):]
                os.rename(pjoin(ipath, path, mp4f), pjoin(ipath, path, save_prefix+path+f"{idx:0>6}"+prefix))
                idx += 1

            for jpg_ in img_list:
                # prefix = jpg[jpg.find("."):]
                if jpg_.endswith("jpg"):
                    os.rename(pjoin(ipath, path, jpg_), pjoin(ipath, path, save_prefix+path+f"{idx:0>6}.jpg"))
                else:
                    Image.open(pjoin(ipath, path, jpg_)).convert("RGB").save(pjoin(ipath, path, save_prefix+path+f"{idx:0>6}.jpg"))
                    os.remove(pjoin(ipath, path, jpg_))
                idx += 1

        imgs = os.listdir(pjoin(ipath, path))
        imgs = [pjoin(ipath, path, img) for img in imgs]
        # print(imgs)
        mp4_list = [i for i in imgs if i.endswith(("mp4", 'mov'))]
        mp4_list = []
        img_list = [i for i in imgs if i.endswith(("jpg","png","JPG","PNG","jpeg","JPEG","BMP",'bmp',"webp")) ]

        # 先处理图片
        sample_list = eval('CustomLoader')(img_list, opt.Test.Dataset.transforms)
        samples = tqdm.tqdm(sample_list, desc='Inference', total=len(
                sample_list), position=0, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')

        for sample in samples:
            sample = to_cuda(sample)
            with torch.no_grad():
                out = model(sample)
                pred = to_numpy(out['pred'].type(torch.float32), sample['shape'])
            img = np.array(sample['original'])

            if args.type == 'map':
                img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
            elif args.type == 'rgba':
                r, g, b = cv2.split(img)
                pred = (pred * 255).astype(np.uint8)
                img = cv2.merge([r, g, b, pred])
            elif args.type == 'green':
                h, w = pred.shape[:2]
                bg = Checkerboard(h, w)
                img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])
            elif args.type == 'blur':
                img = img * pred[..., np.newaxis] + cv2.GaussianBlur(img, (0, 0), 15) * (1 - pred[..., np.newaxis])
            elif args.type == 'overlay':
                bg = (np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155] + img) // 2
                img = bg * pred[..., np.newaxis] + img * (1 - pred[..., np.newaxis])
                border = cv2.Canny(((pred > .5) * 255).astype(np.uint8), 50, 100)
                img[border != 0] = [120, 255, 155]
            elif args.type.lower().endswith(('.jpg', '.jpeg', '.png')):
                if background is None:
                    background = cv2.cvtColor(cv2.imread(args.type), cv2.COLOR_BGR2RGB)
                    background = cv2.resize(background, img.shape[:2][::-1])
                img = img * pred[..., np.newaxis] + background * (1 - pred[..., np.newaxis])
                    
                    
            img = img.astype(np.uint8)
            os.makedirs(os.path.join(ipath, path,"result"),exist_ok=True)
            # np.concatenate([np.array(sample['original']), img],axis=1)
            if args.type == 'rgba':
                Image.fromarray(img).save(os.path.join(ipath, path,"result", sample['name'] + '.png'))
            else:
                Image.fromarray(np.concatenate([np.array(sample['original']), img],axis=1)).save(os.path.join(ipath, path,"result", sample['name'] + '.jpg'))
        # 再处理视频，转成Image类型的序列
        for fmp4 in mp4_list:
            # 解析mp4，把mp4解码成图片集（每个视频随机取10张），先放到tmp文件夹内，然后放到result下的同名文件夹内
            cap = cv2.VideoCapture(fmp4)
            base_name = os.path.basename(fmp4)
            base_dir = os.path.dirname(fmp4)
            total_frames = max(20, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            # 打印
            prefix = base_name[:base_name.find(".")]

            os.makedirs(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"), prefix), exist_ok=True) # , "result"

            cur_frame = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if shold_rename:
                    jg = 10
                else:
                    jg = 1
                if shold_rename:
                    if (cur_frame % (total_frames // 10)) == 0:
                        # 保存这一帧
                        Image.fromarray(frame[...,::-1]).convert("RGB").save(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"), prefix,f"{prefix}_{cur_frame}.jpg"))
                else:
                    Image.fromarray(frame[...,::-1]).convert("RGB").save(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"), prefix,f"{prefix}_{cur_frame}.jpg"))
                cur_frame += 1
                    
            # 保存后再把结果保存到result里面
            img_list = [os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"), prefix, i) for i in os.listdir(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"), prefix))
                        if os.path.isfile(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"), prefix, i))]
            print(img_list)
            sample_list = eval('CustomLoader')(img_list, opt.Test.Dataset.transforms)
            samples = tqdm.tqdm(sample_list, desc='Inference', total=len(
                    sample_list), position=0, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
            

            for sample in samples:
                sample = to_cuda(sample)
                with torch.no_grad():
                    out = model(sample)
                    pred = to_numpy(out['pred'].type(torch.float32), sample['shape'])
                img = np.array(sample['original'])

                if args.type == 'map':
                    img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
                elif args.type == 'rgba':
                    r, g, b = cv2.split(img)
                    pred = (pred * 255).astype(np.uint8)
                    img = cv2.merge([r, g, b, pred])
                elif args.type == 'green':
                    h, w = pred.shape[:2]
                    bg = Checkerboard(h, w)
                    img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])
                elif args.type == 'blur':
                    img = img * pred[..., np.newaxis] + cv2.GaussianBlur(img, (0, 0), 15) * (1 - pred[..., np.newaxis])
                elif args.type == 'overlay':
                    bg = (np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155] + img) // 2
                    img = bg * pred[..., np.newaxis] + img * (1 - pred[..., np.newaxis])
                    border = cv2.Canny(((pred > .5) * 255).astype(np.uint8), 50, 100)
                    img[border != 0] = [120, 255, 155]
                elif args.type.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if background is None:
                        background = cv2.cvtColor(cv2.imread(args.type), cv2.COLOR_BGR2RGB)
                        background = cv2.resize(background, imfvg.shape[:2][::-1])
                    img = img * pred[..., np.newaxis] + background * (1 - pred[..., np.newaxis])
                        
                        
                img = img.astype(np.uint8)
                os.makedirs(os.path.join(ipath, path,"result"),exist_ok=True)
                np.concatenate([np.array(sample['original']), img],axis=1)
                os.makedirs(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"),"result", prefix), exist_ok=True)
                Image.fromarray(np.concatenate([np.array(sample['original']), img],axis=1)).save(os.path.join(ipath, os.path.dirname(fmp4)[len(ipath):].strip("/"),"result", prefix, sample['name'] + '.jpg'))

if __name__ == "__main__":
    args = _args()
    inference(args)
