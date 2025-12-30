import os
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from SAM2UNeXT import SAM2UNeXT
torch.backends.cudnn.enabled = False

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
#from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--dist', default=True, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', default=False, help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
args = parser.parse_args()

config = Config()

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5)],
    )
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.local_process_index
    else:
        device = config.device

if config.rand_seed:
    set_seed(config.rand_seed + device)

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)
config.batch_size = 1

from dataset import custom_collate_fn

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )



# from torch.utils.data import DataLoader
# from torch._utils import _flatten_dense_tensors
# from torch.utils.data._utils.collate import default_collate

# def custom_collate_fn(batch):
#     """
#     自定义 collate 函数，保留字符串字段不进行 tensor 转换
#     假设每个样本是 (video_tensor, label, video_path)
#     """
#     elem = batch[0]
#     # 假设样本是一个元组或字典，我们只对非字符串字段使用 default_collate
#     if isinstance(elem, tuple):
#         # 分离可 collate 和不可 collate 的字段
#         transposed = list(zip(*batch))
#         result = []
#         for samples in transposed:
#             # 尝试用 default_collate，如果失败就保留为 list
#             try:
#                 result.append(default_collate(samples))
#             except TypeError:
#                 result.append(list(samples))  # 字符串等保留为 list
#         return tuple(result)
#     elif isinstance(elem, dict):
#         return {key: custom_collate_fn([d[key] for d in batch]) for key in elem}
#     else:
#         return default_collate(batch)

def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=False),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=False
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    return train_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    model = SAM2UNeXT("sam2_hiera_large.pt", "model.safetensors")
    # model = BiRefNet(bb_pretrained=False)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    state_dict = torch.load("/home/notebook/code/personal/80410839/Matting/LAB/BirefNet_SOD_Mat_RM_ADD0904_LHLR/ckpt/tmp/epoch_45.pth", map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler

import numpy as np
import cv2
from PIL import Image
import tqdm

if os.path.exists('masks2'):
    shutil.rmtree("masks2")
os.makedirs("masks2")
class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader = data_loaders
        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        # self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)

        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
        img_path = batch[2]

        scaled_preds = [self.model(inputs)]
        loss_cls = 0.

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls

        if loss > 5:
            print(f"loss:{loss},img_path:{img_path}")
            # 保存下来，要求保存原图+mask+伪标结果+预测结果
            ori_img_path = img_path[0][0]
            ori_mask_path = img_path[1][0] #ori_img_path.replace("/im/","/gt/")
            img_ = Image.open(ori_img_path).convert("RGB")


            #, n = ori_mask_path.replace(ori_path,"").split("/gt/")
            #db_ = Image.open(new_mask_path+b+"/"+n).convert("L")
            mask_ = Image.open(ori_mask_path).convert("L")

            res = scaled_preds[0].sigmoid().data.cpu()
            res = res.numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            image_ori = np.array(img_.resize(res.shape))
            mask_ = np.array(mask_.resize(res.shape))
            #db_ = np.array(db_.resize(res.shape))
            # ,np.stack([db_] * 3, axis=-1)
            res = cv2.hconcat([image_ori, np.stack([mask_] * 3, axis=-1),np.stack([res] * 3, axis=-1)])

            Image.fromarray(res).save(f"masks2/{os.path.basename(ori_img_path)}")

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.eval()
        self.loss_dict = {}

        self.pix_loss.lambdas_pix_last['mae'] *= 1
        self.pix_loss.lambdas_pix_last['mse'] *= 0.9
        self.pix_loss.lambdas_pix_last['ssim'] *= 0.9

        for batch_idx, batch in tqdm.tqdm(enumerate(self.train_loader)):
            # with nullcontext if not args.use_accelerate or accelerator.gradient_accumulation_steps <= 1 else accelerator.accumulate(self.model):
            self._train_batch(batch)
            # Logger
            if (epoch < 2 and batch_idx < 10 and batch_idx % 2 == 0) or batch_idx % max(100, len(self.train_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Epoch[{epoch}/{args.epochs}] Iter[{batch_idx}/{len(self.train_loader)}].'
                info_loss = 'Training Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = f'@==Final== Epoch[{epoch}/{args.epochs}]  Training Loss: {self.loss_log.avg:.5g}  '
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


def main():

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )
    trainer.train_epoch(0)

    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()
