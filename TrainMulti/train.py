import os
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from SAM2UNeXT import SAM2UNeXT
torch.backends.cudnn.enabled = False

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--dist', default=True, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', default=True, help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
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


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    return train_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    num_outputs = getattr(config, 'num_outputs', 1)  # 默认单 mask 输出
    model = SAM2UNeXT("sam2_hiera_large.pt", "model.safetensors", num_outputs=num_outputs)
    # model = BiRefNet(bb_pretrained=False)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            # 从旧checkpoint加载时，允许部分匹配（如果是从单mask模型加载到多mask模型）
            model.load_state_dict(state_dict, strict=False)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    state_dict = torch.load("epoch_20.pth", map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    # 从旧checkpoint加载时，允许部分匹配（如果是从单mask模型加载到多mask模型）
    model.load_state_dict(state_dict, strict=False)
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


    # param_groups = [
    # {'params': model.dino.parameters(), 'lr': config.lr*0.1},
    # {'params': [p for n, p in model.named_parameters() 
    #             if 'dino.model' not in n], 
    #  'lr': config.lr}]

    param_groups = [
        # 只传入 requires_grad=True 的参数
        {'params': filter(lambda p: p.requires_grad, model.dino.parameters()), 'lr': config.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'dino' not in n], 'lr': config.lr}
    ]

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


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
        full_mask_lambda = getattr(config, 'full_mask_lambda', 0.01)
        decay_rate = getattr(config, 'decay_rate', 0.2)
        self.pix_loss = PixLoss(full_mask_lambda=full_mask_lambda, decay_rate=decay_rate)
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
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        # 处理模型输出：可能是单 mask 或多 mask 格式
        if isinstance(outputs, dict):
            # 多 mask 输出格式
            scaled_preds = outputs
        else:
            # 单 mask 输出格式（保持向后兼容）
            scaled_preds = [outputs]

        # Loss - 传递当前 epoch 用于多 mask 损失衰减
        loss_pix, loss_dict_pix = self.pix_loss(
            scaled_preds, 
            torch.clamp(gts, 0, 1), 
            pix_loss_lambda=1.0,
            epoch=getattr(self, 'current_epoch', 0)
        )
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix

        self.loss_log.update(loss.item(), inputs.size(0))
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
            max_grad_norm = 15.0 if getattr(self, 'current_epoch', 1) == 1 else 5.0
            if hasattr(self.model, 'module'):
                clip_grad_norm_(self.model.module.parameters(), max_grad_norm)
            else:
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
        else:
            loss.backward()
            max_grad_norm = 15.0 if getattr(self, 'current_epoch', 1) == 1 else 5.0
            if hasattr(self.model, 'module'):
                clip_grad_norm_(self.model.module.parameters(), max_grad_norm)
            else:
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.current_epoch = epoch  # 保存当前 epoch 用于损失计算
        self.loss_dict = {}
        
        # 动态调整 alpha_cons 权重（mae使用config.py中的值，不再动态调整）
        if config.task == 'Matting':
            epoch_progress = epoch / args.epochs  # 当前epoch进度 (0-1)
            
            # 动态调整 alpha_cons 权重
            if epoch_progress <= 0.6:
                # 前60%的epoch：alpha_cons设置为0
                self.pix_loss.lambdas_pix_last['alpha_cons'] = 0
            else:
                # 60%以后：alpha_cons设置为5
                self.pix_loss.lambdas_pix_last['alpha_cons'] = 5
        
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        for batch_idx, batch in enumerate(self.train_loader):
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

    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
        if args.use_accelerate:
            state_dict = trainer.model.state_dict()
        else:
            state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
        torch.save(state_dict, os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))
    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()
