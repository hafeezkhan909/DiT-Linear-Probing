# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import timm

# assert timm.__version__ == "0.3.2" # version check
# from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

from engine_finetune import train_one_epoch, evaluate

import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from diffusion import create_diffusion
from download import find_model
from models import DiT_L_2
from diffusers.models import AutoencoderKL

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../DiT/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--use_amp", action='store_true', default=False)
    parser.add_argument("--timestep", default=10, type=int, help='time step conditioning') 
    parser.add_argument("--blockname", default="layer-13", type=str, help='layer to extract features from')

    return parser

def get_model(device):
    model = DiT_L_2().to(device)
    state_dict = find_model(f"DiT-XL-2-256x256.pt")
    # state_dict = torch.load("./DiT-XL-2-256x256.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(None) # 1000-len betas
    return model, diffusion

class DiTFeatureExtractor(nn.Module):
    def __init__(self, dit, diffusion, vae, timestep: int, blockname: str, use_amp: bool):
        super().__init__()
        self.dit = dit
        self.diffusion = diffusion
        self.vae = vae
        self.timestep = timestep
        self.blockname = blockname
        self.use_amp = use_amp

        # freeze everything inside extractor
        for p in self.dit.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False

        self.dit.eval()
        self.vae.eval()

    def forward(self, imgs):
        """
        imgs: [B,3,256,256] in [-1,1]
        returns: [B, hidden]
        """
        device = imgs.device
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215
                latents = latents.float()  # keep stable

        B = latents.shape[0]
        t = torch.full((B,), self.timestep, device=device, dtype=torch.long)

        noise = torch.randn_like(latents)
        x_t = self.diffusion.q_sample(latents, t, noise=noise)

        # classifier-free null label for DiT ImageNet models
        y_null = torch.full((B,), 1000, device=device, dtype=torch.long)

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                _, acts = self.dit(x_t, t, y_null, ret_activation=True)

            if not hasattr(self, "_printed"):
                print("Available act keys:", list(acts.keys()))
                print("Num layers:", len(acts))
                self._printed = True

            feat = acts[self.blockname]          # typically [B, tokens, C]
            feat = feat.float().detach()
            feat = feat.mean(dim=1)              # pool over tokens -> [B, C]

            if not hasattr(self, "_feat_debug"):
                print("feat shape:", feat.shape)
                print("feat mean:", feat.mean().item())
                print("feat std:", feat.std().item())
                print("feat min/max:", feat.min().item(), feat.max().item())
                self._feat_debug = True
            return feat

class LinearProbeModel(nn.Module):
    def __init__(self, extractor: DiTFeatureExtractor, hidden_size: int, num_classes: int):
        super().__init__()
        self.extractor = extractor
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size, affine=False),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: images
        feat = self.extractor(x)   # [B, hidden]
        logits = self.head(feat)   # [B, num_classes]
        return logits
        
def main(args):
    misc.init_distributed_mode(args)
    assert hasattr(args, "gpu"), "init_distributed_mode did not set args.gpu (are you using torchrun?)"

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            RandomResizedCrop(256, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    dit, diffusion = get_model(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # for linear prob only
    extractor = DiTFeatureExtractor(
        dit=dit,
        diffusion=diffusion,
        vae=vae,
        timestep=args.timestep,
        blockname=args.blockname,
        use_amp=args.use_amp,
    ).to(device)

    with torch.no_grad():
        sample_imgs, _ = next(iter(data_loader_val))
        sample_imgs = sample_imgs.to(device, non_blocking=True)
        hidden_size = extractor(sample_imgs).shape[-1]
        print("Hidden size:", hidden_size)

    model = LinearProbeModel(extractor, hidden_size, args.nb_classes).to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)

    if misc.get_rank() == 0:
        w = model_without_ddp.head[1].weight.detach().float()
        b = model_without_ddp.head[1].bias.detach().float()
        print("INIT head weight norm:", w.norm().item(), "bias norm:", b.norm().item())
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if misc.get_rank() == 0 and epoch == 0:
            w = model_without_ddp.head[1].weight.detach().float()
            b = model_without_ddp.head[1].bias.detach().float()
            print("head weight norm:", w.norm().item(), "bias norm:", b.norm().item())
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
