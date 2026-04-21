#!/usr/bin/env python
# Copyright (c) Alibaba Group
import argparse
import builtins
import os
import random
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import secu.loader
import secu.folder
import secu.builder
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=401, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log', type=str)
# options for secu
parser.add_argument('--secu-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--secu-num-ins', default=50000, type=int,
                    help='number of instances (default: 50000)')
parser.add_argument('--secu-num-head', default=3, type=int,
                    help='number of k-means ( default: 10)')
parser.add_argument('--secu-k', default=[4,5,6], type=int, nargs="+", help='multi-clustering head')
parser.add_argument('--secu-tx', default=0.05, type=float,
                    help='temperature for representation (default: 0.05)')
parser.add_argument('--secu-tw', default=0.05, type=float,
                    help='temperature for cluster center (default: 0.05)')
parser.add_argument('--secu-tau', default=0.2, type=float,
                    help='weight of one-hot label (default: 0.2)')
parser.add_argument('--secu-dual-lr', default=0.1, type=float,
                    help='dual learning rate for lower bound (default: 0.1)')
parser.add_argument('--secu-lratio', default=0.9, type=float,
                    help='lower-bound ratio (default: 0.4)')
parser.add_argument('--secu-alpha', default=6000, type=float,
                    help='entropy weight (default: 6000)')
parser.add_argument('--secu-cst', default='size', type=str,
                    help='constraint in secu: size or entropy or size-mml')
parser.add_argument('--clr', default=1.2, type=float,
                    help='learning rate for cluster center')
parser.add_argument('--min-crop', default=0.3, type=float,
                    help='minimal scale for random crop')
parser.add_argument('--data-name', default='cifar10', type=str,
                    help='name of data: cifar10, cifar100, stl10')
parser.add_argument('--use-medoid', default=0, type=int,
                    help='use new revised with SeCu')
parser.add_argument('--warm-up', default=15, type=int,
                    help='use new revised with SeCu')
parser.add_argument('--backbone', default='resnet18', type=str,
                    help='backbone architecture: resnet18 or vit')
parser.add_argument('--bands',default=3, type=int, help= 'channel for dataset: 3 or 4')


def main():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    start_time = time.time()
    args = parser.parse_args()
    print(args)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    print(f"總訓練時間：{elapsed_hours:.2f} 小時")


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    #add log to tensorboard
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter(f"runs/{args.log}")
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    assert (len(args.secu_k) == args.secu_num_head)
    print("=> creating model")
    if args.backbone == 'resnet18':
        if args.data_name == 'stl10':
            from nets.resnet_stl import resnet18
        elif args.data_name == 'cifar10' or args.data_name == 'cifar100':
            from nets.resnet_cifar import resnet18
        elif args.data_name == 'custom':
            from nets.resnet_custom import resnet18
        base_encoder = resnet18
    elif args.backbone == 'vit':
        from nets.vit import ViT
        base_encoder = ViT
    elif args.backbone == 'dinov2':
        from nets.vit import DINOv2
        base_encoder = DINOv2
    else:
        print("Input data set is not supported")
        return

    model = secu.builder.SeCu(
        base_encoder=base_encoder,
        K=args.secu_k,
        tx=args.secu_tx,
        tw=args.secu_tw,
        dim=args.secu_dim,
        num_ins=args.secu_num_ins,
        alpha=args.secu_alpha,
        dual_lr=args.secu_dual_lr,
        lratio=args.secu_lratio,
        constraint=args.secu_cst
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # 取得原始模型（DistributedDataParallel 包裝後需透過 .module 存取）
    _model = model.module if hasattr(model, 'module') else model

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    centers = []
    encoder = []
    for name, param in model.named_parameters():
        if 'center' in name:
            centers.append(param)
        else:
            encoder.append(param)
    if args.backbone == 'vit' or args.backbone == 'dinov2':
        optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.lr},
                                    {"params": centers, "lr": args.clr}],
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([{"params": encoder, "lr": args.lr},
                                    {"params": centers, "lr": args.clr}],
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 修正：透過 _model 呼叫 load_param()
    _model.load_param()
    cudnn.benchmark = True

    traindir = args.data
    if args.data_name == 'cifar10' or args.data_name == 'custom':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        crop_size = 224
    elif args.data_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        crop_size = 32
    elif 'stl' in args.data_name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        crop_size = 96

    aug_1 = [
        transforms.RandomResizedCrop(crop_size, scale=(args.min_crop, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.01)
        ], p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([secu.loader.GaussianBlur([.1, 1.])], p=0.7),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    aug_2 = [
        transforms.RandomResizedCrop(crop_size, scale=(0.4, 0.8)),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.15, 0.05)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([secu.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = secu.folder.ImageFolder(
        traindir,
        secu.loader.DoubleCropsTransform(transforms.Compose(aug_1),
                                         transforms.Compose(aug_2)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    scaler = GradScaler()
    best_epoch_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        avg_loss, loss_x, loss_c = train(train_loader, model, criterion, optimizer, epoch, args, scaler, writer)
        if args.secu_cst == 'size' or args.secu_cst == 'size-mml':
            # 修正：透過 _model 呼叫 reset_count()
            _model.reset_count()
        print('use time :', time.time() - start_time)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='model/{}_{:04d}.pth.tar'.format(args.log, epoch))
        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            save_best_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='model/best_model.pth.tar')
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, scaler, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')

    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Epoch: [{}]".format(epoch))

    # 修正：統一透過 _model 存取原始模型
    _model = model.module if hasattr(model, 'module') else model

    if args.use_medoid and epoch >= args.warm_up:
        _model.recompute_medoids_topk(train_loader)
    pcenters = _model.get_centers()

    model.train()
    end = time.time()
    train_loader_len = len(train_loader)

    for i, (images, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args, i, train_loader_len)
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu)
        with autocast():
            loss_x, loss_c, loss_mml = model(images[0], images[1], pcenters, target, epoch, criterion, args)
            loss = loss_x + loss_c + loss_mml
        losses.update(loss.item(), images[0].size(0))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/loss_swap', loss_x.item(), global_step)
            writer.add_scalar('train/loss_constraint', loss_c.item(), global_step)
            writer.add_scalar('train/loss_cosine_mml', loss_mml.item(), global_step)
            for h in range(args.secu_num_head):
                # 修正：透過 _model 存取 counters
                writer.add_scalar(f'train/cluster{h}_max', _model.counters[h].max().item(), global_step)
                writer.add_scalar(f'train/cluster{h}_min', _model.counters[h].min().item(), global_step)

    progress.display(train_loader_len)
    for i in range(0, args.secu_num_head):
        print('max and min cluster size for {}-class clustering is ({},{})'.format(
            args.secu_k[i],
            torch.max(_model.counters[i].data).item(),
            torch.min(_model.counters[i].data).item()))
    return losses.avg, loss_x.item(), loss_c.item()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if (state['epoch'] - 1) % 50 != 0 or state['epoch'] == 1:
        return
    torch.save(state, filename)


def save_best_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, iteration, num_iter):
    warmup_epoch = args.warm_up
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter
    lr = args.lr * (1 + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < warmup_epoch:
        if epoch == 0:
            lr = 0
        else:
            lr = args.lr * max(1, current_iter - num_iter) / (warmup_iter - num_iter)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = args.clr


if __name__ == '__main__':
    main()