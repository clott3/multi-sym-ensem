from distutils.command.build import build
from pathlib import Path
import argparse
import os
import sys
import random

import time
import json
import math
import numpy as np

from torch import nn, optim
import torch
import torch.distributed as dist
import torchvision

import tensorboard_logger as tb_logger

from utils import gather_from_all
from itertools import permutations
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from utils import gather_from_all, GaussianBlur, Solarization

parser = argparse.ArgumentParser(description='E-SSL or I-SSL Training')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset (imagenet only')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--checkpoint-dir', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path,
                    metavar='LOGDIR', help='path to tensorboard log directory')
parser.add_argument('--lmbd', default=0.0, type=float,
                    help="coefficient of rotation loss") # set to 0.4 for rot-eq models
parser.add_argument('--scale', default='0.05,0.14', type=str)

# Training / loss specific parameters
parser.add_argument('--temp', default=0.2, type=float,
                    help='Temperature for InfoNCE loss')
parser.add_argument('--opt-momentum', default=0.9, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--optimizer', default='lars', type=str,
                    help='Optimizer', choices=['lars', 'sgd'])

# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes')
parser.add_argument("--timeout", default=360, type=int,
                    help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str,
                    help="Partition where to submit")

parser.add_argument("--exp", default="SimCLR", type=str,
                    help="Name of experiment")

parser.add_argument("--seed", default=None, type=int,
                    help="seed")
parser.add_argument('--rotate', type=str, default=None, choices=['inv', 'eq'])
parser.add_argument('--trans_p', default=0.5, type=float,
                    help="probability of applying transformation for inv versions")
parser.add_argument('--downsize', default=96, type=int,
                    help="downsize images for rot prediction to conserve memory")

# wsc setting
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://10.3.1.9:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")


# rotation
def rotate_images(images, gpu, single=False, trans_p=0.5):
    nimages = images.shape[0]

    if single:
        y = []
        for i in range(nimages):
            rotdeg = np.random.choice(4, 1, p=[(1-trans_p), trans_p/3, trans_p/3, trans_p/3]).item()
            y.append(rotdeg)
            images[i] = torch.rot90(images[i], rotdeg, [1, 2])
        y = torch.LongTensor(y).cuda()
        return images.cuda(gpu), y

    n_rot_images = 4 * nimages
    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda(gpu,
                                                                                                         non_blocking=True)
    rot_classes = torch.zeros([n_rot_images]).long().cuda(gpu, non_blocking=True)

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages:2 * nimages] = images.flip(3).transpose(2, 3)
    rot_classes[nimages:2 * nimages] = 1
    # rotate 180
    rotated_images[2 * nimages:3 * nimages] = images.flip(3).flip(2)
    rot_classes[2 * nimages:3 * nimages] = 2
    # rotate 270
    rotated_images[3 * nimages:4 * nimages] = images.transpose(2, 3).flip(3)
    rot_classes[3 * nimages:4 * nimages] = 3

    return rotated_images, rot_classes

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.scale = [float(x) for x in args.scale.split(',')]
    if 'SLURM_JOB_ID' in os.environ:
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58478'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(gpu, args):
    # args.rank += gpu
    if args.seed is not None:
        fix_seed(args.seed)
        print(f"using seed {args.seed} for pre-training")

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = SimCLR(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=exclude_bias_and_norm,
                        lars_adaptation_filter=exclude_bias_and_norm)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0,
                                    momentum=args.opt_momentum, weight_decay=args.weight_decay)


    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    else:
        start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform(args))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for step, inputs in enumerate(loader, start=epoch * len(loader)):

            (y1, y2, y3), labels = inputs
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)

            if args.rotate == 'inv':
                y1, _ = rotate_images(y1, gpu, single=True, trans_p=args.trans_p)
                y2, _ = rotate_images(y2, gpu, single=True, trans_p=args.trans_p)
            
            labels = labels.cuda(gpu, non_blocking=True)

            if args.rotate == 'eq':
                y3 = y3.cuda(gpu, non_blocking=True)
                rotated_images, rotated_labels = rotate_images(y3, gpu)
          
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss, acc = model.forward(y1, y2, labels)

                if args.rotate == 'eq':
                    logits = model.module.forward_rotation(rotated_images)
                    rot_loss = torch.nn.functional.cross_entropy(logits, rotated_labels)
                    loss += args.lmbd * rot_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(f'epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}', flush=True)
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                            optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

            # save checkpoint to epoch
            if epoch % args.save_freq == 0 and epoch != 0:
                torch.save(state, args.checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch))

            # log to tensorboard
            logger.log_value('loss', loss.item(), epoch)
            logger.log_value('acc', acc.item(), epoch)
            logger.log_value('learning_rate', lr, epoch)

    if args.rank == 0:
        # save final model
        torch.save(dict(backbone=model.module.backbone.state_dict(),
                        projector=model.module.projector.state_dict(),
                        head=model.module.online_head.state_dict()),
                args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate #* args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.dataset == 'imagenet':
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        elif args.dataset == 'cifar100':
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)

        self.backbone.fc = nn.Identity()

        # projector
        if args.dataset == 'imagenet':
            sizes = [2048, 2048, 2048, 128]
        elif args.dataset == 'cifar100':
            sizes = [512, 512, 512, 128]

        args.feat_dim = sizes[-1]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        self.projector = nn.Sequential(*layers)

        if args.dataset == 'imagenet':
            self.online_head = nn.Linear(2048, 1000)
        elif args.dataset == 'cifar100':
            self.online_head = nn.Linear(512, 100)

        if args.rotate == 'eq':
            num_rotate_classes = 4
            self.rotation_projector = nn.Sequential(nn.Linear(2048, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.ReLU(inplace=True),  # first layer
                                                    nn.Linear(2048, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.ReLU(inplace=True),  # second layer
                                                    nn.Linear(2048, 128),
                                                    nn.LayerNorm(128),
                                                    nn.Linear(128, num_rotate_classes))  # output layer


    def forward(self, y1, y2, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        # projection
        z1 = self.projector(r1)
        z2 = self.projector(r2)

        loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
            
        logits = self.online_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc

    def forward_rotation(self, x):
        b = self.backbone(x)
        logits = self.rotation_projector(b)

        return logits

def infoNCE(nn, p, temperature=0.2, gather_all=True):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    if gather_all:
        nn = gather_from_all(nn)
        p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class Transform:
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.transform_rotation = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(args.scale[0], args.scale[1])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        y3 = self.transform_rotation(x)
        return y1, y2, y3

def exclude_bias_and_norm(p):
    return p.ndim == 1


if __name__ == "__main__":
    main()
