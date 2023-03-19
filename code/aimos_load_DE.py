import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import pathlib

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Subset

from networks import *
from utils import *
import config as cf
from losses import *
import json
from data_sampler import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--dataset', metavar='DATASET', nargs='?', default='imagenet',
                    help='name of dataset (imagenet, cifar100)')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=36, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--warmup-epochs', default=5, type=int,
                    help='number of epochs to warm up lr')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--decay-epochs', nargs="+", default=[30, 60, 80], type=int,
                    help='epochs at which to decay the lr')
parser.add_argument('--decay-ratio', default=0.1, type=float,
                    help='rate to decay the ratio')
# parser.add_argument('--lr-step', action='store_true')
# parser.add_argument('--lr-step-size', '--learning-rate-step-size', default=30, type=int,
                    # metavar='LR-SS', help='step size for learning rate for step-lr scheduler')
# parser.add_argument('--lr-cosine', action='store_true')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--exp-id', default='', type=str,
                    help='Experiment ID for saving the outputs/checkpoints into a file')

# ensem params
parser.add_argument('--ensem_samples', default=4, type=int)
parser.add_argument('--width', default=2, type=int, help='width of model')
parser.add_argument('--BE', action='store_true')
parser.add_argument('--arch', default='rn50', type=str)
parser.add_argument('--reset_seed', action='store_true', default=True)
parser.add_argument('--load_pt_wseed', action='store_true')
parser.add_argument('--load_pt_noseed', action='store_true')
parser.add_argument('--pt_ckpt', default='', type=str)
parser.add_argument('--mlp_depth', default=1, type=int)
parser.add_argument('--mlp_hidden_dim', default=512, type=int)
parser.add_argument('--proj_dim', default=512, type=int)

# NCE params
parser.add_argument('--nce', action='store_true')
parser.add_argument('--naive', default= '', type=str)
parser.add_argument('--temp', default=0.1, type=float)
parser.add_argument('--neg_opt', default='', type=str)
parser.add_argument('--min_mi_mem', action='store_true')
parser.add_argument('--nce_lamb', default=0.1, type=float)
parser.add_argument('--seq', action='store_true')
parser.add_argument('--separate_proj', action='store_true')
parser.add_argument('--share_proj', action='store_true')

parser.add_argument('--cc', action='store_true')
parser.add_argument('--no_diag', action='store_true')

parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--use_lmbd_sch', action='store_true')

parser.add_argument('--bias_sampler', action='store_true')
parser.add_argument('--ncls_per_batch', default=2, type=int)

parser.add_argument('--submit', action='store_true')
parser.add_argument('--server', default='sc', type=str)
parser.add_argument('--arg_str', default='', type=str)
parser.add_argument('--add_prefix', default='', type=str)
parser.add_argument('--log_id', default='', type=str)
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--no_save_every', action='store_true')
parser.add_argument('--distributed', action='store_true')

best_acc1 = 0
best_acc1_mem = None
best_ece = None


# torch.cuda.set_device(gpu)
# torch.backends.cudnn.benchmark = True
args = parser.parse_args()
args.num_classes = 1000

print("=> creating model '{}'".format(args.arch))
encoder, classifier, projector = init_models(args, args.pt_ckpt)
print(encoder.encoders.keys())
print(classifier.branches.keys())
