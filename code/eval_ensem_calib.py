from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import warnings
import getpass
import copy

from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch
import numpy as np
import tensorboard_logger as tb_logger
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from networks import EnsembleSSL, TS_EnsembleSSL
from datasets import build_dataloaders, dataset_num_classes
from utils import accuracy, AverageMeter, ECELoss
import scipy as sp
from datasets import Split_Dataset

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=('imagenet', 'imagenet-a', 'imagenet-r',
                    'imagenet-v2', 'imagenet-sketch', 'imagenet-100',
                    'imagenet-100-a', 'imagenet-100-r', 'imagenet-100-v2',
                    'imagenet-100-sketch', 'inat-1k', 'cub-200', 'flowers-102',
                    'food', 'cifar10', 'cifar100', 'pets', 'sun-397', 'cars',
                    'aircraft', 'voc-2007', 'dtd', 'caltech-101'),
                    help='dataset name')
parser.add_argument('--pretrained', nargs="+", default=[], type=str,
                    help='paths to pretrained models')
parser.add_argument('--eval-mode', default='freeze', type=str,
                    choices=('finetune', 'linear_probe', 'freeze', 'log_reg', 'temp_scale'),
                    help='finetune, linear probe, logistic regression, or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--val-batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size for validation (uses only 1 gpu)')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=1.0, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--lr-scheduler', default=None, type=str, metavar='LR-SCH',
                    choices=('cosine'),
                    help='scheduler for learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')

# single gpu training params
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# distributed training params
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# ensemble params
parser.add_argument('--num-ensem', default=1, type=int,
                    help='number of members in the ensemble')
parser.add_argument('--arch', default='resnet50', type=str,
                    choices=('resnet18', 'resnet50'),
                    help='architecture for each member in the ensemble')
parser.add_argument('--convert', action='store_true', default=False,
                    help='Whether to convert from single MultiBackbone \
                    checkpoint to use with EnsembleSSL')
parser.add_argument('--combine_sep_ckpts', action='store_true', default=False,
                    help='Whether to combine sep checkpoints from EnsembleSSL')


# submit params
parser.add_argument('--server', type=str, default='sc')
parser.add_argument('--arg_str', default='--', type=str)
parser.add_argument('--add_prefix', default='', type=str)
parser.add_argument('--submit', action='store_true')

# misc
parser.add_argument('--seed', default=None, type=int, metavar='S',
                    help='random seed')
parser.add_argument('--exp-id', default='', type=str,
                    help='Experiment ID for saving the outputs/checkpoints into a file')
parser.add_argument('--stats-filename', default='results.txt', type=str,
                    help='Stats filename to aggregate all results')
parser.add_argument('--save_freq', default=20, type=int)
parser.add_argument('--ensem_pred', default='DE', type=str)
parser.add_argument('--use_default_pretrained', action='store_true')
parser.add_argument('--use_smaller_split', action='store_true')
parser.add_argument('--val_perc', default=20, type=int)
parser.add_argument('--lr-temp', default=0.01, type=float, metavar='LR',
                    help='temperature learning rate')
parser.add_argument('--gate', default='none', choices=('frozen', 'joint', 'none'), type=str)
parser.add_argument('--lp_100_on_full', default='none', choices=('base', 'eq', 'inv', 'none'), type=str)

parser.add_argument('--eval_subset100', action='store_true')
parser.add_argument('--eval_on_train', action='store_true')
parser.add_argument('--max_iter', default=50, type=int)
parser.add_argument('--train_val_split', default=-1, type=int)
parser.add_argument('--use_smaller_split_val', action='store_true')
parser.add_argument('--eval_var_subset', default=None, type=int)
parser.add_argument('--fold', default=None, type=int)


# def main():
#     args = parser.parse_args()
#     args.ngpus_per_node = torch.cuda.device_count()

#     # single-node distributed training
#     args.rank = 0
#     args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
#     args.world_size = args.ngpus_per_node
#     torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def main():
    args = parser.parse_args()

    if args.submit:
        make_sh_and_submit(args)
        return

    if args.use_default_pretrained:
        if args.dataset == 'imagenet':
            args.data = Path('/home/gridsan/groups/datasets/ImageNet')
            args.pretrained = ['/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth',\
            '/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth',\
            '/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth']
        elif args.dataset == 'imagenet-100':
            args.data = Path('/home/gridsan/groups/MAML-Soljacic/imagenet-100')
            if args.gate == 'joint'or (args.gate == 'none' and args.eval_mode == 'linear_probe'):
                args.pretrained = ['/home/gridsan/groups/MAML-Soljacic/pretrain_checkpoints/pretrain_checkpoints/simclr-rotinv-lr6.4_resnet50.pth',\
                                    '/home/gridsan/groups/MAML-Soljacic/pretrain_checkpoints/pretrain_checkpoints/simclr-roteq-lr6.4-lamb0.4_resnet50.pth',\
                                    '/home/gridsan/groups/MAML-Soljacic/pretrain_checkpoints/pretrain_checkpoints/simclr-base-lr6.4-resnet50.pth']
            elif args.gate == 'frozen' or (args.gate == 'none' and args.eval_mode == 'freeze'):
                args.pretrained = ['/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',\
                                '/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth', \
                                '/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth']
        else:
            raise "Only imagenet and imagenet-100 supported"

    elif args.lp_100_on_full != 'none':
        args.data = Path('/home/gridsan/groups/MAML-Soljacic/imagenet-100')
        if args.lp_100_on_full == 'base':
            args.pretrained = ['/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth']
        elif args.lp_100_on_full == 'eq':
            args.pretrained = ['/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth']
        elif args.lp_100_on_full == 'inv':
            args.pretrained = ['/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth']

    # args.exp_id = f'{args.dataset}_' + args.exp_id
    args.checkpoint_dir = Path('./checkpoints/') / args.exp_id
    args.log_dir = Path('./logs/') / args.exp_id
    args.stats_dir = Path('./stats/')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)

    # single node distributed training
    args.ngpus_per_node = torch.cuda.device_count()
    if args.dist_url == '':
        args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'

    args.rank = 0
    # args.world_size = args.ngpus_per_node

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
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

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function / single gpu training
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node=None, args=None):

    if not (args.distributed and args.server == 'aimos'):
        args.rank = args.rank * ngpus_per_node + gpu

    if args.distributed:
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    print(' '.join(sys.argv))

    if args.rank == 0:
        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file() and args.eval_mode != 'temp_scale':
        print('Resuming from previous checkpoint')
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        num_ensem = ckpt['num_ensem']
        args.num_ensem = num_ensem
        if args.eval_mode == 'temp_scale':
            model = TS_EnsembleSSL(args.arch, num_ensem, dataset_num_classes[args.dataset], args.eval_mode).cuda()
        else:
            model = EnsembleSSL(args.arch, num_ensem, dataset_num_classes[args.dataset], args.eval_mode).cuda()
        model.load_state_dict(ckpt['model'])
    elif args.combine_sep_ckpts:
        num_ensem = len(args.pretrained)
        args.num_ensem = num_ensem
        print(f'Combining separate checkpionts where each EnsembleSSL was trained separately: {num_ensem} models')
        if args.eval_mode == 'temp_scale':
            model = TS_EnsembleSSL(args.arch, num_ensem, dataset_num_classes[args.dataset], args.eval_mode).cuda()
        else:
            model = EnsembleSSL(args.arch, num_ensem, dataset_num_classes[args.dataset], args.eval_mode).cuda()
        start_epoch = 0
        model.load_sep_weights(args.pretrained)

    else:
        print('Loading weights from pre-trained weights')
        num_ensem = len(args.pretrained)
        args.num_ensem = num_ensem
        if args.eval_mode == 'temp_scale':
            model = TS_EnsembleSSL(args.arch, args.num_ensem, dataset_num_classes[args.dataset], args.eval_mode).cuda()
        else:
            model = EnsembleSSL(args.arch, args.num_ensem, dataset_num_classes[args.dataset], args.eval_mode).cuda()

        start_epoch = 0
        model.load_weights(args.pretrained, convert_from_single=args.convert)

    classifier_parameters, model_parameters = [], []
    if args.eval_mode == 'temp_scale':
        temp_parameters = []
    for name, param in model.named_parameters():
        if 'fc.weight' in name or 'fc.bias' in name:
            classifier_parameters.append(param)
        elif 'temp' in name:
            temp_parameters.append(param)
        else:
            model_parameters.append(param)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    if args.eval_mode == 'temp_scale':
        param_groups = [dict(params=temp_parameters, lr=args.lr_temp)]
    # set optimizer
    else:
        param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
        if args.eval_mode == 'finetune':
            param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)

    # set scheduler
    scheduler = None
    if args.lr_scheduler is not None:
        if args.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        else:
            raise NotImplementedError(f'Scheduler {args.lr_scheduler} not implemented!')

    if (args.checkpoint_dir / 'checkpoint.pth').is_file() and args.eval_mode != 'temp_scale':
        optimizer.load_state_dict(ckpt['optimizer'])

    best_acc = argparse.Namespace(top1=0, top5=0)

    # create dataloaders
    train_loader, _, val_loader, indices_in_1k = build_dataloaders(args)

    args.start_time = time.time()

    # if freeze weights and evaluate, then no need for training -- reset start_epoch & epochs
    if args.eval_mode in {'freeze', 'log_reg', 'temp_scale'}:
        start_epoch = 0
        args.epochs = 1

    is_best = False
    for epoch in range(start_epoch, args.epochs):
        if args.eval_mode == 'finetune' or args.eval_mode == 'linear_probe':
            print(f'Epoch {epoch} training: \n')
            train(model, train_loader, optimizer, scheduler, criterion, epoch, args)
        elif args.eval_mode == 'log_reg':
            stats_json, best_acc, best_weights, best_bias, best_c = train_validate_logreg(model, train_loader, val_loader, args)
        elif args.eval_mode == 'temp_scale':
            stats_json = train_validate_LBFGS(model, train_loader, val_loader, args)
            acc_per_class = None
        # validate & save checkpoint
        if args.rank == 0:
            print(f'Epoch {epoch} validation: \n')
            if args.eval_mode != 'log_reg' and args.eval_mode != 'temp_scale':
                stats_json, acc_per_class, is_best = validate(model, val_loader, logger, indices_in_1k, epoch, args, best_acc)

            # only save per epoch weights & results if training is being done
            if args.eval_mode != 'freeze':
                if args.distributed:
                    state = dict(epoch=epoch + 1, model=model.module.state_dict(),
                                    optimizer=optimizer.state_dict(), num_ensem=num_ensem)
                else:
                    state = dict(epoch=epoch + 1, model=model.state_dict(),
                                    optimizer=optimizer.state_dict(), num_ensem=num_ensem)

                if args.eval_mode == 'log_reg':
                    state['log_reg_weight'] = best_weights
                    state['log_reg_bias'] = best_bias
                    state['log_reg_c'] = best_c

                torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

                # save checkpoint to epoch
                if epoch % args.save_freq == 0 and epoch != 0:
                    torch.save(state, args.checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch))

                if is_best:
                    torch.save(state, args.checkpoint_dir / 'checkpoint_best.pth')

                # log results per epoch
                with open(args.stats_dir / f'{args.exp_id}_results.txt', 'a+') as f:
                    f.write(f'epoch {epoch} | ')
                    f.write(stats_json)
                    f.write('\n')

            if epoch == (args.epochs - 1):
                with open(args.stats_dir / args.stats_filename, 'a+') as f:
                    f.write(f'{args.exp_id} | ')
                    f.write(stats_json)
                    f.write('\n')

                # saving accuracy per class tensor
                if acc_per_class is not None:
                    torch.save({'acc_per_class': acc_per_class}, args.checkpoint_dir / 'acc_per_class.pth')


def train(model, train_loader, optimizer, scheduler, criterion, epoch, args):
    if args.eval_mode == 'finetune':
        model.train()
    elif args.eval_mode == 'linear_probe' or args.eval_mode == 'freeze':
        model.eval()
    else:
        raise NotImplementedError(f'{args.eval_mode} mode not implemented')

    if args.distributed:
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)

    for step, (images, target) in enumerate(tqdm(train_loader), start=epoch * len(train_loader)):
        output = model(images.cuda(non_blocking=True))
        loss = 0.0
        for m in range(args.num_ensem):
            loss += criterion(output[m], target.cuda(non_blocking=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % args.print_freq == 0:
            if args.distributed:
                torch.distributed.reduce(loss.div_(args.world_size), 0)

            if args.rank == 0:
                pg = optimizer.param_groups
                lr_classifier = pg[0]['lr']
                lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                lr_classifier=lr_classifier, loss=loss.item(),
                                time=int(time.time() - args.start_time))
                print(json.dumps(stats))

    if scheduler is not None:
        scheduler.step()

def train_validate_LBFGS(model, train_loader, val_loader, args):
    if args.eval_mode != 'temp_scale':
        raise NotImplementedError('In the wrong train/validate function not implemented')

    if args.distributed:
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(0)

    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss(input_softmax=False).cuda()
    model.eval()
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    if (args.checkpoint_dir / 'x_train.pth').is_file():
        x_train = torch.load(args.checkpoint_dir / 'x_train.pth')
        y_train = torch.load(args.checkpoint_dir / 'y_train.pth')
        x_val = torch.load(args.checkpoint_dir / 'x_val.pth')
        y_val = torch.load(args.checkpoint_dir / 'y_val.pth')
    else:
        print('Precomputing features')
        with torch.no_grad():
            x_train, y_train, x_val, y_val = [], [], [], []
            for _, (images, target) in enumerate(tqdm(train_loader)):
                output = model(images.cuda(non_blocking=True)).squeeze()
                x_train.append(output)
                y_train.append(target)

            for _, (images, target) in enumerate(tqdm(val_loader)):
                output = model(images.cuda(non_blocking=True)).squeeze()
                x_val.append(output)
                y_val.append(target)


        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0)
        x_val = torch.cat(x_val, dim=0)
        y_val = torch.cat(y_val, dim=0)

        torch.save(x_train, args.checkpoint_dir / 'x_train.pth')
        torch.save(y_train, args.checkpoint_dir / 'y_train.pth')
        torch.save(x_val, args.checkpoint_dir / 'x_val.pth')
        torch.save(y_val, args.checkpoint_dir / 'y_val.pth')

    print(len(x_train), len(y_train))
    print(len(x_val), len(y_val))
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    x_val = x_val.cuda()
    y_val = y_val.cuda()

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(x_train, y_train).item()
    before_temperature_ece = ece_criterion(x_train, y_train).item()
    acc1, acc5 = accuracy(x_train, y_train, topk=(1, 5))

    print('Before temperature - Train NLL: %.3f, ECE: %.3f, Acc@1: %.2f, Acc@5: %.2f' % (before_temperature_nll, before_temperature_ece, acc1[0].item(), acc5[0].item()))

    before_temperature_nll = nll_criterion(x_val, y_val).item()
    before_temperature_ece = ece_criterion(x_val, y_val).item()
    acc1, acc5 = accuracy(x_val, y_val, topk=(1, 5))

    print('Before temperature - Test NLL: %.3f, ECE: %.3f, Acc@1: %.2f, Acc@5: %.2f' % (before_temperature_nll, before_temperature_ece, acc1[0].item(), acc5[0].item()))

    # temp_parameters = []
    # for name, param in model.named_parameters():
    #     if 'temp' in name:
    #         temp_parameters.append(param)
    #     param_groups = [dict(params=temp_parameters, lr=args.lr_temp)]


    optimizer = optim.LBFGS([model.temp], lr=args.lr_temp, max_iter=args.max_iter)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(model.temperature_scale(x_train), y_train)
        loss.backward()
        return loss

    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(model.temperature_scale(x_train), y_train).item()
    after_temperature_ece = ece_criterion(model.temperature_scale(x_train), y_train).item()
    acc1, acc5 = accuracy(model.temperature_scale(x_train), y_train, topk=(1, 5))
    print(f'Params | temp_lr: {args.lr_temp}, max_iter: {args.max_iter}')
    print('Optimal temperature: %.3f' % model.temp)
    print('After temperature - Train NLL: %.3f, ECE: %.3f, Acc@1: %.2f, Acc@5: %.2f' % (after_temperature_nll, after_temperature_ece, acc1[0].item(), acc5[0].item()))

    after_temperature_nll = nll_criterion(model.temperature_scale(x_val), y_val).item()
    after_temperature_ece = ece_criterion(model.temperature_scale(x_val), y_val).item()
    acc1, acc5 = accuracy(model.temperature_scale(x_val), y_val, topk=(1, 5))
    print('After temperature - Test NLL: %.3f, ECE: %.3f, Acc@1: %.2f, Acc@5: %.2f' % (after_temperature_nll, after_temperature_ece, acc1[0].item(), acc5[0].item()))

    stats = dict(dataset=args.dataset, val_perc=args.val_perc, acc1=acc1[0].item(), acc5=acc5[0].item())
    stats_dump = json.dumps(stats)
    print(stats_dump)
    is_best = True

    return stats_dump

def train_validate_logreg(model, train_loader, val_loader, args):
    if args.eval_mode != 'log_reg':
        raise NotImplementedError('In the wrong train/validate function not implemented')

    if args.distributed:
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(0)

    model.eval()
    print('Precomputing features')
    with torch.no_grad():
        x_train, y_train, x_val, y_val = [], [], [], []
        for _, (images, target) in enumerate(tqdm(train_loader)):
            output = model(images.cuda(non_blocking=True)).squeeze()
            x_train.append(output.cpu())
            y_train.append(target)

        for _, (images, target) in enumerate(tqdm(val_loader)):
            output = model(images.cuda(non_blocking=True)).squeeze()
            x_val.append(output.cpu())
            y_val.append(target)

    x_train = torch.cat(x_train, dim=0).numpy()
    y_train = torch.cat(y_train, dim=0).numpy()
    x_val = torch.cat(x_val, dim=0).numpy()
    y_val = torch.cat(y_val, dim=0).numpy()

    clf = LogisticRegression(max_iter=200, warm_start=True, random_state=0)  # follow exactly the LP-FT paper

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    # cs = np.logspace(1e-6, 1e5, 45)
    cs = np.logspace(-7, 2, 100)

    best_acc = -1.
    best_weights, best_bias, best_c, = None, None, None
    best_stats = None

    for c in tqdm(cs):
        clf.C = c
        clf.fit(x_train, y_train)
        pred_train = clf.predict(x_train)
        pred_val = clf.predict(x_val)
        acc_train = np.mean(pred_train == y_train) * 100.
        acc_val = np.mean(pred_val == y_val) * 100.
        stats = dict(c=c, acc_train=acc_train, acc_val=acc_val, time=int(time.time() - args.start_time))
        print(json.dumps(stats))

        if acc_val > best_acc:
            best_acc = acc_val
            best_weights = copy.deepcopy(clf.coef_)
            best_bias = copy.deepcopy(clf.intercept_)
            best_c = c
            best_stats = json.dumps(stats)

    best_weights /= std
    best_bias -= np.matmul(best_weights, np.ones(best_weights.shape[1]) * mean / std)

    return best_stats, best_acc, best_weights, best_bias, best_c

def ensem_pred(outputs, mode='DE', num_ensem=3, target=None):
    if mode == 'DE':
        output = outputs.softmax(dim=-1).mean(dim=0)
        _, ensem_preds = output.max(1)
        batch_acc = (ensem_preds.long() == target).sum()/ len(target)
        print("batch_acc:", batch_acc.item())
        conf, preds = outputs.max(dim=-1) # each has dim (M,B)
        pred_exist = (preds == target).sum(dim=0).bool()
        acc_ub = 1 - (~pred_exist).sum()/len(pred_exist)

        return ensem_preds, target, output, acc_ub * 100

    else:
        outputs = outputs.softmax(dim=-1)
        conf, preds = outputs.max(dim=-1) # each has dim (M,B)
        pred_exist = (preds == target).sum(dim=0).bool()
        acc_ub = 1 - (~pred_exist).sum()/len(pred_exist)

        if mode == 'most_conf':
            _, chosen_mem = conf.max(dim=0)
            chosen_mem = chosen_mem.unsqueeze(0)
            ensem_preds = torch.gather(preds,0,chosen_mem).squeeze(0)
            new_target = target
        if mode == 'hard_vote':
            predmode, count = sp.stats.mode(preds.cpu())
            # print("all preds:", preds)
            tie = (count < num_ensem - 1)
            ensem_preds = torch.Tensor(predmode[~tie]).to(outputs.device) # these are predictions without tie
            new_target = target[~tie]
            new_conf = conf.T[~tie].T
            new_preds = preds.T[~tie].T
            if num_ensem != 3:
                raise NotImplementedError("need to check tie condition")

            if tie.sum() > 0:
                tied_conf = conf.T[tie].T
                tied_preds = preds.T[tie].T
                # For samples under tie, choose the most confident model
                _, chosen_mem = tied_conf.max(dim=0)
                chosen_mem = chosen_mem.unsqueeze(0)
                tpreds = torch.gather(tied_preds,0,chosen_mem).squeeze(0) # these are predictions with tie
                ensem_preds = torch.cat([ensem_preds, tpreds],dim=0)
                new_target = torch.cat([new_target,target[tie]])
                new_conf = torch.cat([new_conf, tied_conf],dim=1)
                new_preds = torch.cat([new_preds, tied_preds],dim=1)

                print(f"tied fraction: {len(tied_preds)/len(ensem_preds)}")
            batch_acc = (ensem_preds.long() == new_target).sum()/ len(target)
            wrong_samples = (new_target != ensem_preds.long())
            wrong_conf = new_conf.T[wrong_samples].T
            wrong_preds = new_preds.T[wrong_samples].T
            wrong_target = new_target[wrong_samples]
            # print("target:", wrong_target)
            # print("pred matrix:", wrong_preds)
            pred_exist = (wrong_preds == wrong_target).sum(dim=0).bool()
            if len(pred_exist) > 0:
                print("conf:", wrong_conf.T[pred_exist].T)
                print("pred:", wrong_preds.T[pred_exist].T)
                print("target:", wrong_target[pred_exist])


            # print("conf matrix:", wrong_conf)

            print("batch_acc:", batch_acc.item())
            print("Acc upper bound:", acc_ub)

            assert(ensem_preds.shape[0] == outputs.shape[1])

        elif 'soft_vote' in mode: # each member's vote is weighted by its confidence
            print("Mode:",mode)
            if num_ensem != 3:
                raise NotImplementedError("need to check tie condition")
            # Can think of a faster way for this later
            # weighting only matters to switch decision when predictions are not tied (if 1 member conf > sum of 2 members)
            predmode, count = sp.stats.mode(preds.cpu())
            tie3 = (count == num_ensem) # if all members agree, prediction is straightforward
            ensem_preds = torch.Tensor(predmode[tie3]).to(outputs.device) # these are predictions without tie
            new_target = target[tie3]
            tie2 = (count == num_ensem - 1)
            if tie2.sum() > 0:
                newpred = preds.T[tie2].T
                newconf =conf.T[tie2].T
                pmode = torch.Tensor(predmode[tie2]).long().unsqueeze(0).to(outputs.device)
                conf_minority = (newconf * (newpred != pmode)).sum(dim=0)
                if mode == 'soft_vote_max':
                    conf_majority, _ = (newconf * (newpred == pmode)).max(dim=0)
                elif mode == 'soft_vote_sum':
                    conf_majority = (newconf * (newpred == pmode)).sum(dim=0)
                else:
                    raise "specify soft_vote_max or soft_vote_sum"
                ind_to_switch = conf_minority > conf_majority
                ensem_pred1 = newpred[(newpred != pmode)][ind_to_switch]
                ensem_pred2 = pmode.squeeze(0)[~ind_to_switch]
                ensem_preds = torch.cat([ensem_preds, ensem_pred1, ensem_pred2])
                new_target = torch.cat([new_target, target[tie2][ind_to_switch], target[tie2][~ind_to_switch]])
                print(f"switch fraction {len(ensem_pred1)/ (len(ensem_pred1)+len(ensem_pred2))}")

            tie = (count == 1)

            if tie.sum() > 0:
                tied_conf = conf.T[tie].T
                tied_preds = preds.T[tie].T
                # For samples under tie, choose the most confident model
                _, chosen_mem = tied_conf.max(dim=0)
                chosen_mem = chosen_mem.unsqueeze(0)
                tied_preds = torch.gather(tied_preds,0,chosen_mem).squeeze(0) # these are predictions with tie
                ensem_preds = torch.cat([ensem_preds, tied_preds],dim=0)
                new_target = torch.cat([new_target, target[tie]])
                print(f"tied fraction: {len(tied_preds)/len(ensem_preds)}")
            assert(ensem_preds.shape[0] == outputs.shape[1])
        return ensem_preds.long(), new_target, None, acc_ub * 100



def validate(model, val_loader, logger, indices_in_1k, epoch, args, best_acc):
    # evaluate
    model.eval()
    top1 = AverageMeter('Acc@1')
    top1_mems = [AverageMeter('Acc@1') for _ in range(args.num_ensem)]
    top5 = AverageMeter('Acc@5')
    top5_mems = [AverageMeter('Acc@5') for _ in range(args.num_ensem)]
    top1_ub = AverageMeter('Acc_UB@1')

    # per class accuracy
    if val_loader.dataset.__class__ != Split_Dataset:
        num_classes = len(val_loader.dataset.classes)
    else:
    # if args.eval_subset100 or args.use_smaller_split or args.use_smaller_split_val or (args.eval_var_subset is not None):
        num_classes = 1000
    correct_per_class = torch.zeros(num_classes, device='cuda')
    total_per_class = torch.zeros(num_classes, device='cuda')

    # only in ensemble freeze eval mode or single model freeze eval mode, calculate ECE
    calc_ece = False
    if args.eval_mode == 'freeze':
        # calc_ece = True # disable for now
        ece = AverageMeter('ECE')
        ece_mems = [AverageMeter('ECE') for _ in range(args.num_ensem)]
        ece_fn = ECELoss()

    with torch.no_grad():
        for images, target in tqdm(val_loader):
            target = target.cuda(non_blocking=True)
            output = model.forward(images.cuda(non_blocking=True))
            output = output[:, :, indices_in_1k] # subsetting the classes if it doesn't cover all 1000 in imagenet
            # output_mean = output.softmax(dim=-1).mean(dim=0)
            preds, new_target, output_mean, acc_ub = ensem_pred(output, mode=args.ensem_pred, target=target)

            if args.ensem_pred == 'DE':
                acc1, acc5 = accuracy(output_mean, target, topk=(1, 5))
                top5.update(acc5[0].item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
            else:
                acc1 = (preds == new_target).sum() / images.size(0) * 100
                top1.update(acc1.item(), images.size(0))
            top1_ub.update(acc_ub.item(), images.size(0))



            # per class accuracy
            correct_vec = (preds == new_target) # if each prediction is correct or not
            ind_per_class = (target.unsqueeze(1) == torch.arange(num_classes, device='cuda')) # indicator variable for each class
            correct_per_class += (correct_vec.unsqueeze(1) * ind_per_class).sum(0)
            total_per_class += ind_per_class.sum(0)

            if calc_ece:
                ece_loss = ece_fn(output_mean, target)
                ece.update(ece_loss.item(), images.size(0))

            for m in range(args.num_ensem):
                acc1, acc5 = accuracy(output[m], target, topk=(1, 5))
                top1_mems[m].update(acc1[0].item(), images.size(0))
                top5_mems[m].update(acc5[0].item(), images.size(0))

                if calc_ece:
                    ece_loss = ece_fn(output[m].softmax(dim=-1), target)
                    ece_mems[m].update(ece_loss.item(), images.size(0))

    # sanity check that the sum of total per class amounts to the whole dataset
    assert total_per_class.sum() == len(val_loader.dataset)
    acc_per_class = correct_per_class / total_per_class

    is_best = False
    if best_acc.top1 < top1.avg:
        best_acc.top1 = top1.avg
        best_acc.top5 = top5.avg

        if calc_ece:
            best_acc.ece = ece.avg
        is_best = True

    stats = dict(dataset=args.dataset, epoch=epoch, acc1=top1.avg, acc1_UB=top1_ub.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)

    logger.log_value('Acc1/ensemble', top1.avg, epoch)
    logger.log_value('Acc5/ensemble', top5.avg, epoch)

    if calc_ece:
        stats['best_ece'] = best_acc.ece
        logger.log_value('ECE/ensemble', ece.avg, epoch)

    for m in range(args.num_ensem):
        stats[f'mem{m}_acc1'] = top1_mems[m].avg
        stats[f'mem{m}_acc5'] = top5_mems[m].avg
        logger.log_value('Acc1/member{m}', top1_mems[m].avg, epoch)
        logger.log_value('Acc5/member{m}', top5_mems[m].avg, epoch)

        if calc_ece:
            stats[f'mem{m}_ece'] = ece_mems[m].avg

    stats_dump = json.dumps(stats)
    print(stats_dump)

    return stats_dump, acc_per_class, is_best


def make_sh_and_submit(args, delay=0):
    os.makedirs('./scripts/submit_scripts/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    options = args.arg_str
    if delay == 0:
        options_split = options.split(" ")[1:-2]
        name = ''.join([opt1.replace("--","").replace("=","") for opt1 in options_split])
        name = args.add_prefix + name

    else: # log_id should be already defined
        name = args.log_id
    print('Submitting the job with options: ')
    # print(options)
    print(f"experiment name: {name}")

    if args.server == 'aimos':
        options += f' --server=aimos --arg_str=\"{args.arg_str}\" '
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:2\n#SBATCH --exclusive\n#SBATCH --cpus-per-task=20\n#SBATCH '
            f'-N 1\n#SBATCH -t 360\n#SBATCH ')
        preamble += f'--begin=now+{delay}hour\n#SBATCH '
        preamble += (f'-o ./logs/{name}.out\n#SBATCH '
                        f'--job-name={name}_{delay}\n#SBATCH '
                        f'--open-mode=append\n\n')

    else:
        username = getpass.getuser()
        options += f' --server={args.server} '
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH '
            f'-o ./logs/{name}.out\n#SBATCH '
            f'--job-name={name}\n#SBATCH '
            f'--open-mode=append\n\n'
        )
    with open(f'./scripts/submit_scripts/{name}_{delay}.sh', 'w') as file:
        file.write(preamble)
        file.write("echo \"current time: $(date)\";\n")
        port = random.randrange(10000, 20000)
        file.write(
            f'python {sys.argv[0]} '
            # f'{options} --log_id={name} --multiprocessing-distributed --dist-url \'tcp://localhost:{port}\' '
            f'{options} --gpu 0 --exp-id={name} '
        )
        # if args.server == 'sc' or args.server == 'rumensc':
        #     if args.dataset == 'imagenet-100':
        #         file.write(f'--data=/home/gridsan/groups/MAML-Soljacic/imagenet100-new --dataset=imagenet-100 ')
        #     elif args.dataset == 'imagenet':
        #         file.write(f'--data=/home/gridsan/groups/datasets/ImageNet --dataset=imagenet ')
        #     else:
        #         file.write(f'--data=/home/gridsan/groups/MAML-Soljacic/ --dataset={args.dataset} ')
                # raise NotImplementedError(f'{args.dataset} path not specified correctly')
    os.system(f'sbatch ./scripts/submit_scripts/{name}_{delay}.sh')

if __name__ == '__main__':
    main()
