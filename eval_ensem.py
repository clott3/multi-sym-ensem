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

from networks import EnsembleSSL
from datasets import build_dataloaders, dataset_num_classes
from utils import accuracy, AverageMeter
import scipy as sp


parser = argparse.ArgumentParser(description='Finetune or evaluate pretrained models/ensemble')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=('imagenet', 'imagenet-r',
                    'imagenet-v2', 'inat-1k', 'flowers-102',
                    'food', 'cifar10', 'cifar100'),
                    help='dataset name')
parser.add_argument('--pretrained', nargs="+", default=[], type=str,
                    help='paths to pretrained models')
parser.add_argument('--eval-mode', default='freeze', type=str,
                    choices=('finetune', 'linear_probe', 'freeze'),
                    help='finetune, linear probe, or freeze resnet weights')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--val-batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size for validation (uses only 1 gpu)')
parser.add_argument('--lr-backbone', default=0.003, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.003, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--lr-scheduler', default='cosine', type=str, metavar='LR-SCH',
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
                    choices=('resnet50'),
                    help='architecture for each member in the ensemble')
parser.add_argument('--convert', action='store_true', default=False,
                    help='Whether to convert from single MultiBackbone \
                    checkpoint to use with EnsembleSSL')

# misc
parser.add_argument('--seed', default=None, type=int, metavar='S',
                    help='random seed')
parser.add_argument('--exp-id', default='', type=str,
                    help='Experiment ID for saving the outputs/checkpoints into a file')
parser.add_argument('--stats-filename', default='results.txt', type=str,
                    help='Stats filename to aggregate all results')
parser.add_argument('--save_freq', default=20, type=int)


def main():
    args = parser.parse_args()

    args.checkpoint_dir = Path('./checkpoints/') / args.exp_id
    args.log_dir = Path('./logs/') / args.exp_id
    args.stats_dir = args.log_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)

    # single node distributed training
    args.ngpus_per_node = torch.cuda.device_count()
    if args.dist_url == '':
        args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'

    args.rank = 0

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
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    os.makedirs(args.stats_dir, exist_ok=True)
    
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

    if args.distributed:
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    print(' '.join(sys.argv))

    if args.rank == 0:
        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    args.ds_num_classes = dataset_num_classes[args.dataset]

    # automatically resume from checkpoint if it exists (for restarting of fine-tuning)
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        print('Resuming from previous checkpoint')
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        num_ensem = ckpt['num_ensem']
        args.num_ensem = num_ensem
        model = EnsembleSSL(args.arch, num_ensem, args.ds_num_classes, args.eval_mode).cuda()
        model.load_state_dict(ckpt['model'])
    else:
        num_ensem = len(args.pretrained)
        args.num_ensem = num_ensem
        print(f'Combining separate checkpionts where each EnsembleSSL was trained separately: {num_ensem} models')
        model = EnsembleSSL(args.arch, num_ensem, args.ds_num_classes, args.eval_mode).cuda()
        start_epoch = 0
        model.load_weights(args.pretrained)

    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if 'fc.weight' in name or 'fc.bias' in name:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # set optimizer
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

    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        optimizer.load_state_dict(ckpt['optimizer'])

    best_acc = argparse.Namespace(top1=0, top5=0)

    # create dataloaders
    train_loader, test_loader, indices_in_1k = build_dataloaders(args)

    args.start_time = time.time()

    # if freeze weights and evaluate, then no need for training -- reset start_epoch & epochs
    if args.eval_mode == 'freeze':
        start_epoch = 0
        args.epochs = 1

    is_best = False

    for epoch in range(start_epoch, args.epochs):
        if args.eval_mode == 'finetune' or args.eval_mode == 'linear_probe':
            print(f'Epoch {epoch} training: \n')
            train(model, train_loader, optimizer, scheduler, criterion, epoch, args)

        # validate & save checkpoint
        if args.rank == 0:
            print(f'Epoch {epoch} validation: \n')
            if test_loader:
                stats_json, acc_per_class, is_best = validate(model, test_loader, logger, indices_in_1k, epoch, args, best_acc)

            # only save per epoch weights & results if training is being done
            if args.eval_mode != 'freeze':
                if args.distributed:
                    state = dict(epoch=epoch + 1, model=model.module.state_dict(),
                                    optimizer=optimizer.state_dict(), num_ensem=num_ensem)
                else:
                    state = dict(epoch=epoch + 1, model=model.state_dict(),
                                    optimizer=optimizer.state_dict(), num_ensem=num_ensem)

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
                torch.save({'acc_per_class': acc_per_class}, args.checkpoint_dir / 'acc_per_class_val.pth')


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


def validate(model, val_loader, logger, indices_in_1k, epoch, args, best_acc):

    # evaluate
    model.eval()
    top1 = AverageMeter('Acc@1')
    top1_mems = [AverageMeter('Acc@1') for _ in range(args.num_ensem)]
    top5 = AverageMeter('Acc@5')
    top5_mems = [AverageMeter('Acc@5') for _ in range(args.num_ensem)]

    num_classes = args.ds_num_classes
    print("num classes: ", num_classes)
  
    correct_per_class = torch.zeros(num_classes, device='cuda')
    total_per_class = torch.zeros(num_classes, device='cuda')
    
    iii = 0
    collect_predictions = []
    if args.num_ensem > 1:
        collect_ps = {}
        for i in range(args.num_ensem):
            collect_ps[i] = []

    collect_targets = []
    with torch.no_grad():
        for images, target in tqdm(val_loader):
            iii += 1
            target = target.cuda(non_blocking=True)

            output = model.forward(images.cuda(non_blocking=True))

            output = output[:, :, indices_in_1k] # subsetting the classes if it doesn't cover all 1000 in imagenet
            output_mean = output.softmax(dim=-1).mean(dim=0)
            sys.stdout.flush()
            acc1, acc5 = accuracy(output_mean, target, topk=(1, 5))
            top5.update(acc5[0].item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            collect_predictions.append(output_mean)
            collect_targets.append(target)
            if args.num_ensem > 1:
                for i in range(args.num_ensem):
                    collect_ps[i].append(output[i])

            # per class accuracy
            _, preds = output_mean.max(1)
            correct_vec = (preds == target) # if each prediction is correct or not
            ind_per_class = (target.unsqueeze(1) == torch.arange(num_classes, device='cuda')) # indicator variable for each class
            correct_per_class += (correct_vec.unsqueeze(1) * ind_per_class).sum(0)
            total_per_class += ind_per_class.sum(0)

            for m in range(args.num_ensem):
                acc1, acc5 = accuracy(output[m], target, topk=(1, 5))
                top1_mems[m].update(acc1[0].item(), images.size(0))
                top5_mems[m].update(acc5[0].item(), images.size(0))

    # save predictions for further analysis
    cp = torch.cat(collect_predictions).cpu()
    ct = torch.cat(collect_targets).cpu()
    torch.save(cp, args.checkpoint_dir / 'all_predictions.pth')
    torch.save(ct, args.checkpoint_dir / 'all_targets.pth')
    if args.num_ensem > 1:
        for i in range(args.num_ensem):
            cp_i = collect_ps[i]
            ckpt_name = (args.pretrained[i]).split("/")[-2]
            torch.save(cp_i, args.checkpoint_dir / f'{ckpt_name}_predictions.pth')


    # sanity check that the sum of total per class amounts to the whole dataset
    assert total_per_class.sum() == len(val_loader.dataset)
    acc_per_class = correct_per_class / total_per_class

    is_best = False
    if best_acc.top1 < top1.avg:
        best_acc.top1 = top1.avg
        best_acc.top5 = top5.avg
        is_best = True

    stats = dict(dataset=args.dataset, epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5,
                )

    logger.log_value('Acc1/ensemble', top1.avg, epoch)
    logger.log_value('Acc5/ensemble', top5.avg, epoch)

    for m in range(args.num_ensem):
        stats[f'mem{m}_acc1'] = top1_mems[m].avg
        stats[f'mem{m}_acc5'] = top5_mems[m].avg
        logger.log_value('Acc1/member{m}', top1_mems[m].avg, epoch)
        logger.log_value('Acc5/member{m}', top5_mems[m].avg, epoch)

    stats_dump = json.dumps(stats)
    print(stats_dump)

    return stats_dump, acc_per_class, is_best

if __name__ == '__main__':
    main()
