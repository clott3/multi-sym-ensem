#!/usr/bin/python

import itertools
from subprocess import Popen
import argparse

parser = argparse.ArgumentParser(description='Run OOD / transfer ensembling evaluations')
parser.add_argument('dataset', type=str, default='imagenet',
                    choices=('imagenet', 'imagenet-a', 'imagenet-r',
                    'imagenet-v2', 'imagenet-sketch', 'imagenet-100',
                    'imagenet-100-a', 'imagenet-100-r', 'imagenet-100-v2',
                    'imagenet-100-sketch', 'inat-1k', 'cub-200', 'flowers-102',
                    'food', 'cifar10', 'cifar100', 'pets', 'sun-397', 'cars',
                    'aircraft', 'voc-2007', 'dtd', 'caltech-101'),
                    help='dataset name')
parser.add_argument('--stats-filename', type=str, default='results.txt')

args = parser.parse_args()


dataset_lrs = {
    'cifar10': 1.0,
    'cifar100': 1.0,
    'cars': 10.0,
    'cub-200': 4.0,
}

models = {
    'baseline':     f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-baselinedataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'baseline2':    f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-baseline2dataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'baseline3':    f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-baseline3dataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'baseline4':    f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-baseline4dataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'baseline5':    f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-baseline5dataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'inverteq':     f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-inverteqdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'roteq':        f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-roteqdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'halfswapeq':   f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-halfswapeqdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'vflipeq':      f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-vflipeqdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'invertinv':    f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-invertinvdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'rotinv':       f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-rotinvdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'halfswapinv':  f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-halfswapinvdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'vflipinv':     f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-vflipinvdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
    'styleinv':     f'/data/scratch/swhan/data/lp/{args.dataset}/simclr-styleinvdataset{args.dataset}eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier{dataset_lrs[args.dataset]}lr-schedulercosineworld-size1rank0/checkpoint_best.pth',
}
bases = ['invert', 'rot', 'halfswap', 'vflip']

perms = list(itertools.combinations(models.keys(), 3))
dataset = args.dataset

perms = [['baseline3', bases[i]+'eq', bases[i]+'inv'] for i in range(len(bases))]
processes = []
for i in range(len(perms)):
    device = 0
    perm = perms[i]
    f = open(f'./logs_ood/simclr_{dataset}_{perm[0]}_{perm[1]}_{perm[2]}.txt', "w")
    command = f'eval_ensem.py --data /data/scratch/swhan/data/ --dataset {dataset} \
        --eval-mode freeze --combine_sep_ckpts --pretrained {models[perm[0]]} {models[perm[1]]} {models[perm[2]]} \
        --lr-backbone 0 --lr-classifier 0 --val-batch-size 256 \
        --exp-id simclr_{dataset}_{perm[0]}_{perm[1]}_{perm[2]} --gpu {device} \
        --stats-filename {args.stats_filename}'

    command = ['python3'] + command.split()
    # processes.append(Popen(command, stdout=f))
    p = Popen(command, stdout=f)
    
    print(f'Queued {dataset} {perm[0]} {perm[1]} {perm[2]}')
    print(f'Running ood evaluation!')
    p.wait()
