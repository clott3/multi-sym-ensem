#!/usr/bin/python

import itertools

from subprocess import Popen


models = {
    'baseline': '/data/scratch/swhan/data/lp/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'baseline2': '/data/scratch/swhan/data/lp/simclr-baseline2datasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'baseline3': '/data/scratch/swhan/data/lp/simclr-baseline3datasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'baseline4': '/data/scratch/swhan/data/lp/simclr-baseline4datasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'baseline5': '/data/scratch/swhan/data/lp/simclr-baseline5datasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'inverteq': '/data/scratch/swhan/data/lp/simclr-invertdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'roteq': '/data/scratch/swhan/data/lp/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'halfswapeq': '/data/scratch/swhan/data/lp/simclr-halfswapeqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier1.0lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'vflipeq': '/data/scratch/swhan/data/lp/simclr-vflipeqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'invertinv': '/data/scratch/swhan/data/lp/simclr-invertinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'rotinv': '/data/scratch/swhan/data/lp/simclr-rotinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'halfswapinv': '/data/scratch/swhan/data/lp/simclr-halfswapinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'vflipinv': '/data/scratch/swhan/data/lp/simclr-vflipinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth',
    'styleinv': '/data/scratch/swhan/data/lp/simclr-styleinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth'
}
bases = ['invert', 'rot', 'halfswap', 'vflip']
perms = list(itertools.combinations(models.keys(), 3))

perms = [['baseline', bases[i]+'eq', bases[i]+'inv'] for i in range(len(bases))]

for dataset in ['imagenet-100', 'imagenet-100-a', 'imagenet-100-r', 'imagenet-100-sketch']:
    processes = []
    for i in range(len(perms)):
        device = 0
        perm = perms[i]
        f = open(f'./logs_ood/simclr_{dataset}_{perm[0]}_{perm[1]}_{perm[2]}.txt', "w")
        command = f'eval_ensem.py --data /data/scratch/swhan/data/ --dataset {dataset} \
            --eval-mode freeze --combine_sep_ckpts --pretrained {models[perm[0]]} {models[perm[1]]} {models[perm[2]]} \
            --lr-backbone 0 --lr-classifier 0 --val-batch-size 256 \
            --exp-id simclr_{dataset}_{perm[0]}_{perm[1]}_{perm[2]} --gpu {device}'

        command = ['python3'] + command.split()
        # processes.append(Popen(command, stdout=f))
        p = Popen(command, stdout=f)
        
        print(f'Queued {dataset} {perm[0]} {perm[1]} {perm[2]}')
        print(f'Running ood evaluation!')
        p.wait()
