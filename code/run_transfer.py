#!/usr/bin/python

import os

model_root = '/gpfs/wscgpfs02/shivsr/ensem_ssl/checkpoints'
models = ['simclr-vflip-lr6.4_resnet50.pth', 'simclr-vflipeq-lr6.4_resnet50.pth', 'simclr-stylizeinv-lr6.4_resnet50.pth',
          'simclr-rotinv-lr6.4_resnet50.pth', 'simclr-roteq-lr6.4_resnet50.pth', 'simclr-invert-lr6.4_resnet50.pth',
          'simclr-inverteq-lr6.4_resnet50.pth', 'simclr-halfswapeq-lr6.4-3_checkpoint_99.pth',
          'simclr-halfswap-lr6.4_resnet50.pth', 'simclr-base-lr6.4-seed7_resnet50.pth',
          'simclr-base-lr6.4-seed69_resnet50.pth', 'simclr-base-lr6.4-seed11_resnet50.pth',
          'simclr-base-lr6.4-seed22_checkpoint_99.pth', 'simclr-base-lr6.4-2_resnet50.pth']
dirs = list(map(lambda x: x.replace('_resnet50.pth', ''), models))
dirs = list(map(lambda x: x.replace('_checkpoint_99.pth', ''), dirs))
dataset = 'food'
lr = 1.0
suffix = 'nov9'


for d, m in zip(dirs, models):

    path = f'{model_root}/{d}/{m}'
    exp_id = f'{d}-lp-{dataset}-lr{lr}'
    cmd = f'python3 wsc_submit.py -n 1 -j {exp_id} -t 70 --suffix {suffix} --model {exp_id} "./run_eval-transfer.sh ' \
          f'--dataset={dataset} --exp-id {exp_id} --lr-classifier {lr} --pretrained {path}"'

    os.system(cmd)