#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python eval_ensem.py --dataset imagenet --data /data/scratch/swhan/data/ --pretrained \
	./ensem_checkpoints/10-18-simclr-rot-inv-800ep-resnet50.pth --num-ensem 1 --eval-mode freeze \
	--exp-id simclr-rotinv-ep800-acc-per-class --gpu 0
# ./ensem_checkpoints/10-16-simclr-rot-0.0-800ep-resnet50.pth
# ./ensem_checkpoints/11-26-simclr-rot-0.4-800ep-resnet50.pth