#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python eval_ensem.py --dataset imagenet --data /data/scratch/swhan/data/imagenet/ --pretrained \
	./ensem_checkpoints/10-16-simclr-rot-0.0-800ep-resnet50.pth \
	 --num-ensem 1 --eval-mode linear_probe \
	--exp-id simclr-baseline-ep800-lp-lr0.3-cosine --multiprocessing-distributed --lr-classifier 0.3 --lr-backbone 0.0 --lr-scheduler cosine

#./ensem_checkpoints/10-18-simclr-rot-inv-800ep-resnet50.pth
#./ensem_checkpoints/11-26-simclr-rot-0.4-800ep-resnet50.pth