#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4 python eval_ensem.py --dataset inat-1k --data /data/scratch/swhan/data/inat-1k/ --pretrained \
	./ensem_checkpoints/roteq-IN1k-e800-seed69-checkpoint_epoch799.pth --num-ensem 1 --eval-mode linear_probe \
	--exp-id inat-roteq-seed69-lp-lr$1-cosine --multiprocessing-distributed --lr-classifier $1 --lr-backbone 0.0 --lr-scheduler cosine

