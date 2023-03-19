#!/bin/bash

module load anaconda/2022a

python eval_ensem.py --submit --server=sc --dataset=$1 --arg_str="--add_prefix $2 --dataset $1 --eval-mode log_reg --batch-size=256 --lr-backbone 0.0 --lr-classifier 0.0 --world-size 1 --rank 0 --pretrained $3"
