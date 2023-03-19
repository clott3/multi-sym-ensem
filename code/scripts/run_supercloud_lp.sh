#!/bin/bash

module load anaconda/2022a

#for lr in 0.3 0.5 0.7 1.0 5.0 10.0
for lr in 0.3 1.0
do
python eval_ensem.py --submit --server=sc --dataset=$1 --arg_str="--add_prefix $2 --dataset $1 --eval-mode linear_probe --batch-size=256 --lr-backbone 0.0 --lr-classifier ${lr} --lr-scheduler cosine --world-size 1 --rank 0 --pretrained $3"
done
