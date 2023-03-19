#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.3
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem.py \
      --server aimos \
      --combine_sep_ckpts \
      --eval-mode linear_probe \
      --lr-classifier ${lr} \
      --dataset imagenet \
      --use_smaller_split \
      --val_perc 10 \
      --lr-scheduler cosine \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id lp90perc_roteq_cos_lr${lr}_bs256 \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \

done

echo "Run completed at:- "
date
