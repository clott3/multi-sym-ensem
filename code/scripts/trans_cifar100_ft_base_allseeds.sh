#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for seed in 24 31 69
do
for lr in 0.1 0.2 0.5 0.8
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem.py \
      --server aimos \
      --combine_sep_ckpts \
      --eval-mode finetune \
      --lr-classifier ${lr} \
      --lr-backbone ${lr} \
      --dataset cifar100 \
      --lr-scheduler cosine \
      --data /gpfs/u/home/BNSS/BNSSlhch/scratch/datasets/ \
      --workers 32 \
      --pretrained $HOME2/scratch/ensem_pt_checkpoints/base-IN1k-e800-seed${seed}-checkpoint_epoch799.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id trans_cifar100_base${seed}_cos_lr${lr}_bs256 \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4
done
done
echo "Run completed at:- "
date