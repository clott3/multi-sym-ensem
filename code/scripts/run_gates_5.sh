#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.0003 0.001 0.00001 0.01 0.1
do
for lmbd in 0.0
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem_gate.py \
      --server aimos \
      --gate_arch vit_small \
      --me_max \
      --lmbd ${lmbd} \
      --use_eps \
      --gate frozen \
      --lr-gate ${lr} \
      --dataset imagenet \
      --combine_sep_ckpts \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth $HOME2/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth $HOME2/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id IN_gate_vitsmall_lmbd${lmbd}_lr${lr}_bs256_on5perc \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \
      --val_perc 5 \
      --use_smaller_split_val \
      --optim adam
done
done
echo "Run completed at:- "
date
