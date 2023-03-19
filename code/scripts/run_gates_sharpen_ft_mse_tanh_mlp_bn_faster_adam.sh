#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 1e-3 1e-2 0.1 1e-4
do
for wd in 5e-4
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem_gate.py \
      --server aimos \
      --gate_arch mlp_bn \
      --use_eps \
      --gate frozen \
      --lr-gate ${lr} \
      --dataset imagenet \
      --combine_sep_ckpts \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth $HOME2/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth $HOME2/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id INft2_gate_mlp_bn_lr${lr}_bs256_on5perc_wd${wd}_mse_tanh_adam \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \
      --val_perc 5 \
      --use_smaller_split_val \
      --weight-decay ${wd} \
      --gate_loss mse_tanh \
      --validate_freq 20 \
      --optim adam
done
done
echo "Run completed at:- "
date
