#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 5e-3 1e-2 0.1
do
for lmbd in 0.0
do
for hd in 512 1024 2048
do
for shp in 1.0
do
for wd in 5e-4
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem_gate.py \
      --server aimos \
      --gate_arch smallmlp \
      --smallmlp_hd ${hd} \
      --me_max \
      --lmbd ${lmbd} \
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
      --exp-id INft_gate_smallmlp${hd}_lmbd${lmbd}_lr${lr}_bs256_on5perc_shp${shp}_wd${wd}_mse_tanh \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \
      --val_perc 5 \
      --use_smaller_split_val \
      --weight-decay ${wd} \
      --sharpen_T ${shp} \
      --gate_loss mse_tanh
done
done
done
done
done
echo "Run completed at:- "
date
