#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for lr in 0.001 0.01
do
for lmbd in 0.01 0.001
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem_gate.py \
      --server aimos \
      --gate_arch smallmlp \
      --smallmlp_hd 256 \
      --me_max \
      --lmbd ${lmbd} \
      --use_eps \
      --lr-scheduler cosine \
      --gate frozen \
      --lr-gate ${lr} \
      --dataset imagenet \
      --combine_sep_ckpts \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth $HOME2/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth $HOME2/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id IN_gate_smallmlp256_memax_lmbd${lmbd}_lr${lr}_bs256 \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4
done
done
echo "Run completed at:- "
date
