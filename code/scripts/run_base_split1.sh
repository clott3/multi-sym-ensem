#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for seed in 69 31 42
do
python \
      $HOME2/scratch/ensem_ssl/launcher.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --epochs 400 \
      --batch-size 4096 \
      --learning-rate 4.8 \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/saved_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/saved_logs/ \
      --rotation 0.0 \
      --ngpus-per-node 4 \
      --nodes 8 \
      --exp simclr-base-seed${seed}-IN1k_400e_bs4096_split1 \
      --seed ${seed} \
      --train_val_split 1
done

echo "Run completed at:- "
date
