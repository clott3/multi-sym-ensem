#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME2/scratch/ensem_ssl/launcher.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --epochs 800 \
      --batch-size 4096 \
      --learning-rate 4.8 \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/saved_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/logs/ \
      --rotation 0.0 \
      --ngpus-per-node 4 \
      --nodes 8 \
      --exp SimCLR-rotinv \
      --rotinv
      
echo "Run completed at:- "
date
