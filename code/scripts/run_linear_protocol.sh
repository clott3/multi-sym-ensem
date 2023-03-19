#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/scratch/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME2/scratch/ssl/imagenet/simclr/launcher_linear.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/ssl/imagenet/simclr/saved_models/SimCLR-default/checkpoint_epoch100.pth \
      --checkpoint-dir $HOME2/scratch/ssl/imagenet/simclr/saved_models/ \
      --log-dir $HOME2/scratch/ssl/imagenet/simclr/logs/ \
      --nodes 1 \
      --exp SimCLR-default-linear

echo "Run completed at:- "
date

