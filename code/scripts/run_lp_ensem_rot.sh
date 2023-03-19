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
      $HOME2/scratch/ensem_ssl/launcher_linear.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/checkpoints/10-16-simclr-rot-0.0-800ep-resnet50.pth $HOME2/scratch/checkpoints/10-18-simclr-rot-inv-800ep-resnet50.pth $HOME2/scratch/checkpoints/11-26-simclr-rot-0.4-800ep-resnet50.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/saved_models/ \
      --log-dir $HOME2/ensem_ssl/logs/ \
      --nodes 1 \
      --ngpus-per-node 4 \
      --exp-id ensem_lp_rot_800ep \
      --exp ensem_lp_rot_800ep \
      --weights freeze \
      --lr-classifier 1.0 


echo "Run completed at:- "
date
