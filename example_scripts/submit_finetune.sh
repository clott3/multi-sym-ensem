#!/bin/bash

## User python environment
HOME=<set-path-to-multi-sym-ensem-folder>
PYTHON_VIRTUAL_ENVIRONMENT=<name-of-conda-env>
CONDA_ROOT=<path-to-conda>

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME/slurm_launcher_ft.py \
      --data <path-to-imagenet-folder> \
      --eval-mode finetune \
      --lr-classifier 0.003 \
      --lr-backbone 0.003 \
      --dataset imagenet \
      --pretrained $HOME/ensem_pt_checkpoints/simclr-roteq-seed69-IN1k-800e-resnet50.pth \
      --checkpoint-dir $HOME/ft_models/ \
      --log-dir $HOME/ft_logs/ \
      --stats-dir $HOME/ft_stats/ \
      --exp-id ft_roteq69_lr0.003 \
      --ngpus-per-node 4 \
      --nodes 2 \
      
echo "Run completed at:- "
date
