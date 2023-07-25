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
      $HOME/slurm_launcher.py \
      --data <path-to-imagenet-folder> \
      --checkpoint-dir $HOME/saved_models/ \
      --log-dir $HOME/logs/ \
      --lmbd 0.4 \
      --ngpus-per-node 4 \
      --nodes 8 \
      --exp SimCLR-rotinv \
      --rotate eq
      
echo "Run completed at:- "
date
