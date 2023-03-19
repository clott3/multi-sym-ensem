#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for seed in 69
do
python \
      $HOME2/scratch/ensem_ssl/launcher_wsc_main.py \
      --data /gpfs/u/home/BNSS/BNSSlhch/scratch/imagenet-100-classes/ \
      --workers 32 \
      --epochs 100 \
      --batch-size 4064 \
      --learning-rate 6.4 \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/saved_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/saved_logs/ \
      --lmbd 0.0 \
      --ngpus-per-node 4 \
      --nodes 8 \
      --exp simclr-halfswapInv-seed${seed}-oldIN100_100e_bs4064 \
      --seed ${seed} \
      --tfm halfswap \
      --tfm_mode inv
done

echo "Run completed at:- "
date
