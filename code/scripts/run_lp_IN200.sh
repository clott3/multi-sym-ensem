#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSlhch
PYTHON_VIRTUAL_ENVIRONMENT=py37
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

for seed in 31 42
do
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem.py \
      --server aimos \
      --combine_sep_ckpts \
      --eval-mode linear_probe \
      --lr-classifier 0.3 \
      --lr-backbone 0.0 \
      --dataset imagenet \
      --lr-scheduler cosine \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/saved_models/simclr-roteq-seed${seed}-IN200_400e_bs4096/checkpoint.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id roteq-IN200-e400-seed${seed}-lp-cos-lr0.3-bs256 \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \
      --use_smaller_split \
      --val_perc 200;
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem.py \
      --server aimos \
      --combine_sep_ckpts \
      --eval-mode linear_probe \
      --lr-classifier 0.3 \
      --lr-backbone 0.0 \
      --dataset imagenet \
      --lr-scheduler cosine \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/saved_models/simclr-base-seed${seed}-IN200_400e_bs4096/checkpoint.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id base-IN200-e400-seed${seed}-lp-cos-lr0.3-bs256 \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \
      --use_smaller_split \
      --val_perc 200;      
python \
      $HOME2/scratch/ensem_ssl/launcher_eval_ensem.py \
      --server aimos \
      --combine_sep_ckpts \
      --eval-mode linear_probe \
      --lr-classifier 0.3 \
      --lr-backbone 0.0 \
      --dataset imagenet \
      --lr-scheduler cosine \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ \
      --workers 32 \
      --pretrained $HOME2/scratch/saved_models/simclr-inv-seed${seed}-IN200_400e_bs4096/checkpoint.pth \
      --checkpoint-dir $HOME2/scratch/ensem_ssl/dist_models/ \
      --log-dir $HOME2/scratch/ensem_ssl/dist_logs/ \
      --stats-dir $HOME2/scratch/ensem_ssl/dist_stats/ \
      --exp-id rotinv-IN200-e400-seed${seed}-lp-cos-lr0.3-bs256 \
      --batch-size 256 \
      --nodes 2 \
      --ngpus-per-node 4 \
      --use_smaller_split \
      --val_perc 200;
done

echo "Run completed at:- "
date
