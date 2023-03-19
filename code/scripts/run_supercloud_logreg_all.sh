#!/bin/bash

module load anaconda/2022a

model_names=(simclr-baseline simclr-baseline2 simclr-baseline3 simclr-baseline4 simclr-baseline5 simclr-rotinv simclr-vflipinv simclr-halfswapinv simclr-invertinv simclr-styleinv simclr-roteq simclr-vflipeq simclr-halfswapeq simclr-inverteq)
model_paths=(simclr-base-lr6.4-resnet50.pth simclr-base-lr6.4-seed7_resnet50.pth simclr-base-lr6.4-seed11_resnet50.pth simclr-base-lr6.4-seed22_checkpoint_ep99.pth simclr-base-lr6.4-seed54_resnet50.pth simclr-rotinv-lr6.4_resnet50.pth simclr-vflipinv-lr6.4_resnet50.pth simclr-halfswapinv-lr6.4_resnet50.pth simclr-invertinv-lr6.4_resnet50.pth simclr-stylizeinv-lr6.4_resnet50.pth simclr-roteq-lr6.4-lamb0.4_resnet50.pth simclr-vflipeq-lr6.4-lamb0.4_resnet50.pth simclr-halfswapeq-lr6.4-lamb0.1_resnet50.pth simclr-inverteq-lr6.4-lamb0.4_resnet50.pth)
len=${#model_names[@]}
lenother=${#model_paths[@]}
if [ "$len" -ne "$lenother" ]; then
    echo "Length not equal"
    exit 1
fi
for ((i=0;i<$len;i++))
do
python eval_ensem.py --submit --server=sc --dataset=$1 --arg_str="--add_prefix ${model_names[$i]} --dataset $1 --eval-mode log_reg --batch-size=256 --lr-backbone 0.0 --lr-classifier 0.0 --world-size 1 --rank 0 --pretrained ../pretrain_checkpoints/${model_paths[$i]}"
done