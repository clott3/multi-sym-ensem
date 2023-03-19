#!/bin/sh

python main_supervised_gate.py /data/scratch/swhan/data/imagenet-100/ -a mlp --dataset imagenet-100 \
    --use-features --hidden-dim 1024 512 --num-classes 4 -b 256 --lr 0.1 --gpu 0 --rank 0 --sampling-method samplewise-single \
    --input-model-weights ./checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth \
    ./checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth ./checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth
