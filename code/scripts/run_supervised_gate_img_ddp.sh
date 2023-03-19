#!/bin/sh

port=$(($RANDOM+10000))
python main_supervised_gate.py /data/scratch/swhan/data/imagenet/ -a resnet50 --dataset imagenet \
    --num-classes 3 -b 256 --lr 0.1 --sampling-method classwise \
    --input-model-weights ./checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth \
    ./checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth ./checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth \
    --dist-url tcp://127.0.0.1:${port} --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0