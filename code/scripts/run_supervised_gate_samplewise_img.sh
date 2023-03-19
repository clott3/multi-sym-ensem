#!/bin/sh

python main_supervised_gate.py /data/scratch/swhan/data/imagenet-100/ -a resnet18 --dataset imagenet-100 -b 256 --lr 0.1 --gpu 1 --rank 0 --num-classes 4 --sampling-method samplewise-single
