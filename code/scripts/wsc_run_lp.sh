#!/bin/bash

script=$1
shift

python eval_ensem.py --dataset imagenet --data /gpfs/wscgpfs02/shivsr/slinks/imagenet_slink --dist-url "tcp://$MASTER_HOSTNAME:10596" \
    --dist-backend 'nccl' --multiprocessing-distributed --world-size $NODES --rank ${PMIX_RANK} --num-ensem 1 --eval-mode linear_probe \
    --pretrained /gpfs/wscgpfs02/shivsr/ensem_ssl_main/models/IN1k-800e/simclr-roteq-seed69-IN1k-800e-resnet50.pth \
    --lr-classifier 0.3 --lr-backbone 0.0 --lr-scheduler cosine --exp-id simclr-roteq-seed69-ep800-lp-lr0.3-cosine --batch-size 252

$script "$@"
