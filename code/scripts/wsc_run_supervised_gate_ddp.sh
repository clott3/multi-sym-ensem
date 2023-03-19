#!/bin/bash

script=$1
shift

python main_supervised_gate.py /gpfs/wscgpfs02/shivsr/slinks/imagenet_slink --dist-url "tcp://$MASTER_HOSTNAME:10596" \
    --dist-backend 'nccl' --multiprocessing-distributed --world-size $NODES --rank ${PMIX_RANK} -a mlp --dataset imagenet \
    --use-features --hidden-dim 1024 512 --num-classes 3 -b 252 --lr 0.1 --sampling-method classwise \
    --input-model-weights /gpfs/wscgpfs02/shivsr/ensem_ssl_main/models/IN1k-800e/simclr-base-seedR-IN1k-800e-resnet50.pth \
    /gpfs/wscgpfs02/shivsr/ensem_ssl_main/models/IN1k-800e/simclr-rotinv-seedR-IN1k-800e-resnet50.pth \
    /gpfs/wscgpfs02/shivsr/ensem_ssl_main/models/IN1k-800e/simclr-roteq-seedR-IN1k-800e-resnet50.pth \

$script "$@"
