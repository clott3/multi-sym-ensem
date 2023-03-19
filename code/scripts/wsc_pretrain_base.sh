#!/bin/bash

script=$1
shift

python wsc_launcher.py --dist-url "tcp://$MASTER_HOSTNAME:10596" --dist-backend 'nccl' --multiprocessing-distributed \
                --world-size $NODES --rank ${PMIX_RANK} --data=/gpfs/wscgpfs02/shivsr/slinks/imagenet_slink --workers 32 \
                --batch-size 4080 --epochs 800 --checkpoint-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/checkpoints \
                --log-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/logs $script "$@"
