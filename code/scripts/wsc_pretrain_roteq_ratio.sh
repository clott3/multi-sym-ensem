#!/bin/bash

script=$1
shift

python wsc_launcher.py --dist-url "tcp://$MASTER_HOSTNAME:10596" --dist-backend 'nccl' --multiprocessing-distributed \
                --world-size $NODES --rank ${PMIX_RANK} --data=/gpfs/wscgpfs02/shivsr/slinks/imagenet_slink --workers 32 \
                --batch-size 4050 --lmbd 0.4 --rotate=eq --epochs 200 --checkpoint-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/checkpoints \
                --log-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/logs --training-ratio 0.50 --learning-rate 6.4 $script "$@"
