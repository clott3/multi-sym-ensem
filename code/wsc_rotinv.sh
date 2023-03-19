#!/bin/bash

script=$1
shift

echo python wsc_launcher.py --dist-url "tcp://$MASTER_HOSTNAME:10596" --dist-backend 'nccl' --multiprocessing-distributed --world-size $NODES --rank ${PMIX_RANK} --data=/gpfs/wscgpfs02/shivsr/slinks/imagenet_slink --workers 60 --epochs 800 --batch-size 4080 --learning-rate 4.8 --checkpoint-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/checkpoint/rotinv/ --log-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/logs/ --rotation 0.0 --exp SimCLR-rotinv --rotinv $script "$@"


python wsc_launcher.py --dist-url "tcp://$MASTER_HOSTNAME:10596" --dist-backend 'nccl' --multiprocessing-distributed --world-size $NODES --rank ${PMIX_RANK} --data=/gpfs/wscgpfs02/shivsr/slinks/imagenet_slink --workers 60 --epochs 800 --batch-size 4080 --learning-rate 4.8 --checkpoint-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/checkpoint/rotinv/ --log-dir /gpfs/wscgpfs02/shivsr/ensem_ssl/logs/ --rotation 0.0 --exp SimCLR-rotinv --rotinv $script "$@"

