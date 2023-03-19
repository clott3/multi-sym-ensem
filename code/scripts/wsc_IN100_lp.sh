#!/bin/bash

script=$1
shift

python eval_ensem.py \
    --dataset imagenet-100 \
    --data /gpfs/wscgpfs02/shivsr/datasets/imagenet-100 \
    --dist-url "tcp://$MASTER_HOSTNAME:10596" \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size $NODES \
    --rank ${PMIX_RANK} \
    --eval-mode linear_probe \
    --lr-classifier 0.3 \
    --lr-backbone 0.0 \
    --lr-scheduler cosine \
    --workers 32 \
    --batch-size 258 \
    --exp-id hseq-IN100-e100-seed69-lp-cos-lr0.3-bs258 \
    --pretrained /gpfs/wscgpfs02/shivsr/ensem_ssl/checkpoints/halfswapeq-seed31-lr6.4/checkpoint_epoch99.pth

$script "$@"