#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WANDB_ENTITY=alelab

MODE="$1"
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    src/train.py --config-name "train/dpo_kl/${MODE}"