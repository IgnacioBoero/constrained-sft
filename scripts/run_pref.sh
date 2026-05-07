#!/bin/bash
# When2Call / DPO_KL training. Configs live under configs/train/paper_experiments/function_calling/.
# Examples:
#   ./scripts/run_pref.sh llama/dpo
#   ./scripts/run_pref.sh sweep/simpo
#   ./scripts/run_pref.sh xlam/resilient_both

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WANDB_ENTITY=alelab

SUBPATH="${1:?usage: $0 <path under paper_experiments/function_calling/>, e.g. llama/dpo or sweep/dpo}"
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    src/train.py --config-name "train/paper_experiments/function_calling/${SUBPATH}"
