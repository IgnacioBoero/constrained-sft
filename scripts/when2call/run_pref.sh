#!/usr/bin/env bash
# When2Call / dpo_kl training (distributed). Configs: configs/train/paper_experiments/function_calling/.
# Run from repo root, or invoke directly (script cds to repo root).
#
# Examples:
#   ./scripts/when2call/run_pref.sh llama/dpo
#   ./scripts/when2call/run_pref.sh sweep/simpo
#   ./scripts/when2call/run_pref.sh xlam/resilient_both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WANDB_ENTITY="${WANDB_ENTITY:-alelab}"

SUBPATH="${1:?usage: $0 <subdir>/<config-name> under paper_experiments/function_calling/ (e.g. llama/dpo, sweep/simpo)}"
SUBDIR="${SUBPATH%/*}"
CONFIG_NAME="${SUBPATH##*/}"

if [[ "$SUBDIR" == "$SUBPATH" ]]; then
  echo "error: expected subdir/name, got: $SUBPATH" >&2
  exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    src/train.py \
    --config-path="train/paper_experiments/function_calling/${SUBDIR}" \
    --config-name="${CONFIG_NAME}"
