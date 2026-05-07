#!/usr/bin/env bash
# Run wandb_alpaca_eval_vllm.py for every W&B run in a project that has a given tag.
#
# Usage:
#   ./alpacaeval_by_tag.sh <wandb_entity> <wandb_project> <tag>
# Environment:
#   CUDA_VISIBLE_DEVICES (default: 0)

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <wandb_entity> <wandb_project> <tag>" >&2
  exit 1
fi

WANDB_ENTITY="$1"
WANDB_PROJECT="$2"
TAG="$3"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_IDS="$(
  WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" TAG="$TAG" python3 - <<'PY'
import os
import sys
import wandb

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
tag = os.environ["TAG"]

api = wandb.Api()
path = f"{entity}/{project}"

runs = api.runs(path, filters={"tags": {"$in": [tag]}})
ids = []
for run in runs:
    ids.append(run.id)

if not ids:
    for run in api.runs(path):
        if tag in (run.tags or []):
            ids.append(run.id)

if not ids:
    print("No runs matched tag; nothing to do.", file=sys.stderr)
    sys.exit(1)

for run_id in ids:
    print(run_id)
PY
)"

for ids in $RUN_IDS; do
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python src/eval/wandb_alpaca_eval_vllm.py \
    wandb.entity="$WANDB_ENTITY" \
    wandb.project="$WANDB_PROJECT" \
    wandb.run_id="$ids"
done
