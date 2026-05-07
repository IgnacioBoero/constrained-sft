#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Bash wrapper for SAFE-long1k AlpacaEval.
#
# Each eval runs as a direct child of this shell, avoiding the
# Python-subprocess → vLLM-multiprocess nesting that causes WorkerProc
# initialisation failures.
#
# Usage:
#   bash scripts/run_safe_long1k_eval.sh                 # all qualifying runs
#   bash scripts/run_safe_long1k_eval.sh --tag my_tag    # only runs with tag
#   bash scripts/run_safe_long1k_eval.sh --include-evaluated
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Configurable defaults (override via env vars) ───────────────────────
ENTITY="${ENTITY:-alelab}"
PROJECT="${PROJECT:-SAFE-long1k}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/alpaca_eval_vllm/${PROJECT}}"

# vLLM environment fixes
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# ── Collect qualifying run IDs via the Python helper ────────────────────
# Forward all CLI args (--tag, --include-evaluated) to the Python script.
echo "[run_safe_long1k_eval.sh] Querying W&B for qualifying runs …"
mapfile -t RUN_IDS < <(python scripts/run_safe_long1k_eval.py --print-ids "$@")

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "[run_safe_long1k_eval.sh] No qualifying runs found. Nothing to do."
    exit 0
fi

TOTAL=${#RUN_IDS[@]}
echo "[run_safe_long1k_eval.sh] Found $TOTAL run(s) to evaluate: ${RUN_IDS[*]}"
echo

# ── Run each evaluation as a top-level process ──────────────────────────
FAILED=()
for i in "${!RUN_IDS[@]}"; do
    RID="${RUN_IDS[$i]}"
    N=$((i + 1))

    CMD=(python src/eval/wandb_alpaca_eval_vllm.py
         "wandb.entity=${ENTITY}"
         "wandb.project=${PROJECT}"
         "wandb.run_id=${RID}"
         "vllm.max_model_len=${MAX_MODEL_LEN}"
         "eval.local_output_dir=${OUTPUT_DIR}")

    echo "[$N/$TOTAL] ${CMD[*]}"
    if "${CMD[@]}"; then
        echo "  ✓ Run $RID done."
    else
        echo "  ⚠ Run $RID failed (exit $?)."
        FAILED+=("$RID")
    fi
    echo
done

# ── Summary ─────────────────────────────────────────────────────────────
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[run_safe_long1k_eval.sh] ${#FAILED[@]} run(s) failed: ${FAILED[*]}"
    exit 1
else
    echo "[run_safe_long1k_eval.sh] All $TOTAL run(s) completed successfully."
fi
