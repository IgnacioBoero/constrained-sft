#!/usr/bin/env python3
"""
Batch-run ``src/eval/wandb_alpaca_eval_vllm.py`` for every qualifying run
in the W&B project **alelab/SAFE-long1k**.

Usage
-----
# Evaluate all finished runs that have LoRA artifacts:
    python scripts/run_safe_long1k_eval.py

# Only runs with a specific tag:
    python scripts/run_safe_long1k_eval.py --tag my_tag

# Dry-run (just print the commands):
    python scripts/run_safe_long1k_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import wandb

ENTITY = "alelab"
PROJECT = "SAFE-long1k"


def _has_lora_artifact(run_obj) -> bool:
    """Return True if *run_obj* has at least one logged ``lora_adapters`` artifact."""
    try:
        for art in run_obj.logged_artifacts():
            if getattr(art, "type", None) == "lora_adapters":
                return True
    except Exception:
        pass
    return False


def _already_has_alpaca_eval(run_obj) -> bool:
    """Return True if the run already has alpaca_eval outputs logged."""
    try:
        for art in run_obj.logged_artifacts():
            if getattr(art, "type", None) == "alpaca_eval_outputs_vllm":
                return True
    except Exception:
        pass
    return False


def get_qualifying_runs(
    tag: str | None = None,
    skip_evaluated: bool = True,
) -> list:
    """Return W&B Run objects from *ENTITY/PROJECT* that should be evaluated."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    qualifying = []
    for run in runs:
        # Only finished runs
        if run.state != "finished":
            continue

        # Tag filter (if requested)
        run_tags = set(getattr(run, "tags", []) or [])
        if tag and tag not in run_tags:
            continue

        # Must have LoRA artifacts
        if not _has_lora_artifact(run):
            continue

        # Optionally skip runs that already have AlpacaEval outputs
        if skip_evaluated and _already_has_alpaca_eval(run):
            continue

        qualifying.append(run)

    return qualifying


def build_command(run_id: str, max_model_len: int | None = None) -> list[str]:
    """Build the CLI command list for one evaluation run."""
    cmd = [
        sys.executable,
        "src/eval/wandb_alpaca_eval_vllm.py",
        # Override only the wandb section; everything else comes from the
        # default config (configs/eval/wandb_alpaca_eval_vllm.yaml).
        f"wandb.entity={ENTITY}",
        f"wandb.project={PROJECT}",
        f"wandb.run_id={run_id}",
    ]
    if max_model_len is not None:
        cmd.append(f"vllm.max_model_len={max_model_len}")
    return cmd


# Environment tweaks that are forwarded to every eval subprocess.
_SUBPROCESS_ENV: dict[str, str] = {
    **os.environ,
    # Use 'spawn' instead of the default 'fork' to avoid vLLM v1
    # WorkerProc termination during multiprocess engine initialisation.
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch AlpacaEval (vLLM) for alelab/SAFE-long1k runs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Only evaluate runs that carry this W&B tag.",
    )
    parser.add_argument(
        "--include-evaluated",
        action="store_true",
        default=False,
        help="Re-evaluate runs that already have alpaca_eval_outputs_vllm artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM max_model_len override (avoids huge KV-cache for 128K-context models). "
             "Set to 0 to use the model default.",
    )
    parser.add_argument(
        "--print-ids",
        action="store_true",
        default=False,
        help="Print qualifying run IDs (one per line) and exit. "
             "Useful for feeding into a bash wrapper.",
    )
    args = parser.parse_args()

    runs = get_qualifying_runs(
        tag=args.tag,
        skip_evaluated=not args.include_evaluated,
    )

    if not runs:
        if not args.print_ids:
            print("[run_safe_long1k_eval] No qualifying runs found. Nothing to do.")
        return

    # --print-ids: just emit IDs for consumption by a shell script.
    if args.print_ids:
        for r in runs:
            print(r.id)
        return

    print(f"[run_safe_long1k_eval] Found {len(runs)} run(s) to evaluate:")
    for r in runs:
        print(f"  - {r.id}  ({r.name})")
    print()

    max_model_len = args.max_model_len if args.max_model_len else None

    failed: list[str] = []
    for i, run in enumerate(runs, 1):
        cmd = build_command(run.id, max_model_len=max_model_len)
        cmd_str = " ".join(cmd)
        print(f"[{i}/{len(runs)}] {cmd_str}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False, env=_SUBPROCESS_ENV)
        if result.returncode != 0:
            print(f"  ⚠ Run {run.id} ({run.name}) failed (exit {result.returncode})")
            failed.append(run.id)
        else:
            print(f"  ✓ Run {run.id} done.")
        print()

    if failed:
        print(f"[run_safe_long1k_eval] {len(failed)} run(s) failed: {failed}")
        sys.exit(1)
    elif not args.dry_run:
        print(f"[run_safe_long1k_eval] All {len(runs)} run(s) completed successfully.")


if __name__ == "__main__":
    main()
