#!/usr/bin/env python3
"""
Compute the upper-tail CVaR of the end-of-training eval slacks.

For each run:
  1. Download the merged ``constraint_slacks_epoch_*_eval`` table.
  2. Compute CVaR_{alpha} = mean of slacks strictly above the ``alpha``
     quantile (alpha=0.95 by default => mean over the upper 5% tail)
     using the RAW slack values stored in the table (no sign flipping).
  3. Log per-category and combined CVaR values back to the wandb run
     summary under ``eval_cvar/*`` keys.

Example:

    python scripts/compute_slack_cvar.py \
        --run-id dojd9qmj --run-id sg59r2xi \
        --project alelab/SAFE-long1k --alpha 0.95
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _find_single_table_json(root: Path) -> Path:
    tbl = [
        p for p in glob.glob(str(root / "**/*.json"), recursive=True)
        if p.endswith(".table.json")
    ]
    if not tbl:
        matches = glob.glob(str(root / "**/*.json"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No JSON found under {root}")
        return Path(matches[0])
    return Path(tbl[0])


def cvar_upper(values: np.ndarray, alpha: float) -> float:
    """Mean of values above the `alpha` quantile (empirical upper-tail CVaR)."""
    if values.size == 0:
        return float("nan")
    q = float(np.quantile(values, alpha))
    tail = values[values > q]
    if tail.size == 0:
        # All values tied at the quantile => fall back to max
        return float(values.max())
    return float(tail.mean())


def process_run(run_id: str, project: str, alpha: float, work_dir: Path, dry_run: bool) -> dict:
    import wandb
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    print(f"\n=== {run_id} ({run.name}) ===")

    key = "constraint_slacks_epoch_3.0_eval"
    entry = run.summary.get(key)
    if entry is None:
        print(f"    [{key}] missing; skipping")
        return {}

    dst = work_dir / run_id / key
    dst.mkdir(parents=True, exist_ok=True)
    f = run.file(entry["path"])
    f.download(root=str(dst), replace=True)
    with open(_find_single_table_json(dst)) as fh:
        data = json.load(fh)

    cols = data["columns"]
    rows = data["data"]
    label_idx = cols.index("safety_label")
    slack_idx = cols.index("constraint_slack")

    slacks_true = np.array(
        [row[slack_idx] for row in rows if bool(row[label_idx])], dtype=np.float64
    )
    slacks_false = np.array(
        [row[slack_idx] for row in rows if not bool(row[label_idx])], dtype=np.float64
    )
    slacks_combined = np.concatenate([slacks_true, slacks_false], axis=0)

    def _report(name: str, v: np.ndarray) -> dict:
        stats = {
            "n": int(v.size),
            "mean": float(v.mean()) if v.size else float("nan"),
            "min": float(v.min()) if v.size else float("nan"),
            "max": float(v.max()) if v.size else float("nan"),
            "quantile_alpha": float(np.quantile(v, alpha)) if v.size else float("nan"),
            "cvar_upper": cvar_upper(v, alpha),
        }
        print(
            f"    [{name}] n={stats['n']} mean={stats['mean']:.4f} "
            f"q{alpha:.2f}={stats['quantile_alpha']:.4f} "
            f"CVaR_upper={stats['cvar_upper']:.4f} "
            f"(min={stats['min']:.4f}, max={stats['max']:.4f})"
        )
        return stats

    print(f"    alpha={alpha} (upper tail mean over top {(1-alpha)*100:.1f}%)")
    stats = {
        "unsafe": _report("unsafe (safe=True)", slacks_true),
        "safe": _report("safe (safe=False)", slacks_false),
        "combined": _report("combined", slacks_combined),
    }

    if dry_run:
        print("    --dry-run set; not logging to wandb.")
        return stats

    entity, project_name = project.split("/")
    wb = wandb.init(project=project_name, entity=entity, id=run_id, resume="must")
    try:
        payload = {
            "eval_cvar/alpha": float(alpha),
            "eval_cvar/cvar_upper_unsafe": stats["unsafe"]["cvar_upper"],
            "eval_cvar/cvar_upper_safe": stats["safe"]["cvar_upper"],
            "eval_cvar/cvar_upper_combined": stats["combined"]["cvar_upper"],
            "eval_cvar/mean_unsafe": stats["unsafe"]["mean"],
            "eval_cvar/mean_safe": stats["safe"]["mean"],
            "eval_cvar/mean_combined": stats["combined"]["mean"],
            # Remove the previously-logged negated-sign metrics.
            "eval_cvar/cvar_upper_safe_negated": None,
            "eval_cvar/mean_safe_negated": None,
        }
        wb.log({k: v for k, v in payload.items() if v is not None})
        wb.summary.update({k: v for k, v in payload.items() if v is not None})
        for k, v in payload.items():
            if v is None:
                try:
                    del wb.summary[k]
                except (KeyError, AttributeError):
                    pass
        print(f"    logged payload: { {k:v for k,v in payload.items() if v is not None} }")
    finally:
        wb.finish()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", action="append", required=True)
    ap.add_argument("--project", default="alelab/SAFE-long1k")
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="Upper-tail CVaR quantile (0.95 = top 5%)")
    ap.add_argument("--work-dir", default="/tmp/slack_cvar")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for rid in args.run_id:
        process_run(rid, args.project, args.alpha, work_dir, args.dry_run)


if __name__ == "__main__":
    main()
