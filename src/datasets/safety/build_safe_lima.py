#!/usr/bin/env python3
"""
Build and push the combined SAFE-ALPACA-2 (safe-only) + LIMA-Alpaca dataset.

- SAFE-ALPACA-2: keep only safety_label == True, per split (train/validation)
- LIMA-Alpaca (xzuyn/lima-alpaca): random 80/20 split into train/validation
- Add safety_label=False to all LIMA examples
- Concatenate per split and push to Hub as ihounie/safe-lima (by default)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, Features, Value, concatenate_datasets, load_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--safe_alpaca2_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[0] / "SAFE-ALPACA-2"
        ),
        help="Path to SAFE-ALPACA-2 directory (contains train/ and validation/).",
    )
    p.add_argument(
        "--lima_repo",
        type=str,
        default="xzuyn/lima-alpaca",
        help="HF dataset repo id to load for LIMA Alpaca.",
    )
    p.add_argument(
        "--repo_id",
        type=str,
        default="ihounie/safe-lima",
        help="HF dataset repo id to push to.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for the 80/20 LIMA split.")
    p.add_argument(
        "--lima_train_frac",
        type=float,
        default=0.8,
        help="Fraction of LIMA to keep in train (rest goes to validation).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/push the dataset as private on the Hub.",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional; otherwise uses HF_TOKEN env var / cached login).",
    )
    return p.parse_args()


def _require_columns(ds: Dataset, required) -> None:
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns {missing}. Found: {ds.column_names}")


def _cast_to_common_schema(ds: Dataset) -> Dataset:
    """
    Ensure schema is exactly: instruction, input, output, safety_label.
    """
    required = ["instruction", "input", "output", "safety_label"]
    _require_columns(ds, required)
    ds = ds.select_columns(required)
    features = Features(
        {
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
            "safety_label": Value("bool"),
        }
    )
    return ds.cast(features)


def load_safe_alpaca2_safe_only(safe_alpaca2_dir: Path) -> DatasetDict:
    train_path = safe_alpaca2_dir / "train" / "safe_alpaca-00000-of-00001.json"
    val_path = safe_alpaca2_dir / "validation" / "safe_alpaca-00000-of-00001.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing SAFE-ALPACA-2 train file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing SAFE-ALPACA-2 validation file: {val_path}")

    safe_ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    for split in list(safe_ds.keys()):
        _require_columns(safe_ds[split], ["instruction", "input", "output", "safety_label"])
        safe_ds[split] = safe_ds[split].filter(lambda ex: bool(ex["safety_label"]))
        safe_ds[split] = _cast_to_common_schema(safe_ds[split])
    return safe_ds


def load_lima_with_safety_label_false(lima_repo: str, seed: int, train_frac: float) -> DatasetDict:
    lima = load_dataset(lima_repo)
    if "train" not in lima:
        raise ValueError(f"Expected {lima_repo} to have a 'train' split. Found: {list(lima.keys())}")
    base = lima["train"]
    _require_columns(base, ["instruction", "input", "output"])
    base = base.map(lambda ex: {"safety_label": False}, remove_columns=[])
    base = base.select_columns(["instruction", "input", "output", "safety_label"])
    base = _cast_to_common_schema(base)

    split = base.train_test_split(test_size=1.0 - train_frac, seed=seed, shuffle=True)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_combined_dataset(safe_only: DatasetDict, lima: DatasetDict) -> DatasetDict:
    combined: Dict[str, object] = {}
    for split in ["train", "validation"]:
        if split not in safe_only or split not in lima:
            raise ValueError(f"Missing split '{split}' in inputs.")
        combined[split] = concatenate_datasets([safe_only[split], lima[split]])
    return DatasetDict(combined)


def main() -> None:
    args = _parse_args()
    safe_dir = Path(args.safe_alpaca2_dir).expanduser().resolve()

    safe_only = load_safe_alpaca2_safe_only(safe_dir)
    lima = load_lima_with_safety_label_false(args.lima_repo, seed=args.seed, train_frac=args.lima_train_frac)
    combined = build_combined_dataset(safe_only, lima)

    print("SAFE-ALPACA-2 safe-only sizes:", {k: v.num_rows for k, v in safe_only.items()})
    print("LIMA sizes:", {k: v.num_rows for k, v in lima.items()})
    print("COMBINED sizes:", {k: v.num_rows for k, v in combined.items()})
    print("Features:", combined["train"].features)

    combined.push_to_hub(args.repo_id, private=args.private, token=args.token)
    print(f"Pushed dataset to {args.repo_id}")


if __name__ == "__main__":
    main()

