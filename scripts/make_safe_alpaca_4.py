#!/usr/bin/env python3
"""
Create SAFE-ALPACA-4 by attaching model generations as `unsafe_response` to the
`iboero16/SAFE-ALPACA-3` validation split.

Input file format: JSON list of objects with keys: {prompt, answer, safe}
where prompt looks like:
  "BEGINNING OF CONVERSATION: USER: <instruction> ASSISTANT:"

What this script does:
  1) Matches each extracted <instruction> to `instruction` in SAFE-ALPACA-3 validation.
  2) Adds the corresponding `answer` as `unsafe_response` for that validation row.
  3) Saves the resulting DatasetDict (train + validation) to disk.
Optionally pushes to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


PROMPT_RE = re.compile(
    r"BEGINNING OF CONVERSATION:\s*USER:\s*(.*?)\s*ASSISTANT:\s*$", re.DOTALL
)


def _norm_text(s: str) -> str:
    # Normalize whitespace so "exact match" works across minor formatting differences
    return re.sub(r"\s+", " ", s.strip())


def extract_instruction(prompt: str) -> str:
    m = PROMPT_RE.match(prompt.strip())
    if m:
        return m.group(1)
    # Fallback: try a looser extraction if prefix/suffix differs slightly
    p = prompt
    p = re.sub(r"^\s*BEGINNING OF CONVERSATION:\s*USER:\s*", "", p, flags=re.DOTALL)
    p = re.sub(r"\s*ASSISTANT:\s*$", "", p, flags=re.DOTALL)
    return p.strip()


def load_generations_json(path: Path) -> Dict[str, str]:
    """Return mapping normalized_instruction -> answer."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data)}")

    mapping: Dict[str, str] = {}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not an object: {type(row)}")
        if "prompt" not in row or "answer" not in row:
            raise ValueError(f"Row {i} missing 'prompt'/'answer' keys: {row.keys()}")
        instr = extract_instruction(str(row["prompt"]))
        ans = row["answer"]
        if ans is None:
            ans = ""
        mapping[_norm_text(instr)] = str(ans)
    return mapping


def try_load_safe_alpaca_3(dataset_name: str) -> DatasetDict:
    """
    Load SAFE-ALPACA-3 from hub if possible; if offline/unavailable, fall back to the
    local JSON copies tracked in this repo.
    """
    try:
        return load_dataset(dataset_name)  # type: ignore[no-any-return]
    except Exception as e:
        repo_root = Path(__file__).resolve().parents[1]
        local_root = repo_root / "src" / "datasets" / "safety" / "SAFE-ALPACA-3"
        train_files = sorted((local_root / "train").glob("*.json"))
        val_files = sorted((local_root / "validation").glob("*.json"))
        if not train_files or not val_files:
            raise RuntimeError(
                f"Failed to load {dataset_name} from hub ({e}) and local fallback "
                f"was not found at {local_root}"
            ) from e
        return load_dataset(
            "json",
            data_files={"train": [str(p) for p in train_files], "validation": [str(p) for p in val_files]},
        )


def attach_unsafe_responses(
    ds: DatasetDict, generations: Dict[str, str], validation_split: str = "validation"
) -> Tuple[DatasetDict, Dict[str, Any]]:
    if validation_split not in ds:
        raise KeyError(f"Dataset has no split '{validation_split}'. Splits: {list(ds.keys())}")
    if "instruction" not in ds[validation_split].column_names:
        raise KeyError(
            f"Split '{validation_split}' has no 'instruction' field. "
            f"Columns: {ds[validation_split].column_names}"
        )

    # Build reverse map to detect instruction duplicates in the validation set
    instr_to_indices: Dict[str, List[int]] = defaultdict(list)
    val_instrs = ds[validation_split]["instruction"]
    for idx, instr in enumerate(val_instrs):
        instr_to_indices[_norm_text(str(instr))].append(idx)

    # Use "" as the default so the feature type is consistently "string" across splits
    # and we avoid null-only columns in other splits (which can confuse schema inference).
    unsafe_response: List[str] = [""] * len(ds[validation_split])
    matched_prompts = 0
    ambiguous = 0
    missing_in_val = 0

    for instr_norm, ans in generations.items():
        indices = instr_to_indices.get(instr_norm)
        if not indices:
            missing_in_val += 1
            continue
        if len(indices) > 1:
            ambiguous += 1
        # Fill all duplicates to be deterministic; you can change this to first-only.
        for idx in indices:
            unsafe_response[idx] = ans
        matched_prompts += 1

    # Ensure *all* splits share the same features. HF `DatasetDict` enforces this in
    # some operations (notably `push_to_hub`).
    ds_out_dict: Dict[str, Dataset] = {}
    for split_name, split_ds in ds.items():
        if "unsafe_response" in split_ds.column_names:
            split_ds = split_ds.remove_columns(["unsafe_response"])

        if split_name == validation_split:
            split_ds = split_ds.add_column("unsafe_response", unsafe_response)
        else:
            split_ds = split_ds.add_column("unsafe_response", [""] * len(split_ds))

        ds_out_dict[split_name] = split_ds

    ds_out = DatasetDict(ds_out_dict)

    stats: Dict[str, Any] = {
        "validation_rows": len(ds[validation_split]),
        "generations_rows": len(generations),
        "matched_generation_instructions": matched_prompts,
        "missing_generation_instructions_in_validation": missing_in_val,
        "ambiguous_instruction_collisions_in_validation": ambiguous,
        "filled_validation_rows": sum(x != "" for x in unsafe_response),
    }
    return ds_out, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generations_json",
        type=Path,
        required=True,
        help="Path to safe_generate_train_end_outputs_epoch_*.json",
    )
    p.add_argument(
        "--source_dataset",
        type=str,
        default="iboero16/SAFE-ALPACA-3",
        help="HF dataset name to load (defaults to iboero16/SAFE-ALPACA-3).",
    )
    p.add_argument(
        "--validation_split",
        type=str,
        default="validation",
        help="Which split to attach unsafe_response to (default: validation).",
    )
    p.add_argument(
        "--save_to_disk",
        type=Path,
        required=True,
        help="Output directory for datasets.save_to_disk(...)",
    )
    p.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="If set (e.g. ihounie/SAFE-ALPACA-4), pushes the resulting DatasetDict to hub.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="If pushing, create/update the repo as private (requires permissions).",
    )
    p.add_argument(
        "--commit_message",
        type=str,
        default="Add unsafe_response generations to validation (SAFE-ALPACA-4)",
        help="Commit message used for push_to_hub.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    generations = load_generations_json(args.generations_json)
    ds = try_load_safe_alpaca_3(args.source_dataset)

    ds_out, stats = attach_unsafe_responses(
        ds=ds, generations=generations, validation_split=args.validation_split
    )

    args.save_to_disk.mkdir(parents=True, exist_ok=True)
    ds_out.save_to_disk(str(args.save_to_disk))

    print(json.dumps({"saved_to_disk": str(args.save_to_disk), **stats}, indent=2))

    if args.push_to_hub:
        # Respect HF_HOME / token env vars; user must be logged in via huggingface-cli.
        ds_out.push_to_hub(
            args.push_to_hub,
            private=bool(args.private),
            commit_message=args.commit_message,
        )
        print(json.dumps({"pushed_to_hub": args.push_to_hub}, indent=2))


if __name__ == "__main__":
    # Reduce tokenizers parallelism warning spam
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

