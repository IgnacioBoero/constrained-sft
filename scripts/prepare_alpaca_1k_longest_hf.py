#!/usr/bin/env python3
"""
Prepare the EPFL "alpaca 1k longest" dataset into standard Alpaca format and optionally push to Hugging Face.

Source: https://github.com/tml-epfl/long-is-more-for-alignment
Raw file used:
  https://raw.githubusercontent.com/tml-epfl/long-is-more-for-alignment/main/data/alpaca/filtered_alpaca_1k_longest.json

Output schema (standard Alpaca):
  - instruction: str
  - input: str
  - output: str
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_RE_INSTRUCTION = re.compile(
    r"###\s*Instruction:\s*(.*?)(?:\s*###\s*(?:Input|Response)\s*:|\Z)",
    flags=re.DOTALL,
)
_RE_INPUT = re.compile(
    r"###\s*Input:\s*(.*?)(?:\s*###\s*Response\s*:|\Z)",
    flags=re.DOTALL,
)


def _extract_instruction_input(human_prompt: str) -> Tuple[str, str]:
    instruction = ""
    input_ = ""

    m_inst = _RE_INSTRUCTION.search(human_prompt)
    if m_inst:
        instruction = m_inst.group(1).strip()

    m_inp = _RE_INPUT.search(human_prompt)
    if m_inp:
        input_ = m_inp.group(1).strip()

    # Fallback: if we didn't find the marker, treat the whole prompt as instruction.
    if not instruction:
        instruction = human_prompt.strip()

    return instruction, input_


def _example_to_alpaca(ex: Dict[str, Any]) -> Dict[str, str]:
    conversations = ex.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"Unexpected conversations format for id={ex.get('id')!r}")

    # Find first human and last assistant message (robust to extra turns).
    human_msg = next((m for m in conversations if m.get("from") == "human"), None)
    gpt_msg = next((m for m in reversed(conversations) if m.get("from") in {"gpt", "assistant"}), None)
    if not human_msg or not gpt_msg:
        raise ValueError(f"Could not locate human/gpt messages for id={ex.get('id')!r}")

    instruction, input_ = _extract_instruction_input(str(human_msg.get("value", "")))
    output = str(gpt_msg.get("value", "")).strip()

    return {"instruction": instruction, "input": input_, "output": output}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source_json",
        type=Path,
        default=Path("data_external/long-is-more-for-alignment/alpaca/filtered_alpaca_1k_longest.json"),
        help="Path to the downloaded EPFL JSON file.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data_processed/alpaca_1k_longest"),
        help="Directory where processed dataset artifacts will be written.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--repo_id", type=str, default="ihounie/alpaca-1k-longest")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument(
        "--private",
        action="store_true",
        help="If set, create/push the Hub repo as private (recommended).",
    )
    args = p.parse_args()

    with args.source_json.open("r", encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    examples = [_example_to_alpaca(ex) for ex in raw]

    # Lazy import so conversion can run even if HF deps are missing.
    from datasets import Dataset, DatasetDict

    ds = Dataset.from_list(examples).shuffle(seed=args.seed)
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    dsd = DatasetDict(train=split["train"], validation=split["test"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(args.output_dir / "hf_dataset"))
    dsd["train"].to_json(str(args.output_dir / "train.jsonl"), orient="records", lines=True, force_ascii=False)
    dsd["validation"].to_json(
        str(args.output_dir / "validation.jsonl"), orient="records", lines=True, force_ascii=False
    )

    if args.push_to_hub:
        dsd.push_to_hub(args.repo_id, private=bool(args.private))


if __name__ == "__main__":
    main()

