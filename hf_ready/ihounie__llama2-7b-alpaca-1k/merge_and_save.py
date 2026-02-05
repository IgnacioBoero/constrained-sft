#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF id or local path of the base model")
    ap.add_argument("--adapter_dir", required=True, help="Path to the LoRA adapter folder (this repo)")
    ap.add_argument("--out_dir", required=True, help="Where to write the merged standalone model")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dtype == "auto":
        torch_dtype = None
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))


if __name__ == "__main__":
    main()
