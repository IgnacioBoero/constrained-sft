#!/usr/bin/env python3
"""
Evaluate LoRA-finetuned models from a W&B project on the PKU-SafeRLHF-30K
*test* split using PKU-Alignment's unified reward & cost models.

Either:
  A) For every W&B run in `--entity/--project` with `--tag` (or `--run_ids`):
     pull `lora_adapters`, attach to the base model, generate, score, optionally
     log back to that run.
  B) For each `--hf_models` Hugging Face repo id (pretrained causal LM only):
     load with `AutoModelForCausalLM`, generate, score; write CSV/metrics under
     `out_root/<repo_id with slashes replaced by __>/`. No W&B unless you log
     manually (these entries have no `run_obj`).

In both cases we greedy-sample on the PKU-SafeRLHF-30K *test* deduped prompts
and score with:
  - PKU-Alignment/beaver-7b-unified-reward
  - PKU-Alignment/beaver-7b-unified-cost

The beaver-7b score models are LLaMA backbones with a single-dim linear head.
They were trained in the `safe-rlhf` repo (not in HF transformers), so we
re-implement a minimal `LlamaForScore` locally and load weights via
`snapshot_download`.

Run from outside the repo root (the local ./wandb directory shadows the
wandb package):

    cd /tmp
    CUDA_VISIBLE_DEVICES=1 python /home/chiche/constrained-sft-3/scripts/eval_saferlhf_beaver.py --entity alelab --project SAFE-long1k --tag xtest

Project docs: configs/train/paper_experiments/README.md · safety detail: configs/train/paper_experiments/safety/README.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from utils import format_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal LlamaForScore (safe-rlhf compatible): LlamaModel + nn.Linear head.
# Weights in the published checkpoints are stored as `model.*` and
# `score_head.weight` / `score_head.bias`.
# ---------------------------------------------------------------------------
def build_llama_for_score(model_repo: str, device: str, dtype: torch.dtype):
    from transformers import LlamaModel, AutoConfig, AutoTokenizer
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    local_dir = snapshot_download(repo_id=model_repo, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "tokenizer*", "special_tokens_map*",
    ])
    config = AutoConfig.from_pretrained(local_dir)

    class LlamaForScore(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = LlamaModel(config)
            score_dim = getattr(config, "score_dim", 1)
            score_bias = getattr(config, "score_bias", True)
            self.score_head = nn.Linear(config.hidden_size, score_dim, bias=score_bias)

        def forward(self, input_ids, attention_mask=None):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (B, L, H)
            scores = self.score_head(hidden)  # (B, L, 1)
            if attention_mask is not None:
                last = attention_mask.sum(dim=1) - 1  # (B,)
            else:
                last = torch.full(
                    (input_ids.shape[0],),
                    input_ids.shape[1] - 1,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            last = last.clamp(min=0)
            end_scores = scores[torch.arange(scores.shape[0], device=scores.device), last]
            return end_scores.squeeze(-1)  # (B,)

    model = LlamaForScore(config)
    state_dict: Dict[str, torch.Tensor] = {}
    shard_files = sorted(Path(local_dir).glob("*.safetensors"))
    if not shard_files:
        raise RuntimeError(f"No .safetensors files found in {local_dir}")
    for shard in shard_files:
        sd = load_file(str(shard), device="cpu")
        state_dict.update(sd)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{model_repo}] missing keys (first 5): {list(missing)[:5]} (total {len(missing)})")
    if unexpected:
        print(f"[{model_repo}] unexpected keys (first 5): {list(unexpected)[:5]} (total {len(unexpected)})")

    model.to(dtype=dtype, device=device)
    model.eval()

    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token_id = getattr(config, "pad_token_id", tok.eos_token_id)
    tok.padding_side = "right"
    return model, tok


@torch.no_grad()
def score_prompts_and_responses(
    model,
    tok,
    prompts: List[str],
    responses: List[str],
    batch_size: int,
    max_length: int = 2048,
) -> List[float]:
    from tqdm.auto import tqdm

    assert len(prompts) == len(responses)
    device = next(model.parameters()).device
    texts = [
        format_prompt(input=p, eos_token=tok.eos_token) + (r if isinstance(r, str) else "")
        for p, r in zip(prompts, responses)
    ]

    scores: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring"):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        end = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        scores.extend(end.float().cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Generation side: base + LoRA adapter, greedy decoding.
# ---------------------------------------------------------------------------
def find_lora_adapter_artifact(run) -> Optional[Any]:
    for art in run.logged_artifacts():
        if art.type == "lora_adapters":
            return art
    return None


def download_adapter(art, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(art.download(root=str(dest_dir)))


def build_model_and_tokenizer(
    base_model: str,
    adapter_dir: Path,
    dtype: torch.dtype,
    device: str,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else base_model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.to(device)
    return model, tok


def build_causal_lm_and_tokenizer(
    model_id: str,
    dtype: torch.dtype,
    device: str,
):
    """Load a standalone HF causal LM (no adapter)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.to(device)
    return model, tok


@torch.no_grad()
def generate_completions(
    model,
    tok,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    eos = tok.eos_token
    formatted = [format_prompt(input=p, eos_token=eos) for p in prompts]

    completions: List[str] = []
    device = next(model.parameters()).device
    from tqdm.auto import tqdm

    for i in tqdm(range(0, len(formatted), batch_size), desc="generating"):
        batch = formatted[i : i + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        completions.extend(decoded)

    return completions


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_test_prompts(num_samples: Optional[int], seed: int = 0) -> List[str]:
    from datasets import load_dataset

    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="test")
    except Exception:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
    prompts = list(dict.fromkeys(ds["prompt"]))  # dedupe, keep order
    print(f"[prompts] Loaded {len(prompts)} unique test prompts.")
    if num_samples is not None and num_samples < len(prompts):
        import random

        rng = random.Random(seed)
        prompts = rng.sample(prompts, num_samples)
        print(f"[prompts] Sampled {len(prompts)} prompts (seed={seed}).")
    return prompts


# ---------------------------------------------------------------------------
# Per-run generation phase.
# ---------------------------------------------------------------------------
def generate_for_run(
    run,
    prompts: List[str],
    out_root: Path,
    base_model_override: Optional[str],
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    run_id = run.id
    print(f"\n========== GEN Run {run_id} ({run.name}) ==========")

    run_out = out_root / run_id
    run_out.mkdir(parents=True, exist_ok=True)

    art = find_lora_adapter_artifact(run)
    if art is None:
        print(f"  [skip] No lora_adapters artifact on {run_id}")
        return None

    adapter_dir = run_out / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading adapter {art.name} ...")
    adapter_dir = download_adapter(art, adapter_dir)

    meta = dict(getattr(art, "metadata", {}) or {})
    base_model = (
        base_model_override
        or meta.get("model_name")
        or meta.get("base_model")
        or (run.config.get("exp", {}) or {}).get("model_name")
    )
    if not base_model:
        raise RuntimeError(f"Could not determine base model for {run_id}")
    print(f"  Base model: {base_model}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tok = build_model_and_tokenizer(base_model, adapter_dir, dtype, device)

    print(f"  Generating {len(prompts)} completions (bs={batch_size}) ...")
    completions = generate_completions(
        model, tok, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens
    )

    del model, tok
    _free_cuda()

    comp_df = pd.DataFrame({"prompt": prompts, "completion": completions})
    comp_csv = run_out / "saferlhf_completions.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"  Wrote {comp_csv} ({len(comp_df)} rows).")
    return {
        "run_id": run_id,
        "run_obj": run,
        "base_model": base_model,
        "completions_csv": comp_csv,
        "run_out": run_out,
    }


def generate_for_hf_model(
    model_id: str,
    prompts: List[str],
    out_root: Path,
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    """Generate completions for a pretrained HF causal LM (no W&B / no LoRA)."""
    run_id = model_id.replace("/", "__")
    print(f"\n========== GEN HF model {model_id} (folder id={run_id}) ==========")

    run_out = out_root / run_id
    run_out.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tok = build_causal_lm_and_tokenizer(model_id, dtype, device)

    print(f"  Generating {len(prompts)} completions (bs={batch_size}) ...")
    completions = generate_completions(
        model, tok, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens
    )

    del model, tok
    _free_cuda()

    comp_df = pd.DataFrame({"prompt": prompts, "completion": completions})
    comp_csv = run_out / "saferlhf_completions.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"  Wrote {comp_csv} ({len(comp_df)} rows).")
    return {
        "run_id": run_id,
        "run_obj": None,
        "base_model": model_id,
        "completions_csv": comp_csv,
        "run_out": run_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf_models",
        nargs="*",
        default=None,
        help="HF repo id(s) of pretrained causal LMs to evaluate (no LoRA, no W&B resume).",
    )
    ap.add_argument("--entity", default="alelab")
    ap.add_argument("--project", default="SAFE-long1k")
    ap.add_argument("--tag", default="xtest")
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--out_root", default="/tmp/saferlhf_beaver_eval")
    ap.add_argument("--num_samples", type=int, default=500,
                    help="Number of test prompts to sample (None = all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--score_batch_size", type=int, default=4)
    ap.add_argument("--reward_model", default="PKU-Alignment/beaver-7b-unified-reward")
    ap.add_argument("--cost_model", default="PKU-Alignment/beaver-7b-unified-cost")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_wandb_push", action="store_true",
                    help="Compute metrics but don't push back to W&B.")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    num_samples = None if (args.num_samples is None or args.num_samples <= 0) else args.num_samples
    prompts = load_test_prompts(num_samples, seed=args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # ---------------- Phase 1: generate all completions ----------------
    gen_results: List[Dict[str, Any]] = []

    if args.hf_models:
        print(f"HF-only mode: {len(args.hf_models)} pretrained model(s).")
        for mid in args.hf_models:
            if not mid or not str(mid).strip():
                continue
            mid = str(mid).strip()
            try:
                res = generate_for_hf_model(
                    mid,
                    prompts=prompts,
                    out_root=out_root,
                    batch_size=args.gen_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                gen_results.append(res)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"[gen] error on HF model {mid}: {e}")
    else:
        import wandb

        api = wandb.Api(timeout=60)
        if args.run_ids:
            runs = [api.run(f"{args.entity}/{args.project}/{rid}") for rid in args.run_ids]
        else:
            runs = list(
                api.runs(
                    f"{args.entity}/{args.project}",
                    filters={"tags": {"$in": [args.tag]}},
                )
            )
        print(f"Found {len(runs)} runs to evaluate.")
        for r in runs:
            print(f"  - {r.id} {r.name} tags={r.tags}")

        for run in runs:
            try:
                res = generate_for_run(
                    run,
                    prompts=prompts,
                    out_root=out_root,
                    base_model_override=args.base_model,
                    batch_size=args.gen_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                if res is not None:
                    gen_results.append(res)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"[gen] error on {run.id}: {e}")

    if not gen_results:
        print("No runs produced completions; exiting.")
        return

    # ---------------- Phase 2: score with reward model ----------------
    print(f"\n========== Loading reward model: {args.reward_model} ==========")
    reward_model, reward_tok = build_llama_for_score(args.reward_model, device, dtype)
    for gr in gen_results:
        comp_df = pd.read_csv(gr["completions_csv"])
        comp_df["completion"] = comp_df["completion"].fillna("").astype(str)
        print(f"  [reward] scoring {len(comp_df)} rows for {gr['run_id']}")
        r_scores = score_prompts_and_responses(
            reward_model,
            reward_tok,
            prompts=comp_df["prompt"].astype(str).tolist(),
            responses=comp_df["completion"].tolist(),
            batch_size=args.score_batch_size,
        )
        comp_df["reward"] = r_scores
        comp_df.to_csv(gr["completions_csv"], index=False)
    del reward_model, reward_tok
    _free_cuda()

    # ---------------- Phase 3: score with cost model ----------------
    print(f"\n========== Loading cost model: {args.cost_model} ==========")
    cost_model, cost_tok = build_llama_for_score(args.cost_model, device, dtype)
    for gr in gen_results:
        comp_df = pd.read_csv(gr["completions_csv"])
        comp_df["completion"] = comp_df["completion"].fillna("").astype(str)
        print(f"  [cost] scoring {len(comp_df)} rows for {gr['run_id']}")
        c_scores = score_prompts_and_responses(
            cost_model,
            cost_tok,
            prompts=comp_df["prompt"].astype(str).tolist(),
            responses=comp_df["completion"].tolist(),
            batch_size=args.score_batch_size,
        )
        comp_df["cost"] = c_scores
        comp_df.to_csv(gr["completions_csv"], index=False)
    del cost_model, cost_tok
    _free_cuda()

    # ---------------- Phase 4: push metrics to W&B ----------------
    summary_rows: List[Dict[str, Any]] = []
    for gr in gen_results:
        comp_df = pd.read_csv(gr["completions_csv"])
        metrics = {
            "saferlhf/reward_mean": float(comp_df["reward"].mean()),
            "saferlhf/reward_std": float(comp_df["reward"].std(ddof=0)),
            "saferlhf/cost_mean": float(comp_df["cost"].mean()),
            "saferlhf/cost_std": float(comp_df["cost"].std(ddof=0)),
            "saferlhf/reward_rate": float((comp_df["reward"] > 0).mean()),
            "saferlhf/cost_rate": float((comp_df["cost"] > 0).mean()),
            "saferlhf/n": int(len(comp_df)),
        }
        metrics_path = gr["run_out"] / "saferlhf_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"\n[{gr['run_id']}] metrics: {metrics}")
        summary_rows.append({"run_id": gr["run_id"], **metrics})

        if args.no_wandb_push:
            continue

        run_obj = gr.get("run_obj")
        if run_obj is None:
            print(f"  [wandb] skip push for {gr['run_id']} (no W&B run object; HF-only).")
            continue

        import wandb

        w_run = wandb.init(
            entity=run_obj.entity,
            project=run_obj.project,
            id=gr["run_id"],
            resume="must",
            reinit=True,
        )
        try:
            w_run.log(metrics)
            for k, v in metrics.items():
                w_run.summary[k] = v
            table = wandb.Table(
                dataframe=comp_df[["prompt", "completion", "reward", "cost"]].astype(
                    {"prompt": str, "completion": str}
                )
            )
            w_run.log({"saferlhf/completions_table": table})

            art_out = wandb.Artifact(
                f"{gr['run_id']}-saferlhf_completions",
                type="saferlhf_completions",
                metadata={
                    "base_model": gr["base_model"],
                    "n_prompts": int(len(comp_df)),
                    "reward_model": args.reward_model,
                    "cost_model": args.cost_model,
                },
            )
            art_out.add_file(str(gr["completions_csv"]))
            w_run.log_artifact(art_out)
        finally:
            w_run.finish()

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = out_root / "saferlhf_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nSummary written to {sum_csv}\n{sum_df}")


if __name__ == "__main__":
    main()
