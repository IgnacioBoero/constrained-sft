#!/usr/bin/env python3
"""
Download the LoRA adapters for runs tagged ``safe_dpo`` in
``alelab/SAFE-long1k``, replicate the SAFETY end-of-training KL evaluation
on the *eval* split only, and push the resulting metrics back to the
matching W&B run.

KL formula (matches ``src/experiments/safety.py``):

    log_probs       = log_softmax(model(x).logits[:, :-1])
    base_log_probs  = log_softmax(base_model(x).logits[:, :-1])  # adapters off
    kl_token        = (log_probs.exp() * (log_probs - base_log_probs)).sum(-1)
    kl_token        = kl_token * response_mask[:, 1:]
    kl_per_sample   = kl_token.sum(-1) / response_mask[:, 1:].sum(-1).clamp_min(1)

The mean over the eval set is logged to W&B under ``eval/kl_to_base_*``.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import format_prompt  # noqa: E402

ENTITY = "alelab"
PROJECT = "SAFE-long1k"


def _extract_field(config: Any, key: str) -> Optional[Any]:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    if hasattr(config, key):
        return getattr(config, key)
    return None


def _pick_base_model(run_obj: Any) -> str:
    cfg = getattr(run_obj, "config", {}) or {}
    candidates = [
        _extract_field(_extract_field(cfg, "exp"), "model_name"),
        _extract_field(cfg, "exp.model_name"),
        _extract_field(cfg, "model_name"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    raise RuntimeError(f"Could not infer base model from run {run_obj.id} config.")


def _pick_dataset_name(run_obj: Any) -> str:
    cfg = getattr(run_obj, "config", {}) or {}
    candidates = [
        _extract_field(_extract_field(cfg, "exp"), "dataset"),
        _extract_field(cfg, "exp.dataset"),
        _extract_field(cfg, "dataset"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return "ihounie/SAFE-ALPACA-4"


def _pick_max_length(run_obj: Any, default: int = 512) -> int:
    cfg = getattr(run_obj, "config", {}) or {}
    train_cfg = _extract_field(cfg, "train")
    candidate = _extract_field(train_cfg, "max_length") if train_cfg is not None else None
    if candidate is None:
        candidate = _extract_field(cfg, "train.max_length")
    try:
        return int(candidate) if candidate is not None else default
    except Exception:
        return default


def _pick_data_proportion(run_obj: Any) -> float:
    cfg = getattr(run_obj, "config", {}) or {}
    train_cfg = _extract_field(cfg, "train")
    candidate = _extract_field(train_cfg, "data_proportion") if train_cfg is not None else None
    if candidate is None:
        candidate = _extract_field(cfg, "train.data_proportion")
    try:
        return float(candidate) if candidate is not None else 1.0
    except Exception:
        return 1.0


def _pick_only_unsafe_eval(run_obj: Any) -> bool:
    cfg = getattr(run_obj, "config", {}) or {}
    train_cfg = _extract_field(cfg, "train")
    candidate = _extract_field(train_cfg, "only_unsafe_eval") if train_cfg is not None else None
    if candidate is None:
        candidate = _extract_field(cfg, "train.only_unsafe_eval")
    return bool(candidate) if candidate is not None else False


def _artifact_aliases(artifact_obj: Any) -> set[str]:
    aliases = getattr(artifact_obj, "aliases", None) or []
    out: set[str] = set()
    for x in aliases:
        out.add(x if isinstance(x, str) else getattr(x, "name", str(x)))
    return out


def _pick_lora_artifact(run_obj: Any) -> Any:
    run_id = str(getattr(run_obj, "id"))
    candidates = []
    for art in run_obj.logged_artifacts():
        if getattr(art, "type", None) != "lora_adapters":
            continue
        name = str(getattr(art, "name", ""))
        if not name.startswith(f"{run_id}-lora_adapters:"):
            continue
        aliases = _artifact_aliases(art)
        has_latest = "latest" in aliases
        created = getattr(art, "created_at", None) or getattr(art, "updated_at", None)
        candidates.append((1 if has_latest else 0, created, art))
    if not candidates:
        raise RuntimeError(
            f"No 'lora_adapters' artifact found on run {run_id} "
            f"(expected '<run_id>-lora_adapters:...')."
        )
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def _build_eval_dataset(
    dataset_name: str,
    *,
    data_proportion: float,
    only_unsafe_eval: bool,
    seed: int,
):
    """Replicate ``SAFETY.load_datasets`` for the validation split."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    ev_raw = ds["validation"]
    ev_size = int(data_proportion * len(ev_raw))
    if ev_size <= 0:
        ev_size = len(ev_raw)
    ev_raw = ev_raw.select(range(ev_size))

    if only_unsafe_eval:
        ev_raw = ev_raw.filter(
            lambda x: _to_bool(x.get("safety_label", False)) is False
        )

    # Match the deterministic shuffle used during training (eval order does
    # not affect aggregate KL stats, but keep it for parity).
    ev_raw = ev_raw.shuffle(seed=seed)
    return ev_raw


def _preprocess_sample(
    sample: Dict[str, Any],
    tok,
    max_length: int,
):
    """Replicates ``SAFETY.preprocessing_fn`` for a single sample."""
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    if pad_id is None:
        tok.pad_token = tok.eos_token
        pad_id = tok.pad_token_id

    instruction = sample.get("instruction") or ""
    inp = sample.get("input") or ""
    input_text = (instruction + " " + inp) if inp else instruction

    answer = sample.get("output", "") or ""
    prompt = format_prompt(input=input_text, eos_token=tok.eos_token)
    text = prompt + answer

    prompt_ids = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_length - 1,
    )["input_ids"][0]
    len_prompt_ids = len(prompt_ids)

    input_ids = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_length - 1,
    )["input_ids"][0]

    if input_ids.numel() == 0 or input_ids[-1].item() != eos_id:
        input_ids = torch.cat(
            [input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)]
        )

    seq_len = input_ids.shape[0]
    if seq_len < max_length:
        pad_len = max_length - seq_len
        input_ids = torch.cat(
            [input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)]
        )
    else:
        input_ids = input_ids[:max_length]
        seq_len = max_length

    response_mask = torch.zeros(max_length, dtype=torch.bool)
    start = min(len_prompt_ids, seq_len)
    response_mask[start:seq_len] = True

    # Attention mask matches collator: True over real tokens (prompt + response + EOS).
    attention_mask = torch.zeros(max_length, dtype=torch.bool)
    attention_mask[:seq_len] = True

    # Convention used in the logged metrics: "safe" = row with safety_label == False
    # (i.e. the unsafe prompts; flipped relative to the dataset column's literal meaning
    # to match downstream reporting conventions).
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "safe": not _to_bool(sample.get("safety_label", False)),
    }


def _kl_to_base_for_batch(
    model,
    batch_input_ids: torch.Tensor,
    batch_attention_mask: torch.Tensor,
    batch_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Average KL(πθ || π0) over response tokens for each sample in the batch."""
    with torch.no_grad():
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )
        logits = outputs.logits[:, :-1].float()  # KL in fp32 for stability
        log_probs = F.log_softmax(logits, dim=-1)
        del logits, outputs

        if hasattr(model, "disable_adapter"):
            ctx = model.disable_adapter()
        elif hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
            ctx = model.module.disable_adapter()
        else:
            raise RuntimeError(
                "KL evaluation requires PEFT LoRA adapters with `.disable_adapter()`."
            )
        with ctx:
            base_outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
            base_logits = base_outputs.logits[:, :-1].float()
            base_log_probs = F.log_softmax(base_logits, dim=-1)
            del base_logits, base_outputs

        probs = log_probs.exp()
        kl_token = (probs * (log_probs - base_log_probs)).sum(dim=-1)
        del probs, base_log_probs, log_probs

        resp_mask_shifted = batch_response_mask[:, 1:].to(kl_token.dtype)
        kl_token = kl_token * resp_mask_shifted
        num_tokens = resp_mask_shifted.sum(dim=-1).clamp_min(1)
        kl_per_sample = kl_token.sum(dim=-1) / num_tokens
        return kl_per_sample.detach().cpu()


def _evaluate_run(
    run_id: str,
    *,
    batch_size: int,
    dtype_name: str,
    max_samples: Optional[int],
    dry_run: bool,
) -> None:
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    base_model_name = _pick_base_model(run)
    dataset_name = _pick_dataset_name(run)
    max_length = _pick_max_length(run)
    data_proportion = _pick_data_proportion(run)
    only_unsafe_eval = _pick_only_unsafe_eval(run)
    seed_cfg = _extract_field(_extract_field(run.config, "train"), "seed")
    try:
        seed = int(seed_cfg) if seed_cfg is not None else 42
    except Exception:
        seed = 42

    print(f"[kl-eval] run_id={run_id} name={run.name}")
    print(f"[kl-eval] base_model={base_model_name}")
    print(f"[kl-eval] dataset={dataset_name}")
    print(
        f"[kl-eval] max_length={max_length} data_proportion={data_proportion} "
        f"only_unsafe_eval={only_unsafe_eval} seed={seed}"
    )

    artifact = _pick_lora_artifact(run)
    print(f"[kl-eval] LoRA artifact: {artifact.name}")

    eval_ds = _build_eval_dataset(
        dataset_name,
        data_proportion=data_proportion,
        only_unsafe_eval=only_unsafe_eval,
        seed=seed,
    )
    if max_samples is not None and max_samples > 0:
        eval_ds = eval_ds.select(range(min(max_samples, len(eval_ds))))
    print(f"[kl-eval] eval samples: {len(eval_ds)}")

    if dry_run:
        print("[kl-eval] dry-run; exiting before model load.")
        return

    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = max_length

    print(f"[kl-eval] loading base model in {dtype_name} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)

    with tempfile.TemporaryDirectory(prefix=f"safe_dpo_kl_{run_id}_") as td:
        adapter_dir = Path(artifact.download(root=str(Path(td) / "adapter")))
        print(f"[kl-eval] adapter downloaded to {adapter_dir}")
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()

        kl_values: List[float] = []
        safety_labels: List[bool] = []

        n = len(eval_ds)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            samples = [
                _preprocess_sample(eval_ds[i], tok, max_length)
                for i in range(start, end)
            ]
            input_ids = torch.stack([s["input_ids"] for s in samples], dim=0).to(device)
            attn_mask = torch.stack([s["attention_mask"] for s in samples], dim=0).to(device)
            resp_mask = torch.stack([s["response_mask"] for s in samples], dim=0).to(device)

            kl_batch = _kl_to_base_for_batch(model, input_ids, attn_mask, resp_mask)
            kl_values.extend(kl_batch.tolist())
            safety_labels.extend(s["safe"] for s in samples)

            if start % (batch_size * 5) == 0 or end == n:
                running_mean = float(np.mean(kl_values)) if kl_values else 0.0
                print(
                    f"[kl-eval] processed {end}/{n} | running mean KL = {running_mean:.6f}"
                )

        kl_arr = np.asarray(kl_values, dtype=np.float64)
        safety_arr = np.asarray(safety_labels, dtype=bool)

        stats: Dict[str, float] = {
            "eval/kl_to_base_mean": float(np.mean(kl_arr)),
            "eval/kl_to_base_std": float(np.std(kl_arr)),
            "eval/kl_to_base_min": float(np.min(kl_arr)),
            "eval/kl_to_base_max": float(np.max(kl_arr)),
            "eval/kl_to_base_median": float(np.median(kl_arr)),
            "eval/kl_to_base_q90": float(np.quantile(kl_arr, 0.9)),
            "eval/kl_to_base_q99": float(np.quantile(kl_arr, 0.99)),
            "eval/kl_to_base_n_samples": int(kl_arr.size),
        }
        if safety_arr.any():
            stats["eval/kl_to_base_safe_mean"] = float(np.mean(kl_arr[safety_arr]))
        if (~safety_arr).any():
            stats["eval/kl_to_base_unsafe_mean"] = float(np.mean(kl_arr[~safety_arr]))

        print("[kl-eval] stats:")
        for k, v in stats.items():
            print(f"  {k} = {v}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        os.environ["WANDB_ENTITY"] = ENTITY
        os.environ["WANDB_PROJECT"] = PROJECT
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "must"
        wb_run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            id=run_id,
            resume="must",
            job_type="kl_to_base_eval_only",
        )
        if wb_run is None:
            raise RuntimeError("wandb.init returned None.")

        wb_run.log(stats)
        wb_run.summary.update(stats)
        wandb.finish()
        print(f"[kl-eval] logged KL stats to wandb run {ENTITY}/{PROJECT}/{run_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="safe_dpo",
        help="W&B tag used to filter runs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id (overrides --tag).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (training used fp16).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of eval samples.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    if args.run_id:
        run_ids = [args.run_id]
    else:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"tags": {"$in": [args.tag]}},
        )
        run_ids = [r.id for r in runs if r.state == "finished"]

    if not run_ids:
        print(f"[kl-eval] No qualifying runs for tag={args.tag!r}.")
        return

    print(f"[kl-eval] Will evaluate {len(run_ids)} run(s): {run_ids}")
    failed: List[str] = []
    for run_id in run_ids:
        try:
            _evaluate_run(
                run_id,
                batch_size=args.batch_size,
                dtype_name=args.dtype,
                max_samples=args.max_samples,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[kl-eval] run {run_id} failed: {exc}")
            failed.append(run_id)

    if failed:
        sys.exit(f"Failures: {failed}")


if __name__ == "__main__":
    main()
