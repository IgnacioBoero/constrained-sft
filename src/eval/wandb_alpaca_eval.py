#!/usr/bin/env python3
"""
Download LoRA adapters logged by `src/train.py`, merge into the base model, run
distributed sampling on AlpacaEval prompts, and log generations back to the same
Weights & Biases run.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def _is_dist_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist_on() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist_on() else 1


def _is_main() -> bool:
    return _rank() == 0


def _setup_distributed() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def _cleanup_distributed() -> None:
    if _is_dist_on():
        dist.barrier()
        dist.destroy_process_group()


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_field(config: Any, key: str) -> Optional[Any]:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    if hasattr(config, key):
        return getattr(config, key)
    return None


def _pick_base_model(run_obj: Any, override: Optional[str]) -> str:
    if override:
        return override

    cfg = getattr(run_obj, "config", {}) or {}
    # W&B often flattens nested config keys, so check both styles.
    candidates = [
        _extract_field(_extract_field(cfg, "exp"), "model_name"),
        _extract_field(cfg, "exp.model_name"),
        _extract_field(cfg, "model_name"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    raise RuntimeError(
        "Could not infer base model from run config. "
        "Pass --base_model explicitly."
    )


def _artifact_aliases(artifact_obj: Any) -> set[str]:
    aliases = getattr(artifact_obj, "aliases", None) or []
    out: set[str] = set()
    for x in aliases:
        if isinstance(x, str):
            out.add(x)
        else:
            out.add(getattr(x, "name", str(x)))
    return out


def _pick_lora_artifact_for_run(run_obj: Any) -> Any:
    run_id = str(getattr(run_obj, "id"))
    candidates = []
    for art in run_obj.logged_artifacts():
        if getattr(art, "type", None) != "lora_adapters":
            continue
        name = str(getattr(art, "name", ""))
        # Saved in train.py as f"{wandb.run.id}-lora_adapters"
        if not name.startswith(f"{run_id}-lora_adapters:"):
            continue
        aliases = _artifact_aliases(art)
        has_latest = "latest" in aliases
        created = getattr(art, "created_at", None) or getattr(art, "updated_at", None)
        candidates.append((1 if has_latest else 0, created, art))

    if not candidates:
        raise RuntimeError(
            f"No logged lora_adapters artifact found on run {run_id}. "
            "Expected a name like '<run_id>-lora_adapters:...'."
        )
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _init_wandb_exact_run(wandb_mod: Any, entity: str, project: str, run_id: str) -> Any:
    """
    Attach to an existing run and fail fast on any mismatch.
    This prevents accidental logging to a new run or the wrong entity/project.
    """
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "must"

    run_obj = wandb_mod.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        job_type="alpaca_eval_sampling",
    )
    if run_obj is None:
        raise RuntimeError("wandb.init returned None on rank 0.")

    got_path = f"{run_obj.entity}/{run_obj.project}/{run_obj.id}"
    want_path = f"{entity}/{project}/{run_id}"
    if got_path != want_path:
        raise RuntimeError(
            "W&B attached to unexpected run path. "
            f"Expected '{want_path}', got '{got_path}'."
        )
    return run_obj


def _load_alpaca_eval_prompts(dataset_path: str, max_samples: int) -> List[Dict[str, Any]]:
    raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    prompts: List[Dict[str, Any]] = []

    # Supported formats:
    # 1) {"instructions": [...], "inputs": [...]}
    # 2) [{"instruction": ..., "input": ...}, ...]
    if isinstance(raw, dict) and isinstance(raw.get("instructions"), list):
        instructions = raw.get("instructions", [])
        inputs = raw.get("inputs", [])
        for i, instruction in enumerate(instructions):
            inp = inputs[i] if i < len(inputs) else ""
            prompts.append(
                {
                    "instruction": instruction or "",
                    "input": inp or "",
                }
            )
    elif isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict):
                prompts.append(
                    {
                        "instruction": row.get("instruction") or row.get("prompt") or "",
                        "input": row.get("input") or "",
                    }
                )
    else:
        raise ValueError(
            "Unsupported AlpacaEval dataset format. "
            "Expected dict with `instructions` or list of prompt dicts."
        )

    prompts = [p for p in prompts if p.get("instruction", "").strip()]
    if max_samples > 0:
        prompts = prompts[:max_samples]
    return prompts


def _build_chat_prompt(tok: AutoTokenizer, instruction: str, input_text: str) -> str:
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\n{input_text.strip()}"
    messages = [{"role": "user", "content": user_content}]
    # Matches dpo_kl chat formatting style: chat template + generation prompt.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class GenCfg:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


def _generate_local_rows(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[Dict[str, Any]],
    gen_cfg: GenCfg,
) -> List[Dict[str, Any]]:
    rank = _rank()
    world = _world_size()
    local_indices = list(range(rank, len(prompts), world))
    rows: List[Dict[str, Any]] = []

    iterator = tqdm(
        local_indices,
        desc=f"[rank {rank}] sampling",
        leave=False,
    )
    for idx in iterator:
        sample = prompts[idx]
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        prompt_text = _build_chat_prompt(tok, instruction=instruction, input_text=input_text)

        tokenized = tok(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            out = model.generate(
                **tokenized,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen_ids = out[0][tokenized["input_ids"].shape[1] :]
        output_text = tok.decode(gen_ids, skip_special_tokens=True)
        rows.append(
            {
                "index": idx,
                "instruction": instruction,
                "input": input_text,
                "prompt": prompt_text,
                "output": output_text,
                "generator": "wandb_lora_merged",
            }
        )
    return rows


@hydra.main(config_path="../../configs", config_name="eval/wandb_alpaca_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = _setup_distributed()
    rank = _rank()

    try:
        import wandb

        api = wandb.Api()
        entity = str(cfg.wandb.entity)
        project = str(cfg.wandb.project)
        run_id = str(cfg.wandb.run_id)
        run_path = f"{entity}/{project}/{run_id}"
        run = api.run(run_path)
        base_model_override = cfg.model.base_model if "model" in cfg else None
        base_model = _pick_base_model(run_obj=run, override=base_model_override)
        artifact = _pick_lora_artifact_for_run(run)

        # Rank-local temp to avoid collisions.
        tmp_root = Path(tempfile.mkdtemp(prefix=f"wandb_lora_eval_r{rank}_"))
        adapter_dir = Path(artifact.download(root=str(tmp_root)))

        # dtype alignment with training defaults (bf16 when available).
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        model = peft_model.merge_and_unload()
        model.to(device)
        model.eval()

        prompts = _load_alpaca_eval_prompts(
            dataset_path=to_absolute_path(str(cfg.eval.dataset_path)),
            max_samples=int(cfg.eval.max_samples),
        )
        if not prompts:
            raise RuntimeError("No prompts found after parsing and filtering dataset.")

        gen_cfg = GenCfg(
            max_new_tokens=int(cfg.eval.max_new_tokens),
            temperature=float(cfg.eval.temperature),
            top_p=float(cfg.eval.top_p),
            do_sample=not bool(cfg.eval.greedy),
        )
        if bool(cfg.eval.greedy):
            gen_cfg.temperature = 1.0
            gen_cfg.top_p = 1.0

        rows_local = _generate_local_rows(model=model, tok=tok, prompts=prompts, gen_cfg=gen_cfg)

        if _is_dist_on():
            gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(_world_size())]
            dist.all_gather_object(gathered, rows_local)
            rows_all: List[Dict[str, Any]] = []
            for part in gathered:
                if part:
                    rows_all.extend(part)
        else:
            rows_all = rows_local

        if _is_main():
            rows_all.sort(key=lambda x: int(x["index"]))
            out_rows = [
                {
                    "instruction": r["instruction"],
                    "input": r["input"],
                    "output": r["output"],
                    "generator": r["generator"],
                }
                for r in rows_all
            ]
            payload = {
                "entity": entity,
                "project": project,
                "run_id": run_id,
                "base_model": base_model,
                "artifact_name": getattr(artifact, "name", ""),
                "generated_at_utc": _iso_now(),
                "max_samples": int(cfg.eval.max_samples),
                "num_outputs": len(out_rows),
                "outputs": out_rows,
            }

            _ = _init_wandb_exact_run(
                wandb_mod=wandb,
                entity=entity,
                project=project,
                run_id=run_id,
            )

            tmp_out = tmp_root / f"alpaca_eval_outputs_{run_id}.json"
            tmp_out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            table = wandb.Table(columns=["instruction", "input", "output", "generator"])
            for row in out_rows:
                table.add_data(
                    row["instruction"],
                    row["input"],
                    row["output"],
                    row["generator"],
                )
            wandb.log(
                {
                    "alpaca_eval/outputs_table": table,
                    "alpaca_eval/num_outputs": len(out_rows),
                    "alpaca_eval/base_model": base_model,
                    "alpaca_eval/artifact_name": getattr(artifact, "name", ""),
                }
            )

            wb_art = wandb.Artifact(
                name=f"{run_id}-alpaca_eval_outputs",
                type="alpaca_eval_outputs",
                metadata={
                    "run_id": run_id,
                    "base_model": base_model,
                    "num_outputs": len(out_rows),
                    "dataset_path": str(cfg.eval.dataset_path),
                    "max_new_tokens": int(cfg.eval.max_new_tokens),
                    "temperature": float(cfg.eval.temperature),
                    "top_p": float(cfg.eval.top_p),
                    "greedy": bool(cfg.eval.greedy),
                },
            )
            wb_art.add_file(str(tmp_out))
            wandb.log_artifact(wb_art, aliases=["latest"])
            wandb.finish()

            print(f"[ok] Logged {len(out_rows)} AlpacaEval outputs to run {run_path}")

    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()

