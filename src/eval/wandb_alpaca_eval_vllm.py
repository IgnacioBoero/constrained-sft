#!/usr/bin/env python3
"""
vLLM-based AlpacaEval sampling for a W&B run trained with LoRA adapters.

Flow:
1) Resolve run and download LoRA adapters from W&B.
2) Merge adapters into the base model and save a temporary merged model folder.
3) Run batched generation with vLLM.
4) Log outputs back to the exact same W&B run.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CHAT_TEMPLATE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# SAFETY prompt format (matches src/experiments/safety.py / src/utils.py)
# Used for Llama-2-era models trained with the SAFETY experiment.
# ---------------------------------------------------------------------------
_PROMPT_BEGIN: str = "BEGINNING OF CONVERSATION: "
_PROMPT_USER: str = "USER: {input} "
_PROMPT_ASSISTANT: str = "ASSISTANT:"


def _is_llama2_style_model(model_name: str) -> bool:
    """Return True when *model_name* is a Llama-2-era (or LLaMA-1) model
    that was trained with the SAFETY ``format_prompt`` template rather than a
    chat template."""
    name_lower = model_name.lower()
    # Explicitly Llama 2
    if "llama-2" in name_lower or "llama2" in name_lower:
        return True
    # huggyllama/llama-7b and derivatives (e.g. ihounie/huggy-*)
    if "huggyllama" in name_lower:
        return True
    if "huggy" in name_lower:
        return True
    # Generic LLaMA 1/2 pattern (but NOT Llama 3+)
    if "llama" in name_lower and not any(
        tag in name_lower for tag in ("llama-3", "llama3")
    ):
        return True
    return False


def _build_safety_prompt(instruction: str, input_text: str) -> str:
    """Build a prompt using the SAFETY format_prompt template
    (``BEGINNING OF CONVERSATION: USER: ... ASSISTANT:``)."""
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content} {input_text.strip()}"
    return f"{_PROMPT_BEGIN}{_PROMPT_USER.format(input=user_content)}{_PROMPT_ASSISTANT}"


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


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    return s


def _pick_base_model(run_obj: Any, override: Optional[str]) -> str:
    if override:
        return override
    cfg = getattr(run_obj, "config", {}) or {}
    candidates = [
        _extract_field(_extract_field(cfg, "exp"), "model_name"),
        _extract_field(cfg, "exp.model_name"),
        _extract_field(cfg, "model_name"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    raise RuntimeError("Could not infer base model from run config. Set model.base_model.")


def _artifact_aliases(artifact_obj: Any) -> set[str]:
    aliases = getattr(artifact_obj, "aliases", None) or []
    out: set[str] = set()
    for x in aliases:
        out.add(x if isinstance(x, str) else getattr(x, "name", str(x)))
    return out


def _pick_lora_artifact_for_run(run_obj: Any) -> Any:
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
            f"No logged lora_adapters artifact found on run {run_id}. "
            "Expected '<run_id>-lora_adapters:...'."
        )
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _init_wandb_exact_run(wandb_mod: Any, entity: str, project: str, run_id: str) -> Any:
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "must"
    run_obj = wandb_mod.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        job_type="alpaca_eval_sampling_vllm",
    )
    if run_obj is None:
        raise RuntimeError("wandb.init returned None.")
    got_path = f"{run_obj.entity}/{run_obj.project}/{run_obj.id}"
    want_path = f"{entity}/{project}/{run_id}"
    if got_path != want_path:
        raise RuntimeError(
            "W&B attached to unexpected run path. "
            f"Expected '{want_path}', got '{got_path}'."
        )
    return run_obj


def _init_wandb_new_run(wandb_mod: Any, entity: str, project: str, mode: str) -> Any:
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    run_obj = wandb_mod.init(
        entity=entity,
        project=project,
        job_type=f"alpaca_eval_sampling_vllm_{mode}",
    )
    if run_obj is None:
        raise RuntimeError("wandb.init returned None.")
    return run_obj


def _load_alpaca_eval_prompts(dataset_path: str, max_samples: int) -> List[Dict[str, Any]]:
    raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    prompts: List[Dict[str, Any]] = []
    if isinstance(raw, dict) and isinstance(raw.get("instructions"), list):
        instructions = raw.get("instructions", [])
        inputs = raw.get("inputs", [])
        for i, instruction in enumerate(instructions):
            inp = inputs[i] if i < len(inputs) else ""
            prompts.append({"instruction": instruction or "", "input": inp or ""})
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
        raise ValueError("Unsupported AlpacaEval format.")

    prompts = [p for p in prompts if p.get("instruction", "").strip()]
    if max_samples > 0:
        prompts = prompts[:max_samples]
    return prompts


def _build_chat_prompt(
    tok: AutoTokenizer,
    instruction: str,
    input_text: str,
    *,
    use_safety_format: bool = False,
) -> str:
    """Build the text prompt for a single AlpacaEval example.

    When *use_safety_format* is True the SAFETY ``format_prompt`` template is
    used (matching the training format for Llama-2-era models).  Otherwise the
    tokenizer's chat template is applied.
    """
    if use_safety_format:
        return _build_safety_prompt(instruction, input_text)
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\n{input_text.strip()}"
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _torch_dtype(name: str) -> torch.dtype:
    x = str(name).lower()
    if x == "bfloat16":
        return torch.bfloat16
    if x == "float16":
        return torch.float16
    if x == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _auto_tensor_parallel_size(requested: int) -> int:
    if requested and requested > 0:
        return requested
    n = torch.cuda.device_count()
    if n <= 0:
        raise RuntimeError("vLLM requires at least one CUDA device.")
    return n


def _get_default_chat_template() -> str:
    fallback_tok = AutoTokenizer.from_pretrained(
        DEFAULT_CHAT_TEMPLATE_MODEL,
        use_fast=True,
    )
    template = getattr(fallback_tok, "chat_template", None)
    if not template:
        raise ValueError(
            "Fallback tokenizer has no chat template: "
            f"{DEFAULT_CHAT_TEMPLATE_MODEL}"
        )
    return template


def _ensure_chat_template(tok: AutoTokenizer) -> AutoTokenizer:
    if getattr(tok, "chat_template", None):
        return tok
    tok.chat_template = _get_default_chat_template()
    print(
        "[wandb_alpaca_eval_vllm] tokenizer has no chat template; "
        f"using template from {DEFAULT_CHAT_TEMPLATE_MODEL}"
    )
    return tok


def _merge_lora_to_local_model(
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
    merge_dtype: str,
    *,
    skip_chat_template: bool = False,
) -> AutoTokenizer:
    merged_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if not skip_chat_template:
        tok = _ensure_chat_template(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=_torch_dtype(merge_dtype),
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))

    del peft_model
    del base
    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_tok = AutoTokenizer.from_pretrained(str(merged_dir), use_fast=True)
    if not skip_chat_template:
        merged_tok = _ensure_chat_template(merged_tok)
    return merged_tok


@hydra.main(config_path="../../configs", config_name="eval/wandb_alpaca_eval_vllm", version_base=None)
def main(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Run vLLM evaluator as a single process. "
            "Use vllm.tensor_parallel_size to use all GPUs."
        )

    from vllm import LLM, SamplingParams


    import wandb

    entity = str(cfg.wandb.entity)
    project = str(cfg.wandb.project)
    run_id = _optional_str(cfg.wandb.run_id if "run_id" in cfg.wandb else None)
    base_model_override = _optional_str(cfg.model.base_model if "model" in cfg else None)
    use_existing_run = run_id is not None

    artifact: Optional[Any] = None
    source_run_path: Optional[str] = None
    mode = "lora_merged" if use_existing_run else "base_model"

    if use_existing_run:
        source_run_path = f"{entity}/{project}/{run_id}"
        api = wandb.Api()
        run = api.run(source_run_path)
        base_model = _pick_base_model(run_obj=run, override=base_model_override)
        artifact = _pick_lora_artifact_for_run(run)
    else:
        if not base_model_override:
            raise RuntimeError(
                "wandb.run_id is null/empty, so base-model evaluation mode is used. "
                "Set model.base_model to a valid model identifier."
            )
        base_model = base_model_override

    # Detect Llama-2 / LLaMA-1 models that use the SAFETY prompt format
    use_safety_format = _is_llama2_style_model(base_model)
    if use_safety_format:
        print(
            f"[wandb_alpaca_eval_vllm] Detected Llama-2-era model ({base_model}); "
            "using SAFETY prompt format instead of chat template."
        )

    with tempfile.TemporaryDirectory(prefix="wandb_eval_vllm_") as td:
        tmp_root = Path(td)
        if use_existing_run:
            assert artifact is not None
            adapter_dir = Path(artifact.download(root=str(tmp_root / "adapter")))
            merged_dir = tmp_root / "merged_model"
            tokenizer = _merge_lora_to_local_model(
                base_model=base_model,
                adapter_dir=adapter_dir,
                merged_dir=merged_dir,
                merge_dtype=str(cfg.vllm.merge_dtype),
                skip_chat_template=use_safety_format,
            )
            model_ref = str(merged_dir)
            generator_name = "wandb_lora_merged_vllm"
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            if not use_safety_format:
                tokenizer = _ensure_chat_template(tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model_ref = base_model
            generator_name = "base_model_vllm"

        prompts = _load_alpaca_eval_prompts(
            dataset_path=to_absolute_path(str(cfg.eval.dataset_path)),
            max_samples=int(cfg.eval.max_samples),
        )
        if not prompts:
            raise RuntimeError("No prompts found after parsing and filtering dataset.")

        prompt_texts = [
            _build_chat_prompt(
                tokenizer, p["instruction"], p.get("input", ""),
                use_safety_format=use_safety_format,
            )
            for p in prompts
        ]

        greedy = bool(cfg.eval.greedy)
        temperature = 0.0 if greedy else float(cfg.eval.temperature)
        top_p = 1.0 if greedy else float(cfg.eval.top_p)
        sampling_params = SamplingParams(
            max_tokens=int(cfg.eval.max_new_tokens),
            temperature=temperature,
            top_p=top_p,
        )

        tp_size = _auto_tensor_parallel_size(int(cfg.vllm.tensor_parallel_size))
        llm = LLM(
            model=model_ref,
            tokenizer=model_ref,
            tensor_parallel_size=tp_size,
            dtype=str(cfg.vllm.dtype),
            gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
            trust_remote_code=bool(cfg.vllm.trust_remote_code),
            max_model_len=(
                int(cfg.vllm.max_model_len)
                if cfg.vllm.max_model_len is not None
                else None
            ),
            enforce_eager=bool(cfg.vllm.enforce_eager),
        )
        outputs = llm.generate(prompt_texts, sampling_params, use_tqdm=True)

        out_rows: List[Dict[str, Any]] = []
        for i, out in enumerate(outputs):
            generated = ""
            if out.outputs:
                generated = out.outputs[0].text
            out_rows.append(
                {
                    "instruction": prompts[i]["instruction"],
                    "input": prompts[i].get("input", ""),
                    "output": generated,
                    "generator": generator_name,
                }
            )

        # AlpacaEval expects a JSON list of generations rather than a metadata envelope.
        alpaca_eval_outputs = [
            {
                "instruction": row["instruction"],
                "output": row["output"],
                "generator": row["generator"],
            }
            for row in out_rows
        ]

        if use_existing_run:
            assert run_id is not None
            wb_run = _init_wandb_exact_run(wandb_mod=wandb, entity=entity, project=project, run_id=run_id)
        else:
            wb_run = _init_wandb_new_run(wandb_mod=wandb, entity=entity, project=project, mode=mode)
        eval_run_id = str(wb_run.id)
        eval_run_path = f"{wb_run.entity}/{wb_run.project}/{wb_run.id}"

        out_path = tmp_root / f"alpaca_eval_outputs_vllm_{eval_run_id}.json"
        out_path.write_text(
            json.dumps(alpaca_eval_outputs, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        local_out_dir = Path(to_absolute_path(str(cfg.eval.local_output_dir)))
        local_out_dir.mkdir(parents=True, exist_ok=True)
        local_out_path = local_out_dir / f"alpaca_eval_outputs_vllm_{eval_run_id}.json"
        local_out_path.write_text(
            json.dumps(alpaca_eval_outputs, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        table = wandb.Table(columns=["instruction", "input", "output", "generator"])
        for row in out_rows:
            table.add_data(row["instruction"], row["input"], row["output"], row["generator"])
        wandb.log(
            {
                "alpaca_eval_vllm/outputs_table": table,
                "alpaca_eval_vllm/num_outputs": len(out_rows),
                "alpaca_eval_vllm/mode": mode,
                "alpaca_eval_vllm/base_model": base_model,
                "alpaca_eval_vllm/artifact_name": getattr(artifact, "name", ""),
                "alpaca_eval_vllm/source_run_path": source_run_path or "",
                "alpaca_eval_vllm/tensor_parallel_size": tp_size,
            }
        )

        wb_art = wandb.Artifact(
            name=f"{eval_run_id}-alpaca_eval_outputs-vllm",
            type="alpaca_eval_outputs_vllm",
            metadata={
                "run_id": eval_run_id,
                "base_model": base_model,
                "mode": mode,
                "source_run_path": source_run_path,
                "num_outputs": len(out_rows),
                "generated_at_utc": _iso_now(),
                "dataset_path": str(cfg.eval.dataset_path),
                "max_new_tokens": int(cfg.eval.max_new_tokens),
                "temperature": float(cfg.eval.temperature),
                "top_p": float(cfg.eval.top_p),
                "greedy": bool(cfg.eval.greedy),
                "tensor_parallel_size": tp_size,
            },
        )
        wb_art.add_file(str(out_path))
        wandb.log_artifact(wb_art, aliases=["latest"])
        wandb.finish()

        print(f"[ok] Logged {len(out_rows)} AlpacaEval outputs with vLLM to run {eval_run_path}")
        print(f"[ok] Saved local AlpacaEval outputs to {local_out_path}")


if __name__ == "__main__":
    main()

