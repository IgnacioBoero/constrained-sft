#!/usr/bin/env python3
"""
Run BFCL v2 evaluation (vLLM backend) for a given W&B run id.

This script:
1) Resolves base model + LoRA adapters from a W&B run.
2) Merges LoRA into a temporary local model.
3) Runs official BFCL generation/evaluation for selected categories only.
4) Logs BFCL metrics and artifacts back to the same W&B run.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def _as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


def _torch_dtype(name: str) -> torch.dtype:
    x = str(name).lower()
    if x == "bfloat16":
        return torch.bfloat16
    if x == "float16":
        return torch.float16
    if x == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _merge_lora_to_local_model(
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
    merge_dtype: str,
) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
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


def _auto_tensor_parallel_size(requested: int) -> int:
    if requested and requested > 0:
        return requested
    n = torch.cuda.device_count()
    if n <= 0:
        raise RuntimeError("vLLM requires at least one CUDA device.")
    return n


def _sanitize_model_name_for_path(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_") or "model"


def _is_xlam_model_name(model_name: str) -> bool:
    lowered = str(model_name).lower()
    return ("salesforce/xlam" in lowered) or ("salesforce/llama-xlam" in lowered)


def _load_bfcl_model_mapping() -> dict[str, Any]:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    return MODEL_CONFIG_MAPPING


def _resolve_bfcl_model_name(cfg: DictConfig, base_model: str) -> tuple[str, Any]:
    requested = _optional_str(getattr(cfg.bfcl, "model_name", None))
    requested_lower = (requested or "").strip().lower()
    auto_mode = requested is None or requested_lower in {"auto", "none", "null"}
    candidate = base_model if auto_mode else requested

    mapping = _load_bfcl_model_mapping()
    if candidate not in mapping:
        if auto_mode:
            raise RuntimeError(
                "Auto BFCL model-name resolution failed: base model "
                f"'{base_model}' is not present in BFCL MODEL_CONFIG_MAPPING. "
                "Set bfcl.model_name to an explicit supported registry key."
            )
        raise RuntimeError(
            f"bfcl.model_name '{candidate}' is not present in BFCL MODEL_CONFIG_MAPPING. "
            "Use an exact BFCL registry key (for xLAM, e.g. Salesforce/xLAM-2-1b-fc-r)."
        )
    return candidate, mapping[candidate]


def _expected_xlam_handler_names(model_name: str) -> Optional[tuple[str, ...]]:
    lowered = str(model_name).lower()
    if "salesforce/xlam-2-" in lowered:
        return (
            "SalesforceQwenHandler",
            "SalesforceXlamDPOPromptHandler",
        )
    if "salesforce/llama-xlam" in lowered:
        return (
            "SalesforceLlamaHandler",
            "SalesforceXlamDPOPromptHandler",
        )
    return None


def _extract_summary_field(run_obj: Any, key: str) -> Optional[Any]:
    summary = getattr(run_obj, "summary", None)
    if summary is None:
        return None
    if isinstance(summary, dict):
        return summary.get(key)
    if hasattr(summary, "get"):
        return summary.get(key)
    return None


def _as_bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _run_command_attempts(
    attempts: list[list[str]],
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, str]:
    last_err = None
    for cmd in attempts:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return {
                "command": " ".join(cmd),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        last_err = (
            f"cmd={' '.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{(proc.stdout or '')[-8000:]}\n"
            f"stderr:\n{(proc.stderr or '')[-8000:]}\n"
        )
    raise RuntimeError(f"Command failed.\n{last_err}")


def _run_bfcl_generate(
    *,
    project_root: Path,
    model_name: str,
    categories: list[str],
    local_model_path: Path,
    num_gpus: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> dict[str, str]:
    categories_csv = ",".join(categories)
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(project_root)
    env["BFCL_TOP_P"] = str(float(top_p))
    env["BFCL_MAX_OUTPUT_TOKENS"] = str(int(max_output_tokens))

    common = [
        "--model",
        model_name,
        "--test-category",
        categories_csv,
    ]
    cli_attempt = [
        "bfcl",
        "generate",
        *common,
        "--backend",
        "vllm",
        "--num-gpus",
        str(num_gpus),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--temperature",
        str(float(temperature)),
        "--local-model-path",
        str(local_model_path),
    ]

    return _run_command_attempts([cli_attempt], cwd=project_root, env=env)


def _supported_bfcl_test_names() -> tuple[set[str], set[str]]:
    import bfcl_eval.utils as bfcl_utils

    categories = set(getattr(bfcl_utils, "ALL_CATEGORIES", []))
    collections = set(getattr(bfcl_utils, "TEST_COLLECTION_MAPPING", {}).keys())
    return categories, collections


def _normalize_test_categories(requested: list[str]) -> list[str]:
    """
    Map user-friendly aliases to categories supported by the installed BFCL version.
    """
    categories, collections = _supported_bfcl_test_names()
    supported = categories | collections

    alias_map: dict[str, list[str]] = {
        # Legacy / shorthand aliases.
        "live_ast": [
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ],
        "hallucination_irrelevance": ["live_irrelevance", "irrelevance"],
        "hallucination_relevance": ["live_relevance"],
    }

    out: list[str] = []
    for item in requested:
        if item in supported:
            out.append(item)
            continue
        if item in alias_map:
            mapped = [x for x in alias_map[item] if x in supported]
            if not mapped:
                raise RuntimeError(
                    f"Alias '{item}' could not be mapped for this BFCL version. "
                    f"Supported categories={sorted(categories)} collections={sorted(collections)}"
                )
            # For irrelevance alias, prefer live irrelevance when available.
            if item == "hallucination_irrelevance" and "live_irrelevance" in mapped:
                out.append("live_irrelevance")
            else:
                out.extend(mapped)
            continue
        raise RuntimeError(
            f"Unsupported BFCL test category '{item}'. "
            f"Supported categories={sorted(categories)} collections={sorted(collections)}"
        )
    return _dedupe_keep_order(out)


def _validate_bfcl_runtime() -> None:
    if shutil.which("bfcl") is None:
        raise RuntimeError(
            "BFCL CLI `bfcl` is not found in PATH for this Python environment. "
            "Install bfcl-eval and ensure its console scripts are available."
        )
    if importlib.util.find_spec("bfcl_eval") is None:
        raise RuntimeError(
            "bfcl_eval is not installed in the current Python environment. "
            "Install it first (e.g. `pip install bfcl-eval`)."
        )
    # bfcl_eval imports qwen_agent at startup in current releases, which requires soundfile.
    if importlib.util.find_spec("soundfile") is None:
        raise RuntimeError(
            "Missing dependency `soundfile`, required by installed bfcl_eval runtime. "
            "Install it in this environment (e.g. `pip install soundfile`)."
        )


def _run_bfcl_evaluate(
    *,
    project_root: Path,
    model_name: str,
    categories: list[str],
) -> dict[str, str]:
    categories_csv = ",".join(categories)
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(project_root)
    common = [
        "--model",
        model_name,
        "--test-category",
        categories_csv,
    ]
    cli_attempt = ["bfcl", "evaluate", *common]
    return _run_command_attempts([cli_attempt], cwd=project_root, env=env)


def _find_primary_score_json(project_root: Path) -> Path:
    candidates = sorted((project_root / "score").rglob("*score.json"))
    if not candidates:
        candidates = sorted((project_root / "score").rglob("*.json"))
    if not candidates:
        raise RuntimeError(f"No BFCL score JSON files found under {project_root / 'score'}")
    return candidates[-1]


def _sanitize_wandb_key_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def _flatten_json_for_wandb(
    obj: Any,
    *,
    prefix: str,
    out: Dict[str, Any],
) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = _sanitize_wandb_key_part(str(key))
            child = f"{prefix}/{k}" if prefix else k
            _flatten_json_for_wandb(value, prefix=child, out=out)
        return
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}/idx_{idx}"
            _flatten_json_for_wandb(value, prefix=child, out=out)
        return
    if isinstance(obj, (int, float, bool)):
        out[prefix] = obj
        return
    if obj is None:
        out[prefix] = "null"
        return
    # Keep strings/raw leaf values so all library outputs are preserved.
    out[prefix] = str(obj)


def _collect_all_bfcl_metrics(project_root: Path) -> Dict[str, Any]:
    score_root = project_root / "score"
    if not score_root.exists():
        raise RuntimeError(f"BFCL score directory not found: {score_root}")

    metrics: Dict[str, Any] = {}
    json_files = sorted(score_root.rglob("*.json"))
    if not json_files:
        raise RuntimeError(f"No BFCL JSON outputs found under {score_root}")

    def _load_json_or_jsonl(path: Path) -> Any:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback for JSONL / concatenated JSON objects.
            rows: list[Any] = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    # Preserve raw line so we don't lose BFCL outputs.
                    rows.append({"_raw_line": stripped})
            if rows:
                return rows
            # Last resort: keep raw payload.
            return {"_raw_text": text}

    for json_path in json_files:
        rel = json_path.relative_to(score_root)
        rel_key = _sanitize_wandb_key_part(str(rel).replace("/", "__"))
        obj = _load_json_or_jsonl(json_path)
        root_prefix = f"bfcl_v2_vllm/library_metrics/{rel_key}"
        _flatten_json_for_wandb(obj, prefix=root_prefix, out=metrics)

    return metrics


def _collect_prompt_alignment_checks(
    run_obj: Any,
    *,
    base_model: str,
    bfcl_model_name: str,
    bfcl_handler_name: str,
) -> dict[str, Any]:
    cfg = getattr(run_obj, "config", {}) or {}
    train_cfg = _extract_field(cfg, "train")
    train_dataset = _extract_field(train_cfg, "dataset")
    use_xlam_prompt_format = _extract_field(train_cfg, "use_xlam_prompt_format")
    strict_prompt = _extract_field(train_cfg, "when2call_strict_prompt")
    dataset_name = str(train_dataset).lower() if train_dataset is not None else ""
    is_when2call_training = "when2call" in dataset_name

    is_xlam_run = _is_xlam_model_name(base_model)
    # In this project xLAM runs force xLAM formatting by model identity; explicit
    # use_xlam_prompt_format=true is still the clearest indicator in config.
    training_uses_xlam_format = (
        _as_bool_or_default(use_xlam_prompt_format, default=is_xlam_run)
        if is_when2call_training
        else _as_bool_or_default(use_xlam_prompt_format, default=is_xlam_run)
    )
    training_uses_strict_prompt = _as_bool_or_default(strict_prompt, default=False)

    expected_handlers = _expected_xlam_handler_names(base_model)
    bfcl_uses_expected_xlam_handler = (
        (expected_handlers is None) or (bfcl_handler_name in expected_handlers)
    )
    bfcl_model_matches_training_model = bfcl_model_name == base_model

    when2call_eval_task_name = _extract_summary_field(run_obj, "when2call_lm_eval_vllm/task_name")
    when2call_eval_task_matches_xlam: Optional[bool] = None
    if is_xlam_run:
        if when2call_eval_task_name is None:
            when2call_eval_task_matches_xlam = None
        else:
            when2call_eval_task_matches_xlam = (
                str(when2call_eval_task_name).strip().lower() == "when2call-xlam"
            )

    alignment_ok = (
        bfcl_model_matches_training_model
        and bfcl_uses_expected_xlam_handler
        and (training_uses_xlam_format or not is_xlam_run)
        and (not training_uses_strict_prompt or not is_xlam_run)
        and (when2call_eval_task_matches_xlam is not False)
    )

    note_parts = [
        f"is_xlam_run={is_xlam_run}",
        f"bfcl_model_matches_training_model={bfcl_model_matches_training_model}",
        f"bfcl_handler={bfcl_handler_name}",
    ]
    if expected_handlers is not None:
        note_parts.append(f"expected_handlers={','.join(expected_handlers)}")
        note_parts.append(f"bfcl_uses_expected_xlam_handler={bfcl_uses_expected_xlam_handler}")
    note_parts.append(f"training_uses_xlam_format={training_uses_xlam_format}")
    note_parts.append(f"training_uses_strict_prompt={training_uses_strict_prompt}")
    if when2call_eval_task_name is not None:
        note_parts.append(f"when2call_eval_task={when2call_eval_task_name}")
    if when2call_eval_task_matches_xlam is not None:
        note_parts.append(
            f"when2call_eval_task_matches_xlam={when2call_eval_task_matches_xlam}"
        )

    return {
        "alignment_ok": bool(alignment_ok),
        "alignment_note": "; ".join(note_parts),
        "is_xlam_run": bool(is_xlam_run),
        "training_uses_xlam_format": bool(training_uses_xlam_format),
        "training_uses_strict_prompt": bool(training_uses_strict_prompt),
        "bfcl_model_matches_training_model": bool(bfcl_model_matches_training_model),
        "bfcl_uses_expected_xlam_handler": bool(bfcl_uses_expected_xlam_handler),
        "bfcl_handler_name": str(bfcl_handler_name),
        "expected_xlam_handler_name": (
            ",".join(expected_handlers) if expected_handlers is not None else ""
        ),
        "when2call_eval_task_name": (
            str(when2call_eval_task_name) if when2call_eval_task_name is not None else ""
        ),
        "when2call_eval_task_matches_xlam": (
            "unknown"
            if when2call_eval_task_matches_xlam is None
            else str(bool(when2call_eval_task_matches_xlam)).lower()
        ),
    }


@hydra.main(config_path="../../configs", config_name="eval/wandb_bfcl_v2_vllm", version_base=None)
def main(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Run this evaluator as a single process. "
            "Use vllm.tensor_parallel_size / bfcl.num_gpus for multi-GPU."
        )

    import wandb
    _validate_bfcl_runtime()

    entity = str(cfg.wandb.entity)
    project = str(cfg.wandb.project)
    run_id = _optional_str(cfg.wandb.run_id if "run_id" in cfg.wandb else None)
    if run_id is None:
        raise RuntimeError("wandb.run_id is required.")

    source_run_path = f"{entity}/{project}/{run_id}"
    api = wandb.Api()
    source_run = api.run(source_run_path)

    base_model_override = _optional_str(cfg.model.base_model if "model" in cfg else None)
    base_model = _pick_base_model(source_run, override=base_model_override)
    artifact = _pick_lora_artifact_for_run(source_run)

    requested_categories = _as_list_of_str(cfg.bfcl.test_categories)
    if not requested_categories:
        raise RuntimeError("bfcl.test_categories must contain at least one category.")
    categories = _normalize_test_categories(requested_categories)

    bfcl_model_name, bfcl_model_config = _resolve_bfcl_model_name(cfg, base_model=base_model)
    bfcl_handler_name = getattr(bfcl_model_config.model_handler, "__name__", "unknown_handler")

    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "must"
    wb_run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        job_type="bfcl_v2_vllm",
    )
    if wb_run is None:
        raise RuntimeError("wandb.init returned None.")

    with tempfile.TemporaryDirectory(prefix="wandb_bfcl_v2_vllm_") as td:
        tmp_root = Path(td)
        adapter_dir = Path(artifact.download(root=str(tmp_root / "adapter")))
        merged_dir = tmp_root / "merged_model"
        _merge_lora_to_local_model(
            base_model=base_model,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            merge_dtype=str(cfg.vllm.merge_dtype),
        )

        local_out_dir = Path(to_absolute_path(str(cfg.bfcl.local_output_dir)))
        local_out_dir.mkdir(parents=True, exist_ok=True)
        run_out_dir = local_out_dir / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)

        bfcl_project_root = run_out_dir / "bfcl_project_root"
        bfcl_project_root.mkdir(parents=True, exist_ok=True)

        num_gpus = int(getattr(cfg.bfcl, "num_gpus", 0) or 0)
        if num_gpus <= 0:
            num_gpus = _auto_tensor_parallel_size(int(cfg.vllm.tensor_parallel_size))

        gen_result = _run_bfcl_generate(
            project_root=bfcl_project_root,
            model_name=bfcl_model_name,
            categories=categories,
            local_model_path=merged_dir,
            num_gpus=num_gpus,
            gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
            temperature=float(getattr(cfg.bfcl, "temperature", 0.0)),
            top_p=float(getattr(cfg.bfcl, "top_p", 1.0)),
            max_output_tokens=int(getattr(cfg.bfcl, "max_output_tokens", 2048)),
        )
        (run_out_dir / "bfcl_generate_stdout.log").write_text(gen_result["stdout"], encoding="utf-8")
        (run_out_dir / "bfcl_generate_stderr.log").write_text(gen_result["stderr"], encoding="utf-8")
        (run_out_dir / "bfcl_generate_command.txt").write_text(gen_result["command"] + "\n", encoding="utf-8")

        eval_result = _run_bfcl_evaluate(
            project_root=bfcl_project_root,
            model_name=bfcl_model_name,
            categories=categories,
        )
        (run_out_dir / "bfcl_evaluate_stdout.log").write_text(eval_result["stdout"], encoding="utf-8")
        (run_out_dir / "bfcl_evaluate_stderr.log").write_text(eval_result["stderr"], encoding="utf-8")
        (run_out_dir / "bfcl_evaluate_command.txt").write_text(eval_result["command"] + "\n", encoding="utf-8")

        score_path = _find_primary_score_json(bfcl_project_root)
        all_bfcl_metrics = _collect_all_bfcl_metrics(bfcl_project_root)

        alignment = _collect_prompt_alignment_checks(
            source_run,
            base_model=base_model,
            bfcl_model_name=bfcl_model_name,
            bfcl_handler_name=bfcl_handler_name,
        )

        payload = {
            "bfcl_v2_vllm/base_model": base_model,
            "bfcl_v2_vllm/source_run_path": source_run_path,
            "bfcl_v2_vllm/model_name_for_bfcl": bfcl_model_name,
            "bfcl_v2_vllm/bfcl_handler_name": bfcl_handler_name,
            "bfcl_v2_vllm/test_categories_requested": ",".join(requested_categories),
            "bfcl_v2_vllm/test_categories_resolved": ",".join(categories),
            "bfcl_v2_vllm/num_gpus": num_gpus,
            "bfcl_v2_vllm/top_p": float(getattr(cfg.bfcl, "top_p", 1.0)),
            "bfcl_v2_vllm/max_output_tokens": int(getattr(cfg.bfcl, "max_output_tokens", 2048)),
            "bfcl_v2_vllm/metric_source": "score/*.json (all BFCL library outputs)",
            "bfcl_v2_vllm/prompt_consistent_with_training_eval": bool(alignment["alignment_ok"]),
            "bfcl_v2_vllm/prompt_consistency_note": alignment["alignment_note"],
            "bfcl_v2_vllm/alignment/is_xlam_run": bool(alignment["is_xlam_run"]),
            "bfcl_v2_vllm/alignment/training_uses_xlam_format": bool(
                alignment["training_uses_xlam_format"]
            ),
            "bfcl_v2_vllm/alignment/training_uses_strict_prompt": bool(
                alignment["training_uses_strict_prompt"]
            ),
            "bfcl_v2_vllm/alignment/bfcl_model_matches_training_model": bool(
                alignment["bfcl_model_matches_training_model"]
            ),
            "bfcl_v2_vllm/alignment/bfcl_uses_expected_xlam_handler": bool(
                alignment["bfcl_uses_expected_xlam_handler"]
            ),
            "bfcl_v2_vllm/alignment/expected_xlam_handler_name": str(
                alignment["expected_xlam_handler_name"]
            ),
            "bfcl_v2_vllm/alignment/when2call_eval_task_name": str(
                alignment["when2call_eval_task_name"]
            ),
            "bfcl_v2_vllm/alignment/when2call_eval_task_matches_xlam": str(
                alignment["when2call_eval_task_matches_xlam"]
            ),
        }
        payload.update(all_bfcl_metrics)
        wandb.log(payload)

        summary_path = run_out_dir / "bfcl_all_metrics.json"
        summary_obj = {
            "source_run_path": source_run_path,
            "base_model": base_model,
            "bfcl_model_name": bfcl_model_name,
            "test_categories_requested": requested_categories,
            "test_categories_resolved": categories,
            "score_json_path": str(score_path),
            "metric_source": "score/*.json (all BFCL library outputs)",
            "top_p": float(getattr(cfg.bfcl, "top_p", 1.0)),
            "max_output_tokens": int(getattr(cfg.bfcl, "max_output_tokens", 2048)),
            "all_metrics_flat": all_bfcl_metrics,
            "prompt_consistent_with_training_eval": alignment["alignment_ok"],
            "prompt_consistency_note": alignment["alignment_note"],
            "alignment": alignment,
            "generated_at_utc": _iso_now(),
        }
        summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

        art = wandb.Artifact(
            name=f"{run_id}-bfcl-v2-vllm",
            type="bfcl_v2_vllm",
            metadata={
                "run_id": run_id,
                "source_run_path": source_run_path,
                "base_model": base_model,
                "bfcl_model_name": bfcl_model_name,
                "test_categories": categories,
                "num_gpus": num_gpus,
                "top_p": float(getattr(cfg.bfcl, "top_p", 1.0)),
                "max_output_tokens": int(getattr(cfg.bfcl, "max_output_tokens", 2048)),
                "artifact_name": getattr(artifact, "name", ""),
                "score_json": str(score_path),
                "generated_at_utc": _iso_now(),
            },
        )
        art.add_dir(str(run_out_dir))
        wandb.log_artifact(art, aliases=["latest"])
        wandb.finish()

        print(f"[ok] Logged BFCL v2 metrics to run {entity}/{project}/{run_id}")
        print(f"[ok] Saved local outputs to {run_out_dir}")
        print(f"[ok] Logged {len(all_bfcl_metrics)} flattened BFCL metric keys from score/*.json")
        print(
            "[info] Prompt consistency with training/eval: "
            f"{alignment['alignment_ok']} ({alignment['alignment_note']})"
        )


if __name__ == "__main__":
    main()
