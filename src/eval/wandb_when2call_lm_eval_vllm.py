#!/usr/bin/env python3
"""
Run official When2Call LM-Eval-Harness task with vLLM on a merged LoRA model,
then log metrics/artifacts back to the same W&B run.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from when2call_additional_metrics import compute_additional_metrics


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


def _download_text(url: str) -> str:
    with urlopen(url, timeout=30) as f:  # nosec B310
        return f.read().decode("utf-8")


def _download_text_with_fallback(urls: list[str]) -> str:
    last_err: Exception | None = None
    for url in urls:
        try:
            return _download_text(url)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to download from all candidate URLs: {urls}") from last_err


def _prepare_official_when2call_task_layout(root: Path) -> Path:
    """
    Build:
      <root>/lm_eval/tasks/when2call/{when2call-llama3_2.yaml,utils.py,when2call_test_mcq.jsonl}
    so official YAML `data_files: lm_eval/tasks/when2call/when2call_test_mcq.jsonl`
    resolves unchanged.
    """
    task_dir = root / "lm_eval" / "tasks" / "when2call"
    task_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "when2call-llama3_2.yaml": [
            "https://raw.githubusercontent.com/NVIDIA/When2Call/main/evaluation/mcq/lm_eval_harness/when2call/when2call-llama3_2.yaml",
        ],
        "utils.py": [
            "https://raw.githubusercontent.com/NVIDIA/When2Call/main/evaluation/mcq/lm_eval_harness/when2call/utils.py",
        ],
        # In NVIDIA/When2Call, the MCQ test data is stored in data/test.
        "when2call_test_mcq.jsonl": [
            "https://raw.githubusercontent.com/NVIDIA/When2Call/main/data/test/when2call_test_mcq.jsonl",
            # Keep old path as best-effort fallback in case upstream layout changes.
            "https://raw.githubusercontent.com/NVIDIA/When2Call/main/evaluation/mcq/lm_eval_harness/when2call/when2call_test_mcq.jsonl",
        ],
    }
    for filename, urls in files.items():
        (task_dir / filename).write_text(_download_text_with_fallback(urls), encoding="utf-8")

    return task_dir


def _build_model_args(cfg: DictConfig, model_ref: str, tp_size: int) -> str:
    parts = [
        f"pretrained={model_ref}",
        f"tensor_parallel_size={tp_size}",
        f"dtype={str(cfg.vllm.dtype)}",
        f"gpu_memory_utilization={float(cfg.vllm.gpu_memory_utilization)}",
        f"trust_remote_code={bool(cfg.vllm.trust_remote_code)}",
    ]
    if cfg.vllm.max_model_len is not None:
        parts.append(f"max_model_len={int(cfg.vllm.max_model_len)}")
    return ",".join(parts)


def _build_hf_model_args(cfg: DictConfig, model_ref: str) -> str:
    parts = [
        f"pretrained={model_ref}",
        "parallelize=True",
        f"dtype={str(cfg.vllm.dtype)}",
        f"trust_remote_code={bool(cfg.vllm.trust_remote_code)}",
    ]
    return ",".join(parts)


def _run_lm_eval_command(
    cfg: DictConfig,
    *,
    working_dir: Path,
    include_path: Path,
    output_path: Path,
    cache_path: Optional[Path],
    model_ref: str,
    model_args: str,
) -> Dict[str, str]:
    backend = str(getattr(cfg.lm_eval, "backend", "hf")).strip().lower()
    if backend not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported lm_eval backend: {backend}. Use 'hf' or 'vllm'.")

    configured_batch_size = str(getattr(cfg.lm_eval, "batch_size", "1"))

    def _base_args_for(
        model_backend: str, backend_model_args: str, batch_size_value: str
    ) -> list[str]:
        return [
            "--model",
            model_backend,
            "--model_args",
            backend_model_args,
            "--tasks",
            str(cfg.lm_eval.task_name),
            "--batch_size",
            str(batch_size_value),
            "--num_fewshot",
            "0",
            "--output_path",
            str(output_path),
            "--log_samples",
            "--write_out",
            "--include_path",
            str(include_path),
        ]

    selected_model_args = (
        _build_hf_model_args(cfg, model_ref=model_ref) if backend == "hf" else model_args
    )
    base_args = _base_args_for(backend, selected_model_args, configured_batch_size)
    if cache_path is not None:
        base_args.extend(["--use_cache", str(cache_path)])

    attempts = [
        ["lm_eval", *base_args],
        ["lm_eval", "run", *base_args],
    ]
    last_err = None
    last_stderr = ""
    last_cmd = ""
    for cmd in attempts:
        proc = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "command": " ".join(cmd),
                "backend": backend,
            }
        last_stderr = proc.stderr or ""
        last_cmd = " ".join(cmd)
        last_err = (
            f"cmd={' '.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-8000:]}\n"
            f"stderr:\n{proc.stderr[-8000:]}\n"
        )

    if last_cmd:
        print(f"[lm-eval] Last failed command: {last_cmd}")
    if last_stderr.strip():
        print("[lm-eval] stderr from failed command:")
        print(last_stderr)
    else:
        print("[lm-eval] stderr from failed command is empty.")

    # Retry with safe batch size on known HF auto-batching CUDA failures.
    is_hf_backend = backend == "hf"
    cuda_launch_failure = "cuda error: unspecified launch failure" in last_stderr.lower()
    auto_batching = configured_batch_size.lower() == "auto"
    if is_hf_backend and auto_batching and cuda_launch_failure:
        retry_args = _base_args_for(backend, selected_model_args, "1")
        if cache_path is not None:
            retry_args.extend(["--use_cache", str(cache_path)])
        retry_attempts = [
            ["lm_eval", *retry_args],
            ["lm_eval", "run", *retry_args],
        ]
        for cmd in retry_attempts:
            proc = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "command": " ".join(cmd),
                    "backend": "hf",
                }
            last_err = (
                f"cmd={' '.join(cmd)}\n"
                f"exit={proc.returncode}\n"
                f"stdout:\n{proc.stdout[-8000:]}\n"
                f"stderr:\n{proc.stderr[-8000:]}\n"
            )

    # Fallback path for known lm-eval <-> vLLM API mismatch.
    allow_hf_fallback = backend == "vllm" and bool(
        getattr(cfg.lm_eval, "fallback_to_hf_on_vllm_import_error", True)
    )
    vllm_import_mismatch = (
        "cannot import name 'get_open_port' from 'vllm.utils'" in last_stderr.lower()
        or "cannot import name \"get_open_port\" from \"vllm.utils\"" in last_stderr.lower()
    )
    if allow_hf_fallback and vllm_import_mismatch:
        hf_model_args = _build_hf_model_args(cfg, model_ref=model_ref)
        hf_base_args = _base_args_for("hf", hf_model_args, configured_batch_size)
        if cache_path is not None:
            hf_base_args.extend(["--use_cache", str(cache_path)])
        hf_attempts = [
            ["lm_eval", *hf_base_args],
            ["lm_eval", "run", *hf_base_args],
        ]
        for cmd in hf_attempts:
            proc = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "command": " ".join(cmd),
                    "backend": "hf_fallback",
                }
            last_err = (
                f"cmd={' '.join(cmd)}\n"
                f"exit={proc.returncode}\n"
                f"stdout:\n{proc.stdout[-8000:]}\n"
                f"stderr:\n{proc.stderr[-8000:]}\n"
            )

    raise RuntimeError(f"lm_eval command failed.\n{last_err}")


def _find_primary_results_json(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("results*.json"))
    if not candidates:
        candidates = sorted(output_dir.rglob("*.json"))
    if not candidates:
        raise RuntimeError(f"No JSON outputs found in {output_dir}")
    return candidates[-1]


def _extract_task_metrics(results_json: Dict[str, Any], task_name: str) -> Dict[str, float]:
    task_metrics = (
        results_json.get("results", {}).get(task_name, {})
        if isinstance(results_json.get("results"), dict)
        else {}
    )
    out: Dict[str, float] = {}
    for k, v in task_metrics.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def _auto_tensor_parallel_size(requested: int) -> int:
    if requested and requested > 0:
        return requested
    n = torch.cuda.device_count()
    if n <= 0:
        raise RuntimeError("vLLM requires at least one CUDA device.")
    return n


def _sanitize_metric_key_part(value: str) -> str:
    s = str(value).strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unknown"


def _append_additional_metrics_to_payload(
    payload: dict[str, Any], additional: dict[str, Any]
) -> None:
    root = "when2call_lm_eval_vllm"
    payload[f"{root}/hallucination_rate"] = float(additional["hallucination_rate"])

    cls = additional.get("classification_metrics", {})
    macro = cls.get("macro_avg", {})
    micro = cls.get("micro_avg", {})
    payload[f"{root}/macro_f1"] = float(macro.get("f1", 0.0))
    payload[f"{root}/micro_f1"] = float(micro.get("f1", 0.0))
    payload[f"{root}/macro_acc"] = float(macro.get("acc", 0.0))
    payload[f"{root}/micro_acc"] = float(micro.get("acc", 0.0))

    per_category = cls.get("per_category", {})
    if isinstance(per_category, dict):
        for label, values in per_category.items():
            safe_label = _sanitize_metric_key_part(str(label))
            if not isinstance(values, dict):
                continue
            for metric_name in ["precision", "recall", "f1", "acc"]:
                v = values.get(metric_name)
                if isinstance(v, (int, float)):
                    payload[f"{root}/per_category/{safe_label}/{metric_name}"] = float(v)

    confusion = additional.get("confusion_matrix", {})
    if isinstance(confusion, dict):
        for true_label, row in confusion.items():
            safe_true = _sanitize_metric_key_part(str(true_label).replace("true:", ""))
            if not isinstance(row, dict):
                continue
            for pred_label, value in row.items():
                safe_pred = _sanitize_metric_key_part(str(pred_label).replace("pred:", ""))
                if isinstance(value, (int, float)):
                    payload[f"{root}/confusion_matrix/{safe_true}/{safe_pred}"] = float(value)


@hydra.main(config_path="../../configs", config_name="eval/wandb_when2call_lm_eval_vllm", version_base=None)
def main(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Run this evaluator as a single process. "
            "Use vllm.tensor_parallel_size to use all GPUs."
        )

    import wandb

    entity = str(cfg.wandb.entity)
    project = str(cfg.wandb.project)
    run_id = _optional_str(cfg.wandb.run_id if "run_id" in cfg.wandb else None)
    if run_id is None:
        raise RuntimeError("wandb.run_id is required for post-train evaluation logging.")
    base_model_override = _optional_str(cfg.model.base_model if "model" in cfg else None)

    source_run_path = f"{entity}/{project}/{run_id}"
    api = wandb.Api()
    source_run = api.run(source_run_path)
    base_model = _pick_base_model(run_obj=source_run, override=base_model_override)
    artifact = _pick_lora_artifact_for_run(source_run)

    # Attach to the exact same run so metrics/artifacts are added to training run.
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "must"
    wb_run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        job_type="when2call_lm_eval_vllm",
    )
    if wb_run is None:
        raise RuntimeError("wandb.init returned None.")

    with tempfile.TemporaryDirectory(prefix="when2call_lm_eval_vllm_") as td:
        tmp_root = Path(td)
        adapter_dir = Path(artifact.download(root=str(tmp_root / "adapter")))
        merged_dir = tmp_root / "merged_model"
        _merge_lora_to_local_model(
            base_model=base_model,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            merge_dtype=str(cfg.vllm.merge_dtype),
        )

        task_dir = _prepare_official_when2call_task_layout(tmp_root)
        include_path = tmp_root / "lm_eval" / "tasks"

        local_out_dir = Path(to_absolute_path(str(cfg.lm_eval.local_output_dir)))
        local_out_dir.mkdir(parents=True, exist_ok=True)
        run_out_dir = local_out_dir / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = None
        if bool(cfg.lm_eval.use_cache):
            cache_path = run_out_dir / "cache"
            cache_path.mkdir(parents=True, exist_ok=True)

        tp_size = _auto_tensor_parallel_size(int(cfg.vllm.tensor_parallel_size))
        model_args = _build_model_args(cfg, model_ref=str(merged_dir), tp_size=tp_size)
        cmd_result = _run_lm_eval_command(
            cfg,
            working_dir=tmp_root,
            include_path=include_path,
            output_path=run_out_dir,
            cache_path=cache_path,
            model_ref=str(merged_dir),
            model_args=model_args,
        )

        (run_out_dir / "lm_eval_stdout.log").write_text(cmd_result["stdout"], encoding="utf-8")
        (run_out_dir / "lm_eval_stderr.log").write_text(cmd_result["stderr"], encoding="utf-8")
        (run_out_dir / "lm_eval_command.txt").write_text(cmd_result["command"] + "\n", encoding="utf-8")

        results_path = _find_primary_results_json(run_out_dir)
        results_obj = json.loads(results_path.read_text(encoding="utf-8"))
        metrics = _extract_task_metrics(results_obj, task_name=str(cfg.lm_eval.task_name))
        additional_metrics = compute_additional_metrics(run_out_dir)
        additional_metrics_json_path = run_out_dir / "when2call_additional_metrics.json"
        additional_metrics_json_path.write_text(
            json.dumps({k: v for k, v in additional_metrics.items() if k != "_debug"}, indent=2),
            encoding="utf-8",
        )

        payload = {
            "when2call_lm_eval_vllm/mode": "lora_merged",
            "when2call_lm_eval_vllm/base_model": base_model,
            "when2call_lm_eval_vllm/source_run_path": source_run_path,
            "when2call_lm_eval_vllm/task_name": str(cfg.lm_eval.task_name),
            "when2call_lm_eval_vllm/tensor_parallel_size": tp_size,
            "when2call_lm_eval_vllm/lm_eval_backend": str(cmd_result.get("backend", "vllm")),
        }
        for k, v in metrics.items():
            payload[f"when2call_lm_eval_vllm/{k}"] = v
        _append_additional_metrics_to_payload(payload, additional_metrics)
        wandb.log(payload)

        art = wandb.Artifact(
            name=f"{run_id}-when2call-lm-eval-vllm",
            type="when2call_lm_eval_vllm",
            metadata={
                "run_id": run_id,
                "source_run_path": source_run_path,
                "base_model": base_model,
                "task_name": str(cfg.lm_eval.task_name),
                "tensor_parallel_size": tp_size,
                "artifact_name": getattr(artifact, "name", ""),
                "generated_at_utc": _iso_now(),
                "official_yaml": str(task_dir / "when2call-llama3_2.yaml"),
            },
        )
        art.add_dir(str(run_out_dir))
        wandb.log_artifact(art, aliases=["latest"])

        wandb.finish()
        print(f"[ok] Logged When2Call lm_eval(vllm) metrics to run {entity}/{project}/{run_id}")
        print(f"[ok] Saved local outputs to {run_out_dir}")


if __name__ == "__main__":
    main()

