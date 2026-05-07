#!/usr/bin/env python3
"""
Download a W&B LoRA adapters artifact (type: lora_adapters) and assemble a
Hugging Face-ready folder containing:
  - adapter weights + adapter_config.json (+ tokenizer files if present)
  - README.md (model card)
  - merge_and_save.py (helper to merge adapters into the base model)
  - .gitattributes (LFS patterns)

IMPORTANT:
This repo contains a local ./wandb directory (run logs) that can shadow the
real `wandb` Python package when running from the repo root. Prefer running
this script from outside the repo root, e.g.:
  cd /tmp
  python /home/chiche/constrained-sft-2/scripts/export_wandb_lora_to_hf.py ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_copytree(src: Path, dst: Path) -> None:
    _ensure_dir(dst)
    for root, _dirs, files in os.walk(src):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        out_root = dst / rel
        _ensure_dir(out_root)
        for f in files:
            in_f = root_p / f
            out_f = out_root / f
            if out_f.exists():
                out_f.unlink()
            shutil.copy2(in_f, out_f)


def _pick_latest_lora_artifact(
    entity: str,
    project: str,
    type_name: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (fully qualified artifact ref, metadata) for the newest artifact version.

    W&B's public API varies across SDK versions; the most robust approach here is:
      - list artifact *collections* for (entity/project, type_name)
      - list artifact *versions* for each collection
      - choose the newest version (prefer alias 'latest' when present)
    """
    import wandb  # noqa: WPS433

    api = wandb.Api()
    project_path = f"{entity}/{project}"
    collections = list(api.artifact_collections(project_path, type_name, per_page=100))
    if not collections:
        raise RuntimeError(f"No artifact collections found for {project_path} with type={type_name!r}")

    def _aliases(a: Any) -> set[str]:
        try:
            als = a.aliases
        except Exception:
            return set()
        if als is None:
            return set()
        # Sometimes strings, sometimes objects with `.name`
        out: set[str] = set()
        for x in als:
            if isinstance(x, str):
                out.add(x)
            else:
                out.add(getattr(x, "name", str(x)))
        return out

    best_art = None
    best_key = None

    for col in collections:
        col_name = getattr(col, "name", None)
        if not col_name:
            continue
        # List versions for this collection
        versions = list(api.artifacts(type_name, f"{project_path}/{col_name}", per_page=100))
        for art in versions:
            created = getattr(art, "created_at", None) or getattr(art, "updated_at", None)
            als = _aliases(art)
            has_latest = "latest" in als
            # Key: prefer 'latest' alias, then newest timestamp
            key = (1 if has_latest else 0, created)
            if best_key is None or key > best_key:
                best_key = key
                best_art = art

    if best_art is None:
        raise RuntimeError(f"Could not select an artifact version for {project_path} type={type_name!r}")

    # best_art.name is like: "<collection>:v0"
    name_with_version = getattr(best_art, "name", None)
    if not name_with_version:
        raise RuntimeError("Selected artifact is missing .name")

    # Use explicit :latest ref if alias exists for that version; otherwise use versioned name.
    if "latest" in _aliases(best_art):
        collection_name = name_with_version.split(":")[0]
        ref = f"{project_path}/{collection_name}:latest"
    else:
        ref = f"{project_path}/{name_with_version}"

    meta = dict(getattr(best_art, "metadata", {}) or {})
    return ref, meta


def _download_artifact(ref: str, download_root: Path) -> Path:
    import wandb  # noqa: WPS433

    api = wandb.Api()
    art = api.artifact(ref)
    out_dir = Path(art.download(root=str(download_root)))
    return out_dir


def _write_gitattributes(out_dir: Path) -> None:
    (out_dir / ".gitattributes").write_text(
        "\n".join(
            [
                "*.safetensors filter=lfs diff=lfs merge=lfs -text",
                "*.bin filter=lfs diff=lfs merge=lfs -text",
                "*.pt filter=lfs diff=lfs merge=lfs -text",
                "*.model filter=lfs diff=lfs merge=lfs -text",
                "*.tiktoken filter=lfs diff=lfs merge=lfs -text",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _detect_base_model(meta: Dict[str, Any], fallback: Optional[str]) -> str:
    cand = meta.get("model_name") or meta.get("base_model") or meta.get("base_model_name_or_path")
    if isinstance(cand, str) and cand.strip():
        return cand.strip()
    if fallback:
        return fallback
    return "CHANGE_ME_BASE_MODEL"


def _write_readme(
    out_dir: Path,
    hf_repo_id: str,
    base_model: str,
    wandb_ref: str,
    meta: Dict[str, Any],
) -> None:
    global_step = meta.get("global_step", None)
    exp_name = meta.get("exp_name", None)
    trained_with = meta.get("output_dir", None)

    yaml_front_matter = "\n".join(
        [
            "---",
            f"base_model: {base_model}",
            "library_name: peft",
            "tags:",
            "  - lora",
            "  - peft",
            "  - alpaca",
            "  - safety",
            "license: other",
            "---",
        ]
    )

    body = f"""{yaml_front_matter}

## {hf_repo_id}

This repository contains **LoRA adapters** trained on an Alpaca-style dataset (1k longest unsafe responses filtering used in training config).

- **Base model**: `{base_model}`
- **Artifact source (Weights & Biases)**: `{wandb_ref}`
- **Exported at**: `{_iso_utc_now()}`
"""
    if exp_name:
        body += f"- **Experiment**: `{exp_name}`\n"
    if global_step is not None:
        body += f"- **Global step**: `{global_step}`\n"
    if trained_with:
        body += f"- **Training output_dir**: `{trained_with}`\n"

    body += """

## Usage (load adapters)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("{base_model}", use_fast=True)

model = PeftModel.from_pretrained(base_model, "{hf_repo_id}")
```

## Merge adapters into a standalone model

This repo ships a helper script:

```bash
python merge_and_save.py --base_model "{base_model}" --adapter_dir . --out_dir ./merged
```

The merged folder can then be uploaded as a fully standalone model (no PEFT dependency).

## Files

- `adapter_config.json`, `adapter_model.*`: LoRA adapter weights/config
- Tokenizer files (if present): `tokenizer.*`, `special_tokens_map.json`, etc.
"""
    (out_dir / "README.md").write_text(
        body.format(base_model=base_model, hf_repo_id=hf_repo_id),
        encoding="utf-8",
    )


def _write_merge_script(out_dir: Path) -> None:
    (out_dir / "merge_and_save.py").write_text(
        """#!/usr/bin/env python3
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
""",
        encoding="utf-8",
    )
    os.chmod(out_dir / "merge_and_save.py", 0o755)


def _assert_adapter_files_present(out_dir: Path) -> None:
    cfg = out_dir / "adapter_config.json"
    if not cfg.exists():
        raise RuntimeError(f"Missing {cfg.name} in {out_dir}")

    weights_ok = any((out_dir / n).exists() for n in ["adapter_model.safetensors", "adapter_model.bin"])
    if not weights_ok:
        raise RuntimeError(f"Missing adapter weights in {out_dir} (expected adapter_model.safetensors or adapter_model.bin)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="alelab")
    ap.add_argument("--project", default="SAFE-llama2-long1k")
    ap.add_argument("--type_name", default="lora_adapters")
    ap.add_argument(
        "--artifact_ref",
        default=None,
        help="Optional explicit ref: entity/project/name:alias_or_version (overrides auto-pick)",
    )
    ap.add_argument("--hf_repo_id", default="ihounie/llama2-7b-alpaca-1k")
    ap.add_argument(
        "--base_model",
        default=None,
        help="Optional override for base model HF id; if unset uses artifact metadata model_name",
    )
    ap.add_argument(
        "--out_root",
        default=str(Path.cwd() / "hf_ready"),
        help="Root output directory where the HF repo folder will be created",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    repo_dir_name = args.hf_repo_id.replace("/", "__")
    out_dir = out_root / repo_dir_name

    _ensure_dir(out_root)

    if args.artifact_ref:
        wandb_ref = args.artifact_ref
        meta: Dict[str, Any] = {}
    else:
        wandb_ref, meta = _pick_latest_lora_artifact(args.entity, args.project, args.type_name)

    # Download into a temporary dir inside out_root to keep everything together.
    dl_root = out_root / "_wandb_download"
    _ensure_dir(dl_root)
    dl_dir = _download_artifact(wandb_ref, dl_root)

    # Assemble HF folder
    if out_dir.exists():
        shutil.rmtree(out_dir)
    _safe_copytree(dl_dir, out_dir)

    base_model = _detect_base_model(meta, args.base_model)
    _write_gitattributes(out_dir)
    _write_merge_script(out_dir)
    _write_readme(out_dir, args.hf_repo_id, base_model, wandb_ref, meta)

    # Keep a tiny provenance file
    (out_dir / "export_provenance.json").write_text(
        json.dumps(
            {
                "wandb_ref": wandb_ref,
                "wandb_metadata": meta,
                "hf_repo_id": args.hf_repo_id,
                "base_model": base_model,
                "exported_at": _iso_utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _assert_adapter_files_present(out_dir)

    print(f"[ok] Downloaded: {wandb_ref}")
    print(f"[ok] HF folder:  {out_dir}")
    print("[next] Push to HF (example):")
    print(f"  huggingface-cli repo create {args.hf_repo_id} --type model")
    print(f"  cd {out_dir} && git init && git lfs install")
    print(f"  git remote add origin https://huggingface.co/{args.hf_repo_id}")
    print("  git add -A && git commit -m 'Add LoRA adapter'")
    print("  git push -u origin main")


if __name__ == "__main__":
    main()

