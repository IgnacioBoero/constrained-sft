# Repo cleanup notes (optional)

The items that used to live here (aux training scripts, BFCL/IFEval runners, `hf_ready/`, `pda_*`, Deepspeed debug JSON, etc.) have been **removed** from the tree.

Remaining optional housekeeping:

## Packaging metadata drift

| Path | Notes |
|------|-------|
| **`pyproject.toml`** | `description` / `keywords` still mention “bias mitigation”; project scripts entry point may not match how you launch training (`python src/train.py`). Optional cleanup only. |

When you change dependencies or entry points, keep **`docs/tooling_paper_aligned.md`** in sync.
