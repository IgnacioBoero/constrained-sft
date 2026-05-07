# Review list — not part of the three paper experiments (or obsolete)

Use this file to decide what to **delete**, **move to a personal branch**, or **archive off-repo**.
None of these are required to reproduce **Appendix D.2 / D.3 / When2Call (D)** training configs under `configs/train/paper_experiments/`.

## Already removed in this pass

| Item | Reason |
|------|--------|
| `wandb_download.py` (repo root) | One-off W&B artifact fetch; hard-coded entity/project/run ids; **invalid Python** in the list literal (missing commas). |
| `test_eval.py` | Ad-hoc smoke test importing the old root `eval.py`; paths pointed at removed **ModernBERT-large** layout. |

## Auxiliary training & exploration (not appendix recipes)

| Path | What it does | Suggested action |
|------|----------------|------------------|
| **`scripts/aux_training/train_sft_ultrachat_1b.py`** | 1‑epoch Ultrachat SFT on **Llama‑3.2‑1B base** toward a hobby Hub ID (`ihounie/1B-ultrachat`). **Not** the paper’s When2Call or safety backbone scripts. | Delete or relocate to **`experiments/aux/`** on a non-submission branch. |
| **`scripts/aux_training/length_training.py`** | Standalone ModernBERT‑**large** pairwise length classifier on MS MARCO v1.1-style negatives (**not** the `iboero16/reranker` + λLoss stack in Appendix D.2). | Same — keep only if you still run this historical probe. |

## Supplementary benchmarks (HF eval runners; not Tables 5–9 / 12)

| Path | Notes |
|------|--------|
| **`src/eval/wandb_bfcl_v2_vllm.py`** | BFCL v2 tool-use benchmark via vLLM; **orthogonal** to the paper’s When2Call preference setup. Config may live outside `configs/eval/` or be inlined in-repo history. |
| **`src/eval/wandb_ifeval_vllm.py`** + **`configs/eval/wandb_ifeval_vllm.yaml`** | IFEval-style instruction adherence; interesting but **not** one of the three headline tasks. |

## Miscellaneous repo cruft / personal assets

| Path | Notes |
|------|-------|
| **`evaluation_results.csv`** (repo root) | Output from batch rerank evaluation; regenerate with **`scripts/paper_eval/eval_reranker_checkpoints.py`** rather than versioning. Prefer **gitignore**. |
| **`hf_ready/`** | Example **`merge_and_save.py`** subtree for Llama‑2 alpaca merges; overlaps conceptually with `scripts/release/export_wandb_lora_to_hf.py`. |
| **`pda_full.yaml`**, **`pda_pip.txt`** | Look like machine / environment snippets; unlikely to belong in a public reproducibility tree. |
| **`configs/ds_zero3_debug.json`** | Deepspeed debug stub; trim if unused. |

## Packaging metadata drift

| Path | Notes |
|------|-------|
| **`pyproject.toml`** | `description` / `keywords` still mention “bias mitigation”; project scripts entry point may not match how you launch training (`python src/train.py`). Optional cleanup only. |

## If you delete anything

- Grep for old import paths / `scripts/<name>.py` after moving files.
- Keep **`docs/tooling_paper_aligned.md`** in sync with what remains.
