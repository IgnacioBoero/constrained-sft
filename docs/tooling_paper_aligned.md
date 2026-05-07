# Tooling aligned with the paper experiments

Everything here supports **instruction-following safety (Appendix D.3)**,
**When2Call / preference fine-tuning (Appendix D, Tables 5–6)**,
or **length-aware reranking (Appendix D.2)** as described in
*Every Sample Counts: Supervised Fine-Tuning of Language Models with Pointwise Constraints*.

## Core library (always)

| Artifact | Role |
|----------|------|
| `src/train.py` | Hydra entry for fine-tuning; loads `experiment`/`exp.name` maps in `experiments/base.py`. |
| `src/experiments/safety.py` | Appendix D.3: helpfulness KL + pairwise refusal thresholds (`tol`, `tol2`). |
| `src/experiments/dpo_kl.py` | When2Call / margin objectives referenced in Appendix D (`loss_type`, etc.). |
| `src/experiments/reranker.py` | Cross-encoder + length constraint / Λ surrogates (Appendix D.2). |
| `configs/train/paper_experiments/**` | Paper-matched grids and shared YAMLs (`paper_*_shared.yaml`). |

Supporting notes: **`src/experiments/README_dpo_kl_losses.md`**.

---

## Appendix D.3 — Safety

| Artifact | Metrics / experiment tie-in |
|----------|-----------------------------|
| `src/experiments/safety.py` | Training-time helpfulness surrogate (KL-shaped) + refusal logits; logged slacks / samples when W&B tables are enabled. |
| **`scripts/paper_eval/eval_saferlhf_beaver.py`** | **PKU-SafeRLHF** test prompts + **Beaver-7B unified reward/cost** (Table-style safety vs harm tradeoffs). |
| **`src/eval/wandb_alpaca_eval_vllm.py`** + `configs/eval/wandb_alpaca_eval_vllm.yaml` | Generations on **benign** [`alpaca_eval.json`](../src/datasets/safety/evaluation/alpaca_eval.json) prompts for upstream **AlpacaEval** LC win rate. |
| **`src/eval/wandb_alpaca_eval.py`** + `configs/eval/wandb_alpaca_eval.yaml` | Same flow without vLLM (slower HF generate). |
| **`scripts/paper_eval/run_safe_long1k_eval.py`** / **`run_safe_long1k_eval.sh`** | Batch-queue AlpacaEval vLLM jobs for SAFE-long–style projects on W&B. |
| **`scripts/paper_eval/alpacaeval_by_tag.sh`** | Run AlpacaEval vLLM for all runs carrying a chosen W&B tag. |
| **`scripts/paper_eval/compute_refusal_metrics.py`** | Parses `safe_generate_train_end_outputs_epoch_*` tables → refusal phrase ratios (Fig. 3 / refusal storyline). |
| **`scripts/paper_eval/compute_kl_safe_dpo.py`** | Post-hoc **eval-split KL** to base (matches KL construction in `safety.py`; useful when logging was incomplete). |
| **`scripts/paper_eval/compute_slack_cvar.py`** | Tail **CVaR** over logged constraint **slacks** (Abstract C4 / violation distribution emphasis). |

---

## Appendix D — When2Call / function calling

| Artifact | Metrics / experiment tie-in |
|----------|-----------------------------|
| `src/experiments/dpo_kl.py` + YAMLs under `paper_experiments/function_calling/` | Table 6 loss types (`dpo`, `simpo`, `penalty_both`, …); Table 5 shared knobs in each YAML cluster. |
| **`scripts/when2call/run_pref.sh`** | Launcher for **`torch.distributed.run`** + **`src/train.py --config-path train/paper_experiments/function_calling/<subdir> --config-name <stem>`**. |
| `src/train.py` · `train.post_train_when2call_lm_eval*` | Optional post-epoch hook invoking lm-eval harness. |
| **`src/eval/wandb_when2call_lm_eval_vllm.py`** + `configs/eval/wandb_when2call_lm_eval_vllm.yaml` | Stand-alone **lm-eval** over a finished W&B run (merge LoRA → score). |
| **`src/eval/when2call_additional_metrics.py`** | Extra When2Call metrics from **lm-eval `samples_*.jsonl`** (NVIDIA-style add-ons; same data as post-train hook). |

---

## Appendix D.2 — Reranking

| Artifact | Metrics / experiment tie-in |
|----------|-----------------------------|
| `src/experiments/reranker.py` | Task loss + length constraint + logged ranking / length metrics (NDCG@10, MRR, constraint diagnostics in `compute_metrics`). |
| `configs/train/paper_experiments/reranker/*.yaml` | Table 7 shared block + Table 12 / Fig. 4 / Appendix F sweeps. |
| **`scripts/paper_eval/eval_reranker_checkpoints.py`** | Hydra batch driver: walks **`train.output_dir`** (incl. merged `defaults:`) for multirun sweeps, evaluates every `checkpoint-*`, writes **`evaluation_results.csv`**. Config: `configs/eval/reranker.yaml` (default chain `eval/default.yaml`). |

---

## Dataset / release helpers (paper pipeline)

| Artifact | Role |
|----------|------|
| **`scripts/datasets/make_safe_alpaca_4.py`** | Builds **SAFE-ALPACA-4** from model generations + `iboero16/SAFE-ALPACA-3` validation alignments (dataset evolution *around* the safety mix; not every reader needs to rerun). |
| **`scripts/release/export_wandb_lora_to_hf.py`** | Pack W&B **lora_adapters** artifacts into an HF-uploadable folder (+ model card stub). |

---

## `configs/eval/` quick map

| File | Used with |
|------|-----------|
| `eval/default.yaml` | Defaults to reranker eval stack (see `reranker.yaml`). |
| `eval/reranker.yaml` | `train_config` → `paper_experiments/reranker/paper_rerank_shared.yaml` + eval batch settings. |
| `wandb_alpaca_eval*.yaml` | `src/eval/wandb_alpaca_eval*.py` |
| `wandb_when2call_lm_eval_vllm.yaml` | `src/eval/wandb_when2call_lm_eval_vllm.py` |

---

## Layout

See **`scripts/README.md`** for directory conventions after the repo reorg (`paper_eval/`, `when2call/`, `datasets/`, `aux_training/`, `release/`).
