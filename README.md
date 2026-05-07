# Constrained supervised fine-tuning

Research code aligned with **“Every Sample Counts: Supervised Fine-Tuning of Language Models with Pointwise Constraints”** (PDF in-repo as `Paper.pdf` if you add it locally).

Training is **[Hydra](https://hydra.cc/)–driven** (`src/train.py`); implementations live under `src/experiments/`.

---

## Requirements

Python **3.11+**, CUDA optional but recommended for full runs. Typical install mixes **conda** (PyTorch, core deps per `environment.yml`) plus **pip** extras (`transformers`, `peft`, `hydra-core`, `wandb`, `vllm` for accelerated eval)—adjust to your stack.

Always run trainers with **`PYTHONPATH` including the repo root** so imports resolve:

```bash
export PYTHONPATH=.
```

Training entry point:

```bash
python src/train.py --config-path train/paper_experiments/<area>/<subdir> --config-name=<stem>
# Reranker paper grids live one level shorter (YAMLs alongside paper_rerank_shared):
python src/train.py --config-path train/paper_experiments/reranker --config-name=sweep_pointwise
```

Add **`-m`** when the chosen YAML declares `hydra.mode: MULTIRUN`.

---

## Paper-aligned experiments (`configs/train/paper_experiments/`)

Unified layout:

| Folder | Experiment | Experiment class | Detailed docs |
|--------|-------------|-----------------|---------------|
| **`paper_experiments/safety/`** | Appendix D.3 — helpfulness KL + refusal constraints | **`safety`** | [`configs/train/paper_experiments/safety/README.md`](configs/train/paper_experiments/safety/README.md) |
| **`paper_experiments/function_calling/`** | Appendix D — When2Call / preference margins | **`dpo_kl`** | [`configs/train/paper_experiments/function_calling/README.md`](configs/train/paper_experiments/function_calling/README.md) |
| **`paper_experiments/reranker/`** | Appendix D.2 — cross-encoder relevance + length \(\Lambda\) loss | **`reranker`** | [`configs/train/paper_experiments/reranker/README.md`](configs/train/paper_experiments/reranker/README.md) |

Index + cross-links: **`configs/train/paper_experiments/README.md`**.

The default **`configs/train/default.yaml`** composes **`paper_experiments/safety/paper_safety_shared`** for a minimal safety-shape run—override **`--config-path` / `--config-name`** for When2Call, reranking, or other setups.

**Tooling:** maps scripts → paper metrics → [`docs/tooling_paper_aligned.md`](docs/tooling_paper_aligned.md).  
Optional packaging / metadata cleanup → [`docs/tooling_review_for_deletion.md`](docs/tooling_review_for_deletion.md).

---

## Losses / objectives

Structured notes for pairwise / margin objectives on When2Call: **`src/experiments/README_dpo_kl_losses.md`**.

---

## Evaluation & tooling (high level)

| Purpose | Starting point |
|--------|----------------|
| **AlpacaEval** generations (benign prompts) | `src/eval/wandb_alpaca_eval_vllm.py`, `configs/eval/wandb_alpaca_eval_vllm.yaml`; batch wrappers: **`scripts/paper_eval/run_safe_long1k_eval.sh`**, **`scripts/paper_eval/run_safe_long1k_eval.py`**; tag loop: **`scripts/paper_eval/alpacaeval_by_tag.sh`** |
| **Refusal phrase counts** from W&B train-end dumps | **`scripts/paper_eval/compute_refusal_metrics.py`** (see **`safety` README**) |
| **PKU SafeRLHF + Beaver-7B unified cost / reward** | **`scripts/paper_eval/eval_saferlhf_beaver.py`** |
| **Safety: post-hoc KL to base / CVaR of slacks** (tail diagnostics) | **`scripts/paper_eval/compute_kl_safe_dpo.py`**, **`scripts/paper_eval/compute_slack_cvar.py`** |
| **When2Call lm-eval** (optional post-train hook) | `configs/eval/wandb_when2call_lm_eval_vllm.yaml`, `src/eval/wandb_when2call_lm_eval_vllm.py`; extra metrics helper: **`src/eval/when2call_additional_metrics.py`** |
| **MS MARCO length-aware reranker** | Train with `configs/train/paper_experiments/reranker/*.yaml`; batch checkpoint **metrics** → **`scripts/paper_eval/eval_reranker_checkpoints.py`**; see [`paper_experiments/reranker/README.md`](configs/train/paper_experiments/reranker/README.md) |

Layout: **`scripts/README.md`** indexes subfolders (`paper_eval/`, `when2call/`, `datasets/`, …).

---

## Branches

**`submission`** further merges **`main`** (reranker experiment codepaths) atop **`safe-func`** (**safety + eval + when2call**). **`paper_experiments/reranker/`** documents the appendix reranking grids on that line.
