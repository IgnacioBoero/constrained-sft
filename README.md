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
```

Add **`-m`** when the chosen YAML declares `hydra.mode: MULTIRUN`.

---

## Paper-aligned experiments (`configs/train/paper_experiments/`)

Unified layout:

| Folder | Experiment | Experiment class | Detailed docs |
|--------|-------------|-----------------|---------------|
| **`paper_experiments/safety/`** | Appendix D.3 — helpfulness KL + refusal constraints | **`safety`** | [`configs/train/paper_experiments/safety/README.md`](configs/train/paper_experiments/safety/README.md) |
| **`paper_experiments/function_calling/`** | Appendix D — When2Call / preference margins | **`dpo_kl`** | [`configs/train/paper_experiments/function_calling/README.md`](configs/train/paper_experiments/function_calling/README.md) |

Index + cross-links: **`configs/train/paper_experiments/README.md`**.

The default **`configs/train/default.yaml`** composes **`paper_experiments/safety/paper_safety_shared`** for a minimal safety-shape run—override **`--config-path` / `--config-name`** when working on When2Call or other tasks.

Legacy task folders under **`configs/train/`** (`bias`, `reranker`, `debug`, etc.) remain for exploratory configs.

---

## Losses / objectives

Structured notes for pairwise / margin objectives on When2Call: **`src/experiments/README_dpo_kl_losses.md`.

---

## Evaluation & tooling (high level)

| Purpose | Starting point |
|--------|----------------|
| **AlpacaEval** generations (benign prompts) | `src/eval/wandb_alpaca_eval_vllm.py`, `configs/eval/wandb_alpaca_eval_vllm.yaml`; batch: `scripts/run_safe_long1k_eval.sh` |
| **Refusal phrase counts** from W&B train-end dumps | `scripts/compute_refusal_metrics.py` (see **`safety` README**) |
| **PKU SafeRLHF + Beaver-7B unified cost / reward** | `scripts/eval_saferlhf_beaver.py` |
| **When2Call lm-eval** (optional post-train hook) | `configs/eval/wandb_when2call_lm_eval_vllm.yaml`, `src/eval/wandb_when2call_lm_eval_vllm.py` |

Scripts live in **`scripts/`**; HF datasets bundled or downloaded as configured in YAML.

---

## Branches

**`safe-func`** merges **safety-double-sided**, **eval**, and **when2call** tooling: paper configs split into **`safety/`** and **`function_calling/`** with per-task READMEs plus this overview.
