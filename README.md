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

When2Call / pairwise preference objectives (**Table 6**) are implemented in **`src/experiments/dpo_kl.py`** and configured via the YAMLs in **`paper_experiments/function_calling/`** (see that folder’s README).

---

## Evaluation Utilities

Evaluation is split by **role**: **training-time** logging (e.g. refusal samples, slacks) inside the experiment classes, and **offline** runners that score finished checkpoints or W&B runs.

**What’s covered**

- **Safety (Appendix D.3):** benign-instruction generations for external win-rate tooling, phrase-based refusal summaries from logged tables, PKU / Beaver-style cost–reward scoring, and optional post-hoc KL / slack–tail summaries for analysis.
- **When2Call / preferences:** lm-eval-style task runs (including optional post-train hooks) plus add-on metrics parsed from harness logs.
- **Reranking:** batch re-evaluation of saved cross-encoder checkpoints into a single results table.

**Where things live**

- **`src/eval/`** — Hydra-facing eval entry points (often vLLM-backed) and small metric helpers they call.
- **`scripts/paper_eval/`** — W&B batch workflows, safety aggregates, and reranker checkpoint sweeps.
- **`scripts/datasets/`** and **`scripts/release/`** — dataset construction and packaging LoRA weights for the Hub.
- **`configs/eval/`** — defaults shared by those eval Hydra apps.

For a **file-level** index tied to appendix tables, see [`docs/tooling_paper_aligned.md`](docs/tooling_paper_aligned.md) and [`scripts/README.md`](scripts/README.md).

---

## Branches

**`submission`** further merges **`main`** (reranker experiment codepaths) atop **`safe-func`** (**safety + eval + when2call**). **`paper_experiments/reranker/`** documents the appendix reranking grids on that line.
