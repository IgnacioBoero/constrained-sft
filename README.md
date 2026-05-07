# Constrained supervised fine-tuning

Code for **“Every Sample Counts: Supervised Fine-Tuning of Language Models with Pointwise Constraints”** .

Training uses [Hydra](https://hydra.cc/) for configuration. Methods are implemented in `src/experiments/`.

Main training entry point is `src/train.py`.

```bash
python src/train.py --config-path train/paper_experiments/<area>/<subdir> --config-name=<stem>
# Reranker paper grids live one level shorter (YAMLs alongside paper_rerank_shared):
python src/train.py --config-path train/paper_experiments/reranker --config-name=sweep_pointwise
```
---

## Running experiments (`configs/train/paper_experiments/`)

The yaml config files for the experiments in the submission are located in the `configs/train/paper_experiments/` folder.
It contains:

| Folder | Experiment | Experiment class | Detailed docs |
|--------|-------------|-----------------|---------------|
| **`paper_experiments/safety/`** | Appendix D.3 — helpfulness KL + refusal constraints | **`safety`** | [`configs/train/paper_experiments/safety/README.md`](configs/train/paper_experiments/safety/README.md) |
| **`paper_experiments/function_calling/`** | Appendix D — When2Call / preference margins | **`dpo_kl`** | [`configs/train/paper_experiments/function_calling/README.md`](configs/train/paper_experiments/function_calling/README.md) |
| **`paper_experiments/reranker/`** | Appendix D.2 — cross-encoder relevance + length \(\Lambda\) loss | **`reranker`** | [`configs/train/paper_experiments/reranker/README.md`](configs/train/paper_experiments/reranker/README.md) |

Evaluation metrics are computed during training and logged using W&B (e.g. refusal samples, slacks), and others in **offline** scripts that load checkpoints (by default from W&B runs) and evaluate them.

We also include some notebooks for plotting the results in the `notebooks/` folder.
