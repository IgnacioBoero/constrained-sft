# Paper configs: reranking with length-aware objective (Appendix D.2)

Cross-encoder **ModernBERT-base** tuning on MS MARCO v2.1–derived pairs with **nine negatives**, optimizing a **LambdaLoss surrogate** \(\Lambda\) (\(K{=}k_\lambda{=}3\)) plus **ranking-margin constraints**
\(s_{\mathrm{pos}} - s_{\mathrm{neg},i} \ge \varepsilon\) implemented as `slack = loss_tol − (s_pos − s_neg)` (see **`src/experiments/reranker.py`**).

## Paper \(\rightarrow\) code map

| Symbol / term | YAML field |
|---------------|-------------|
| ε (margin tol. on score differences) | **`exp.loss_tol`** |
| α (Augmented Lagrangian curvature) | **`exp.loss_alpha`** |
| \(\eta_{\mathrm{dual}}\) (dual ascent step) | **`exp.dual_step_size`** |
| β (Resilient / quadratic smoothing) | **`exp.resilient_coef`** |
| λ₀ (penalty weight on slack) | **`exp.loss_alpha`** with **`loss_type: penalty`** |
| Shared backbone / batch / schedule | **`paper_rerank_shared.yaml`** (**Table 7**) |

## Data

**`datasets.load_dataset("<hf-user>/reranker", cfg.train.size, …)`**. Use **`train.size: large`** (default) as the pooled **55 k-example** stratified subset described in appendix D.2.

## Config layout

| YAML | Appendix ref |
|------|----------------|
| **`paper_rerank_shared.yaml`** | Table 7 — 10 epochs; micro-batch 8; `gradient_accumulation_steps: 8` (×1 GPU ⇒ **64** seq. groups per optimizer step); lr \(2\times10^{-5}\); linear schedule; warmup **0.1**; weight decay **0**; BF16; dropout **0.05** |
| **`sweep_pointwise.yaml`** | Table 12 **point**, ε ∈ {1, 3}; α = 1, η = 0.1 |
| **`sweep_penalty.yaml`** | λ₀ ∈ {0.1, 0.2, 0.4, 1, 10} |
| **`sweep_average.yaml`** | Table 12 **avg**, ε ∈ {50, 60, 75, 90, 100} |
| **`sweep_resilient.yaml`** | Table 12 **relax**, β grid; **`loss_tol = 3`** |
| **`sweep_dual_eta.yaml`**, **`sweep_aug_dual_eta.yaml`** | Fig. 14 dual vs augmented η grid |
| **`sweep_aug_dual_alpha.yaml`** | Table 13 α ∈ {1, 10, 100, 1000} |
| **`seeds_*.yaml`**, **`baseline_mono_bert.yaml`** | Fig. 4 illustrative points (**erm** + **`length_constraint: false`** = relevance-only cross-encoder) |

## Checkpoint batch evaluation

**`scripts/paper_eval/eval_reranker_checkpoints.py`** (Hydra entry) discovers checkpoints under **`train.output_dir`** from a training YAML **`train_config`** (see **`configs/eval/reranker.yaml`**) and writes **`evaluation_results.csv`**. Prefer this over ad-hoc one-off notebooks when sweeping Table 12 / Appendix F.

## Training

From repo root (`export PYTHONPATH=.`):

```bash
export PYTHONPATH=.
python src/train.py --config-path train/paper_experiments/reranker --config-name=sweep_pointwise
python src/train.py -m --config-path train/paper_experiments/reranker --config-name=sweep_average
python src/train.py --config-path train/paper_experiments/reranker --config-name=seeds_point_eps3
```

Optional: `train.use_wandb=true` on the CLI.

**Metrics:** `MRR@10`, `Hit@3`, `LenRank@10`, `AvgLength@3` (trainer utilities under the reranker experiment code).

More scripts and paths: [`docs/tooling_paper_aligned.md`](../../../../docs/tooling_paper_aligned.md).
