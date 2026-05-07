# Paper configs: safety tuning (Alpaca longest 1k + BeaverTails mix)

Safety tuning setups from **Appendix D.3** (instruction following + safety refusal), matching **Table 8** (shared training / LoRA) and **Table 9** (per-method hyperparameter grids).

**Branch lineage:** Evaluation helpers (AlpacaEval vLLM, refusal metrics, etc.) come from merges with `eval`; this folder is authoritative for safety **paper sweeps**.

## Location

Configs live under **`configs/train/paper_experiments/safety/`**:

- **`paper_safety_shared.yaml`** — shared recipe (epochs, LR, cosine schedule, AdamW weight decay 0.01, global grad clip 1, LoRA \(r{=}64,\ \alpha{=}128\) on `\{q,k,v,o\}_proj`, max length 2048, BF16 effective batch \(1 \times 32\) grad accumulation \(=32\), dataset `ihounie/SAFE-ALPACA-BTails-llama2`, base `ihounie/huggy-2-1k-alpaca-f16`).
- **`sweep_*.yaml`** — Hydra `MULTIRUN` grids (Table 9).
- **`seeds_fig3_*.yaml`** — three RNG seeds \(\{42,43,44\}\) for **Figure 3 right-table** rows (excluding **Base**, which has no separate safety train stage).

Notation (config → paper): `\(\alpha\)` → `exp.loss_alpha`; unsafe refusal level \(\varepsilon_U\) → `exp.tol`; `\(\varepsilon_H\)` → `exp.tol2`; `\(\eta_{\mathrm{dual}}\)` → `exp.dual_step_size`; relaxation \(\beta\) → `exp.resilient_coef`. Safe-DPO margin \(\Delta\) → `exp.tol` with `loss_type: daa` (Table 9 only lists \(\Delta\); `loss_alpha = 0.1` elsewhere unless overridden).

## How to run training

Hydra resolves from `configs/`. Prefer explicit path + stem:

```bash
export PYTHONPATH=.
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_safe_dpo
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_avg
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_aug_dual
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_resilient
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_penalty
```

Figure 3 seeded rows:

```bash
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_safe_dpo_delta5
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_safe_dpo_delta10
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_pen_lambda5
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_avg_eps_m032
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_avg_eps_m024
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_point_eps008
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_relax_beta20
python src/train.py --config-path train/paper_experiments/safety --config-name=seeds_fig3_relax_beta10
```

Distributed / DeepSpeed: add your usual launcher flags or `Acceleration`/`deepspeed` via `cfg.train.hf_args` overrides.

Paper **Base**: alpaca-long-1k SFT backbone only—no extra safety YAML row.

## Evaluation: PKU SafeRLHF + Beaver-7B cost

[`scripts/paper_eval/eval_saferlhf_beaver.py`](../../../../scripts/paper_eval/eval_saferlhf_beaver.py) loads **`PKU-Alignment/PKU-SafeRLHF-30K`** test prompts (by default), **greedy-generates**, and scores **`beaver-7b-unified-reward`** / **`beaver-7b-unified-cost`**. Example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/paper_eval/eval_saferlhf_beaver.py \
  --hf_models ihounie/huggy-2-1k-alpaca-f16 \
  --out_root ./outputs/beaver_eval \
  --num_samples 500
```

(`python scripts/paper_eval/eval_saferlhf_beaver.py --help` for W&B-backed runs.)

## AlpacaEval sampling (benign instructions)

Prompts shipped as **`src/datasets/safety/evaluation/alpaca_eval.json`**.

- **`src/eval/wandb_alpaca_eval_vllm.py`** + **`configs/eval/wandb_alpaca_eval_vllm.yaml`**
- Batch: **`scripts/paper_eval/run_safe_long1k_eval.py`** / **`scripts/paper_eval/run_safe_long1k_eval.sh`**
- No vLLM: **`src/eval/wandb_alpaca_eval.py`** + **`configs/eval/wandb_alpaca_eval.yaml`**

```bash
export PYTHONPATH=.
python src/eval/wandb_alpaca_eval_vllm.py \
  wandb.entity=YOUR_ENTITY \
  wandb.project=YOUR_PROJECT \
  wandb.run_id=RUN_ID
```

Feed JSON into upstream [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) for **LC win rate**.

## Refusal metrics

From W&B **`safe_generate_train_end_outputs_epoch_<N>`** artifacts ( **`train.use_wandb`** + train-end sampling in **`src/experiments/safety.py`**):

```bash
python scripts/paper_eval/compute_refusal_metrics.py --entity YOUR_ENTITY --project YOUR_PROJECT --tag YOUR_TAG
python scripts/paper_eval/compute_refusal_metrics.py --dry-run --tag YOUR_TAG
```

Phrase list / ratios: **`scripts/paper_eval/compute_refusal_metrics.py`** (orthogonal to benign AlpacaEval prompts).

## Post-hoc W&B metrics (optional)

- **`scripts/paper_eval/compute_kl_safe_dpo.py`** — recomputes **mean KL** to the frozen base model on the **eval** split (matches the helpfulness surrogate in **`safety.py`**) and can log **`eval/kl_to_base_*`** back to tagged runs (e.g. Safe-DPO / Table 9 rows).
- **`scripts/paper_eval/compute_slack_cvar.py`** — upper-tail **CVaR** of constraint **slacks** from end-of-training W&B tables; supports tail analyses aligned with abstract claim (C4) about violation distributions.

See **[`docs/tooling_paper_aligned.md`](../../../../docs/tooling_paper_aligned.md)** for full paths and prerequisites.
