# Paper experiment configs

Safety tuning setups from **Appendix D.3** (instruction following + safety refusal), matching **Table 8** (shared training / LoRA) and **Table 9** (per-method hyperparameter grids).

## Location

YAML files live under:

`configs/train/paper_experiments/safety/`

- `paper_safety_shared.yaml` — shared recipe (epochs, LR, cosine schedule, AdamW weight decay 0.01, global grad clip 1, LoRA \(r{=}64,\ \alpha{=}128\) on `\{q,k,v,o\}_proj`, max length 2048, BF16 effective batch \(1 \times 32\) grad accumulation \(=32\), dataset `ihounie/SAFE-ALPACA-BTails-llama2`, base `ihounie/huggy-2-1k-alpaca-f16`).
- **`sweep_*.yaml`** — Hydra `MULTIRUN` grids (“sweeps for each method” as in Table 9).
- **`seeds_fig3_*.yaml`** — three RNG seeds \(\{42,43,44\}\) for the **Figure 3 right-table** labelled runs (excluding the non-trained Base row).

Notation (config → paper): `\(\alpha\)` → `exp.loss_alpha`; unsafe refusal level \(\varepsilon_U\) → `exp.tol`; `\(\varepsilon_H\)` → `exp.tol2` (don’t-over-refuse bound); `\(\eta_{\mathrm{dual}}\)` → `exp.dual_step_size`; relaxation \(\beta\) → `exp.resilient_coef`. Safe-DPO margin \(\Delta\) → `exp.tol` with `loss_type: daa` ( \(\beta\) fixed in repo at `loss_alpha = 0.1` unless you override—Table 9 lists only \(\Delta\) for Safe-DPO).

## How to run

From the repository root (`PYTHONPATH` must include the project so `callbacks`, `experiments`, etc. resolve):

```bash
export PYTHONPATH=.
```

Hydra resolves package configs from `configs/`. Use **`--config-path=train/paper_experiments/safety`** and **`--config-name=<file_stem>`** (omit `.yaml`).

### Sweeps (Table 9 grids)

```bash
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_safe_dpo
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_avg
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_aug_dual
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_resilient
python src/train.py --config-path train/paper_experiments/safety --config-name=sweep_penalty
```

### Three-seed repeats (Figure 3 plotted rows)

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

Distributed / DeepSpeed: add your usual launcher flags or `Acceleration`/`deepspeed` entries via `cfg.train.hf_args` overrides on the CLI.

Results in the paper for **Base** correspond to evaluating the alpaca-long-1k SFT backbone without safety fine-tuning; there is no training config row for Base.
