# Paper experiment configs

Safety tuning setups from **Appendix D.3** (instruction following + safety refusal), matching **Table 8** (shared training / LoRA) and **Table 9** (per-method hyperparameter grids).

**Branch:** `origin/eval` tooling (AlpacaEval sampling, batch helpers, refusal metrics) is merged into `safety-double-sided`; downstream commands assume that tree.

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

## Evaluation: PKU SafeRLHF prompts + Beaver-7B unified cost

After fine-tuning, you can reproduce the **harmlessness / Beaver cost-model**–style scoring used alongside AlpacaEval in the paper via:

[`scripts/eval_saferlhf_beaver.py`](../../../scripts/eval_saferlhf_beaver.py)

That script pulls deduplicated prompts from **`PKU-Alignment/PKU-SafeRLHF-30K`** (test split by default), **runs greedy generation** from your checkpoints (either Hugging Face model IDs via `--hf_models`, or LoRA checkpoints downloaded from Weight & Biases runs via `--entity` / `--project` / `--tag` or `--run_ids`), then **scores completions** with **`PKU-Alignment/beaver-7b-unified-reward`** and **`PKU-Alignment/beaver-7b-unified-cost`** (`--reward_model` / `--cost_model` override the defaults). Outputs go under `--out_root` (CSV plus aggregate metrics).

Run from outside a directory that shadows the `wandb` package when using W&B mode (see the docstring at the top of that file). Example (HF checkpoints only):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_saferlhf_beaver.py \
  --hf_models ihounie/huggy-2-1k-alpaca-f16 \
  --out_root ./outputs/beaver_eval \
  --num_samples 500
```

For full options (batch sizes, `--max_new_tokens`, logging back to runs, etc.), use `python scripts/eval_saferlhf_beaver.py --help`.

## AlpacaEval sampling (benign Alpaca-style instructions)

Figure 3 (left) **length-controlled win rate** is computed with the official [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) harness on model outputs for **benign** evaluation prompts. In this repo, prompts are shipped as:

`src/datasets/safety/evaluation/alpaca_eval.json`

**vLLM path (recommended for SAFE-long1k / LoRA runs logged to W&B):**

- Entry point: [`src/eval/wandb_alpaca_eval_vllm.py`](../../../src/eval/wandb_alpaca_eval_vllm.py) (Hydra config [`configs/eval/wandb_alpaca_eval_vllm.yaml`](../../../configs/eval/wandb_alpaca_eval_vllm.yaml)).
- Downloads the run’s `lora_adapters` artifact, merges adapters into the logged base model, generates with vLLM, writes JSON under `eval.local_output_dir`, and can log an `alpaca_eval_outputs_vllm` artifact back to the same run.

```bash
export PYTHONPATH=.
python src/eval/wandb_alpaca_eval_vllm.py \
  wandb.entity=YOUR_ENTITY \
  wandb.project=YOUR_PROJECT \
  wandb.run_id=RUN_ID
```

**Batch over many finished runs** (default project `alelab` / `SAFE-long1k`): [`scripts/run_safe_long1k_eval.py`](../../../scripts/run_safe_long1k_eval.py) lists qualifying runs; [`scripts/run_safe_long1k_eval.sh`](../../../scripts/run_safe_long1k_eval.sh) launches each eval as a **direct child process** (avoids vLLM multiprocess nesting issues).

**Transformers path (no vLLM):** [`src/eval/wandb_alpaca_eval.py`](../../../src/eval/wandb_alpaca_eval.py) with [`configs/eval/wandb_alpaca_eval.yaml`](../../../configs/eval/wandb_alpaca_eval.yaml).

Feed the produced JSON into upstream AlpacaEval to obtain **LC win rate** vs. a reference model.

## Refusal metrics (held-out harmful / safety-eval prompts)

Figure 3 (left) **refusal rate on harmful prompts** follows the paper’s rule: treat a generation as a refusal if it **starts with** certain phrases (substring list in code; close to Appendix D.3). That is **orthogonal** to AlpacaEval sampling above: Alpaca inputs are **safe** instructions; refusal rates come from generations on **unsafe** rows in the safety evaluation mix.

After training with **`train.use_wandb: true`** and train-end generation enabled (see `eval_at_end` / `SAFETYTrainer` in [`src/experiments/safety.py`](../../../src/experiments/safety.py)), runs log W&B artifacts named like:

`safe_generate_train_end_outputs_epoch_<N>`

(JSON bundles `outputs` = sampled text and `safe` = prompt labels; the refusal script’s docstring describes how that flag is interpreted.)

To aggregate refusal counts / ratios **from those artifacts** and optionally log metrics back into each run:

```bash
python scripts/compute_refusal_metrics.py --entity YOUR_ENTITY --project YOUR_PROJECT --tag YOUR_TAG
# Preview only:
python scripts/compute_refusal_metrics.py --dry-run --tag YOUR_TAG
```

See [`scripts/compute_refusal_metrics.py`](../../../scripts/compute_refusal_metrics.py) for flags (`--epoch`, phrase list, default denominator used in ratios).

**Related:** [`scripts/eval_saferlhf_beaver.py`](../../../scripts/eval_saferlhf_beaver.py) (section above) samples **PKU-SafeRLHF** test prompts and scores with **Beaver-7B cost** — another safety-related evaluation path, distinct from AlpacaEval and from `compute_refusal_metrics`.
