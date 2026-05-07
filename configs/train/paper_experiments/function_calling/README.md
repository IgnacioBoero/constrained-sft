# Paper configs: When2Call function calling preferences

Experiment configs for **Appendix D**, **“Tool Calling / Function Calling Preferences”** (When2Call-style setup), matching **shared hyperparameters Table 5** and **per-method grids Table 6**.

The main narrative uses **Llama-3.2-1B-Instruct**-class checkpoints; **`function_calling/llama/`** targets **`meta-llama/Llama-3.2-1B-Instruct`** (public ID used in-repo). **`function_calling/xlam/`** uses **`Salesforce/xLAM-2-1b-fc-r`** with **`train.use_xlam_prompt_format`** (Appendix notes reusing Llama-found hyperparameters for xLAM).

## Directory layout

All YAMLs live under **`configs/train/paper_experiments/function_calling/`**.

| Directory | Role |
|-----------|------|
| **`sweep/`** | **Llama-only** Cartesian sweeps (**Table 6**) — each file runs `hydra.mode: MULTIRUN` with matching `params`. |
| **`llama/`** | **Final Llama:** validation-best hyperparameters, multirun **only seeds** \(\{42,43,44\}\). Includes **`resilient_both.yaml`** (no Table 6 grid in appendix). |
| **`xlam/`** | **Final xLAM:** hyperparameters mirrored from **`llama/`**, plus xLAM prompting flags — same seed sweep. |

**Sweep files:** `simpo.yaml`, `dpo.yaml`, `cal_dpo.yaml`, `cpo.yaml`, `average_both.yaml`, `pointwise.yaml`, `penalty.yaml`.

## Launch

Canonical (matches safety paper configs):

```bash
export PYTHONPATH=.
python src/train.py --config-path train/paper_experiments/function_calling/sweep --config-name=dpo
python src/train.py --config-path train/paper_experiments/function_calling/llama --config-name=resilient_both
python src/train.py --config-path train/paper_experiments/function_calling/xlam --config-name=dpo
```

Add **`-m`** for Hydra multirun sweeps:

```bash
python src/train.py -m --config-path train/paper_experiments/function_calling/sweep --config-name=dpo
```

**Multi-GPU (example)** — from repo root, mirror the **`--config-path` / `--config-name`** splits above inside **`torch.distributed.run`** (two processes in this snippet; adjust GPUs / rendezvous flags as needed):

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
python -m torch.distributed.run \
  --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  src/train.py \
  --config-path train/paper_experiments/function_calling/llama \
  --config-name dpo
```

Historical entry point still supported:

```bash
python src/train.py train=paper_experiments/function_calling/sweep/dpo
```

## Table 6 \(\rightarrow\) `exp.loss_type`

| Paper (Table 6) | `exp.loss_type` | Sweep YAML |
|-----------------|-----------------|------------|
| SimPO | `simpo` | `sweep/simpo.yaml` (\(\gamma \rightarrow\) `exp.tol`, \(\beta\) `exp.loss_alpha`) |
| DPO | `dpo` | `sweep/dpo.yaml` |
| Cal-DPO | `cal_dpo` | `sweep/cal_dpo.yaml` |
| CPO | `cpo` | `sweep/cpo.yaml` — preference β → **`exp.loss_beta`**; NLL-style weight → **`exp.loss_alpha`** (YAML + `dpo_kl.py`) |
| Average | `average_both` | `sweep/average_both.yaml` (\(\varepsilon_{\mathrm{win}}\) / \(\varepsilon_{\mathrm{lose}}\) \(\rightarrow\) `tol_win`, `tol_loose`) |
| Pointwise | `aug_dual_both` | `sweep/pointwise.yaml` |
| Penalty | `penalty_both` | `sweep/penalty.yaml` |

**Resilient** (`resilient_both`): **`llama/`**, **`xlam/`** finals only.

## Shared training knobs (Table 5 gist)

Effective batch **32**, **AdamW \(5{\times}10^{-5}\)** (see each YAML), **cosine**, **warmup 0**, **weight decay 0**, **BF16**, **max prompt 1024** / **max length 2048**, LoRA **r=8, α=16, dropout=0.05**, **5% validation** subsample default, **`when2call`** dataset flag.

Defaults for W&B: **`paper_when2call`** (override with `train.wandb_project=`).

## Post-train LM eval / metrics

`src/train.py` can spawn **When2Call lm-eval–style** subprocesses via `train.post_train_when2call_*` hooks (see **`configs/eval/wandb_when2call_lm_eval_vllm.yaml`** and **`src/eval/wandb_when2call_lm_eval_vllm.py`**). Loss definitions: **`src/experiments/README_dpo_kl_losses.md`**.  
Tooling catalogue: **[`docs/tooling_paper_aligned.md`](../../../../docs/tooling_paper_aligned.md)**.

## xLAM vs Llama

**`xlam/`** configs set **`train.use_xlam_prompt_format: true`** and relax strict When2Call prompting where indicated. **`llama/`** keeps **`when2call_strict_prompt: true`** defaults.
