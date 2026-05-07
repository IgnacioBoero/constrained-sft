# Paper experiments

This directory will hold experiment configs tied to the paper. Each **task** (e.g. function calling) lives in its own subfolder.

## Function calling (When2Call, Appendix D)

Configs for **Appendix D, “Function Calling Preferences”**: **When2Call** [BFCL / APIGen-style setup as described there], **LoRA** fine-tuning, **shared hyperparameters Table 5**, **per-method search grids Table 6**.

The main text cites both **Llama-3.3-1B-Instruct** and **Figure 2** uses **Llama-3.2-1B-Instruct**; these configs use **`meta-llama/Llama-3.2-1B-Instruct`** for sweeps and Llama finals to match the public checkpoint used in this repo. **xLAM** finals use **`Salesforce/xLAM-2-1b-fc-r`** with **xLAM prompt formatting** (see `train.use_xlam_prompt_format`).

### Directory layout

All paths are under **`configs/train/paper_experiments/function_calling/`**.

| Directory | Role |
|-----------|------|
| **`sweep/`** | **Llama 3.2 1B only.** One YAML per baseline/constrained **method**; each file sets `hydra.mode: MULTIRUN` and a sweeper `params` block matching **Table 6** (Cartesian product of the listed grids). |
| **`llama/`** | **Final Llama runs:** validation-selected hyperparameters (same as reported main results), **only** `train.seed ∈ {42, 43, 44}` multirun (3 seeds). |
| **`xlam/`** | **Final xLAM runs:** **identical** hyperparameters to **`llama/`** (Appendix D: no separate xLAM search), same **3 seeds**. |

#### Files

- **`sweep/`:** `simpo.yaml`, `dpo.yaml`, `cal_dpo.yaml`, `cpo.yaml`, `average_both.yaml`, `pointwise.yaml`, `penalty.yaml`  
- **`llama/`** and **`xlam/`:** same method set plus **`resilient_both.yaml`** (main table; no Table 6 grid in the paper — only seed sweep here).

### Launch

From the repository root:

```bash
python src/train.py train=paper_experiments/function_calling/sweep/dpo
python src/train.py train=paper_experiments/function_calling/llama/resilient_both
python src/train.py train=paper_experiments/function_calling/xlam/dpo
```

For multirun configs, run with Hydra’s multirun flag as you normally do (e.g. `-m`).

### Table 6 → config mapping

| Paper (Table 6) | `exp.loss_type` | Sweep YAML | Notes |
|-----------------|-----------------|------------|--------|
| SimPO | `simpo` | `function_calling/sweep/simpo.yaml` | γ → `exp.tol`; β → `exp.loss_alpha` (implementation: `gamma = cfg.tol`, `beta_simpo = cfg.loss_alpha`). |
| DPO | `dpo` | `function_calling/sweep/dpo.yaml` | β → `exp.loss_alpha`. |
| Cal-DPO | `cal_dpo` | `function_calling/sweep/cal_dpo.yaml` | β → `exp.loss_alpha`; **10 epochs** fixed. The paper fixes additional calibration notation (λ); this code path uses the Cal-DPO objective in `dpo_kl.py`. |
| CPO | `cpo` | `function_calling/sweep/cpo.yaml` | Preference β → `exp.loss_beta`; `exp.loss_alpha` holds the NLL-style weight (see CPO branch in `dpo_kl.py`). |
| Average | `average_both` | `function_calling/sweep/average_both.yaml` | ε_win → `exp.tol_win`; ε_loose → `exp.tol_loose`. |
| Pointwise | `aug_dual_both` | `function_calling/sweep/pointwise.yaml` | Same ε’s; **augmented dual** (pointwise) objective. |
| Penalty | `penalty_both` | `function_calling/sweep/penalty.yaml` | λ₀ → `exp.loss_alpha`. |

**Resilient** (`resilient_both`) appears in the main results / Table 10; final configs only live under **`function_calling/llama/`** and **`function_calling/xlam/`**.

### Table 5 → shared training settings

Effective batch **32** (`per_device_train_batch_size: 1`, `gradient_accumulation_steps: 32`), **AdamW** `5e-5`, **cosine**, **warmup 0**, **weight decay 0**, **bf16**, **max_prompt_length 1024**, **max_length 2048**, LoRA **r=8, α=16, dropout=0.05**, **5% val** holdout from training split, **`when2call`** dataset.

### Final hyperparameters (fixed across seeds)

The **`function_calling/llama/`** and **`function_calling/xlam/`** YAMLs freeze the values used for the **main-result rows** (best on the paper’s validation protocol). Update these blocks if your validation argmax differs.

| Method | Key settings |
|--------|----------------|
| SimPO | `tol=1`, `loss_alpha=2.5`, 10 epochs |
| DPO | `loss_alpha=0.1`, 10 epochs |
| Cal-DPO | `loss_alpha=0.1`, 10 epochs |
| CPO | `loss_alpha=1.0`, `loss_beta=0.25`, 10 epochs |
| Average | `tol_win=-2`, `tol_loose=-40`, `loss_alpha=10`, `dual_step_size=1`, 10 epochs |
| Pointwise | `tol_win=-2`, `tol_loose=-20`, `loss_alpha=10`, `dual_step_size=0.1`, 10 epochs |
| Penalty | `loss_alpha=10`, `dual_step_size=1`, `tol_win=-2`, `tol_loose=-20`, 10 epochs |
| Resilient | `tol_win=-2`, `tol_loose=-20`, `loss_alpha=10`, `dual_step_size=1`, `resilient_coef=10`, 10 epochs |

### xLAM vs Llama

**`function_calling/xlam/`** sets `train.use_xlam_prompt_format: true`, `when2call_strict_prompt: false`, and optional objective/constraint fractions (matching other xLAM When2Call configs in this repo). **`function_calling/llama/`** uses strict When2Call prompting and no xLAM format flag.

### W&B

Default project: **`paper_when2call`** (override via `train.wandb_project=...`).
