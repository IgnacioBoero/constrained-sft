# `dpo_kl.py` Losses Explained

This document describes every `loss_type` implemented in `src/experiments/dpo_kl.py`, including shared notation, equations, constraints, and key config knobs.

## Shared Setup And Notation

The trainer builds paired samples `(chosen, rejected)` and computes per-token log-probabilities only on response tokens.

- `logp_chosen`: average response-token log-prob of the chosen answer.
- `logp_rejected`: average response-token log-prob of the rejected answer.
- `gap = logp_chosen - logp_rejected`.
- `tol` / `tol_win` / `tol_loose`: constraint thresholds from config.

For KL-based losses (all except `simpo`, `slic_hf`, `dpo`, and `cal_dpo`), a reference/base policy is computed by running the same model with adapters disabled, then:

- token KL: `sum_v p_theta(v) * (log p_theta(v) - log p_ref(v))`
- sequence KL: average over response tokens
- pair KL objective: `kl_mean = 0.5 * (kl_chosen + kl_rejected)`

The raw unconstrained objective is:

- `objective = kl_mean` (KL-based family)
- `objective = softplus(-score)` (SimPO, DPO)

## Constraint Slacks

The code uses slack variables where `slack <= 0` means the constraint is satisfied.

- Pairwise margin slack: `slack = tol - gap`
  - equivalent to `tol + logp_rejected - logp_chosen`
  - enforces `logp_chosen - logp_rejected >= tol`
- Chosen floor slack: `slack_win = tol_win - logp_chosen`
  - fallback keys: `epsilon_win`, then `tol`
- Rejected ceiling slack: `slack_loose = logp_rejected - tol_loose`
  - fallback keys: `epsilon_loose`, then `tol2`, then `tol`

## Loss Types

### 1) `simpo`

No KL term and no base-model pass.

- `gamma = tol`
- `beta = loss_alpha`
- `score = beta * gap - gamma`
- `loss = mean(softplus(-score))`

Interpretation: logistic margin objective directly on `gap`.

### 2) `dpo`

No KL term. Uses reference/base gap subtraction.

- `beta = loss_alpha` (default 1.0)
- `base_gap = logp_ref(chosen) - logp_ref(rejected)`
- `score = beta * (gap - base_gap)`
- `loss = mean(softplus(-score))`

Interpretation: standard DPO preference score against frozen reference behavior.

### 3) `cal_dpo`

No KL term. Uses the same reference-adjusted log-ratios as DPO:

- `rel_logp_chosen = logp_chosen - base_logp_chosen`
- `rel_logp_rejected = logp_rejected - base_logp_rejected`
- `score = rel_logp_chosen - rel_logp_rejected`
- `beta = loss_alpha`
- `target = 1 / (2 * beta)`
- `loss = mean(softplus(-score) + (rel_logp_chosen - target)^2 + (rel_logp_rejected + target)^2)`

Interpretation: DPO with an unscaled preference term (`beta = 1` for the logistic score) plus quadratic calibration penalties that pull chosen and rejected log-ratios toward `+1/(2*loss_alpha)` and `-1/(2*loss_alpha)`.

### 4) `slic_hf`

No KL term and no base-model pass.

- `delta = tol`
- `lambda = loss_alpha`
- `gap = logp_chosen - logp_rejected`
- `score = gap - delta`
- `loss = mean(relu(-score) - lambda * logp_chosen)`

Interpretation: hinge margin on the pairwise preference (`gap >= delta`) plus a chosen-logprob reward term weighted by `loss_alpha`.

### 5) `erm`

Objective-only KL minimization, no constraint penalty.

- `loss = mean(kl_mean)`

### 6) `avg`

Single average (global) dual variable for pairwise slack.

- Dual update (train only):  
  `mu <- clamp(mu + dual_step_size * mean(slack), min=0)`
- Loss:  
  `loss = mean(kl_mean + mu * slack)`

### 7) `aug_dual`

Single per-example augmented-Lagrangian dual variable (`dual_vars[index]`) for pairwise slack.

- `a = slack`
- `b = lambda / loss_alpha`
- `z = 2*a + b`
- Dual gradient (piecewise):
  - if `z > 0`: `dual_grad = a`
  - else: `dual_grad = -0.5 * b`
- Dual update (train only):  
  `lambda <- lambda + dual_step_size * dual_grad`
- Penalty term:
  - `loss += loss_alpha/4 * (relu(z)^2 - b^2)`
- Final: batch mean.

### 8) `resilient`

Variant of `aug_dual` with additional resilience coefficient.

- `a = slack`
- `a_resilient = slack - (lambda/2) * resilient_coef`
- `b = lambda / loss_alpha`
- `coef = resilient_coef / (loss_alpha + resilient_coef)`
- `z = 2*a + b`
- Dual gradient (piecewise):
  - if `z > 0`: `dual_grad = coef * a_resilient`
  - else: `dual_grad = -0.5 * b`
- Dual update (train only):  
  `lambda <- lambda + dual_step_size * dual_grad`
- Penalty term:
  - `loss += loss_alpha/4 * (coef * relu(z)^2 - b^2)`
- Final: batch mean.

### 9) `penalty`

Linear pairwise slack penalty.

- `loss = mean(kl_mean + loss_alpha * slack)`

### 10) `_both_avg` aliases

Aliases: `_both_avg`, `both_avg`, `avg_both`

Two global duals, one for `slack_win` and one for `slack_loose`.

- Updates (train only):
  - `mu_win <- clamp(mu_win + dual_step_size * mean(slack_win), min=0)`
  - `mu_loose <- clamp(mu_loose + dual_step_size * mean(slack_loose), min=0)`
- Weights:
  - `alpha_win = loss_alpha_win` (fallback `loss_alpha`)
  - `alpha_loose = loss_alpha_loose` (fallback `loss_alpha`)
- Loss:
  - `loss = mean(kl_mean + alpha_win*mu_win*slack_win + alpha_loose*mu_loose*slack_loose)`

### 11) `_both_aug_dual` aliases

Aliases: `_both_aug_dual`, `both_aug_dual`, `aug_dual_both`

Two per-example augmented-dual terms, one for `slack_win`, one for `slack_loose`, each with its own dual vector and alpha (`loss_alpha_win`, `loss_alpha_loose`, each clamped to `>= 1e-12`).

Each side uses the same augmented form as `aug_dual` (`a`, `b`, `z`, piecewise dual gradient), and the final loss adds both penalty terms to `kl_mean`, then averages.

### 12) `_both_penalty` aliases

Aliases: `_both_penalty`, `both_penalty`, `penalty_both`

Linear penalties on both explicit constraints.

- `loss = mean(kl_mean + alpha_win*slack_win + alpha_loose*slack_loose)`
- `alpha_win = loss_alpha_win` fallback `loss_alpha`
- `alpha_loose = loss_alpha_loose` fallback `loss_alpha`

### 13) `aug_dual_three` aliases

Aliases: `aug_dual_three`, `_aug_dual_three`, `three_aug_dual`

Three per-example augmented-dual constraints:

1. pairwise gap slack (`slack = tol - gap`) with `loss_alpha_gap`
2. chosen floor slack (`slack_win`) with `loss_alpha_win`
3. rejected ceiling slack (`slack_loose`) with `loss_alpha_loose`

Each uses the same augmented piecewise update as `aug_dual` but with independent dual vectors and alpha values. Final loss is KL objective plus all three augmented terms, then mean.

## Which Losses Use A Reference/Base Pass?

- Uses reference pass (`disable_adapter`) and KL machinery:
  - `erm`, `avg`, `aug_dual`, `resilient`, `penalty`
  - `_both_avg` aliases
  - `_both_aug_dual` aliases
  - `_both_penalty` aliases
  - `aug_dual_three` aliases
- No reference pass:
  - `simpo`, `slic_hf`
- Reference pass used for preference score/log-ratio penalties only (not KL objective):
  - `dpo`, `cal_dpo`

## Key Config Fields By Family

- Always relevant:
  - `loss_type`
- Margin-style:
  - `tol`
- SimPO/Slic-HF/DPO/calibration scale:
  - `loss_alpha`
- Dual updates:
  - `dual_step_size`
- Resilient variant:
  - `resilient_coef`
- Two/three-constraint families:
  - `tol_win`, `tol_loose`
  - `loss_alpha_win`, `loss_alpha_loose`
  - `loss_alpha_gap` (three-constraint only)

## Metrics Behavior

`compute_metrics` always logs `logp_gap` and slack statistics. It interprets the 3rd prediction column based on loss family:

- `simpo`: third column is SimPO `score`; logs `objective_simpo_*` and `simpo_margin_sat_rate`.
- `slic_hf`: third column is Slic-HF `score = gap - delta`; logs hinge term, regularizer term, total loss, and `slic_hf_margin_sat_rate`.
- `dpo`: third column is DPO `score`; logs `objective_dpo_*` and `dpo_pref_sat_rate`.
- `cal_dpo`: third column is unscaled DPO `score`; logs DPO term, quadratic penalty, total loss, and `cal_dpo_pref_sat_rate`.
- all others: third column is `kl_mean`; logs `objective_kl`.

For augmented-dual losses, it also logs dual-variable distribution stats (mean/min/max/q90/q99/CVaR), with per-constraint breakdown for the multi-constraint variants.
