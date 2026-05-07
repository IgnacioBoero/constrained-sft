# `dpo_kl.py` Losses Explained

`loss_type` **must** be one of:

`simpo`, `dpo`, `cal_dpo`, `cpo`, `average_both`, `penalty_both`, `aug_dual_both`, `resilient_both`, `dual_both`.

(`CPO` in configs is normalized to `cpo`.)

## Shared notation

- `logp_chosen`, `logp_rejected`: average response-token log-probs under the policy.
- `gap = logp_chosen - logp_rejected`.
- For `dpo` / `cal_dpo` and KL-based objectives: a reference policy is the same model with LoRA adapters disabled; `rel_*` are logits relative to that reference.

## Constraint slacks (two-constraint family)

For `average_both`, `penalty_both`, `aug_dual_both`, `resilient_both`, `dual_both`:

- `slack_win = tol_win - logp_chosen_eff` (effective logp uses `use_pretrained_slacks` like training).
- `slack_loose = logp_rejected_eff - tol_loose`.
- Satisfied when each slack is ≤ 0.

Unconstrained term in the objective: `kl_mean = 0.5 * (kl_chosen + kl_rejected)` vs the frozen reference (sequence-mean token KL on responses).

## Loss summaries

| `loss_type` | Objective |
|-------------|-----------|
| `simpo` | `softplus(-(loss_alpha * gap - tol))` |
| `dpo` | `softplus(-(loss_alpha * rel_gap))` |
| `cal_dpo` | `softplus(-rel_gap) + quadratic penalties on rel logp chosen/rejected` |
| `cpo` | `softplus(-loss_beta * gap) - loss_alpha * logp_chosen` (no ref subtraction) |
| `average_both` | `kl_mean` + avg duals × `slack_win`, `slack_loose` |
| `dual_both` | `kl_mean` + per-example duals × slacks (Lagrangian) |
| `aug_dual_both` | `kl_mean` + augmented dual on win/loose slacks |
| `resilient_both` | `kl_mean` + resilient augmented dual (`resilient_coef`) |
| `penalty_both` | `kl_mean` + `loss_alpha_win * slack_win + loss_alpha_loose * slack_loose` |

See the docstring on `DPO_KL` and `compute_loss` in `dpo_kl.py` for exact update rules.
