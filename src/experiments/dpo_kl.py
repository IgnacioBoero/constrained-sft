import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate.utils import extract_model_from_parallel
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from peft import LoraConfig, get_peft_model

from .base import Experiment
from utils import right_padding


class DPO_KL(Experiment):
    """
    Constrained preference finetuning:
      - Objective: minimize KL(πθ || π0) on response tokens (π0 = pretrained/base model)
      - Constraint: logp(chosen) - logp(rejected) >= tol
        slack := tol + logp(rejected) - logp(chosen)   (constraint satisfied when slack <= 0)
      - Optional two-constraint mode:
        logp(chosen) >= tol_win and logp(rejected) <= tol_loose

    Loss types (mirroring `SAFETY`):
      - erm: objective only
      - avg: average dual variable
      - aug_dual: augmented Lagrangian with per-example dual variables
      - penalty: linear penalty
      - _both_avg: avg dual on (win, loose) slacks with separate duals
      - _both_aug_dual: augmented dual on (win, loose) slacks with separate per-sample duals
      - _both_penalty: linear penalty with separate multipliers for (win, loose) slacks
      - simpo: SimPO objective (no KL regularization), margin gamma = tol
    """
    DEFAULT_CHAT_TEMPLATE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    def _get_default_chat_template(self) -> str:
        cached = getattr(self, "_default_chat_template", None)
        if cached:
            return cached

        fallback_tok = AutoTokenizer.from_pretrained(
            self.DEFAULT_CHAT_TEMPLATE_MODEL,
            use_fast=True,
        )
        template = getattr(fallback_tok, "chat_template", None)
        if not template:
            raise ValueError(
                "Fallback tokenizer has no chat template: "
                f"{self.DEFAULT_CHAT_TEMPLATE_MODEL}"
            )
        self._default_chat_template = template
        return template

    def _ensure_chat_template(self, tok):
        if getattr(tok, "chat_template", None):
            return tok
        tok.chat_template = self._get_default_chat_template()
        print(
            "[dpo_kl] tokenizer has no chat template; "
            f"using template from {self.DEFAULT_CHAT_TEMPLATE_MODEL}"
        )
        return tok

    def load_model_and_tok(self, cfg):
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name, use_fast=True)
        tok = self._ensure_chat_template(tok)
        tok.model_max_length = int(getattr(cfg.train, "max_length", 2048))

        # Ensure pad token exists (common for Llama-family)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Right padding keeps response masks aligned with shifted labels in training.
        tok.padding_side = "right"

        dtype = torch.bfloat16 if getattr(cfg.train.hf_args, "bf16", False) else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            cfg.exp.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        # Training/eval in this script never uses KV-cache; disable it to reduce GPU memory.
        model.config.use_cache = False

        if getattr(cfg.train, "lora", False):
            model = get_peft_model(
                model,
                LoraConfig(
                    r=int(cfg.train.lora.r),
                    lora_alpha=int(cfg.train.lora.lora_alpha),
                    lora_dropout=float(getattr(cfg.train.lora, "lora_dropout", 0.05)),
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                    ],
                ),
            )
        if hasattr(model, "config"):
            model.config.use_cache = False

        # With gradient checkpointing + PEFT, inputs must require grads or loss has no grad_fn.
        if (
            getattr(cfg.train, "hf_args", None) is not None
            and getattr(cfg.train.hf_args, "gradient_checkpointing", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        return model, tok

    def load_datasets(self, cfg):
        cache_dir = getattr(cfg.train, "cache_dir", None)
        val_fraction = float(getattr(cfg.train, "val_fraction", 0.1))
        sanity_check = bool(getattr(cfg.train, "sanity_check", False))
        do_shuffle = bool(getattr(cfg.train, "shuffle", True))

        dataset_name = str(getattr(cfg.train, "dataset", "orca")).lower()

        if dataset_name == "ultra":
            ds = load_dataset(
                "HuggingFaceH4/ultrafeedback_binarized",
                split="train_prefs",
                cache_dir=cache_dir,
            )
            if sanity_check:
                ds = ds.select(range(min(len(ds), 1000)))
        else:
            # Default: Orca DPO pairs
            ds = load_dataset(
                "argilla/distilabel-intel-orca-dpo-pairs",
                split="train",
                cache_dir=cache_dir,
            )
            # Filter as requested
            ds = ds.filter(
                lambda r: (r["status"] != "tie")
                and (r["chosen_score"] >= 8)
                and (not r["in_gsm8k_train"])
            )

        # Optional: proportion cap
        data_prop = float(getattr(cfg.train, "data_proportion", 1.0))
        if not (0.0 < data_prop <= 1.0):
            raise ValueError(f"cfg.train.data_proportion must be in (0,1], got {data_prop}")
        if data_prop < 1.0:
            keep = max(1, int(data_prop * len(ds)))
            ds = ds.select(range(keep))

        # Remove unused / heavy columns up-front (keeps behavior similar to the provided snippet)
        original_columns = ds.column_names

        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name, use_fast=True)
        tok = self._ensure_chat_template(tok)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        if dataset_name == "ultra":
            def chatml_format(r):
                # UltraFeedback: chosen/rejected are message lists;
                # extract the assistant response from index 1.
                msgs = [{"role": "user", "content": r.get("prompt") or ""}]
                prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                chosen = (r["chosen"][1]["content"] if r.get("chosen") else "") + "<|im_end|>\n"
                rejected = (r["rejected"][1]["content"] if r.get("rejected") else "") + "<|im_end|>\n"
                return {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
        else:
            def chatml_format(r):
                # Match the reference notebook formatting (ChatML-style strings)
                system_text = r.get("system") or ""
                msgs = []
                if len(system_text) > 0:
                    msgs.append({"role": "system", "content": system_text})
                msgs.append({"role": "user", "content": r.get("input") or ""})
                prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                chosen = (r.get("chosen") or "") + "<|im_end|>\n"
                rejected = (r.get("rejected") or "") + "<|im_end|>\n"
                return {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }

        ds = ds.map(chatml_format, remove_columns=original_columns)

        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"cfg.train.val_fraction must be in (0,1), got {val_fraction}")

        n_val = max(1, int(len(ds) * val_fraction))
        ev_raw = ds.select(range(n_val))
        tr_raw = ds.select(range(n_val, len(ds)))

        if sanity_check:
            tr_raw = tr_raw.select(range(min(len(tr_raw), 16)))
            ev_raw = ev_raw.select(range(min(len(ev_raw), 16)))

        tr_raw = tr_raw.add_column("split", ["train"] * len(tr_raw))
        ev_raw = ev_raw.add_column("split", ["validation"] * len(ev_raw))

        complete_ds = concatenate_datasets([tr_raw, ev_raw])
        complete_ds = complete_ds.add_column("index", list(range(len(complete_ds))))

        tr = complete_ds.filter(lambda x: x["split"] == "train")
        ev = complete_ds.filter(lambda x: x["split"] == "validation")
        if do_shuffle:
            tr = tr.shuffle(seed=int(cfg.train.seed))
            ev = ev.shuffle(seed=int(cfg.train.seed))

        # Keep only the shortest `filter_shortest_train` fraction of training
        # samples, measured by character length of the "chosen" response.
        filter_shortest = float(getattr(cfg.train, "filter_shortest_train", 1.0))
        if not (0.0 < filter_shortest <= 1.0):
            raise ValueError(
                f"cfg.train.filter_shortest_train must be in (0,1], got {filter_shortest}"
            )
        if filter_shortest < 1.0:
            chosen_lengths = [len(r["chosen"]) for r in tr]
            keep_n = max(1, int(filter_shortest * len(tr)))
            sorted_indices = sorted(range(len(tr)), key=lambda i: chosen_lengths[i])
            tr = tr.select(sorted_indices[:keep_n])

        print(f"Training samples: {len(tr)}, Eval samples: {len(ev)}")
        return tr, ev, complete_ds

    def preprocessing_fn(self, tok, cfg):
        max_length = int(getattr(cfg.train, "max_length", 1536))
        max_prompt_length = int(getattr(cfg.train, "max_prompt_length", max_length))

        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        eos_id = tok.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id set for this experiment.")

        def _prompt_ids(prompt_text: str) -> List[int]:
            prompt_text = prompt_text or ""
            ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

            # Truncate from the LEFT to keep the end-of-prompt (assistant header) intact.
            if len(ids) > max_prompt_length:
                ids = ids[-max_prompt_length:]
            # Ensure room for EOS at end of full sequence.
            if len(ids) > max_length - 1:
                ids = ids[-(max_length - 1) :]
            return ids

        def _build_full(prompt_ids: List[int], answer_text: str) -> Tuple[List[int], List[bool]]:
            ans_ids = tok(answer_text or "", add_special_tokens=False)["input_ids"]
            # Reserve space for EOS
            max_ans = max(0, max_length - len(prompt_ids) - 1)
            ans_ids = ans_ids[:max_ans]
            full = prompt_ids + ans_ids + [eos_id]
            # Response mask: score only answer+EOS (not prompt)
            resp = [False] * len(prompt_ids) + [True] * (len(full) - len(prompt_ids))
            return full, resp

        def fn(sample: Dict[str, Any]):
            prompt_text = sample.get("prompt", "")
            chosen_text = sample.get("chosen", "")
            rejected_text = sample.get("rejected", "")

            p_ids = _prompt_ids(prompt_text)
            chosen_ids, chosen_mask = _build_full(p_ids, chosen_text)
            rejected_ids, rejected_mask = _build_full(p_ids, rejected_text)

            return {
                "chosen_input_ids": chosen_ids,
                "rejected_input_ids": rejected_ids,
                "chosen_response_mask": chosen_mask,
                "rejected_response_mask": rejected_mask,
                "index": int(sample["index"]),
            }

        return fn

    def get_collator(self, tok):
        pad_id = tok.pad_token_id
        if pad_id is None:
            tok.pad_token = tok.eos_token
            pad_id = tok.pad_token_id
        tok.padding_side = "right"

        class Collator:
            def __init__(self, pad_token_id: int) -> None:
                self.pad_token_id = pad_token_id

            def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                B = len(samples)

                # Pad chosen + rejected together to a shared max length for a single stacked forward pass.
                seqs = (
                    [torch.tensor(s["chosen_input_ids"], dtype=torch.long) for s in samples]
                    + [torch.tensor(s["rejected_input_ids"], dtype=torch.long) for s in samples]
                )
                padded = right_padding(seqs, padding_value=self.pad_token_id)  # (2B, L)
                chosen_input_ids = padded[:B]
                rejected_input_ids = padded[B:]

                masks = (
                    [torch.tensor(s["chosen_response_mask"], dtype=torch.bool) for s in samples]
                    + [torch.tensor(s["rejected_response_mask"], dtype=torch.bool) for s in samples]
                )
                padded_masks = right_padding(masks, padding_value=0).bool()  # (2B, L)
                chosen_response_mask = padded_masks[:B]
                rejected_response_mask = padded_masks[B:]

                chosen_attn = chosen_input_ids.ne(self.pad_token_id)
                rejected_attn = rejected_input_ids.ne(self.pad_token_id)

                index = torch.tensor([s["index"] for s in samples], dtype=torch.long)

                return {
                    "chosen_input_ids": chosen_input_ids,
                    "rejected_input_ids": rejected_input_ids,
                    "chosen_attention_mask": chosen_attn.bool(),
                    "rejected_attention_mask": rejected_attn.bool(),
                    "chosen_response_mask": chosen_response_mask,
                    "rejected_response_mask": rejected_response_mask,
                    "index": index,
                }

        return Collator(pad_id)

    def get_trainer_class(self):
        class CustomTrainer(Trainer):
            def __init__(
                self,
                *args,
                custom_cfg=None,
                complete_dataset=None,
                experiment=None,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.custom_cfg = custom_cfg
                self.complete_ds = complete_dataset
                self.experiment = experiment
                self.init_dual_vars()
                # Use per-example `index` as the label returned to metrics/prediction loops.
                self.label_names = ["index"]
                self._generated_eval_answers = False
                self._generated_eval_keys = set()
                self.compute_metrics = self._compute_metrics

            def init_dual_vars(self):
                loss_type = str(self.custom_cfg.exp.loss_type)
                both_avg_aliases = {"_both_avg", "both_avg", "avg_both"}
                both_aug_aliases = {"_both_aug_dual", "both_aug_dual", "aug_dual_both"}

                needs_single_avg_dual = loss_type == "avg"
                needs_both_avg_duals = loss_type in both_avg_aliases
                needs_single_aug_dual = loss_type in {"aug_dual", "resilient"}
                needs_both_aug_duals = loss_type in both_aug_aliases
                needs_any_per_example_duals = needs_single_aug_dual or needs_both_aug_duals

                # Always define attributes so other code paths can safely check/use them.
                self.dual_vars = None
                self.avg_dual = None
                self.dual_vars_win = None
                self.dual_vars_loose = None
                self.avg_dual_win = None
                self.avg_dual_loose = None

                if needs_any_per_example_duals:
                    n = (
                        len(self.complete_ds)
                        if self.complete_ds is not None
                        else len(self.train_dataset or []) + len(self.eval_dataset or [])
                    )
                    if needs_single_aug_dual:
                        self.dual_vars = torch.zeros(
                            n, dtype=torch.float, requires_grad=False, device=self.model.device
                        )
                    if needs_both_aug_duals:
                        # State for two-constraint (_both*) augmented-dual losses.
                        self.dual_vars_win = torch.zeros(
                            n, dtype=torch.float, requires_grad=False, device=self.model.device
                        )
                        self.dual_vars_loose = torch.zeros(
                            n, dtype=torch.float, requires_grad=False, device=self.model.device
                        )

                if needs_single_avg_dual:
                    self.avg_dual = torch.tensor(0.0, device=self.model.device, dtype=torch.float)
                if needs_both_avg_duals:
                    self.avg_dual_win = torch.tensor(0.0, device=self.model.device, dtype=torch.float)
                    self.avg_dual_loose = torch.tensor(0.0, device=self.model.device, dtype=torch.float)

            def train(self, *args, **kwargs):
                out = super().train(*args, **kwargs)
                self._generate_eval_answers_end_of_training()
                return out

            def evaluate(self, *args, **kwargs):
                metrics = super().evaluate(*args, **kwargs)
                self._generate_eval_answers_on_epoch_eval()
                return metrics

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                cfg = self.custom_cfg.exp
                loss_type = str(cfg.loss_type)
                both_avg_aliases = {"_both_avg", "both_avg", "avg_both"}
                both_aug_aliases = {"_both_aug_dual", "both_aug_dual", "aug_dual_both"}
                both_penalty_aliases = {"_both_penalty", "both_penalty", "penalty_both"}
                needs_pairwise_slack = loss_type in {"avg", "aug_dual", "resilient", "penalty"}
                needs_two_constraint_slacks = (
                    loss_type in both_avg_aliases
                    or loss_type in both_aug_aliases
                    or loss_type in both_penalty_aliases
                )

                chosen_ids = inputs["chosen_input_ids"]
                rejected_ids = inputs["rejected_input_ids"]
                chosen_attn = inputs["chosen_attention_mask"]
                rejected_attn = inputs["rejected_attention_mask"]
                chosen_resp = inputs["chosen_response_mask"]
                rejected_resp = inputs["rejected_response_mask"]
                index = inputs["index"]

                B = chosen_ids.shape[0]

                input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)  # (2B, L)
                attention_mask = torch.cat([chosen_attn, rejected_attn], dim=0)
                response_mask = torch.cat([chosen_resp, rejected_resp], dim=0)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits[:, :-1]  # (2B, L-1, V)
                del outputs
                log_probs = F.log_softmax(logits, dim=-1)

                # logp of the realized tokens (average over response tokens)
                token_logp = torch.gather(
                    log_probs,
                    dim=-1,
                    index=input_ids[:, 1:].unsqueeze(-1),
                ).squeeze(-1)  # (2B, L-1)
                token_logp = token_logp * response_mask[:, 1:]
                num_tokens = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                avg_logp = token_logp.sum(dim=-1) / num_tokens  # (2B,)
                logp_chosen = avg_logp[:B]
                logp_rejected = avg_logp[B:]
                gap = None
                if loss_type == "simpo" or needs_pairwise_slack:
                    gap = logp_chosen - logp_rejected  # (B,)

                # SimPO objective (no KL regularization / no base model pass):
                #   log σ( β/|y_w| logπ(y_w|x) - β/|y_l| logπ(y_l|x) - γ )
                # We use per-token average logprobs, so this becomes:
                #   log σ( β*(logp_chosen - logp_rejected) - γ )
                # and set γ from the existing constraint tolerance (cfg.tol).
                if loss_type == "simpo":
                    gamma = cfg.tol
                    beta_simpo = cfg.loss_alpha
                    if gap is None:
                        gap = logp_chosen - logp_rejected
                    score = beta_simpo * gap - gamma  # (B,)
                    # -log σ(score) == softplus(-score)
                    loss = F.softplus(-score).mean()
                    if return_outputs:
                        packed = torch.stack([logp_chosen, logp_rejected, score], dim=-1)
                        return loss, {"logits": packed}
                    return loss

                # KL objective vs pretrained/base (adapters disabled)
                with torch.no_grad():
                    if hasattr(model, "disable_adapter"):
                        with model.disable_adapter():
                            base_out = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                            )
                    elif hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
                        with model.module.disable_adapter():
                            base_out = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                            )
                    else:
                        raise RuntimeError(
                            "KL objective requires PEFT LoRA adapters. "
                            "Expected model to expose `.disable_adapter()` (PeftModel)."
                        )
                    base_logits = base_out.logits[:, :-1]
                    del base_out
                    base_log_probs = F.log_softmax(base_logits, dim=-1)

                probs = log_probs.exp()
                kl_token = (probs * (log_probs - base_log_probs)).sum(dim=-1)  # (2B, L-1)
                kl_token = kl_token * response_mask[:, 1:]
                kl_seq = kl_token.sum(dim=-1) / num_tokens  # (2B,)
                kl_chosen = kl_seq[:B]
                kl_rejected = kl_seq[B:]
                kl_mean = 0.5 * (kl_chosen + kl_rejected)  # (B,)

                # Objective: minimize KL (matches `SAFETY`; no separate beta scaling)
                loss = kl_mean

                slack = None
                if needs_pairwise_slack:
                    if gap is None:
                        gap = logp_chosen - logp_rejected
                    # Constraint slack: tol + logp(rejected) - logp(chosen) == tol - (logp_chosen - logp_rejected)
                    slack = cfg.tol - gap  # (B,)

                slack_win = None
                slack_loose = None
                if needs_two_constraint_slacks:
                    # Two explicit constraints:
                    #   logp(chosen) >= tol_win      -> slack_win <= 0
                    #   logp(rejected) <= tol_loose  -> slack_loose <= 0
                    eps_win = float(
                        getattr(
                            cfg,
                            "tol_win",
                            getattr(cfg, "epsilon_win", getattr(cfg, "tol", 0.0)),
                        )
                    )
                    eps_loose = float(
                        getattr(
                            cfg,
                            "tol_loose",
                            getattr(cfg, "epsilon_loose", getattr(cfg, "tol2", getattr(cfg, "tol", 0.0))),
                        )
                    )
                    slack_win = eps_win - logp_chosen
                    slack_loose = logp_rejected - eps_loose

                # Loss schemes (as in SAFETY)
                if loss_type == "erm":
                    pass

                elif loss_type == "avg":
                    if model.training:
                        with torch.no_grad():
                            dual_avg = self.avg_dual.clone()
                            dual_avg = torch.clamp(
                                dual_avg + cfg.dual_step_size * slack.mean(),
                                min=0.0,
                            )
                        self.avg_dual = dual_avg.detach()
                    loss = loss + self.avg_dual * slack

                elif loss_type == "aug_dual":
                    dual_var = self.dual_vars[index].clone()
                    a = slack
                    b = dual_var / (cfg.loss_alpha)
                    z = 2 * a + b
                    with torch.no_grad():
                        dual_grad = torch.where(z > 0, a, -0.5 * b)
                        if model.training:
                            dual_var = dual_var + cfg.dual_step_size * dual_grad
                            self.dual_vars[index] = dual_var.detach()
                    loss = loss + cfg.loss_alpha / 4 * (torch.clamp(z, min=0.0) ** 2 - b**2)
                    loss = loss.mean()
                elif loss_type == "resilient":
                    
                    dual_var = self.dual_vars[index].clone()
                    a = slack
                    a_resilient = slack - dual_var / 2 * (cfg.resilient_coef)
                    b = dual_var / (cfg.loss_alpha)
                    coef = (cfg.resilient_coef) / (cfg.loss_alpha + cfg.resilient_coef)
                    z = 2 * a + b
                    with torch.no_grad():
                        dual_grad = torch.where(z > 0, coef * a_resilient, -0.5 * b)
                        if model.training:
                            dual_var += cfg.dual_step_size * dual_grad
                            self.dual_vars[index] = dual_var.detach()
                    loss = loss + cfg.loss_alpha / 4 * (
                        coef * torch.clamp(z, min=0.0) ** 2 - b**2
                    )

                elif loss_type == "penalty":
                    loss = loss + cfg.loss_alpha * slack

                elif loss_type in both_avg_aliases:
                    alpha_win = float(getattr(cfg, "loss_alpha_win", getattr(cfg, "loss_alpha", 1.0)))
                    alpha_loose = float(getattr(cfg, "loss_alpha_loose", getattr(cfg, "loss_alpha", 1.0)))
                    with torch.no_grad():
                        if model.training:
                            dual_avg_win = self.avg_dual_win.clone()
                            dual_avg_win = torch.clamp(
                                dual_avg_win + cfg.dual_step_size * slack_win.mean(),
                                min=0.0,
                            )
                            self.avg_dual_win = dual_avg_win.detach()

                            dual_avg_loose = self.avg_dual_loose.clone()
                            dual_avg_loose = torch.clamp(
                                dual_avg_loose + cfg.dual_step_size * slack_loose.mean(),
                                min=0.0,
                            )
                            self.avg_dual_loose = dual_avg_loose.detach()

                    loss = (
                        loss
                        + alpha_win * self.avg_dual_win.detach() * slack_win
                        + alpha_loose * self.avg_dual_loose.detach() * slack_loose
                    )

                elif loss_type in both_aug_aliases:
                    alpha_win = float(getattr(cfg, "loss_alpha_win", getattr(cfg, "loss_alpha", 1.0)))
                    alpha_loose = float(getattr(cfg, "loss_alpha_loose", getattr(cfg, "loss_alpha", 1.0)))
                    alpha_win = max(alpha_win, 1e-12)
                    alpha_loose = max(alpha_loose, 1e-12)

                    dual_var_win = self.dual_vars_win[index].clone()
                    a_win = slack_win
                    b_win = dual_var_win / alpha_win
                    z_win = 2 * a_win + b_win
                    with torch.no_grad():
                        dual_grad_win = torch.where(z_win > 0, a_win, -0.5 * b_win)
                        if model.training:
                            dual_var_win = dual_var_win + cfg.dual_step_size * dual_grad_win
                            self.dual_vars_win[index] = dual_var_win.detach()
                    loss = loss + alpha_win / 4 * (torch.clamp(z_win, min=0.0) ** 2 - b_win**2)

                    dual_var_loose = self.dual_vars_loose[index].clone()
                    a_loose = slack_loose
                    b_loose = dual_var_loose / alpha_loose
                    z_loose = 2 * a_loose + b_loose
                    with torch.no_grad():
                        dual_grad_loose = torch.where(z_loose > 0, a_loose, -0.5 * b_loose)
                        if model.training:
                            dual_var_loose = dual_var_loose + cfg.dual_step_size * dual_grad_loose
                            self.dual_vars_loose[index] = dual_var_loose.detach()
                    loss = loss + alpha_loose / 4 * (torch.clamp(z_loose, min=0.0) ** 2 - b_loose**2)

                elif loss_type in both_penalty_aliases:
                    alpha_win = float(getattr(cfg, "loss_alpha_win", getattr(cfg, "loss_alpha", 1.0)))
                    alpha_loose = float(getattr(cfg, "loss_alpha_loose", getattr(cfg, "loss_alpha", 1.0)))
                    loss = loss + alpha_win * slack_win + alpha_loose * slack_loose

                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

                loss = loss.mean()

                if return_outputs:
                    # Return compact predictions for metrics: [logp_chosen, logp_rejected, third]
                    # third = KL mean for KL-based methods
                    packed = torch.stack([logp_chosen, logp_rejected, kl_mean], dim=-1)
                    return loss, {"logits": packed}
                return loss

            def _compute_metrics(self, pred):
                cfg = self.custom_cfg
                exp_cfg = cfg.exp
                loss_type = str(exp_cfg.loss_type)
                both_avg_aliases = {"_both_avg", "both_avg", "avg_both"}
                both_aug_aliases = {"_both_aug_dual", "both_aug_dual", "aug_dual_both"}
                arr = pred.predictions
                # Flatten prediction chunks robustly (match SAFETY behavior):
                # keep all rows from the full evaluated dataset.
                if isinstance(arr, (list, tuple)):
                    parts = []
                    for x in arr:
                        x_np = np.asarray(x)
                        if x_np.ndim == 1:
                            x_np = x_np[:, None]
                        parts.append(x_np)
                    if not parts:
                        return {}
                    arr = np.concatenate(parts, axis=0)
                else:
                    arr = np.asarray(arr)
                if arr.ndim != 2 or arr.shape[1] < 3:
                    return {}
                idx_chunks = pred.label_ids
                if isinstance(idx_chunks, (list, tuple)):
                    indexes = np.concatenate([np.asarray(x) for x in idx_chunks], axis=0)
                else:
                    indexes = np.asarray(idx_chunks)
                indexes = np.asarray(indexes).reshape(-1)
                logp_chosen = arr[:, 0]
                logp_rejected = arr[:, 1]
                gap = logp_chosen - logp_rejected
                eps_win = float(
                    getattr(
                        exp_cfg,
                        "tol_win",
                        getattr(exp_cfg, "epsilon_win", getattr(exp_cfg, "tol", 0.0)),
                    )
                )
                eps_loose = float(
                    getattr(
                        exp_cfg,
                        "tol_loose",
                        getattr(exp_cfg, "epsilon_loose", getattr(exp_cfg, "tol2", getattr(exp_cfg, "tol", 0.0))),
                    )
                )
                slack_win_values = eps_win - logp_chosen
                slack_loose_values = logp_rejected - eps_loose
                # Match SAFETY convention: "constraint" is the negated log-ratio.
                # constraint = -(logp_chosen - logp_rejected)
                constraint_values = -gap
                if indexes.shape[0] != constraint_values.shape[0]:
                    # Fallback for unexpected prediction/label packing mismatch.
                    indexes = np.arange(constraint_values.shape[0], dtype=np.int64)

                # Store full-eval tensors for shared eval callbacks (eval/train prefixes).
                self._last_constraint_slacks = torch.tensor(constraint_values, dtype=torch.float).detach().cpu()
                self._last_constraint_indexes = torch.tensor(indexes, dtype=torch.long).detach().cpu()
                self._last_constraint_slacks_win = torch.tensor(slack_win_values, dtype=torch.float).detach().cpu()
                self._last_constraint_slacks_loose = torch.tensor(slack_loose_values, dtype=torch.float).detach().cpu()
                self._last_objective_ratios = torch.tensor(gap, dtype=torch.float).detach().cpu()
                self._last_objective_indexes = torch.tensor(indexes, dtype=torch.long).detach().cpu()

                def _slack_stats(values_np: np.ndarray) -> Dict[str, float]:
                    mean_v = float(np.mean(values_np))
                    min_v = float(np.min(values_np))
                    max_v = float(np.max(values_np))
                    q90_v = float(np.quantile(values_np, 0.9))
                    q99_v = float(np.quantile(values_np, 0.99))
                    tail_v = values_np[values_np > q90_v]
                    cvar_v = float(np.mean(tail_v)) if tail_v.size > 0 else max_v
                    return {
                        "mean": mean_v,
                        "min": min_v,
                        "max": max_v,
                        "q90": q90_v,
                        "q99": q99_v,
                        "cvar": cvar_v,
                    }

                constraint_stats = _slack_stats(constraint_values)
                win_stats = _slack_stats(slack_win_values)
                loose_stats = _slack_stats(slack_loose_values)

                out = {
                    "logp_gap": float(np.mean(gap)),
                    "constraint_mean": constraint_stats["mean"],
                    "constraint_min": constraint_stats["min"],
                    "constraint_max": constraint_stats["max"],
                    "constraint_q90": constraint_stats["q90"],
                    "constraint_q99": constraint_stats["q99"],
                    "constraint_cvar": constraint_stats["cvar"],
                    "slack_win_mean": win_stats["mean"],
                    "slack_win_min": win_stats["min"],
                    "slack_win_max": win_stats["max"],
                    "slack_win_q90": win_stats["q90"],
                    "slack_win_q99": win_stats["q99"],
                    "slack_win_cvar": win_stats["cvar"],
                    "slack_loose_mean": loose_stats["mean"],
                    "slack_loose_min": loose_stats["min"],
                    "slack_loose_max": loose_stats["max"],
                    "slack_loose_q90": loose_stats["q90"],
                    "slack_loose_q99": loose_stats["q99"],
                    "slack_loose_cvar": loose_stats["cvar"],
                }
                dual_stats = None
                dual_stats_win = None
                dual_stats_loose = None
                if loss_type in {"aug_dual", "resilient"} or loss_type in both_aug_aliases:
                    if (
                        self.train_dataset is not None
                        and hasattr(self.train_dataset, "column_names")
                        and "index" in self.train_dataset.column_names
                    ):
                        train_indexes = list(self.train_dataset["index"])
                    else:
                        train_indexes = (
                            list(range(len(self.train_dataset)))
                            if self.train_dataset is not None
                            else []
                        )

                    def _dual_stats_for(dual_tensor):
                        if not train_indexes:
                            return None
                        dual_vals = dual_tensor.detach().cpu().numpy()
                        selected = [
                            float(dual_vals[int(idx)])
                            for idx in train_indexes
                            if 0 <= int(idx) < len(dual_vals)
                        ]
                        if not selected:
                            return None
                        selected_np = np.asarray(selected, dtype=np.float64)
                        dual_q90 = float(np.quantile(selected_np, 0.9))
                        dual_q99 = float(np.quantile(selected_np, 0.99))
                        dual_tail = selected_np[selected_np > dual_q90]
                        dual_cvar = (
                            float(np.mean(dual_tail))
                            if dual_tail.size > 0
                            else float(np.max(selected_np))
                        )
                        return {
                            "dual_vars_mean": float(np.mean(selected_np)),
                            "dual_vars_min": float(np.min(selected_np)),
                            "dual_vars_max": float(np.max(selected_np)),
                            "dual_vars_q90": dual_q90,
                            "dual_vars_q99": dual_q99,
                            "dual_vars_cvar": dual_cvar,
                            "_train_indexes": train_indexes,
                            "_dual_vals": dual_vals if train_indexes else None,
                        }

                    if loss_type in {"aug_dual", "resilient"}:
                        dual_stats = _dual_stats_for(self.dual_vars)
                        if dual_stats is not None:
                            out.update(
                                {
                                    "dual_vars_mean": dual_stats["dual_vars_mean"],
                                    "dual_vars_min": dual_stats["dual_vars_min"],
                                    "dual_vars_max": dual_stats["dual_vars_max"],
                                    "dual_vars_q90": dual_stats["dual_vars_q90"],
                                    "dual_vars_q99": dual_stats["dual_vars_q99"],
                                    "dual_vars_cvar": dual_stats["dual_vars_cvar"],
                                }
                            )
                    else:
                        dual_stats_win = _dual_stats_for(self.dual_vars_win)
                        dual_stats_loose = _dual_stats_for(self.dual_vars_loose)
                        if dual_stats_win is not None:
                            out.update(
                                {
                                    "dual_vars_win_mean": dual_stats_win["dual_vars_mean"],
                                    "dual_vars_win_min": dual_stats_win["dual_vars_min"],
                                    "dual_vars_win_max": dual_stats_win["dual_vars_max"],
                                    "dual_vars_win_q90": dual_stats_win["dual_vars_q90"],
                                    "dual_vars_win_q99": dual_stats_win["dual_vars_q99"],
                                    "dual_vars_win_cvar": dual_stats_win["dual_vars_cvar"],
                                }
                            )
                        if dual_stats_loose is not None:
                            out.update(
                                {
                                    "dual_vars_loose_mean": dual_stats_loose["dual_vars_mean"],
                                    "dual_vars_loose_min": dual_stats_loose["dual_vars_min"],
                                    "dual_vars_loose_max": dual_stats_loose["dual_vars_max"],
                                    "dual_vars_loose_q90": dual_stats_loose["dual_vars_q90"],
                                    "dual_vars_loose_q99": dual_stats_loose["dual_vars_q99"],
                                    "dual_vars_loose_cvar": dual_stats_loose["dual_vars_cvar"],
                                }
                            )

                if getattr(cfg.train, "use_wandb", False):
                    try:
                        import wandb

                        if wandb.run is not None:
                            epoch = self.state.epoch
                            table = wandb.Table(columns=["constraint_slack"])
                            for s in constraint_values.tolist():
                                table.add_data(s)
                            table_win = wandb.Table(columns=["slack_win"])
                            for s in slack_win_values.tolist():
                                table_win.add_data(s)
                            table_loose = wandb.Table(columns=["slack_loose"])
                            for s in slack_loose_values.tolist():
                                table_loose.add_data(s)
                            prefix = getattr(self, "_current_eval_prefix", "eval")
                            wandb.log(
                                {
                                    f"constraint_slacks_epoch_{epoch}_{prefix}": table,
                                    f"constraint_slacks_win_epoch_{epoch}_{prefix}": table_win,
                                    f"constraint_slacks_loose_epoch_{epoch}_{prefix}": table_loose,
                                }
                            )
                    except Exception as exc:
                        print(f"[dpo_kl] wandb logging failed: {exc}")

                if loss_type in {"aug_dual", "resilient"} and getattr(cfg.train, "use_wandb", False):
                    try:
                        import wandb

                        if wandb.run is not None and dual_stats is not None:
                            epoch = self.state.epoch
                            table = wandb.Table(columns=["index", "dual_var"])
                            train_indexes = dual_stats["_train_indexes"]
                            dual_vals = dual_stats["_dual_vals"]
                            if train_indexes:
                                for idx in train_indexes:
                                    if 0 <= int(idx) < len(dual_vals):
                                        table.add_data(int(idx), float(dual_vals[int(idx)]))
                            prefix = getattr(self, "_current_eval_prefix", "eval")
                            wandb.log(
                                {
                                    f"dual_vars_{prefix}_epoch_{epoch}": table,
                                    f"dual_vars_mean_{prefix}": dual_stats["dual_vars_mean"],
                                    f"dual_vars_min_{prefix}": dual_stats["dual_vars_min"],
                                    f"dual_vars_max_{prefix}": dual_stats["dual_vars_max"],
                                    f"dual_vars_q90_{prefix}": dual_stats["dual_vars_q90"],
                                    f"dual_vars_q99_{prefix}": dual_stats["dual_vars_q99"],
                                    f"dual_vars_cvar_{prefix}": dual_stats["dual_vars_cvar"],
                                }
                            )
                    except Exception as exc:
                        print(f"[dpo_kl] wandb logging failed: {exc}")

                if loss_type in both_aug_aliases and getattr(cfg.train, "use_wandb", False):
                    try:
                        import wandb

                        if wandb.run is not None:
                            epoch = self.state.epoch
                            prefix = getattr(self, "_current_eval_prefix", "eval")
                            payload = {}

                            if dual_stats_win is not None:
                                table_win_dual = wandb.Table(columns=["index", "dual_var_win"])
                                train_indexes = dual_stats_win["_train_indexes"]
                                dual_vals = dual_stats_win["_dual_vals"]
                                if train_indexes:
                                    for idx in train_indexes:
                                        if 0 <= int(idx) < len(dual_vals):
                                            table_win_dual.add_data(int(idx), float(dual_vals[int(idx)]))
                                payload.update(
                                    {
                                        f"dual_vars_win_{prefix}_epoch_{epoch}": table_win_dual,
                                        f"dual_vars_win_mean_{prefix}": dual_stats_win["dual_vars_mean"],
                                        f"dual_vars_win_min_{prefix}": dual_stats_win["dual_vars_min"],
                                        f"dual_vars_win_max_{prefix}": dual_stats_win["dual_vars_max"],
                                        f"dual_vars_win_q90_{prefix}": dual_stats_win["dual_vars_q90"],
                                        f"dual_vars_win_q99_{prefix}": dual_stats_win["dual_vars_q99"],
                                        f"dual_vars_win_cvar_{prefix}": dual_stats_win["dual_vars_cvar"],
                                    }
                                )

                            if dual_stats_loose is not None:
                                table_loose_dual = wandb.Table(columns=["index", "dual_var_loose"])
                                train_indexes = dual_stats_loose["_train_indexes"]
                                dual_vals = dual_stats_loose["_dual_vals"]
                                if train_indexes:
                                    for idx in train_indexes:
                                        if 0 <= int(idx) < len(dual_vals):
                                            table_loose_dual.add_data(int(idx), float(dual_vals[int(idx)]))
                                payload.update(
                                    {
                                        f"dual_vars_loose_{prefix}_epoch_{epoch}": table_loose_dual,
                                        f"dual_vars_loose_mean_{prefix}": dual_stats_loose["dual_vars_mean"],
                                        f"dual_vars_loose_min_{prefix}": dual_stats_loose["dual_vars_min"],
                                        f"dual_vars_loose_max_{prefix}": dual_stats_loose["dual_vars_max"],
                                        f"dual_vars_loose_q90_{prefix}": dual_stats_loose["dual_vars_q90"],
                                        f"dual_vars_loose_q99_{prefix}": dual_stats_loose["dual_vars_q99"],
                                        f"dual_vars_loose_cvar_{prefix}": dual_stats_loose["dual_vars_cvar"],
                                    }
                                )

                            if payload:
                                wandb.log(payload)
                    except Exception as exc:
                        print(f"[dpo_kl] wandb logging failed: {exc}")

                if loss_type == "avg":
                    out["avg_dual"] = float(self.avg_dual.detach().cpu().item())
                if loss_type in both_avg_aliases:
                    out["avg_dual_win"] = float(self.avg_dual_win.detach().cpu().item())
                    out["avg_dual_loose"] = float(self.avg_dual_loose.detach().cpu().item())

                if loss_type == "simpo":
                    # third column is the SimPO score used inside logsigmoid
                    score = arr[:, 2]
                    # logsigmoid(score) = -softplus(-score) = -logaddexp(0, -score)
                    out["objective_simpo_logsigmoid"] = float(np.mean(-np.logaddexp(0.0, -score)))
                    # softplus(-score) = logaddexp(0, -score)
                    out["objective_simpo_loss"] = float(np.mean(np.logaddexp(0.0, -score)))
                    out["simpo_margin_sat_rate"] = float(np.mean(score >= 0.0))
                else:
                    kl_mean = arr[:, 2]
                    out["objective_kl"] = float(np.mean(kl_mean))

                return out

            def _generate_eval_answers_end_of_training(self):
                if self._generated_eval_answers:
                    return None
                self._generated_eval_answers = True
                return self._generate_eval_answers(file_tag="end_of_training")

            def _generate_eval_answers_on_epoch_eval(self):
                # Run generation when eval is configured per-epoch.
                strategy = getattr(self.args, "eval_strategy", None)
                if strategy is None:
                    strategy = getattr(self.args, "evaluation_strategy", None)
                strategy_str = str(strategy).lower() if strategy is not None else ""
                if "epoch" not in strategy_str:
                    return None

                epoch = self.state.epoch
                if epoch is None:
                    key = f"step_{int(self.state.global_step)}"
                else:
                    key = f"epoch_{int(round(float(epoch)))}"
                if key in self._generated_eval_keys:
                    return None
                self._generated_eval_keys.add(key)

                return self._generate_eval_answers(file_tag=key)

            def _generate_eval_answers(self, file_tag: str):

                # Rank-0 only
                rank = int(os.environ.get("RANK", "0"))
                if rank != 0:
                    return None

                cfg = self.custom_cfg
                n = int(getattr(cfg.train, "max_gen", 10) or 10)
                if n <= 0:
                    return None

                ev_ds = self.eval_dataset
                if ev_ds is None or len(ev_ds) == 0:
                    return None

                n = min(n, len(ev_ds))
                # Deterministic subset selection for comparability across runs:
                # pick the rows with the smallest global `index` values (stable across shuffles/DDP).
                row_idxs = None
                try:
                    if hasattr(ev_ds, "column_names") and "index" in ev_ds.column_names:
                        idx_col = np.asarray(ev_ds["index"], dtype=np.int64)
                        row_idxs = np.argsort(idx_col)[:n].tolist()
                except Exception:
                    row_idxs = None
                if row_idxs is None:
                    row_idxs = list(range(n))

                model = extract_model_from_parallel(self.model)
                was_training = model.training
                model.eval()

                rows = []
                for i in row_idxs:
                    sample = ev_ds[int(i)]
                    ids = torch.tensor(sample["chosen_input_ids"], dtype=torch.long, device=model.device)
                    mask = torch.tensor(sample["chosen_response_mask"], dtype=torch.bool, device=model.device)
                    true_pos = torch.where(mask)[0]
                    if true_pos.numel() == 0:
                        continue
                    prompt_end = int(true_pos[0].item())
                    if prompt_end <= 0:
                        continue
                    prompt_ids = ids[:prompt_end].unsqueeze(0)
                    attn = torch.ones_like(prompt_ids, device=model.device)
                    with torch.no_grad():
                        out_ids = model.generate(
                            input_ids=prompt_ids,
                            attention_mask=attn,
                            max_new_tokens=512,
                        )
                    gen_ids = out_ids[0][prompt_ids.shape[1] :]
                    prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True) if self.tokenizer is not None else ""
                    decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True) if self.tokenizer is not None else ""
                    rows.append({"index": int(sample.get("index", -1)), "prompt": prompt_text, "output": decoded})

                if was_training:
                    model.train()

                if not rows:
                    return None

                # Log to wandb + json artifact in output_dir
                out_dir = Path(cfg.train.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"dpo_kl_generate_val10_{file_tag}.json"
                out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

                if cfg.train.use_wandb:
                    try:
                        import wandb

                        if wandb.run is not None:
                            table = wandb.Table(columns=["prompt", "output"])
                            for r in rows:
                                table.add_data(r.get("prompt"), r.get("output"))
                            wandb.log({f"dpo_kl_val_generations_{file_tag}": table})
                            artifact = wandb.Artifact(f"dpo_kl_val_generations_{file_tag}", type="generations")
                            artifact.add_file(str(out_path))
                            wandb.log_artifact(artifact)
                    except Exception as exc:
                        print(f"[dpo_kl] wandb logging failed: {exc}")

                return rows

        return CustomTrainer

