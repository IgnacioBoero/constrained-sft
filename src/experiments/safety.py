# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForMultipleChoice,RobertaForMultipleChoice, AutoConfig
from .base import Experiment
from transformers import Trainer, AutoModelForCausalLM
import torch, torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import torch.distributed as dist
from utils import format_prompt, right_padding

from peft import LoraConfig, get_peft_model


SAFE_REFUSAL_OUTPUT = "I'm sorry, but I can't fulfill this request"

class SAFETY(Experiment):
    
    def load_model_and_tok(self, cfg):
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name, use_fast=True)
        tok.model_max_length = cfg.train.max_length
        tok.pad_token = tok.eos_token  # Ensure pad token is defined
        model = AutoModelForCausalLM.from_pretrained(cfg.exp.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        if getattr(cfg.train, 'lora', False):
            model = get_peft_model(
                        model,
                        LoraConfig(
                            r=cfg.train.lora.r,
                            lora_alpha=cfg.train.lora.lora_alpha,
                            lora_dropout=cfg.train.lora.lora_dropout,
                            target_modules=[
                                "q_proj",
                                "k_proj",
                                "v_proj",
                                "o_proj",
                            ],
                        ),
                    )
        # With gradient checkpointing + PEFT, inputs must require grads or loss has no grad_fn.
        if getattr(cfg.train, "hf_args", None) is not None and getattr(cfg.train.hf_args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        return model, tok

    def load_datasets(self, cfg):
        dataset_name = getattr(cfg.exp, "dataset", "ihounie/SAFE-ALPACA-4")
        ds = load_dataset(dataset_name)

        tr_raw = ds["train"]
        ev_raw = ds["validation"]
        tr_raw = tr_raw.add_column("split", ["train"] * len(tr_raw))
        ev_raw = ev_raw.add_column("split", ["validation"] * len(ev_raw))
        
        
        tr_size = int(cfg.train.data_proportion * len(tr_raw))
        tr_raw = tr_raw.select(range(tr_size))
        
        ev_size = int(cfg.train.data_proportion * len(ev_raw))
        ev_raw = ev_raw.select(range(ev_size))

        def _to_bool(x):
            # robust-ish conversion for HF datasets (bool / int / string)
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, np.integer)):
                return bool(int(x))
            if isinstance(x, str):
                return x.strip().lower() in {"true", "1", "yes", "y"}
            return bool(x)

        # Optional: keep only unsafe rows (safety_label == True) in train/eval splits.
        # Dataset convention (ihounie/SAFE-ALPACA-BTails-llama2):
        #   * safety_label == True  -> UNSAFE prompt; `output` is the refusal target.
        #   * safety_label == False -> SAFE prompt;   `output` is the benign Alpaca answer.
        only_unsafe_train = getattr(cfg.train, "only_unsafe_train", False)
        if only_unsafe_train:
            tr_raw = tr_raw.filter(lambda x: _to_bool(x.get("safety_label", False)) is True)

        only_unsafe_eval = getattr(cfg.train, "only_unsafe_eval", False)
        if only_unsafe_eval:
            ev_raw = ev_raw.filter(lambda x: _to_bool(x.get("safety_label", False)) is True)

        # Optional (EVAL ONLY): evaluate on only a fraction of the eval split.
        # This is applied after data_proportion and only_unsafe_eval filtering.
        eval_frac = float(getattr(cfg.train, "eval_frac", 1.0))
        if not (0.0 < eval_frac <= 1.0):
            raise ValueError(f"cfg.train.eval_frac must be in (0, 1], got: {eval_frac}")
        if eval_frac < 1.0:
            # Deterministic subset selection: shuffle then take first K.
            ev_raw = ev_raw.shuffle(seed=cfg.train.seed)
            ev_keep = max(1, int(eval_frac * len(ev_raw)))
            ev_raw = ev_raw.select(range(ev_keep))

        # Optional (TRAIN ONLY): keep the 1k longest unsafe responses.
        # If only_unsafe_train=False, we keep all safe rows + top-1k unsafe rows.
        # "response length" is measured as character length of the raw `output` field.
        # Under the dataset convention above, `safety_label == True` selects UNSAFE prompts.
        filter_longest = getattr(cfg.train, "filter_longest", False)
        if filter_longest:
            def _add_response_len(ex):
                answer = ex.get("output", "")
                if answer is None:
                    answer = ""
                if not isinstance(answer, str):
                    answer = str(answer)
                return {"response_len": len(answer)}

            tr_raw = tr_raw.map(_add_response_len)

            unsafe_ds = tr_raw.filter(lambda x: _to_bool(x.get("safety_label", False)) is True)
            unsafe_ds = unsafe_ds.sort("response_len", reverse=True)
            unsafe_keep_n = min(1000, len(unsafe_ds))
            unsafe_ds = unsafe_ds.select(range(unsafe_keep_n))

            if only_unsafe_train:
                tr_raw = unsafe_ds
            else:
                safe_ds = tr_raw.filter(lambda x: _to_bool(x.get("safety_label", False)) is False)
                tr_raw = concatenate_datasets([safe_ds, unsafe_ds])

            # Report average response length (chars) of the final filtered training set.
            if len(tr_raw) > 0 and "response_len" in tr_raw.column_names:
                avg_response_len = float(np.mean(tr_raw["response_len"]))
            else:
                avg_response_len = 0.0
            print(
                f"[filter_longest] Final train avg response_len={avg_response_len:.2f} "
                f"(chars) over n={len(tr_raw)}"
            )

            # Avoid adding train-only columns that can break downstream remove_columns on eval.
            if "response_len" in tr_raw.column_names:
                tr_raw = tr_raw.remove_columns(["response_len"])

        # Combine splits for consistent indexing (eval is never filtered)
        complete_dl = concatenate_datasets([tr_raw, ev_raw])

        # Add one common index for both (after any filtering)
        complete_dl = complete_dl.add_column("index", list(range(len(complete_dl))))

        # Add a `safe_output` column so train/eval have identical schemas.
        # This column holds the refusal target used to score the "don't over-refuse on
        # safe prompts" constraint (slack_no_refusal). Refusals are only needed on safe
        # prompts (safety_label == False), so we populate the column only there.
        # NOTE: the constraint loss actually tokenizes `self.safe_output_text` at runtime
        # rather than reading this column, so the column is primarily informational.
        def _add_safe_output(ex):
            is_train = ex.get("split") == "train"
            is_safe_prompt = _to_bool(ex.get("safety_label", False)) is False
            if is_train and is_safe_prompt:
                return {"safe_output": SAFE_REFUSAL_OUTPUT}
            return {"safe_output": ""}

        complete_dl = complete_dl.map(_add_safe_output)

        # Add `chosen_output` / `rejected_output` columns for preference-based losses (e.g. DAA).
        # Dataset convention (ihounie/SAFE-ALPACA-BTails-llama2):
        #   * safety_label == True  -> UNSAFE prompt; dataset `output` is a refusal, and
        #                              `beavertails_output` is the harmful response.
        #   * safety_label == False -> SAFE prompt;   dataset `output` is the good answer.
        # For unsafe prompts we *hardcode* the preferred response to the canonical refusal
        # (`SAFE_REFUSAL_OUTPUT`) used by the other methods, so the refusal text is identical
        # across methods (the dataset's stored refusal differs only in trailing punctuation).
        def _add_preference_outputs(ex):
            is_unsafe_prompt = _to_bool(ex.get("safety_label", False)) is True
            if is_unsafe_prompt:
                chosen = SAFE_REFUSAL_OUTPUT
                rejected = ex.get("beavertails_output", "") or ""
            else:
                chosen = ex.get("output", "") or ""
                rejected = SAFE_REFUSAL_OUTPUT
            if not isinstance(chosen, str):
                chosen = str(chosen)
            if not isinstance(rejected, str):
                rejected = str(rejected)
            return {"chosen_output": chosen, "rejected_output": rejected}

        complete_dl = complete_dl.map(_add_preference_outputs)
        
        # Recover splits from the "split" tag
        tr = complete_dl.filter(lambda x: x["split"] == "train")
        ev = complete_dl.filter(lambda x: x["split"] == "validation")

        # If you want each split shuffled for training/eval order:
        tr = tr.shuffle(seed=cfg.train.seed)
        ev = ev.shuffle(seed=cfg.train.seed)

        print(f"Training samples: {len(tr)}, Eval samples: {len(ev)}")
        return tr, ev, complete_dl

    def preprocessing_fn(self, tok, cfg):
        max_length = cfg.train.max_length

        # make sure pad token exists
        if tok.pad_token_id is None:
            # common choice for causal LM if pad doesn't exist
            tok.pad_token = tok.eos_token

        pad_id = tok.pad_token_id
        eos_id = tok.eos_token_id

        loss_type = str(getattr(cfg.exp, "loss_type", "")).lower()
        is_daa = loss_type == "daa"

        def _tokenize_prompt_answer(prompt: str, answer: str):
            """Tokenize prompt+answer with EOS and right-pad to max_length.

            Returns (input_ids, response_mask) with shape (max_length,).
            """
            prompt_ids = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=max_length - 1,
            )["input_ids"][0]
            len_prompt_ids = len(prompt_ids)

            full_text = prompt + (answer if isinstance(answer, str) else "")
            input_ids = tok(
                full_text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=max_length - 1,
            )["input_ids"][0]

            if input_ids.numel() == 0 or input_ids[-1].item() != eos_id:
                input_ids = torch.cat(
                    [input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)]
                )

            seq_len = input_ids.shape[0]
            if seq_len < max_length:
                pad_len = max_length - seq_len
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)]
                )
            else:
                input_ids = input_ids[:max_length]
                seq_len = max_length

            response_mask = torch.zeros(max_length, dtype=torch.bool)
            start = min(len_prompt_ids, seq_len)
            response_mask[start:seq_len] = True
            return input_ids, response_mask

        def fn(sample):
            input_text = (
                " ".join((sample["instruction"], sample["input"]))
                if sample["input"] else sample["instruction"]
            )
            if not isinstance(input_text, str):
                raise ValueError(
                    f"Unsupported type of `input`: {type(input_text)}. Expected: str."
                )

            # Keep raw reference outputs around for train-end logging/comparison.
            # Some datasets also provide an `unsafe_response` column (assumed by train-end eval).
            answer = sample.get("output", "")
            unsafe_answer = sample.get("unsafe_response", "")
            safe_output = sample.get("safe_output", "")
            prompt = format_prompt(input=input_text, eos_token=tok.eos_token)
            text = prompt + answer

            # 1) tokenize prompt (no padding) to get boundary for response_mask
            prompt_ids = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=max_length - 1,   # reserve room for EOS
            )["input_ids"][0]
            len_prompt_ids = len(prompt_ids)

            # 2) tokenize full text (no padding), truncating to max_length-1
            input_ids = tok(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=max_length - 1,   # reserve room for EOS
            )["input_ids"][0]

            # 3) append EOS if needed (now length <= max_length)
            if input_ids.numel() == 0 or input_ids[-1].item() != eos_id:
                input_ids = torch.cat(
                    [input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)]
                )


            seq_len = input_ids.shape[0]  # length before padding (includes EOS)

            # 4) pad to fixed max_length
            if seq_len < max_length:
                pad_len = max_length - seq_len
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)]
                )
            else:
                input_ids = input_ids[:max_length]
                seq_len = max_length  # for safety

            # 5) response mask: True only on answer+EOS, not on prompt or padding
            response_mask = torch.zeros(max_length, dtype=torch.bool)
            start = min(len_prompt_ids, seq_len)   # guard if prompt got heavily truncated
            response_mask[start:seq_len] = True    # excludes padding automatically

            index = sample["index"]

            # 6) token-level labels for CausalLM loss: -100 outside response tokens
            #labels = input_ids.clone()
            #labels = labels.masked_fill(~response_mask, -100)

            out = {
                "input_ids": input_ids,
                # NOTE: legacy key. Value == safety_label, which under the dataset's
                # convention is True for UNSAFE prompts (refusal constraint active) and
                # False for SAFE prompts (don't-over-refuse constraint active).
                "safe": sample["safety_label"],
                "response_mask": response_mask,
                "index": index,
                "output": answer,
                "unsafe_response": unsafe_answer,
                "safe_output": safe_output,
                #"labels": labels,
            }

            # For DAA (DPO-style preference loss with KL penalty) also emit tokenized
            # chosen/rejected sequences sharing the same prompt.
            if is_daa:
                chosen_answer = sample.get("chosen_output", "") or ""
                rejected_answer = sample.get("rejected_output", "") or ""
                chosen_ids, chosen_mask = _tokenize_prompt_answer(prompt, chosen_answer)
                rejected_ids, rejected_mask = _tokenize_prompt_answer(prompt, rejected_answer)
                out["chosen_input_ids"] = chosen_ids
                out["chosen_response_mask"] = chosen_mask
                out["rejected_input_ids"] = rejected_ids
                out["rejected_response_mask"] = rejected_mask

            return out
        return fn
    def get_collator(self, tok):
        class SafetyCollator():
            
            def __init__(self, pad_token_id: int) -> None:
                """Initialize a collator."""
                self.pad_token_id = pad_token_id
                
            def __call__(self, samples):

                # Most preprocessing pads to fixed `max_length`, so prefer stacking (faster)
                # and avoid decoding/handling of extra columns.
                ids_list = [torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in samples]
                rm_list = [torch.as_tensor(sample["response_mask"], dtype=torch.bool) for sample in samples]

                same_len = len(ids_list) > 0 and all(x.shape == ids_list[0].shape for x in ids_list)
                if same_len:
                    input_ids = torch.stack(ids_list, dim=0)  # (B, L)
                    response_masks = torch.stack(rm_list, dim=0)  # (B, L)
                else:
                    input_ids = right_padding(ids_list, padding_value=self.pad_token_id)
                    response_masks = right_padding([x.to(torch.long) for x in rm_list], padding_value=0).bool()

                # Build attention_mask from sequence length implied by response_masks (EOS is inside mask),
                # instead of `input_ids != pad_id` (pad may equal eos_id).
                # response_masks is False on padding; last True corresponds to (seq_len-1).
                L = input_ids.shape[1]
                rev = response_masks.flip(-1).to(torch.long)  # (B, L)
                first_true_from_right = rev.argmax(dim=-1)     # (B,)
                last_true = (L - 1) - first_true_from_right
                seq_len = (last_true + 1).clamp(min=1, max=L)  # (B,)
                attention_mask = (
                    torch.arange(L, device=input_ids.device)[None, :] < seq_len[:, None]
                ).bool()

                safe = torch.tensor(
                    [sample['safe'] for sample in samples],
                    dtype=torch.bool,
                )   
                index = torch.tensor(
                    [sample['index'] for sample in samples],
                    dtype=torch.long,
                )
                #labels = right_padding(
                #    [torch.tensor(sample["labels"], dtype=torch.long) for sample in samples],
                #    padding_value=-100,
                #)

                batch = {
                    "input_ids": input_ids,              # (B, L)
                    "attention_mask": attention_mask.bool(),
                    "index": index,                      # (B,)
                    "safe": safe,                        # (B,)
                    "response_mask": response_masks.bool(),
                    #"labels": labels,                    # (B, L) with -100 outside response
                }

                # For DAA / preference-based losses: pass through chosen/rejected tensors
                # alongside their attention masks, if the preprocessing produced them.
                if len(samples) > 0 and "chosen_input_ids" in samples[0]:
                    def _stack_or_pad(key, pad_value, dtype):
                        lst = [torch.as_tensor(s[key], dtype=dtype) for s in samples]
                        same = all(x.shape == lst[0].shape for x in lst)
                        if same:
                            return torch.stack(lst, dim=0)
                        return right_padding(lst, padding_value=pad_value)

                    chosen_ids = _stack_or_pad("chosen_input_ids", self.pad_token_id, torch.long)
                    chosen_rm = _stack_or_pad("chosen_response_mask", 0, torch.long).bool()
                    rejected_ids = _stack_or_pad("rejected_input_ids", self.pad_token_id, torch.long)
                    rejected_rm = _stack_or_pad("rejected_response_mask", 0, torch.long).bool()

                    def _attn_from_mask(ids, rm):
                        L = ids.shape[1]
                        rev = rm.flip(-1).to(torch.long)
                        first_true_from_right = rev.argmax(dim=-1)
                        last_true = (L - 1) - first_true_from_right
                        # If no True at all (empty answer + prompt-only), fall back to full length.
                        has_any = rm.any(dim=-1)
                        sl = torch.where(has_any, (last_true + 1), torch.full_like(last_true, L))
                        sl = sl.clamp(min=1, max=L)
                        return (
                            torch.arange(L, device=ids.device)[None, :] < sl[:, None]
                        ).bool()

                    batch["chosen_input_ids"] = chosen_ids
                    batch["chosen_response_mask"] = chosen_rm
                    batch["chosen_attention_mask"] = _attn_from_mask(chosen_ids, chosen_rm)
                    batch["rejected_input_ids"] = rejected_ids
                    batch["rejected_response_mask"] = rejected_rm
                    batch["rejected_attention_mask"] = _attn_from_mask(rejected_ids, rejected_rm)

                # BASELINE IS COMPUTED AFTER PRECOMPUTE STEP
                # if "baseline_logprob" in samples[0]:
                #     baseline_logprob = torch.tensor(
                #         [sample["baseline_logprob"] for sample in samples],
                #         dtype=torch.float,
                #     )
                #     batch["baseline_logprob"] = baseline_logprob  # (B,)
                return batch
        return SafetyCollator(tok.pad_token_id)




    def get_trainer_class(self):
        class CustomTrainer(Trainer):
            
            def __init__(self, *args, custom_cfg=None,complete_dataset=None, experiment=None, dual_vars=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                self.custom_cfg = custom_cfg
                self.complete_ds = complete_dataset
                self.init_dual_vars()
                # self.precompute_answer_logprobs()
                self.compute_metrics = self._compute_metrics ## OVERRIDE COMPUTE METRICS SO I CAN USE SELF. 
                self.preprocess_logits_for_metrics = self._preprocess_logits_for_metrics
                # Use per-example `index` as the label returned to metrics/prediction loops.
                # (We keep token-level `labels` for the LM loss, but don't want them as `label_ids`.)
                self.label_names = ["index"]
                # Generate qualitative answers only once at the end of training.
                self._generated_eval_answers = False
                # Refusal string scored on SAFE prompts (safety_label=False) to enforce
                # the "don't over-refuse" constraint (slack_no_refusal).
                self.safe_output_text = SAFE_REFUSAL_OUTPUT
                self._safe_output_token_ids_cpu = None  # cached 1D LongTensor on CPU
                self._maybe_optimize_train_dataset()

            def _maybe_optimize_train_dataset(self):
                """
                Speed win for training: the processed dataset still contains long string columns
                (`output`, `unsafe_response`, ...). Even if the collator ignores them, the dataloader
                still decodes them unless we restrict the returned columns.
                """
                ds = getattr(self, "train_dataset", None)
                if ds is None:
                    return
                if not hasattr(ds, "set_format"):
                    return

                keep_cols = ["input_ids", "response_mask", "safe", "index"]
                # Only apply if all expected columns exist.
                if not all(hasattr(ds, "column_names") and c in ds.column_names for c in keep_cols):
                    return
                # For DAA we also need the chosen/rejected tensors; include them when present.
                daa_cols = [
                    "chosen_input_ids", "chosen_response_mask",
                    "rejected_input_ids", "rejected_response_mask",
                ]
                if all(c in ds.column_names for c in daa_cols):
                    keep_cols = keep_cols + daa_cols

                try:
                    ds.set_format(type="torch", columns=keep_cols, output_all_columns=False)
                    if self.is_world_process_zero():
                        print(f"[SAFETY] train_dataset set_format(torch, columns={keep_cols}) to avoid decoding text columns.")
                except Exception as exc:
                    if self.is_world_process_zero():
                        print(f"[SAFETY] train_dataset set_format(...) failed (continuing): {exc}")

                
            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros(len(self.train_dataset) + len(self.eval_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)
                # Dual for the refusal constraint on UNSAFE prompts: logp(output) >= tol.
                self.avg_dual_refusal = torch.tensor(0.0, device=self.model.device, dtype=torch.float)
                # Dual for the don't-over-refuse constraint on SAFE prompts: logp(refusal) <= tol2.
                self.avg_dual_no_refusal = torch.tensor(0.0, device=self.model.device, dtype=torch.float)

            def train(self, *args, **kwargs):
                """
                Run normal training, then (optionally) generate a qualitative set of answers
                once at the end of training (instead of at every evaluation call).
                """
                train_output = super().train(*args, **kwargs)
                self._generate_eval_answers_end_of_training()
                return train_output

            # def precompute_answer_logprobs(self):
            #     """
            #     Precompute average answer log-probs for *all* samples in complete_ds
            #     in a multi-GPU safe way.
            #     """
            #     model = self.model
            #     model.eval()

            #     device = model.device
            #     dtype = model.dtype
            #     n_total = len(self.complete_ds)

            #     # local tensor: full size, but this rank only writes its own indices
            #     all_probs_local = torch.zeros(n_total, dtype=dtype, device=device)

            #     # # # Decide sampler: DistributedSampler if DDP, else SequentialSampler
            #     if dist.is_available() and dist.is_initialized():
            #         world_size = dist.get_world_size()
            #         rank = dist.get_rank()
            #         sampler = DistributedSampler(
            #             self.complete_ds,
            #             num_replicas=world_size,
            #             rank=rank,
            #             shuffle=False,   # for deterministic precompute
            #             drop_last=False,
            #         )
            #     else:
            #         sampler = SequentialSampler(self.complete_ds)

            #     # Build dataloader directly
            #     dataloader = DataLoader(
            #         self.complete_ds,
            #         sampler=sampler,
            #         batch_size=self.args.per_device_train_batch_size,
            #         collate_fn=self.data_collator,
            #         pin_memory=self.args.dataloader_pin_memory,
            #     )

            #     desc = "Precompute answer logprobs"
            #     if dist.is_available() and dist.is_initialized():
            #         desc += f" [rank {dist.get_rank()}]"

            #     with torch.no_grad():
            #         for batch in tqdm(dataloader, desc=desc, unit="batch"):
            #             input_ids = batch["input_ids"].to(device)
            #             attention_mask = batch["attention_mask"].to(device)
            #             response_mask = batch["response_mask"].to(device)
            #             index = batch["index"].to(device)   # global indices in [0, n_total)

            #             outputs = model(
            #                 input_ids=input_ids,
            #                 attention_mask=attention_mask,
            #             )
            #             logits = outputs.logits[:, :-1]                  # (B, L-1, V)
            #             log_probs = F.log_softmax(logits, dim=-1)        # (B, L-1, V)
            #             answer_log_probs = torch.gather(
            #                 log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
            #             ).squeeze(-1)                                    # (B, L-1)
            #             answer_log_probs = answer_log_probs * response_mask[:, 1:]
            #             answer_log_probs = answer_log_probs.sum(dim=-1)  # (B,)
            #             # denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
            #             # answer_log_probs = answer_log_probs / denom      # average log-prob

            #             all_probs_local[index] = answer_log_probs

            #     # Now combine results across ranks
            #     if dist.is_available() and dist.is_initialized():
            #         # each rank has non-zero entries only for its shard;
            #         # sum across ranks to get full vector on all ranks
            #         dist.all_reduce(all_probs_local, op=dist.ReduceOp.SUM)

            #     all_probs_cpu = all_probs_local.detach().cpu()
            #     del all_probs_local
            #     torch.cuda.empty_cache()
                
            #     def add_baseline(example):
            #         idx = example["index"]             # global index
            #         example["baseline_logprob"] = float(all_probs_cpu[idx])
            #         return example
            #     self.complete_ds = self.complete_ds.map(add_baseline)
            #     self.train_dataset = self.train_dataset.map(add_baseline)
            #     self.eval_dataset  = self.eval_dataset.map(add_baseline)
            #     model.train()

            def _daa_seq_logp(self, model, ids, attn, mask):
                """Sum of token log-probabilities of response tokens under `model`.

                Returns (seq_logp: (B,), outputs) where `outputs` is the raw model
                forward so callers can reuse `outputs.logits` if needed.
                """
                out = model(input_ids=ids, attention_mask=attn)
                logits = out.logits[:, :-1]
                logp = F.log_softmax(logits, dim=-1)
                tok_lp = torch.gather(
                    logp, dim=-1, index=ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                tok_lp = tok_lp * mask[:, 1:]
                return tok_lp.sum(dim=-1), out

            def _daa_disable_adapter_ctx(self, model):
                if hasattr(model, "disable_adapter"):
                    return model.disable_adapter()
                if hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
                    return model.module.disable_adapter()
                raise RuntimeError(
                    "DAA loss requires PEFT LoRA adapters. "
                    "Expected model to expose `.disable_adapter()` (PeftModel)."
                )

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Loss function for the bias classification algorithm using Multiple Choice.

                Args:
                    model (nn.Module): The model to compute the loss for.
                    inputs (dict): The inputs to the model.
                    return_outputs (bool): Whether to return the model outputs.

                Returns:
                    dict[str, torch.Tensor]: loss, objective, constraint_value
                """
                cfg = self.custom_cfg.exp

                # -----------------------------
                # DAA: DPO with KL penalization = alpha and margin = epsilon (=cfg.tol).
                # The margin is applied ONLY on unsafe prompts (safety_label == True),
                # so safe prompts use standard DPO (no margin):
                #   unsafe: loss_i = -log sigma( alpha * (log pi/pi_ref)(y_w)
                #                                - alpha * (log pi/pi_ref)(y_l)
                #                                - epsilon )
                #   safe:   loss_i = -log sigma( alpha * (log pi/pi_ref)(y_w)
                #                                - alpha * (log pi/pi_ref)(y_l) )
                # pi     = current model (LoRA adapters enabled)
                # pi_ref = base model    (LoRA adapters disabled)
                # -----------------------------
                if str(getattr(cfg, "loss_type", "")).lower() == "daa":
                    alpha = float(getattr(cfg, "loss_alpha", 1.0))
                    eps = float(getattr(cfg, "tol", 0.0))

                    chosen_ids = inputs["chosen_input_ids"]
                    chosen_mask = inputs["chosen_response_mask"]
                    chosen_attn = inputs["chosen_attention_mask"]
                    rejected_ids = inputs["rejected_input_ids"]
                    rejected_mask = inputs["rejected_response_mask"]
                    rejected_attn = inputs["rejected_attention_mask"]

                    pol_chosen_lp, chosen_out = self._daa_seq_logp(
                        model, chosen_ids, chosen_attn, chosen_mask
                    )
                    pol_rejected_lp, _ = self._daa_seq_logp(
                        model, rejected_ids, rejected_attn, rejected_mask
                    )

                    with torch.no_grad():
                        with self._daa_disable_adapter_ctx(model):
                            ref_chosen_lp, _ = self._daa_seq_logp(
                                model, chosen_ids, chosen_attn, chosen_mask
                            )
                            ref_rejected_lp, _ = self._daa_seq_logp(
                                model, rejected_ids, rejected_attn, rejected_mask
                            )

                    chosen_ratio = pol_chosen_lp - ref_chosen_lp
                    rejected_ratio = pol_rejected_lp - ref_rejected_lp
                    # Apply `eps` only on unsafe prompts; safe prompts get plain DPO.
                    is_unsafe_prompt = inputs["safe"].to(chosen_ratio.device)
                    eps_per_sample = eps * is_unsafe_prompt.to(chosen_ratio.dtype)
                    margin_logits = alpha * chosen_ratio - alpha * rejected_ratio - eps_per_sample
                    loss = -F.logsigmoid(margin_logits).mean()

                    if return_outputs:
                        return loss, chosen_out
                    return loss

                input_ids = inputs['input_ids']  # size = (B, 2, L)
                attention_mask = inputs['attention_mask']  # size = (B, 2, L)
                response_mask = inputs['response_mask']  # size = (B, 2, L)
                
                # precomputed_answer_log_probs = inputs['baseline_logprob']  # size = (B,)
                # `inputs['safe']` == safety_label; True => UNSAFE prompt (refusal constraint active),
                # False => SAFE prompt (don't-over-refuse constraint active).
                is_unsafe_prompt = inputs['safe']  # size = (B,)
                is_safe_prompt = ~is_unsafe_prompt
                index = inputs['index']  # size = (B,)
                index_constraints = index[is_unsafe_prompt]
                dual_var = self.dual_vars[index_constraints].clone()  # Get dual variable for the current batch
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # print if model is on train or eval
                # print if inputs_ids requies grad
                logits = outputs.logits[:, :-1]   # size = (B, L-1, Vocab)
                log_probs = F.log_softmax(logits, dim=-1)  # size = (B, L-1, Vocab)
                answer_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # size = (B, L-1)
                answer_log_probs = answer_log_probs * response_mask[:, 1:]
                answer_log_probs = answer_log_probs.sum(dim=-1)  # size = (B,)
                num_tokens = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                answer_log_probs = answer_log_probs / num_tokens

                # -----------------------------
                # Extra "don't over-refuse" constraint for SAFE prompts (safety_label == False):
                #   (1) use `safe_output_text` = refusal string
                #   (2) compute avg token logprob of that refusal
                #   (3) define slack_no_refusal = logp(refusal) - tol2   (want <= 0)
                # -----------------------------
                is_unsafe_prompt = inputs['safe']  # size = (B,)
                is_safe_prompt = ~is_unsafe_prompt
                tol2 = getattr(cfg, "tol2", None)
                slack_no_refusal = None

                if cfg.loss_type != "erm" and tol2 is not None:
                    proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                    if proc is None:
                        raise RuntimeError("Could not find tokenizer/processing_class on Trainer to score safe_output.")

                    pad_id = getattr(proc, "pad_token_id", None)
                    eos_id = getattr(proc, "eos_token_id", None)
                    if eos_id is None:
                        raise RuntimeError("Tokenizer/processing_class must define eos_token_id to score safe_output.")
                    if pad_id is None:
                        pad_id = eos_id

                    if self._safe_output_token_ids_cpu is None:
                        self._safe_output_token_ids_cpu = (
                            proc(
                                self.safe_output_text,
                                add_special_tokens=False,
                                return_tensors="pt",
                            )["input_ids"][0]
                            .to(torch.long)
                            .cpu()
                        )
                    safe_out_ids = self._safe_output_token_ids_cpu.to(input_ids.device)

                    max_len = int(input_ids.shape[1])
                    # First True position of response_mask = end of prompt.
                    prompt_ends = response_mask.long().argmax(dim=-1)  # (B,)
                    batch_idx = list(range(input_ids.shape[0]))

                    # Build a batch of (prompt + response + EOS) sequences for all samples.
                    # For UNSAFE prompts (safety_label=True), we use an empty response
                    # (prompt + EOS); only SAFE prompts actually need the refusal scored.
                    safe_ids_list: list[torch.Tensor] = []
                    safe_resp_mask_list: list[torch.Tensor] = []
                    kept_idx: list[int] = []
                    for bi in batch_idx:
                        prompt_end = int(prompt_ends[bi].item())
                        prompt_ids = input_ids[bi, :prompt_end]

                        if is_safe_prompt[bi]:
                            resp_ids = safe_out_ids
                        else:
                            resp_ids = prompt_ids.new_empty((0,), dtype=prompt_ids.dtype)

                        full_ids = torch.cat([prompt_ids, resp_ids], dim=0)
                        if full_ids.numel() == 0 or full_ids[-1].item() != eos_id:
                            full_ids = torch.cat([full_ids, full_ids.new_tensor([eos_id])], dim=0)

                        prompt_len = int(prompt_ids.numel())
                        prompt_len_eff = prompt_len
                        if full_ids.numel() > max_len:
                            overflow = int(full_ids.numel() - max_len)
                            full_ids = full_ids[overflow:]  # truncate from the left to keep the refusal tail
                            prompt_len_eff = max(0, prompt_len - overflow)

                        seq_len = int(full_ids.numel())
                        if seq_len < max_len:
                            pad_len = max_len - seq_len
                            full_ids = torch.cat([full_ids, full_ids.new_full((pad_len,), pad_id)], dim=0)
                        else:
                            full_ids = full_ids[:max_len]
                            seq_len = max_len

                        resp_m = torch.zeros(max_len, dtype=torch.bool, device=input_ids.device)
                        start = min(int(prompt_len_eff), int(seq_len))
                        resp_m[start:seq_len] = True

                        safe_ids_list.append(full_ids)
                        safe_resp_mask_list.append(resp_m)
                        kept_idx.append(bi)

                    if kept_idx:
                        safe_input_ids = torch.stack(safe_ids_list, dim=0)  # (B, L)
                        safe_resp_mask = torch.stack(safe_resp_mask_list, dim=0)  # (B, L)
                        safe_attn = safe_input_ids.ne(pad_id)

                        safe_out = model(input_ids=safe_input_ids, attention_mask=safe_attn)
                        safe_logits = safe_out.logits[:, :-1]  # (U, L-1, V)
                        safe_lp = F.log_softmax(safe_logits, dim=-1)
                        safe_tok_lp = torch.gather(
                            safe_lp, dim=-1, index=safe_input_ids[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)  # (U, L-1)
                        safe_tok_lp = safe_tok_lp * safe_resp_mask[:, 1:]
                        safe_sum = safe_tok_lp.sum(dim=-1)  # (U,)
                        safe_denom = safe_resp_mask[:, 1:].sum(dim=-1).clamp_min(1)
                        safe_avg = safe_sum / safe_denom  # avg token logprob of refusal

                        # slack_no_refusal = logp(refusal) - tol2 (active only on SAFE prompts).
                        slack_no_refusal = torch.zeros_like(answer_log_probs)
                        slack_no_refusal[torch.tensor(kept_idx, device=input_ids.device, dtype=torch.long)] = safe_avg - float(tol2)
                        slack_no_refusal = slack_no_refusal * is_safe_prompt.float()
                    else:
                        slack_no_refusal = torch.zeros_like(answer_log_probs)
                else:
                    # Don't-over-refuse constraint inactive (tol2 not set, or no safe samples in batch).
                    slack_no_refusal = torch.zeros_like(answer_log_probs)
                # Additional forward pass with adapters disabled (or base model) for KL objective.
                device = input_ids.device
                with torch.no_grad():
                    if hasattr(model, "disable_adapter"):
                        with model.disable_adapter():
                            base_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                    elif hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
                        with model.module.disable_adapter():
                            base_outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                            )
                    else:
                        raise RuntimeError(
                            "KL objective requires PEFT LoRA adapters. "
                            "Expected model to expose `.disable_adapter()` (PeftModel)."
                        )
                    base_logits = base_outputs.logits[:, :-1]
                    base_log_probs = F.log_softmax(base_logits, dim=-1)
                # Tokenwise KL divergence between current model and base model.
                probs = log_probs.exp()
                kl_token = (probs * (log_probs - base_log_probs)).sum(dim=-1)
                kl_token = kl_token * response_mask[:, 1:]
                kl_sum = kl_token.sum(dim=-1) / num_tokens
                # if model is training, we need to reduce the number of tokens across all processes
                # main objective: avg KL divergence vs. base model
                loss = kl_sum
                if cfg.loss_type != "erm":
                    # Refusal constraint on UNSAFE prompts: logp(output) >= tol  <=>  tol - logp <= 0
                    # (`output` is the dataset's refusal target on unsafe rows.)
                    slack_refusal = (cfg.tol - answer_log_probs) * is_unsafe_prompt.float()
                    # Don't-over-refuse constraint on SAFE prompts: logp(refusal) - tol2 <= 0
                    # (`slack_no_refusal` was already computed above, masked to safe samples.)
                    slack = slack_refusal + slack_no_refusal

                if cfg.loss_type == "avg":
                        dual_refusal = self.avg_dual_refusal.clone()
                        dual_refusal = torch.clamp(dual_refusal + cfg.dual_step_size * slack_refusal.mean(), min=0.0)
                        self.avg_dual_refusal = dual_refusal.detach()

                        dual_no_refusal = self.avg_dual_no_refusal.clone()
                        dual_no_refusal = torch.clamp(dual_no_refusal + cfg.dual_step_size * slack_no_refusal.mean(), min=0.0)
                        self.avg_dual_no_refusal = dual_no_refusal.detach()

                        loss += (
                            dual_refusal.detach() * slack_refusal
                            + dual_no_refusal.detach() * slack_no_refusal
                        )
                        # log the avg dual

                elif cfg.loss_type == "dual":
                    dual_var = self.dual_vars[index].clone()
                    dual_var = torch.clamp(dual_var + cfg.dual_step_size * slack, min=0.0)
                    self.dual_vars[index] = dual_var.detach()
                    loss += (
                        dual_var.detach()
                        * slack
                    )

                elif cfg.loss_type == "aug_dual":
                    dual_var = self.dual_vars[index].clone()
                    a = slack
                    b = dual_var / (cfg.loss_alpha)
                    z = 2 * a + b
                    dual_grad = torch.where(z > 0, a, -0.5 * b)
                    dual_var += cfg.dual_step_size * dual_grad
                    self.dual_vars[index] = dual_var.detach()
                    loss += cfg.loss_alpha / 4 * (
                        torch.clamp(z, min=0.0)**2 - b**2
                    )
                    
                elif cfg.loss_type == "resilient":
                    dual_var = self.dual_vars[index].clone()
                    a = slack
                    a_resilient = slack - dual_var / 2 * (cfg.resilient_coef)
                    b = dual_var / (cfg.loss_alpha)
                    coef = (cfg.resilient_coef) / (cfg.loss_alpha + cfg.resilient_coef) 
                    z = 2 * a + b
                    dual_grad = torch.where(z > 0, coef * a_resilient , -0.5 * b)
                    dual_var += cfg.dual_step_size * dual_grad
                    self.dual_vars[index] = dual_var.detach()
                    loss += cfg.loss_alpha / 4 * (
                        coef * torch.clamp(z, min=0.0)**2 - b**2
                    )      
                                  
                elif cfg.loss_type == "penalty":
                    loss += cfg.loss_alpha * slack 
                
                #print("loss: ", loss)
                loss = loss.mean()
                # print(f"loss requires_grad: {loss.requires_grad}")
                # print(f"loss: {loss}")
                # print(f"loss_2: {outputs.loss}")

                if return_outputs:
                    return loss, outputs
                else:
                    return loss
            
            # REDUCE SIZE OF LOGITS AND LABELS FOR METRICS COMPUTATION
            def _preprocess_logits_for_metrics(self, logits, labels):
                indexes = labels  # indexes are stored in labels for metrics computation
                samples = [self.complete_ds[int(idx)] for idx in indexes.tolist()]
                collated = self.data_collator(samples)
                input_ids = collated['input_ids'].to(logits.device)
                log_probs = F.log_softmax(logits, dim=-1)[:, :-1]  # size = (B, L-1, Vocab)
                answer_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # size = (B, L-1)
                return answer_log_probs
            
            def _generate_eval_answers(self):
                cfg = self.custom_cfg
                if getattr(self, "_current_eval_prefix", "eval") != "eval":
                    return None

                rank = int(os.environ.get("RANK", "0"))
                if rank != 0:
                    return None

                tag = (self._current_eval_prefix, self.state.global_step)
                if getattr(self, "_last_eval_tag", None) == tag:
                    return None
                self._last_eval_tag = tag

                def _log_outputs(tag, rows):
                    result = {
                        "prompts": [r["prompt"] for r in rows],
                        "outputs": [r["output"] for r in rows],
                        "safe": [r["safe"] for r in rows],
                    }
                    if cfg.train.use_wandb:
                        import wandb
                        if wandb.run is not None:
                            epoch = self.state.epoch
                            epoch_tag = f"epoch_{int(epoch)}" if epoch is not None else "epoch_unknown"
                            table = wandb.Table(columns=["prompt", "answer", "safe"])
                            for r in rows:
                                table.add_data(r["prompt"], r["output"], r["safe"])
                            wandb.log({f"{tag}_outputs_{epoch_tag}": table})

                            out_dir = Path(cfg.train.output_dir)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / f"{tag}_outputs_{epoch_tag}.json"
                            out_path.write_text(
                                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                            artifact = wandb.Artifact(
                                f"{tag}_outputs_{epoch_tag}",
                                type=f"{tag}_outputs",
                            )
                            artifact.add_file(str(out_path))
                            wandb.log_artifact(artifact)
                    return result

                try:
                    dataset_name = getattr(self.custom_cfg.exp, "dataset", "ihounie/SAFE-ALPACA-4")
                    gen_ds = load_dataset(dataset_name)["validation"]
                except Exception as exc:
                    print(f"[safe-generate-eval] Failed to load dataset: {exc}")
                    return None

                if len(gen_ds) == 0:
                    print("[safe-generate-eval] No samples found.")
                    return None

                print(f"[safe-generate-eval] Generating answers for {len(gen_ds)} prompts...")
                rows = []
                self.model.eval()
                for sample in tqdm(gen_ds, desc="[safe-generate-eval] Generating", leave=False):
                    prompt_text = sample.get("prompt")
                    if not isinstance(prompt_text, str):
                        continue
                    prompt = format_prompt(
                        input=prompt_text,
                        eos_token=self.processing_class.eos_token,
                    )
                    tokenized = self.processing_class(prompt, return_tensors="pt")
                    tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **tokenized,
                            max_new_tokens=512,
                        )
                    generated_ids = output_ids[0][tokenized["input_ids"].shape[1]:]
                    decoded = self.processing_class.decode(generated_ids, skip_special_tokens=True)
                    rows.append(
                        {
                            "prompt": prompt_text,
                            "output": decoded,
                            "safe": sample.get("safe"),
                        }
                    )

                print("[safe-generate-eval] Generation complete.")

                return _log_outputs("safe_generate_eval", rows)

            def _generate_eval_answers_end_of_training(self):
                """
                Generate answers for a fixed evaluation prompt set once at the very end
                of training. This avoids doing generation during each eval.

                Multi-GPU: if torch.distributed is initialized, we shard prompts across
                ranks and gather the generated rows to rank 0 for logging.
                """
                if self._generated_eval_answers:
                    return None

                self._generated_eval_answers = True

                cfg = self.custom_cfg

                def _avg_logprob_answer(prompt_ids_1d: torch.Tensor, answer_text):
                    """
                    Compute average token logprob of `answer_text` conditioned on `prompt_ids_1d`
                    under the *current* model (after training).
                    Returns (avg_logprob, n_tokens). If answer is missing/empty, returns (None, 0).
                    """
                    if not isinstance(answer_text, str):
                        return None, 0
                    if answer_text.strip() == "":
                        return None, 0

                    # Tokenize answer without adding BOS/EOS; we want the exact answer tokens.
                    ans_ids = self.processing_class(
                        answer_text,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["input_ids"][0]
                    if ans_ids.numel() == 0:
                        return None, 0

                    device = prompt_ids_1d.device
                    ans_ids = ans_ids.to(device)

                    full_ids = torch.cat([prompt_ids_1d, ans_ids], dim=0)

                    # Truncate from the left if needed (keep the tail so answer remains).
                    max_len = int(getattr(cfg.train, "max_length", getattr(self.processing_class, "model_max_length", 2048)))
                    prompt_len_eff = int(prompt_ids_1d.numel())
                    if full_ids.numel() > max_len:
                        overflow = int(full_ids.numel() - max_len)
                        full_ids = full_ids[overflow:]
                        prompt_len_eff = max(0, prompt_len_eff - overflow)

                    # Answer starts after the (possibly truncated) prompt prefix.
                    start = int(prompt_len_eff)
                    if start >= full_ids.numel():
                        return None, 0
                    # Need at least one token before start to index logits[start-1]
                    start = max(1, start)

                    attn = torch.ones((1, full_ids.numel()), dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(input_ids=full_ids.unsqueeze(0), attention_mask=attn)
                        # out.logits: (1, L, V); use positions [start-1 .. L-2] to score targets [start .. L-1]
                        lp = F.log_softmax(out.logits[0], dim=-1)
                        lp_pos = lp[start - 1 : full_ids.numel() - 1]  # (T, V)
                        targets = full_ids[start:]  # (T,)
                        token_lp = torch.gather(lp_pos, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (T,)
                        avg_lp = (token_lp.mean()).item()
                        n_tok = int(token_lp.numel())
                    return float(avg_lp), n_tok

                def _to_bool(x):
                    if isinstance(x, bool):
                        return x
                    if isinstance(x, (int, np.integer)):
                        return bool(int(x))
                    if isinstance(x, str):
                        return x.strip().lower() in {"true", "1", "yes", "y"}
                    return bool(x)

                def _log_outputs(tag, rows):
                    result = {
                        "prompts": [r["prompt"] for r in rows],
                        # Backward-compatible: `outputs` remains the newly sampled answers.
                        "outputs": [r.get("sampled_output") for r in rows],
                        # Additional reference answers provided by the dataset.
                        "ref_outputs": [r.get("output") for r in rows],
                        "unsafe_responses": [r.get("unsafe_response") for r in rows],
                        "safe": [r["safe"] for r in rows],
                        # Logprobs: dataset `output`, dataset `unsafe_response`, and sampled output.
                        "logprob_output": [r.get("logprob_output") for r in rows],
                        "logprob_unsafe_response": [r.get("logprob_unsafe_response") for r in rows],
                        "logprob_sampled_output": [r.get("logprob_sampled_output") for r in rows],
                    }
                    if cfg.train.use_wandb:
                        import wandb
                        if wandb.run is not None:
                            epoch = self.state.epoch
                            epoch_tag = f"epoch_{int(epoch)}" if epoch is not None else "epoch_unknown"
                            table = wandb.Table(
                                columns=[
                                    "prompt",
                                    "sampled_output",
                                    "output",
                                    "unsafe_response",
                                    "safe",
                                    "logprob_output",
                                    "logprob_unsafe_response",
                                    "logprob_sampled_output",
                                ]
                            )
                            for r in rows:
                                table.add_data(
                                    r.get("prompt"),
                                    r.get("sampled_output"),
                                    r.get("output"),
                                    r.get("unsafe_response"),
                                    r.get("safe"),
                                    r.get("logprob_output"),
                                    r.get("logprob_unsafe_response"),
                                    r.get("logprob_sampled_output"),
                                )
                            wandb.log({f"{tag}_outputs_{epoch_tag}": table})

                            out_dir = Path(cfg.train.output_dir)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / f"{tag}_outputs_{epoch_tag}.json"
                            out_path.write_text(
                                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                            artifact = wandb.Artifact(
                                f"{tag}_outputs_{epoch_tag}",
                                type=f"{tag}_outputs",
                            )
                            artifact.add_file(str(out_path))
                            wandb.log_artifact(artifact)
                    return result

                # If provided, generate from an explicit dataset path (keeps old behavior).
                gen_data = getattr(cfg.train, "gen_data", None)
                if isinstance(gen_data, str) and not gen_data.strip():
                    gen_data = None

                # Dist info
                dist_on = dist.is_available() and dist.is_initialized()
                rank = dist.get_rank() if dist_on else 0
                world_size = dist.get_world_size() if dist_on else 1

                # Optional: cap number of prompts to generate/log at train end.
                # Disabled by default (None) to preserve existing behavior.
                max_gen = getattr(cfg.train, "max_gen", None)
                if max_gen is not None:
                    max_gen = int(max_gen)
                    if max_gen <= 0:
                        raise ValueError(f"cfg.train.max_gen must be a positive int or None, got: {max_gen}")

                rows_local = []
                model = extract_model_from_parallel(self.model)
                was_training = model.training
                model.eval()
                try:
                    if gen_data is not None:
                        def _select_split(ds_dict):
                            for name in ("eval", "validation", "test", "train"):
                                if name in ds_dict:
                                    return ds_dict[name]
                            return ds_dict[next(iter(ds_dict.keys()))]

                        try:
                            ds_dict = load_dataset(gen_data)
                            gen_ds = _select_split(ds_dict) if hasattr(ds_dict, "keys") else ds_dict
                        except Exception as exc:
                            if rank == 0:
                                print(f"[gen_data] Failed to load dataset '{gen_data}': {exc}")
                            return None

                        if len(gen_ds) == 0:
                            if rank == 0:
                                print(f"[gen_data] Dataset '{gen_data}' is empty.")
                            return None

                        n_total = len(gen_ds)
                        if max_gen is not None:
                            n_total = min(n_total, max_gen)
                        if rank == 0:
                            print(f"[gen_data] (train end) Generating answers for {n_total} prompts from '{gen_data}' across {world_size} ranks...")

                        # Shard indices across ranks (strided to avoid needing a sampler)
                        local_indices = list(range(rank, n_total, world_size))
                        for i in tqdm(local_indices, desc=f"[gen_data] (train end) Generating [rank {rank}]", leave=False):
                            sample = gen_ds[i]
                            prompt_text = sample.get("prompt") if hasattr(sample, "get") else None
                            if not isinstance(prompt_text, str):
                                continue
                            prompt = format_prompt(
                                input=prompt_text,
                                eos_token=self.processing_class.eos_token,
                            )
                            tokenized = self.processing_class(prompt, return_tensors="pt")
                            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                            with torch.no_grad():
                                output_ids = model.generate(
                                    **tokenized,
                                    max_new_tokens=512,
                                    no_repeat_ngram_size=5
                                )
                            generated_ids = output_ids[0][tokenized["input_ids"].shape[1]:]
                            decoded = self.processing_class.decode(generated_ids, skip_special_tokens=True)

                            prompt_ids_1d = tokenized["input_ids"][0]
                            ref_out = sample.get("output") if hasattr(sample, "get") else None
                            unsafe_out = sample.get("unsafe_response") if hasattr(sample, "get") else None
                            lp_out, _ = _avg_logprob_answer(prompt_ids_1d, ref_out)
                            lp_unsafe, _ = _avg_logprob_answer(prompt_ids_1d, unsafe_out)
                            lp_sampled, _ = _avg_logprob_answer(prompt_ids_1d, decoded)

                            rows_local.append(
                                {
                                    "i": int(i),
                                    "prompt": prompt_text,
                                    "output": ref_out,
                                    "unsafe_response": unsafe_out,
                                    "sampled_output": decoded,
                                    "safe": sample.get("safe") if hasattr(sample, "get") else None,
                                    "logprob_output": lp_out,
                                    "logprob_unsafe_response": lp_unsafe,
                                    "logprob_sampled_output": lp_sampled,
                                }
                            )
                    else:
                        # Default: generate from the validation set used in normal eval.
                        # We generate answers for safe=True samples, and ALSO for an equal number
                        # of safe=False samples. The unsafe subset is selected deterministically
                        # by taking the first N (so results are comparable across runs/tasks).
                        ev_ds = self.eval_dataset
                        if not hasattr(ev_ds, "filter"):
                            if rank == 0:
                                print("[gen_data] (train end) eval_dataset does not support filtering; skipping generation.")
                            return None

                        safe_ds = ev_ds.filter(lambda x: _to_bool(x.get("safe", False)) is True)
                        unsafe_ds = ev_ds.filter(lambda x: _to_bool(x.get("safe", False)) is False)

                        n_safe_total = len(safe_ds)
                        if n_safe_total == 0:
                            if rank == 0:
                                print("[gen_data] (train end) No safe=True samples found in validation set.")
                            return None

                        n_safe = n_safe_total
                        if max_gen is not None:
                            n_safe = min(n_safe, max_gen)
                        if hasattr(safe_ds, "select") and n_safe < n_safe_total:
                            safe_ds = safe_ds.select(range(n_safe))

                        n_unsafe_total = len(unsafe_ds)
                        n_unsafe = min(n_unsafe_total, n_safe)
                        if hasattr(unsafe_ds, "select") and n_unsafe < n_unsafe_total:
                            # Deterministic subset: take the first N unsafe samples.
                            unsafe_ds = unsafe_ds.select(range(n_unsafe))

                        if rank == 0:
                            print(
                                f"[gen_data] (train end) Generating answers for {n_safe} safe=True and "
                                f"{n_unsafe} safe=False validation prompts across {world_size} ranks..."
                            )
                            if n_unsafe_total == 0:
                                print("[gen_data] (train end) No safe=False samples found in validation set; skipping unsafe generation.")
                            elif n_unsafe < n_safe:
                                print(
                                    f"[gen_data] (train end) Only found {n_unsafe_total} safe=False samples; "
                                    f"generating for {n_unsafe} (requested {n_safe})."
                                )

                        def _as_tensor(x):
                            if isinstance(x, torch.Tensor):
                                return x
                            return torch.tensor(x)

                        def _generate_rows(ds, n_total, expected_safe: bool, i_offset: int, desc: str):
                            local_indices = list(range(rank, n_total, world_size))
                            for i in tqdm(local_indices, desc=desc, leave=False):
                                sample = ds[i]
                                if _to_bool(sample.get("safe", expected_safe)) != expected_safe:
                                    continue

                                input_ids = _as_tensor(sample["input_ids"]).to(model.device)
                                response_mask = _as_tensor(sample["response_mask"]).bool()
                                # Prompt ends where response_mask turns True for the first time.
                                true_pos = torch.where(response_mask)[0]
                                if true_pos.numel() == 0:
                                    # Fallback: can't find boundary; skip.
                                    continue
                                prompt_end = int(true_pos[0].item())
                                if prompt_end <= 0:
                                    continue

                                prompt_ids = input_ids[:prompt_end].unsqueeze(0)
                                attn = torch.ones_like(prompt_ids, device=model.device)
                                prompt_text = self.processing_class.decode(prompt_ids[0], skip_special_tokens=True)

                                with torch.no_grad():
                                    output_ids = model.generate(
                                        input_ids=prompt_ids,
                                        attention_mask=attn,
                                        max_new_tokens=512,
                                    )
                                generated_ids = output_ids[0][prompt_ids.shape[1]:]
                                decoded = self.processing_class.decode(generated_ids, skip_special_tokens=True)

                                # Reference answers are expected to be present on the (preprocessed) dataset.
                                ref_out = sample.get("output")
                                unsafe_out = sample.get("unsafe_response")
                                prompt_ids_1d = prompt_ids[0]
                                lp_out, _ = _avg_logprob_answer(prompt_ids_1d, ref_out)
                                lp_unsafe, _ = _avg_logprob_answer(prompt_ids_1d, unsafe_out)
                                lp_sampled, _ = _avg_logprob_answer(prompt_ids_1d, decoded)

                                rows_local.append(
                                    {
                                        "i": int(i_offset + i),
                                        "prompt": prompt_text,
                                        "output": ref_out,
                                        "unsafe_response": unsafe_out,
                                        "sampled_output": decoded,
                                        "safe": expected_safe,
                                        "logprob_output": lp_out,
                                        "logprob_unsafe_response": lp_unsafe,
                                        "logprob_sampled_output": lp_sampled,
                                    }
                                )

                        _generate_rows(
                            safe_ds,
                            n_safe,
                            True,
                            0,
                            f"[gen_data] (train end) Generating (val safe=True) [rank {rank}]",
                        )
                        if n_unsafe > 0:
                            _generate_rows(
                                unsafe_ds,
                                n_unsafe,
                                False,
                                n_safe,
                                f"[gen_data] (train end) Generating (val safe=False) [rank {rank}]",
                            )
                finally:
                    if was_training:
                        model.train()

                # Gather results to rank 0 for a single log/artifact.
                if dist_on and world_size > 1:
                    gathered = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered, rows_local)
                    if rank == 0:
                        rows_all = []
                        for part in gathered:
                            if part:
                                rows_all.extend(part)
                        # Stable-ish ordering
                        rows_all.sort(key=lambda r: r.get("i", 0))
                        for r in rows_all:
                            r.pop("i", None)
                        print("[gen_data] (train end) Generation complete.")
                        return _log_outputs("safe_generate_train_end", rows_all)
                    return None
                else:
                    # Single process
                    rows_local.sort(key=lambda r: r.get("i", 0))
                    for r in rows_local:
                        r.pop("i", None)
                    if rank == 0:
                        print("[gen_data] (train end) Generation complete.")
                        return _log_outputs("safe_generate_train_end", rows_local)
                    return None

            def _compute_metrics(self, pred):

                logits_chunks = pred.predictions
                idx_chunks = pred.label_ids
                # ---- flatten indexes to 1D ----
                if isinstance(idx_chunks, (list, tuple)):
                    indexes = np.concatenate([np.asarray(x) for x in idx_chunks], axis=0)
                else:
                    indexes = np.asarray(idx_chunks)

                # ---- flatten logits to (N, L, V) with single L ----
                if isinstance(logits_chunks, (list, tuple)):
                    # global max length across batches (or use cfg.train.max_length)
                    max_L = max(x.shape[1] for x in logits_chunks)

                    padded = []
                    for x in logits_chunks:
                        x = np.asarray(x)  # (B, Lb, V)
                        Lb = x.shape[1]
                        if Lb < max_L:
                            pad_width = ((0, 0), (0, max_L - Lb))
                            x = np.pad(x, pad_width, mode="constant", constant_values=0.0)
                        padded.append(x)

                    logits = np.concatenate(padded, axis=0)  # (N, max_L, V)
                else:
                    logits = np.asarray(logits_chunks)
                    
                answer_log_probs = torch.tensor(logits); indexes = torch.tensor(indexes); 
                cfg = self.custom_cfg.exp

                # Get safety labels, response_mask, input_ids from complete dataset and index.
                # `collated['safe']` == safety_label; True => UNSAFE prompt, False => SAFE prompt.
                samples = [self.complete_ds[int(idx)] for idx in indexes.tolist()]
                collated = self.data_collator(samples)
                input_ids = collated["input_ids"]
                response_mask = collated['response_mask'].to(answer_log_probs.device)
                is_unsafe_prompt = collated['safe'].to(answer_log_probs.device)  # size
                # precomputed_answer_log_probs = collated['baseline_logprob'].to(answer_log_probs.device)

                is_safe_prompt = ~is_unsafe_prompt

                answer_log_probs = answer_log_probs * response_mask[:, 1:]
                answer_log_probs = answer_log_probs.sum(dim=-1)  # size = (B,)
                denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                answer_log_probs = answer_log_probs / denom
                
                # answer_log_ratios = answer_log_probs - precomputed_answer_log_probs
                
                # Objective tracked on SAFE prompts (keep producing the benign answer).
                answer_log_ratios_objective = answer_log_probs[is_safe_prompt]
                tol2 = getattr(cfg, "tol2", None)
                if tol2 is None:
                    # -----------------------------
                    # Existing behavior (when tol2 is NOT provided):
                    # only compute/log constraint slacks for UNSAFE prompts (safety_label=True).
                    # -----------------------------
                    self._last_constraint_indexes = indexes[is_unsafe_prompt].detach().cpu()
                    self._last_objective_indexes = indexes[is_safe_prompt].detach().cpu()

                    answer_log_ratios_constraint = -1 * answer_log_probs[is_unsafe_prompt]
                    if is_unsafe_prompt.sum() == 0:
                        answer_log_ratios_constraint = torch.tensor([0.0], device=answer_log_probs.device)
                        self._last_constraint_indexes = torch.tensor([0], dtype=torch.long)
                        self._last_constraint_safety_labels = torch.tensor([True], dtype=torch.bool)
                    else:
                        # all True, but keep shape aligned with slacks/indexes
                        self._last_constraint_safety_labels = is_unsafe_prompt[is_unsafe_prompt].detach().cpu()

                    slacks = answer_log_ratios_constraint
                    self._last_constraint_slacks = slacks.detach().cpu()
                    self._last_objective_ratios = answer_log_ratios_objective.detach().cpu()

                    epoch = self.state.epoch
                    constraint_slacks = slacks.flatten()

                    import wandb
                    if wandb.run is not None:
                        table = wandb.Table(columns=["constraint_slack", "safety_label"])
                        for slack in constraint_slacks.detach().cpu().tolist():
                            table.add_data(slack, True)
                        wandb.log({f"constraint_slacks_epoch_{epoch}_{self._current_eval_prefix}": table})

                    objective = -1 * answer_log_ratios_objective.mean().item()
                    constraint_mean = slacks.mean().item()
                    constrain_min = slacks.min().item()
                    constrain_max = slacks.max().item()
                    contriant_cvar = slacks[
                        slacks > np.quantile(slacks.detach().cpu().numpy(), 0.9)
                    ].mean().item()

                    return {
                        "objective": objective,
                        "constraint_mean": constraint_mean,
                        "constraint_min": constrain_min,
                        "constraint_max": constrain_max,
                        "constraint_cvar": contriant_cvar,
                    }

                # -----------------------------
                # tol2 is provided => compute BOTH constraints in eval:
                #  - UNSAFE prompts (safety_label=True): slack_refusal = tol - logp(output)
                #  - SAFE prompts   (safety_label=False): slack_no_refusal = logp(refusal) - tol2
                # -----------------------------
                slack_refusal = (cfg.tol - answer_log_probs) * is_unsafe_prompt.float()

                safe_output_logp = torch.zeros_like(answer_log_probs)
                slack_no_refusal = torch.zeros_like(answer_log_probs)
                if bool(is_safe_prompt.any()):
                    proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                    if proc is None:
                        raise RuntimeError("Could not find tokenizer/processing_class on Trainer to score safe_output.")

                    pad_id = getattr(proc, "pad_token_id", None)
                    eos_id = getattr(proc, "eos_token_id", None)
                    if eos_id is None:
                        raise RuntimeError("Tokenizer/processing_class must define eos_token_id to score safe_output.")
                    if pad_id is None:
                        pad_id = eos_id

                    if self._safe_output_token_ids_cpu is None:
                        self._safe_output_token_ids_cpu = (
                            proc(
                                self.safe_output_text,
                                add_special_tokens=False,
                                return_tensors="pt",
                            )["input_ids"][0]
                            .to(torch.long)
                            .cpu()
                        )
                    safe_out_ids_cpu = self._safe_output_token_ids_cpu  # (T,)

                    prompt_ends = response_mask.long().argmax(dim=-1)  # (N,)
                    # Indices of SAFE prompts (where we score the refusal to check for over-refusal).
                    safe_batch_idx = (
                        torch.nonzero(is_safe_prompt, as_tuple=False).squeeze(-1).detach().cpu().tolist()
                    )

                    model = extract_model_from_parallel(self.model)
                    was_training = model.training
                    model.eval()
                    try:
                        model_device = next(model.parameters()).device
                        bs = int(getattr(self.args, "per_device_eval_batch_size", 8) or 8)
                        bs = max(1, min(32, bs))

                        batch_ids = []
                        batch_masks = []
                        batch_pos = []

                        def _flush():
                            if not batch_pos:
                                return
                            safe_input_ids = torch.stack(batch_ids, dim=0).to(model_device)  # (U, L)
                            safe_resp_mask = torch.stack(batch_masks, dim=0).to(model_device)  # (U, L)
                            safe_attn = safe_input_ids.ne(pad_id)
                            with torch.no_grad():
                                out = model(input_ids=safe_input_ids, attention_mask=safe_attn)
                                logits = out.logits[:, :-1]
                                lp = F.log_softmax(logits, dim=-1)
                                tok_lp = torch.gather(
                                    lp, dim=-1, index=safe_input_ids[:, 1:].unsqueeze(-1)
                                ).squeeze(-1)
                                tok_lp = tok_lp * safe_resp_mask[:, 1:]
                                sum_lp = tok_lp.sum(dim=-1)
                                denom_lp = safe_resp_mask[:, 1:].sum(dim=-1).clamp_min(1)
                                avg_lp = (sum_lp / denom_lp).float().detach().cpu()

                            pos_t = torch.tensor(batch_pos, dtype=torch.long)
                            safe_output_logp[pos_t] = avg_lp
                            batch_ids.clear()
                            batch_masks.clear()
                            batch_pos.clear()

                        max_len = int(input_ids.shape[1])
                        for bi in safe_batch_idx:
                            prompt_end = int(prompt_ends[bi].item())
                            prompt_ids = input_ids[bi, :prompt_end].to(torch.long).cpu()

                            full_ids = torch.cat([prompt_ids, safe_out_ids_cpu], dim=0)
                            if full_ids.numel() == 0 or full_ids[-1].item() != eos_id:
                                full_ids = torch.cat([full_ids, full_ids.new_tensor([eos_id])], dim=0)

                            prompt_len = int(prompt_ids.numel())
                            prompt_len_eff = prompt_len
                            if full_ids.numel() > max_len:
                                overflow = int(full_ids.numel() - max_len)
                                full_ids = full_ids[overflow:]
                                prompt_len_eff = max(0, prompt_len - overflow)

                            seq_len = int(full_ids.numel())
                            if seq_len < max_len:
                                pad_len = max_len - seq_len
                                full_ids = torch.cat([full_ids, full_ids.new_full((pad_len,), pad_id)], dim=0)
                            else:
                                full_ids = full_ids[:max_len]
                                seq_len = max_len

                            resp_m = torch.zeros(max_len, dtype=torch.bool)
                            start = min(int(prompt_len_eff), int(seq_len))
                            resp_m[start:seq_len] = True

                            batch_ids.append(full_ids)
                            batch_masks.append(resp_m)
                            batch_pos.append(bi)
                            if len(batch_pos) >= bs:
                                _flush()
                        _flush()
                    finally:
                        if was_training:
                            model.train()

                    slack_no_refusal = (safe_output_logp - float(tol2)) * is_safe_prompt.float()

                slacks = slack_refusal + slack_no_refusal

                # Keep for wandb callback logging
                self._last_constraint_slacks = slacks.detach().cpu()
                self._last_constraint_indexes = indexes.detach().cpu()
                self._last_constraint_safety_labels = is_unsafe_prompt.detach().cpu()
                self._last_objective_ratios = answer_log_ratios_objective.detach().cpu()
                self._last_objective_indexes = indexes[is_safe_prompt].detach().cpu()

                epoch = self.state.epoch
                # Flatten slacks and output table to wandb
                constraint_slacks = slacks.flatten()

                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["constraint_slack", "safety_label"])
                    safety_labels = is_unsafe_prompt.detach().cpu().flatten().tolist()
                    for slack, lbl in zip(constraint_slacks.detach().cpu().tolist(), safety_labels):
                        table.add_data(slack, bool(lbl))
                    wandb.log({f"constraint_slacks_epoch_{epoch}_{self._current_eval_prefix}": table})
            
                # Metrics
                if answer_log_ratios_objective.numel() > 0:
                    objective = -1 * answer_log_ratios_objective.mean().item()
                else:
                    objective = 0.0

                # Combined constraint stats (refusal-on-unsafe + no-refusal-on-safe)
                if slacks.numel() > 0:
                    constraint_mean = slacks.mean().item()
                    constrain_min = slacks.min().item()
                    constrain_max = slacks.max().item()
                    sl_np = slacks.detach().cpu().numpy()
                    q = float(np.quantile(sl_np, 0.9)) if sl_np.size > 0 else 0.0
                    tail = slacks[slacks > q]
                    contriant_cvar = tail.mean().item() if tail.numel() > 0 else 0.0
                else:
                    constraint_mean = 0.0
                    constrain_min = 0.0
                    constrain_max = 0.0
                    contriant_cvar = 0.0

                # Per-group means for debugging.
                # `refusal_slack_mean`    : mean slack of the refusal constraint on UNSAFE prompts.
                # `no_refusal_slack_mean` : mean slack of the don't-over-refuse constraint on SAFE prompts.
                refusal_slack_mean = (
                    slack_refusal[is_unsafe_prompt].mean().item() if bool(is_unsafe_prompt.any()) else 0.0
                )
                no_refusal_slack_mean = (
                    slack_no_refusal[is_safe_prompt].mean().item() if bool(is_safe_prompt.any()) else 0.0
                )

                return {
                    "objective": objective,
                    "constraint_mean": constraint_mean,
                    "constraint_min": constrain_min,
                    "constraint_max": constrain_max,
                    "constraint_cvar": contriant_cvar,
                    # NOTE: legacy wandb keys kept for dashboard backward-compatibility.
                    # `constraint_safe_mean`   = refusal-constraint slack mean (on UNSAFE prompts).
                    # `constraint_unsafe_mean` = no-refusal-constraint slack mean (on SAFE prompts).
                    "constraint_safe_mean": refusal_slack_mean,
                    "constraint_unsafe_mean": no_refusal_slack_mean,
                }
        return CustomTrainer
    
