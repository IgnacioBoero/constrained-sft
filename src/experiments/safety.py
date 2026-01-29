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


class SAFETY(Experiment):
    
    def load_model_and_tok(self, cfg):
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name, use_fast=True)
        tok.model_max_length = cfg.train.max_length
        tok.pad_token = tok.eos_token  # Ensure pad token is defined
        model = AutoModelForCausalLM.from_pretrained(cfg.exp.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
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
                            "gate_proj",
                            "down_proj",
                            "up_proj",
                            "lm_head",
                        ],
                    ),
                )
        return model, tok

    def load_datasets(self, cfg):
        ds = load_dataset("iboero16/SAFE-ALPACA-3")

        tr_raw = ds["train"]
        ev_raw = ds["validation"]
        tr_raw = tr_raw.add_column("split", ["train"] * len(tr_raw))
        ev_raw = ev_raw.add_column("split", ["validation"] * len(ev_raw))
        
        
        tr_size = int(cfg.train.data_proportion * len(tr_raw))
        tr_raw = tr_raw.select(range(tr_size))
        
        ev_size = int(cfg.train.data_proportion * len(ev_raw))
        ev_raw = ev_raw.select(range(ev_size))

        # Add one common index for both
        complete_dl = concatenate_datasets([tr_raw, ev_raw])
        complete_dl = complete_dl.add_column("index", list(range(len(complete_dl))))
        
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

        def fn(sample):
            input_text = (
                " ".join((sample["instruction"], sample["input"]))
                if sample["input"] else sample["instruction"]
            )
            if not isinstance(input_text, str):
                raise ValueError(
                    f"Unsupported type of `input`: {type(input_text)}. Expected: str."
                )

            answer = sample["output"]
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
            
            return {
                "input_ids": input_ids,
                "safe": sample["safety_label"],
                "response_mask": response_mask,
                "index": index,
                "labels": index,  # unused, keep your trick
            }
        return fn
    def get_collator(self, tok):
        class SafetyCollator():
            
            def __init__(self, pad_token_id: int) -> None:
                """Initialize a collator."""
                self.pad_token_id = pad_token_id
                
            def __call__(self, samples):
                
                input_ids = right_padding(
                            [torch.tensor(sample['input_ids']) for sample in samples],
                            padding_value=self.pad_token_id,
                        )

                response_masks = right_padding(
                    [torch.tensor(sample['response_mask']) for sample in samples],
                    padding_value=0,
                )

                attention_mask = input_ids.ne(self.pad_token_id)
                safe = torch.tensor(
                    [sample['safe'] for sample in samples],
                    dtype=torch.bool,
                )   
                index = torch.tensor(
                    [sample['index'] for sample in samples],
                    dtype=torch.long,
                )
                labels = torch.tensor(
                    [sample['labels'] for sample in samples],
                    dtype=torch.long,
                )

                batch = {
                    "input_ids": input_ids,              # (B, L)
                    "attention_mask": attention_mask.bool(),
                    "index": index,                      # (B,)
                    "safe": safe,                        # (B,)
                    "response_mask": response_masks.bool(),
                    "labels": labels,
                }

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

                
            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros(len(self.train_dataset) + len(self.eval_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)
                self.avg_dual = torch.tensor(0.0, device=self.model.device, dtype=torch.float)

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
                
                input_ids = inputs['input_ids']  # size = (B, 2, L)
                attention_mask = inputs['attention_mask']  # size = (B, 2, L)
                response_mask = inputs['response_mask']  # size = (B, 2, L)
                
                # precomputed_answer_log_probs = inputs['baseline_logprob']  # size = (B,)
                is_constraint = inputs['safe']  # size = (B,)                
                is_not_constraint = ~is_constraint
                index = inputs['index']  # size = (B,)
                index_constraints = index[is_constraint]
                dual_var = self.dual_vars[index_constraints].clone()  # Get dual variable for the current batch
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids.clone().masked_fill(response_mask == 0, -100),
                )
                # print if model is on train or eval
                # print if inputs_ids requies grad
                logits = outputs.logits[:, :-1]  # size = (B, L-1, Vocab)
                log_probs = F.log_softmax(logits, dim=-1)  # size = (B, L-1, Vocab)
                answer_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # size = (B, L-1)
                answer_log_probs = answer_log_probs * response_mask[:, 1:]
                answer_log_probs = answer_log_probs.sum(dim=-1)  # size = (B,)
                denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                answer_log_probs = answer_log_probs / denom
                

                # answer_log_ratios = answer_log_probs - precomputed_answer_log_probs
                

                
                loss = -1 * answer_log_probs * is_not_constraint.float()
                slack = (cfg.tol - answer_log_probs) * is_constraint.float()

                if cfg.loss_type == "erm":
                    loss = -1 * answer_log_probs
                elif cfg.loss_type == "avg":
                        dual_avg = self.avg_dual.clone()
                        dual_avg = torch.clamp(dual_avg + cfg.dual_step_size * slack.mean(), min=0.0)
                        self.avg_dual = dual_avg.detach()
                        loss += (
                            dual_avg.detach()
                            * slack
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
                
                # print(loss)
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
                    gen_ds = load_dataset("iboero16/safe-generate-eval")["eval"]
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
                        eos_token=self.tokenizer.eos_token,
                    )
                    tokenized = self.tokenizer(prompt, return_tensors="pt")
                    tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **tokenized,
                            max_new_tokens=64,
                        )
                    generated_ids = output_ids[0][tokenized["input_ids"].shape[1]:]
                    decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    rows.append(
                        {
                            "prompt": prompt_text,
                            "output": decoded,
                            "safe": sample.get("safe"),
                        }
                    )

                print("[safe-generate-eval] Generation complete.")

                return _log_outputs("safe_generate_eval", rows)

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
                
                self._generate_eval_answers()

                # Get is_constraint, response_mask, input_ids from complete dataset and index
                samples = [self.complete_ds[int(idx)] for idx in indexes.tolist()]
                collated = self.data_collator(samples)
                response_mask = collated['response_mask'].to(answer_log_probs.device)
                is_constraint = collated['safe'].to(answer_log_probs.device)  # size
                # precomputed_answer_log_probs = collated['baseline_logprob'].to(answer_log_probs.device)

                is_not_constraint = ~is_constraint
                self._last_constraint_indexes = indexes[is_constraint].detach().cpu()
                self._last_objective_indexes = indexes[is_not_constraint].detach().cpu()
                
                answer_log_probs = answer_log_probs * response_mask[:, 1:]
                answer_log_probs = answer_log_probs.sum(dim=-1)  # size = (B,)
                denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                answer_log_probs = answer_log_probs / denom
                
                # answer_log_ratios = answer_log_probs - precomputed_answer_log_probs
                
                
                answer_log_ratios_objective = answer_log_probs[is_not_constraint]
                answer_log_ratios_constraint = -1 * answer_log_probs[is_constraint]
                
                if is_constraint.sum() == 0:
                    answer_log_ratios_constraint = torch.tensor([0.0], device=answer_log_probs.device)
                    self._last_constraint_indexes = torch.tensor([0], dtype=torch.long)
                    
                
                objective = -1 * answer_log_ratios_objective.mean().item()
                constraint_mean = answer_log_ratios_constraint.mean().item()
                constrain_min = answer_log_ratios_constraint.min().item()
                constrain_max = answer_log_ratios_constraint.max().item()
                contriant_cvar = answer_log_ratios_constraint[answer_log_ratios_constraint > np.quantile(answer_log_ratios_constraint.cpu().numpy(), 0.9)].mean().item()
                
                slacks = answer_log_ratios_constraint
                
                self._last_constraint_slacks = slacks.detach().cpu()
                self._last_objective_ratios = answer_log_ratios_objective.detach().cpu()

                epoch = self.state.epoch
                # Flatten slacks and output table to wandb
                constraint_slacks = slacks.flatten()

                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["constraint_slack"])
                    for slack in constraint_slacks.tolist():
                        table.add_data(slack)
                    wandb.log({f"constraint_slacks_epoch_{epoch}_{self._current_eval_prefix}": table})
            
                return {
                    "objective": objective,
                    "constraint_mean": constraint_mean,
                    "constraint_min": constrain_min,
                    "constraint_max": constrain_max,
                    "constraint_cvar": contriant_cvar,
                        }
        return CustomTrainer
    
