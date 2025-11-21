# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
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
        ds = load_dataset("iboero16/SAFE-ALPACA-2")

        tr_raw = ds["train"]
        ev_raw = ds["validation"]
        tr_raw = tr_raw.add_column("split", ["train"] * len(tr_raw))
        ev_raw = ev_raw.add_column("split", ["validation"] * len(ev_raw))

        complete_dl = concatenate_datasets([tr_raw, ev_raw])
        complete_dl = complete_dl.add_column("index", list(range(len(complete_dl))))

        # Optional subsample while keeping global indices consistent
        complete_dl = complete_dl.shuffle(seed=cfg.train.seed)
        complete_size = int(cfg.train.data_proportion * len(complete_dl))
        complete_dl = complete_dl.select(range(complete_size))

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
        def fn(sample):
            input_text = ' '.join((sample['instruction'], sample['input'])) if sample['input'] else sample['instruction']
            if not isinstance(input_text, str):
                raise ValueError(f'Unsupported type of `input`: {type(input_text)}. Expected: str.')
            answer = sample["output"]
            prompt = format_prompt(input=input_text, eos_token=tok.eos_token)

            text = prompt + answer
            
            prompt_ids = tok(prompt, return_tensors='pt', truncation=True, max_length=max_length-1)['input_ids'][0]
            len_prompt_ids = len(prompt_ids)
        
            input_ids = tok(text,  return_tensors='pt', truncation=True, max_length=max_length-1)['input_ids'][0]
            if input_ids[-1].item() != tok.eos_token_id:
                input_ids = torch.cat(
                    [input_ids, torch.tensor([tok.eos_token_id], dtype=input_ids.dtype)]
                )   
            index = sample['index']  # Get the index for dual variable updates
  
            response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            response_mask[len_prompt_ids:] = True
    

            return {
                'input_ids': input_ids,
                'safe': sample["safe"],
                'response_mask': response_mask,
                'index': index,
                'labels': index, # AS LABELS ARE UNUSED, STORE INDEX HERE FOR METRICS COMPUTATION
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
                if "baseline_logprob" in samples[0]:
                    baseline_logprob = torch.tensor(
                        [sample["baseline_logprob"] for sample in samples],
                        dtype=torch.float,
                    )
                    batch["baseline_logprob"] = baseline_logprob  # (B,)

                return batch
        return SafetyCollator(tok.pad_token_id)

    def compute_metrics(self, tok, cfg):
        return None


    def get_trainer_class(self):
        class CustomTrainer(Trainer):
            
            def __init__(self, *args, custom_cfg=None,complete_dataset=None, experiment=None, dual_vars=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                self.custom_cfg = custom_cfg
                self.complete_ds = complete_dataset
                self.init_dual_vars()
                self.precompute_answer_logprobs()
                self.compute_metrics = self._compute_metrics ## OVERRIDE COMPUTE METRICS SO I CAN USE SELF. 

                
            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros(len(self.train_dataset) + len(self.eval_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)

            def precompute_answer_logprobs(self):
                """
                Precompute average answer log-probs for *all* samples in complete_ds
                in a multi-GPU safe way.
                """
                model = self.model
                model.eval()

                device = model.device
                dtype = model.dtype
                n_total = len(self.complete_ds)

                # local tensor: full size, but this rank only writes its own indices
                all_probs_local = torch.zeros(n_total, dtype=dtype, device=device)

                # # Decide sampler: DistributedSampler if DDP, else SequentialSampler
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    sampler = DistributedSampler(
                        self.complete_ds,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=False,   # for deterministic precompute
                        drop_last=False,
                    )
                else:
                    sampler = SequentialSampler(self.complete_ds)

                # Build dataloader directly
                dataloader = DataLoader(
                    self.complete_ds,
                    sampler=sampler,
                    batch_size=self.args.per_device_train_batch_size,
                    collate_fn=self.data_collator,
                    pin_memory=self.args.dataloader_pin_memory,
                )

                desc = "Precompute answer logprobs"
                if dist.is_available() and dist.is_initialized():
                    desc += f" [rank {dist.get_rank()}]"

                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=desc, unit="batch"):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        response_mask = batch["response_mask"].to(device)
                        index = batch["index"].to(device)   # global indices in [0, n_total)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        logits = outputs.logits[:, :-1]                  # (B, L-1, V)
                        log_probs = F.log_softmax(logits, dim=-1)        # (B, L-1, V)
                        answer_log_probs = torch.gather(
                            log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
                        ).squeeze(-1)                                    # (B, L-1)
                        answer_log_probs = answer_log_probs * response_mask[:, 1:]
                        answer_log_probs = answer_log_probs.sum(dim=-1)  # (B,)
                        denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                        answer_log_probs = answer_log_probs / denom      # average log-prob

                        all_probs_local[index] = answer_log_probs

                # Now combine results across ranks
                if dist.is_available() and dist.is_initialized():
                    # each rank has non-zero entries only for its shard;
                    # sum across ranks to get full vector on all ranks
                    dist.all_reduce(all_probs_local, op=dist.ReduceOp.SUM)

                all_probs_cpu = all_probs_local.detach().cpu()
                del all_probs_local
                torch.cuda.empty_cache()
                
                def add_baseline(example):
                    idx = example["index"]             # global index
                    example["baseline_logprob"] = float(all_probs_cpu[idx])
                    return example
                self.complete_ds = self.complete_ds.map(add_baseline)
                self.train_dataset = self.train_dataset.map(add_baseline)
                self.eval_dataset  = self.eval_dataset.map(add_baseline)
                model.train()

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
                precomputed_answer_log_probs = inputs['baseline_logprob']  # size = (B,)
                is_constraint = inputs['safe']  # size = (B,)                
                is_not_constraint = ~is_constraint
                index = inputs['index']  # size = (B,)
                index_constraints = index[is_constraint]
                dual_var = self.dual_vars[index_constraints].clone()  # Get dual variable for the current batch
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, :-1]  # size = (B, L-1, Vocab)
                log_probs = F.log_softmax(logits, dim=-1)  # size = (B, L-1, Vocab)
                answer_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # size = (B, L-1)
                answer_log_probs = answer_log_probs * response_mask[:, 1:]
                answer_log_probs = answer_log_probs.sum(dim=-1)  # size = (B,)
                denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                avg_answer_log_probs = answer_log_probs / denom
                
                avg_answer_log_ratios = avg_answer_log_probs - precomputed_answer_log_probs
                
                avg_answer_log_ratios_objective = avg_answer_log_ratios[is_not_constraint]
                avg_answer_log_ratios_constraint = avg_answer_log_ratios[is_constraint]
                
                loss = -1 * avg_answer_log_ratios_objective.mean()
                slack = cfg.tol - avg_answer_log_ratios_constraint

                if cfg.loss_type == "erm":
                    pass
                elif cfg.loss_type == "penalty":
                    loss += cfg.loss_alpha * slack.sum()
                elif cfg.loss_type == "l2":
                    loss += (
                        cfg.loss_alpha / 2
                        * torch.clamp(slack, min=0.0) ** 2
                    ).sum()
                elif cfg.loss_type == "dual":                   
                    dual_var = torch.clamp(dual_var + 2 * cfg.dual_lr * slack, min=0.0)
                    self.dual_vars[index_constraints] = dual_var.detach()  
                    loss += (
                        dual_var.detach()
                        * slack
                    ).sum()

                loss = loss.mean()
                
                if return_outputs:
                    return loss, outputs
                else:
                    return loss
                
                
            def _compute_metrics(self, pred):
                
        
                logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
                indexes = pred.label_ids
                logits = torch.tensor(logits); indexes = torch.tensor(indexes); 
                
                cfg = self.custom_cfg.exp
                
                # Get is_constraint, response_mask, input_ids from complete dataset and index
                samples = [self.complete_ds[int(idx)] for idx in indexes.tolist()]
                collated = self.data_collator(samples)
                input_ids = collated['input_ids'].to(logits.device)
                response_mask = collated['response_mask'].to(logits.device)
                is_constraint = collated['safe'].to(logits.device)  # size
                precomputed_answer_log_probs = collated['baseline_logprob'].to(logits.device)
                is_not_constraint = ~is_constraint
                self._last_constraint_indexes = indexes[is_constraint].detach().cpu()
                self._last_objective_indexes = indexes[is_not_constraint].detach().cpu()
                
                log_probs = F.log_softmax(logits, dim=-1)[:, :-1]  # size = (B, L-1, Vocab)
                answer_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # size = (B, L-1)
                answer_log_probs = answer_log_probs * response_mask[:, 1:]
                answer_log_probs = answer_log_probs.sum(dim=-1)  # size = (B,)
                denom = response_mask[:, 1:].sum(dim=-1).clamp_min(1)
                avg_answer_log_probs = answer_log_probs / denom

                avg_answer_log_ratios = avg_answer_log_probs - precomputed_answer_log_probs
                
                avg_answer_log_ratios_objective = avg_answer_log_ratios[is_not_constraint]
                avg_answer_log_ratios_constraint = avg_answer_log_ratios[is_constraint]
                
                if is_constraint.sum() == 0:
                    avg_answer_log_ratios_constraint = torch.tensor([0.0], device=avg_answer_log_ratios.device)
                    self._last_constraint_indexes = torch.tensor([0], dtype=torch.long)
                    
                
                objective = -1 * avg_answer_log_ratios_objective.mean().item()
                constraint_mean = avg_answer_log_ratios_constraint.mean().item()
                constrain_min = avg_answer_log_ratios_constraint.min().item()
                constrain_max = avg_answer_log_ratios_constraint.max().item()
                contriant_cvar = avg_answer_log_ratios_constraint[avg_answer_log_ratios_constraint > np.quantile(avg_answer_log_ratios_constraint.cpu().numpy(), 0.9)].mean().item()
                
                slacks = cfg.tol - avg_answer_log_ratios_constraint
                
                self._last_constraint_slacks = slacks.detach().cpu()
                self._last_objective_ratios = avg_answer_log_ratios_objective.detach().cpu()
                
                return {
                    "objective": objective,
                    "constraint_mean": constraint_mean,
                    "constraint_min": constrain_min,
                    "constraint_max": constrain_max,
                    "constraint_cvar": contriant_cvar,
                        }
        return CustomTrainer
    
