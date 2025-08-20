# src/your_proj/experiments/reranker_cmnrl.py
import math
from collections import defaultdict
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from .base import Experiment  # your existing base class
from transformers import Trainer
from accelerate.utils import extract_model_from_parallel


class RERANKER(Experiment):
    """
    Cross-Encoder reranker with a Cached-MNRL-style loss:
      - Data: MS MARCO v1.1 train split -> (query, 1 pos, K negs) per sample
      - Model: HF AutoModelForSequenceClassification, num_labels=1 (score)
      - Loss: -log sigmoid(scale * (s_pos - s_neg)) averaged over negatives
      - Metrics: nDCG@10 / MRR@10 / MAP on the held-out set created from train
    """

    # ---------- Model & Tokenizer ----------
    def load_model_and_tok(self, cfg, load_checkpoint=False, checkpoint_path=None):
        if load_checkpoint:
            # Load from a checkpoint if specified
            name = checkpoint_path
        else:
            name = cfg.exp.model_name
        configuration = AutoConfig.from_pretrained(name)
        # Set a 1-dim regression head (score)
        configuration.num_labels = 1
        configuration.problem_type = "regression"
        
        model = AutoModelForSequenceClassification.from_pretrained(name, config=configuration)
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            # Prefer SEP if available, otherwise EOS if present
            tok.pad_token = getattr(tok, "sep_token", None) or getattr(tok, "eos_token", None)

        # Pass training constants to the model config for convenience
        model.config.num_negatives = int(cfg.exp.num_negatives)
        model.config.scale = float(cfg.exp.scale)
        model.config.loss_type = cfg.exp.loss_type
        model.config.loss_tol = float(cfg.exp.loss_tol)

        return model, tok

    # ---------- Dataset ----------
    def load_datasets(self, cfg):
        """
        Replicates the ST mapping:
          - Keep queries that have >= num_negatives negatives and exactly one positive
          - Produce columns: query, positive, negative_1..K
          - Split: train/test via train_test_split(test_size=10_000)
        """
        num_neg = int(cfg.exp.num_negatives)
        assert num_neg >= 1

        train = load_dataset("microsoft/ms_marco", "v1.1", split="train")
        eval = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

        def mnrl_mapper(batch):
            out = defaultdict(list)
            for query, passages_info in zip(batch["query"], batch["passages"]):
                is_sel = passages_info["is_selected"]
                if (1 not in is_sel) or (sum(1 for b in is_sel if b == 0) < num_neg):
                    continue
                pos_idx = is_sel.index(1)
                neg_idxes = [i for i, b in enumerate(is_sel) if b == 0][:num_neg]

                out["query"].append(query)
                out["positive"].append(passages_info["passage_text"][pos_idx])
                for j, ni in enumerate(neg_idxes):
                    out[f"negative_{j+1}"].append(passages_info["passage_text"][ni])
            return out

        train = train.map(mnrl_mapper, batched=True, remove_columns=train.column_names)
        eval = eval.map(mnrl_mapper, batched=True, remove_columns=eval.column_names)
        tr_size = int(cfg.train.data_proportion * len(train))
        ev_size = int(cfg.train.data_proportion * len(eval))
        train = train.select(range(tr_size))
        eval = eval.select(range(ev_size))
        # Add a simple query id for grouping if you want to log it
        train = train.add_column("index", list(range(len(train))))
        eval = eval.add_column("index", list(range(len(eval))))

        print(f"Training samples: {len(train)}, Eval samples: {len(eval)}")
        return train, eval

    # ---------- Preprocessing ----------
    def preprocessing_fn(self, tok, cfg):
        num_neg = int(cfg.exp.num_negatives)
        max_len = int(cfg.exp.max_length)

        def fn(sample):
            q = sample["query"]
            pos = sample["positive"]
            negs = [sample[f"negative_{i+1}"] for i in range(num_neg)]
            left = [q] * (1 + num_neg)
            right = [pos] + negs

            enc = tok(
                left,
                right,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            # Labels: 1 for positive (index 0), 0 for negatives
            labels = [1] + [0] * num_neg
            out = {
                "input_ids": enc["input_ids"],                 # (1+K, L)
                "attention_mask": enc["attention_mask"],       # (1+K, L)
                "labels": labels,                               # (1+K,)
                "index": int(sample["index"]),
            }
            if "token_type_ids" in enc:
                out["token_type_ids"] = enc["token_type_ids"]  # (1+K, L)
            return out

        return fn

    # ---------- Collator ----------
    def get_collator(self, tok):
        use_token_type = "token_type_ids" in tok.model_input_names

        class Collator:
            def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                # Shapes per sample: (G, L), with G = 1 + num_negatives
                input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)         # (B, G, L)
                attn = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)         # (B, G, L)
                labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)               # (B, G)
                index = torch.tensor([ex["index"] for ex in batch], dtype=torch.long)                    # (B,)

                features = {
                    "input_ids": input_ids,
                    "attention_mask": attn.bool(),
                    "labels": labels,
                    "index": index,
                }

                if use_token_type and ("token_type_ids" in batch[0]):
                    tti = torch.tensor([ex["token_type_ids"] for ex in batch], dtype=torch.long)      # (B, G, L)
                    features["token_type_ids"] = tti

                return features

        return Collator()

    # ---------- Metrics ----------
    def compute_metrics(self, tok, cfg):
        # With one positive per group, these reduce to simple rank-based formulas
        def ndcg_at_k(rank: int, k: int = 10) -> float:
            if rank <= 0 or rank > k:
                return 0.0
            # IDCG for a single relevant item is 1 / log2(1 + 1) == 1
            return 1.0 / math.log2(rank + 1)

        def metric_fn(pred):
            # predictions could be (N*G, 1) or (N*G,) depending on model head
            logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            labels = pred.label_ids  # expected shape (N, G), 1 positive per row

            # Ensure shapes
            logits = torch.tensor(logits)
            if logits.ndim == 2 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            labels = torch.tensor(labels).long()

            # Infer group size G from labels
            if labels.ndim == 1:
                raise ValueError("Labels should be grouped per query: shape (N, G).")
            N, G = labels.shape

            if logits.numel() == N * G:
                logits = logits.view(N, G)

            # Compute rank of the positive for each query
            scores = logits.detach().cpu().numpy()
            s_pos = scores[:, 0]  # positive scores
            s_neg = scores[:, 1:]  # negative scores
            y = labels.detach().cpu().numpy()
            constraint_slacks = s_pos[:, None] - s_neg  # (N, K)
            constraint_slacks = constraint_slacks.flatten()
            # Flatten slacks and output table to wandb
            if cfg.train.use_wandb:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["constraint_slack"])
                    for slack in constraint_slacks.tolist():
                        table.add_data(slack)
                    wandb.log({"constraint_slacks": table})

            ranks = []
            for i in range(N):
                order = scores[i].argsort()[::-1]  # descending
                # position (1-based) of the positive (label == 1)
                pos_idx = int((y[i] == 1).nonzero()[0])
                rank = int((order == pos_idx).nonzero()[0]) + 1
                ranks.append(rank)

            # Metrics
            mrr10 = sum((1.0 / r) if r <= 10 else 0.0 for r in ranks) / N
            mean_ndcg10 = sum(ndcg_at_k(r, 10) for r in ranks) / N
            # For a single relevant doc, MAP reduces to precision@rank = 1/rank

            out = {
                "eval_NanoBEIR_R100_mean_ndcg@10": float(mean_ndcg10),
                "eval_mrr@10": float(mrr10),
                "mean_constraint_violation": constraint_slacks.abs().mean().item(),
                "min_constraint_violation": constraint_slacks.abs().min().item(),
                "max_constraint_violation": constraint_slacks.abs().max().item(),
            }
            breakp = getattr(cfg.train, "debug_metrics", False)
            if breakp:
                breakpoint()
            # If the trainer provided per-batch losses, report mean
            if hasattr(pred, "losses") and pred.losses is not None:
                try:
                    out["eval_loss"] = float(torch.tensor(pred.losses).mean().item())
                except Exception:
                    pass
            return out

        return metric_fn

    # ---------- Trainer ----------
    def get_trainer_class(self):
        class CustomTrainer(Trainer):
            """
            Implements Cached-MNRL-like loss:
                L = mean_{i in batch} mean_{neg in 1..K} -log sigma(scale * (s_pos - s_neg))
                where s_* are raw logits (higher is more relevant)
            """
            def __init__(self, *args, experiment=None, dual_vars=None,eval=False, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                if not eval:
                    self.init_dual_vars()

            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros(len(self.train_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")  # (B, G) with exactly one '1' per row
                index = inputs.pop("index", None)
                core_model = extract_model_from_parallel(model)
                cfg = core_model.config
                # Flatten (B, G, L) -> (B*G, L)
                B, G, L = inputs["input_ids"].shape
                flat = {k: v.view(B * G, L) for k, v in inputs.items() if k in {"input_ids", "attention_mask", "token_type_ids"} and k in inputs}

                outputs = model(**flat)
                logits = outputs.logits.squeeze(-1)  # (B*G,)
                logits = logits.view(B, G)           # (B, G)

                # Assume index 0 is positive, 1..K negatives (we encoded that in preprocessing)
                s_pos = logits[:, 0].unsqueeze(1)    # (B, 1)
                s_neg = logits[:, 1:]                # (B, K)
                slack = -(s_pos - s_neg) + cfg.exp.loss_tol

                if cfg.loss_type == "ls":
                    loss = -F.logsigmoid(slack * cfg.loss_alpha).sum()
                elif cfg.loss_type == "l2":
                    loss = (torch.clamp(slack , 0.0)**2).sum()
                elif cfg.loss_type == "l1":
                    loss = torch.clamp(slack, 0.0).sum()
                elif cfg.loss_type == "dual":
                    dual_var = self.dual_vars[index].clone()
                    dual_var += 2 * cfg.loss_alpha * slack
                    self.dual_vars[index] = dual_var.detach()
                    loss += (
                        dual_var.detach()
                        * slack
                    ).sum()
                if return_outputs:
                    # Return flattened outputs so HF eval loop collects predictions consistently
                    outputs.logits = logits.view(B * G, 1)
                    return loss, outputs
                return loss

        return CustomTrainer
