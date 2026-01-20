# src/your_proj/experiments/reranker_cmnrl.py
import math
import numpy as np
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

from utils import ndcg_at_k, compute_passage_lengths_from_tokens


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
        configuration.attention_dropout = cfg.train.dropout
        configuration.mlp_dropout = cfg.train.dropout
        configuration.classifier_dropout = cfg.train.dropout
        model = AutoModelForSequenceClassification.from_pretrained(name, config=configuration)
        tok = AutoTokenizer.from_pretrained(name)
        
        if tok.pad_token is None:
            # Prefer SEP if available, otherwise EOS if present
            tok.pad_token = getattr(tok, "sep_token", None) or getattr(tok, "eos_token", None)

        return model, tok

    # ---------- Dataset ----------
    def load_datasets(self, cfg):
        
        num_neg = int(cfg.exp.num_negatives)
        assert num_neg >= 1, f"num_negatives must be >= 1, got {num_neg}"
        dataset_size = getattr(cfg.train, 'size',' large')
        
        # Load all splits
        train = load_dataset("iboero16/reranker", dataset_size, split="train")
        eval = load_dataset("iboero16/reranker", dataset_size, split="validation")

        # Apply data proportion if specified
        tr_size = int(cfg.train.data_proportion * len(train))
        ev_size = int(cfg.train.data_proportion * len(eval))
        train = train.select(range(tr_size))
        eval = eval.select(range(ev_size))
        
        # Add index columns for tracking
        train = train.add_column("index", list(range(len(train))))
        eval = eval.add_column("index", list(range(len(eval))))

        print(f"Training samples: {len(train)}, Eval samples: {len(eval)}")
        return train, eval, None

    # ---------- Preprocessing ----------
    def preprocessing_fn(self, tok, cfg):
        num_neg = int(cfg.exp.num_negatives)
        max_len = int(cfg.exp.max_length)

        def fn(sample):
            q = sample["query"]
            pos = sample["positive"]
            negs = sample["negative"]
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
            
            # Extract length information from the lengths dict
            lengths_dict = sample["lengths"]
            passage_lengths = [lengths_dict["positive"]] + lengths_dict["negatives"]
            
            # Labels: 1 for positive (index 0), 0 for negatives
            labels = [1] + [0] * num_neg
            out = {
                "input_ids": enc["input_ids"],                 # (1+K, L)
                "attention_mask": enc["attention_mask"],       # (1+K, L)
                "labels": labels,                               # (1+K,)
                "index": int(sample["index"]),
                "passage_lengths": passage_lengths,            # (1+K,) - lengths of positive + negatives
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
                
                # Handle new length features
                passage_lengths = torch.tensor([ex["passage_lengths"] for ex in batch], dtype=torch.long)  # (B, G)

                features = {
                    "input_ids": input_ids,
                    "attention_mask": attn.bool(),
                    "labels": labels,
                    "index": index,
                    "passage_lengths": passage_lengths,
                }

                if use_token_type and ("token_type_ids" in batch[0]):
                    tti = torch.tensor([ex["token_type_ids"] for ex in batch], dtype=torch.long)      # (B, G, L)
                    features["token_type_ids"] = tti

                return features

        return Collator()



    # ---------- Trainer ----------
    def get_trainer_class(self):
        class CustomTrainer(Trainer):

            def __init__(self, *args, experiment=None,complete_dataset=None, tokenizer=None, custom_cfg=None, eval=False, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                self.custom_cfg = custom_cfg
                self.tok = tokenizer
                if not eval:
                    self.init_dual_vars()
                self.compute_metrics = self._compute_metrics
                
            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros((len(self.train_dataset), self.custom_cfg.exp.num_negatives), dtype=torch.float, requires_grad=False).to(self.model.device)
                self.avg_dual = torch.tensor(0.0, device=self.model.device, dtype=torch.float)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                eps = 1e-8
                index = inputs.pop("index", None)
                passage_lengths = inputs.pop("passage_lengths", None)
                
                cfg = self.custom_cfg.exp
                

                # Flatten (B, G, L) -> (B*G, L)
                B, G, L = inputs["input_ids"].shape
                flat = {k: v.view(B * G, L) for k, v in inputs.items() if k in {"input_ids", "attention_mask", "token_type_ids"} and k in inputs}

                outputs = model(**flat)
                logits = outputs.logits.squeeze(-1).view(B, G)    # (B, G)
                
                # Compute objective function based on configured type
                s = logits[:, 1:]                        # scores for negatives, (B, G-1)
                L = passage_lengths[:, 1:].to(s.dtype)   # lengths for negatives, (B, G-1)

                obj_type = cfg.obj_type
                match obj_type:
                    case "pearson_corr":
                        # Anti-correlation between scores and lengths
                        s_centered = s - s.mean(dim=1, keepdim=True)
                        L_centered = L - L.mean(dim=1, keepdim=True)
                        cov = (s_centered * L_centered).mean(dim=1)  # (B,)
                        std_s = s_centered.pow(2).mean(dim=1).sqrt().clamp_min(eps)  # (B,)
                        std_L = L_centered.pow(2).mean(dim=1).sqrt().clamp_min(eps)  # (B,)
                        corr = cov / (std_s * std_L)              # Pearson corr(s, L) per sample, (B,)
                        obj = (1.0 + corr) * 0.5                  # anti-corr loss in [0,1]; 0 is best

                    case "spearman_corr":
                        # Spearman rank correlation between scores and lengths
                        s_ranks = s.argsort(dim=1, descending=True).argsort(dim=1).to(s.dtype) + 1
                        L_ranks = L.argsort(dim=1).argsort(dim=1).to(s.dtype) + 1
                        s_ranks_centered = s_ranks - s_ranks.mean(dim=1, keepdim=True)
                        L_ranks_centered = L_ranks - L_ranks.mean(dim=1, keepdim=True)
                        num = (s_ranks_centered * L_ranks_centered).sum(dim=1)
                        denom = torch.sqrt((s_ranks_centered**2).sum(dim=1) * (L_ranks_centered**2).sum(dim=1) + eps)
                        spearman_corr = num / denom
                        obj = (1.0 + spearman_corr) * 0.5  # Anti-correlation

                    case "kendall_tau":
                        # Kendall's Tau rank correlation (smooth approximation)
                        n = s.size(1)
                        concordant = torch.zeros(s.size(0), device=s.device, dtype=s.dtype)
                        for i in range(n):
                            for j in range(i + 1, n):
                                s_diff = s[:, i] - s[:, j]           # (B,)
                                L_diff = L[:, j] - L[:, i]           # (B,) reversed for anti-corr
                                concordant += torch.tanh(s_diff * L_diff)
                        denom_pairs = (n * (n - 1) / 2)
                        tau = concordant / denom_pairs
                        obj = (1.0 - tau) * 0.5

                    case "lambda_loss":
                        B_obj, N = s.shape
                        K = int(getattr(cfg, "k_lambda", 2) or 2)
                        K = max(1, min(K, N))

                        # Gains: higher for shorter lengths (any monotone map works)
                        gains = (L.max(dim=1, keepdim=True).values - L).clamp_min(0)  # (B, N)

                        # Current ranks from model scores (1 = top)
                        order = s.argsort(dim=1, descending=True)
                        ranks = torch.empty_like(order, dtype=torch.long)
                        ranks.scatter_(1, order, torch.arange(1, N + 1, device=s.device).unsqueeze(0).expand(B_obj, N))

                        # Discount 1 / log2(rank+1)
                        discount = 1.0 / torch.log2(ranks.to(s.dtype) + 1.0)  # (B, N)

                        # Ideal DCG@K for normalization (sort by gains desc)
                        ideal_idx = gains.argsort(dim=1, descending=True)
                        ideal_ranks = torch.arange(1, N + 1, device=s.device).unsqueeze(0).expand(B_obj, N)
                        ideal_discount = 1.0 / torch.log2(ideal_ranks.to(s.dtype) + 1.0)
                        idcg = (gains.gather(1, ideal_idx) * (ideal_discount * (ideal_ranks <= K)).to(s.dtype)).sum(dim=1).clamp_min(eps)  # (B,)

                        # All upper-triangular pairs i<j (per list)
                        I, J = torch.triu_indices(N, N, offset=1, device=s.device)  # (P,)
                        obj_per_list = []
                        for b in range(B_obj):
                            gb = gains[b]         # (N,)
                            sb = s[b]             # (N,)
                            rb = ranks[b]         # (N,)
                            db = discount[b]      # (N,)
                            idcgb = idcg[b]       # scalar

                            # Ground-truth pair preference sign ∈{-1,0,1}; 0=ties ignored
                            y_ij = torch.sign(gb[I] - gb[J])

                            # ΔDCG@K if we swap items at current ranks r_i and r_j
                            ri, rj = rb[I], rb[J]
                            in_top_i = (ri <= K).to(s.dtype)
                            in_top_j = (rj <= K).to(s.dtype)
                            dcg_before = gb[I] * db[I] * in_top_i + gb[J] * db[J] * in_top_j
                            dcg_after = gb[I] * db[J] * in_top_j + gb[J] * db[I] * in_top_i
                            w = (dcg_after - dcg_before).abs() / idcgb  # (P,)

                            # Weighted pairwise logistic: log(1 + exp(-y_ij * (s_i - s_j)))
                            sdiff = sb[I] - sb[J]
                            mask = (y_ij != 0)
                            pair_loss = torch.nn.functional.softplus(-y_ij.to(s.dtype) * sdiff) * w * mask.to(s.dtype)

                            denom = (w * mask.to(s.dtype)).sum().clamp_min(eps)
                            obj_per_list.append((pair_loss.sum() / denom).unsqueeze(0))

                        obj = torch.cat(obj_per_list, dim=0)  # (B,)

                    case _:
                        # Unknown objective
                        raise ValueError(f"Unknown objective type: {obj_type}")
                
            
                loss = torch.zeros(B, dtype=logits.dtype, device=logits.device)
                if cfg.length_constraint:
                    loss += obj


                s_pos = logits[:, 0]    # (B, 1)
                s_neg = logits[:, 1:]                # (B, K) 
                slack = cfg.loss_tol -(s_pos[:, None] - s_neg)  # (B, K)

                # Loss schemes
                
                ## BASELINES
                if cfg.loss_type == "erm":
                    # Pairwise logistic (your current ERM baseline)
                    loss += (-1 * F.logsigmoid(10 * (s_pos[:, None] - s_neg))).mean(dim=1)

                # elif cfg.loss_type == "hinge":
                #     # Pairwise hinge loss: max(0, margin - (s_pos - s_neg))
                #     # i.e. max(0, slack)
                #     margin_violation = F.relu(slack)          # (B, K)
                #     loss += margin_violation.mean(dim=1)      # (B,)

                # elif cfg.loss_type == "sq_hinge":
                #     # Squared hinge: max(0, margin - (s_pos - s_neg))^2
                #     margin_violation = F.relu(slack)          # (B, K)
                #     loss += (margin_violation ** 2).mean(dim=1)

                ## DUALS METHODS
                elif cfg.loss_type == "avg":
                    dual_avg = self.avg_dual.clone()
                    dual_avg = torch.clamp(dual_avg + cfg.dual_step_size * slack.mean(), min=0.0)
                    self.avg_dual = dual_avg.detach()
                    loss += (
                        dual_avg.detach()
                        * slack
                    ).mean(dim=1)
                    # log the avg dual
                    #self.log({"avg_dual_var": dual_avg.detach()})           

                elif cfg.loss_type == "dual":
                    dual_var = self.dual_vars[index].clone()
                    dual_var = torch.clamp(dual_var + cfg.dual_step_size * slack, min=0.0)
                    self.dual_vars[index] = dual_var.detach()
                    loss += (
                        dual_var.detach()
                        * slack
                    ).mean(dim=1)

                elif cfg.loss_type == "aug_dual":
                    dual_var = self.dual_vars[index].clone()
                    a = slack
                    b = dual_var / (cfg.alpha)
                    z = 2 * a + b
                    dual_grad = torch.where(z > 0, a, -0.5 * b)
                    dual_var += cfg.dual_step_size * dual_grad
                    self.dual_vars[index] = dual_var.detach()
                    loss += cfg.alpha / 4 * (
                        torch.clamp(z, min=0.0)**2 - b**2
                    ).mean(dim=1)
                
                
                elif cfg.loss_type == "l2":
                    loss += cfg.alpha / 2 * (torch.clamp(slack,min=0.0) ** 2)
                    
                elif cfg.loss_type == "penalty":
                    loss += cfg.alpha * slack 
                    
                loss = loss.mean()
                
                if return_outputs:
                    outputs.logits = logits.view(B * G, 1)
                    return loss, outputs
                return loss


            def _compute_metrics(self,pred):
                cfg = self.custom_cfg

                inputs = getattr(pred, 'inputs', None)
                
                # predictions could be (N*G, 1) or (N*G,) depending on model head
                logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
                labels = pred.label_ids  # expected shape (N, G), 1 positive per row

                # Ensure shapes
                # inputs = torch.tensor(inputs)
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
                else:
                    raise ValueError(f"Logits shape {logits.shape} is incompatible with labels shape {labels.shape}.")

                # Compute passage lengths if inputs are available
                passage_lengths = None
                all_lengths = compute_passage_lengths_from_tokens(inputs, self.tok)
                passage_lengths = torch.tensor(all_lengths).view(N, G)  # (N, G)

                # Compute rank of the positive for each query
                scores = logits.detach().cpu().numpy()
                s_pos = scores[:, 0]  # positive scores
                s_neg = scores[:, 1:]  # negative scores
                y = labels.detach().cpu().numpy()
                constraint_slacks = cfg.exp.loss_tol - (s_pos[:, None] - s_neg)  # (N, K)
                constraint_slacks = constraint_slacks.flatten()
                
                epoch = self.state.epoch
                # Flatten slacks and output table to wandb
                if cfg.train.use_wandb:
                    import wandb
                    if wandb.run is not None:
                        table = wandb.Table(columns=["constraint_slack"])
                        for slack in constraint_slacks.tolist():
                            table.add_data(slack)
                        wandb.log({f"constraint_slacks_epoch_{epoch}_{self._current_eval_prefix}": table})

                ranks = []
                length_weighted_scores = []
                acc_at_3_scores = []
                top3_avg_lengths = []
                
                for i in range(N):
                    order = scores[i].argsort()[::-1]  # descending order of scores
                    # position (1-based) of the positive (label == 1)
                    pos_idx = int((y[i] == 1).nonzero()[0])
                    rank = int((order == pos_idx).nonzero()[0]) + 1
                    ranks.append(rank)
                    
                    # Acc@3: 1 if positive is in top 3, 0 otherwise
                    acc_at_3_scores.append(1.0 if rank <= 3 else 0.0)
                    
                    if passage_lengths is not None:
                        # Length-weighted ranking metric (only for negatives)
                        neg_lengths = passage_lengths[i, 1:].numpy()  # lengths of negatives only
                        neg_scores = scores[i, 1:]  # scores of negatives only
                        neg_order = neg_scores.argsort()[::-1]  # descending order of negative scores
                        
                        # Compute length-weighted score: sum of (length / rank) for each negative
                        length_weighted_score = 0.0
                        for rank_idx, neg_idx in enumerate(neg_order):
                            weight = 1.0 / (rank_idx + 1)  # 1/1, 1/2, 1/3, etc.
                            length_weighted_score += neg_lengths[neg_idx] * weight
                        length_weighted_scores.append(length_weighted_score)
                        
                        # Average length of top-3 passages (including positive if in top 3)
                        top3_indices = order[:3]
                        top3_lengths = [passage_lengths[i, idx].item() for idx in top3_indices]
                        top3_avg_lengths.append(np.mean(top3_lengths))
                
                # Standard metrics
                mrr10 = sum((1.0 / r) if r <= 10 else 0.0 for r in ranks) / N
                mean_ndcg10 = sum(ndcg_at_k(r, 10) for r in ranks) / N
                acc_at_3 = np.mean(acc_at_3_scores)
                
                # Compute objective value
                eps = 1e-8
                s = torch.tensor(scores[:, 1:])  # scores for negatives, (N, G-1)
                L = passage_lengths[:, 1:].to(s.dtype)  # lengths for negatives, (N, G-1)
                obj_type = cfg.exp.obj_type
                
                match obj_type:
                    case "pearson_corr":
                        # Anti-correlation between scores and lengths
                        s_centered = s - s.mean(dim=1, keepdim=True)
                        L_centered = L - L.mean(dim=1, keepdim=True)
                        cov = (s_centered * L_centered).mean(dim=1)  # (N,)
                        std_s = s_centered.pow(2).mean(dim=1).sqrt().clamp_min(eps)  # (N,)
                        std_L = L_centered.pow(2).mean(dim=1).sqrt().clamp_min(eps)  # (N,)
                        corr = cov / (std_s * std_L)  # Pearson corr(s, L) per sample, (N,)
                        obj = (1.0 + corr) * 0.5  # anti-corr loss in [0,1]; 0 is best (perfect inverse)

                    case "spearman_corr":
                        # Spearman rank correlation between scores and lengths
                        s_ranks = s.argsort(dim=1, descending=True).argsort(dim=1).to(s.dtype) + 1
                        L_ranks = L.argsort(dim=1).argsort(dim=1).to(s.dtype) + 1
                        s_ranks_centered = s_ranks - s_ranks.mean(dim=1, keepdim=True)
                        L_ranks_centered = L_ranks - L_ranks.mean(dim=1, keepdim=True)
                        spearman_corr = (s_ranks_centered * L_ranks_centered).sum(dim=1) / torch.sqrt(
                            (s_ranks_centered**2).sum(dim=1) * (L_ranks_centered**2).sum(dim=1) + eps)
                        obj = (1.0 + spearman_corr) * 0.5  # Anti-correlation

                    case "kendall_tau":
                        # Kendall's Tau rank correlation (smooth approximation)
                        n = s.size(1)
                        concordant = torch.zeros(s.size(0), device=s.device, dtype=s.dtype)
                        for i in range(n):
                            for j in range(i + 1, n):
                                s_diff = s[:, i] - s[:, j]           # (N,)
                                L_diff = L[:, j] - L[:, i]           # (N,) reversed for anti-corr
                                concordant += torch.tanh(s_diff * L_diff)
                        denom_pairs = (n * (n - 1) / 2)
                        tau = concordant / denom_pairs
                        obj = (1.0 - tau) * 0.5

                    case "lambda_loss":
                        N_obj, N = s.shape
                        K = int(getattr(cfg.exp, "k_lambda", 2) or 2)
                        K = max(1, min(K, N))

                        # Gains: higher for shorter lengths (any monotone map works)
                        gains = (L.max(dim=1, keepdim=True).values - L).clamp_min(0)  # (N_obj, N)

                        # Current ranks from model scores (1 = top)
                        order = s.argsort(dim=1, descending=True)
                        ranks = torch.empty_like(order, dtype=torch.long)
                        ranks.scatter_(1, order, torch.arange(1, N + 1, device=s.device).unsqueeze(0).expand(N_obj, N))

                        # Discount 1 / log2(rank+1)
                        discount = 1.0 / torch.log2(ranks.to(s.dtype) + 1.0)  # (N_obj, N)

                        # Ideal DCG@K for normalization (sort by gains desc)
                        ideal_idx = gains.argsort(dim=1, descending=True)
                        ideal_ranks = torch.arange(1, N + 1, device=s.device).unsqueeze(0).expand(N_obj, N)
                        ideal_discount = 1.0 / torch.log2(ideal_ranks.to(s.dtype) + 1.0)
                        idcg = (gains.gather(1, ideal_idx) * (ideal_discount * (ideal_ranks <= K)).to(s.dtype)).sum(dim=1).clamp_min(eps)  # (N_obj,)

                        # All upper-triangular pairs i<j (per list)
                        I, J = torch.triu_indices(N, N, offset=1, device=s.device)  # (P,)
                        obj_per_list = []
                        for b in range(N_obj):
                            gb = gains[b]         # (N,)
                            sb = s[b]             # (N,)
                            rb = ranks[b]         # (N,)
                            db = discount[b]      # (N,)
                            idcgb = idcg[b]       # scalar

                            # Ground-truth pair preference sign ∈{-1,0,1}; 0=ties ignored
                            y_ij = torch.sign(gb[I] - gb[J])

                            # ΔDCG@K if we swap items at current ranks r_i and r_j
                            ri, rj = rb[I], rb[J]
                            in_top_i = (ri <= K).to(s.dtype)
                            in_top_j = (rj <= K).to(s.dtype)
                            dcg_before = gb[I] * db[I] * in_top_i + gb[J] * db[J] * in_top_j
                            dcg_after = gb[I] * db[J] * in_top_j + gb[J] * db[I] * in_top_i
                            w = (dcg_after - dcg_before).abs() / idcgb  # (P,)

                            # Weighted pairwise logistic: log(1 + exp(-y_ij * (s_i - s_j)))
                            sdiff = sb[I] - sb[J]
                            mask = (y_ij != 0)
                            pair_loss = torch.nn.functional.softplus(-y_ij.to(s.dtype) * sdiff) * w * mask.to(s.dtype)

                            denom = (w * mask.to(s.dtype)).sum().clamp_min(eps)
                            obj_per_list.append((pair_loss.sum() / denom).unsqueeze(0))

                        obj = torch.cat(obj_per_list, dim=0)  # (N_obj,)

                    case _:
                        # Unknown objective
                        raise ValueError(f"Unknown objective type: {obj_type}")

                out = {
                    "NanoBEIR_R100_mean_ndcg@10": float(mean_ndcg10),
                    "mrr@10": float(mrr10),
                    "acc@3": float(acc_at_3),
                    "mean_constraint_violation": constraint_slacks.mean().item(),
                    "cvar_constraint_violation": float(torch.tensor(constraint_slacks).float().kthvalue(int(0.9 * len(constraint_slacks))).values.item()),
                    "objective_value": float(obj.mean().item()),
                }
                
                # Add length-based metrics if available
                if passage_lengths is not None and length_weighted_scores:
                    out["length_weighted_mrr"] = float(np.mean(length_weighted_scores))
                    out["avg_top3_length"] = float(np.mean(top3_avg_lengths))
                
                # add dual if using avg
                if cfg.exp.loss_type == "avg":
                    out["avg_dual_var"] = float(self.avg_dual.detach().item())
                    
                return out
        return CustomTrainer
