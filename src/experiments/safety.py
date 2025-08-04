# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from .base import Experiment

class SFTCEKL(Experiment):
    def build_tokenizer(self, cfg):
        tok = AutoTokenizer.from_pretrained(cfg.model.name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def build_model(self, cfg):
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
        # save knobs in config for checkpoints
        model.config.loss_alpha = cfg.exp.loss_alpha
        model.config.loss_temperature = cfg.exp.temperature
        return model

    def load_datasets(self, cfg):
        tr = load_dataset(cfg.data.name, split=cfg.data.train_split)
        ev = load_dataset(cfg.data.name, split=cfg.data.eval_split)
        return tr, ev

    def preprocessing_fn(self, tok, cfg):
        field = cfg.data.text_field
        max_len = cfg.model.max_length
        def fn(batch):
            x = tok(batch[field], truncation=True, max_length=max_len)
            x["labels"] = x["input_ids"].copy()
            return x
        return fn

    def build_collator(self, tok, cfg):
        return DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    def compute_metrics(self, tok, cfg):
        def fn(pred):
            import torch, torch.nn.functional as F
            logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            labels = pred.label_ids
            logits = torch.tensor(logits); labels = torch.tensor(labels).long()
            mask = labels.ne(-100)
            logp = F.log_softmax(logits, -1)
            nll = F.nll_loss(logp.transpose(-1,-2), labels, ignore_index=-100, reduction="none")
            per_ex = (nll*mask).sum(1)/mask.sum(1).clamp_min(1)
            return {"eval_ppl": float(torch.exp(per_ex.mean()))}
        return fn

    def extra_loss(self, trainer, outputs, inputs):
        alpha = float(getattr(trainer.model.config, "loss_alpha", 0.0))
        if alpha == 0.0: return outputs.loss*0
        temp = float(getattr(trainer.model.config, "loss_temperature", 0.07))
        logits = outputs.logits
        labels = inputs["labels"]
        mask = labels.ne(-100)
        with torch.no_grad():
            tgt = torch.zeros_like(logits).float()
            tgt.scatter_(-1, labels.clamp(min=0).unsqueeze(-1), 1.0)
            tgt = F.softmax(tgt / temp, dim=-1)
        kl = F.kl_div(F.log_softmax(logits, -1), tgt, reduction="none").sum(-1)
        kl = (kl * mask).sum() / mask.sum().clamp_min(1)
        return alpha * kl
