# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForMultipleChoice,RobertaForMultipleChoice, AutoConfig
from .base import Experiment
from transformers import Trainer
import torch, torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
import numpy as np
import os
from utils import freeze_for_mc


class BIAS(Experiment):
    
    def load_model_and_tok(self, cfg):
        configuration = AutoConfig.from_pretrained(cfg.exp.model_name)
        if cfg.exp.model_name == "answerdotai/ModernBERT-base" or cfg.exp.model_name == "answerdotai/ModernBERT-large":
            configuration.attention_dropout = cfg.train.dropout
            configuration.mlp_dropout = cfg.train.dropout
            configuration.classifier_dropout = cfg.train.dropout
        elif cfg.exp.model_name == "FacebookAI/roberta-large" or cfg.exp.model_name == "FacebookAI/roberta-base":
            
            configuration.classifier_dropout = cfg.train.dropout
        model = AutoModelForMultipleChoice.from_pretrained(cfg.exp.model_name, config=configuration)
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model.main_input_name = 'probs'
        model = freeze_for_mc(model, train_classifier=True, unfreeze_last_n_layers=cfg.train.unfreeze_last_n_layers) if cfg.train.freeze_base_model else model
        return model, tok

    def load_datasets(self, cfg):
        # Load and shuffle the full datasets first
        tr = load_dataset("iboero16/winogrande_bias", split="train").shuffle(seed=cfg.train.seed)
        ev = load_dataset("iboero16/winogrande_bias", split="eval").shuffle(seed=cfg.train.seed)

        # Take the required proportion after shuffling
        tr_size = int(cfg.train.data_proportion * len(tr))
        ev_size = int(cfg.train.data_proportion * len(ev))
        tr = tr.select(range(tr_size))
        ev = ev.select(range(ev_size))
        
        # Add index column
        tr = tr.add_column("index", list(range(len(tr))))
        ev = ev.add_column("index", list(range(len(ev))))
        print(f"Training samples: {len(tr)}, Eval samples: {len(ev)}")
        return tr, ev, None

    def preprocessing_fn(self, tok, cfg):
        def fn(sample):
            input_text = sample['sentence']
            if not isinstance(input_text, str):
                raise ValueError(f'Unsupported type of `input`: {type(input_text)}. Expected: str.')
            opt1, opt2 = sample["option1"], sample["option2"]

            if "_" in input_text:
                prefix, suffix = input_text.split("_", 1)
            else:
                raise ValueError(f'Unsupported type of `input`: {type(input_text)}. Expected: str.')

            text_a = [prefix, prefix]
            text_b = [opt1 + suffix, opt2 + suffix]

            tokenized = tok(text_a, text_b, return_token_type_ids=True, return_attention_mask=True, return_tensors=None, padding='max_length', max_length=128, truncation=True, padding_side='right')
      
            # Stack input_ids and attention_mask for both choices
            input_ids = tokenized["input_ids"]  # shape: (2, L)
            attention_mask = tokenized["attention_mask"]  # shape: (2, L)
            index = sample['index']  # Get the index for dual variable updates
            token_type_ids = tokenized["token_type_ids"]  # shape: (2, L)

            # Determine correct label (0 for option1, 1 for option2)
            correct = sample["answer"]
            if correct == sample["option1"]:  # answer is option1
                label = 0
            elif correct == sample["option2"]:  # other_answer is option2
                label = 1
            else:
                raise ValueError(f'Invalid `correct` value: {correct}. Expected one of: {sample["option1"]}, {sample["option2"]}.')
            if bool(sample['gender_bias']):
                if cfg.exp.ratio == "fair":
                    probs = sample['stereo_prob'] / (1 + sample['stereo_prob'])
                elif cfg.exp.ratio == "ones":
                    probs = 0.5
                elif cfg.exp.ratio == "reverse":
                    probs = 1 - (sample['stereo_prob'] / (1 + sample['stereo_prob']))
            else:
                probs = -1.0
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids':token_type_ids,
                'labels': int(label),
                'is_constraint': bool(sample['gender_bias']),
                'probs': probs,  # Add ratio for fair/ones
                'index': int(index)
            }
        return fn

    def get_collator(self, tok):
        class ClassificationBiasCollator():
            
            def __init__(self, pad_token_id: int) -> None:
                """Initialize a collator."""
                self.pad_token_id = pad_token_id
                
            def __call__(self, samples):
                input_ids = torch.stack([torch.tensor(sample['input_ids']) for sample in samples])  # (B, 2, L)
                attention_mask = torch.stack([torch.tensor(sample['attention_mask']) for sample in samples])  # (B, 2, L)
                token_type_ids = torch.stack([torch.tensor(sample["token_type_ids"]) for sample in samples])      # (B, 2, L) ← NEW

                labels = torch.tensor(
                    [sample['labels'] for sample in samples],
                    dtype=torch.long,
                )
                is_constraint = torch.tensor(
                    [sample['is_constraint'] for sample in samples],
                    dtype=torch.bool,
                )
                probs = torch.tensor(
                    [sample['probs'] for sample in samples],
                    dtype=torch.float,
                )
                index = torch.tensor(
                    [sample['index'] for sample in samples],
                    dtype=torch.long,
                )
                return {
                    'input_ids': input_ids,  # size = (B, 2, L)
                    'attention_mask': attention_mask.bool(),  # size = (B, 2, L)
                    "token_type_ids": token_type_ids,
                    'labels': labels,  # size = (B,)
                    'is_constraint': is_constraint,  # size = (B,)
                    'probs': probs,  # size = (B,)
                    'index': index,  # size = (B,)
                }
        return ClassificationBiasCollator(tok.pad_token_id)



    def get_trainer_class(self):
        class CustomTrainer(Trainer):
            def __init__(self, *args, experiment=None,complete_dataset=None, tokenizer=None, custom_cfg=None, eval=False, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                self.tok = tokenizer
                self.init_dual_vars()
                self.compute_metrics = self._compute_metrics
                self.custom_cfg = custom_cfg

        
            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros(len(self.train_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)
                self.avg_dual = torch.tensor(0.0, device=self.model.device, dtype=torch.float)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Loss function for the bias classification algorithm using Multiple Choice.

                Args:
                    model (nn.Module): The model to compute the loss for.
                    inputs (dict): The inputs to the model.
                    return_outputs (bool): Whether to return the model outputs.

                Returns:
                    dict[str, torch.Tensor]: loss, objective, constraint_value
                """
                core_model = extract_model_from_parallel(model)
                cfg = self.custom_cfg.exp
                
                input_ids = inputs['input_ids']  # size = (B, 2, L)
                attention_mask = inputs['attention_mask']  # size = (B, 2, L)
                token_type_ids = inputs['token_type_ids']  # size = (B, 2, L)
                labels = inputs['labels']  # size = (B,)
                target = inputs['probs']  # size = (B,)
                incorrect_labels = (1 - labels)  # flip 0->1, 1->0
                is_constraint = inputs['is_constraint'].squeeze()  # size = (B,)
                is_not_constraint = ~is_constraint
                index = inputs['index']  # size = (B,)
                dual_var = self.dual_vars[index].clone()  # Get dual variable for the current batch

                # Get model outputs - for multiple choice, input shape should be (B, 2, L)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits  # size = (B, 2) - logits for each choice
                log_probs = F.log_softmax(logits, dim=-1)  # size = (B, 2)
                # Get probabilities for correct and incorrect options
                correct_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # size = (B,)
                incorrect_log_probs = log_probs.gather(1, incorrect_labels.unsqueeze(1)).squeeze(1)  # size = (B,)
                target = target.clamp(1e-7, 1 - 1e-7)  # Avoid log(-1) issues on non-constraint samples
                # Compute log probability difference (correct - incorrect)
                dkl = target * (torch.log(target) - correct_log_probs) + (1 - target) * (torch.log(1 - target) - incorrect_log_probs) # KL divergence

                
                # For constraint samples, we want the probabilities to be equal (difference = 0)
                # For non-constraint samples, we want correct to have higher probability
                slack = (dkl - cfg.tol) * is_constraint.float()  # constraint violation
                loss = -1 * correct_log_probs * is_not_constraint.float() # objective

                
                # Add constraint penalization
                if cfg.loss_type == "erm":
                    pass
                
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
                                  
                loss = loss.mean()
                
                if return_outputs:
                    return loss, outputs
                else:
                    return loss
                
            def _compute_metrics(self,pred):
                cfg = self.custom_cfg
                logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
                labels = pred.label_ids
                loss = pred.losses
                probs = pred.inputs
                logits = torch.tensor(logits); labels = torch.tensor(labels).long();  loss = torch.tensor(loss); probs = torch.tensor(probs)
                is_constraint = probs != -1.0
                is_not_constraint = ~is_constraint
                log_probs = F.log_softmax(logits, dim=-1)
                correct_log_probs = log_probs.gather(1, labels.reshape(-1, 1)).squeeze(1)
                incorrect_log_probs = log_probs.gather(1, (1 - labels).reshape(-1, 1)).squeeze(1)
                epoch = self.state.epoch

                constraint_slacks = probs * (torch.log(probs) - correct_log_probs) + (1 - probs) * (torch.log(1 - probs) - incorrect_log_probs)  # KL divergence
                constraint_slacks = constraint_slacks[is_constraint]  # Only keep constraint samples
                metric_constraints = torch.abs(probs - torch.exp(correct_log_probs))[is_constraint]
                objective = -1 * correct_log_probs * is_not_constraint.float()
                acc =  ((logits.argmax(dim=-1) == labels)[~is_constraint]).float().mean().item()
                bias = ((logits.argmax(dim=-1) == labels)[is_constraint]).float().mean().item()

                # Log table with constraint slacks for wandb
                rank = int(os.environ.get("RANK", "0"))
                if cfg.train.use_wandb and rank == 0:
                    import wandb
                    if wandb.run is not None:
                        table_metrics = wandb.Table(columns=["constraint_slack", "metric_constraint"])
                        for slack, metric in zip(constraint_slacks.tolist(), metric_constraints.tolist()):
                            table_metrics.add_data(slack, metric)
                        wandb.log({f"constraint_slacks_epoch_{epoch}_{self._current_eval_prefix}": table_metrics})
                    # Add histogram of constraint slacks
                    wandb.log({f"constraint_slacks_histogram_epoch_{epoch}_{self._current_eval_prefix}": wandb.Histogram(constraint_slacks.cpu().numpy())})
                    wandb.log({f"metric_constraints_histogram_epoch_{epoch}_{self._current_eval_prefix}": wandb.Histogram(metric_constraints.cpu().numpy())})

                return {"accuracy": acc,
                        "bias_accuracy": bias,
                        "loss": loss.mean().item(),
                        "objective": objective.mean().item(), 
                        "mean_dkl_violation": constraint_slacks.mean().item(),
                        "cvar_dkl_violation": constraint_slacks[constraint_slacks > 0].mean().item() if (constraint_slacks > 0).any() else 0.0,
                        "mean_metric_constraint": metric_constraints.mean().item(),
                        "cvar_metric_constraint": metric_constraints[metric_constraints > 0].mean().item() if (metric_constraints > 0).any() else 0.0
                        }
        return CustomTrainer
