# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForMultipleChoice,RobertaForMultipleChoice, AutoConfig
from .base import Experiment
from transformers import Trainer
import torch, torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel

class BIAS(Experiment):
    
    def load_model_and_tok(self, cfg):
        # config = AutoConfig.from_pretrained(cfg.exp.model_name, hidden_dropout_prob=cfg.train.dropout, attention_probs_dropout_prob=cfg.train.dropout)
        model = AutoModelForMultipleChoice.from_pretrained(cfg.exp.model_name)#, config=config)
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.config.loss_alpha = cfg.exp.loss_alpha
        model.config.loss_type = cfg.exp.loss_type
        model.main_input_name = 'is_constraint'
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
        return tr, ev

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

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids':token_type_ids,
                'labels': int(label),
                'is_constraint': bool(sample['gender_bias']),
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
                    'index': index,  # size = (B,)
                }
        return ClassificationBiasCollator(tok.pad_token_id)

    def compute_metrics(self, tok, cfg):
        def metric_fn(pred):
            logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            labels = pred.label_ids
            loss = pred.losses
            is_constraint = pred.inputs
            logits = torch.tensor(logits); labels = torch.tensor(labels).long();  loss = torch.tensor(loss); is_constraint = torch.tensor(is_constraint).bool()
            log_probs = F.log_softmax(logits, dim=-1)

            constraint_slacks = (logits[:, 0] - logits[:, 1])[is_constraint]
            objective = -1 * log_probs.gather(1, labels.reshape(-1, 1)).squeeze(1)[~is_constraint]
            acc =  ((logits.argmax(dim=-1) == labels)[~is_constraint]).float().mean().item()
            bias = ((logits.argmax(dim=-1) == labels)[is_constraint]).float().mean().item()

            # Log table with constraint slacks for wandb
            if cfg.train.use_wandb:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["constraint_slack"])
                    for slack in constraint_slacks.tolist():
                        table.add_data(slack)
                    wandb.log({"constraint_slacks": table})

            return {"accuracy": acc, 
                    "bias_accuracy": bias,
                    "loss": loss.mean().item(),
                    "objective": objective.mean().item(), 
                    "mean_constraint_violation": constraint_slacks.abs().mean().item(),
                    "min_constraint_violation": constraint_slacks.abs().min().item(),
                    "max_constraint_violation": constraint_slacks.abs().max().item()}
        return metric_fn


    def get_trainer_class(self):
        class CustomTrainer(Trainer):
            def __init__(self, *args, experiment=None, dual_vars=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                self.init_dual_vars()
                
        
            def init_dual_vars(self):
                """Initialize dual variables for the current batch."""
                self.dual_vars = torch.zeros(len(self.train_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)

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
                cfg = core_model.config
                
                input_ids = inputs['input_ids']  # size = (B, 2, L)
                attention_mask = inputs['attention_mask']  # size = (B, 2, L)
                token_type_ids = inputs['token_type_ids']  # size = (B, 2, L)
                labels = inputs['labels']  # size = (B,)
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

                # Compute log probability difference (correct - incorrect)
                log_prob_diff = correct_log_probs - incorrect_log_probs
                
                # For constraint samples, we want the probabilities to be equal (difference = 0)
                # For non-constraint samples, we want correct to have higher probability
                slack = log_prob_diff * is_constraint.float()  # constraint violation

                if is_not_constraint.any():
                    loss = -correct_log_probs[is_not_constraint]
                else:
                    loss = torch.tensor(0.0, device=correct_log_probs.device, requires_grad=True)
                
                # Add constraint penalization
                if cfg.loss_type == "erm":
                    pass
                elif cfg.loss_type == "l2":
                    loss += (
                        cfg.loss_alpha
                        / 2
                        * slack ** 2
                    ).sum()

                elif cfg.loss_type == "l1":
                    loss += (
                        cfg.loss_alpha
                        / 2
                        * slack.abs()
                    ).sum()

                elif cfg.loss_type == "dual":                   
                    dual_var += 2 * cfg.loss_alpha * slack
                    self.dual_vars[index] = dual_var.detach()  
                    loss += (
                        dual_var.detach()
                        * slack
                    ).sum()
                    
                elif cfg.loss_type == "dual_res":
                    # Update duals before computing the loss
                    dual_var += -1 * dual_var + 2 * cfg.loss_alpha * (slack)
                    self.dual_vars[index] = dual_var
                    loss += (
                        dual_var
                        * slack
                    ).sum()

                loss = loss.mean()
                if return_outputs:
                    return loss, outputs
                else:
                    return loss
        return CustomTrainer
