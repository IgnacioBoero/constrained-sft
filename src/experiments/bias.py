# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForMultipleChoice
from .base import Experiment
from transformers import Trainer
import torch, torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel

class BIAS(Experiment):
    
    def load_model_and_tok(self, cfg):
        model = AutoModelForMultipleChoice.from_pretrained(cfg.exp.model_name)
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.config.loss_alpha = cfg.exp.loss_alpha
        model.config.loss_type = cfg.exp.loss_type
        model.main_input_name = 'is_constraint'
        return model, tok

    def load_datasets(self, cfg):
        tr = load_dataset("iboero16/winogrande_bias", split=f"train[:{int(cfg.train.data_proportion*100)}%]")
        ev = load_dataset("iboero16/winogrande_bias", split=f"eval[:{int(cfg.train.data_proportion*100)}%]")
        tr = tr.add_column("index", list(range(len(tr))))
        ev = ev.add_column("index", list(range(len(ev))))
        return tr, ev

    def preprocessing_fn(self, tok, model):
        def fn(sample):
            input_text = sample['sentence']
            if not isinstance(input_text, str):
                raise ValueError(f'Unsupported type of `input`: {type(input_text)}. Expected: str.')

            # Tokenize both choices
            tokenized = tok([input_text, input_text], [sample['option1'], sample['option2']], return_tensors="pt",return_attention_mask=True, padding='max_length',max_length=128, truncation=True,padding_side='right')
            
            # Stack input_ids and attention_mask for both choices
            input_ids = tokenized["input_ids"]  # shape: (2, L)
            attention_mask = tokenized["attention_mask"]  # shape: (2, L)
            index = sample['index']  # Get the index for dual variable updates
            
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
                'labels': torch.tensor(label, dtype=torch.long),
                'is_constraint': sample['gender_bias'],
                'index': torch.tensor(index, dtype=torch.long)
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
                    'labels': labels,  # size = (B,)
                    'is_constraint': is_constraint,  # size = (B,)
                    'index': index,  # size = (B,)
                }
        return ClassificationBiasCollator(tok.pad_token_id)

    def compute_metrics(self, tok, cfg):
        def metric_fn(pred):
            logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            labels = pred.label_ids
            is_constraint = pred.inputs
            logits = torch.tensor(logits); labels = torch.tensor(labels).long(); is_constraint = torch.tensor(is_constraint).bool()
            constraint_slacks = (logits[:, 0] - logits[:, 1])[is_constraint]
            objective = -1 * logits.gather(1, labels.reshape(-1, 1)).squeeze(1) * (~is_constraint).float()
            acc =  (logits.argmax(dim=-1) == labels).float().mean().item()

            # Log table with constraint slacks for wandb
            if cfg.train.use_wandb:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["constraint_slack"])
                    for slack in constraint_slacks.tolist():
                        table.add_data(slack)
                    wandb.log({"constraint_slacks": table})

            return {"accuracy": acc, 
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
                objective = -1 * correct_log_probs * is_not_constraint.float()  # maximize correct for non-constraint
                loss = objective

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
                    # Update duals before computing the loss
                    
                    dual_var += 2 * cfg.loss_alpha * slack
                    self.dual_vars[index] = dual_var  
                    loss += (
                        dual_var
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

                # Total loss
                loss = loss.mean()
                if return_outputs:
                    return loss, outputs
                else:
                    return loss
        return CustomTrainer
