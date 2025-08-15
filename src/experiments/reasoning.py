# src/your_proj/experiments/sft_cekl.py
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .base import Experiment
from transformers import Trainer
import torch, torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
from utils import right_padding
from torch.backends.cuda import sdp_kernel
from torch.autograd.profiler import record_function
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class REASONING(Experiment):
    
    def load_model_and_tok(self, cfg):
        model = AutoModelForCausalLM.from_pretrained(
            cfg.exp.model_name, 
            torch_dtype=torch.bfloat16 if cfg.train.hf_args.bf16 else torch.float32,
            trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(cfg.exp.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        # Apply LoRA if enabled
        if getattr(cfg.exp, 'use_lora', False):
            lora_config = LoraConfig(
                r=cfg.exp.lora.r,
                lora_alpha=cfg.exp.lora.lora_alpha,
                target_modules=cfg.exp.lora.target_modules,
                lora_dropout=cfg.exp.lora.lora_dropout,
                bias=cfg.exp.lora.bias,
                task_type=getattr(TaskType, cfg.exp.lora.task_type)
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            model.enable_input_require_grads()
        model.config.loss_tol = cfg.exp.loss_tol
        model.config.loss_type = cfg.exp.loss_type
        model.config.use_cache = False
        model.config.sliding_window = None
        return model, tok

    def load_datasets(self, cfg):
        tr = load_dataset(cfg.train.train_ds, split="train")
        tr_size = int(cfg.train.data_proportion * len(tr))
        tr = tr.select(range(tr_size))
        tr, ev = tr.train_test_split(test_size=0.1).values()
        tr = tr.add_column("index", list(range(len(tr))))
        ev = ev.add_column("index", list(range(len(ev))))
        return tr, ev

    def preprocessing_fn(self, tok, cfg):
        def fn(sample):
            input_text = tok.apply_chat_template(
                [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                 {"role": "user", "content": sample['question']},
                 {"role": "assistant", "content": sample['solution']}],
                tokenize=False,
                max_length=cfg.train.max_length,
                truncation=True,
            )
            prompt_text = tok.apply_chat_template(
                [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                 {"role": "user", "content": sample['question']}],
                tokenize=False,
                max_length=cfg.train.max_length,
                truncation=True,
                add_generation_prompt=True
            )
            enc = tok(input_text, max_length=cfg.train.max_length, truncation=True)
            prompt_ids = tok(prompt_text, max_length=cfg.train.max_length, truncation=True)["input_ids"]
            full_ids = enc["input_ids"]

            labels = full_ids.copy()
            cutoff = len(prompt_ids)
            labels[:cutoff] = [-100] * cutoff
            index = sample['index']  # Get the index for dual variable updates
            return {
                'input_ids': full_ids,
                'labels': labels,
                'index': int(index)
            }
        return fn

    def get_collator(self, tok):
        class Collator:
            def __init__(self, pad_token_id: int) -> None:
                """Initialize a collator."""
                self.pad_token_id = pad_token_id
                
            def __call__(self, samples):
                input_ids = right_padding(
                    [torch.tensor(sample['input_ids']) for sample in samples],
                    padding_value=self.pad_token_id,
                )
                labels = right_padding(
                    [torch.tensor(sample['labels']) for sample in samples],
                    padding_value=-100,
                )
                attention_mask = input_ids.ne(self.pad_token_id)
                index = torch.tensor(
                    [sample['index'] for sample in samples],
                    dtype=torch.long,
                )

                return {
                    'input_ids': input_ids,  # size = (B,  L)
                    'attention_mask': attention_mask,  # size = (B, L)
                    'labels': labels,  # size = (B, L)
                    'index': index,  # size = (B,)
                }
                
        return Collator(tok.pad_token_id)

    def compute_metrics(self, tok, cfg):
        def metric_fn(pred):
            logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            labels = pred.label_ids
            loss = pred.losses
            input_ids = pred.inputs
            logits = torch.tensor(logits); labels = torch.tensor(labels).long();  loss = torch.tensor(loss); input_ids = torch.tensor(input_ids)
            log_probs = torch.gather(F.log_softmax(logits[:,:-1], dim=-1), dim=-1, index=input_ids[:, 1:].unsqueeze(dim=-1)).squeeze(-1)
            log_probs = log_probs[labels[:,1:] != -100]
            constraint_slacks = log_probs
            objective = -log_probs.mean().item()
            constraint_satisfaction = (constraint_slacks >= cfg.exp.loss_tol).float().mean().item()
            if cfg.train.use_wandb:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["constraint_slack"])
                    for slack in constraint_slacks.tolist():
                        table.add_data(slack)
                    wandb.log({"constraint_slacks": table})

            return {"loss": loss.mean().item(),
                    "objective": objective,
                    "constraint_satisfaction": constraint_satisfaction,
                    "mean_constraint_violation": constraint_slacks.abs().mean().item(),
                    "min_constraint_violation": constraint_slacks.abs().min().item(),
                    "max_constraint_violation": constraint_slacks.abs().max().item()}
        return metric_fn


    def get_trainer_class(self):
        
        class CustomTrainer(Trainer):
            def __init__(self, *args, experiment=None, dual_vars=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.experiment = experiment
                # self.init_dual_vars()
                
            def _save(self, output_dir: str, state_dict=None):
                # Save LoRA adapter if using PEFT
                if isinstance(self.model, PeftModel):
                    self.model.save_pretrained(output_dir)
                    # Also save tokenizer
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        self.tokenizer.save_pretrained(output_dir)
                else:
                    super()._save(output_dir, state_dict)
                
        
            # def init_dual_vars(self):
            #     """Initialize dual variables for the current batch."""
            #     self.dual_vars = torch.zeros(len(self.train_dataset), dtype=torch.float, requires_grad=False).to(self.model.device)
            #     print(f"Initialized dual variables with shape: {self.dual_vars.shape} on device {self.dual_vars.device}")
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Loss function for the bias classification algorithm using Multiple Choice.

                Args:
                    model (nn.Module): The model to compute the loss for.
                    inputs (dict): The inputs to the model.
                    return_outputs (bool): Whether to return the model outputs.

                Returns:
                    dict[str, torch.Tensor]: loss, objective, constraint_value
                """

                inputs = self._prepare_inputs(inputs)
                cfg = self.model.config
                input_ids = inputs['input_ids']  # size = (B, 2, L)
                attention_mask = inputs['attention_mask']  # size = (B, 2, L)
                labels = inputs['labels']  # size = (B,)
                if cfg.loss_type == "hf":
                    with record_function("forward"):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                    if return_outputs:
                        return outputs.loss, outputs
                    else:
                        return outputs.loss
                
                
                with record_function("forward"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )                        
                logits = outputs.logits  # size = (B, 2) - logits for each choice
                log_probs = torch.gather(F.log_softmax(logits[:,:-1], dim=-1), dim=-1, index=input_ids[:, 1:].unsqueeze(dim=-1)).squeeze(-1)
                log_probs_answer = log_probs[labels[:,1:] != -100]
                slack = cfg.loss_tol - log_probs_answer

                if cfg.loss_type == "erm":
                    loss = -log_probs_answer
                    
                elif cfg.loss_type == "l2":
                    loss = (
                        torch.clamp(slack, 0, None) ** 2
                    )

                elif cfg.loss_type == "l1":
                    loss = (
                        torch.clamp(slack, 0, None)
                    )
                    
                loss = loss.mean()
                
                if return_outputs:
                    return loss, outputs
                else:
                    return loss
                
        return CustomTrainer
