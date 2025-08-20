# src/eval.py
import hydra
import torch
import os
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed
from experiments.registry import EXPERIMENTS
import wandb


@hydra.main(config_path="../configs", config_name="eval/bias", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    
    # Initialize wandb if enabled
    if cfg.train.use_wandb:
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    ExpCls = EXPERIMENTS[cfg.exp.name]
    exp = ExpCls()
    
    # Check if model path exists
    model_path = cfg.train.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load base model and tokenizer first
    model, tok = exp.load_model_and_tok(cfg)
    
    import re

    # Search for checkpoint folders in model_path with names like "checkpoint-<number>"
    checkpoint_folders = []
    for entry in os.listdir(model_path):
        full_path = os.path.join(model_path, entry)
        if os.path.isdir(full_path):
            match = re.match(r'checkpoint-(\d+)', entry)
            if match:
                checkpoint_folders.append((int(match.group(1)), full_path))
    
    if checkpoint_folders:
        # Get the folder with the largest number
        latest_checkpoint_folder = max(checkpoint_folders, key=lambda x: x[0])[1]
        checkpoint_path = os.path.join(latest_checkpoint_folder, "model.safetensors")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            model, tok = exp.load_model_and_tok(cfg, load_checkpoint=True,checkpoint_path=latest_checkpoint_folder)
        else:
            print(f"Warning: 'model.safetensors' not found in {latest_checkpoint_folder}, using base model")
    else:
        print(f"Warning: No checkpoint folders found in {model_path}, using base model")
    # Load datasets
    train_ds, eval_ds = exp.load_datasets(cfg)
    preprocess = exp.preprocessing_fn(tok, cfg)
    cols = eval_ds.column_names
    
    # Preprocess evaluation dataset
    eval_ds = eval_ds.map(preprocess, remove_columns=cols)
    
    # Preprocess training dataset if eval_on_train is enabled
    if cfg.train.eval_on_train:
        train_cols = train_ds.column_names
        train_ds = train_ds.map(preprocess, remove_columns=train_cols)
    
    collator = exp.get_collator(tok)
    
    # Create compute_metrics function with debug support
    compute_metrics_fn = exp.compute_metrics(tok, cfg)
    
    # if cfg.train.debug:
    #     def debug_compute_metrics(eval_pred):
    #         import pdb; pdb.set_trace()  # Debug breakpoint
    #         return metric_fn(eval_pred)
    #     compute_metrics_fn = debug_compute_metrics
    # else:
    # compute_metrics_fn = metric_fn

    # Configure evaluation arguments
    args = TrainingArguments(
        output_dir="./temp_eval",  # Temporary directory for eval
        **cfg.train.hf_args,
        do_train=True,
        do_eval=True,
        report_to=["wandb"] if cfg.train.use_wandb else [],
        run_name=cfg.train.run_name if cfg.train.use_wandb else None,
        logging_steps=1,
    )

    # Create trainer for evaluation
    trainer = exp.get_trainer_class()(
        model=model,
        args=args,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics_fn,
        experiment=exp,
        eval=True,
    )
    
    print("Starting evaluation on test dataset...")
    eval_results = trainer.evaluate(metric_key_prefix="eval")
    print("Evaluation results (test set):")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # Evaluate on training set if requested
    if cfg.train.eval_on_train:
        print("\nStarting evaluation on training dataset...")
        trainer.eval_dataset = train_ds
        train_results = trainer.evaluate(metric_key_prefix="train")
        print("Evaluation results (train set):")
        for key, value in train_results.items():
            print(f"  {key}: {value}")
    
    # Finish wandb run
    if cfg.train.use_wandb:
        wandb.finish()

    print("Evaluation completed!")


if __name__ == "__main__":
    main()