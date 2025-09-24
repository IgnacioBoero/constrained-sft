# eval.py
"""
Evaluation script for trained models from constrained-sft experiments.

This script evaluates all models trained during a specific training run, handling 
hyperparameter sweeps and producing comprehensive evaluation metrics.

Usage:
    python eval.py --config-path configs/eval --config-name reranker.yaml

Features:
- Automatically discovers all trained model checkpoints from a training run
- Handles Hydra multirun sweeps by evaluating all parameter combinations
- Evaluates on both training and evaluation datasets
- Robust error handling: fills metrics with -1 if model loading fails
- Outputs results to CSV for analysis

The evaluation config should specify:
- train_config: Path to the original training config used
- All experiment parameters matching the training setup
"""
import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import hydra
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed
from experiments.base import EXPERIMENTS
import numpy as np

def is_global_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"

def find_all_trained_models(output_dir: str) -> List[str]:
    """Find all trained model checkpoints in the output directory."""
    checkpoint_paths = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Warning: Output directory {output_dir} does not exist")
        return []
    
    # Search for checkpoint directories
    for root, dirs, files in os.walk(output_path):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_path = os.path.join(root, dir_name)
                # Verify it has the necessary model files
                model_file = os.path.join(checkpoint_path, "model.safetensors")
                config_file = os.path.join(checkpoint_path, "config.json")
                if os.path.exists(model_file) and os.path.exists(config_file):
                    checkpoint_paths.append(checkpoint_path)
    
    return sorted(checkpoint_paths)

def evaluate_model(exp, checkpoint_path: str, train_ds, eval_ds, cfg: DictConfig) -> Dict[str, Any]:
    """Evaluate a single model checkpoint."""
    try:
        print(f"Evaluating model at: {checkpoint_path}")
        
        # Load model and tokenizer from checkpoint
        model, tok = exp.load_model_and_tok(cfg, load_checkpoint=True, checkpoint_path=checkpoint_path)
        
        # Prepare datasets
        preprocess = exp.preprocessing_fn(tok, cfg)
        cols_train = train_ds.column_names
        cols_eval = eval_ds.column_names
        
        train_ds_processed = train_ds.map(preprocess, remove_columns=cols_train)
        eval_ds_processed = eval_ds.map(preprocess, remove_columns=cols_eval)
        
        collator = exp.get_collator(tok)
        metric_fn = exp.compute_metrics(tok, cfg)
        
        # Create training arguments for evaluation
        args = TrainingArguments(
            output_dir=cfg.train.output_dir,
            per_device_eval_batch_size=cfg.train.hf_args.get("per_device_eval_batch_size", 1),
            bf16=cfg.train.hf_args.get("bf16", True),
            fp16=cfg.train.hf_args.get("fp16", False),
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=[],  # Disable wandb for eval
        )
        
        # Create trainer for evaluation
        trainer = exp.get_trainer_class()(
            model=model,
            args=args,
            eval_dataset=eval_ds_processed,
            data_collator=collator,
            tokenizer=tok,
            compute_metrics=metric_fn,
            experiment=exp,
            eval=True  # Flag to skip dual variable initialization
        )
        
        # Evaluate on both train and eval datasets
        results = {}
        
        # Evaluate on eval dataset
        print(f"  Evaluating on validation set...")
        eval_results = trainer.evaluate(eval_dataset=eval_ds_processed, metric_key_prefix="eval")
        results["eval"] = {k.replace("eval_", ""): v for k, v in eval_results.items()}
        
        # Evaluate on train dataset 
        print(f"  Evaluating on training set...")
        train_results = trainer.evaluate(eval_dataset=train_ds_processed, metric_key_prefix="train")
        results["train"] = {k.replace("train_", ""): v for k, v in train_results.items()}
        
        return results
        
    except Exception as e:
        print(f"Error evaluating model at {checkpoint_path}: {str(e)}")
        # Return -1 for all metrics to indicate failure
        return {
            "eval": {
                "NanoBEIR_R100_mean_ndcg@10": -1,
                "mrr@10": -1,
                "acc@3": -1,
                "mean_constraint_violation": -1,
                "min_constraint_violation": -1,
                "max_constraint_violation": -1,
                "length_weighted_mrr": -1,
                "avg_top3_length": -1,
                "loss": -1
            },
            "train": {
                "NanoBEIR_R100_mean_ndcg@10": -1,
                "mrr@10": -1,
                "acc@3": -1,
                "mean_constraint_violation": -1,
                "min_constraint_violation": -1,
                "max_constraint_violation": -1,
                "length_weighted_mrr": -1,
                "avg_top3_length": -1,
                "loss": -1
            }
        }

@hydra.main(config_path="configs", config_name="eval/default", version_base=None)
def main(cfg: DictConfig):
    """
    Evaluate all trained models from a specific training run.
    
    Usage:
        python eval.py --config-path configs/eval --config-name reranker.yaml
    """
    
    set_seed(cfg.train.seed)
    
    # Load the corresponding train config to get the output directory pattern
    train_config_path = cfg.get("train_config", None)
    if train_config_path is None:
        raise ValueError("eval config must specify 'train_config' pointing to the train config file")
    
    # Load train config to understand the hyperparameter sweep structure
    # Load without resolving interpolations to get the raw template
    train_cfg = OmegaConf.load(train_config_path)
    
    # Get experiment class
    ExpCls = EXPERIMENTS[cfg.exp.name]
    exp = ExpCls()
    
    # Load datasets (same as training)
    train_ds, eval_ds = exp.load_datasets(cfg)
    
    # Get all possible output directories based on hydra multirun sweep
    if "hydra" in train_cfg and "sweeper" in train_cfg.hydra:
        # Extract parameter sweep combinations
        sweep_params = train_cfg.hydra.sweeper.params
        results_all = []
        
        # Parse sweep parameters
        param_combinations = []
        param_names = list(sweep_params.keys())
        param_values = []
        
        for param_name, param_value_str in sweep_params.items():
            # Handle different parameter types
            if isinstance(param_value_str, (bool, int, float)):
                # Single value parameter
                values = [param_value_str]
            else:
                # String parameter that might have multiple values
                values = [v.strip() for v in str(param_value_str).split(",")]
            param_values.append(values)
        
        # Generate all combinations
        from itertools import product
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        print(f"Found {len(param_combinations)} parameter combinations to evaluate")
        
        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            print(f"\n[{i+1}/{len(param_combinations)}] Evaluating parameter combination: {params}")
            
            # Build output directory for this combination
            # Get the raw template string from the config
            output_dir_template = "./outputs/reranker/${exp.model_name}-${exp.loss_type}-${exp.obj_type}-${exp.loss_tol}-${train.size}"
            
            # Create a mapping of all variables for substitution
            substitutions = {}
            
            # Add base config values
            substitutions.update({
                "exp.model_name": train_cfg.exp.model_name,
                "train.size": train_cfg.train.size if hasattr(train_cfg.train, 'size') else "small"
            })
            
            # Add sweep parameter values
            substitutions.update(params)
            
            # Replace all placeholders in the output directory
            output_dir = output_dir_template
            for key, value in substitutions.items():
                placeholder = "${" + key + "}"
                output_dir = output_dir.replace(placeholder, str(value))
            
            # Find all model checkpoints in this output directory
            checkpoint_paths = find_all_trained_models(output_dir)
            
            if not checkpoint_paths:
                print(f"  No trained models found in {output_dir}")
                # Add failed entry with -1 metrics
                result_entry = {
                    "params": params,
                    "output_dir": output_dir,
                    "checkpoint_path": None,
                    "status": "no_models_found"
                }
                # Add -1 metrics for both train and eval
                for split in ["train", "eval"]:
                    result_entry[f"{split}_NanoBEIR_R100_mean_ndcg@10"] = -1
                    result_entry[f"{split}_mrr@10"] = -1
                    result_entry[f"{split}_acc@3"] = -1
                    result_entry[f"{split}_mean_constraint_violation"] = -1
                    result_entry[f"{split}_min_constraint_violation"] = -1
                    result_entry[f"{split}_max_constraint_violation"] = -1
                    result_entry[f"{split}_length_weighted_mrr"] = -1
                    result_entry[f"{split}_avg_top3_length"] = -1
                    result_entry[f"{split}_loss"] = -1
                results_all.append(result_entry)
                continue
            
            # Evaluate each checkpoint
            for checkpoint_path in checkpoint_paths:
                # Update config with current parameters
                eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
                for param_key, param_value in params.items():
                    # Set parameter values in config
                    keys = param_key.split(".")
                    config_section = eval_cfg
                    for key in keys[:-1]:
                        if key not in config_section:
                            config_section[key] = {}
                        config_section = config_section[key]
                    
                    # Convert value to appropriate type
                    if isinstance(param_value, bool):
                        config_section[keys[-1]] = param_value
                    elif isinstance(param_value, (int, float)):
                        config_section[keys[-1]] = param_value
                    elif isinstance(param_value, str):
                        if param_value.lower() in ["true", "false"]:
                            config_section[keys[-1]] = param_value.lower() == "true"
                        elif param_value.replace(".", "").replace("-", "").isdigit():
                            if "." in param_value:
                                config_section[keys[-1]] = float(param_value)
                            else:
                                config_section[keys[-1]] = int(param_value)
                        else:
                            config_section[keys[-1]] = param_value
                    else:
                        config_section[keys[-1]] = param_value
                
                # Evaluate this checkpoint
                results = evaluate_model(exp, checkpoint_path, train_ds, eval_ds, eval_cfg)
                
                # Create result entry
                result_entry = {
                    "params": params,
                    "output_dir": output_dir,
                    "checkpoint_path": checkpoint_path,
                    "status": "success" if results["eval"]["NanoBEIR_R100_mean_ndcg@10"] != -1 else "failed"
                }
                
                # Flatten results for CSV
                for split in ["train", "eval"]:
                    for metric, value in results[split].items():
                        result_entry[f"{split}_{metric}"] = value
                
                results_all.append(result_entry)
        
        # Save results to CSV
        df = pd.DataFrame(results_all)
        output_csv = "evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        
        # Print summary
        successful_runs = len([r for r in results_all if r["status"] == "success"])
        total_runs = len(results_all)
        print(f"Successfully evaluated: {successful_runs}/{total_runs} model checkpoints")
        
    else:
        # Single run evaluation (no multirun)
        output_dir = train_cfg.train.output_dir
        checkpoint_paths = find_all_trained_models(output_dir)
        
        if not checkpoint_paths:
            print(f"No trained models found in {output_dir}")
            return
        
        results_all = []
        for checkpoint_path in checkpoint_paths:
            results = evaluate_model(exp, checkpoint_path, train_ds, eval_ds, cfg)
            result_entry = {
                "checkpoint_path": checkpoint_path,
                "status": "success" if results["eval"]["NanoBEIR_R100_mean_ndcg@10"] != -1 else "failed"
            }
            
            # Flatten results
            for split in ["train", "eval"]:
                for metric, value in results[split].items():
                    result_entry[f"{split}_{metric}"] = value
            
            results_all.append(result_entry)
        
        # Save results
        df = pd.DataFrame(results_all)
        output_csv = "evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()