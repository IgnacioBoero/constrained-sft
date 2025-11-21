#!/usr/bin/env python3
# test_eval.py - Simple test of eval.py functionality

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from eval import find_all_trained_models, evaluate_model
from experiments.base import EXPERIMENTS
from omegaconf import OmegaConf

def test_find_models():
    """Test finding trained models"""
    output_dir = "outputs/reranker/answerdotai/ModernBERT-large-ls-spearman_corr-10-small"
    checkpoints = find_all_trained_models(output_dir)
    print(f"Found checkpoints in {output_dir}:")
    for cp in checkpoints:
        print(f"  {cp}")
    return checkpoints

def test_evaluate_single_model():
    """Test evaluating a single model"""
    # Load a simple config
    cfg = OmegaConf.create({
        'exp': {
            'name': 'reranker',
            'model_name': 'answerdotai/ModernBERT-large',
            'num_negatives': 9,
            'max_length': 512,
            'loss_type': 'ls',
            'loss_tol': 10.0,
            'alpha': 1,
            'length_constraint': False,
            'ls_clamp': False,
            'obj_type': 'spearman_corr',
            'top_k': 3
        },
        'train': {
            'seed': 42,
            'output_dir': './outputs/test',
            'data_proportion': 0.1,  # Use small subset for testing
            'size': 'small',
            'hf_args': {
                'per_device_eval_batch_size': 1,
                'bf16': True,
                'fp16': False,
            }
        }
    })
    
    # Get experiment
    ExpCls = EXPERIMENTS[cfg.exp.name]
    exp = ExpCls()
    
    # Load small datasets
    train_ds, eval_ds = exp.load_datasets(cfg)
    print(f"Loaded {len(train_ds)} train and {len(eval_ds)} eval samples")
    
    # Test with first checkpoint found
    output_dir = "outputs/reranker/answerdotai/ModernBERT-large-ls-spearman_corr-10-small"
    checkpoints = find_all_trained_models(output_dir)
    
    if checkpoints:
        checkpoint_path = checkpoints[0]
        print(f"Testing evaluation on: {checkpoint_path}")
        
        results = evaluate_model(exp, checkpoint_path, train_ds, eval_ds, cfg)
        print("Evaluation results:")
        for split, metrics in results.items():
            print(f"  {split}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")
    else:
        print("No checkpoints found for testing")

if __name__ == "__main__":
    print("Testing eval.py functionality...")
    
    print("\n1. Testing model discovery:")
    checkpoints = test_find_models()
    
    if checkpoints:
        print("\n2. Testing single model evaluation:")
        test_evaluate_single_model()
    else:
        print("\n2. Skipping model evaluation - no checkpoints found")
    
    print("\nTest completed!")