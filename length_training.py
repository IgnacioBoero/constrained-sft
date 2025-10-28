#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_ID = "answerdotai/ModernBERT-large"
SEED = 42


def pick_shortest_and_longest_negative(example: Dict) -> Optional[Dict]:
    """
    From an MS MARCO v1.1 example, return dict with:
      query, short (shortest negative), long (longest negative).
    Return None if fewer than 2 negatives exist.
    """
    passages = example["passages"]["passage_text"]
    flags = example["passages"]["is_selected"]
    negatives = [p for p, f in zip(passages, flags) if f == 0]
    if len(negatives) < 2:
        return None

    shortest = min(negatives, key=lambda s: len(s))
    longest  = max(negatives, key=lambda s: len(s))
    return {"query": example["query"], "short": shortest, "long": longest}


def make_pair(record: Dict) -> Dict:
    """
    Construct (query, ans_a, ans_b, label) with random order to avoid positional bias.
    label = 1 iff ans_a is the longer one.
    """
    if random.random() < 0.5:
        ans_a, ans_b = record["long"], record["short"]
        label = 1
    else:
        ans_a, ans_b = record["short"], record["long"]
        label = 0
    return {"query": record["query"], "ans_a": ans_a, "ans_b": ans_b, "label": label}


def build_pairs_split(hf_split) -> Dataset:
    rows = []
    for ex in hf_split:
        picked = pick_shortest_and_longest_negative(ex)
        if picked is None:
            continue
        rows.append(make_pair(picked))
    return Dataset.from_list(rows)


def tokenize_batch(examples, tokenizer, max_length=None):
    """
    Pack each example as a *pair input*:
      text = query + [SEP] + ans_a
      text_pair = ans_b
    This uses the model’s native [SEP] for the first sequence boundary.
    Label meaning: 1 iff ans_a is longer.
    """
    # build “first” string as query + sep + ans_a
    sep = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"
    first_texts  = [q + f" {sep} " + a for q, a in zip(examples["query"], examples["ans_a"])]
    second_texts = examples["ans_b"]

    enc = tokenizer(
        first_texts,
        second_texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # use dynamic padding via DataCollatorWithPadding
        return_tensors=None,
    )
    enc["labels"] = examples["label"]
    return enc


def freeze_encoder_keep_head(model):
    """
    Freeze all params, then unfreeze only the classification head.
    Works for most HF seq-class models (e.g., Bert*, ModernBERT*, RoBERTa*).
    """
    for p in model.parameters():
        p.requires_grad = False

    # Common head attribute names:
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "score"):  # some models use .score
        for p in model.score.parameters():
            p.requires_grad = True
    else:
        # Fallback: unfreeze last module parameters (rarely needed)
        head = list(model.modules())[-1]
        for p in head.parameters():
            p.requires_grad = True


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": float(acc)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "head-only"], default="full",
                        help="Fine-tune full model or only the classification head.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None, help="Default depends on mode.")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./modernbert_len_pred_seqcls")
    parser.add_argument("--max_train_examples", type=int, default=50000)
    parser.add_argument("--max_eval_examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Override tokenizer.model_max_length (e.g., 1024).")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset
    ds_hf: DatasetDict = load_dataset("microsoft/ms_marco", "v1.1")
    train_pairs = build_pairs_split(ds_hf["train"])
    val_pairs   = build_pairs_split(ds_hf["validation"])

    if args.max_train_examples and args.max_train_examples < len(train_pairs):
        train_pairs = train_pairs.select(range(args.max_train_examples))
    if args.max_eval_examples and args.max_eval_examples < len(val_pairs):
        val_pairs = val_pairs.select(range(args.max_eval_examples))

    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # num_labels=2 for softmax classification
    config = AutoConfig.from_pretrained(MODEL_ID, num_labels=2)
    model  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)

    # Pick a sane default LR per mode
    if args.lr is None:
        args.lr = 2e-5 if args.mode == "full" else 1e-3

    # Tokenize (map) with dynamic padding at collator time
    tgt_max_len = args.max_length or min(tokenizer.model_max_length, 2048)
    train_tok = train_pairs.map(
        lambda ex: tokenize_batch(ex, tokenizer, max_length=tgt_max_len),
        batched=True,
        remove_columns=train_pairs.column_names,
        desc="Tokenizing train",
    )
    val_tok = val_pairs.map(
        lambda ex: tokenize_batch(ex, tokenizer, max_length=tgt_max_len),
        batched=True,
        remove_columns=val_pairs.column_names,
        desc="Tokenizing val",
    )

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    if args.mode == "head-only":
        freeze_encoder_keep_head(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        # bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation:", metrics)


if __name__ == "__main__":
    main()
