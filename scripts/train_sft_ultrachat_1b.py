#!/usr/bin/env python3
"""
One-epoch SFT on HuggingFaceH4/ultrachat_200k with Llama-3.2-1B (base model).

Defaults match the requested setup:
- model: meta-llama/Llama-3.2-1B
- dataset: HuggingFaceH4/ultrachat_200k
- instruct template: meta-llama/Llama-3.2-1B-Instruct chat template
- micro batch size: 1
- grad accumulation: 32
- optimizer: adamw_torch (default AdamW in Transformers)
- learning rate: 5e-07
- epochs: 1
- push target: ihounie/1B-ultrachat
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_INSTRUCT_TEMPLATE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_HUB_MODEL_ID = "ihounie/1B-ultrachat"


class StopOnLossThresholdCallback(TrainerCallback):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.triggered or logs is None:
            return control

        loss = logs.get("loss")
        if loss is None:
            return control

        try:
            loss_value = float(loss)
        except (TypeError, ValueError):
            return control

        if loss_value <= self.threshold:
            self.triggered = True
            print(
                "[train_sft_ultrachat_1b] Early stopping: "
                f"loss={loss_value:.4f} <= {self.threshold:.4f} "
                f"at global_step={state.global_step}."
            )
            control.should_training_stop = True
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one epoch of SFT on UltraChat and push to Hugging Face Hub.",
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--instruct-template-model-id",
        type=str,
        default=DEFAULT_INSTRUCT_TEMPLATE_MODEL_ID,
        help="Model whose tokenizer chat template will be used if base tokenizer lacks one.",
    )
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--hub-model-id", type=str, default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--output-dir", type=str, default="outputs/ultrachat-1b-sft")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-acc-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--logging-steps", type=int, default=5)
    return parser.parse_args()


def resolve_train_split(dataset_dict: DatasetDict) -> Dataset:
    for split_name in ("train_sft", "train"):
        if split_name in dataset_dict:
            return dataset_dict[split_name]

    for split_name in dataset_dict.keys():
        if "train" in split_name:
            return dataset_dict[split_name]

    raise ValueError(
        "Could not find a train split in dataset. "
        f"Available splits: {list(dataset_dict.keys())}"
    )


def ensure_chat_template(
    tokenizer: AutoTokenizer,
    template_source_model_id: str,
) -> AutoTokenizer:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    template_tok = AutoTokenizer.from_pretrained(
        template_source_model_id,
        use_fast=True,
    )
    template = getattr(template_tok, "chat_template", None)
    if not template:
        raise ValueError(
            "No chat template found on template source tokenizer: "
            f"{template_source_model_id}"
        )

    tokenizer.chat_template = template
    print(
        "[train_sft_ultrachat_1b] base tokenizer has no chat template; "
        f"using template from {template_source_model_id}"
    )
    return tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer = ensure_chat_template(tokenizer, args.instruct_template_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float32
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            use_bf16 = True
        else:
            dtype = torch.float16
            use_fp16 = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    ds = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    if not isinstance(ds, DatasetDict):
        raise ValueError(
            f"Expected DatasetDict from {args.dataset_id}, got {type(ds)} instead."
        )

    train_raw = resolve_train_split(ds)

    def _format_chat(example: Dict[str, Any]) -> Dict[str, str]:
        messages = example.get("messages")
        if not isinstance(messages, list):
            raise ValueError(
                "Expected `messages` to be a list in UltraChat examples. "
                f"Got type={type(messages)}."
            )
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_text = train_raw.map(
        _format_chat,
        remove_columns=train_raw.column_names,
        desc="Applying instruct chat template",
    )

    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
        )

    train_tokenized = train_text.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training data",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1.0,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        do_eval=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_strategy="end",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[StopOnLossThresholdCallback(threshold=0.5)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    trainer.push_to_hub(
        commit_message="SFT Llama-3.2-1B on UltraChat 200k (1 epoch)"
    )

    print(f"[train_sft_ultrachat_1b] Done. Pushed model to {args.hub_model_id}")


if __name__ == "__main__":
    main()
