
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
import contextlib, torch, traceback, sys
import os

def dump_cuda_memory(path=None):
    try:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        if path:
            # Experimental/private API: only if available in your torch
            torch.cuda.memory._dump_snapshot(path)  # noqa: SLF001
            print(f"Wrote CUDA memory snapshot to {path}")
    except Exception:
        traceback.print_exc()
def right_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
def trace_handler(logdir="./log/profile"):
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK") or "0"
    return torch.profiler.tensorboard_trace_handler(
        logdir,
        worker_name=f"rank{rank}",   # avoid DDP/DeepSpeed collisions
        use_gzip=True,
    )



## SAFETY PROMPTING FORMATTING

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end

def format_prompt(
    input: str | list[str],  # pylint: disable=redefined-builtin
    eos_token: str,
    tokenizer=None,
) -> str:
    if isinstance(input, str):
        input = [input]
    elif not isinstance(input, list):
        raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

    if len(input) % 2 != 1:
        raise ValueError(
            'The length of `input` must be odd, while `input` must end at the user question.',
        )

    # Auto-switch to "open-instruct" style if the tokenizer supports the special tokens.
    # Expected format:
    # <|user|>\n...user...\n<|assistant|>\n...assistant...<eos>
    use_open_instruct = False
    if tokenizer is not None:
        try:
            special = set(getattr(tokenizer, "all_special_tokens", []) or [])
            # Some tokenizers only list these in additional_special_tokens
            special |= set(getattr(tokenizer, "additional_special_tokens", []) or [])
            use_open_instruct = ("<|user|>" in special) and ("<|assistant|>" in special)
        except Exception:
            use_open_instruct = False

    buffer: list[str] = []
    if use_open_instruct:
        user_tok = "<|user|>"
        assistant_tok = "<|assistant|>"
        for i, line in enumerate(input):
            if i % 2 == 0:
                buffer.append(f"{user_tok}\n{line}\n{assistant_tok}\n")
            else:
                # Separate assistant turns with EOS; keep a trailing newline for cleanliness.
                buffer.append(f"{line}{eos_token}\n")
        return "".join(buffer)

    # Legacy prompt format (kept for backwards compatibility)
    buffer = [PROMPT_BEGIN]
    for i, line in enumerate(input):
        if i % 2 == 0:
            buffer.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
        else:
            buffer.extend((line, eos_token))
    return "".join(buffer)

import math

def ndcg_at_k(rank: int, k: int = 10) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    # IDCG for a single relevant item is 1 / log2(1 + 1) == 1
    return 1.0 / math.log2(rank + 1)

def compute_passage_lengths_from_tokens(input_ids, tok):
    """Compute actual passage lengths from tokenized input_ids"""
    # input_ids shape: (N*G, L) where G = 1 + num_negatives
    # We need to compute the length of the second part (passage) for each pair
    lengths = []
    for i in range(input_ids.shape[0]):
        tokens = input_ids[i]
        
        
        # Find the separator token (usually [SEP] or equivalent)
        sep_token_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        
        # Find positions of separator tokens
        mask = (tokens == sep_token_id)
        first_pos = mask.argmax(axis=1)
        second_pos = (mask.cumsum(axis=1) == 2).argmax(axis=1)
        
        if len(first_pos) != tokens.shape[0] or len(second_pos) != tokens.shape[0]:
            raise ValueError("Could not find two separator tokens in the input sequence.")
    
        passage_length = second_pos - first_pos
        lengths += list(passage_length)
    return lengths


def freeze_for_mc(model, *, unfreeze_last_n_layers: int = 0, train_classifier: bool = True):
    """
    Freeze all params, then optionally unfreeze:
      - classifier head
      - last N transformer layers of the base encoder
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze classifier head (HF uses different names across models, so be flexible)
    if train_classifier:
        for name, p in model.named_parameters():
            # Common head names in MultipleChoice models
            if any(k in name for k in ["classifier", "score", "out_proj"]):
                p.requires_grad = True

    # 3) find the base encoder module inside the MC wrapper
    base = getattr(model, "roberta", None) or getattr(model, "bert", None) or getattr(model, "deberta", None) \
           or getattr(model, "electra", None) or getattr(model, "backbone", None) or getattr(model, "model", None)

    if base is None:
        # As a fallback, try model.base_model (HF convention)
        base = getattr(model, "base_model", None)

    if base is None:
        raise ValueError("Couldn't locate the base encoder module inside the MultipleChoice model.")

    # 4) locate the transformer layer stack
    # RoBERTa/BERT: base.encoder.layer
    # DeBERTa: base.encoder.layer (often)
    layers = None
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        layers = base.encoder.layer
    elif hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
        layers = base.encoder.layers
    elif hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        layers = base.transformer.layer
    elif hasattr(base, "layers"):
        layers = base.layers

    if layers is None:
        raise ValueError("Couldn't locate transformer layers (encoder.layer / encoder.layers / etc.).")

    # 5) unfreeze last N layers
    if unfreeze_last_n_layers > 0:
        for layer in list(layers)[-unfreeze_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    return model
