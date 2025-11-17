
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
) -> str:
    if isinstance(input, str):
        input = [input]
    elif not isinstance(input, list):
        raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

    if len(input) % 2 != 1:
        raise ValueError(
            'The length of `input` must be odd, while `input` must end at the user question.',
        )

    buffer = [PROMPT_BEGIN]
    for i, line in enumerate(input):
        if i % 2 == 0:
            # User input
            buffer.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
        else:
            # Assistant response
            buffer.extend((line, eos_token))

    return ''.join(buffer)