# Hugging Face Trainer callback example
from transformers import TrainerCallback
import torch
import torch.distributed as dist
try:
    # Optional: only available in some environments.
    from deepspeed.accelerator import get_accelerator  # type: ignore
except Exception:  # pragma: no cover
    get_accelerator = None

class SyncEmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps=1):
        self.every_n_steps = every_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if get_accelerator is not None:
                get_accelerator().empty_cache()
            else:
                # Fallback when DeepSpeed isn't installed.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()