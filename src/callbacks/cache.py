# Hugging Face Trainer callback example
from transformers import TrainerCallback
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

class SyncEmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps=1):
        self.every_n_steps = every_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            get_accelerator().empty_cache()