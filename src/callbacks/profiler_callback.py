from transformers import TrainerCallback


class ProfilerCallback(TrainerCallback):
    """Callback to step the PyTorch profiler during training."""
    
    def __init__(self, profiler):
        self.profiler = profiler
    
    def on_step_end(self, args, state, control, **kwargs):
        """Step the profiler after each training step."""
        if self.profiler is not None:
            self.profiler.step()