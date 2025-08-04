from transformers import TrainerCallback
from copy import deepcopy
import random


class TrainSetEvalCallback(TrainerCallback):
    def __init__(self, trainer, sample_size: int | None = 5000, prefix="train"):
        super().__init__()
        self.trainer = trainer
        self.sample_size = sample_size
        self.prefix = prefix

    def on_epoch_end(self, args, state, control, **kwargs):
        if not control.should_evaluate:          # keep default val-split eval logic
            return control                       # unchanged control object

        ds = self.trainer.train_dataset
        if self.sample_size and self.sample_size < len(ds):
            import random
            idx = random.sample(range(len(ds)), self.sample_size)
            ds = ds.select(idx)                  # fast “mini-train” slice

        self.trainer.evaluate(
            eval_dataset=ds,
            metric_key_prefix=self.prefix       # results appear as train/…
        )
        return control

