from transformers import TrainerCallback
from copy import deepcopy
import random


from transformers import TrainerCallback

class TrainSetEvalCallback(TrainerCallback):
    def __init__(self, trainer, sample_size: int | None = None, prefix="train"):
        super().__init__()
        self.trainer = trainer
        self.sample_size = sample_size
        self.prefix = prefix
        self._inside = False

    def on_evaluate(self, args, state, control, **kwargs):
        if self._inside:
            return control
        self._inside = True

        ds = self.trainer.train_dataset
        if self.sample_size and self.sample_size < len(ds):
            import random
            idx = random.sample(range(len(ds)), self.sample_size)
            ds = ds.select(idx)

        # backup eval slacks/indexes (so outer eval can still see them)
        prev_slacks = getattr(self.trainer, "_last_constraint_slacks", None)
        prev_indexes = getattr(self.trainer, "_last_constraint_indexes", None)
        prev_labels = getattr(self.trainer, "_last_constraint_safety_labels", None)
        prev_prefix = getattr(self.trainer, "_current_eval_prefix", "eval")

        # tag as train and run nested eval
        self.trainer._current_eval_prefix = self.prefix  # "train"
        self.trainer.evaluate(
            eval_dataset=ds,
            metric_key_prefix=self.prefix,
        )

        # restore eval state for the outer on_evaluate
        if prev_slacks is not None:
            self.trainer._last_constraint_slacks = prev_slacks
        if prev_indexes is not None:
            self.trainer._last_constraint_indexes = prev_indexes
        if prev_labels is not None:
            self.trainer._last_constraint_safety_labels = prev_labels
        self.trainer._current_eval_prefix = prev_prefix

        self._inside = False
        return control


