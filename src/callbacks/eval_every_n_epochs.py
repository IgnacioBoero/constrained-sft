from __future__ import annotations

from transformers import TrainerCallback


class EvalEveryNEpochsCallback(TrainerCallback):
    """
    Gate evaluation that is triggered at epoch boundaries.

    Use with TrainingArguments(eval_strategy="epoch"). This callback will skip
    evaluation unless the (1-indexed) epoch number is divisible by N.
    """

    def __init__(self, every_n_epochs: int):
        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch is 0-index-ish float; at epoch end it's typically an int-ish value.
        if state.epoch is None:
            return control
        epoch_1_indexed = int(round(state.epoch))
        if epoch_1_indexed < 1:
            epoch_1_indexed = 1

        if (epoch_1_indexed % self.every_n_epochs) != 0:
            control.should_evaluate = False
        return control

