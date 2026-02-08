from transformers import TrainerCallback

class ConstraintSlackWandbCallback(TrainerCallback):
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = self.trainer
        if trainer is None:
            return

        cfg = trainer.custom_cfg
        if not (cfg.train.use_wandb and trainer.is_world_process_zero()):
            return

        if not hasattr(trainer, "_last_constraint_slacks"):
            return

        import wandb
        if wandb.run is None:
            return
        
        prefix = getattr(trainer, "_current_eval_prefix", "eval")

        slacks = trainer._last_constraint_slacks.numpy()
        idxs_constraint = trainer._last_constraint_indexes.numpy()
        safety_labels = getattr(trainer, "_last_constraint_safety_labels", None)
        if safety_labels is not None:
            safety_labels = safety_labels.numpy()
        
        objective_ratios = trainer._last_objective_ratios.numpy()
        idxs_objective = trainer._last_objective_indexes 

        table_constraints = wandb.Table(columns=["index", "slack", "safety_label"])
        if safety_labels is None:
            for idx, slack in zip(idxs_constraint.tolist(), slacks.tolist()):
                table_constraints.add_data(idx, slack, None)
        else:
            for idx, slack, lbl in zip(idxs_constraint.tolist(), slacks.tolist(), safety_labels.tolist()):
                table_constraints.add_data(idx, slack, bool(lbl))
        
        table_objective = wandb.Table(columns=["index", "objective_log_ratio"])
        for idx, ratio in zip(idxs_objective.tolist(), objective_ratios.tolist()):
            table_objective.add_data(idx, ratio)
        
        dual_vars = trainer.dual_vars.detach().cpu().numpy()
        dual_vars = dual_vars[idxs_constraint]
        table_dual_vars = wandb.Table(columns=["index", "dual_var"])
        for idx, dual_var in zip(idxs_constraint.tolist(), dual_vars.tolist()):
            table_dual_vars.add_data(idx, dual_var)
        epoch = state.epoch
        if epoch is None:
            epoch = -1
        else:
            epoch = int(epoch)

        wandb.log(
            {
                f"{prefix}_constraint_slacks_table_epoch_{epoch}": table_constraints,
                f"{prefix}_constraint_slacks_histogram_epoch_{epoch}": wandb.Histogram(slacks),
                f"{prefix}_objective_log_ratios_table_epoch_{epoch}": table_objective,
                f"{prefix}_objective_log_ratios_histogram_epoch_{epoch}": wandb.Histogram(objective_ratios),
                f"{prefix}_dual_vars_table_epoch_{epoch}": table_dual_vars,
            },
            step=state.global_step,
        )
        print(f"Logged {prefix} constraint slacks and objective ratios to wandb.")
        return control
