from transformers import TrainerCallback

class ConstraintSlackWandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return

        cfg = trainer.custom_cfg.exp
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
        
        objective_ratios = trainer._last_objective_ratios.numpy()
        idxs_objective = trainer.last_objective_indexes 

        table_constraints = wandb.Table(columns=["index", "slack"])
        for idx, slack in zip(idxs_constraint.tolist(), slacks.tolist()):
            table_constraints.add_data(idx, slack)
        
        table_objective = wandb.Table(columns=["index", "objective_log_ratio"])
        for idx, ratio in zip(idxs_objective.tolist(), objective_ratios.tolist()):
            table_objective.add_data(idx, ratio)
        
        dual_vars = trainer.dual_vars.detach().cpu().numpy()
        dual_vars = dual_vars[idxs_constraint]
        table_dual_vars = wandb.Table(columns=["index", "dual_var"])
        for idx, dual_var in zip(idxs_constraint.tolist(), dual_vars.tolist()):
            table_dual_vars.add_data(idx, dual_var)
            
        wandb.log(
            {
                f"{prefix}_constraint_slacks_table": table_constraints,
                f"{prefix}_constraint_slacks_histogram": wandb.Histogram(slacks),
                f"{prefix}_objective_log_ratios_table": table_objective,
                f"{prefix}_objective_log_ratios_histogram": wandb.Histogram(objective_ratios),
                f"{prefix}_dual_vars_table": table_dual_vars,
            },
            step=int(state.epoch),
        )
        return control
