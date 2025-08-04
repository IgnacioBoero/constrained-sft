# src/run.py
from callbacks.trainer_eval import TrainSetEvalCallback
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed
from experiments.registry import EXPERIMENTS


@hydra.main(config_path="../configs", config_name="train/default", version_base=None)
def main(cfg: DictConfig):
    
    set_seed(cfg.train.seed)
    
    # Initialize wandb if enabled
    if cfg.train.use_wandb:
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.run_name or f"{cfg.exp.name}-{cfg.model.name}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    ExpCls = EXPERIMENTS[cfg.exp.name]
    exp = ExpCls()

    # Model
    model,tok = exp.load_model_and_tok(cfg)

    # Data
    train_ds, eval_ds = exp.load_datasets(cfg)
    preprocess = exp.preprocessing_fn(tok, model)
    cols = train_ds.column_names

    train_ds = train_ds.map(preprocess, remove_columns=cols)
    eval_ds  = eval_ds.map(preprocess, remove_columns=cols)
    collator = exp.get_collator(tok)

    # Loss and Metrics
    metric_fn  = exp.compute_metrics(tok, cfg)

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        **cfg.train.hf_args,
        include_for_metrics=['inputs','loss'],
        report_to=["wandb"] if cfg.train.use_wandb else [],
        run_name=cfg.train.run_name if cfg.train.use_wandb else None,
        remove_unused_columns=False,
    )

    trainer = exp.get_trainer_class()(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, tokenizer=tok,
        compute_metrics=metric_fn, experiment=exp
    )
    trainer.add_callback(TrainSetEvalCallback(trainer))
    if cfg.train.do_initial_eval: 
        trainer.evaluate(metric_key_prefix="eval")
        trainer.evaluate(train_ds, metric_key_prefix="train")
    if cfg.train.do_train: trainer.train()

    # Finish wandb run
    if cfg.train.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
