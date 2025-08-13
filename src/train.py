# src/run.py
from callbacks.trainer_eval import TrainSetEvalCallback
from callbacks.profiler_callback import ProfilerCallback
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed
from experiments.registry import EXPERIMENTS
from utils import dump_cuda_memory, trace_handler

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
    preprocess = exp.preprocessing_fn(tok, cfg)
    cols = train_ds.column_names

    train_ds = train_ds.map(preprocess, remove_columns=cols)
    eval_ds = eval_ds.map(preprocess, remove_columns=cols) 
    
    collator = exp.get_collator(tok)
    metric_fn  = exp.compute_metrics(tok, cfg)
    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        **cfg.train.hf_args,
        include_for_metrics=['inputs','loss'],
        report_to=["wandb"] if cfg.train.use_wandb else [],
        run_name=cfg.train.run_name if cfg.train.use_wandb else None,
        remove_unused_columns=False,
        logging_steps= 1,
    )

    trainer = exp.get_trainer_class()(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, tokenizer=tok,
        compute_metrics=metric_fn, experiment=exp
    )
    if cfg.train.eval_on_train:
        trainer.add_callback(TrainSetEvalCallback(trainer))


    
    if cfg.train.do_train:
        # Configure profiler for GPU memory tracking
        profiler_enabled = getattr(cfg.train, 'enable_profiler', False)
        
        if profiler_enabled and torch.cuda.is_available():
            print("Profiler is enabled. Tracking GPU memory usage.")
            schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=9999)

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=trace_handler("./log/profile"),
                record_shapes=True,
                profile_memory=True,
                with_flops=True
            ) as prof:
                # Add profiler callback to step the profiler
                trainer.add_callback(ProfilerCallback(prof))
                trainer.train()
                # try:
                #     if cfg.train.do_initial_eval:
                #         trainer.evaluate(metric_key_prefix="eval")
                #     trainer.train()
                # except RuntimeError as e:
                #     if "CUDA out of memory" in str(e):
                #         print("CUDA out of memory error encountered. Dumping CUDA memory snapshot.")
                #         dump_cuda_memory(path="./log/cuda_mem_snapshot.pickle")
                #     raise
        else:
            if cfg.train.do_initial_eval:
                trainer.evaluate(metric_key_prefix="eval")
            trainer.train()

    # Finish wandb run
    if cfg.train.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
