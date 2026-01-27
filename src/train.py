# src/train.py
from callbacks.trainer_eval import TrainSetEvalCallback
from callbacks.profiler_callback import ProfilerCallback
from callbacks.cache import SyncEmptyCacheCallback
from callbacks.log_slack import ConstraintSlackWandbCallback
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed
from experiments.base import EXPERIMENTS
from utils import dump_cuda_memory, trace_handler
from transformers.integrations import HfDeepSpeedConfig
import os
import atexit, torch, torch.distributed as dist


def is_global_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"

@hydra.main(config_path="../configs", config_name="train/default", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    print("RANK", os.environ.get("RANK"),
      "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
      "CUDA", torch.cuda.current_device())
    
    # Initialize wandb if enabled
    if cfg.train.use_wandb and is_global_main_process():
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.run_name or f"{cfg.exp.name}-{cfg.model.name}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    ExpCls = EXPERIMENTS[cfg.exp.name]
    exp = ExpCls()
    
    # deepSpeed configuration
    if "deepspeed" in cfg.train.hf_args and cfg.train.hf_args.deepspeed:
        ds_cfg_path = cfg.train.hf_args.deepspeed
        _ = HfDeepSpeedConfig(ds_cfg_path)

    # Model
    model,tok = exp.load_model_and_tok(cfg)

    # Data
    train_ds, eval_ds, complete_ds = exp.load_datasets(cfg)
    preprocess = exp.preprocessing_fn(tok, cfg)
    cols = train_ds.column_names

    train_ds = train_ds.map(preprocess, remove_columns=cols)
    eval_ds = eval_ds.map(preprocess, remove_columns=cols)
    if complete_ds is not None:
        complete_ds = complete_ds.map(preprocess, remove_columns=cols)
    
    collator = exp.get_collator(tok)
    include_for_metrics = []
    if getattr(cfg.train, "include_inputs_for_metrics", False):
        include_for_metrics = ['inputs','loss']
    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        **cfg.train.hf_args,
        include_for_metrics=include_for_metrics,
        report_to=["wandb"] if cfg.train.use_wandb else [],
        run_name=cfg.train.run_name if cfg.train.use_wandb else None,
        remove_unused_columns=False,
        logging_steps= 1,
    )

    trainer = exp.get_trainer_class()(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, tokenizer=tok, experiment=exp,
        custom_cfg=cfg, complete_dataset=complete_ds
    )
    
    trainer._current_eval_prefix = "eval"  # default for normal evals
    log_tables = getattr(cfg.train, "log_tables", False)
    if log_tables:
        trainer.add_callback(ConstraintSlackWandbCallback(trainer))
        
    eval_on_train = getattr(cfg.train, "eval_on_train", False)
    if eval_on_train:
        trainer.add_callback(TrainSetEvalCallback(trainer))
        

    
    sync_empty_cache = getattr(cfg.train, "sync_empty_cache", False)
    if sync_empty_cache:
        trainer.add_callback(SyncEmptyCacheCallback(every_n_steps=1))
    if cfg.train.do_train:  
        # Configure profiler for GPU memory tracking
        profiler_enabled = getattr(cfg.train, 'enable_profiler', False)
        
        if profiler_enabled and torch.cuda.is_available():
            print("Profiler is enabled. Tracking GPU memory usage.")
            schedule = torch.profiler.schedule(wait=cfg.train.profiler.wait, warmup=cfg.train.profiler.warmup, active=cfg.train.profiler.active, repeat=cfg.train.profiler.repeat)

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=trace_handler(cfg.train.profiler.output_dir),
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                # with_stack=True,
                with_modules=True,
            ) as prof:
                trainer.add_callback(ProfilerCallback(prof))
                if cfg.train.do_initial_eval:
                    trainer.evaluate(metric_key_prefix="eval")
                trainer.train()

        else:
            if cfg.train.do_initial_eval:
                trainer.evaluate(metric_key_prefix="eval")
            trainer.train()
    if cfg.train.eval_at_end:
        trainer._current_eval_prefix = "eval"
        trainer.evaluate(metric_key_prefix="eval")
    # Finish wandb run
    if cfg.train.use_wandb and is_global_main_process():
        wandb.finish()

if __name__ == "__main__":
    main()
