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
from pathlib import Path
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
    eval_at_end = getattr(cfg.train, "eval_at_end", False)
    if eval_at_end:
        trainer._current_eval_prefix = "eval"
        trainer.evaluate(metric_key_prefix="eval")
    eval_train_at_end = getattr(cfg.train, "eval_train_at_end", False)
    if eval_train_at_end:
        ds = trainer.train_dataset
        trainer._current_eval_prefix = "train"
        trainer.evaluate(
            eval_dataset=ds,
            metric_key_prefix="train",
        )


    push_adapters_to_wandb = getattr(cfg.train, "push_adapters_to_wandb", False)
    should_save_adapters = bool(getattr(cfg.train, "save_model", False) or push_adapters_to_wandb)
    adapters_out_dir = Path(cfg.train.output_dir) / "lora_adapters"

    if should_save_adapters and is_global_main_process():
        adapters_out_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(adapters_out_dir)
            if trainer.tokenizer is not None:
                trainer.tokenizer.save_pretrained(adapters_out_dir)
        else:
            print("[save-model] Model does not support save_pretrained; skipping adapter save.")

    if push_adapters_to_wandb and is_global_main_process():
        if not cfg.train.use_wandb:
            print("[wandb] push_adapters_to_wandb=true but train.use_wandb=false; skipping artifact upload.")
        elif wandb.run is None:
            print("[wandb] push_adapters_to_wandb=true but wandb.run is None; skipping artifact upload.")
        else:
            if not adapters_out_dir.exists():
                print(f"[wandb] Adapter directory not found at {adapters_out_dir}; skipping artifact upload.")
            else:
                artifact_name = f"{wandb.run.id}-lora_adapters"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="lora_adapters",
                    metadata={
                        "output_dir": str(cfg.train.output_dir),
                        "exp_name": str(getattr(cfg.exp, "name", "")),
                        "model_name": str(getattr(cfg.exp, "model_name", "")),
                        "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
                    },
                )
                artifact.add_dir(str(adapters_out_dir))
                wandb.log_artifact(artifact, aliases=["latest"])
                try:
                    artifact.wait()
                except Exception as exc:
                    print(f"[wandb] Artifact upload did not complete cleanly: {exc}")

    # Finish wandb run
    if cfg.train.use_wandb and is_global_main_process():
        wandb.finish()

if __name__ == "__main__":
    main()
