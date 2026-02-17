import wandb
from pathlib import Path

entity = "alelab"
project = "SAFE-long1k"
run_id = "5dwh4stq"#n8oe8pxk"#"acsip14q"#"694jyduz"#"ppocpwoy"#"h4h0gxko" #"83q43pt7"#"bqp0t0qs"
    
api = wandb.Api()
artifact = api.artifact(f"{entity}/{project}/{run_id}-alpaca_eval_outputs-vllm:latest")
out_dir = Path("./wandb_downloads")
out_dir.mkdir(parents=True, exist_ok=True)
local_dir = Path(artifact.download(root=str(out_dir)))
print("Downloaded to:", local_dir)
for p in local_dir.glob("*.json"):
    print("JSON:", p)