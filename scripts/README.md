# Scripts layout

Run Python from the **repository root** with `export PYTHONPATH=.` unless a script cds there for you.

| Directory | Contents |
|-----------|----------|
| **`paper_eval/`** | Safety + reranking **evaluation** and W&B post-processing tied to the paper metrics (Beaver, AlpacaEval batch, refusal / KL / CVaR, reranker checkpoint sweeps). |
| **`when2call/`** | Distributed training launcher for **`paper_experiments/function_calling`**. |
| **`datasets/`** | Dataset construction utilities (e.g. SAFE-ALPACA-4). |
| **`release/`** | Export W&B LoRA artifacts to Hugging Face upload layout. |

Full mapping to appendix tables: **`docs/tooling_paper_aligned.md`**.
