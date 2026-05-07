# Paper experiment configs

Hydra YAMLs for **“Every Sample Counts”** under `configs/train/paper_experiments/`. Each subfolder documents appendix tables, YAML layout, and how to run training.

| Subfolder | Task | Appendix / paper refs |
|-----------|------|------------------------|
| [`safety/`](./safety/) | Instruction following + safety refusal | Appendix D.3, Tables 8–9, Fig. 3 |
| [`function_calling/`](./function_calling/) | When2Call–style tool-use preferences | Appendix D (tool / function calling), Tables 5–6, Fig. 2 |
| [`reranker/`](./reranker/) | MS MARCO subset; length-aware \(\Lambda\) + margin constraints | Appendix D.2, Table 7, Tables 12–13, Figs 4 & 14 |

## Training

From repo root:

```bash
export PYTHONPATH=.
python src/train.py --config-path train/paper_experiments/<area>/<subdir> --config-name=<stem>
```

Example:

```bash
python src/train.py --config-path train/paper_experiments/reranker --config-name=sweep_pointwise
```

The default [`configs/train/default.yaml`](../default.yaml) composes **`paper_experiments/safety/paper_safety_shared`** for a minimal smoke run; override **`--config-path`** and **`--config-name`** for other setups.

Legacy Hydra grouping (When2Call sweeps):

```bash
python src/train.py train=paper_experiments/function_calling/sweep/dpo
```

## Per-task docs

| | |
|--|--|
| **Safety** | [`safety/README.md`](./safety/README.md) — sweeps, seeds, AlpacaEval / refusal / Beaver eval |
| **Function calling** | [`function_calling/README.md`](./function_calling/README.md) — Llama vs xLAM, sweeps vs finals, Table 6 \(\leftrightarrow\) `exp.*` |
| **Reranker** | [`reranker/README.md`](./reranker/README.md) — cross-encoder grids, checkpoint batch eval |

**Scripts and eval index:** [`docs/tooling_paper_aligned.md`](../../../docs/tooling_paper_aligned.md)
