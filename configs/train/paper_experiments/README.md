# Paper experiment configs

Hydra YAMLs aligned with **“Every Sample Counts”** (Appendix setups under `configs/train/paper_experiments/`).

**Layout** (pick the task-first path; each area has its own README):

| Subfolder | Task | Appendix / paper refs |
|-----------|------|------------------------|
| [`safety/`](./safety/) | Instruction following + safety refusal | Appendix D.3, Tables 8–9, Fig. 3 |
| [`function_calling/`](./function_calling/) | When2Call / tool-use preferences | Appendix D (“Function Calling Preferences”), Tables 5–6, Fig. 2 |

Training entry point:

```bash
export PYTHONPATH=.
python src/train.py --config-path train/paper_experiments/<task>/<...> --config-name=<stem>
```

The default [`configs/train/default.yaml`](../default.yaml) points at **`paper_experiments/safety/paper_safety_shared`** for a quick sanity run—override **`--config-path` / `--config-name`** when training function-calling configs.

Alternative (Hydra grouping) launches used historically for When2Call still work:

```bash
python src/train.py train=paper_experiments/function_calling/sweep/dpo
```

---

- **Safety** — sweeps/seeds for constrained vs baseline methods + AlpacaEval / refusal / Beaver-cost evaluation pointers: **[`safety/README.md`](./safety/README.md)**  
- **Function calling** — Llama vs xLAM, sweep vs seeded finals + Table 5–6 mapping: **[`function_calling/README.md`](./function_calling/README.md)**
