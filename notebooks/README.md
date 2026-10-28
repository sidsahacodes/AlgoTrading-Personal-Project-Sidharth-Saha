# `notebooks/` – exploration

Put exploratory notebooks here; keep reusable logic in `src/momentum_rebalancer/` (notebooks should import it, not copy it).
Saved outputs are stripped on commit (`pre-commit install`, `nbstripout --install`), so a notebook in git has no charts until you run it.

| Notebook | What it shows |
|---|---|
| [`01_walk_forward_explained.ipynb`](01_walk_forward_explained.ipynb) | Train, validation and out-of-sample test explained in plain English, drawn from **your own** `mr walkforward --save` results: the fold timeline, what was chosen, validation vs test, and the stitched out-of-sample record |

**To use it:** run `mr walkforward --save`, open the notebook (in VS Code or JupyterLab), choose the project's `.venv` as the kernel, and **Run All**. It finds the newest `walkforward_*.csv` files in `data/results/`, in `$MR_DATA_ROOT/results/`, or in a sibling `mr-research-data/results/` folder; set `RESULTS_DIR` in its first code cell to point elsewhere. A longer written explanation is in [`../docs/VALIDATION.md`](../docs/VALIDATION.md).

## Ideas for further notebooks

| Idea | Purpose |
|---|---|
| Cluster diagnostics | Cluster sizes, silhouette scores, feature scales (see limitation A4) |
| Stress windows | The strategy against SPY through 2020 and 2022 (or `scripts/stress_windows.py`) |
| Regime review | Monthly regime indicators over time |

Earlier notebooks are archived in [`../old/`](../old/).
