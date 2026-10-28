# `notebooks/` – exploration

Put exploratory notebooks here; keep reusable logic in `src/momentum_rebalancer/` (notebooks should import it, not copy it).

Suggested names, mirroring the numbered convention used in sibling project repos:

| Notebook | Purpose |
|---|---|
| `00_project_orientation.ipynb` | Walk through one month end to end on cached data |
| `01_cluster_diagnostics.ipynb` | Cluster sizes, silhouette scores, feature scales (see limitation A4) |
| `02_stress_windows.ipynb` | Strategy vs SPY in the 2020 and 2022 windows (or `scripts/stress_windows.py`) |
| `03_regime_review.ipynb` | Monthly regime indicators over time |

Outputs are stripped on commit (`pre-commit install`, `nbstripout --install`). Earlier notebooks are archived in [`../old/`](../old/).
