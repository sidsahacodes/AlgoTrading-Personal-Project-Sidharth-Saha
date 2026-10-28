# `scripts/` – repeatable jobs outside the `mr` CLI

| Script | Purpose | Needs |
|---|---|---|
| `stress_windows.py` | Strategy vs SPY on the named windows in `configs/stress_windows.yaml` (2020 crash, 2020 rebound, 2022, full sample), net of costs | Cached modeling data, clusters and monthly returns (`mr pipeline`) |

```bash
python scripts/stress_windows.py
python scripts/stress_windows.py --regime rebound
python scripts/stress_windows.py --top-n 5 --momentum return_6m
python scripts/stress_windows.py --cost-bps 20
python scripts/stress_windows.py --save      # also writes data/results/stress_windows_*.csv
```

The backtest behind it models HRP weights, the exposure scalar, the trade band, the size cap and costs, and applies the active regime's settings through every window. It is a scenario, not a trigger rule. Read [`../docs/VALIDATION.md`](../docs/VALIDATION.md) and [`../docs/REGIME_PLAYBOOK.md`](../docs/REGIME_PLAYBOOK.md) section 5.
