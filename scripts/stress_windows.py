"""
scripts/stress_windows.py — evaluate the strategy on named stress windows.

Usage:
    python scripts/stress_windows.py
    python scripts/stress_windows.py --top-n 5 --momentum return_6m
    python scripts/stress_windows.py --regime rebound
    python scripts/stress_windows.py --save          # also writes data/results/*.csv

Requires cached modeling data and clusters (`mr pipeline`). The backtest is
net of costs; read docs/VALIDATION.md and docs/KNOWN_LIMITATIONS.md before drawing conclusions.
"""

from __future__ import annotations

import argparse
import os

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime")
    ap.add_argument("--top-n", type=int)
    ap.add_argument("--momentum")
    ap.add_argument("--cost-bps", type=float, help="override the cost assumption")
    ap.add_argument("--save", action="store_true", help="write the table to data/results/")
    args = ap.parse_args()
    if args.regime:
        os.environ["MR_REGIME"] = args.regime

    from momentum_rebalancer import config as C
    from momentum_rebalancer.strategy.backtest import run_backtest
    from momentum_rebalancer.strategy.regime import window_metrics

    windows = yaml.safe_load((C.CONFIG_DIR / "stress_windows.yaml").read_text())["windows"]
    windows = {name: tuple(span) for name, span in windows.items()}
    windows["Full sample"] = (None, None)

    kwargs = {}
    if args.top_n:
        kwargs["top_n"] = args.top_n
    if args.momentum:
        kwargs["momentum_feature"] = args.momentum
    if args.cost_bps is not None:
        kwargs["cost_bps"] = args.cost_bps

    res = run_backtest(**kwargs)
    table = window_metrics(res.monthly_returns, res.benchmark_returns, windows)
    print(f"\nregime={C.REGIME}  {kwargs or ''}\n")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    if args.save:
        import pandas as pd
        path = C.RESULTS_DIR / f"stress_windows_{C.REGIME}_{pd.Timestamp.today():%Y-%m-%d}.csv"
        table.to_csv(path)
        print(f"\nsaved -> {path}")
    print(f"\nNet of {kwargs.get('cost_bps', C.COST_BPS):.0f} bps costs; HRP weights, exposure scalar, trade band and size cap included.")
    print("The active regime's settings are applied through every window (a scenario, not a trigger rule).")
    print("Short windows: judge cumulative return and drawdown only.")


if __name__ == "__main__":
    main()
