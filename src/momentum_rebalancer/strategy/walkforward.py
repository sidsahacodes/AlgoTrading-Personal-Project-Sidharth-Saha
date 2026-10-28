"""
strategy/walkforward.py — out-of-sample evaluation of the strategy's hyperparameters.

Nothing in this strategy is fitted on prices: clusters are refit independently each month and
HRP uses past returns only. What *is* chosen using data are the hyperparameters in the grid
(momentum lookback, names per cluster, number of clusters). Choosing them on the full sample and
then reporting the same sample's performance would be look-ahead. Instead, for each fold:

    |-------- train --------|-- validation --|-- embargo --|--- test ---|
    (reported only)          (picks the config)              (never used to choose)

1. Every config is simulated month by month (point-in-time, net of costs).
2. The config with the best *validation* objective is chosen using only months before the test block.
3. That frozen config's *test* months are recorded.
4. The blocks step forward; the stitched test months are the out-of-sample record.

The simulation of a config over the whole sample is reused for every fold. That is valid because
each month's decisions depend only on data up to that month (tested), so slicing a long run gives
the same numbers as running each block separately, apart from the position carried between months.

Caveats: with ~12 years of data there are only a handful of folds; the more configs in the grid,
the more likely the validation winner is luck; and costs are an assumption (`cost_bps`).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from momentum_rebalancer.strategy.backtest import compute_metrics
from momentum_rebalancer.strategy.simulate import SimConfig, simulate

logger = logging.getLogger(__name__)


# ── Folds ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Fold:
    k:     int
    train: tuple[pd.Timestamp, pd.Timestamp]
    val:   tuple[pd.Timestamp, pd.Timestamp]
    test:  tuple[pd.Timestamp, pd.Timestamp]


def make_folds(
    dates: pd.DatetimeIndex,
    train_months: int,
    val_months: int,
    test_months: int,
    step_months: int,
    embargo_months: int = 0,
) -> list[Fold]:
    """Expanding-window folds over consecutive signal months (inclusive date ranges)."""
    if step_months < test_months:
        raise ValueError("step_months must be >= test_months so test blocks do not overlap")
    n, folds, a, k = len(dates), [], train_months, 0
    while True:
        v_end = a + val_months
        t_start = v_end + embargo_months
        t_end = t_start + test_months
        if t_end > n:
            break
        folds.append(Fold(
            k=k,
            train=(dates[0], dates[a - 1]),
            val=(dates[a], dates[v_end - 1]),
            test=(dates[t_start], dates[t_end - 1]),
        ))
        a += step_months
        k += 1
    return folds


# ── Objective ─────────────────────────────────────────────────────────────────

def objective_value(r: pd.Series, objective: str = "sharpe") -> float:
    """Net Sharpe (annualised, rf = 0) or cumulative return; -inf when undefined."""
    r = r.dropna()
    if len(r) < 2:
        return -np.inf
    if objective == "cumulative_return":
        return float((1 + r).prod() - 1)
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(12)) if sd > 0 else -np.inf


def _cum(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() - 1) if len(r) else float("nan")


# ── Grid ──────────────────────────────────────────────────────────────────────

def config_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def cluster_kwargs(method: str, k: int) -> dict:
    if method in ("kmeans", "pca_kmeans"):
        return {"n_clusters": k}
    if method == "gmm":
        return {"n_components": k}
    raise ValueError(f"Walk-forward over n_clusters needs kmeans, gmm or pca_kmeans, not '{method}'.")


@dataclass
class WalkForwardResult:
    folds:            pd.DataFrame                    # one row per fold: chosen config, train/val/test metrics
    oos_returns:      pd.Series                       # chosen configs, test months only, labelled by month earned
    baseline_returns: pd.Series                       # fixed baseline config over the same months
    summary:          dict
    selection_counts: pd.Series
    n_configs:        int
    diagnostics:      dict = field(default_factory=dict)


def run_walk_forward(
    modeling: pd.DataFrame,
    returns_wide: pd.DataFrame,
    wf: dict,
    base: SimConfig,
    baseline: dict,
    cluster_method: str,
    feature_cols: list[str],
    cluster_fn: Callable[[pd.DataFrame, list[str], str, dict], pd.DataFrame] | None = None,
    benchmark_loader: Callable[[pd.DatetimeIndex], pd.Series] | None = None,
) -> WalkForwardResult:
    """
    Parameters
    ----------
    modeling, returns_wide : cached features and the raw monthly return panel.
    wf        : parsed configs/walk_forward.yaml.
    base      : SimConfig carrying everything that is *not* in the grid (optimiser, costs, ...).
    baseline  : the fixed configuration to compare against ({momentum_feature, top_n_per_cluster, n_clusters}).
    cluster_fn: (data, feature_cols, method, kwargs) -> clustered frame. Injectable for tests.
    benchmark_loader : optional (index of months earned) -> benchmark monthly returns.
    """
    from momentum_rebalancer.clustering.methods import cluster as _cluster
    cluster_fn = cluster_fn or (lambda d, cols, m, kw: _cluster(d, cols, method=m, **kw))

    sp = wf["splits"]
    objective, min_months = wf.get("objective", "sharpe"), int(wf.get("min_months", 12))
    configs = config_grid(wf["grid"])
    if baseline not in [{k: c[k] for k in baseline} for c in configs]:
        logger.info("Baseline config is not in the grid; it is simulated separately.")

    # 1. cluster once per k (clusters are fit independently per month: no information crosses months)
    clustered_by_k: dict[int, pd.DataFrame] = {}
    for k in sorted({c["n_clusters"] for c in configs} | {baseline["n_clusters"]}):
        logger.info("Clustering with k=%d …", k)
        clustered_by_k[k] = cluster_fn(modeling, feature_cols, cluster_method, cluster_kwargs(cluster_method, k))

    # 2. simulate every config over the whole sample
    def run(cfg: dict):
        sc = SimConfig(
            momentum_feature=cfg["momentum_feature"], top_n=cfg["top_n_per_cluster"],
            portfolio_size=cfg["top_n_per_cluster"] * cfg["n_clusters"],
            optimiser=base.optimiser, hrp_lookback_months=base.hrp_lookback_months,
            exposure_scalar=base.exposure_scalar, min_trade_pct=base.min_trade_pct,
            cost_bps=base.cost_bps, delisting_return=base.delisting_return,
        )
        return simulate(clustered_by_k[cfg["n_clusters"]], returns_wide, sc)

    sims, nets = {}, {}
    for i, cfg in enumerate(configs, 1):
        logger.info("Simulating config %d/%d: %s", i, len(configs), cfg)
        sims[i - 1] = run(cfg)
        nets[i - 1] = sims[i - 1].net
    base_sim = run(baseline)
    base_net = base_sim.net

    dates = base_net.index
    earned_in = base_sim.earned_in
    folds = make_folds(dates, int(sp["train_months"]), int(sp["val_months"]), int(sp["test_months"]),
                       int(sp["step_months"]), int(sp["embargo_months"]))
    if not folds:
        raise ValueError(
            f"Not enough months ({len(dates)}) for one fold: need train {sp['train_months']} + validation "
            f"{sp['val_months']} + embargo {sp['embargo_months']} + test {sp['test_months']}. "
            f"Increase lookback_years or shrink the blocks in configs/walk_forward.yaml."
        )

    # 3. choose on validation, record test
    rows, oos_parts, base_parts = [], [], []
    for f in folds:
        tr = lambda s: s.loc[f.train[0]:f.train[1]]
        va = lambda s: s.loc[f.val[0]:f.val[1]]
        te = lambda s: s.loc[f.test[0]:f.test[1]]

        scores = []
        for i, cfg in enumerate(configs):
            v = va(nets[i])
            eligible = v.notna().sum() >= min_months
            scores.append(objective_value(v, objective) if eligible else -np.inf)
        best = int(np.argmax(scores))                       # first best in grid order on ties
        chosen = configs[best]
        n = nets[best]

        rows.append({
            "fold": f.k,
            "train": f"{f.train[0]:%Y-%m}..{f.train[1]:%Y-%m}",
            "validation": f"{f.val[0]:%Y-%m}..{f.val[1]:%Y-%m}",
            "test": f"{f.test[0]:%Y-%m}..{f.test[1]:%Y-%m}",
            **{f"chosen_{k}": v for k, v in chosen.items()},
            "train_sharpe": objective_value(tr(n)),
            "val_sharpe": objective_value(va(n)),
            "test_sharpe": objective_value(te(n)),
            "test_return": _cum(te(n)),
            "baseline_test_sharpe": objective_value(te(base_net)),
            "baseline_test_return": _cum(te(base_net)),
        })
        oos_parts.append(te(n))
        base_parts.append(te(base_net))

    oos_sig = pd.concat(oos_parts)
    base_sig = pd.concat(base_parts)
    oos = oos_sig.set_axis(pd.DatetimeIndex(earned_in.loc[oos_sig.index].values))
    base_oos = base_sig.set_axis(oos.index)

    benchmark = benchmark_loader(oos.index) if benchmark_loader else None
    fold_df = pd.DataFrame(rows).set_index("fold")
    chosen_cols = [c for c in fold_df.columns if c.startswith("chosen_")]
    counts = fold_df.groupby(chosen_cols).size().sort_values(ascending=False)

    def strip(m):  # drop the heavy series from the metrics dict
        return {k: v for k, v in m.items() if not hasattr(v, "values")}

    summary = {
        "oos_months": len(oos),
        "oos_period": f"{oos.index.min():%Y-%m} .. {oos.index.max():%Y-%m}",
        "walk_forward": strip(compute_metrics(oos, benchmark)),
        "fixed_baseline": strip(compute_metrics(base_oos, benchmark)),
        "mean_val_sharpe_of_chosen": float(fold_df["val_sharpe"].replace(-np.inf, np.nan).mean()),
        "mean_test_sharpe_of_chosen": float(fold_df["test_sharpe"].replace(-np.inf, np.nan).mean()),
        "cost_bps": base.cost_bps,
        "n_folds": len(folds),
    }
    return WalkForwardResult(
        folds=fold_df, oos_returns=oos, baseline_returns=base_oos, summary=summary,
        selection_counts=counts, n_configs=len(configs), diagnostics=base_sim.diagnostics,
    )


def format_walk_forward(res: WalkForwardResult) -> str:
    s, wf, fb = res.summary, res.summary["walk_forward"], res.summary["fixed_baseline"]
    pct = lambda x: f"{x * 100:6.1f}%" if x is not None and np.isfinite(x) else "   n/a"
    num = lambda x: f"{x:6.2f}" if x is not None and np.isfinite(x) else "   n/a"
    out = ["", "WALK-FORWARD (out-of-sample, net of costs)", "=" * 72,
           f"Folds: {s['n_folds']}   configs per fold: {res.n_configs}   test months: {s['oos_months']} ({s['oos_period']})",
           f"Cost assumption: {s['cost_bps']:.0f} bps per unit traded", "",
           res.folds.drop(columns=["train", "validation"]).to_string(float_format=lambda x: f"{x:.2f}"), "",
           f"{'':26s}{'walk-forward':>14s}{'fixed baseline':>16s}"]
    for label, key, f in [("Annualised return", "annualised_return", pct), ("Sharpe", "sharpe", num),
                          ("Max drawdown", "max_drawdown", pct), ("Cumulative return", "cumulative_return", pct)]:
        out.append(f"{label:26s}{f(wf.get(key)):>14s}{f(fb.get(key)):>16s}")
    if "benchmark_ann_return" in wf:
        out.append(f"{'SPY annualised return':26s}{pct(wf['benchmark_ann_return']):>14s}")
    out += ["",
            f"Mean Sharpe of the chosen config: validation {num(s['mean_val_sharpe_of_chosen'])}  ->  test {num(s['mean_test_sharpe_of_chosen'])}",
            "(a large drop from validation to test is the signature of selection on noise)", "",
            "Most-chosen configs (folds):", res.selection_counts.to_string(), "",
            "Read with care: few folds, a grid of "
            f"{res.n_configs} configs (the winner of each validation block is partly luck), and costs are assumed."]
    return "\n".join(out)
