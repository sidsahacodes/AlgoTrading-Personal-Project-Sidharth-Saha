"""
strategy/backtest.py — fully vectorised backtest, no Python loops.

Instead of iterating month by month, we operate on the entire
dataset at once using pandas groupby + transform operations.
Should complete in under 10 seconds.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from config import CLUSTER_METHOD, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    monthly_returns:   pd.Series
    benchmark_returns: pd.Series
    positions:         pd.DataFrame
    metrics:           dict = field(default_factory=dict)
    cluster_perf:      pd.DataFrame = field(default_factory=pd.DataFrame)


# ── Core vectorised backtest ──────────────────────────────────────────────────

def run_backtest(
    method:           str = CLUSTER_METHOD,
    momentum_feature: str = MOMENTUM_FEATURE,
    top_n:            int = TOP_N_PER_CLUSTER,
) -> BacktestResult:
    from clustering.runner import load_clustered
    from features.pipeline import load_modeling_data

    logger.info("Loading data …")
    clustered = load_clustered(method)
    modeling  = load_modeling_data()

    # ── Step 1: rank stocks within each (Date, cluster) group ────────────────
    # For each month and cluster, rank by momentum descending
    logger.info("Ranking stocks …")
    df = clustered[[momentum_feature, "cluster"]].copy()
    df = df[df["cluster"] != -1]   # drop HDBSCAN noise

    df["rank"] = (
        df.groupby(["Date", "cluster"])[momentum_feature]
        .rank(ascending=False, method="first")
    )

    # ── Step 2: select top-N per cluster ─────────────────────────────────────
    selected = df[df["rank"] <= top_n].copy()
    selected["weight"] = (
        1.0 / selected.groupby("Date")["rank"].transform("count")
    )

    # ── Step 3: get forward 1-month returns ───────────────────────────────────
    # return_1m at date T is the return OF month T.
    # We want the return of the NEXT month, so we shift by -1 within each ticker.
    logger.info("Computing forward returns …")
    fwd = (
        modeling["return_1m"]
        .unstack("Ticker")
        .shift(-1)          # shift entire dataframe back 1 month
        .stack("Ticker")
        .rename("fwd_return")
    )

    selected = selected.join(fwd, how="left")

    # ── Step 4: portfolio return = weighted average of forward returns ─────────
    logger.info("Computing portfolio returns …")
    selected["weighted_return"] = selected["weight"] * selected["fwd_return"]

    monthly_returns = (
        selected.groupby("Date")["weighted_return"]
        .sum()
        .dropna()
    )
    # Remove last month (no forward return available)
    monthly_returns = monthly_returns.iloc[:-1]

    # ── Step 5: benchmark ─────────────────────────────────────────────────────
    benchmark = _load_benchmark(monthly_returns.index)

    # ── Step 6: metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(monthly_returns, benchmark)

    # ── Step 7: cluster performance ───────────────────────────────────────────
    cluster_perf = _cluster_performance(selected)

    logger.info(
        "Backtest complete. Ann. return: %.1f%%, Sharpe: %.2f",
        metrics.get("annualised_return", 0) * 100,
        metrics.get("sharpe", 0),
    )

    return BacktestResult(
        monthly_returns   = monthly_returns,
        benchmark_returns = benchmark,
        positions         = selected,
        metrics           = metrics,
        cluster_perf      = cluster_perf,
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────

def _load_benchmark(index: pd.DatetimeIndex) -> pd.Series:
    start = index.min() - pd.DateOffset(months=2)
    end   = index.max() + pd.DateOffset(months=1)
    try:
        spy = yf.download("SPY", start=start, end=end, progress=False,
                          auto_adjust=True, multi_level_index=False)
        if spy.empty:
            raise ValueError("Empty SPY data")

        close = spy["Close"]

        # Flatten any remaining MultiIndex
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(close.values, index=pd.to_datetime(close.index), name="SPY")

        monthly = close.resample("ME").last().pct_change().dropna()
        monthly.index = monthly.index.to_period("M").to_timestamp("M")

        index_ts = index.to_period("M").to_timestamp("M")
        aligned = monthly.reindex(index_ts).fillna(0)
        result = pd.Series(aligned.values, index=index, name="SPY")
        logger.info("SPY benchmark loaded: %d months", result.notna().sum())
        return result
    except Exception as e:
        logger.warning("Could not load SPY benchmark: %s", e)
        return pd.Series(0.0, index=index, name="SPY")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {}

    n_months   = len(r)
    n_years    = n_months / 12
    cum        = (1 + r).prod() - 1
    ann_return = (1 + cum) ** (1 / n_years) - 1
    ann_vol    = r.std() * np.sqrt(12)
    sharpe     = (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else 0

    downside   = r[r < 0]
    down_std   = downside.std() * np.sqrt(12) if len(downside) > 1 else 1e-8
    sortino    = ann_return / down_std if down_std > 0 else 0

    wealth     = (1 + r).cumprod()
    peak       = wealth.cummax()
    drawdown   = (wealth - peak) / peak
    max_dd     = drawdown.min()
    calmar     = ann_return / abs(max_dd) if max_dd != 0 else 0

    gains      = r[r > 0].sum()
    losses     = abs(r[r < 0].sum())
    omega      = gains / losses if losses > 0 else float("inf")
    win_rate   = (r > 0).mean()

    metrics = {
        "n_months":           n_months,
        "cumulative_return":  round(float(cum), 4),
        "annualised_return":  round(float(ann_return), 4),
        "annualised_vol":     round(float(ann_vol), 4),
        "sharpe":             round(float(sharpe), 3),
        "sortino":            round(float(sortino), 3),
        "calmar":             round(float(calmar), 3),
        "omega":              round(float(omega), 3),
        "max_drawdown":       round(float(max_dd), 4),
        "win_rate":           round(float(win_rate), 4),
        "avg_monthly_return": round(float(r.mean()), 4),
        "monthly_returns_series": r,
        "drawdown_series":        drawdown,
        "wealth_series":          wealth,
    }

    if benchmark is not None:
        b       = benchmark.reindex(r.index).fillna(0)
        b_cum   = (1 + b).prod() - 1
        b_ann   = (1 + b_cum) ** (1 / n_years) - 1
        b_sharpe = (b.mean() / b.std()) * np.sqrt(12) if b.std() > 0 else 0
        alpha   = r.mean() - b.mean()
        beta    = r.cov(b) / b.var() if b.var() > 0 else 0
        metrics.update({
            "alpha":                round(float(alpha * 12), 4),
            "beta":                 round(float(beta), 3),
            "benchmark_ann_return": round(float(b_ann), 4),
            "benchmark_sharpe":     round(float(b_sharpe), 3),
            "benchmark_series":     b,
        })

    return metrics


def _cluster_performance(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or "fwd_return" not in positions.columns:
        return pd.DataFrame()
    perf = (
        positions.groupby("cluster")["fwd_return"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_return", "std": "volatility", "count": "n_trades"})
    )
    perf["sharpe"]   = (perf["avg_return"] / perf["volatility"] * np.sqrt(12)).round(3)
    perf["win_rate"] = positions.groupby("cluster")["fwd_return"].apply(lambda x: (x > 0).mean())
    return perf.round(4)


# ── Period slicing ─────────────────────────────────────────────────────────────

PERIODS = {"1M":1, "3M":3, "6M":6, "1Y":12, "3Y":36, "5Y":60, "All":None}

def slice_period(returns: pd.Series, period: str) -> pd.Series:
    months = PERIODS.get(period)
    if months is None:
        return returns
    return returns.iloc[-months:]