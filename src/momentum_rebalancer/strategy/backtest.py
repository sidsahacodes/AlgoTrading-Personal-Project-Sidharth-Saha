"""
strategy/backtest.py — backtest, metrics and period slicing.

`run_backtest` runs the month-by-month engine in `simulate.py` on the cached clusters and the
raw monthly return panel, so it follows the live path (same selection, HRP weights, exposure
scalar, trade band) and charges transaction costs. Returns are labelled by the month they were
earned, so they line up with the SPY benchmark.

Still not modelled: market impact beyond `cost_bps`, taxes, and anything about stocks for which
the data source has no prices at all (see `mr coverage`).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from momentum_rebalancer.config import (
    CLUSTER_METHOD, COST_BPS, DELISTING_RETURN, EXPOSURE_SCALAR, HRP_LOOKBACK_MONTHS,
    MIN_TRADE_PCT, MOMENTUM_FEATURE, N_CLUSTERS, OPTIMISER, PORTFOLIO_SIZE, TOP_N_PER_CLUSTER,
)
from momentum_rebalancer.strategy.simulate import SimConfig, simulate

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    monthly_returns:   pd.Series            # net of costs, indexed by the month the return was earned
    benchmark_returns: pd.Series            # SPY, same index (NaN if it could not be downloaded)
    positions:         pd.DataFrame
    metrics:           dict = field(default_factory=dict)
    cluster_perf:      pd.DataFrame = field(default_factory=pd.DataFrame)
    gross_returns:     pd.Series = field(default_factory=pd.Series)
    turnover:          pd.Series = field(default_factory=pd.Series)      # traded notional per rebalance
    diagnostics:       dict = field(default_factory=dict)


def default_portfolio_size(top_n: int) -> int:
    """The configured cap when `top_n` is the configured value, otherwise top_n x clusters."""
    return PORTFOLIO_SIZE if top_n == TOP_N_PER_CLUSTER else top_n * N_CLUSTERS


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    method:           str = CLUSTER_METHOD,
    momentum_feature: str = MOMENTUM_FEATURE,
    top_n:            int = TOP_N_PER_CLUSTER,
    *,
    portfolio_size:      int | None = None,
    optimiser:           str = OPTIMISER,
    hrp_lookback_months: int | None = HRP_LOOKBACK_MONTHS,
    exposure_scalar:     float = EXPOSURE_SCALAR,
    min_trade_pct:       float = MIN_TRADE_PCT,
    cost_bps:            float = COST_BPS,
    delisting_return:    float = DELISTING_RETURN,
) -> BacktestResult:
    from momentum_rebalancer.clustering.runner import load_clustered
    from momentum_rebalancer.features.pipeline import load_monthly_returns

    logger.info("Loading data …")
    clustered = load_clustered(method)
    returns_wide = load_monthly_returns()

    cfg = SimConfig(
        momentum_feature=momentum_feature, top_n=top_n,
        portfolio_size=portfolio_size or default_portfolio_size(top_n),
        optimiser=optimiser, hrp_lookback_months=hrp_lookback_months,
        exposure_scalar=exposure_scalar, min_trade_pct=min_trade_pct,
        cost_bps=cost_bps, delisting_return=delisting_return,
    )
    logger.info("Simulating %s …", cfg)
    sim = simulate(clustered, returns_wide, cfg)

    # Label by the month the return was earned (so it lines up with the benchmark).
    net   = sim.net.set_axis(pd.DatetimeIndex(sim.earned_in.values))
    gross = sim.gross.set_axis(net.index)
    turnover = sim.traded.set_axis(net.index)

    benchmark = _load_benchmark(net.index)
    metrics = compute_metrics(net, benchmark)
    metrics.update({
        "gross_annualised_return": round(float(_annualise(gross)), 4),
        "avg_monthly_traded":      round(float(sim.traded.mean()), 4),
        "annual_cost_drag":        round(float(sim.costs.mean() * 12), 4),
    })
    positions = sim.positions
    cluster_perf = _cluster_performance(positions)

    logger.info("Backtest complete. Net ann. return: %.1f%%, Sharpe: %.2f, cost drag: %.2f%%/yr",
                metrics.get("annualised_return", 0) * 100, metrics.get("sharpe", 0),
                metrics["annual_cost_drag"] * 100)
    if sim.diagnostics.get("delisted") or sim.diagnostics.get("data_gaps"):
        logger.warning("Held stocks with no next-month price: %d (assumed %.0f%%), data gaps: %d (assumed 0%%)",
                       sim.diagnostics["delisted"], delisting_return * 100, sim.diagnostics["data_gaps"])

    return BacktestResult(
        monthly_returns=net, benchmark_returns=benchmark, positions=positions,
        metrics=metrics, cluster_perf=cluster_perf,
        gross_returns=gross, turnover=turnover, diagnostics=sim.diagnostics,
    )


def _annualise(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (12 / len(r)) - 1 if len(r) else float("nan")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def _load_benchmark(index: pd.DatetimeIndex) -> pd.Series:
    """
    SPY monthly total returns aligned to `index` (month-ends = the month each return was earned).
    Returns NaN where SPY is unavailable; it never substitutes 0%.
    """
    nan = pd.Series(np.nan, index=index, name="SPY")
    if len(index) == 0:
        return nan
    start = index.min() - pd.DateOffset(months=2)
    end   = index.max() + pd.DateOffset(months=1)
    try:
        spy = yf.download("SPY", start=start, end=end, progress=False,
                          auto_adjust=True, multi_level_index=False)
        if spy.empty:
            raise ValueError("Empty SPY data")

        close = spy["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(close.values, index=pd.to_datetime(close.index), name="SPY")

        monthly = close.resample("ME").last().pct_change().dropna()
        monthly.index = monthly.index.to_period("M").to_timestamp("M")

        index_ts = index.to_period("M").to_timestamp("M")
        aligned = monthly.reindex(index_ts)
        result = pd.Series(aligned.values, index=index, name="SPY")
        logger.info("SPY benchmark loaded: %d of %d months", result.notna().sum(), len(result))
        return result
    except Exception as e:
        logger.warning("Could not load SPY benchmark: %s. Benchmark metrics will be unavailable.", e)
        return nan


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

    # Sortino: annualised return over annualised downside deviation (target 0, all months in the denominator)
    down_dev   = np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2)) * np.sqrt(12)
    sortino    = ann_return / down_dev if down_dev > 0 else float("inf")

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
        "sortino":            round(float(sortino), 3) if np.isfinite(sortino) else float("inf"),
        "calmar":             round(float(calmar), 3),
        "omega":              round(float(omega), 3) if np.isfinite(omega) else float("inf"),
        "max_drawdown":       round(float(max_dd), 4),
        "win_rate":           round(float(win_rate), 4),
        "avg_monthly_return": round(float(r.mean()), 4),
        "monthly_returns_series": r,
        "drawdown_series":        drawdown,
        "wealth_series":          wealth,
    }

    if benchmark is not None and benchmark.reindex(r.index).notna().any():
        b       = benchmark.reindex(r.index).dropna()
        r_b     = r.reindex(b.index)
        b_cum   = (1 + b).prod() - 1
        b_ann   = (1 + b_cum) ** (12 / len(b)) - 1
        b_sharpe = (b.mean() / b.std()) * np.sqrt(12) if b.std() > 0 else 0
        beta    = r_b.cov(b) / b.var() if b.var() > 0 else 0
        excess  = r_b.mean() - b.mean()                     # plain average difference, not beta-adjusted
        alpha   = r_b.mean() - beta * b.mean()              # CAPM alpha (risk-free rate taken as 0)
        metrics.update({
            "excess_return":        round(float(excess * 12), 4),
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