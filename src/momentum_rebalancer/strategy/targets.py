"""
strategy/targets.py — from latest clusters to a target portfolio and trade list.

One place for logic that used to be duplicated in run.py and dashboard/app.py:
  current_selection()  -> stocks chosen for the latest month
  target_weights()     -> HRP/equal weights from raw monthly returns, optional recent window, exposure scalar
  plan_rebalance()     -> (selected, target, current, value, trades)
"""

from __future__ import annotations

import logging

import pandas as pd

from momentum_rebalancer.clustering.runner import get_current_clusters, load_clustered
from momentum_rebalancer.config import (
    CLUSTER_METHOD, EXPOSURE_SCALAR, HRP_LOOKBACK_MONTHS, MIN_TRADE_PCT,
    MOMENTUM_FEATURE, OPTIMISER, PORTFOLIO_VALUE, TOP_N_PER_CLUSTER,
)
from momentum_rebalancer.features.pipeline import load_monthly_returns
from momentum_rebalancer.strategy.optimiser import equal_weight, weights_from_history
from momentum_rebalancer.strategy.rebalancer import compute_trades
from momentum_rebalancer.strategy.signals import select_top_momentum

logger = logging.getLogger(__name__)


def current_selection() -> pd.DataFrame:
    clustered  = load_clustered(CLUSTER_METHOD)
    month_data = get_current_clusters(clustered)
    return select_top_momentum(month_data, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER)


def history_for(tickers: list[str], returns_wide: pd.DataFrame | None = None,
                lookback_months: int | None = HRP_LOOKBACK_MONTHS) -> pd.DataFrame:
    """Wide monthly *raw* return frame for `tickers` (all history, or the last N months)."""
    returns_wide = load_monthly_returns() if returns_wide is None else returns_wide
    valid = [t for t in tickers if t in returns_wide.columns]
    missing = sorted(set(tickers) - set(valid))
    if missing:
        logger.warning("No return history for %s; excluded from HRP.", missing)
    if not valid:
        raise ValueError("None of the selected tickers has return history.")
    returns = returns_wide.loc[:, valid].dropna(how="all")
    if lookback_months:
        returns = returns.tail(lookback_months)
    return returns


def scale_exposure(weights: dict[str, float], scalar: float) -> dict[str, float]:
    return {t: w * scalar for t, w in weights.items()}


def target_weights(
    tickers: list[str],
    returns_wide: pd.DataFrame | None = None,
    optimiser: str = OPTIMISER,
    hrp_lookback_months: int | None = HRP_LOOKBACK_MONTHS,
    exposure_scalar: float = EXPOSURE_SCALAR,
) -> dict[str, float]:
    """Target weights for the selected tickers (same weighting code the backtest uses)."""
    if optimiser == "hrp":
        history = history_for(tickers, returns_wide, hrp_lookback_months)
        weights, how = weights_from_history(list(history.columns), history, "hrp", hrp_lookback_months)
        if how != "hrp":
            logger.warning("Weights fell back to %s.", how)
    else:
        weights = equal_weight(tickers)
    return scale_exposure(weights, exposure_scalar)


def plan_rebalance():
    """Return (selected, target, current, value, trades) using the configured broker."""
    from momentum_rebalancer import broker      # lazy: avoids loading IBKR code for pure-math users

    selected = current_selection()
    target   = target_weights(selected.index.tolist())
    current  = broker.get_holdings()
    value    = broker.get_value()
    value    = float(value) if value else float(PORTFOLIO_VALUE)
    trades   = compute_trades(target, current, portfolio_value=value, min_trade_pct=MIN_TRADE_PCT)
    return selected, target, current, value, trades
