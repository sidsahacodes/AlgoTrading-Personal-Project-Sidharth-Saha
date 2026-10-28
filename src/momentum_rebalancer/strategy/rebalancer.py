"""
strategy/rebalancer.py — translate target weights into a trade list.

Takes target weights (from optimiser) and current holdings (from broker)
and produces a minimal list of BUY / SELL orders to reach the target.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import pandas as pd

from momentum_rebalancer.config import PORTFOLIO_VALUE

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    ticker:     str
    action:     str        # "BUY" | "SELL"
    shares:     int
    target_pct: float      # target weight (0–1)
    current_pct: float     # current weight (0–1)
    delta_pct:  float      # target - current


def compute_trades(
    target_weights: dict[str, float],
    current_holdings: dict[str, float],   # {ticker: current weight (0–1)}
    portfolio_value: float = PORTFOLIO_VALUE,
    min_trade_value: float = 100.0,       # ignore rounding noise below $100
    min_trade_pct: float = 0.0,           # skip entries/resizes smaller than this weight change
) -> list[Trade]:
    """
    Compute the trades needed to move from current_holdings to target_weights.

    Parameters
    ----------
    target_weights   : {ticker: target weight}, must sum to ~1
    current_holdings : {ticker: current weight} — pass {} for fresh start
    portfolio_value  : total account value in $
    min_trade_value  : skip trades smaller than this dollar amount
    min_trade_pct    : skip entries/resizes whose |weight change| is below this
                       (e.g. 0.005 = 0.5%). Full exits (target == 0) are never skipped
                       by this band, so small leftover positions do not linger.

    Returns
    -------
    List of Trade objects, sorted: sells first, then buys (frees cash first).
    """
    all_tickers = set(target_weights) | set(current_holdings)
    trades = []

    for ticker in all_tickers:
        target  = target_weights.get(ticker, 0.0)
        current = current_holdings.get(ticker, 0.0)
        delta   = target - current

        dollar_delta = delta * portfolio_value

        if abs(dollar_delta) < min_trade_value:
            continue

        if target > 0 and abs(delta) < min_trade_pct:
            continue

        # We don't know prices here — shares are computed by the broker layer.
        # Use a placeholder of 0; broker fills in the actual share count.
        trades.append(Trade(
            ticker      = ticker,
            action      = "BUY" if delta > 0 else "SELL",
            shares      = 0,         # filled by broker
            target_pct  = target,
            current_pct = current,
            delta_pct   = delta,
        ))

    # Sells first so cash is available for buys
    trades.sort(key=lambda t: (0 if t.action == "SELL" else 1, t.ticker))

    logger.info(
        "Rebalance: %d buys, %d sells",
        sum(1 for t in trades if t.action == "BUY"),
        sum(1 for t in trades if t.action == "SELL"),
    )
    return trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert trade list to a tidy DataFrame for display / export."""
    return pd.DataFrame([
        {
            "ticker":       t.ticker,
            "action":       t.action,
            "shares":       t.shares,
            "target_pct":   round(t.target_pct * 100, 2),
            "current_pct":  round(t.current_pct * 100, 2),
            "delta_pct":    round(t.delta_pct * 100, 2),
        }
        for t in trades
    ])
