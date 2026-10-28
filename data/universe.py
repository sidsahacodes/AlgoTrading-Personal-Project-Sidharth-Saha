"""
data/universe.py — build a clean, bias-free monthly trading universe.

Survivorship bias fix:
  At each month-end, only stocks that were actually in the S&P 500 on
  that date are eligible. This uses the point-in-time (PIT) constituent
  list from loader.py.

  Without this, a backtest run today would only see current survivors —
  missing delisted companies like Enron, Lehman, Bear Stearns etc. that
  would have been in the universe historically and could have been traded.
"""

from __future__ import annotations
import logging

import pandas as pd

from config import MIN_HISTORY_MONTHS, UNIVERSE_SIZE

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "adj close", "volume"]


# ── Month-end alignment ───────────────────────────────────────────────────────

def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Take the last daily observation in each calendar month per ticker."""
    df = prices.reset_index()
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    return (
        df.groupby(["Date", "Ticker"], as_index=False)
        .last()
        .set_index(["Date", "Ticker"])
        .sort_index()
    )


def drop_incomplete(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(subset=REQUIRED_COLS)


def add_dollar_volume(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["dollar_volume"] = data["close"] * data["volume"]
    return data


def filter_min_history(
    data: pd.DataFrame,
    min_months: int = MIN_HISTORY_MONTHS,
) -> pd.DataFrame:
    counts = data.groupby("Ticker").size()
    valid  = counts[counts >= min_months].index
    return data[data.index.get_level_values("Ticker").isin(valid)]


# ── PIT membership filter ─────────────────────────────────────────────────────

def apply_pit_filter(
    data: pd.DataFrame,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    """
    At each month-end date, remove any ticker that was NOT in the S&P 500
    on that date according to the point-in-time constituent list.

    This is the core survivorship bias fix:
    - A stock delisted in 2008 is only in the universe up to 2008
    - A stock added in 2020 is only in the universe from 2020 onward
    - We never look at a stock we couldn't have known about at the time
    """
    logger.info("Applying point-in-time membership filter …")

    dates   = data.index.get_level_values("Date").unique()
    keep    = []

    for date in dates:
        # Get constituents as of this exact month-end
        available = pit[pit.index <= date]
        if available.empty:
            pit_tickers = set(pit.iloc[0]["tickers"])
        else:
            pit_tickers = set(available.iloc[-1]["tickers"])

        month_data = data.xs(date, level="Date")
        valid      = month_data.index[month_data.index.isin(pit_tickers)]

        for ticker in valid:
            keep.append((date, ticker))

    if not keep:
        logger.warning("PIT filter removed all rows — check constituent data.")
        return data

    keep_idx = pd.MultiIndex.from_tuples(keep, names=["Date", "Ticker"])
    filtered = data[data.index.isin(keep_idx)]

    removed = len(data) - len(filtered)
    logger.info(
        "PIT filter: kept %d rows, removed %d (%.1f%% survivorship bias reduction)",
        len(filtered), removed, removed / len(data) * 100,
    )
    return filtered


# ── Liquidity universe ────────────────────────────────────────────────────────

def build_universe(
    data: pd.DataFrame,
    top_n: int = UNIVERSE_SIZE,
) -> pd.DataFrame:
    """
    Each month, keep top_n tickers by 5-year rolling average dollar volume.
    """
    data = data.copy()

    monthly_dv = (
        data["dollar_volume"]
        .unstack("Ticker")
        .rolling(60, min_periods=1)
        .mean()
        .stack("Ticker")
        .rename("dv_rolling")
    )

    data = data.join(monthly_dv, how="inner")

    data["dv_rank"] = (
        data.groupby("Date")["dv_rolling"]
        .rank(ascending=False, method="first")
    )

    universe = data[data["dv_rank"] <= top_n].copy()
    return universe.drop(columns=["dv_rolling"])


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_monthly_universe(
    prices: pd.DataFrame,
    pit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full universe pipeline: daily prices → clean monthly universe.

    Pass pit=None to skip the survivorship bias fix (not recommended).
    Pass pit=load_pit_members() for a bias-free backtest universe.
    """
    data = (
        prices
        .pipe(to_month_end)
        .pipe(drop_incomplete)
        .pipe(add_dollar_volume)
        .pipe(filter_min_history)
    )

    if pit is not None:
        data = apply_pit_filter(data, pit)
    else:
        logger.warning(
            "No PIT data provided — universe uses current survivors only. "
            "Backtest results will be optimistically biased."
        )

    return build_universe(data)
