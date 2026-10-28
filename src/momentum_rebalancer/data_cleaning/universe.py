"""
data_cleaning/universe.py — build a clean, point-in-time monthly trading universe.

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

from momentum_rebalancer.config import MAX_STALE_DAYS, MIN_HISTORY_MONTHS, UNIVERSE_SIZE
from momentum_rebalancer.data_ingestion.loader import get_pit_tickers_for_date

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
    """
    Keep a (Date, Ticker) row only once the ticker already has `min_months` month-end rows
    up to and including that date. (Counting a ticker's whole-sample history would use the
    future to decide eligibility today.)
    """
    seen = data.groupby(level="Ticker").cumcount() + 1
    return data[seen >= min_months]


def _bars(prices: pd.DataFrame) -> pd.Series:
    """Prices of real trading bars: rows that have a close (CRSP delisting rows carry only a return)."""
    return prices["close"] if "close" in prices.columns else prices["adj close"]


def month_end_bar_dates(prices: pd.DataFrame) -> pd.Series:
    """Date of the last real price bar in each (month-end, Ticker)."""
    bars = _bars(prices).dropna().reset_index()[["Date", "Ticker"]]
    bars["Month"] = bars["Date"] + pd.offsets.MonthEnd(0)
    return bars.groupby(["Month", "Ticker"])["Date"].max().rename("bar_date").rename_axis(["Date", "Ticker"])


def delisted_in_month(prices: pd.DataFrame) -> pd.Index:
    """(month-end, Ticker) pairs in which the data carry a return but no trading bar, i.e. a delisting."""
    if "close" not in prices.columns:
        return pd.MultiIndex.from_arrays([[], []], names=["Date", "Ticker"])
    ev = prices.loc[prices["adj close"].notna() & prices["close"].isna()].reset_index()[["Date", "Ticker"]]
    ev["Date"] = ev["Date"] + pd.offsets.MonthEnd(0)
    return pd.MultiIndex.from_frame(ev.drop_duplicates())


def drop_stale_bars(data: pd.DataFrame, bar_dates: pd.Series, max_days: int = MAX_STALE_DAYS,
                    delisted: pd.Index | None = None) -> pd.DataFrame:
    """
    Drop (month-end, Ticker) rows whose last price bar is more than `max_days` before the
    month-end: a stock that stopped trading mid-month (delisted or acquired) is not tradable
    at that month-end, and its stale row must not be ranked or selected. A stock whose delisting
    falls in the month is dropped for that month-end even if its last bar is recent.
    """
    last_bar = bar_dates.reindex(data.index)
    month_end = pd.Series(data.index.get_level_values("Date"), index=data.index)
    age = month_end - last_bar
    keep = age <= pd.Timedelta(days=max_days)
    if delisted is not None and len(delisted):
        keep &= ~data.index.isin(delisted)
    return data[keep]


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
        # Constituents as of this exact month-end
        pit_tickers = set(get_pit_tickers_for_date(pit, date))

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
        .pipe(drop_stale_bars, bar_dates=month_end_bar_dates(prices), delisted=delisted_in_month(prices))
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
