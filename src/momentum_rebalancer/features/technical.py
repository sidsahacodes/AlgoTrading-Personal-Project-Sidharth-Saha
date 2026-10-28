"""
features/technical.py — daily technical indicators snapshotted at month-end.

Each function takes the full daily MultiIndex DataFrame and returns it
with new columns added. All computation is purely backward-looking.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

from momentum_rebalancer.config import (RSI_PERIOD, ATR_PERIOD, BB_PERIOD,
                                        MACD_FAST, MACD_SLOW, MACD_SIGNAL)

logger = logging.getLogger(__name__)


def add_garman_klass_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """Garman-Klass volatility estimator (uses OHLC)."""
    df = prices.copy()
    high      = df["high"].astype(float)
    low       = df["low"].astype(float)
    adj_close = df["adj close"].astype(float)
    open_     = df["open"].astype(float)
    log_hl = (np.log(high) - np.log(low)) ** 2
    log_co = (np.log(adj_close) - np.log(open_)) ** 2
    df["gkv"] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return df


def _per_ticker(df: pd.DataFrame, fn, columns: list[str]) -> pd.DataFrame:
    """
    Run `fn(rows of one ticker)` for every ticker and stack the results, aligned to `df.index`.

    `fn` returns a Series (one column) or a DataFrame (the first len(columns) columns are kept).
    An indicator library returns nothing for a series shorter than its window, and a ticker can
    fail for other reasons; either way that ticker gets NaN for these columns and nothing else is
    affected. Results are attached by position to each ticker's own index, so ordering and
    alignment cannot mislabel rows.
    """
    parts = []
    for _, g in df.groupby(level="Ticker", sort=False):
        try:
            out = fn(g)
        except Exception:
            out = None
        arr = None
        if out is not None and len(out) == len(g):
            arr = out.to_numpy().reshape(len(g), -1) if isinstance(out, pd.Series) else out.to_numpy()
            if arr.shape[1] < len(columns):
                arr = None
        if arr is None:
            arr = np.full((len(g), len(columns)), np.nan)
        parts.append(pd.DataFrame(arr[:, :len(columns)], index=g.index, columns=columns))
    if not parts:
        return pd.DataFrame(np.nan, index=df.index, columns=columns)
    return pd.concat(parts).reindex(df.index)


def add_rsi(prices: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = prices.copy()
    df[["rsi"]] = _per_ticker(df, lambda g: ta.rsi(g["adj close"].astype(float), length=period), ["rsi"])
    return df


def add_macd(prices: pd.DataFrame) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA), first column of pandas_ta output."""
    df = prices.copy()
    df[["macd"]] = _per_ticker(
        df,
        lambda g: ta.macd(g["adj close"].astype(float), fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL).iloc[:, 0],
        ["macd"],
    )
    return df


def _expanding_zscore(x: pd.Series, min_periods: int = 60) -> pd.Series:
    """z-score against the mean/std of the series *up to each point* (no look-ahead)."""
    mean = x.expanding(min_periods=min_periods).mean()
    std  = x.expanding(min_periods=min_periods).std()
    return (x - mean) / (std + 1e-8)


def add_atr(prices: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """ATR, z-scored against each ticker's own history up to that day (expanding window)."""
    df = prices.copy()

    def _atr_z(g: pd.DataFrame):
        raw = ta.atr(g["high"].astype(float), g["low"].astype(float), g["close"].astype(float), length=period)
        return None if raw is None else _expanding_zscore(raw)

    df[["atr"]] = _per_ticker(df, _atr_z, ["atr"])
    return df


def add_bollinger_bands(prices: pd.DataFrame, period: int = BB_PERIOD) -> pd.DataFrame:
    """Lower, mid, upper Bollinger Bands on log(1+price)."""
    df = prices.copy()
    df[["bblow", "bbmid", "bbhigh"]] = _per_ticker(
        df, lambda g: ta.bbands(np.log1p(g["adj close"].astype(float)), length=period), ["bblow", "bbmid", "bbhigh"])
    return df


def compute_all_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    """Chain all technical indicators onto the daily price DataFrame."""
    if not prices.index.is_unique:
        logger.warning("Price frame has duplicate (Date, Ticker) rows; keeping the last of each.")
        prices = prices[~prices.index.duplicated(keep="last")]
    return (
        prices
        .pipe(add_garman_klass_vol)
        .pipe(add_rsi)
        .pipe(add_macd)
        .pipe(add_atr)
        .pipe(add_bollinger_bands)
    )


def snapshot_at_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    """Take the last daily value in each month as the month-end feature snapshot."""
    df = daily.reset_index()
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    return (
        df.groupby(["Date", "Ticker"])
        .last()
        .sort_index()
    )