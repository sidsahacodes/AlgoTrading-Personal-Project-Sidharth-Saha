"""
features/technical.py — daily technical indicators snapshotted at month-end.

Each function takes the full daily MultiIndex DataFrame and returns it
with new columns added. All computation is purely backward-looking.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pandas_ta as ta

from config import RSI_PERIOD, ATR_PERIOD, BB_PERIOD


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


def add_rsi(prices: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = prices.copy()
    df["rsi"] = (
        df.groupby("Ticker")["adj close"]
        .transform(lambda x: ta.rsi(x.astype(float), length=period))
    )
    return df


def add_macd(prices: pd.DataFrame) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA), first column of pandas_ta output."""
    df = prices.copy()

    def _safe_macd(x):
        try:
            return ta.macd(x["adj close"].astype(float)).iloc[:, 0]
        except Exception:
            return pd.Series(float("nan"), index=x.index)

    df["macd"] = (
        df.groupby("Ticker", group_keys=False)
        .apply(_safe_macd)
    )
    return df


def add_atr(prices: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """ATR normalised cross-sectionally (z-score per date)."""
    df = prices.copy()

    def _safe_atr(x):
        try:
            h = x["high"].astype(float)
            l = x["low"].astype(float)
            c = x["close"].astype(float)
            return ta.atr(h, l, c, length=period)
        except Exception:
            return pd.Series(float("nan"), index=x.index)

    raw_atr = (
        df.groupby("Ticker", group_keys=False)
        .apply(_safe_atr)
    )
    if isinstance(raw_atr, pd.DataFrame):
        raw_atr = raw_atr.iloc[:, 0]
    df["atr"] = raw_atr.groupby(df.index.get_level_values("Ticker")).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return df


def add_bollinger_bands(prices: pd.DataFrame, period: int = BB_PERIOD) -> pd.DataFrame:
    """Upper, mid, lower Bollinger Bands on log(1+price)."""
    df = prices.copy()

    def _safe_bb(x):
        try:
            return ta.bbands(np.log1p(x["adj close"].astype(float)), length=period)
        except Exception:
            return pd.DataFrame(float("nan"), index=x.index, columns=["bblow","bbmid","bbhigh","a","b"])

    bb = (
        df.groupby("Ticker", group_keys=False)
        .apply(_safe_bb)
    )
    df["bblow"]  = bb.iloc[:, 0]
    df["bbmid"]  = bb.iloc[:, 1]
    df["bbhigh"] = bb.iloc[:, 2]
    return df


def compute_all_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    """Chain all technical indicators onto the daily price DataFrame."""
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