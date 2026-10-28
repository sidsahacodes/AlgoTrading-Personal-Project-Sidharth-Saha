"""
features/returns.py — point-in-time return features and rolling Fama-French betas.

Every value stamped with month-end *t* is computed from data available at *t*:

* Returns come from a complete month-end adjusted-close panel (calendar lags, so a
  stock that leaves and re-enters the universe does not get a misaligned lag).
* Winsorising is cross-sectional (per date), never against a ticker's full history.
* FF5 betas use excess returns, full rolling windows only, and are lagged one month
  (the factor library publishes with a delay). Months for which the library has not
  published yet carry the last available beta forward (a limited number of months).

`tests/test_no_lookahead.py` verifies this by truncating the future and checking that
nothing changes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas_datareader import data as web
from statsmodels.regression.rolling import RollingOLS

from momentum_rebalancer.config import FF_BETA_WINDOW, RETURN_LAGS

logger = logging.getLogger(__name__)

OUTLIER_CUTOFF = 0.005          # cross-sectional winsorising at the 0.5th / 99.5th percentile
FF_MAX_CARRY_MONTHS = 3         # months a beta may be carried forward when factors are unpublished
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
BETA_COLS = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]


# ── Month-end panels ──────────────────────────────────────────────────────────

def monthly_adjclose_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Month-end adjusted close, wide (Date × Ticker), on a complete monthly index.

    Uses the last observation in each calendar month, labelled with the month-end date.
    Months with no observation are NaN (never forward-filled).
    """
    adj = prices["adj close"].astype(float).dropna().reset_index()
    adj["Date"] = adj["Date"] + pd.offsets.MonthEnd(0)
    wide = adj.groupby(["Date", "Ticker"])["adj close"].last().unstack("Ticker").sort_index()
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="ME")
    return wide.reindex(full).rename_axis("Date")


def monthly_return_panel(adj: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Raw total return over `lag` calendar months; NaN where either price is missing."""
    return adj.pct_change(lag, fill_method=None)


def last_valid_dates(returns_wide: pd.DataFrame) -> pd.Series:
    """Last month with a return, per ticker (used to tell delistings from data gaps)."""
    return returns_wide.apply(lambda s: s.last_valid_index())


# ── Return features ───────────────────────────────────────────────────────────

def winsorize_cross_section(df: pd.DataFrame, cols: list[str], q: float = OUTLIER_CUTOFF) -> pd.DataFrame:
    """Clip each column to its own [q, 1-q] percentiles *within each date*."""
    out = df.copy()
    grouped = out.groupby(level="Date")
    for col in cols:
        lo = grouped[col].transform(lambda x: x.quantile(q))
        hi = grouped[col].transform(lambda x: x.quantile(1 - q))
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def lagged_return_features(
    adj: pd.DataFrame,
    index: pd.MultiIndex,
    lags: list[int] = RETURN_LAGS,
    q: float = OUTLIER_CUTOFF,
) -> pd.DataFrame:
    """
    `return_<n>m` for the (Date, Ticker) rows in `index`:
    the n-month return, winsorised cross-sectionally, as a per-month-equivalent
    ``(1 + r_n)^(1/n) - 1``.
    """
    out = pd.DataFrame(index=index)
    for lag in lags:
        raw = (monthly_return_panel(adj, lag).stack(future_stack=True)
               .rename_axis(["Date", "Ticker"]).rename(f"raw_{lag}m"))
        out = out.join(raw, how="left")
    out = winsorize_cross_section(out, [f"raw_{lag}m" for lag in lags], q)
    for lag in lags:
        out[f"return_{lag}m"] = (1 + out.pop(f"raw_{lag}m")).pow(1 / lag) - 1
    return out


def cross_sectional_zscore(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Z-score each feature cross-sectionally (within each month).
    Ensures clustering operates on relative, not absolute, values.
    """
    df = data.copy()
    for col in cols:
        df[col] = (
            df.groupby("Date")[col]
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        )
    return df


# ── Fama-French betas ─────────────────────────────────────────────────────────

def load_ff5(start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Monthly FF5 factors and RF as decimals, indexed by month-end timestamp."""
    ff = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[0]
    ff = ff.div(100)
    ff.index = ff.index.to_timestamp(how="end").normalize()
    ff.index.name = "Date"
    return ff


def estimate_ff5_betas(
    returns_wide: pd.DataFrame,
    window: int = FF_BETA_WINDOW,
    ff: pd.DataFrame | None = None,
    max_carry: int = FF_MAX_CARRY_MONTHS,
) -> pd.DataFrame:
    """
    Rolling FF5 betas per ticker from raw monthly returns (Date × Ticker).

    * y = excess return (return - RF); X = the five factors; OLS over `window` valid months.
    * Only full windows produce a beta (no 6-observation fits).
    * The beta stamped at month *t* uses data through *t-1* (factors are published late).
    * If the factor library has not published recent months, the last beta is carried
      forward for at most `max_carry` months.

    Returns a long frame indexed by (Date, Ticker) with the five beta columns (NaN rows dropped).
    """
    if ff is None:
        start = returns_wide.index.min()
        logger.info("Fetching FF5 factors from %s …", start.date())
        ff = load_ff5(start)
    logger.info("FF5 factors available through %s", ff.index.max().date())

    idx = returns_wide.index.intersection(ff.index)
    X = sm.add_constant(ff.loc[idx, FACTOR_COLS])
    rf = ff.loc[idx, "RF"]

    per_ticker = {}
    logger.info("Estimating rolling FF5 betas (window=%d months) for %d tickers …",
                window, returns_wide.shape[1])
    for ticker in returns_wide.columns:
        y = (returns_wide.loc[idx, ticker] - rf).dropna()
        if len(y) < window:
            continue
        res = RollingOLS(endog=y, exog=X.loc[y.index], window=window, min_nobs=window).fit(params_only=True)
        betas = res.params.drop(columns="const")
        betas.columns = BETA_COLS
        per_ticker[ticker] = betas

    if not per_ticker:
        return pd.DataFrame(columns=BETA_COLS, index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Ticker"]))

    long = pd.concat(per_ticker, names=["Ticker", "Date"]).swaplevel().sort_index()
    wide = long.unstack("Ticker").reindex(returns_wide.index)          # (beta, Ticker) columns
    wide = wide.shift(1).ffill(limit=max_carry)                         # known one month later; carry if unpublished
    out = wide.stack("Ticker", future_stack=True).dropna(how="all")
    out.index.names = ["Date", "Ticker"]
    return out[BETA_COLS].dropna()
