"""
features/returns.py — lagged momentum returns and rolling Fama-French betas.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas_datareader import data as web
from statsmodels.regression.rolling import RollingOLS

from config import RETURN_LAGS, FF_BETA_WINDOW

logger = logging.getLogger(__name__)

OUTLIER_CUTOFF = 0.005  # clip returns below 0.5th and above 99.5th percentile


def add_lagged_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add past N-month returns as features.
    All returns are winsorised to remove extreme outliers.
    """
    df = data.copy()
    for lag in RETURN_LAGS:
        raw = df["adj close"].pct_change(lag)
        clipped = raw.clip(
            raw.quantile(OUTLIER_CUTOFF),
            raw.quantile(1 - OUTLIER_CUTOFF),
        )
        # Convert to per-month equivalent
        df[f"return_{lag}m"] = (1 + clipped).pow(1 / lag) - 1
    return df


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


def estimate_ff5_betas(
    data: pd.DataFrame,
    window: int = FF_BETA_WINDOW,
) -> pd.DataFrame:
    """
    Rolling Fama-French 5-factor betas estimated per ticker.

    Factor data is lagged by 1 month to avoid any lookahead.
    Returns a DataFrame with columns: beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma.
    """
    start_date = data.index.get_level_values("Date").min()
    logger.info("Fetching FF5 factors from %s …", start_date.date())

    ff = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start=start_date,
    )[0]
    ff = (
        ff.drop(columns="RF")
        .div(100)
        .to_timestamp("M")
        .shift(1)           # lag factors by 1 month
    )
    ff.index.name = "Date"

    merged = data.join(ff, how="inner")
    beta_cols = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    def _rolling_betas(x: pd.DataFrame) -> pd.DataFrame:
        if x.shape[0] < window:
            return pd.DataFrame(index=x.index, columns=beta_cols)

        y = x["return_1m"]
        X = sm.add_constant(x[factor_cols])

        result = RollingOLS(
            endog=y,
            exog=X,
            window=window,
            min_nobs=len(factor_cols) + 1,
        ).fit(params_only=True)

        betas = result.params.drop(columns="const")
        betas.columns = beta_cols
        return betas

    logger.info("Estimating rolling FF5 betas (window=%d months) …", window)
    betas = (
        merged
        .groupby("Ticker", group_keys=False)
        .apply(_rolling_betas)
        .dropna()
    )
    return betas
