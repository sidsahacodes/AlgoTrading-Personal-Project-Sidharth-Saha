"""
features/pipeline.py — full feature engineering pipeline.

Input:  daily prices (Yahoo, plus optional vendor files) and the point-in-time membership list
Output: modeling_data.parquet     (Date × Ticker, the feature columns)
        monthly_returns.parquet   (Date × Ticker, raw total return for EVERY ticker with prices,
                                   including stocks outside the universe; used by the backtest)

Every feature stamped with month-end t uses only data available at t
(see tests/test_no_lookahead.py).
"""

from __future__ import annotations
import logging

import pandas as pd

from momentum_rebalancer.config import DATA_DIR, FEATURE_COLS, PIT_ALLOW_FALLBACK, RETURN_LAGS, SETTINGS
from momentum_rebalancer.data_cleaning.universe import build_monthly_universe
from momentum_rebalancer.data_ingestion.loader import download_prices, load_pit_members
from momentum_rebalancer.features.returns import (
    cross_sectional_zscore, estimate_ff5_betas, lagged_return_features,
    monthly_adjclose_panel, monthly_return_panel,
)
from momentum_rebalancer.features.technical import compute_all_technicals, snapshot_at_month_end
from momentum_rebalancer.storage.fingerprint import FEATURE_KEYS, write_meta

logger = logging.getLogger(__name__)

MODELING_DATA_PATH   = DATA_DIR / "modeling_data.parquet"
MONTHLY_RETURNS_PATH = DATA_DIR / "monthly_returns.parquet"

ZSCORE_COLS = ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank"]
TECH_COLS   = ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd"]


def build_modeling_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Steps
    -----
    1.  Load the point-in-time constituent list
    2.  Download daily prices for every ticker that was a member in the window
    3.  Compute daily technical indicators; snapshot at month-end
    4.  Build the monthly universe (PIT membership, tradable at month-end, history, liquidity)
    5.  Add lagged returns (from a complete month-end panel)
    6.  Cross-sectional z-score the technical features
    7.  Estimate rolling FF5 betas and lag them one month
    8.  Select the final feature columns; save features and the raw monthly return panel
    """
    if MODELING_DATA_PATH.exists() and not force_refresh:
        logger.info("Loading modeling data from cache: %s", MODELING_DATA_PATH)
        return pd.read_parquet(MODELING_DATA_PATH)

    logger.info("Step 1/8: Loading point-in-time S&P 500 constituent history …")
    pit = load_pit_members(force_refresh=force_refresh)
    logger.info("  PIT data: %d dates from %s to %s",
                len(pit), pit.index.min().date(), pit.index.max().date())

    logger.info("Step 2/8: Downloading prices (all tickers that were members in the window) …")
    prices = download_prices(force_refresh=force_refresh, use_pit=True)

    first_price = prices.index.get_level_values("Date").min()
    if pit.index.min() > first_price + pd.Timedelta(days=31):
        msg = (f"The membership list starts {pit.index.min().date()}, after the first price "
               f"({first_price.date()}): earlier months have no point-in-time membership, "
               f"so their universe would be survivorship-biased.")
        if not PIT_ALLOW_FALLBACK:
            raise RuntimeError(msg + " Fix the download (pit_url) or set pit_allow_fallback: true.")
        logger.warning(msg)

    logger.info("Step 3/8: Computing daily technical indicators …")
    daily = compute_all_technicals(prices)
    monthly_technicals = snapshot_at_month_end(daily)[TECH_COLS]

    logger.info("Step 4/8: Building monthly universe …")
    universe = build_monthly_universe(prices, pit=pit)
    data = universe.join(monthly_technicals, how="inner")[["dv_rank", *TECH_COLS]]

    logger.info("Step 5/8: Computing lagged returns …")
    adj = monthly_adjclose_panel(prices)
    raw_1m = monthly_return_panel(adj, 1)
    raw_1m.to_parquet(MONTHLY_RETURNS_PATH)
    data = data.join(lagged_return_features(adj, data.index, RETURN_LAGS), how="left").dropna()

    logger.info("Step 6/8: Cross-sectional z-scoring …")
    data = cross_sectional_zscore(data, ZSCORE_COLS)

    logger.info("Step 7/8: Estimating rolling FF5 betas …")
    betas = estimate_ff5_betas(raw_1m)
    data = data.join(betas, how="inner")

    missing = set(FEATURE_COLS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    logger.info("Step 8/8: Saving …")
    data = data[FEATURE_COLS].copy()
    data.to_parquet(MODELING_DATA_PATH)
    write_meta(MODELING_DATA_PATH, SETTINGS, FEATURE_KEYS)
    logger.info(
        "Saved modeling data → %s  (%d rows, %d tickers, %s to %s)",
        MODELING_DATA_PATH, len(data), data.index.get_level_values("Ticker").nunique(),
        data.index.get_level_values("Date").min().date(), data.index.get_level_values("Date").max().date(),
    )
    return data


def load_modeling_data() -> pd.DataFrame:
    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError("modeling_data.parquet not found. Run: mr build-features")
    return pd.read_parquet(MODELING_DATA_PATH)


def load_monthly_returns() -> pd.DataFrame:
    """Raw monthly total returns (Date × Ticker) for every ticker with prices."""
    if not MONTHLY_RETURNS_PATH.exists():
        raise FileNotFoundError("monthly_returns.parquet not found. Run: mr build-features --force")
    return pd.read_parquet(MONTHLY_RETURNS_PATH)
