"""
features/pipeline.py — full feature engineering pipeline.

Input:  raw daily prices
Output: modeling_data.parquet (Date × Ticker, all features)

Survivorship bias is addressed here by passing the point-in-time
constituent list to build_monthly_universe().
"""

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from config import DATA_DIR, FEATURE_COLS
from data.loader import download_prices, load_pit_members
from data.universe import build_monthly_universe
from features.technical import compute_all_technicals, snapshot_at_month_end
from features.returns import add_lagged_returns, cross_sectional_zscore, estimate_ff5_betas

logger = logging.getLogger(__name__)

MODELING_DATA_PATH = DATA_DIR / "modeling_data.parquet"

ZSCORE_COLS = [
    "gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank",
]


def build_modeling_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Full feature engineering pipeline with survivorship bias fix.

    Steps
    -----
    1.  Download point-in-time constituent list (all historical S&P 500 members)
    2.  Download raw daily prices for all historical tickers
    3.  Compute daily technical indicators
    4.  Snapshot technicals at each month-end
    5.  Build liquidity universe — with PIT filter to prevent survivorship bias
    6.  Add lagged returns (1m … 12m)
    7.  Cross-sectional z-score technical features
    8.  Estimate rolling FF5 betas
    9.  Select final feature columns
    """
    if MODELING_DATA_PATH.exists() and not force_refresh:
        logger.info("Loading modeling data from cache: %s", MODELING_DATA_PATH)
        return pd.read_parquet(MODELING_DATA_PATH)

    # ── 1. Point-in-time constituent list ─────────────────────────────────────
    logger.info("Step 1/8: Loading point-in-time S&P 500 constituent history …")
    pit = load_pit_members(force_refresh=force_refresh)
    logger.info("  PIT data: %d dates from %s to %s",
                len(pit), pit.index.min().date(), pit.index.max().date())

    # ── 2. Raw prices (all historical tickers, not just current survivors) ─────
    logger.info("Step 2/8: Downloading prices for all historical S&P 500 tickers …")
    logger.info("  NOTE: This downloads ~600+ tickers and may take 15-20 mins.")
    prices = download_prices(force_refresh=force_refresh, use_pit=True)

    # ── 3. Daily technicals ────────────────────────────────────────────────────
    logger.info("Step 3/8: Computing daily technical indicators …")
    daily = compute_all_technicals(prices)

    # ── 4. Month-end snapshot ──────────────────────────────────────────────────
    logger.info("Step 4/8: Snapshotting technicals at month-end …")
    tech_cols = ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd"]
    monthly_technicals = snapshot_at_month_end(daily)[tech_cols]

    # ── 5. Liquidity universe WITH PIT filter ──────────────────────────────────
    logger.info("Step 5/8: Building monthly universe (with survivorship bias fix) …")
    universe = build_monthly_universe(prices, pit=pit)

    # ── 6. Join technicals ─────────────────────────────────────────────────────
    data = universe.join(monthly_technicals, how="inner")

    # ── 7. Lagged returns ──────────────────────────────────────────────────────
    logger.info("Step 6/8: Computing lagged returns …")
    data = (
        data.groupby("Ticker", group_keys=False)
        .apply(add_lagged_returns)
        .dropna()
    )

    # ── 8. Cross-sectional z-score ─────────────────────────────────────────────
    logger.info("Step 7/8: Cross-sectional z-scoring …")
    data = cross_sectional_zscore(data, ZSCORE_COLS)

    # ── 9. FF5 betas ───────────────────────────────────────────────────────────
    logger.info("Step 8/8: Estimating rolling FF5 betas …")
    betas = estimate_ff5_betas(data)
    data  = data.join(betas, how="inner")

    # ── Final feature selection ────────────────────────────────────────────────
    missing = set(FEATURE_COLS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    data = data[FEATURE_COLS].copy()
    data.to_parquet(MODELING_DATA_PATH)
    logger.info(
        "Saved modeling data → %s  (%d rows, %d tickers)",
        MODELING_DATA_PATH, len(data),
        data.index.get_level_values("Ticker").nunique(),
    )
    return data


def load_modeling_data() -> pd.DataFrame:
    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            "modeling_data.parquet not found. Run: python run.py build-features"
        )
    return pd.read_parquet(MODELING_DATA_PATH)
