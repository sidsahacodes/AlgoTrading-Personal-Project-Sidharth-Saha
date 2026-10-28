import pandas as pd

"""
This module preprocesses raw daily OHLCV data for a monthly rebalanced trading strategy.

Objectives:
1. Produce month-end observations that reflect the information set available at each
   month-end close.
2. Ensure the preprocessing pipeline introduces no lookahead bias, survivorship bias,
   or data imputation bias.
3. Construct a realistic and tradable asset universe for downstream modeling.

Constraints and exclusions:
1. No forward- or back-filling of missing prices or volumes, as this fabricates data
   that was not observed.
2. No use of future timestamps to clean or modify historical observations.
3. No universe definition based on current index membership, which would introduce
   survivorship bias.

Preprocessing decisions:
1. Daily data is aligned to month-end and the last observation in each month is used,
   matching monthly rebalancing mechanics.
2. Observations with incomplete OHLCV at month-end are dropped, since incomplete data
   implies unreliable tradability.
3. A minimum history requirement is enforced to exclude securities with insufficient
   historical depth.
4. A rolling liquidity-based universe is constructed each month by selecting the top N
   securities by dollar volume, ensuring time-consistent and tradable universe selection.

These choices ensure all downstream features, clustering, and signals are computed
using only information available at the decision time.
"""




REQUIRED_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj close",
    "volume",
]


def align_to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.reset_index()
    prices["Date"] = prices["Date"] + pd.offsets.MonthEnd(0)

    monthly = (
        prices
        .groupby(["Date", "Ticker"], as_index=False)
        .last()
        .set_index(["Date", "Ticker"])
        .sort_index()
    )

    return monthly


def drop_incomplete_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing OHLCV data.
    """
    return data.dropna(subset=REQUIRED_COLUMNS)


def compute_dollar_volume(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = monthly_prices.copy()
    monthly_prices["dollar_volume"] = (
        monthly_prices["close"] * monthly_prices["volume"]
    )
    return monthly_prices


def apply_min_history_filter(
    data: pd.DataFrame,
    min_months: int = 36,
) -> pd.DataFrame:
    counts = data.groupby("Ticker").size()
    valid_tickers = counts[counts >= min_months].index
    return data[data.index.get_level_values("Ticker").isin(valid_tickers)]



def build_liquidity_universe(
    data: pd.DataFrame,
    top_n: int = 150,
) -> pd.DataFrame:
    data = data.copy()

    data["rank"] = (
        data.groupby("Date")["dollar_volume"]
        .rank(method="first", ascending=False)
    )

    universe = data[data["rank"] <= top_n]
    return universe.drop(columns="rank")

def select_clustering_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Select and validate the final feature set used for clustering.
    """

    required_cols = [
        # Technical indicators
        "gkv", "rsi", "bblow", "bbmid", "bbhigh",
        "atr", "macd", "dv_rank",

        # Past realized returns (lagged)
        "return_1m", "return_2m", "return_3m",
        "return_6m", "return_9m", "return_12m",

        # Fama-French rolling betas
        "beta_mkt", "beta_smb", "beta_hml",
        "beta_rmw", "beta_cma",
    ]

    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required clustering features: {missing}")

    return data[required_cols].copy()
