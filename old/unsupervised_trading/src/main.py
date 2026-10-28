from pathlib import Path
import pandas as pd
from pandas_datareader import data as web
from data_loader import load_price_data

from preprocessing import (
    align_to_month_end,
    drop_incomplete_rows,
    compute_dollar_volume,
    apply_min_history_filter,
    build_liquidity_universe,
    select_clustering_features
)

from features import (
    compute_daily_indicators,
    snapshot_month_end_features,
    compute_past_returns,
    cross_sectional_zscore,
    estimate_rolling_ff_betas,
)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    raw_path = data_dir / "prices_raw.csv"

    # --------------------------------------------------
    # 1. LOAD DAILY PRICE DATA
    # --------------------------------------------------
    if not raw_path.exists():
        print("prices_raw.csv not found — loading data")
        prices = load_price_data(lookback_years=10)
        prices.to_csv(raw_path)
    else:
        prices = pd.read_csv(raw_path, parse_dates=["Date"])
        prices = prices.set_index(["Date", "Ticker"]).sort_index()

    # --------------------------------------------------
    # 2. DAILY TECHNICAL INDICATORS
    # --------------------------------------------------
    daily = compute_daily_indicators(prices)

    # --------------------------------------------------
    # 3. MONTH-END SNAPSHOT OF TECHNICALS
    # --------------------------------------------------
    monthly_features = snapshot_month_end_features(daily)

    tech_cols = [
        "gkv",
        "rsi",
        "bblow",
        "bbmid",
        "bbhigh",
        "atr",
        "macd",
    ]
    monthly_features = monthly_features[tech_cols]

    # --------------------------------------------------
    # 4. BUILD MONTHLY LIQUIDITY UNIVERSE
    # --------------------------------------------------
    monthly_prices = align_to_month_end(prices)
    monthly_prices = drop_incomplete_rows(monthly_prices)
    monthly_prices = compute_dollar_volume(monthly_prices)
    monthly_prices = apply_min_history_filter(monthly_prices)
    universe = build_liquidity_universe(monthly_prices)

    # --------------------------------------------------
    # 5. DOLLAR VOLUME RANK (5-YEAR ROLLING)
    # --------------------------------------------------
    dv = prices.copy()
    dv["dv"] = (dv["adj close"] * dv["volume"]) / 1e6

    monthly_dv = (
        dv["dv"]
        .unstack("Ticker")
        .resample("M")
        .mean()
        .stack("Ticker")
        .to_frame("dv")
    )

    monthly_dv["dv_rolling"] = (
        monthly_dv.groupby("Ticker")["dv"]
        .transform(lambda x: x.rolling(60, min_periods=1).mean())
    )

    dv_rank = (
        monthly_dv.groupby("Date")["dv_rolling"]
        .rank(ascending=False)
        .to_frame("dv_rank")
    )

    # --------------------------------------------------
    # 6. JOIN UNIVERSE + TECHNICALS + LIQUIDITY
    # --------------------------------------------------
    data = (
        universe
        .join(monthly_features, how="inner")
        .join(dv_rank, how="inner")
    )

    # --------------------------------------------------
    # 7. PAST-HORIZON RETURNS (FEATURES)
    # --------------------------------------------------
    data = (
        data.groupby("Ticker", group_keys=False)
        .apply(compute_past_returns)
        .dropna()
    )

    # --------------------------------------------------
    # 8. CROSS-SECTIONAL Z-SCORE (STOCK-RELATIVE ONLY)
    # --------------------------------------------------
    z_cols = ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank"]
    data = cross_sectional_zscore(data, z_cols)

    # --------------------------------------------------
    # 9. ROLLING FAMA–FRENCH BETAS
    # --------------------------------------------------
    betas = estimate_rolling_ff_betas(data)
    data = data.join(betas, how="inner")

    # Final feature selection for clustering
    # --------------------------------------------------
    data = select_clustering_features(data)
    # --------------------------------------------------
    # 10. SAVE FINAL MODELING DATASET
    # --------------------------------------------------
    output_path = data_dir / "modeling_data.csv"
    data.to_csv(output_path)
    print(f"Saved modeling dataset → {output_path}")


if __name__ == "__main__":
    main()
