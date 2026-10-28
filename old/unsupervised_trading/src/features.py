import pandas as pd
import numpy as np
import pandas_ta as ta
import statsmodels.api as sm
from pandas_datareader import data as web
from statsmodels.regression.rolling import RollingOLS


# ==================================================
# DAILY TECHNICAL INDICATORS
# ==================================================
def compute_daily_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()

    # ---------------------------
    # Garman–Klass volatility
    # ---------------------------
    data["gkv"] = (
        ((np.log(data["high"]) - np.log(data["low"])) ** 2) / 2
        - (2 * np.log(2) - 1)
        * ((np.log(data["adj close"]) - np.log(data["open"])) ** 2)
    )

    # ---------------------------
    # RSI (1D → transform OK)
    # ---------------------------
    data["rsi"] = (
        data.groupby("Ticker")["adj close"]
        .transform(lambda x: ta.rsi(x, length=14))
    )

    # ---------------------------
    # MACD (select ONE column)
    # ---------------------------
    macd = (
        data.groupby("Ticker", group_keys=False)
        .apply(lambda x: ta.macd(x["adj close"]).iloc[:, 0])
    )
    data["macd"] = macd

    # ---------------------------
    # ATR (apply, normalize per stock)
    # ---------------------------
    atr = (
        data.groupby("Ticker", group_keys=False)
        .apply(lambda x: ta.atr(x["high"], x["low"], x["close"], length=14))
    )
    if isinstance(atr, pd.DataFrame):
        atr = atr.iloc[:, 0]

    data["atr"] = atr.groupby(data.index.get_level_values("Ticker")).transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # ---------------------------
    # Bollinger Bands (3 columns)
    # ---------------------------
    bb = (
        data.groupby("Ticker", group_keys=False)
        .apply(lambda x: ta.bbands(np.log1p(x["adj close"]), length=20))
    )

    data["bblow"]  = bb.iloc[:, 0]
    data["bbmid"]  = bb.iloc[:, 1]
    data["bbhigh"] = bb.iloc[:, 2]

    return data



# ==================================================
# MONTH-END SNAPSHOT
# ==================================================
def snapshot_month_end_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.reset_index()
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)

    return (
        df.groupby(["Date", "Ticker"])
        .last()
        .sort_index()
    )


# ==================================================
# PAST-HORIZON RETURNS (FEATURES)
# ==================================================
def compute_past_returns(df: pd.DataFrame) -> pd.DataFrame:
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        r = df["adj close"].pct_change(lag)
        r = r.clip(r.quantile(outlier_cutoff), r.quantile(1 - outlier_cutoff))
        df[f"return_{lag}m"] = (1 + r).pow(1 / lag) - 1

    return df


# ==================================================
# CROSS-SECTIONAL Z-SCORE
# ==================================================
def cross_sectional_zscore(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data = data.copy()
    for col in cols:
        data[col] = (
            data.groupby("Date")[col]
            .transform(lambda x: (x - x.mean()) / x.std())
        )
    return data


# ==================================================
# ROLLING FAMA–FRENCH BETAS (FINAL FEATURES)
# ==================================================
def estimate_rolling_ff_betas(
    data: pd.DataFrame,
    window: int = 24,
) -> pd.DataFrame:
    """
    Estimate rolling FF5 betas using lagged factor data.
    """

    start_date = data.index.get_level_values("Date").min()

    ff = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start=start_date,
    )[0]

    ff = (
        ff.drop(columns="RF")
        .div(100)
        .to_timestamp("M")
        .shift(1)
    )
    ff.index.name = "Date"

    merged = data.join(ff, how="inner")

    def fit(x):
    # Require enough observations
        if x.shape[0] < 24:
            return pd.DataFrame(
                index=x.index,
                columns=["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"],
            )

        y = x["return_1m"]
        X = sm.add_constant(x[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])

        model = RollingOLS(
            endog=y,
            exog=X,
            window=24,
            min_nobs=6 + 1,  # regressors + constant
        ).fit(params_only=True)

        betas = model.params.drop(columns="const")
        betas.columns = [
            "beta_mkt",
            "beta_smb",
            "beta_hml",
            "beta_rmw",
            "beta_cma",
        ]

        return betas


    return (
        merged.groupby("Ticker", group_keys=False)
        .apply(fit)
        .dropna()
    )
