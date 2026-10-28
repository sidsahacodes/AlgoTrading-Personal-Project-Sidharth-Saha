# `features/` – the 19 modelling features

Builds two files: `data/store/modeling_data.parquet` (one row per `(Date, Ticker)` with exactly the columns in `feature_cols`) and `data/store/monthly_returns.parquet` (raw monthly total return for **every** ticker with prices, including stocks outside the universe; the backtest and HRP use it).

Every value stamped with month-end *t* uses only data available at *t*; `tests/test_no_lookahead.py` checks this by removing the future and confirming nothing earlier changes.

| File | Role |
|---|---|
| `technical.py` | Daily indicators, snapshotted at month-end |
| `returns.py` | Month-end price panel, lagged returns, cross-sectional winsorising and z-scoring, rolling Fama-French betas |
| `pipeline.py` | Orchestrates the steps; `build_modeling_data()` / `load_modeling_data()`; writes the parameter fingerprint |

| Group | Columns | Notes |
|---|---|---|
| Volatility | `gkv` | Garman-Klass estimator (OHLC) |
| Oscillators | `rsi`, `macd` | Windows from config; MACD uses `macd_fast/slow/signal` |
| Bands | `bblow`, `bbmid`, `bbhigh` | Bollinger bands on `log1p(price)` |
| Range | `atr` | z-scored against the ticker's own history up to that day (expanding window) |
| Liquidity | `dv_rank` | Rank of 60-month rolling dollar volume |
| Returns | `return_1m … return_12m` | Calendar-month lags, winsorised across stocks within each date; per-month-equivalent `(1+r)^(1/n) − 1` |
| Factors | `beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma` | Rolling OLS of excess returns on FF5 factors over `ff_beta_window` valid months (full windows only); stamped with a one-month lag, because the factor library publishes late |

`gkv, rsi, bblow, bbmid, bbhigh, atr, macd, dv_rank` are z-scored **cross-sectionally** each month. Returns and betas are **not** scaled.

Order: prices → daily technicals → month-end snapshot → universe → join → lagged returns (from the complete month-end panel) → z-score → FF5 betas (from the full return panel) → select `feature_cols` → parquet + `.meta.json`. Months for which the factor library has not published yet carry the last beta forward, for at most three months.

**Parameters** (all L2): `rsi_period`, `atr_period`, `bb_period`, `macd_*`, `ff_beta_window`, `return_lags`, `feature_cols`. `return_lags` and `feature_cols` must agree (validated on load).

**In stress:** features are *relative* (z-scored), so they cannot tell you the absolute volatility level; use `mr regime` for that. Lengthening `ff_beta_window` (24 → 36) steadies betas; shortening makes 6-coefficient regressions on fewer than 24 points noisier.

**Gotchas:** returns/betas are on a much smaller numeric scale than the z-scored columns (A4); the newest betas can be up to three months stale when the factor library lags (B6).

**Testing:** `tests/test_feature_pipeline.py` runs this whole pipeline on synthetic prices with the network stubbed out; `tests/test_no_lookahead.py` and `tests/test_return_features.py` check the point-in-time properties and the beta timing.
