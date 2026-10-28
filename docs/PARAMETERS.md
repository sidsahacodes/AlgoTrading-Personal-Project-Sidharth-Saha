# Parameter Reference

Every parameter, its YAML key, where the code reads it, and **what you must re-run for it to take effect**. Getting that last column wrong is the easiest way to look at stale results.

Re-run levels:

- **L0 none**: read at signal/trade/backtest time; effective immediately.
- **L1** `mr cluster --force`: re-cluster from cached features.
- **L2** `mr build-features --force`: rebuild features **and re-download all prices** (15 years, several hundred tickers; allow 15-20+ minutes), because `--force` is passed through to the download.
- **Code**: hard-coded in a module; edit the file, then re-run the relevant level.

`mr` records the parameters used for each cache (`*.meta.json`) and prints a warning if the active config differs, but it does not rebuild automatically.

Baseline values are in `configs/strategy.yaml`; overlays in `configs/regimes.yaml`; walk-forward settings in `configs/walk_forward.yaml`. A regime preset may override any key in `strategy.yaml`.

## Universe (L2)

| Key | Default | Read in | Effect |
|---|---|---|---|
| `lookback_years` | 15 | `data_ingestion/loader.py` | Years of prices downloaded. The first ~3 years are warm-up (36 months of history, then the beta window), so the evaluable record is shorter |
| `min_history_months` | 36 | `data_cleaning/universe.py` | A ticker is eligible on a date only after it has this many month-end rows *before* that date |
| `universe_size` | 150 | `data_cleaning/universe.py` | Top-N by 60-month rolling dollar volume, per month |
| `max_stale_days` | 7 | `data_cleaning/universe.py` | A stock must have a price bar within this many days of the month-end to be tradable then |
| `price_source` | `yahoo` | `data_ingestion/loader.py` | `yahoo` or `crsp` (WRDS). Env `MR_PRICE_SOURCE` overrides. Use a separate `MR_DATA_ROOT` per source; switching sources makes the caches look stale |
| `pit_url` | maintained "(Updated)" file | `data_ingestion/loader.py` | Source of the index membership history (Yahoo path) |
| `pit_allow_fallback` | `false` | `data_ingestion/loader.py` | `false`: stop if the membership list cannot be downloaded. `true`: use today's members for every date (survivorship-biased) |

## Features (L2)

| Key | Default | Read in | Effect |
|---|---|---|---|
| `rsi_period`, `atr_period`, `bb_period` | 14, 14, 20 | `features/technical.py` | Indicator windows (days) |
| `macd_fast/slow/signal` | 12/26/9 | `features/technical.py` | MACD parameters |
| `ff_beta_window` | 24 | `features/returns.py` | Rolling window (valid months) for FF5 betas; 24 observations for 6 coefficients is thin |
| `return_lags` | 1,2,3,6,9,12 | `features/returns.py` | Which `return_<n>m` columns are built; **must match `feature_cols`** (validated on load) |
| `feature_cols` | 19 names | `features/pipeline.py`, `clustering/runner.py` | Exactly which columns are clustered on |

## Clustering (L1)

| Key | Default | Read in | Effect |
|---|---|---|---|
| `cluster_method` | `kmeans` | `clustering/`, `strategy/targets.py` | `kmeans` / `gmm` / `pca_kmeans` / `hdbscan` (the walk-forward grid over `n_clusters` needs one of the first three) |
| `n_clusters` | 5 | `clustering/methods.py` | Clusters (K-Means, PCA+K-Means) or components (GMM); ignored by HDBSCAN |
| `pca_components` | 6 | `clustering/methods.py` | `pca_kmeans` only |
| `hdbscan_min_size` | 15 | `clustering/methods.py` | With ~150 stocks/month a large value can yield very few clusters |
| `random_state` | 42 | `clustering/methods.py` | Reproducibility |

## Strategy (L0)

These apply to live trading **and** to `mr backtest` / `scripts/stress_windows.py`.

| Key | Default | Read in | Effect |
|---|---|---|---|
| `momentum_feature` | `return_3m` | `strategy/signals.py`, `simulate.py` | Ranking column inside each cluster; must be in `feature_cols` |
| `top_n_per_cluster` | 4 | `strategy/signals.py`, `simulate.py` | Names per cluster |
| `portfolio_size` | 20 | `strategy/signals.py`, `simulate.py` | Hard cap; trims the lowest `momentum_score`. Keep `n_clusters × top_n_per_cluster = portfolio_size` (checked for every preset in tests). `run_backtest(top_n=...)` with a non-default `top_n` uses `top_n × n_clusters` |
| `optimiser` | `hrp` | `strategy/targets.py`, `simulate.py` | `hrp` or `equal` |
| `hrp_lookback_months` | `null` | `strategy/targets.py`, `simulate.py` | `null` = all history up to the decision date. Else the most recent N months only; use ≥ 36 for ~20 assets |
| `exposure_scalar` | 1.0 | `strategy/targets.py`, `simulate.py` | Multiplies all target weights; the remainder is cash earning 0% |
| `min_trade_pct` | 0.0 | `strategy/rebalancer.py`, `simulate.py` | Skips entries/resizes whose weight change is below this (0.005 = 0.5%). Exits are never skipped |

## Backtest assumptions (L0)

| Key | Default | Read in | Effect |
|---|---|---|---|
| `cost_bps` | 10 | `strategy/simulate.py` | Cost per unit of traded notional, one way, in basis points. An **assumption**; override with `--cost-bps` |
| `delisting_return` | -0.30 | `strategy/simulate.py` | Return given to a held stock with no price at all in the next month. An **assumption**; override with `--delist-return` |

## Regime monitor (L0, used by `mr regime`)

`spy_dd_caution`, `spy_dd_stress`, `vix_caution`, `vix_stress`, `corr_jump`, `cluster_max_share`. Illustrative thresholds; calibrate them against your own history. They only drive flags in the report; nothing acts on them automatically.

## Walk-forward (`configs/walk_forward.yaml`, L0)

| Key | Default | Effect |
|---|---|---|
| `splits.train_months` | 48 | Initial training block (reported, not used to choose) |
| `splits.val_months` | 12 | Block on which the config is chosen |
| `splits.test_months` | 12 | Out-of-sample block |
| `splits.step_months` | 12 | Fold-to-fold step; must be ≥ `test_months` |
| `splits.embargo_months` | 1 | Gap between the last validation signal and the first test signal |
| `objective` | `sharpe` | `sharpe` or `cumulative_return`, net of costs, on the validation block |
| `min_months` | 12 | A config needs this many validation months to be eligible |
| `grid.momentum_feature` | `return_3m, return_6m, return_12m` | Candidates |
| `grid.top_n_per_cluster` | 3, 4, 5 | Candidates |
| `grid.n_clusters` | 4, 5, 6 | Candidates (`portfolio_size` is set to `top_n × n_clusters`) |

The grid multiplies: 27 configs by default. Each extra value adds simulation time and another chance for the winner to be luck.

## Broker & dashboard

| Key | Default | Effect |
|---|---|---|
| `broker` | `paper` | `paper` (local ledger, fills at last yfinance close) or `ibkr`. Env `MR_BROKER` overrides |
| `portfolio_value` | 1,000,000 | Fallback NAV and starting cash of a fresh paper ledger |
| `tws_host`, `tws_port` | 127.0.0.1, 7497 | 7497 = paper, 7496 = live |
| `dashboard_host/port/debug` | 127.0.0.1, 5001, false | Keep the host on localhost |

The IBKR client id is chosen at random for each connection, so there is no `client_id` setting.

## Hard-coded values worth knowing (Code)

| Value | File | What it does |
|---|---|---|
| `OUTLIER_CUTOFF = 0.005` | `features/returns.py` | Winsorises lagged returns at the 0.5/99.5 percentile **across stocks within each date** |
| `FF_MAX_CARRY_MONTHS = 3` | `features/returns.py` | Months a beta may be carried forward when factor data are unpublished |
| `min_periods=60` | `features/technical.py` | Days of history before the expanding ATR z-score is defined |
| `.rolling(60)` | `data_cleaning/universe.py` | 60-month window for the dollar-volume rank |
| `MIN_HRP_OBS = 24` | `strategy/optimiser.py` | Months of returns a ticker needs to enter HRP |
| `method="single"` | `strategy/optimiser.py` | HRP linkage |
| `min_trade_value = 100.0` | `strategy/rebalancer.py` | Ignores trades under $100 notional |
| `BATCH = 100` | `data_ingestion/loader.py` | yfinance batch size |
| `ib.sleep(2)`, `ib.sleep(1)` | `broker/ibkr.py` | Waits for market data / order status |

## CRSP / WRDS (`configs/crsp.yaml`, used when `price_source: crsp`)

| Key | Default | Effect |
|---|---|---|
| `wrds.username_env` | `WRDS_USERNAME` | Environment variable holding your WRDS username |
| `libraries.stock`, `libraries.indexes` | `crsp_m_*`, `crsp_q_*`, `crsp_a_*`, `crsp` | Libraries searched, in order, for each table |
| `tables.daily` / `membership` / `security_info` | `dsf_v2` / `dsp500list_v2` or `msp500list_v2` / `stksecurityinfohist` | CIZ tables used |
| `columns.*` | CIZ names (`dlycaldt`, `dlyret`, ...) | Map this project's fields to your subscription's columns; edit if `mr crsp-check` reports a missing column |
| `extend_membership` | `false` | `false`: the sample ends where CRSP's S&P 500 list ends. `true`: later months reuse its last list (not point-in-time) |

`cost_bps` and `delisting_return` apply to both sources; with CRSP, `delisting_return` is also used for any delisting row whose return is missing. Changing anything here needs L2 (`mr build-features --force`).

## Environment variables

| Variable | Effect |
|---|---|
| `MR_REGIME` | Regime preset name |
| `MR_BROKER` | `paper` / `ibkr` |
| `MR_PRICE_SOURCE` | `yahoo` / `crsp` |
| `WRDS_USERNAME` | Your WRDS username (name configurable in `crsp.yaml`) |
| `MR_DATA_ROOT` | Where `store/`, `results/` and `external/` live |
| `MR_CONFIG_DIR` | Folder holding the YAML configs |
