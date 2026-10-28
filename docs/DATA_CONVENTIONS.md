# Data Conventions

## Where data lives

```
<MR_DATA_ROOT>/            default: <repo>/data   (git-ignored contents)
├── store/                 caches, meta files, paper ledger
└── results/               exports you save by hand
```

Set `MR_DATA_ROOT` in `.env` to move it (for example to a synced folder). Nothing under `data/store/` or `data/results/` is committed.

## Files in `store/`

| File | Written by | Contents | Rebuild with |
|---|---|---|---|
| `pit_members.parquet` | `data_ingestion.loader.load_pit_members` | Date-indexed S&P 500 membership; column `tickers` is a list | `mr build-features --force` |
| `prices_raw.parquet` | `data_ingestion.loader.download_prices` | Daily OHLCV, `(Date, Ticker)` index, lowercase column names | `mr build-features --force` (re-downloads) |
| `modeling_data.parquet` | `features.pipeline.build_modeling_data` | Monthly `(Date, Ticker)` × the 19 `feature_cols` | `mr build-features --force` |
| `monthly_returns.parquet` | same | Raw monthly total return, Date × Ticker, for **every** ticker with prices (not only the universe); used by the backtest and HRP | `mr build-features --force` |
| `price_coverage.json` | `data_ingestion.loader.download_prices` | Requested tickers, which returned prices, which came from vendor files (Yahoo path) or the CRSP delisting counts, which are missing | `mr build-features --force` |
| `pit_members.meta.json` | `data_ingestion.loader` | The `pit_url` the membership cache was built from | with the parquet |
| `modeling_data.meta.json` | same | Parameters and timestamp that produced the file | with the parquet |
| `clustered_<method>.parquet` | `clustering.runner.run_clustering` | Modeling data plus a `cluster` column (HDBSCAN noise is -1) | `mr cluster --force` |
| `clustered_<method>.meta.json` | same | Cluster parameters, timestamp, and the modeling build it came from | with the parquet |
| `paper_ledger.json` | `broker.paper` | Cash, positions, fill history of the simulated account | delete to reset the paper account |

Meta files let `mr` warn when a cache was built with different parameters from the active ones.

## Files in `results/`

Written only when you pass `--save`:

| File | Written by |
|---|---|
| `signals_<month>_<regime>.csv` | `mr signals --save` |
| `trades_<date>_<regime>.csv` | `mr trades --save` |
| `stress_windows_<regime>_<date>.csv` | `python scripts/stress_windows.py --save` |
| `backtest_<regime>_<date>.csv` | `mr backtest --save` (net, gross, benchmark, traded notional by month earned) |
| `walkforward_folds_*.csv`, `walkforward_oos_*.csv` | `mr walkforward --save` |
| `coverage_<date>.csv` | `mr coverage --save` |

Copy anything you want to share into `reports/tables/`.

## Conventions

- **Tickers (CRSP path):** one label per company for its whole life, the last ticker CRSP recorded (`META` for the whole of Facebook's history); a clash becomes `LABEL_permno`.

- **Index:** `Date` is the calendar month-end timestamp; `Ticker` is the symbol as downloaded from yfinance. Always `(Date, Ticker)` in that order.
- **Column names:** lowercase for prices (`adj close`), snake_case for features (`return_3m`, `beta_mkt`).
- **Returns:** `return_<n>m` is the *per-month-equivalent* return over the last *n* calendar months, `(1 + r_n)^(1/n) − 1`, winsorised across stocks within each date. `monthly_returns.parquet` holds the unwinsorised 1-month returns.
- **Time labels:** features and signals are stamped with the month-end they are known at; backtest returns are labelled by the month they were *earned* (one month later).
- **No forward-filling or back-filling** of prices or volumes; rows with incomplete OHLCV at month-end are dropped. The only carry-forward is the FF betas for months the factor library has not published (at most three).
- **Weights:** fractions of NAV (0-1), not percentages, everywhere except display tables (`*_pct` columns in `trades_to_dataframe` are percent).
- **Caches are not self-invalidating.** Use `--force` after changing parameters (see `docs/PARAMETERS.md`).
- **No large files in git.** Keep downloads, parquet and ledgers in `data/`.
