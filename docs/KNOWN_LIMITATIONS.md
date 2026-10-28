# Known Limitations

What to keep in mind before trusting a number or sending an order. Found by reading the code and running synthetic tests; nothing here has been run on a real-data backtest. ⚠ marks items that affect real orders.

## A. Backtest and evaluation

| # | Issue | Where | Consequence |
|---|---|---|---|
| A1 | **Missing-price returns are assumed.** A held stock with no price at all in the next month gets `delisting_return` (default -30%); one with a gap gets 0% | `strategy/simulate.py` | Both are counted and printed by `mr backtest`; re-run with `--delist-return` to see the sensitivity. The right value depends on whether the stock was acquired or failed |
| A2 | **Cluster ids are arbitrary each month** (K-Means "cluster 2" now is not last month's cluster 2) | `clustering/methods.py`, `backtest._cluster_performance`, dashboard cluster tab | Per-cluster performance averages unrelated groups; the tab is not meaningful |
| A3 | **Costs are an assumption.** A flat `cost_bps` on traded notional; no market impact, no spread differences across names, no taxes, no financing. Monthly turnover can be near 100% of NAV | `configs/strategy.yaml` `backtest` | Net results move with the assumption; re-run with `--cost-bps` |
| A4 | **Feature scale in clustering.** Z-scored features (~±1) sit beside raw `return_*` (~0.01-0.03) and betas; Euclidean distance is dominated by the larger-scale columns | `features/pipeline.py`, `clustering/methods.py` | Momentum features may barely influence clusters. Check `modeling_data.std()` |
| A5 | **Walk-forward is statistically thin.** About seven folds, a 27-config grid, no correction for multiple testing; the validation winner is partly luck | `strategy/walkforward.py` | Read the validation-to-test Sharpe drop and how often the same config is chosen, not the headline number |
| A6 | **No real-data run is recorded here.** All tests and demonstrations use synthetic data | whole project | Behaviour on Yahoo data (coverage, delistings, factor-library lag) and on CRSP data (table and column names on your subscription, the update lag) is unverified until you run it |
| A7 | **Hyperparameters outside the grid were chosen with history in view** (universe size, feature set, FF window, the 3-month momentum definition) | `configs/strategy.yaml` | Walk-forward cannot remove that selection |
| A8 | **Regime presets are untested hypotheses**, and the backtest applies a preset to the whole sample or window rather than switching it on by a rule | `configs/regimes.yaml` | Use `scripts/stress_windows.py` to see a preset's effect in a window, not to prove it works |
| A9 | **Monitor thresholds are uncalibrated** | `configs/strategy.yaml` `regime_monitor` | Flags may fire too often or too rarely |
| A10 | **Execution timing differs.** The backtest trades at the month-end adjusted close; live orders are market orders sent after the close, filling at later prices | `strategy/simulate.py` vs `broker/` | Live slippage is not in the backtest beyond `cost_bps` |

## B. Data

| # | Issue | Where | Consequence |
|---|---|---|---|
| B1 | **Yahoo lacks many removed and delisted names.** Index members without prices are invisible to the backtest | `data_ingestion/loader.py` | Residual survivorship bias on the Yahoo path; measure with `mr coverage`. The CRSP path (`price_source: crsp`) closes it ([`DATA_SOURCES.md`](DATA_SOURCES.md)) |
| B2 | **Renamed companies on the Yahoo path.** The membership list has the symbol of the time; Yahoo has the history under the new symbol only | `data_ingestion/loader.py` | The company is missing for its old period unless a vendor file supplies it under the old symbol. CRSP uses one label per company |
| B3 | **The membership list ends at its last update** (Yahoo path: checked against a 120-day freshness threshold; CRSP path: the sample is cut at the list's end) | `configs/strategy.yaml` `pit_url`, CRSP list | Later index changes are missed; on the Yahoo path the last member list is reused |
| B4 | **Adjusted closes are re-based by Yahoo**, so downloads are never appended and re-downloading can change past values | `data_ingestion/loader.py` | Results can shift slightly between refreshes |
| B5 | **Caches do not rebuild themselves.** A fingerprint warns when a cache was built with different parameters, but you must re-run with `--force` | `storage/fingerprint.py` | Stale results if the warning is ignored |
| B6 | **The factor library publishes with a delay.** For recent months the last available beta is carried forward for at most three months | `features/returns.py` | The newest betas are slightly stale |
| B7 | **Dependency risk.** The `ib_insync` repository was archived on 14 Mar 2024 after its author's passing; [`ib_async`](https://github.com/ib-api-reloaded/ib_async) is the maintained successor. `pandas-ta` also has several incompatible release lines | `requirements.txt`, `broker/ibkr.py` | Future breakage, e.g. when IBKR changes its API. Migrating means changing the imports in `broker/ibkr.py`; test on paper first |

| B8 | **The CRSP queries have not been run against WRDS from here.** They follow WRDS and CRSP documentation for the CIZ tables and are tested against a fake connection | `data_ingestion/crsp.py`, `configs/crsp.yaml` | Run `mr crsp-check` first; a different table or column name on your subscription is a `crsp.yaml` edit |
| B9 | **CRSP lags today.** Its S&P 500 list is updated annually and the stock files quarterly or monthly, so a CRSP research sample ends months before today | `data_ingestion/crsp.py` | The newest months are not in the CRSP research set; use the Yahoo path for live signals. Joining the two into one series is not built |
| B10 | **Missing delisting returns are imputed** with `delisting_return` (default -30%) when CRSP has none for a delisting row | `data_ingestion/crsp.py` | Counted in `price_coverage.json`; the right value depends on whether the company failed or was acquired |
| B11 | **CRSP daily bars: no-trade days become flat bars** (open = high = low = close = price) when CRSP has no open/high/low; early CRSP history has no open prices | `data_ingestion/crsp.py` | Slightly understates range-based indicators on those days |

## C. Execution & safety ⚠

| # | Issue | Where | Consequence |
|---|---|---|---|
| C1 ⚠ | **Exit orders are sized from weight, not from the actual position:** `shares = int(\|delta\| × NAV / price)` | `broker/ibkr.py` | A full exit can leave a residual share, or oversell by one and open a short. `paper.py` caps with `min(shares, held)`; `ibkr.py` does not. **Resolve before using `ibkr` with real money** |
| C2 ⚠ | **The dashboard has no authentication.** POST endpoints require a JSON content-type and a same-origin `Origin`, which blocks cross-site pages, but any local process can still call them | `dashboard/app.py` | Keep the host on 127.0.0.1 |
| C3 ⚠ | **Market orders, no fill confirmation.** Status is read ~1 s after placement and the recorded price is the *estimated* price | `broker/ibkr.py` | Fill records are not real fills; poor prices in stress |
| C4 | **Market-data subscriptions are never cancelled** | `broker/ibkr.py` `_get_price` | Can hit TWS data-line limits |
| C5 | **Long-only assumption:** holdings use `abs(marketValue)` | `broker/ibkr.py` | Existing shorts would look like longs |
| C6 | **The dry run shows 0 shares.** `Trade.shares` is a placeholder; the broker sizes shares at execution | `strategy/rebalancer.py` | `mr trades` shows weights, not quantities |
