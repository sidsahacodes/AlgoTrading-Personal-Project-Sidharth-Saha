# How Results Are Validated

What the backtest and the walk-forward guarantee, how those guarantees are tested, and what they cannot tell you.

## Timeline

```
month-end d                          month-end d+1
   │  features, clusters, selection,    │
   │  HRP weights use data up to d      │  return earned over (d, d+1] is realised
   └─────────── trade at d ─────────────┴────────────────────────────────────────►
```

- A **signal date** `d` is the decision date. Everything used to decide (features, clusters, HRP history) is available at `d`.
- The return of the *following* month is the only future quantity the simulator touches, and it is used only to score the decision.
- `mr backtest` labels each return by the month it was **earned** (d+1), so it lines up with SPY. The walk-forward folds use signal dates.

## Point-in-time features

Every feature stamped with month-end `t` is built from data available at `t`:

| Feature | How it stays point-in-time |
|---|---|
| Technical indicators | Backward-looking windows only; ATR is z-scored against each ticker's own history *up to that day* (expanding window) |
| Lagged returns | Calendar-month lags from a complete month-end price panel; winsorised **across stocks within each date**, never against a stock's full history |
| FF5 betas | Excess returns; full rolling windows only; the beta stamped at `t` uses data through `t-1` (the factor library publishes late); when recent factor months are unpublished the last beta is carried forward for at most 3 months |
| Universe | Index membership **as of that month-end**; a stock needs 36 month-end rows *before* the date; it must have traded within `max_stale_days` of the month-end |

## The simulator

`strategy/simulate.py` runs the same path as live trading, one month at a time:

1. `select_top_momentum` on that month's clusters (same function and `portfolio_size` cap as live).
2. Weights from `weights_from_history`, the same function live trading uses (HRP or equal), on returns **up to and including `d`**, then × `exposure_scalar`.
3. The trade band: entries and resizes smaller than `min_trade_pct` are skipped; exits always trade.
4. Hold through the next month using **actual returns from the full price panel**, so a stock that leaves the universe or the index still earns what it earned.
5. Costs: `traded notional × cost_bps / 10,000`, where traded notional is measured against the *drifted* weights (not the previous target).
6. Drift the weights to the next rebalance. A stock with no price at all in the next month gets `delisting_return` and is cashed out; a stock priced later but missing that month counts 0%. Both are counted and printed. With CRSP data the delisting return is already in the price series, so this assumption is rarely triggered.

## How the guarantees are tested

| Guarantee | Test |
|---|---|
| Features never use the future | `tests/test_no_lookahead.py` builds the whole feature set from all prices and from prices cut at mid-2023; every row up to the cut must be identical |
| A ticker's indicators do not depend on other tickers | `tests/test_no_lookahead.py` |
| Beta timing, carry-forward and excess returns | `tests/test_return_features.py` |
| Decisions on or before `d` do not change if later returns or later features are altered | `tests/test_simulate.py` |
| HRP weights equal the shared helper applied to history up to `d` | `tests/test_simulate.py` |
| Costs, band, exposure scalar, exits, delistings, data gaps | hand-computed cases in `tests/test_simulate.py` |
| Walk-forward blocks are ordered, do not overlap, and selection ignores the test block | `tests/test_walkforward.py` (including an "oracle" feature the walk-forward must find, and a tamper test) |

The simulator and walk-forward tests were checked by deliberately breaking the code (peeking at the test block, giving HRP the full history, removing costs, removing the beta lag) and confirming a test fails each time.

## Walk-forward

`mr walkforward` (settings in `configs/walk_forward.yaml`).

Nothing in the strategy is fitted on prices: clusters are refit each month and HRP uses past returns only. What *is* chosen using data are the hyperparameters in the grid: momentum lookback, names per cluster and number of clusters. Choosing them on the whole sample and reporting the same sample would be look-ahead, so for each fold:

```
|-------- train --------|-- validation --|-- embargo --|--- test ---|
 reported, not used       picks the config               never used to choose
```

- Every config in the grid is simulated net of costs. The config with the best **validation** objective is frozen, and only then are its **test** months recorded.
- The train block expands; blocks step forward by `step_months`. The stitched test months are the out-of-sample record.
- Defaults: train 48, validation 12, embargo 1, test 12, step 12 signal months; objective net Sharpe; grid 3 × 3 × 3 = 27 configs. With about 12 years of eligible signals this gives about seven folds.
- The report compares the walk-forward with the fixed baseline configuration over the same months, shows which configs were chosen, and prints the mean Sharpe of the chosen config on validation versus test. **A large drop from validation to test is selection on noise.**

Runtime scales with the grid: 27 configs took about 4.5 minutes on a synthetic 12-year, 70-stock panel; a single backtest about 12 seconds.

## What this cannot tell you

- **Whether the data are complete.** Missing delisted names cannot be corrected by any simulation. On the Yahoo path run `mr coverage`; the CRSP path ([`DATA_SOURCES.md`](DATA_SOURCES.md)) removes most of the gap.
- **Whether the costs are right.** `cost_bps` and `delisting_return` are assumptions. Re-run with `--cost-bps` and `--delist-return` to see how much the result depends on them.
- **Hyperparameters outside the grid** (universe size, feature set, FF window, the 3-month momentum definition) were chosen by people who had seen the history. Walk-forward cannot undo that.
- **Statistical significance.** Few folds, a 27-config grid, and no correction for multiple testing.
- **Any real-data result.** The tests use synthetic data; see [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).
