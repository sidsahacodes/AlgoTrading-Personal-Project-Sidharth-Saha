# `strategy/` – from clusters to trades, and how they are evaluated

| File | Role |
|---|---|
| `signals.py` | `select_top_momentum()`: top N by momentum within each cluster, capped at `portfolio_size` |
| `optimiser.py` | `hrp` / `equal` weights, and `weights_from_history()`, the single function both live trading and the backtest use |
| `targets.py` | **Live path**: `current_selection()`, `target_weights()`, `plan_rebalance()` |
| `rebalancer.py` | `compute_trades()`: target vs. current weights → sells-then-buys list, dollar floor and optional % band |
| `simulate.py` | **Engine**: month-by-month, point-in-time simulation with costs, drift, real exit returns |
| `backtest.py` | `run_backtest()` on top of the engine; metrics, SPY benchmark, period slicing |
| `walkforward.py` | Out-of-sample train / validation / test evaluation of the hyperparameters |
| `regime.py` | Regime indicators (pure functions), the `mr regime` report, and `window_metrics()` for stress windows |

## Live path (`mr signals / trades / rebalance`)

```
latest month's clusters
  └─ select_top_momentum(momentum_feature, top_n_per_cluster)     noise cluster -1 dropped
       └─ trim to portfolio_size by momentum_score
            └─ weights_from_history():  HRP on raw monthly returns (all history or last hrp_lookback_months) or equal
                 └─ × exposure_scalar                              remainder is cash
                      └─ compute_trades(min_trade_value, min_trade_pct)
                           └─ broker.execute(trades)
```

**HRP in two lines.** Cluster the selected stocks by correlation distance `sqrt(0.5·(1−ρ))` with single linkage, order them, then split risk recursively between halves in inverse proportion to sub-portfolio variance. No matrix inversion and no expected-return inputs. A stock needs at least 24 months of returns to enter HRP; fewer than two such stocks, or an HRP failure, falls back to equal weight.

**Rebalancer.** Ignores trades under `min_trade_value` ($100). Entries and resizes with a weight change below `min_trade_pct` are skipped; **full exits are never skipped by the band**. Sells come first. `Trade.shares` is 0 here; the broker sizes shares at execution.

## Backtest (`mr backtest`, `run_backtest`)

The engine in `simulate.py` follows the live path one month at a time: same selection function, same weighting function, exposure scalar and band. It uses returns up to the decision date for HRP, earns the **actual next-month return from the full price panel** (so a stock that left the universe still earns what it earned), and charges `cost_bps` on traded notional measured against drifted weights. Returns are labelled by the month they were earned and compared with SPY. If SPY cannot be downloaded the benchmark metrics are omitted; they are never replaced by 0%.

Reported: net and gross annualised return, volatility, Sharpe, Sortino (downside deviation), Calmar, Omega, max drawdown, win rate, alpha/beta, average monthly traded notional, annual cost drag, and counts of held names with no next-month price (delisted) or a data gap.

Assumptions you can override: `--cost-bps`, `--delist-return`. Details and limits: [`../../../docs/VALIDATION.md`](../../../docs/VALIDATION.md).

## Walk-forward (`mr walkforward`)

Chooses the momentum lookback, names per cluster and number of clusters on a validation block using only earlier data, freezes the winner, and records its test block. See [`../../../docs/VALIDATION.md`](../../../docs/VALIDATION.md) and `configs/walk_forward.yaml`.

## Parameters

All **L0**: `momentum_feature`, `top_n_per_cluster`, `portfolio_size`, `optimiser`, `hrp_lookback_months`, `exposure_scalar`, `min_trade_pct`, `cost_bps`, `delisting_return`. Hard-coded: HRP linkage, `MIN_HRP_OBS`, `min_trade_value`.

## In stress

See [`../../../docs/REGIME_PLAYBOOK.md`](../../../docs/REGIME_PLAYBOOK.md). Levers that live here: `momentum_feature` (slower signal), `top_n_per_cluster`/`portfolio_size` (breadth), `hrp_lookback_months` (react faster; stay ≥ 36 for ~20 assets), `exposure_scalar` (cash), `min_trade_pct` (less churn). All of them can be evaluated with `scripts/stress_windows.py`.

## Gotchas

Per-cluster stats group by arbitrary cluster ids (A2). Missing-price returns are assumptions (A1), costs are an assumption (A3), and the backtest trades at the month-end close (A10). See [`../../../docs/KNOWN_LIMITATIONS.md`](../../../docs/KNOWN_LIMITATIONS.md).
