# `momentum_rebalancer/` – package map

Data flows left to right; each package has its own README.

```
data_ingestion ─► data_cleaning ─► features ─► clustering ─► strategy ─► broker
 (prices, PIT)    (universe)       (19 feats)   (monthly)    (signals,    (paper | IBKR)
                                                              HRP, trades)
        config  ◄── every package reads parameters from here (configs/*.yaml)
        storage ── cache fingerprints       validation ── health checks
        dashboard ── web UI over strategy + broker        cli ── `mr` commands
```

| Package / file | Role | Who calls it |
|---|---|---|
| `config.py` | Loads and validates YAML, applies the regime overlay | everything |
| `cli.py`, `__main__.py` | `mr ...` commands | you |
| `data_ingestion/` | PIT membership, prices, vendor files, symbol normalisation | `features.pipeline` |
| `data_cleaning/` | Monthly universe, coverage report | `features.pipeline`, `cli` |
| `features/` | Builds `modeling_data.parquet` | `cli`, `dashboard` |
| `clustering/` | Monthly clusters, cached | `cli`, `strategy.targets`, `strategy.backtest` |
| `strategy/` | Selection, weights, trades, simulation engine, backtest, walk-forward, regime report | `cli`, `dashboard`, `scripts/` |
| `broker/` | Holdings, NAV and execution | `strategy.targets`, `cli`, `dashboard` |
| `storage/`, `validation/` | Stale-cache warnings, health checks | `cli`, `dashboard` |
| `dashboard/` | Flask UI | `mr dashboard` |

Start with the root [`README.md`](../../README.md) and [`docs/`](../../docs/README.md).
