# Workflow

## Monthly routine

| Step | Command | Notes |
|---|---|---|
| 1. After the month-end close | `mr build-features --force` | Re-downloads prices and rebuilds features (15-20+ min). Prints price coverage and warns if the membership list is stale |
| 2. | `mr cluster --force` | Always `--force`; caches do not detect parameter changes |
| 3. | `mr signals` | Check names, cluster sizes and the printed health checks |
| 3b. Data check | `mr coverage` | Share of index members with prices; add vendor files if the gap is large |
| 4. Regime review | `mr regime` | Indicators and health report; decide whether to change anything (`docs/REGIME_PLAYBOOK.md`) |
| 5. Dry run | `mr trades --save` (optionally `--regime NAME`) | Review trade count and turnover; `--save` keeps a CSV in `data/results/` for the log |
| 6. Execute | `mr rebalance` | Asks for confirmation; non-baseline regimes need you to type the regime name |
| 7. Log | `research_logs/YYYY-MM/` | Copy `research_logs/TEMPLATE.md` |

With `broker: ibkr`, run `mr broker-check` first (TWS must be open and logged in). Keep `broker: paper` until the trade list has looked right for several months and you have read `docs/KNOWN_LIMITATIONS.md` section C.

## Research data vs live data

Keep two data folders, because they come from different sources and are refreshed on different schedules:

| Purpose | Settings | Commands |
|---|---|---|
| **Live** signals and trades (the default) | `price_source: yahoo`, `MR_DATA_ROOT` unset (uses `data/`) | the monthly routine above |
| **Research**: backtest, walk-forward, stress windows (needs WRDS) | `MR_PRICE_SOURCE=crsp`, `MR_DATA_ROOT=<repo>\data_research`, `WRDS_USERNAME` | `mr crsp-check`, `mr build-features --force`, `mr cluster --force`, `mr coverage`, `mr backtest`, `mr walkforward` |

Set the variables in the shell you use for research (`set MR_PRICE_SOURCE=crsp` and `set MR_DATA_ROOT=...` in cmd) so the live folder is never overwritten. CRSP data lag today (limitation B9), so the research record ends earlier than the live one. Details: [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Periodic review (quarterly, or after any change to features or parameters)

| Command | Purpose |
|---|---|
| `mr backtest` | Net-of-cost record up to the last completed month. Check cost drag, turnover, and the counts of held names with no price |
| `mr backtest --cost-bps 20 --delist-return -0.5` | Sensitivity: does the result survive harsher assumptions? |
| `mr walkforward` | Out-of-sample check of the hyperparameters. Look at how often the same config is chosen and at the validation-to-test Sharpe drop |
| `mr coverage` | Whether the survivorship gap has changed |

The walk-forward takes minutes (27 configs); the backtest seconds. How to read them: [`VALIDATION.md`](VALIDATION.md).

## Changing parameters

1. One change at a time, with a stated hypothesis.
2. Edit `configs/strategy.yaml`, or choose a preset from `configs/regimes.yaml`.
3. Re-run the level given in `docs/PARAMETERS.md`.
4. Compare stress windows (`python scripts/stress_windows.py`), remembering what the backtest cannot see.
5. Log the decision and a revert criterion.
6. Commit config changes with a message explaining why.

## Code changes

- Reusable code goes in `src/momentum_rebalancer/`, not in notebooks.
- Add or update a test in `tests/` for any new function; run `pytest` before committing.
- Work on a branch and open a pull request for anything touching `broker/`, `strategy/targets.py` or `strategy/rebalancer.py`; mistakes there can send wrong orders.
- Never commit data, `.env`, ledgers or account details.

## Notebooks

Outputs are stripped on commit:

```bash
pip install pre-commit nbstripout
pre-commit install
nbstripout --install
```

## Testing

```bash
pytest                      # whole suite; no network, no TWS needed
pytest tests/test_config.py # one file
```

The suite uses synthetic data. It checks that the code does what it says, not that the strategy makes money.
