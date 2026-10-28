# `tests/` – what is and is not covered

```bash
pytest          # no network, no TWS, a couple of minutes
```

| File | Covers |
|---|---|
| `test_no_lookahead.py` | **Look-ahead detector**: the feature set built from all prices and from prices cut at mid-2023 must be identical up to the cut; a ticker's indicators do not depend on other tickers |
| `test_return_features.py` | Calendar-month lags and gaps, cross-sectional winsorising, FF beta recovery, beta timing, carry-forward limit |
| `test_simulate.py` | Engine: gross/net returns, costs by hand, band, exposure scalar, exits, delistings, data gaps; **causality** (later returns or features cannot change earlier decisions) |
| `test_walkforward.py` | Fold structure, an oracle feature the walk-forward must find, the tamper test (the test block cannot influence selection), error cases |
| `test_data_layer.py` | Ticker normalisation, membership parsing and the fail-loud guard, window tickers, the exclusive `end` date, price download with vendor fill, coverage, stale-bar and history filters, benchmark alignment, metrics |
| `test_crsp.py` | CRSP source against a fake WRDS connection: labels and collisions, membership, price rebuild through splits, delisting rows (used once, never double-counted), the membership cap, tradability, loader and pipeline integration, `mr crsp-check` |
| `test_config.py` | Pinned baseline values; every regime loads; invalid configs are rejected |
| `test_feature_pipeline.py` | The full feature pipeline on synthetic prices (network stubbed) |
| `test_optimiser.py`, `test_rebalancer.py`, `test_signals.py`, `test_targets.py` | HRP/equal weights and the shared history helper, trade list and band, selection, live target weights |
| `test_backtest_metrics.py`, `test_regime.py` | Metrics and period slicing; regime indicators and report; stress-window helper |
| `test_validation.py`, `test_fingerprint.py`, `test_universe_pit.py` | Health checks, stale-cache warnings, point-in-time membership lookup and filter |
| `test_end_to_end.py`, `test_cli.py` | Clustering → selection → trades with the paper broker; `--save`; broker pre-flight; dashboard backtest JSON; CLI behaviour |
| `test_paper_broker.py`, `test_dashboard_guard.py` | Paper ledger; dashboard POST guard |
| `test_macd_wiring.py` | Config MACD values equal library defaults |

The simulator, walk-forward and beta tests were checked by deliberately breaking the code and confirming a test fails (see [`../docs/VALIDATION.md`](../docs/VALIDATION.md)).

**Not covered:** the IBKR backend (needs TWS), real downloads (Yahoo, GitHub, Fama-French), and anything about whether the strategy makes money. Add a test with any new function; run `pytest` before committing.
