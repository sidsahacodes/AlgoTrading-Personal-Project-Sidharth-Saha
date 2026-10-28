# `configs/` – every parameter lives here

| File | Purpose |
|---|---|
| `strategy.yaml` | **Baseline** parameters: universe, features, clustering, strategy, regime-monitor thresholds, broker, dashboard |
| `regimes.yaml` | Named **overlays** applied on top of the baseline (`stress`, `crash`, `rebound`, `rotation`). Everything except `baseline` is an untested hypothesis |
| `crsp.yaml` | WRDS/CRSP tables, columns and options, used only when `price_source: crsp` (run `mr crsp-check`) |
| `walk_forward.yaml` | Folds, objective and hyperparameter grid for `mr walkforward` |
| `stress_windows.yaml` | Named date windows used by `scripts/stress_windows.py` |

## How it fits together

```
strategy.yaml  ──►  + overrides from regimes.yaml[<active regime>]  ──►  validate  ──►  momentum_rebalancer.config
                          ▲
        mr --regime NAME   |   MR_REGIME=NAME   |   `regime:` in strategy.yaml
```

- Keys are unique across the whole of `strategy.yaml` (sections are only for reading), so an overlay refers to a key by name, e.g. `universe_size`.
- Loading fails fast on inconsistent settings: unknown override keys, a `momentum_feature` that is not a feature, `return_lags` that disagree with `feature_cols`, `exposure_scalar` outside (0, 1], `hrp_lookback_months` below 12, and so on.
- After changing a value, check [`../docs/PARAMETERS.md`](../docs/PARAMETERS.md) for the re-run you need. `mr` also prints a warning when a cache was built with different parameters from the active ones.

## Changing a regime safely

1. `mr regime` and read the indicators; decide whether anything should change.
2. Preview: `mr regimes` lists presets and their overrides.
3. Run with the preset: `mr --regime crash trades` (dry run).
4. `mr rebalance` asks you to type the regime name when it is not `baseline`.
5. Record the decision in `research_logs/`.

Full guidance: [`../docs/REGIME_PLAYBOOK.md`](../docs/REGIME_PLAYBOOK.md).
