# `storage/` – cache fingerprints

| File | Role |
|---|---|
| `fingerprint.py` | Writes `<cache>.meta.json` next to each cache and compares it with the active config |

Caches are reused unless you pass `--force`, so changing a parameter would otherwise keep stale results silently. To make that visible:

- `load_pit_members()` records `pit_url`; `build_modeling_data()` records the feature parameters (including the membership source); `run_clustering()` records only the parameters that matter for the chosen method plus which modeling build it used.
- `mr signals/trades/rebalance/regime/build-features/cluster` and `/api/status` print `!` warnings when the active config differs, or when clusters are older than the features.
- A cache with no record (for example one copied in from elsewhere) produces a single "no parameter record" note; re-run with `--force` once to record one.

It warns; it does not rebuild automatically.
