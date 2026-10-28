# `clustering/` – peer groups

Groups the monthly universe so momentum is compared **among similar stocks**, not across the whole market.

| File | Role |
|---|---|
| `methods.py` | Four algorithms behind one interface: `kmeans`, `gmm`, `pca_kmeans`, `hdbscan` |
| `runner.py` | Applies the chosen method, caches `clustered_<method>.parquet` and its `.meta.json`, helpers for the latest month |

For each month **independently**, take that month's features, fit the model, add a `cluster` column. Nothing is shared across months (no look-ahead), but **cluster ids are arbitrary labels**: cluster 2 now is not cluster 2 last month.

| Method | Key parameters | Notes |
|---|---|---|
| `kmeans` (default) | `n_clusters`, `random_state` | Hard assignment, spherical clusters |
| `gmm` | `n_clusters` (components) | Full covariance; label = most likely component |
| `pca_kmeans` | `pca_components`, `n_clusters` | Denoises first |
| `hdbscan` | `hdbscan_min_size` | Density-based; noise `-1` is excluded from selection |

**Parameters** (all L1: `mr cluster --force`): `cluster_method`, `n_clusters`, `pca_components`, `hdbscan_min_size`, `random_state`.

**In stress:** check cluster sizes (`mr cluster`, `mr signals`, `mr regime` all report health): a cluster smaller than `top_n_per_cluster`, or one cluster holding most stocks, means structure is degenerating, as can happen when cross-sectional differences collapse in a crash. Try `n_clusters` ± 1 or `pca_kmeans`, and compare on stress windows. HDBSCAN can return very few clusters with `hdbscan_min_size = 15` on ~150 stocks, leaving fewer than 20 names.

**Gotchas:** the cache is keyed by method name, so use `--force` after any change; the fingerprint warns when you forget. Features are not re-scaled before clustering (A4); per-cluster performance on cluster ids is not meaningful (A2).
