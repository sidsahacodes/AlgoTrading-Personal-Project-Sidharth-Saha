# `validation/` – data and cluster health checks

| File | Role |
|---|---|
| `checks.py` | `check_modeling_data`, `check_clusters`, `check_selection`, `check_pit_freshness`; each returns a list of `Issue(level, msg)` |

| Check | Flags |
|---|---|
| Modeling data | empty frame, missing feature columns, fewer than `min_tickers` in the latest month, NaNs in the latest month, data older than 45 days |
| Clusters | wrong cluster count, clusters smaller than `top_n_per_cluster`, one cluster above `cluster_max_share` of stocks, HDBSCAN noise share above 20% |
| Selection | no stocks selected, fewer than `portfolio_size` |
| Membership list | the list ends more than 120 days ago, so later months reuse its last member list |

Shown by `mr signals`, `mr regime`, `mr coverage` and `mr build-features`. Checks inform; they never stop the pipeline.
