# `data/` – generated data (git-ignored contents)

Nothing in here is source code. This folder is created automatically and its contents are ignored by git.

```
data/
├── store/      parquet caches, meta files, price_coverage.json, paper ledger
├── results/    CSVs written by `--save` (signals, trades, backtest, walk-forward, coverage, stress windows)
└── external/   vendor price files you provide (see docs/DATA_SOURCES.md)
```

Move it elsewhere with `MR_DATA_ROOT` (see `.env.example`). Layout and file meanings: [`../docs/DATA_CONVENTIONS.md`](../docs/DATA_CONVENTIONS.md).
