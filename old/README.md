# `old/` – archived baseline

The original baseline strategy, kept for reference. It is not imported by anything in `src/`.

| Item | Description |
|---|---|
| `unsupervised_trading/` | K-Means on the same 19 features, top-5 momentum per cluster, equal weight. CSV-based scripts (`main.py` → `generate_clusters.py` → `long_only_strategy.py`), exploratory notebooks, `data.zip` |
| `Unsupervised Learning AlgoTrading.ipynb` | Original notebook version of the strategy |
| `Sentiment Analysis Based Strategy.ipynb` | A separate sentiment-based idea |

It builds its universe from today's Wikipedia S&P 500 list, so its backtest is survivorship-biased. Its notebooks still contain outputs.
