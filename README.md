# Unsupervised Trading Strategy

Long-only momentum strategy using **K-Means clustering** to group S&P 500 stocks, then selecting top momentum performers within each cluster.

## Strategy

1. Cluster stocks into 5 groups based on technical, momentum, and factor features
2. Rank stocks within each cluster by 3-month momentum
3. Long top 5 stocks per cluster (~25 positions)
4. Rebalance monthly

## Quick Start

```bash
pip install pandas numpy scikit-learn hdbscan yfinance pandas_ta statsmodels matplotlib seaborn

cd src
python main.py                    # Generate features
python generate_clusters.py       # Apply clustering
python long_only_strategy.py      # Backtest strategy
```

## Project Structure

```
├── data/                    # CSV outputs
├── src/
│   ├── main.py             # Feature engineering pipeline
│   ├── clustering.py       # K-Means, GMM, PCA, HDBSCAN
│   └── long_only_strategy.py  # Strategy backtest
└── notebooks/
    └── strategy_analysis.ipynb  # Performance analysis
```

## Features (19 total)

- **Technical (7):** GKV, RSI, MACD, ATR, Bollinger Bands, Dollar Volume Rank
- **Momentum (6):** 1m, 2m, 3m, 6m, 9m, 12m returns
- **Factors (5):** Fama-French betas (Mkt, SMB, HML, RMW, CMA)

**Period:** Oct 2018 - Nov 2025 | **Positions:** ~25/month | **Rebalancing:** Monthly

## Key Design Choices

**Why cluster first?** Comparing all stocks directly mixes fundamentally different companies. Clustering creates homogeneous peer groups.

**Why 3-month momentum?** Balances signal strength with stability. Short-term momentum is academically validated.

**Why long-only?** Shorts often underperform due to costs and structural market bias. Simpler implementation.

## Bias Prevention

✅ No lookahead bias  
✅ No survivorship bias (rolling universe)  
✅ Cross-sectional z-scoring  
✅ Month-end alignment

## Notes

- Transaction costs not included (~0.5-1% annual drag expected)
- Universe: Top 150 stocks by liquidity
- Minimum history: 36 months

---

**For educational purposes only. Not financial advice.**
