# `data_cleaning/` – monthly universe and coverage

| File | Role |
|---|---|
| `universe.py` | Month-end alignment, cleaning, tradability, point-in-time filters, liquidity ranking |
| `coverage.py` | `coverage_by_month()` and `missing_member_months()`: how many index members have prices |

`build_monthly_universe(prices, pit)`:

1. last observation per month, per ticker
2. drop rows with incomplete OHLCV (no forward/back-filling)
3. drop rows whose last real price bar (a row with a close) is more than `max_stale_days` before the month-end, and rows for stocks with a delisting event in that month (a stock that stopped trading or was delisted is not tradable at that month-end)
4. `dollar_volume = close × volume`
5. keep a row only if the ticker already has `min_history_months` month-end rows **before** that date
6. keep only tickers that were in the S&P 500 **on that month-end** (`data_ingestion.loader.get_pit_tickers_for_date`)
7. keep the top `universe_size` by 60-month rolling dollar volume each month; keep the rank as `dv_rank`

**Parameters:** `min_history_months`, `universe_size`, `max_stale_days` (all L2). The 60-month window is hard-coded.

**In stress:** shrinking `universe_size` (150 → 100) concentrates on the most liquid names (the `stress` preset). It leaves fewer candidates per cluster; check cluster sizes afterwards.

**Coverage:** `mr coverage` reports, per month, the share of index members that have a price and lists the members missing from the most months.
