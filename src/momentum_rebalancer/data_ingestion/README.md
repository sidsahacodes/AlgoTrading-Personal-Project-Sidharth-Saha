# `data_ingestion/` – membership history and prices

| File | Role |
|---|---|
| `loader.py` | Point-in-time S&P 500 membership (`load_pit_members`, `get_pit_tickers_for_date`, `get_window_tickers`) and daily OHLCV download (`download_prices`) |
| `crsp.py` | CRSP via WRDS as the research price and membership source (`price_source: crsp`): queries, label assignment, return-based price rebuild, `mr crsp-check` |
| `external.py` | Optional vendor price files for names Yahoo does not carry |
| `symbols.py` | `clean_symbol()`: one normalisation of tickers (`BRK.B` → `BRK-B`, `ALOG-201806` → `ALOG`) |

```
membership CSV (pit_url, "(Updated)" file)
        │ load_pit_members()          -> data/store/pit_members.parquet (+ .meta.json)
        ▼
tickers that were members at any time in the window
        │ download_prices()  yfinance, batches of 100, auto_adjust=False, end = last completed month-end
        │   + vendor files for tickers Yahoo returned nothing for
        ▼                              -> data/store/prices_raw.parquet, data/store/price_coverage.json
daily OHLCV (Date, Ticker)
```

- The window is `lookback_years` back from the last completed month-end; only tickers that were members during it are requested.
- If the membership list cannot be downloaded the loader **raises**; it uses today's members only if `pit_allow_fallback: true`.
- Prices are always re-downloaded in full, never appended (Yahoo re-bases adjusted closes).
- After each download `price_coverage.json` records how many requested tickers have prices and which do not; `mr coverage` turns that into a per-month survivorship report.

**CRSP path** (`price_source: crsp`, tables and columns in `configs/crsp.yaml`): membership is CRSP's S&P 500 list expanded to month-ends; each company has one label for its whole life; prices are rebuilt from CRSP returns so they are continuous through splits and dividends; the delisting return already in CRSP's CIZ return series is used as is (never compounded again), and a delisting row is a return without a trading bar. The sample ends where CRSP's membership list and prices end. Run `mr crsp-check` first. See [`../../../docs/DATA_SOURCES.md`](../../../docs/DATA_SOURCES.md).

**Parameters:** `price_source`, `lookback_years`, `pit_url`, `pit_allow_fallback` (L2).

**In stress:** the main risk is data failure (outages, missing tickers). `mr regime` checks ticker counts, staleness and the age of the membership list.

**Gotchas:** Yahoo lacks many removed and delisted names (B1) and renamed companies appear only under their new symbol (B2); the membership list ends at its last update (B3). See [`../../../docs/DATA_SOURCES.md`](../../../docs/DATA_SOURCES.md) and [`../../../docs/KNOWN_LIMITATIONS.md`](../../../docs/KNOWN_LIMITATIONS.md).
