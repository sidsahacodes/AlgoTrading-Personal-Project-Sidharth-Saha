"""
data_ingestion/external.py — optional vendor price files for names Yahoo does not carry.

Drop CSV or Parquet files into `<MR_DATA_ROOT>/external/prices/`. Each file needs these
columns (case-insensitive; `adj_close`/`adjusted close` are accepted for `adj close`):

    date, ticker, open, high, low, close, adj close, volume

* `ticker` must match the symbol the membership list used at the time. Dots and delisting
  suffixes are normalised for you (`BRK.B` -> `BRK-B`, Norgate-style `ALOG-201806` -> `ALOG`).
  A company that was renamed needs its history filed under the *old* symbol for the old period.
* `adj close` should be adjusted for splits **and dividends** (total return). If it is
  missing, `close` is used and a warning is logged, which is only right for non-dividend payers.
* External rows fill only tickers that Yahoo returned nothing for; they never overwrite Yahoo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from momentum_rebalancer.data_ingestion.symbols import clean_symbol

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close", "adj close", "volume"]
_ALIASES = {"adj_close": "adj close", "adjclose": "adj close", "adjusted close": "adj close",
            "adjusted_close": "adj close", "symbol": "ticker", "datetime": "date"}


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    return df


def load_external_prices(directory: Path) -> pd.DataFrame | None:
    """Combined vendor prices as a (Date, Ticker) frame with the standard price columns, or None."""
    directory = Path(directory)
    files = sorted([*directory.rglob("*.csv"), *directory.rglob("*.parquet")]) if directory.exists() else []
    if not files:
        return None

    frames = []
    for f in files:
        df = _read(f)
        need = {"date", "ticker", "open", "high", "low", "close", "volume"} - set(df.columns)
        if need:
            raise ValueError(f"{f.name}: missing columns {sorted(need)}")
        if "adj close" not in df.columns:
            logger.warning("%s has no 'adj close'; using 'close' (correct only without dividends).", f.name)
            df["adj close"] = df["close"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["ticker"] = df["ticker"].map(clean_symbol)
        frames.append(df[["date", "ticker", *PRICE_COLS]])

    out = pd.concat(frames, ignore_index=True)
    out[PRICE_COLS] = out[PRICE_COLS].astype(float)
    out = (out.rename(columns={"date": "Date", "ticker": "Ticker"})
              .drop_duplicates(["Date", "Ticker"], keep="last")
              .set_index(["Date", "Ticker"]).sort_index())
    logger.info("Loaded %d external price rows for %d tickers from %d file(s).",
                len(out), out.index.get_level_values("Ticker").nunique(), len(files))
    return out
