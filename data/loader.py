"""
data/loader.py — download S&P 500 tickers and raw OHLCV price data.

Survivorship bias fix:
  Uses fja05680/sp500 on GitHub — a point-in-time constituent list going
  back to 1996. Each date row contains exactly which tickers were in the
  S&P 500 on that day, including companies later delisted or removed.

  This means the backtest will include stocks like Enron, Lehman Brothers,
  Bear Stearns etc. that existed historically but are gone today — giving
  a realistic view of what was actually tradeable at each point in time.
"""

from __future__ import annotations
import logging
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from config import DATA_DIR, LOOKBACK_YEARS

logger = logging.getLogger(__name__)

RAW_PRICES_PATH  = DATA_DIR / "prices_raw.parquet"
PIT_MEMBERS_PATH = DATA_DIR / "pit_members.parquet"

# Point-in-time S&P 500 constituent history (fja05680, back to 1996)
PIT_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)


# ── Point-in-time constituent list ────────────────────────────────────────────

def load_pit_members(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load point-in-time S&P 500 membership.

    Returns a DataFrame indexed by date with a column 'tickers' (list of str).
    Falls back to current S&P 500 members if GitHub is unreachable.
    """
    if PIT_MEMBERS_PATH.exists() and not force_refresh:
        return pd.read_parquet(PIT_MEMBERS_PATH)

    logger.info("Downloading point-in-time S&P 500 constituent history …")

    raw = None

    # Try requests first (better proxy/SSL handling than urllib)
    try:
        import requests as req_lib
        resp = req_lib.get(PIT_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        raw = resp.text
        logger.info("Downloaded PIT data via requests.")
    except Exception as e:
        logger.warning("requests failed: %s — trying urllib …", e)

    # Fallback to urllib
    if raw is None:
        try:
            request = Request(PIT_URL, headers={"User-Agent": "Mozilla/5.0"})
            raw = urlopen(request, timeout=30).read().decode("utf-8")
            logger.info("Downloaded PIT data via urllib.")
        except Exception as e:
            logger.warning("urllib also failed: %s", e)

    # Final fallback: build a single-row PIT from current Wikipedia members
    if raw is None:
        logger.warning(
            "Could not reach GitHub for PIT data — falling back to current "
            "S&P 500 members only. Backtest will have some survivorship bias."
        )
        current = load_sp500_tickers()
        today   = pd.Timestamp.today().normalize()
        df = pd.DataFrame({"tickers": [current]}, index=[today])
        df.index.name = "date"
        df.to_parquet(PIT_MEMBERS_PATH)
        return df

    df = pd.read_csv(StringIO(raw), parse_dates=["date"])
    df = df.set_index("date").sort_index()

    def _clean(tickers_str: str) -> list[str]:
        tickers = []
        for t in str(tickers_str).split(","):
            clean = t.split("-")[0].strip()
            if clean:
                tickers.append(clean)
        return tickers

    df["tickers"] = df["tickers"].apply(_clean)
    df.to_parquet(PIT_MEMBERS_PATH)
    logger.info("Saved PIT membership: %d dates, earliest %s",
                len(df), df.index.min().date())
    return df


def get_pit_tickers_for_date(pit: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """
    Return the S&P 500 constituents as of a specific date.
    Uses the most recent available row on or before that date.
    """
    available = pit[pit.index <= date]
    if available.empty:
        # Fall back to earliest available
        return pit.iloc[0]["tickers"]
    return available.iloc[-1]["tickers"]


def get_all_historical_tickers(pit: pd.DataFrame) -> list[str]:
    """
    Return the union of all tickers that ever appeared in the S&P 500.
    Used to build the full price download list.
    """
    all_tickers = set()
    for tickers in pit["tickers"]:
        all_tickers.update(tickers)
    return sorted(all_tickers)


# ── Current tickers (for live trading only) ───────────────────────────────────

def load_sp500_tickers() -> list[str]:
    """Current S&P 500 tickers from Wikipedia (used for live trading, not backtest)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tbl = pd.read_html(urlopen(req))[0]
    tickers = (
        tbl["Symbol"]
        .str.replace(",", "-", regex=False)
        .str.replace(".", "-", regex=False)
        .unique()
        .tolist()
    )
    logger.info("Loaded %d current S&P 500 tickers", len(tickers))
    return tickers


# ── Price download ─────────────────────────────────────────────────────────────

def _month_end_yesterday() -> pd.Timestamp:
    return (pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1))


def download_prices(
    tickers: list[str] | None = None,
    lookback_years: int = LOOKBACK_YEARS,
    force_refresh: bool = False,
    use_pit: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data.

    If use_pit=True (default), downloads ALL historical S&P 500 tickers
    ever — fixing survivorship bias. This is a larger download (~600+ tickers)
    but gives a realistic backtest universe.

    If use_pit=False, only downloads current S&P 500 tickers (faster but biased).
    """
    if RAW_PRICES_PATH.exists() and not force_refresh:
        logger.info("Loading prices from cache: %s", RAW_PRICES_PATH)
        return pd.read_parquet(RAW_PRICES_PATH)

    end   = _month_end_yesterday()
    start = end - pd.DateOffset(years=lookback_years)

    if tickers is None:
        if use_pit:
            pit     = load_pit_members(force_refresh=force_refresh)
            tickers = get_all_historical_tickers(pit)
            logger.info(
                "PIT universe: %d unique tickers ever in S&P 500 (bias-free)",
                len(tickers),
            )
        else:
            tickers = load_sp500_tickers()
            logger.warning(
                "Using current S&P 500 only — backtest will have survivorship bias."
            )

    logger.info(
        "Downloading %d tickers from %s to %s …",
        len(tickers), start.date(), end.date(),
    )

    # Download in batches to avoid timeouts
    BATCH = 100
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        logger.info("  Batch %d/%d …", i // BATCH + 1, (len(tickers) + BATCH - 1) // BATCH)
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if not raw.empty:
                frames.append(raw)
        except Exception as e:
            logger.warning("Batch %d failed: %s", i // BATCH + 1, e)

    if not frames:
        raise RuntimeError("No price data downloaded.")

    combined = pd.concat(frames, axis=1)
    # Drop duplicate columns (same ticker in multiple batches)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    prices = (
        combined
        .stack(future_stack=True)
        .rename_axis(["Date", "Ticker"])
    )
    prices.columns = prices.columns.str.lower()

    prices.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(prices.index.get_level_values("Date")).tz_localize(None),
        prices.index.get_level_values("Ticker"),
    ])
    prices = prices.loc[:end].sort_index()

    prices.to_parquet(RAW_PRICES_PATH)
    logger.info("Saved %d rows to %s", len(prices), RAW_PRICES_PATH)
    return prices