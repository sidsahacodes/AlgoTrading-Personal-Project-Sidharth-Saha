"""
data_ingestion/loader.py — point-in-time S&P 500 membership and daily OHLCV prices.

Membership comes from fja05680/sp500 (its maintained "(Updated)" file): for each date,
the tickers that were in the index, including companies later removed or delisted.

Yahoo does not carry every removed or delisted company. After each download the loader
records which requested tickers returned nothing (`price_coverage.json`) and fills gaps
from vendor files in `<MR_DATA_ROOT>/external/prices/` if you provide them.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from momentum_rebalancer.config import (
    DATA_DIR, DELISTING_RETURN, EXTERNAL_PRICES_DIR, LOOKBACK_YEARS, PIT_ALLOW_FALLBACK, PIT_URL,
    PRICE_SOURCE, SETTINGS, load_crsp_settings,
)
from momentum_rebalancer.data_ingestion import crsp
from momentum_rebalancer.data_ingestion.external import load_external_prices
from momentum_rebalancer.data_ingestion.symbols import clean_symbol
from momentum_rebalancer.storage.fingerprint import diff_params, write_meta

logger = logging.getLogger(__name__)

RAW_PRICES_PATH  = DATA_DIR / "prices_raw.parquet"
PIT_MEMBERS_PATH = DATA_DIR / "pit_members.parquet"
COVERAGE_PATH    = DATA_DIR / "price_coverage.json"

# ── Point-in-time constituent list ────────────────────────────────────────────

def _download_text(url: str) -> str | None:
    try:
        import requests as req_lib
        resp = req_lib.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning("requests failed: %s — trying urllib …", e)
    try:
        return urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode("utf-8")
    except Exception as e:
        logger.warning("urllib also failed: %s", e)
    return None


def load_pit_members(force_refresh: bool = False, allow_fallback: bool = PIT_ALLOW_FALLBACK) -> pd.DataFrame:
    """
    Point-in-time S&P 500 membership: a frame indexed by date with a `tickers` column (list of str).

    If the list cannot be downloaded this raises, unless `allow_fallback` is true, in which case
    today's members are used for every date (survivorship-biased; logged loudly).
    """
    if PIT_MEMBERS_PATH.exists() and not force_refresh:
        for w in diff_params(PIT_MEMBERS_PATH, SETTINGS, ["pit_url", "price_source"]):
            logger.warning("%s  -> `mr build-features --force`", w)
        return pd.read_parquet(PIT_MEMBERS_PATH)

    if PRICE_SOURCE == "crsp":
        crsp.clear_bundle()
        bundle = _crsp_bundle(today=None)
        bundle.pit.to_parquet(PIT_MEMBERS_PATH)
        write_meta(PIT_MEMBERS_PATH, SETTINGS, ["pit_url", "price_source"])
        logger.info("Saved CRSP S&P 500 membership: %d month-ends, %s to %s",
                    len(bundle.pit), bundle.pit.index.min().date(), bundle.pit.index.max().date())
        return bundle.pit

    logger.info("Downloading point-in-time S&P 500 constituent history …")
    raw = _download_text(PIT_URL)

    if raw is None:
        if not allow_fallback:
            raise RuntimeError(
                "Could not download the S&P 500 membership history, and falling back to today's "
                "members would silently reintroduce survivorship bias. Check your connection, or "
                "set `pit_allow_fallback: true` in configs/strategy.yaml to accept that."
            )
        logger.warning("Falling back to TODAY'S S&P 500 members for every date (survivorship-biased).")
        current = load_sp500_tickers()
        df = pd.DataFrame({"tickers": [current]}, index=[pd.Timestamp.today().normalize()])
        df.index.name = "date"
        df.to_parquet(PIT_MEMBERS_PATH)
        write_meta(PIT_MEMBERS_PATH, SETTINGS, ["pit_url", "price_source"])
        return df

    df = pd.read_csv(StringIO(raw), parse_dates=["date"]).set_index("date").sort_index()
    df["tickers"] = df["tickers"].apply(
        lambda s: sorted({clean_symbol(t) for t in str(s).split(",") if t.strip()})
    )
    df.to_parquet(PIT_MEMBERS_PATH)
    write_meta(PIT_MEMBERS_PATH, SETTINGS, ["pit_url", "price_source"])
    logger.info("Saved PIT membership: %d dates, %s to %s",
                len(df), df.index.min().date(), df.index.max().date())
    return df


def get_pit_tickers_for_date(pit: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """
    Return the S&P 500 constituents as of a specific date.
    Uses the most recent available row on or before that date; if the date is
    earlier than the whole list, the earliest row is used.
    """
    available = pit[pit.index <= date]
    if available.empty:
        return pit.iloc[0]["tickers"]
    return available.iloc[-1]["tickers"]


def get_all_historical_tickers(pit: pd.DataFrame) -> list[str]:
    """Union of all tickers that ever appeared in the list."""
    all_tickers: set[str] = set()
    for tickers in pit["tickers"]:
        all_tickers.update(tickers)
    return sorted(all_tickers)


def get_window_tickers(pit: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Tickers that were members at any point in [start, end] (including those already members at start)."""
    before = pit.index[pit.index <= start]
    first = before.max() if len(before) else pit.index.min()
    sub = pit[(pit.index >= first) & (pit.index <= end)]
    return get_all_historical_tickers(sub)


# ── Current tickers (fallback only) ───────────────────────────────────────────

def load_sp500_tickers() -> list[str]:
    """Current S&P 500 tickers from Wikipedia."""
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

def last_completed_month_end(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent month-end strictly before `today` (if today is a month-end, the previous one)."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    return today - pd.offsets.MonthEnd(1)


def _crsp_bundle(today: pd.Timestamp | None):
    return crsp.get_bundle(load_crsp_settings(), last_completed_month_end(today),
                           LOOKBACK_YEARS, DELISTING_RETURN)


def read_price_coverage() -> dict | None:
    return json.loads(COVERAGE_PATH.read_text()) if COVERAGE_PATH.exists() else None


def download_prices(
    tickers: list[str] | None = None,
    lookback_years: int = LOOKBACK_YEARS,
    force_refresh: bool = False,
    use_pit: bool = True,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for every ticker that was a member during the window.

    Prices are always re-downloaded in full (never appended): Yahoo re-bases adjusted
    closes whenever a dividend or split occurs, so old and new downloads do not join up.
    """
    if RAW_PRICES_PATH.exists() and not force_refresh:
        logger.info("Loading prices from cache: %s", RAW_PRICES_PATH)
        return pd.read_parquet(RAW_PRICES_PATH)

    if PRICE_SOURCE == "crsp":
        bundle = _crsp_bundle(today)
        i = bundle.info
        COVERAGE_PATH.write_text(json.dumps({
            "requested": len(set(bundle.labels.values())), "with_prices": len(set(bundle.labels.values())) - len(i["missing"]),
            "from_external": [], "missing": i["missing"], "window": [str(i["start"].date()), str(i["end"].date())],
            "source": "crsp", "delisting_rows": i["delisting_rows"],
            "missing_delisting_returns_imputed": i["missing_delisting_returns_imputed"],
        }, indent=2))
        bundle.prices.to_parquet(RAW_PRICES_PATH)
        logger.info("CRSP prices: %d tickers, %d rows; %d delisting rows (%d missing returns imputed); %d members without prices.",
                    bundle.prices.index.get_level_values("Ticker").nunique(), len(bundle.prices),
                    i["delisting_rows"], i["missing_delisting_returns_imputed"], len(i["missing"]))
        return bundle.prices

    end   = last_completed_month_end(today)
    start = end - pd.DateOffset(years=lookback_years)

    if tickers is None:
        if use_pit:
            pit = load_pit_members(force_refresh=force_refresh)
            tickers = get_window_tickers(pit, start, end)
            logger.info("PIT universe: %d tickers were index members between %s and %s",
                        len(tickers), start.date(), end.date())
        else:
            tickers = load_sp500_tickers()
            logger.warning("Using current S&P 500 only — backtest will have survivorship bias.")

    logger.info("Downloading %d tickers from %s to %s …", len(tickers), start.date(), end.date())

    BATCH = 100
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        logger.info("  Batch %d/%d …", i // BATCH + 1, (len(tickers) + BATCH - 1) // BATCH)
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end + pd.Timedelta(days=1),     # yfinance's `end` is exclusive
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
    combined = combined.loc[:, ~combined.columns.duplicated()]

    prices = combined.stack(future_stack=True).rename_axis(["Date", "Ticker"])
    prices.columns = prices.columns.str.lower()
    prices.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(prices.index.get_level_values("Date")).tz_localize(None),
        prices.index.get_level_values("Ticker"),
    ], names=["Date", "Ticker"])
    prices = prices[prices.index.get_level_values("Date") <= end].sort_index()

    have = set(prices.loc[prices["adj close"].notna()].index.get_level_values("Ticker"))

    # Vendor files fill only the tickers Yahoo could not supply.
    external_used: list[str] = []
    ext = load_external_prices(EXTERNAL_PRICES_DIR)
    if ext is not None:
        ext_dates = ext.index.get_level_values("Date")
        ext_tkrs = ext.index.get_level_values("Ticker")
        keep = ext_tkrs.isin(set(tickers) - have) & (ext_dates >= start) & (ext_dates <= end)
        ext = ext[keep]
        if len(ext):
            external_used = sorted(set(ext.index.get_level_values("Ticker")))
            prices = pd.concat([prices, ext[prices.columns]]).sort_index()
            have |= set(external_used)
            logger.info("Filled %d tickers from external vendor files.", len(external_used))

    missing = sorted(set(tickers) - have)
    COVERAGE_PATH.write_text(json.dumps({
        "requested": len(tickers), "with_prices": len(have & set(tickers)),
        "from_external": external_used, "missing": missing,
        "window": [str(start.date()), str(end.date())],
    }, indent=2))
    logger.info("Price coverage: %d of %d requested tickers have prices (%d missing). "
                "Run `mr coverage` for the survivorship impact.",
                len(have & set(tickers)), len(tickers), len(missing))

    prices.to_parquet(RAW_PRICES_PATH)
    logger.info("Saved %d rows to %s", len(prices), RAW_PRICES_PATH)
    return prices
