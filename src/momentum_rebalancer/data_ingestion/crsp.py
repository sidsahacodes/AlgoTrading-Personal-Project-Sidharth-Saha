"""
data_ingestion/crsp.py — CRSP (via WRDS) as the price and membership source.

Why: Yahoo lacks many removed and delisted companies and has no delisting returns. CRSP has
both, plus its own point-in-time S&P 500 membership. Used when `price_source: crsp`.

Design (see configs/crsp.yaml for every table and column name):

* One label per company for its whole life: the *last* ticker CRSP recorded for the permno.
  That removes ticker changes (FB -> META) and keeps live trading symbols valid. If two permnos
  share a label, the one that traded most recently keeps it and the other becomes `LABEL_permno`.
* Membership comes from CRSP's S&P 500 list, expanded to one row per month-end.
* Prices are rebuilt from CRSP returns: `adj close` is a total-return index; open/high/low/close
  are the raw prices times the same factor (so they are continuous through splits and dividends);
  volume is divided by that factor so that close x volume is the *raw* dollar volume.
* **Delisting returns.** In CIZ the delisting return is already inside `dlyret`, posted on a
  separate row (`dlydelflg = 'Y'`). It is used as is. It is never compounded with anything else
  (that would double-count the loss). Only when that return is missing is `delisting_return`
  (an assumption) imputed, and the count is logged.
* CRSP's membership list is updated annually and the stock files quarterly or monthly, so the
  sample ends at the last month both cover (unless `extend_membership: true`).

Everything below `fetch_*` is pure and unit-tested on synthetic CIZ-shaped frames. The queries
themselves have not been run against WRDS from here: run `mr crsp-check` first.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from momentum_rebalancer.data_ingestion.symbols import clean_symbol

logger = logging.getLogger(__name__)

CHUNK = 200                       # permnos per query
EXAMPLE_PERMNOS = {"AAPL": 14593, "MSFT": 10107}


# ── connection and discovery ──────────────────────────────────────────────────

def connect(cfg: dict):
    try:
        import wrds
    except ImportError as e:                                        # pragma: no cover
        raise ImportError("price_source: crsp needs the `wrds` package: pip install wrds") from e
    env = cfg["wrds"]["username_env"]
    user = os.getenv(env)
    if not user:
        raise RuntimeError(f"Set {env} to your WRDS username (for example in .env).")
    return wrds.Connection(wrds_username=user)


def resolve(db, libraries: list[str], tables) -> tuple[str, str]:
    """First (library, table) pair that exists on this subscription."""
    tables = [tables] if isinstance(tables, str) else list(tables)
    available = set(db.list_libraries())
    for lib in libraries:
        if lib not in available:
            continue
        listed = set(db.list_tables(library=lib))
        for t in tables:
            if t in listed:
                return lib, t
    raise LookupError(f"None of {tables} found in libraries {libraries}. Run `mr crsp-check` and "
                      f"adjust configs/crsp.yaml.")


def table_columns(db, library: str, table: str) -> set[str]:
    d = db.describe_table(library=library, table=table)
    col = "name" if "name" in d.columns else "column_name"
    return set(d[col].astype(str).str.lower())


# ── dates ─────────────────────────────────────────────────────────────────────

def complete_month_end(date: pd.Timestamp, tolerance_days: int = 3) -> pd.Timestamp:
    """
    The latest month-end whose month is fully covered by data that run to `date`.
    A month counts as covered once `date` is within `tolerance_days` of its last business day.
    """
    date = pd.Timestamp(date).normalize()
    me = date + pd.offsets.MonthEnd(0)
    last_bday = pd.Timestamp(pd.offsets.BDay().rollback(me))        # last business day on or before the month-end
    if date >= last_bday - pd.Timedelta(days=tolerance_days):
        return me
    return date - pd.offsets.MonthEnd(1)


def sample_end(today_end: pd.Timestamp, price_max: pd.Timestamp, member_max: pd.Timestamp,
               extend_membership: bool) -> pd.Timestamp:
    """Last month-end of the research sample: limited by today, by prices, and (normally) by membership."""
    limits = [pd.Timestamp(today_end), complete_month_end(price_max)]
    if not extend_membership:
        limits.append(complete_month_end(member_max))
    return min(limits)


# ── labels ────────────────────────────────────────────────────────────────────

def assign_labels(info: pd.DataFrame, permnos: list[int]) -> dict[int, str]:
    """
    One label per permno. `info` has columns permno, sec_start, sec_end, ticker.
    Label = last non-blank ticker (normalised); collisions keep the label for the permno that
    traded most recently, the others get `LABEL_permno`; permnos with no ticker become `P<permno>`.
    """
    info = info[info["permno"].isin(permnos)].copy()
    info["ticker"] = info["ticker"].astype("string").str.strip()
    info = info[info["ticker"].notna() & (info["ticker"] != "")]
    last = info.sort_values(["permno", "sec_end"]).groupby("permno").tail(1)
    base = {int(r.permno): clean_symbol(r.ticker.upper()) for r in last.itertuples()}
    end = {int(r.permno): r.sec_end for r in last.itertuples()}

    labels: dict[int, str] = {}
    owners: dict[str, list[int]] = {}
    for p, lab in base.items():
        owners.setdefault(lab, []).append(p)
    for lab, ps in owners.items():
        ps = sorted(ps, key=lambda p: (end[p], p))
        labels[ps[-1]] = lab                                     # most recent keeps the plain label
        for p in ps[:-1]:
            labels[p] = f"{lab}_{p}"
    for p in permnos:
        labels.setdefault(int(p), f"P{int(p)}")
    return labels


# ── membership ────────────────────────────────────────────────────────────────

def membership_to_pit(members: pd.DataFrame, labels: dict[int, str], month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """
    One row per month-end: the labels of permnos whose [mbr_start, mbr_end] contains it.
    `members` has columns permno, mbr_start, mbr_end. Returns the loader's membership format
    (date index, `tickers` list column).
    """
    rows = []
    for d in month_ends:
        active = members[(members["mbr_start"] <= d) & (members["mbr_end"] >= d)]
        rows.append(sorted({labels[int(p)] for p in active["permno"] if int(p) in labels}))
    return pd.DataFrame({"tickers": rows}, index=pd.DatetimeIndex(month_ends, name="date"))


# ── prices ────────────────────────────────────────────────────────────────────

def daily_to_prices(daily: pd.DataFrame, labels: dict[int, str], delisting_return: float) -> tuple[pd.DataFrame, dict]:
    """
    CIZ daily rows -> the project's price frame, indexed (Date, Ticker) with columns
    open, high, low, close, adj close, volume.

    `daily` columns: permno, dt, delflg, price, ret, vol, px_open, px_high, px_low.
    """
    df = daily[daily["permno"].isin(labels)].copy()
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
    df = df.sort_values(["permno", "dt"]).reset_index(drop=True)

    is_del = df["delflg"].astype("string").str.upper().eq("Y").fillna(False).to_numpy(dtype=bool)
    price = df["price"].astype(float).abs()
    ret = df["ret"].astype(float)

    # The delisting return is already in `ret` (CIZ). Impute only if it is missing.
    missing_payoff = is_del & ret.isna().to_numpy()
    ret = ret.mask(missing_payoff, delisting_return)

    keep = is_del | (price > 0).to_numpy()
    df, price, ret, is_del = df[keep], price[keep], ret[keep], is_del[keep]
    df = df.reset_index(drop=True); price = price.reset_index(drop=True); ret = ret.reset_index(drop=True)

    first = ~df.duplicated("permno").to_numpy()
    ret = ret.where(~first, 0.0).fillna(0.0)                     # a series starts at its first price
    growth = (1.0 + ret).groupby(df["permno"]).cumprod()
    base = price.where(price > 0).groupby(df["permno"]).transform("first")
    adj = base * growth
    f = (adj / price).where(price > 0)                           # NaN on delisting rows without a price

    def scaled(col: str) -> pd.Series:
        raw = df[col].astype(float).abs()
        raw = raw.where(raw > 0, price)                          # no-trade days: flat bar at the price
        return raw * f

    out = pd.DataFrame({
        "open": scaled("px_open"), "high": scaled("px_high"), "low": scaled("px_low"),
        "close": price * f, "adj close": adj,
        "volume": df["vol"].astype(float) / f,
        "Date": df["dt"], "Ticker": df["permno"].map(labels),
    })
    bar_cols = ["open", "high", "low", "close", "volume"]
    out.loc[is_del, bar_cols] = np.nan                           # a delisting row carries the return, not a trading bar
    out = out.dropna(subset=["adj close"]).set_index(["Date", "Ticker"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out, {"delisting_rows": int(is_del.sum()), "missing_delisting_returns_imputed": int(missing_payoff.sum())}


# ── queries ───────────────────────────────────────────────────────────────────

def _chunks(items, n=CHUNK):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def fetch_membership(db, cfg) -> pd.DataFrame:
    lib, tab = resolve(db, cfg["libraries"]["indexes"], cfg["tables"]["membership"])
    c = cfg["columns"]["membership"]
    sql = f"select {c['permno']} as permno, {c['start']} as mbr_start, {c['end']} as mbr_end from {lib}.{tab}"
    df = db.raw_sql(sql, date_cols=["mbr_start", "mbr_end"])
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_max_date(db, cfg) -> pd.Timestamp:
    lib, tab = resolve(db, cfg["libraries"]["stock"], cfg["tables"]["daily"])
    d = cfg["columns"]["daily"]["date"]
    return pd.Timestamp(db.raw_sql(f"select max({d}) as d from {lib}.{tab}", date_cols=["d"])["d"].iloc[0])


def fetch_security_info(db, cfg, permnos: list[int]) -> pd.DataFrame:
    lib, tab = resolve(db, cfg["libraries"]["stock"], cfg["tables"]["security_info"])
    c = cfg["columns"]["security_info"]
    frames = [db.raw_sql(
        f"select {c['permno']} as permno, {c['start']} as sec_start, {c['end']} as sec_end, {c['ticker']} as ticker "
        f"from {lib}.{tab} where {c['permno']} in %(p)s",
        params={"p": tuple(int(x) for x in chunk)}, date_cols=["sec_start", "sec_end"]) for chunk in _chunks(permnos)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["permno", "sec_start", "sec_end", "ticker"])


def fetch_daily(db, cfg, permnos: list[int], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    lib, tab = resolve(db, cfg["libraries"]["stock"], cfg["tables"]["daily"])
    c = cfg["columns"]["daily"]
    select = (f"{c['permno']} as permno, {c['date']} as dt, {c['delisting_flag']} as delflg, {c['price']} as price, "
              f"{c['ret']} as ret, {c['volume']} as vol, {c['open']} as px_open, {c['high']} as px_high, {c['low']} as px_low")
    frames = []
    for i, chunk in enumerate(_chunks(permnos), 1):
        logger.info("  CRSP daily: permno chunk %d/%d …", i, -(-len(permnos) // CHUNK))
        frames.append(db.raw_sql(
            f"select {select} from {lib}.{tab} where {c['date']} between %(s)s and %(e)s and {c['permno']} in %(p)s",
            params={"s": str(start.date()), "e": str(end.date()), "p": tuple(int(x) for x in chunk)}, date_cols=["dt"]))
    return pd.concat(frames, ignore_index=True)


# ── the research bundle ───────────────────────────────────────────────────────

@dataclass
class CrspBundle:
    pit: pd.DataFrame
    prices: pd.DataFrame
    labels: dict
    info: dict = field(default_factory=dict)


_BUNDLE: dict = {}


def clear_bundle() -> None:
    _BUNDLE.clear()


def build_bundle(cfg: dict, today_end: pd.Timestamp, lookback_years: int, delisting_return: float, db=None) -> CrspBundle:
    """Query CRSP and return the membership frame, the price frame and diagnostics."""
    db = db or connect(cfg)
    members = fetch_membership(db, cfg)
    price_max = fetch_max_date(db, cfg)
    member_max = members["mbr_end"].max()
    extend = bool(cfg.get("extend_membership", False))
    end = sample_end(today_end, price_max, member_max, extend)
    start = end - pd.DateOffset(years=lookback_years)
    logger.info("CRSP: prices end %s, membership ends %s -> sample %s .. %s",
                price_max.date(), member_max.date(), start.date(), end.date())
    if end < today_end:
        logger.warning("CRSP data end before the last completed month (%s). Months after %s are not in this "
                       "dataset; use price_source: yahoo for the latest signals.", today_end.date(), end.date())
    if extend and complete_month_end(member_max) < end:
        logger.warning("extend_membership is on: months after %s reuse CRSP's last S&P 500 list "
                       "(not point-in-time).", member_max.date())

    window = members[(members["mbr_end"] >= start) & (members["mbr_start"] <= end)]
    permnos = sorted(window["permno"].unique())
    info = fetch_security_info(db, cfg, permnos)
    labels = assign_labels(info, permnos)

    month_ends = pd.date_range(start + pd.offsets.MonthEnd(0), end, freq="ME")
    pit = membership_to_pit(window, labels, month_ends)

    daily = fetch_daily(db, cfg, permnos, start, end)
    prices, dstats = daily_to_prices(daily, labels, delisting_return)
    have = set(prices.index.get_level_values("Ticker"))
    missing = sorted(set(labels.values()) - have)
    return CrspBundle(pit=pit, prices=prices, labels=labels, info={
        "start": start, "end": end, "price_max": price_max, "member_max": member_max,
        "permnos": len(permnos), "missing": missing, **dstats,
    })


def get_bundle(cfg: dict, today_end: pd.Timestamp, lookback_years: int, delisting_return: float, db=None) -> CrspBundle:
    key = (pd.Timestamp(today_end), lookback_years, delisting_return)
    if key not in _BUNDLE:
        _BUNDLE.clear()
        _BUNDLE[key] = build_bundle(cfg, today_end, lookback_years, delisting_return, db)
    return _BUNDLE[key]


# ── `mr crsp-check` ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_checks(db, cfg: dict, today_end: pd.Timestamp | None = None,
               expected_members: tuple[int, int] = (450, 550)) -> list[Check]:
    """Verify tables, columns and a few sanity facts on this WRDS subscription. Reads only."""
    out: list[Check] = []

    def add(name, ok, detail):
        out.append(Check(name, bool(ok), detail))

    resolved = {}
    for kind, libs, tabs in [("daily", cfg["libraries"]["stock"], cfg["tables"]["daily"]),
                             ("membership", cfg["libraries"]["indexes"], cfg["tables"]["membership"]),
                             ("security_info", cfg["libraries"]["stock"], cfg["tables"]["security_info"])]:
        try:
            lib, tab = resolve(db, libs, tabs)
            resolved[kind] = (lib, tab)
            have = table_columns(db, lib, tab)
            want = {v.lower() for v in cfg["columns"][kind].values()}
            miss = sorted(want - have)
            add(f"{kind}: {lib}.{tab}", not miss, "all columns present" if not miss else f"missing columns {miss}; edit configs/crsp.yaml")
        except Exception as e:
            add(f"{kind} table", False, str(e))

    if len(resolved) == 3:
        try:
            members = fetch_membership(db, cfg)
            price_max = fetch_max_date(db, cfg)
            member_max = members["mbr_end"].max()
            end = sample_end(today_end or pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1),
                             price_max, member_max, bool(cfg.get("extend_membership", False)))
            add("data coverage", True, f"prices to {price_max.date()}, S&P 500 list to {member_max.date()}; "
                f"research sample would end {end.date()}")
            n = int(((members["mbr_start"] <= member_max) & (members["mbr_end"] >= member_max)).sum())
            lo, hi = expected_members
            add("S&P 500 members at the end of the list", lo <= n <= hi, f"{n} members (expect about 500)")
        except Exception as e:
            add("membership / coverage", False, str(e))

        try:
            lib, tab = resolved["daily"]
            c = cfg["columns"]["daily"]
            d = db.raw_sql(f"select count(*) as n, sum(case when {c['ret']} is null then 1 else 0 end) as n_null "
                           f"from {lib}.{tab} where {c['delisting_flag']} = 'Y' and {c['date']} >= '2015-01-01'")
            n, n_null = int(d["n"].iloc[0]), int(d["n_null"].iloc[0])
            add("delisting rows (dlydelflg = 'Y')", n > 0,
                f"{n} since 2015, {n_null} with no return ({(n_null / n if n else 0):.0%}); missing ones get the "
                f"`delisting_return` assumption")
        except Exception as e:
            add("delisting rows", False, str(e))

        try:
            lib, tab = resolved["daily"]
            c = cfg["columns"]["daily"]
            s = db.raw_sql(f"select {c['date']} as dt, {c['price']} as price, {c['ret']} as ret from {lib}.{tab} "
                           f"where {c['permno']} = {EXAMPLE_PERMNOS['AAPL']} order by {c['date']} desc limit 3", date_cols=["dt"])
            add("sample (Apple, permno 14593)", len(s) > 0 and bool((s["price"] > 0).all()),
                "; ".join(f"{r.dt.date()} price {r.price:.2f} ret {r.ret:+.4f}" for r in s.itertuples()) if len(s) else "no rows")
        except Exception as e:
            add("sample row", False, str(e))
    return out
