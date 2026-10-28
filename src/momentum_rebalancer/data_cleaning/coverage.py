"""
data_cleaning/coverage.py — how much of the point-in-time index do we actually have prices for?

Survivorship bias returns through the back door when a company that *was* in the index has no
price history in the data source (typical for delisted names on Yahoo). These helpers measure
that hole: per month, the share of index members that have a price.
"""

from __future__ import annotations

import pandas as pd

from momentum_rebalancer.data_ingestion.loader import get_pit_tickers_for_date


def coverage_by_month(pit: pd.DataFrame, adj: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Per month-end in the price panel `adj` (Date × Ticker, adjusted close):
    index members, members with a price that month, and the share.
    """
    rows = []
    for d in adj.index:
        if start is not None and d < start:
            continue
        members = set(get_pit_tickers_for_date(pit, d))
        priced = set(adj.columns[adj.loc[d].notna()])
        rows.append((d, len(members), len(members & priced)))
    out = pd.DataFrame(rows, columns=["Date", "members", "with_prices"]).set_index("Date")
    out["share"] = out["with_prices"] / out["members"]
    return out


def missing_member_months(pit: pd.DataFrame, adj: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.Series:
    """Per ticker: number of month-ends it was an index member but had no price. Sorted, non-zero only."""
    counts: dict[str, int] = {}
    for d in adj.index:
        if start is not None and d < start:
            continue
        priced = set(adj.columns[adj.loc[d].notna()])
        for t in set(get_pit_tickers_for_date(pit, d)) - priced:
            counts[t] = counts.get(t, 0) + 1
    return pd.Series(counts, dtype=int).sort_values(ascending=False)
