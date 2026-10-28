"""
validation/checks.py — cheap data-health and cluster-health checks.

Each check returns a list of `Issue`s; an empty list means healthy. These do
not stop the pipeline. They exist so a human sees problems *before* trading
(see docs/REGIME_PLAYBOOK.md, regime dashboard).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Issue:
    level: str      # "error" | "warn" | "info"
    msg: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.msg}"


def check_modeling_data(
    df: pd.DataFrame,
    expected_cols: list[str],
    min_tickers: int = 50,
    today: pd.Timestamp | None = None,
    max_age_days: int = 45,
) -> list[Issue]:
    issues: list[Issue] = []
    if df is None or df.empty:
        return [Issue("error", "modeling data is empty")]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        issues.append(Issue("error", f"missing feature columns: {missing}"))

    dates = df.index.get_level_values("Date")
    latest = dates.max()
    month = df.xs(latest, level="Date")

    if len(month) < min_tickers:
        issues.append(Issue("warn", f"only {len(month)} tickers in latest month {latest.date()} "
                                    f"(expected >= {min_tickers}); possible download problem"))

    present = [c for c in expected_cols if c in df.columns]
    if present:
        nan_cols = month[present].isna().any()
        bad = nan_cols[nan_cols].index.tolist()
        if bad:
            issues.append(Issue("warn", f"NaNs in latest month for: {bad}"))

    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    age = (today - latest).days
    if age > max_age_days:
        issues.append(Issue("warn", f"latest data month is {latest.date()} ({age} days old)"))
    return issues


def check_clusters(
    month_data: pd.DataFrame,
    top_n: int,
    n_clusters: int,
    method: str,
    max_share: float = 0.5,
) -> list[Issue]:
    """`month_data` is one month's frame with a `cluster` column."""
    issues: list[Issue] = []
    if "cluster" not in month_data.columns or month_data.empty:
        return [Issue("error", "no cluster assignments for the latest month")]

    sizes = month_data["cluster"].value_counts()
    real = sizes[sizes.index != -1]

    if method == "hdbscan":
        noise = (month_data["cluster"] == -1).mean()
        if noise > 0.2:
            issues.append(Issue("warn", f"{noise:.0%} of stocks are HDBSCAN noise (-1) and excluded"))
        if len(real) < 2:
            issues.append(Issue("warn", f"HDBSCAN found {len(real)} cluster(s)"))
    elif len(real) != n_clusters:
        issues.append(Issue("warn", f"{len(real)} clusters found, expected {n_clusters}"))

    small = real[real < top_n]
    if len(small):
        issues.append(Issue("warn", f"clusters smaller than top_n={top_n}: "
                                    f"{ {int(k): int(v) for k, v in small.items()} } (all members taken)"))

    if len(real):
        share = real.max() / real.sum()
        if share > max_share:
            issues.append(Issue("warn", f"largest cluster holds {share:.0%} of stocks (> {max_share:.0%}); "
                                        f"structure may be degenerate"))
    return issues


def check_selection(selected: pd.DataFrame, portfolio_size: int) -> list[Issue]:
    if selected is None or selected.empty:
        return [Issue("error", "no stocks selected")]
    if len(selected) < portfolio_size:
        return [Issue("warn", f"selected {len(selected)} stocks, target {portfolio_size}")]
    return []


def check_pit_freshness(pit: pd.DataFrame, today: pd.Timestamp | None = None, max_age_days: int = 120) -> list[Issue]:
    """Warn when the membership list stops long before today (the universe is then frozen at its last row)."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    last = pit.index.max()
    age = (today - last).days
    if age > max_age_days:
        return [Issue("warn", f"membership list ends {last.date()} ({age} days ago): every later month "
                              f"reuses that member list. Check pit_url and re-run `mr build-features --force`")]
    return []
