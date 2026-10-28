"""
strategy/regime.py — regime indicators for the monthly review.

This module REPORTS. It never changes parameters or sends orders. Thresholds
come from configs/strategy.yaml (`regime_monitor`) and are illustrative.
What to do about a flag is a human decision; see docs/REGIME_PLAYBOOK.md.

Pure functions (no network) are unit-tested. `fetch_market()` uses yfinance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LEVELS = ("ok", "caution", "stress")


@dataclass(frozen=True)
class Flag:
    name:  str
    value: float | None
    level: str            # ok | caution | stress | n/a
    note:  str = ""


# ── Pure computations ─────────────────────────────────────────────────────────

def drawdown_from_high(prices: pd.Series, window: int = 252) -> float:
    """Current drawdown vs the trailing `window`-observation high (<= 0)."""
    p = prices.dropna()
    if p.empty:
        return float("nan")
    high = p.tail(window).max()
    return float(p.iloc[-1] / high - 1.0)


def avg_pairwise_corr(returns: pd.DataFrame) -> float:
    """Mean off-diagonal correlation across columns."""
    r = returns.dropna(axis=1, how="all")
    if r.shape[1] < 2:
        return float("nan")
    c = r.corr().values
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def corr_shift(returns: pd.DataFrame, recent_months: int = 12) -> tuple[float, float]:
    """(recent avg pairwise corr, long-run avg pairwise corr) of monthly returns."""
    return avg_pairwise_corr(returns.tail(recent_months)), avg_pairwise_corr(returns)


def dispersion_percentile(modeling: pd.DataFrame, col: str = "return_3m") -> tuple[float, float]:
    """(latest cross-sectional std of `col`, its percentile rank over history)."""
    s = modeling[col].groupby("Date").std().dropna()
    if s.empty:
        return float("nan"), float("nan")
    return float(s.iloc[-1]), float((s <= s.iloc[-1]).mean())


def _level(value: float, caution: float, stress: float, higher_is_worse: bool) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    if higher_is_worse:
        return "stress" if value >= stress else "caution" if value >= caution else "ok"
    return "stress" if value <= stress else "caution" if value <= caution else "ok"


def window_metrics(returns: pd.Series, benchmark: pd.Series, windows: dict) -> pd.DataFrame:
    """Cumulative return / max drawdown / Sharpe for named (start, end) windows."""
    from momentum_rebalancer.strategy.backtest import compute_metrics   # lazy: pulls yfinance
    rows = {}
    for name, (s, e) in windows.items():
        r = returns.loc[s:e]
        if r.empty:
            rows[name] = {"months": 0, "strategy_cum": np.nan, "benchmark_cum": np.nan,
                          "max_drawdown": np.nan, "sharpe": np.nan}
            continue
        m = compute_metrics(r, benchmark.loc[s:e])
        b = benchmark.loc[s:e].reindex(r.index).dropna()
        rows[name] = {
            "months": len(r),
            "strategy_cum":  m["cumulative_return"],
            "benchmark_cum": float((1 + b).prod() - 1) if len(b) else np.nan,
            "max_drawdown":  m["max_drawdown"],
            "sharpe":        m["sharpe"],
        }
    return pd.DataFrame(rows).T


# ── Market data (network) ─────────────────────────────────────────────────────

def fetch_market(period: str = "2y") -> tuple[pd.Series | None, pd.Series | None]:
    """Daily closes for SPY and ^VIX; None for either if the download fails."""
    import yfinance as yf

    def _get(sym: str) -> pd.Series | None:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True,
                             multi_level_index=False)
            s = df["Close"].dropna()
            return s if len(s) else None
        except Exception as exc:                       # network / schema changes
            logger.warning("Could not download %s: %s", sym, exc)
            return None

    return _get("SPY"), _get("^VIX")


# ── Report ────────────────────────────────────────────────────────────────────

def build_flags(
    spy: pd.Series | None,
    vix: pd.Series | None,
    holdings_returns: pd.DataFrame | None,
    modeling: pd.DataFrame | None,
    *,
    spy_dd_caution: float, spy_dd_stress: float,
    vix_caution: float, vix_stress: float,
    corr_jump: float,
) -> list[Flag]:
    flags: list[Flag] = []

    if spy is not None:
        dd = drawdown_from_high(spy)
        flags.append(Flag("SPY drawdown from 12m high", dd,
                          _level(dd, spy_dd_caution, spy_dd_stress, higher_is_worse=False)))
    else:
        flags.append(Flag("SPY drawdown from 12m high", None, "n/a", "download failed"))

    if vix is not None:
        v = float(vix.iloc[-1])
        flags.append(Flag("VIX", v, _level(v, vix_caution, vix_stress, higher_is_worse=True)))
    else:
        flags.append(Flag("VIX", None, "n/a", "download failed"))

    if holdings_returns is not None and holdings_returns.shape[1] >= 2:
        recent, longrun = corr_shift(holdings_returns)
        jump = recent - longrun
        lvl = "n/a" if np.isnan(jump) else ("stress" if jump >= corr_jump else "ok")
        flags.append(Flag("Holdings avg pairwise corr (12m) minus long-run", jump, lvl,
                          f"recent={recent:.2f}, long-run={longrun:.2f}"))

    if modeling is not None and "return_3m" in modeling.columns:
        d, pct = dispersion_percentile(modeling)
        flags.append(Flag("Cross-sectional dispersion of return_3m", d, "ok",
                          f"{pct:.0%} percentile vs history (info only)"))
    return flags


def format_report(regime: str, overrides: dict, flags: list[Flag], issues: list, warnings: list[str]) -> str:
    out = ["", "REGIME REPORT", "=" * 60,
           f"Active regime : {regime}" + (f"   overrides: {overrides}" if overrides else "")]
    out.append("")
    out.append("Indicators (thresholds are illustrative; see configs/strategy.yaml):")
    for f in flags:
        val = "n/a" if f.value is None or (isinstance(f.value, float) and np.isnan(f.value)) else f"{f.value:+.3f}"
        out.append(f"  [{f.level:7s}] {f.name:55s} {val:>9s}  {f.note}")
    out.append("")
    if issues:
        out.append("Data / cluster health:")
        out += [f"  {i}" for i in issues]
    else:
        out.append("Data / cluster health: no issues found")
    if warnings:
        out.append("")
        out.append("Cache vs config:")
        out += [f"  ! {w}" for w in warnings]
    n_stress = sum(f.level == "stress" for f in flags)
    n_caut   = sum(f.level == "caution" for f in flags)
    out.append("")
    n_na = sum(f.level == "n/a" for f in flags)
    summary = f"Summary: {n_stress} stress flag(s), {n_caut} caution flag(s)"
    if n_na:
        summary += f", {n_na} indicator(s) UNAVAILABLE (this is not an all-clear)"
    out.append(summary + ". Review docs/REGIME_PLAYBOOK.md before deciding; nothing was changed.")
    return "\n".join(out)
