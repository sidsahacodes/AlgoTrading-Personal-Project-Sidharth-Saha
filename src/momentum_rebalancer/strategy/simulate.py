"""
strategy/simulate.py — month-by-month portfolio simulation.

This is the engine behind `run_backtest` and the walk-forward. It follows the live path:

    clusters at month-end d  ->  select_top_momentum  ->  weights (HRP or equal, history up to d)
        ->  exposure scalar  ->  trade band  ->  hold through month d+1  ->  repeat

Properties that matter for honest results:

* **Point in time.** Selection uses month-d features; HRP uses returns up to and including d.
  The only future information used is the realised return of the month being earned.
* **Real exit returns.** Returns come from the full price panel (`returns_wide`), not from the
  modelling universe, so a stock that drops out of the universe, or out of the index, still
  earns what it actually earned. Only a stock with no price at all in the next month is given
  `delisting_return` (and a stock with a data gap, 0%); both are counted in `diagnostics`.
* **Costs.** Each rebalance trades the difference between the target and the *drifted* weights;
  cost = traded notional x cost_bps / 10,000, deducted from that month's return.

Return series are indexed by *signal date* d (the decision date); the return is earned over
the following month. `BacktestResult` re-labels them by the month they were earned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from momentum_rebalancer.features.returns import last_valid_dates
from momentum_rebalancer.strategy.optimiser import weights_from_history
from momentum_rebalancer.strategy.signals import select_top_momentum

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimConfig:
    momentum_feature:    str
    top_n:               int
    portfolio_size:      int
    optimiser:           str = "hrp"
    hrp_lookback_months: int | None = None
    exposure_scalar:     float = 1.0
    min_trade_pct:       float = 0.0
    cost_bps:            float = 0.0
    delisting_return:    float = -0.30


@dataclass
class SimResult:
    gross:       pd.Series                 # return earned over the month after each signal date
    net:         pd.Series                 # gross minus costs
    traded:      pd.Series                 # traded notional at each rebalance, as a fraction of NAV
    costs:       pd.Series                 # cost paid at each rebalance, as a fraction of NAV
    earned_in:   pd.Series                 # signal date -> the month the return was earned
    positions:   pd.DataFrame              # one row per (signal date, ticker) held
    diagnostics: dict = field(default_factory=dict)


def simulate(clustered: pd.DataFrame, returns_wide: pd.DataFrame, cfg: SimConfig) -> SimResult:
    """
    Parameters
    ----------
    clustered    : (Date, Ticker) frame with a `cluster` column and the momentum feature.
    returns_wide : (Date x Ticker) raw monthly total returns for every ticker with prices,
                   on a complete month-end index.
    """
    r_index = returns_wide.index
    last_valid = last_valid_dates(returns_wide)
    by_date = {d: g.droplevel("Date") for d, g in clustered.groupby(level="Date")}

    signal_dates = [d for d in sorted(by_date) if d in r_index and r_index.get_loc(d) + 1 < len(r_index)]

    gross, net, traded, costs, earned = {}, {}, {}, {}, {}
    rows = []
    diag = {"signals": 0, "holdings": 0, "delisted": 0, "data_gaps": 0, "weight_fallbacks": 0}
    w_prev: dict[str, float] = {}
    prev_date = None

    for d in signal_dates:
        loc = r_index.get_loc(d)
        d_next = r_index[loc + 1]

        # Positions only carry over from the immediately preceding month; after a gap start flat.
        if prev_date is None or r_index.get_loc(prev_date) + 1 != loc:
            w_prev = {}

        month_data = by_date[d]
        sel = select_top_momentum(month_data, cfg.momentum_feature, cfg.top_n,
                                  portfolio_size=cfg.portfolio_size)
        if sel.empty:
            continue
        tickers = sel.index.tolist()

        # ── target weights, from information available at d ───────────────────
        history = returns_wide.loc[:d, [t for t in tickers if t in returns_wide.columns]]
        w_target, how = weights_from_history(tickers, history, cfg.optimiser, cfg.hrp_lookback_months)
        if how not in ("hrp", "equal"):
            diag["weight_fallbacks"] += 1
        w_target = {t: w * cfg.exposure_scalar for t, w in w_target.items()}

        # ── trade band: small resizes are skipped, exits always trade ─────────
        eff: dict[str, float] = {}
        for t in set(w_target) | set(w_prev):
            tgt, cur = w_target.get(t, 0.0), w_prev.get(t, 0.0)
            eff[t] = cur if (tgt > 0 and abs(tgt - cur) < cfg.min_trade_pct) else tgt
        eff = {t: w for t, w in eff.items() if w > 0}

        trade = sum(abs(eff.get(t, 0.0) - w_prev.get(t, 0.0)) for t in set(eff) | set(w_prev))
        cost = trade * cfg.cost_bps / 10_000.0

        # ── realised returns over the following month ─────────────────────────
        r_held: dict[str, float] = {}
        gone: set[str] = set()
        for t in eff:
            r = returns_wide.at[d_next, t] if t in returns_wide.columns else np.nan
            if pd.isna(r):
                lv = last_valid.get(t)
                if lv is None or pd.isna(lv) or lv < d_next:      # no price at all from here on
                    r = cfg.delisting_return
                    gone.add(t)
                    diag["delisted"] += 1
                else:                                              # priced later, missing this month
                    r = 0.0
                    diag["data_gaps"] += 1
            r_held[t] = float(r)

        g = float(sum(w * r_held[t] for t, w in eff.items()))
        gross[d], costs[d], traded[d] = g, cost, trade
        net[d] = g - cost
        earned[d] = d_next

        for t, w in eff.items():
            rows.append((d, t, sel["cluster"].get(t, np.nan), sel["momentum_score"].get(t, np.nan), w, r_held[t]))

        # ── drift weights to the next rebalance (delisted names are cashed out) ─
        denom = 1.0 + g
        w_prev = {t: w * (1.0 + r_held[t]) / denom for t, w in eff.items() if t not in gone}

        diag["signals"] += 1
        diag["holdings"] += len(eff)
        prev_date = d

    idx = pd.DatetimeIndex(sorted(net))
    as_series = lambda m, name: pd.Series(m, name=name, dtype=float).reindex(idx)
    positions = pd.DataFrame(rows, columns=["Date", "Ticker", "cluster", "momentum_score", "weight", "fwd_return"])
    positions = positions.set_index(["Date", "Ticker"]) if len(positions) else positions
    return SimResult(
        gross=as_series(gross, "gross"), net=as_series(net, "net"),
        traded=as_series(traded, "traded"), costs=as_series(costs, "costs"),
        earned_in=pd.Series(earned, name="earned_in").reindex(idx),
        positions=positions, diagnostics=diag,
    )
