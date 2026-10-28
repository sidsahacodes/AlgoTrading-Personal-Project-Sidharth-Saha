import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.strategy.optimiser import weights_from_history
from momentum_rebalancer.strategy.simulate import SimConfig, simulate

MONTHS = pd.date_range("2018-01-31", periods=8, freq="ME")


def _clustered(picks_by_date, months=MONTHS):
    """picks_by_date: {date: {ticker: cluster}}; every listed ticker gets the top momentum in its cluster."""
    rows = [(d, t, c, 1.0) for d, picks in picks_by_date.items() for t, c in picks.items()]
    df = pd.DataFrame(rows, columns=["Date", "Ticker", "cluster", "return_3m"]).set_index(["Date", "Ticker"])
    return df.sort_index()


def _returns(table, months=MONTHS):
    """table: {ticker: list of monthly returns aligned to `months` (None = no price)}"""
    return pd.DataFrame({t: pd.Series(v, index=months, dtype=float) for t, v in table.items()})


def _cfg(**kw):
    base = dict(momentum_feature="return_3m", top_n=1, portfolio_size=20, optimiser="equal")
    base.update(kw)
    return SimConfig(**base)


def _two_stock_setup():
    picks = {d: {"X": 0, "Y": 1} for d in MONTHS}
    rets = _returns({"X": [0.10] * 8, "Y": [0.0] * 8})
    return _clustered(picks), rets


# ── basic mechanics ───────────────────────────────────────────────────────────

def test_gross_return_is_weighted_next_month_return():
    clustered, rets = _two_stock_setup()
    sim = simulate(clustered, rets, _cfg())
    assert sim.gross.iloc[0] == pytest.approx(0.05)            # 0.5 x 10% + 0.5 x 0%
    assert len(sim.gross) == len(MONTHS) - 1                    # the last month has no following return


def test_returns_are_earned_in_the_following_month():
    clustered, rets = _two_stock_setup()
    sim = simulate(clustered, rets, _cfg())
    assert (sim.earned_in.values == MONTHS[1:].values).all()
    assert (sim.earned_in.index == MONTHS[:-1]).all()


# ── costs (hand-computed) ─────────────────────────────────────────────────────

def test_cost_is_charged_on_drifted_turnover():
    clustered, rets = _two_stock_setup()
    sim = simulate(clustered, rets, _cfg(cost_bps=100))
    # month 1: build from cash -> trade 100% of NAV, cost 1%
    assert sim.traded.iloc[0] == pytest.approx(1.0)
    assert sim.costs.iloc[0] == pytest.approx(0.01)
    assert sim.net.iloc[0] == pytest.approx(0.05 - 0.01)
    # month 2: weights drifted to X=0.5/1.05*1.1, Y=0.5/1.05; rebalance back to 50/50
    wx = 0.5 * 1.10 / 1.05
    expected_trade = abs(0.5 - wx) + abs(0.5 - (1 - wx))
    assert sim.traded.iloc[1] == pytest.approx(expected_trade)
    assert sim.costs.iloc[1] == pytest.approx(expected_trade * 0.01)


def test_zero_cost_net_equals_gross():
    clustered, rets = _two_stock_setup()
    sim = simulate(clustered, rets, _cfg(cost_bps=0))
    assert np.allclose(sim.net, sim.gross)


def test_trade_band_skips_small_rebalances_but_not_the_initial_build():
    clustered, rets = _two_stock_setup()
    sim = simulate(clustered, rets, _cfg(min_trade_pct=0.05))
    assert sim.traded.iloc[0] == pytest.approx(1.0)
    assert sim.traded.iloc[1] == pytest.approx(0.0)             # drift of ~2.4% is inside the 5% band


def test_exposure_scalar_scales_gross_and_trades():
    clustered, rets = _two_stock_setup()
    full = simulate(clustered, rets, _cfg())
    half = simulate(clustered, rets, _cfg(exposure_scalar=0.5))
    assert half.gross.iloc[0] == pytest.approx(0.5 * full.gross.iloc[0])
    assert half.traded.iloc[0] == pytest.approx(0.5)


# ── exits: the survivorship-sensitive cases ───────────────────────────────────

def test_stock_that_leaves_the_universe_still_earns_its_real_return():
    """A is selected in month 3 but is no longer in the modelling universe in month 4.
    Its real month-4 return (-60%) exists in the price panel and must be used, not 0%."""
    picks = {d: {"A": 0, "B": 1} for d in MONTHS[:4]}
    picks.update({d: {"B": 1, "C": 0} for d in MONTHS[4:]})
    rets = _returns({"A": [0.02, 0.02, 0.02, -0.60, 0.0, 0.0, 0.0, 0.0],
                     "B": [0.02] * 8, "C": [0.02] * 8})
    sim = simulate(_clustered(picks), rets, _cfg())
    d3 = MONTHS[3]                                               # signal at the end of month index 3
    # return earned in month index 4... A is held at signal date index 3 and earns rets["A"][4]
    sim2 = simulate(_clustered(picks), _returns({"A": [0.02, 0.02, 0.02, 0.02, -0.60, 0, 0, 0],
                                                 "B": [0.02] * 8, "C": [0.02] * 8}), _cfg())
    assert sim2.gross.loc[d3] == pytest.approx(0.5 * -0.60 + 0.5 * 0.02)
    assert sim2.diagnostics["delisted"] == 0


def test_delisted_stock_gets_the_assumed_return_and_is_counted():
    picks = {d: {"A": 0, "B": 1} for d in MONTHS}
    a = [0.01, 0.01, 0.01, None, None, None, None, None]        # no price at all after month index 2
    rets = _returns({"A": a, "B": [0.02] * 8})
    sim = simulate(_clustered(picks), rets, _cfg(delisting_return=-0.30))
    d_sig = MONTHS[2]                                            # A held, then no price in month index 3
    assert sim.gross.loc[d_sig] == pytest.approx(0.5 * -0.30 + 0.5 * 0.02)
    assert sim.diagnostics["delisted"] >= 1


def test_delisted_position_is_cashed_out_not_traded_again():
    picks = {d: {"A": 0, "B": 1} for d in MONTHS[:3]}
    picks.update({d: {"B": 1, "C": 0} for d in MONTHS[3:]})
    rets = _returns({"A": [0.01, 0.01, 0.01, None, None, None, None, None],
                     "B": [0.02] * 8, "C": [0.02] * 8})
    sim = simulate(_clustered(picks), rets, _cfg(cost_bps=100, delisting_return=-0.30))
    # after A vanishes it must not appear as a sale (it cannot be sold): only B is rebalanced / C bought
    d_after = MONTHS[3]
    drifted_b = 0.5 * 1.02 / (1 + sim.gross.loc[MONTHS[2]])
    expected = abs(0.5 - drifted_b) + 0.5                        # resize B + buy C; A's exit is free
    assert sim.traded.loc[d_after] == pytest.approx(expected)


def test_data_gap_counts_as_zero_not_delisting():
    picks = {d: {"A": 0, "B": 1} for d in MONTHS}
    rets = _returns({"A": [0.01, 0.01, None, 0.01, 0.01, 0.01, 0.01, 0.01], "B": [0.02] * 8})
    sim = simulate(_clustered(picks), rets, _cfg())
    assert sim.diagnostics["data_gaps"] >= 1 and sim.diagnostics["delisted"] == 0
    assert sim.gross.loc[MONTHS[1]] == pytest.approx(0.5 * 0.0 + 0.5 * 0.02)


# ── causality: nothing in the future can change the past ──────────────────────

def _random_world(seed=0, n_months=60, n_tickers=24):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets = pd.DataFrame(rng.normal(0.01, 0.06, (n_months, n_tickers)), index=months, columns=tickers)
    rows = []
    for d in months:
        for i, t in enumerate(tickers):
            rows.append((d, t, i % 4, rng.normal()))
    clustered = pd.DataFrame(rows, columns=["Date", "Ticker", "cluster", "return_3m"]).set_index(["Date", "Ticker"])
    return clustered, rets, months


def test_future_returns_cannot_change_earlier_decisions():
    clustered, rets, months = _random_world()
    cfg = _cfg(top_n=2, optimiser="hrp", cost_bps=10)
    cut = months[40]

    base = simulate(clustered, rets, cfg)
    tampered = rets.copy()
    tampered.loc[tampered.index > cut] = np.random.default_rng(9).normal(0, 0.3, tampered.loc[tampered.index > cut].shape)
    alt = simulate(clustered, tampered, cfg)

    b = base.positions[base.positions.index.get_level_values("Date") <= cut]["weight"]
    a = alt.positions[alt.positions.index.get_level_values("Date") <= cut]["weight"]
    pd.testing.assert_series_equal(b, a)                         # identical weights on/before the cut
    early = base.net.index < cut                                 # returns earned strictly before the cut-month's follow-up
    assert np.allclose(base.net[early], alt.net[early])


def test_future_features_cannot_change_earlier_decisions():
    clustered, rets, months = _random_world(seed=3)
    cfg = _cfg(top_n=2, optimiser="hrp")
    cut = months[35]
    base = simulate(clustered, rets, cfg)
    tampered = clustered.copy()
    late = tampered.index.get_level_values("Date") > cut
    tampered.loc[late, "return_3m"] = np.random.default_rng(1).normal(size=late.sum())
    alt = simulate(tampered, rets, cfg)
    sel = lambda s: s.positions[s.positions.index.get_level_values("Date") <= cut]["weight"]
    pd.testing.assert_series_equal(sel(base), sel(alt))


def test_hrp_weights_use_history_up_to_the_decision_date_only():
    clustered, rets, months = _random_world(seed=5)
    cfg = _cfg(top_n=2, optimiser="hrp")
    sim = simulate(clustered, rets, cfg)
    d = months[45]
    held = sim.positions.xs(d, level="Date")
    expected, how = weights_from_history(held.index.tolist(), rets.loc[:d, held.index.tolist()], "hrp")
    assert how == "hrp"
    for t, w in expected.items():
        assert held.loc[t, "weight"] == pytest.approx(w)
