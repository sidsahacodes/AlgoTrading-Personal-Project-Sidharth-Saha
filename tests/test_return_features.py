import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.features.returns import (
    BETA_COLS, FACTOR_COLS, estimate_ff5_betas, lagged_return_features,
    monthly_adjclose_panel, monthly_return_panel, winsorize_cross_section,
)


def _idx(n, start="2015-01-31"):
    return pd.date_range(start, periods=n, freq="ME", name="Date")


# ── lagged returns ────────────────────────────────────────────────────────────

def test_lagged_returns_use_calendar_months_and_respect_gaps():
    idx = _idx(10)
    adj = pd.DataFrame({"A": 2.0 ** np.arange(10)}, index=idx)          # doubles every month
    adj.iloc[3, 0] = np.nan                                              # a missing month
    rows = pd.MultiIndex.from_product([idx, ["A"]], names=["Date", "Ticker"])
    f = lagged_return_features(adj, rows, lags=[1, 3])
    r3 = f["return_3m"].droplevel("Ticker")
    assert r3.iloc[5] == pytest.approx(1.0)                              # (8x)^(1/3) - 1 = 100% per month
    assert np.isnan(r3.iloc[3]) and np.isnan(r3.iloc[6])                 # either end missing -> NaN, not a stretched lag
    assert np.isnan(f["return_1m"].droplevel("Ticker").iloc[4])          # needs the missing month


def test_monthly_panel_takes_the_last_observation_of_each_month():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    px = pd.DataFrame({"adj close": np.arange(len(days), dtype=float) + 1, "Date": days, "Ticker": "A"}).set_index(["Date", "Ticker"])
    adj = monthly_adjclose_panel(px)
    assert list(adj.index) == list(pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
    assert adj.loc["2020-02-29", "A"] == px.xs("A", level="Ticker").loc["2020-02-28", "adj close"]   # 29 Feb 2020 was a Saturday


def test_cross_sectional_winsorising_is_per_date():
    idx = pd.MultiIndex.from_product([_idx(2), [f"T{i}" for i in range(200)]], names=["Date", "Ticker"])
    vals = np.concatenate([np.r_[np.linspace(-0.1, 0.1, 199), 5.0],           # month 1 has an outlier
                           np.linspace(-0.1, 0.1, 200) * 0.0 + 0.01])        # month 2 is flat
    df = pd.DataFrame({"x": vals}, index=idx)
    w = winsorize_cross_section(df, ["x"], q=0.01)
    assert w.xs(idx.levels[0][0], level="Date")["x"].max() < 1.0              # outlier clipped within its own month
    assert (w.xs(idx.levels[0][1], level="Date")["x"] == 0.01).all()          # other months untouched


# ── Fama-French betas ─────────────────────────────────────────────────────────

def _ff_and_returns(n=80, ff_months=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    ff = pd.DataFrame(rng.normal(0, 0.04, (n, 5)), index=idx, columns=FACTOR_COLS)
    ff["RF"] = rng.uniform(0.0, 0.01, n)                                      # varying, so omitting RF would bias the fit
    ret = pd.DataFrame({"X": ff["RF"] + 1.5 * ff["Mkt-RF"] - 0.5 * ff["HML"]}, index=idx)   # exact, no noise
    if ff_months is not None:
        ff = ff.iloc[:ff_months]
    return ff, ret


def test_betas_recover_the_true_loadings_from_excess_returns():
    ff, ret = _ff_and_returns()
    b = estimate_ff5_betas(ret, window=24, ff=ff)
    x = b.xs("X", level="Ticker")
    assert x["beta_mkt"].tolist() == pytest.approx([1.5] * len(x), abs=1e-6)
    assert x["beta_hml"].tolist() == pytest.approx([-0.5] * len(x), abs=1e-6)
    assert x["beta_smb"].abs().max() < 1e-6


def test_beta_stamped_at_t_uses_data_only_through_t_minus_1_and_needs_a_full_window():
    ff, ret = _ff_and_returns()
    b = estimate_ff5_betas(ret, window=24, ff=ff).xs("X", level="Ticker")
    assert b.index.min() == ret.index[24]            # first full window ends at index 23, known one month later


def test_future_returns_cannot_change_an_earlier_beta():
    ff, ret = _ff_and_returns()
    base = estimate_ff5_betas(ret, window=24, ff=ff).xs("X", level="Ticker")
    tampered = ret.copy()
    tampered.iloc[60:] = np.random.default_rng(5).normal(0, 0.2, tampered.iloc[60:].shape)
    alt = estimate_ff5_betas(tampered, window=24, ff=ff).xs("X", level="Ticker")
    cut = ret.index[60]                              # a beta stamped at month 60 uses data through month 59
    pd.testing.assert_frame_equal(base.loc[:cut], alt.loc[:cut])
    assert not np.allclose(base.loc[ret.index[61]:], alt.loc[ret.index[61]:])


def test_unpublished_factor_months_carry_the_last_beta_for_a_limited_time():
    ff, ret = _ff_and_returns(n=80, ff_months=70)    # factors end at index 69; returns continue to 79
    b = estimate_ff5_betas(ret, window=24, ff=ff, max_carry=3).xs("X", level="Ticker")
    last_est = 69                                    # last estimate; stamped at index 70 after the one-month lag
    assert b.index.max() == ret.index[last_est + 1 + 3]
    carried = b.loc[ret.index[last_est + 2]:]
    assert (carried.values == b.loc[ret.index[last_est + 1]].values).all()      # carried, not re-estimated


def test_ticker_with_fewer_than_a_window_of_returns_gets_no_beta():
    ff, ret = _ff_and_returns()
    ret["Y"] = np.nan
    ret.iloc[-10:, ret.columns.get_loc("Y")] = 0.01
    b = estimate_ff5_betas(ret, window=24, ff=ff)
    assert "Y" not in set(b.index.get_level_values("Ticker"))
