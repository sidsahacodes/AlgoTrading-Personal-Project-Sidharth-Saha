import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.strategy import regime as R


def test_drawdown_from_high():
    s = pd.Series([100, 110, 120, 90.0])
    assert R.drawdown_from_high(s) == pytest.approx(90 / 120 - 1)
    assert R.drawdown_from_high(pd.Series([100, 101, 102.0])) == 0.0
    assert np.isnan(R.drawdown_from_high(pd.Series([], dtype=float)))


def test_drawdown_uses_trailing_window():
    s = pd.Series([200] + [100] * 10 + [90.0])            # old 200 high outside the window
    assert R.drawdown_from_high(s, window=5) == pytest.approx(90 / 100 - 1)


def test_avg_pairwise_corr_extremes():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    corr = pd.DataFrame({f"c{i}": base for i in range(4)})
    indep = pd.DataFrame(rng.normal(size=(5000, 4)))
    assert R.avg_pairwise_corr(corr) == pytest.approx(1.0)
    assert abs(R.avg_pairwise_corr(indep)) < 0.05
    assert np.isnan(R.avg_pairwise_corr(corr[["c0"]]))


def test_corr_shift_detects_recent_jump():
    rng = np.random.default_rng(1)
    calm = rng.normal(size=(100, 5))
    common = rng.normal(size=(12, 1))
    stressed = common + 0.1 * rng.normal(size=(12, 5))
    df = pd.DataFrame(np.vstack([calm, stressed]))
    recent, longrun = R.corr_shift(df, recent_months=12)
    assert recent - longrun > 0.5


def test_level_thresholds():
    assert R._level(-0.05, -0.10, -0.20, higher_is_worse=False) == "ok"
    assert R._level(-0.12, -0.10, -0.20, higher_is_worse=False) == "caution"
    assert R._level(-0.25, -0.10, -0.20, higher_is_worse=False) == "stress"
    assert R._level(20, 25, 30, higher_is_worse=True) == "ok"
    assert R._level(35, 25, 30, higher_is_worse=True) == "stress"
    assert R._level(float("nan"), 25, 30, higher_is_worse=True) == "n/a"


def test_build_flags_and_report():
    spy = pd.Series(np.r_[np.linspace(100, 120, 250), np.linspace(120, 90, 20)])
    vix = pd.Series([18.0, 33.0])
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal(size=(60, 5)))
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-31", periods=24, freq="ME"), list("ab")],
                                     names=["Date", "Ticker"])
    modeling = pd.DataFrame({"return_3m": rng.normal(size=len(idx))}, index=idx)
    flags = R.build_flags(spy, vix, rets, modeling, spy_dd_caution=-0.10, spy_dd_stress=-0.20,
                          vix_caution=25, vix_stress=30, corr_jump=0.15)
    by = {f.name: f for f in flags}
    assert by["SPY drawdown from 12m high"].level == "stress"        # -25%
    assert by["VIX"].level == "stress"
    text = R.format_report("baseline", {}, flags, [], [])
    assert "REGIME REPORT" in text and "nothing was changed" in text


def test_build_flags_handles_failed_downloads():
    flags = R.build_flags(None, None, None, None, spy_dd_caution=-0.1, spy_dd_stress=-0.2,
                          vix_caution=25, vix_stress=30, corr_jump=0.15)
    assert all(f.level == "n/a" for f in flags)
    text = R.format_report("baseline", {}, flags, [], [])
    assert "UNAVAILABLE" in text and "not an all-clear" in text


def test_window_metrics():
    idx = pd.date_range("2019-01-31", periods=36, freq="ME")
    r = pd.Series(0.01, index=idx)
    b = pd.Series(0.005, index=idx)
    out = R.window_metrics(r, b, {"w": ("2020-01", "2020-06"), "empty": ("2030-01", "2030-02"), "all": (None, None)})
    assert out.loc["w", "months"] == 6
    assert out.loc["w", "strategy_cum"] == pytest.approx(1.01**6 - 1, abs=1e-4)
    assert out.loc["empty", "months"] == 0
    assert out.loc["all", "months"] == 36
