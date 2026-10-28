import pandas as pd
import pytest

from momentum_rebalancer.strategy.backtest import PERIODS, compute_metrics, slice_period


def test_constant_return_metrics():
    r = pd.Series(0.01, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    m = compute_metrics(r)
    assert m["cumulative_return"] == pytest.approx(1.01**24 - 1, abs=1e-4)
    assert m["annualised_return"] == pytest.approx(1.01**12 - 1, abs=1e-4)
    assert m["max_drawdown"] == 0
    assert m["win_rate"] == 1.0


def test_drawdown_known_value():
    r = pd.Series([0.10, -0.20, 0.05], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    assert compute_metrics(r)["max_drawdown"] == pytest.approx(-0.20)


def test_benchmark_alpha_beta():
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    b = pd.Series([0.01, -0.02, 0.03] * 12, index=idx)
    m = compute_metrics(2 * b, b)
    assert m["beta"] == pytest.approx(2.0, abs=1e-3)


def test_empty_returns():
    assert compute_metrics(pd.Series([], dtype=float)) == {}


def test_slice_period():
    r = pd.Series(range(100), index=pd.date_range("2018-01-31", periods=100, freq="ME"))
    assert len(slice_period(r, "1Y")) == 12
    assert len(slice_period(r, "All")) == 100
    assert set(PERIODS) >= {"1M", "3M", "6M", "1Y", "3Y", "5Y", "All"}
