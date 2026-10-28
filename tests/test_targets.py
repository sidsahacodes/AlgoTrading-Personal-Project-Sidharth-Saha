import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.strategy.targets import history_for, scale_exposure, target_weights


def _returns(n_months=60, tickers=("A", "B", "C", "D", "E", "F"), seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME", name="Date")
    return pd.DataFrame(rng.normal(0, 0.05, (n_months, len(tickers))), index=idx, columns=list(tickers))


def test_scale_exposure():
    assert scale_exposure({"A": 0.6, "B": 0.4}, 0.5) == {"A": 0.3, "B": 0.2}


def test_hrp_weights_sum_to_exposure_scalar():
    w = target_weights(list("ABCDEF"), returns_wide=_returns(), optimiser="hrp",
                       hrp_lookback_months=None, exposure_scalar=0.6)
    assert pytest.approx(sum(w.values())) == 0.6


def test_lookback_limits_history():
    r = _returns()
    assert len(history_for(list("ABC"), r, lookback_months=None)) == 60
    assert len(history_for(list("ABC"), r, lookback_months=36)) == 36


def test_recent_window_changes_weights():
    r = _returns()
    w_all = target_weights(list("ABCDEF"), returns_wide=r, hrp_lookback_months=None, exposure_scalar=1.0)
    w_36 = target_weights(list("ABCDEF"), returns_wide=r, hrp_lookback_months=36, exposure_scalar=1.0)
    assert w_all != w_36


def test_missing_tickers_are_dropped_not_fatal():
    w = target_weights(["A", "B", "C", "ZZZ"], returns_wide=_returns(), hrp_lookback_months=None, exposure_scalar=1.0)
    assert "ZZZ" not in w and pytest.approx(sum(w.values())) == 1.0


def test_no_valid_tickers_raises():
    with pytest.raises(ValueError):
        history_for(["ZZZ"], _returns())


def test_equal_optimiser_ignores_history():
    w = target_weights(["A", "B"], returns_wide=None, optimiser="equal", exposure_scalar=0.5)
    assert w == {"A": 0.25, "B": 0.25}


def test_ticker_with_too_little_history_is_left_out_of_hrp():
    r = _returns()
    r.loc[r.index[:-10], "F"] = np.nan                       # F has only 10 observations
    w = target_weights(list("ABCDEF"), returns_wide=r, hrp_lookback_months=None, exposure_scalar=1.0)
    assert "F" not in w and pytest.approx(sum(w.values())) == 1.0
