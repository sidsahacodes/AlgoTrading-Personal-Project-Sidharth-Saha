import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.strategy.optimiser import compute_weights, equal_weight, hrp


def _returns(n=120, seed=1):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    cols = {}
    for i in range(4):                      # group 1: high-vol, driven by f1
        cols[f"A{i}"] = 0.08 * f1 + 0.01 * rng.normal(size=n)
    for i in range(4):                      # group 2: low-vol, driven by f2
        cols[f"B{i}"] = 0.02 * f2 + 0.005 * rng.normal(size=n)
    return pd.DataFrame(cols)


def test_equal_weight_sums_to_one():
    w = equal_weight(["a", "b", "c", "d"])
    assert pytest.approx(sum(w.values())) == 1.0
    assert set(w.values()) == {0.25}


def test_hrp_sums_to_one_positive_and_favours_low_vol():
    w = hrp(_returns())
    assert pytest.approx(sum(w.values())) == 1.0
    assert all(v > 0 for v in w.values())
    assert sum(w[f"B{i}"] for i in range(4)) > sum(w[f"A{i}"] for i in range(4))


def test_hrp_single_asset():
    assert hrp(_returns()[["A0"]]) == {"A0": 1.0}


def test_hrp_runs_with_fewer_observations_than_assets():
    """Documented behaviour: singular covariance, but no inversion, so it still returns weights."""
    w = hrp(_returns(n=6))
    assert pytest.approx(sum(w.values())) == 1.0


def test_compute_weights_falls_back_to_equal_without_returns():
    w = compute_weights(["a", "b"], returns=None, method="hrp")
    assert w == {"a": 0.5, "b": 0.5}


def test_compute_weights_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_weights(["a"], method="mvo")
