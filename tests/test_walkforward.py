import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.strategy.simulate import SimConfig
from momentum_rebalancer.strategy.walkforward import (
    config_grid, format_walk_forward, make_folds, objective_value, run_walk_forward,
)

# ── folds ─────────────────────────────────────────────────────────────────────

DATES = pd.date_range("2012-01-31", periods=120, freq="ME")


def test_blocks_are_ordered_and_never_overlap():
    folds = make_folds(DATES, train_months=48, val_months=12, test_months=12, step_months=12, embargo_months=1)
    assert len(folds) >= 4
    for f in folds:
        assert f.train[0] == DATES[0]
        assert f.train[1] < f.val[0] <= f.val[1] < f.test[0] <= f.test[1]
        gap = DATES.get_loc(f.test[0]) - DATES.get_loc(f.val[1]) - 1
        assert gap == 1                                          # the embargo month
    for a, b in zip(folds, folds[1:]):
        assert a.test[1] < b.test[0]                             # test blocks do not overlap
        assert b.train[1] > a.train[1]                           # the training window expands


def test_the_last_fold_fits_inside_the_data():
    folds = make_folds(DATES, 48, 12, 12, 12, 1)
    assert folds[-1].test[1] <= DATES[-1]


def test_step_smaller_than_test_is_rejected():
    with pytest.raises(ValueError):
        make_folds(DATES, 48, 12, 12, 6, 1)


def test_too_little_history_gives_no_folds():
    assert make_folds(DATES[:50], 48, 12, 12, 12, 1) == []


def test_objective_value():
    r = pd.Series([0.02, 0.01, 0.03, 0.0, 0.02])
    assert objective_value(r) == pytest.approx(r.mean() / r.std() * np.sqrt(12))
    assert objective_value(r, "cumulative_return") == pytest.approx((1 + r).prod() - 1)
    assert objective_value(pd.Series([0.01])) == -np.inf        # undefined
    assert objective_value(pd.Series([0.01, 0.01, 0.01])) == -np.inf   # zero variance


def test_grid_is_the_full_product():
    g = config_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and {"a": 2, "b": "y"} in g


# ── a world where the answer is known ─────────────────────────────────────────

N_M, N_T = 100, 12


def _world(seed=0):
    """`return_6m` is an oracle: it equals the NEXT month's return. `return_3m` is noise."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2012-01-31", periods=N_M, freq="ME", name="Date")
    tickers = [f"T{i:02d}" for i in range(N_T)]
    rets = pd.DataFrame(rng.normal(0.008, 0.06, (N_M, N_T)), index=months, columns=tickers)
    nxt = rets.shift(-1)
    rows = []
    for d in months[:-1]:
        for t in tickers:
            rows.append((d, t, rng.normal(), nxt.at[d, t]))
    modeling = pd.DataFrame(rows, columns=["Date", "Ticker", "return_3m", "return_6m"]).set_index(["Date", "Ticker"])
    return modeling, rets


def _fake_cluster(data, cols, method, kw):
    k = kw["n_clusters"]
    out = data.copy()
    out["cluster"] = [int(t[1:]) % k for t in out.index.get_level_values("Ticker")]
    return out


WF = {
    "splits": {"train_months": 48, "val_months": 12, "test_months": 12, "step_months": 12, "embargo_months": 1},
    "objective": "sharpe", "min_months": 12,
    "grid": {"momentum_feature": ["return_3m", "return_6m"], "top_n_per_cluster": [2, 3], "n_clusters": [3, 4]},
}
BASE = SimConfig(momentum_feature="return_3m", top_n=2, portfolio_size=6, optimiser="equal", cost_bps=10)
BASELINE = {"momentum_feature": "return_3m", "top_n_per_cluster": 2, "n_clusters": 3}


def _run(modeling, rets, wf=WF):
    return run_walk_forward(modeling, rets, wf, BASE, BASELINE, "kmeans", ["return_3m", "return_6m"],
                            cluster_fn=_fake_cluster)


@pytest.fixture(scope="module")
def world_and_result():
    modeling, rets = _world()
    return modeling, rets, _run(modeling, rets)


def test_walk_forward_finds_the_oracle_feature_out_of_sample(world_and_result):
    _, _, res = world_and_result
    assert res.n_configs == 8 and len(res.folds) >= 3
    assert (res.folds["chosen_momentum_feature"] == "return_6m").all()
    # the oracle config, frozen before each test block, wins out of sample; the noise baseline does not
    assert res.summary["walk_forward"]["cumulative_return"] > res.summary["fixed_baseline"]["cumulative_return"] + 0.5
    assert (res.folds["test_return"] > 0).all()


def test_oos_series_covers_only_test_months_and_is_labelled_by_month_earned(world_and_result):
    _, _, res = world_and_result
    n_expected = len(res.folds) * WF["splits"]["test_months"]
    assert len(res.oos_returns) == n_expected == len(res.baseline_returns)
    assert res.oos_returns.index.is_monotonic_increasing and res.oos_returns.index.is_unique
    first_test_signal = pd.Timestamp(res.folds["test"].iloc[0].split("..")[0] + "-01") + pd.offsets.MonthEnd(0)
    assert res.oos_returns.index[0] == first_test_signal + pd.offsets.MonthEnd(1)


def test_changing_the_test_period_cannot_change_what_was_chosen(world_and_result):
    """Selection for fold 0 must not depend on anything after its validation block."""
    modeling, rets, base = world_and_result
    f0_test_start = pd.Timestamp(base.folds["test"].iloc[0].split("..")[0] + "-01") + pd.offsets.MonthEnd(0)

    tampered_rets = rets.copy()
    later = tampered_rets.index > f0_test_start
    tampered_rets.loc[later] = np.random.default_rng(99).normal(-0.02, 0.2, tampered_rets.loc[later].shape)
    tampered_mod = modeling.copy()
    late = tampered_mod.index.get_level_values("Date") >= f0_test_start
    tampered_mod.loc[late, "return_6m"] = np.random.default_rng(98).normal(size=late.sum())
    alt = _run(tampered_mod, tampered_rets)

    cols = [c for c in base.folds.columns if c.startswith("chosen_")] + ["train_sharpe", "val_sharpe"]
    pd.testing.assert_series_equal(base.folds.loc[0, cols], alt.folds.loc[0, cols], check_names=False)
    assert base.folds.loc[0, "test_return"] != alt.folds.loc[0, "test_return"]      # the test result does change


def test_a_config_with_too_few_validation_months_is_not_eligible(world_and_result):
    modeling, rets, _ = world_and_result
    wf = {**WF, "min_months": 13, "grid": {"momentum_feature": ["return_6m"], "top_n_per_cluster": [2], "n_clusters": [3]}}
    res = _run(modeling, rets, wf)                               # validation has 12 months: nothing is eligible
    assert res.folds["chosen_momentum_feature"].notna().all()    # falls back to the first config rather than crashing


def test_report_mentions_the_caveats(world_and_result):
    text = format_walk_forward(world_and_result[2])
    assert "out-of-sample" in text and "net of costs" in text and "partly luck" in text


def test_not_enough_months_raises_a_helpful_error():
    modeling, rets = _world()
    short_rets = rets.iloc[:60]
    short_mod = modeling[modeling.index.get_level_values("Date") <= short_rets.index[-2]]
    with pytest.raises(ValueError, match="Not enough months"):
        _run(short_mod, short_rets)
