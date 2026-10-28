"""Synthetic data through clustering -> fingerprints -> selection -> weights -> trades (no network)."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_modeling
from momentum_rebalancer import config as C
from momentum_rebalancer.storage.fingerprint import FEATURE_KEYS, stale_warnings, write_meta


@pytest.fixture
def populated(monkeypatch):
    """Write a synthetic modeling_data.parquet + meta into the (temp) data dir."""
    from momentum_rebalancer.features import pipeline
    df = make_modeling(n_months=30, n_tickers=80, seed=7)
    df.to_parquet(pipeline.MODELING_DATA_PATH)
    write_meta(pipeline.MODELING_DATA_PATH, C.SETTINGS, FEATURE_KEYS)
    dates = df.index.get_level_values("Date").unique().sort_values()
    tickers = df.index.get_level_values("Ticker").unique()
    rng = np.random.default_rng(1)
    full = pd.date_range(dates.min(), periods=len(dates) + 1, freq="ME", name="Date")   # one extra month to earn
    pd.DataFrame(rng.normal(0.01, 0.05, (len(full), len(tickers))), index=full, columns=tickers
                 ).to_parquet(pipeline.MONTHLY_RETURNS_PATH)
    yield df
    for p in C.DATA_DIR.glob("*"):
        p.unlink()


@pytest.mark.parametrize("method", ["kmeans", "gmm", "pca_kmeans"])
def test_cluster_and_select(populated, method, monkeypatch):
    from momentum_rebalancer.clustering import runner
    from momentum_rebalancer.strategy.signals import select_top_momentum

    clustered = runner.run_clustering(populated, method=method, force_refresh=True)
    assert clustered["cluster"].nunique() == C.N_CLUSTERS
    month = runner.get_current_clusters(clustered)
    sel = select_top_momentum(month, C.MOMENTUM_FEATURE, C.TOP_N_PER_CLUSTER)
    sizes = month["cluster"].value_counts()
    expected_n = min(C.PORTFOLIO_SIZE, int(sum(min(C.TOP_N_PER_CLUSTER, n) for n in sizes)))
    assert len(sel) == expected_n          # clusters smaller than top_n contribute all members
    # selected names really are the top momentum within their cluster
    for cid, grp in month.groupby("cluster"):
        expected = set(grp.nlargest(C.TOP_N_PER_CLUSTER, C.MOMENTUM_FEATURE).index)
        assert expected == set(sel[sel.cluster == cid].index)


def test_cache_warns_when_cluster_params_change(populated):
    from momentum_rebalancer.clustering import runner
    runner.run_clustering(populated, method="kmeans", force_refresh=True)
    assert stale_warnings(C.SETTINGS, C.DATA_DIR) == []
    changed = {**C.SETTINGS, "n_clusters": 7}
    assert any("n_clusters" in w for w in stale_warnings(changed, C.DATA_DIR))


def test_plan_rebalance_paper_broker(populated, monkeypatch):
    from momentum_rebalancer.clustering import runner
    from momentum_rebalancer.strategy import targets

    runner.run_clustering(populated, method=C.CLUSTER_METHOD, force_refresh=True)
    ledger = C.DATA_DIR / "paper_ledger.json"
    ledger.unlink(missing_ok=True)

    selected, target, current, value, trades = targets.plan_rebalance()
    assert current == {}                                   # empty paper ledger
    assert value == C.PORTFOLIO_VALUE
    assert len(selected) == C.PORTFOLIO_SIZE
    assert sum(target.values()) == pytest.approx(C.EXPOSURE_SCALAR)
    assert all(t.action == "BUY" for t in trades)
    assert {t.ticker for t in trades} == set(selected.index)


def test_exposure_scalar_reduces_buys(populated):
    from momentum_rebalancer.clustering import runner
    from momentum_rebalancer.strategy import targets
    from momentum_rebalancer.strategy.rebalancer import compute_trades

    runner.run_clustering(populated, method=C.CLUSTER_METHOD, force_refresh=True)
    tickers = targets.current_selection().index.tolist()
    full = targets.target_weights(tickers, exposure_scalar=1.0)
    half = targets.target_weights(tickers, exposure_scalar=0.5)
    t_full = sum(t.delta_pct for t in compute_trades(full, {}, 1_000_000))
    t_half = sum(t.delta_pct for t in compute_trades(half, {}, 1_000_000))
    assert t_half == pytest.approx(0.5 * t_full)


def test_save_flags_write_csvs_to_results_dir(populated, capsys):
    from momentum_rebalancer import cli
    from momentum_rebalancer.clustering import runner

    runner.run_clustering(populated, method=C.CLUSTER_METHOD, force_refresh=True)
    (C.DATA_DIR / "paper_ledger.json").unlink(missing_ok=True)
    before = set(C.RESULTS_DIR.glob("*.csv"))

    cli.main(["signals", "--save"])
    cli.main(["trades", "--save"])
    new = {p.name for p in set(C.RESULTS_DIR.glob("*.csv")) - before}

    assert any(n.startswith("signals_") and n.endswith("_baseline.csv") for n in new)
    assert any(n.startswith("trades_") and n.endswith("_baseline.csv") for n in new)
    for p in C.RESULTS_DIR.glob("*.csv"):
        p.unlink()


def test_no_save_flag_writes_nothing(populated):
    from momentum_rebalancer import cli
    from momentum_rebalancer.clustering import runner

    runner.run_clustering(populated, method=C.CLUSTER_METHOD, force_refresh=True)
    before = set(C.RESULTS_DIR.glob("*.csv"))
    cli.main(["signals"])
    assert set(C.RESULTS_DIR.glob("*.csv")) == before


def test_dashboard_backtest_route_returns_valid_json_without_a_benchmark(populated, monkeypatch):
    """No SPY (offline) must give nulls, never NaN/Infinity, which the browser cannot parse."""
    import json
    from momentum_rebalancer.clustering import runner
    from momentum_rebalancer.dashboard import app as appmod
    from momentum_rebalancer.strategy import backtest as bt

    def _offline(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(bt.yf, "download", _offline)
    runner.run_clustering(populated, method=C.CLUSTER_METHOD, force_refresh=True)
    appmod._backtest_cache.clear()

    r = appmod.app.test_client().get("/api/backtest")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    body = r.get_data(as_text=True)
    assert "NaN" not in body and "Infinity" not in body
    data = json.loads(body)
    assert all(v is None for v in data["equity"]["benchmark"])
    assert "annual_cost_drag" in data["full_metrics"]
    appmod._backtest_cache.clear()
