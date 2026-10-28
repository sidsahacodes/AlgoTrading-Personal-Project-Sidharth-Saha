import shutil
from pathlib import Path

import pytest
import yaml

from momentum_rebalancer import config as C

ORIGINAL_FEATURE_COLS = [
    "gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank",
    "return_1m", "return_2m", "return_3m", "return_6m", "return_9m", "return_12m",
    "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
]


def test_baseline_values():
    """Pin the documented baseline so an accidental edit to strategy.yaml is caught."""
    assert C.REGIME == "baseline"
    assert (C.LOOKBACK_YEARS, C.MIN_HISTORY_MONTHS, C.UNIVERSE_SIZE) == (15, 36, 150)
    assert (C.RSI_PERIOD, C.ATR_PERIOD, C.BB_PERIOD) == (14, 14, 20)
    assert (C.MACD_FAST, C.MACD_SLOW, C.MACD_SIGNAL) == (12, 26, 9)
    assert C.FF_BETA_WINDOW == 24
    assert C.RETURN_LAGS == [1, 2, 3, 6, 9, 12]
    assert C.FEATURE_COLS == ORIGINAL_FEATURE_COLS
    assert (C.CLUSTER_METHOD, C.N_CLUSTERS, C.PCA_COMPONENTS) == ("kmeans", 5, 6)
    assert (C.HDBSCAN_MIN_SIZE, C.RANDOM_STATE) == (15, 42)
    assert (C.MOMENTUM_FEATURE, C.TOP_N_PER_CLUSTER, C.PORTFOLIO_SIZE) == ("return_3m", 4, 20)
    assert C.OPTIMISER == "hrp"
    assert C.PORTFOLIO_VALUE == 1_000_000
    assert (C.TWS_HOST, C.TWS_PORT) == ("127.0.0.1", 7497)
    assert (C.DASHBOARD_HOST, C.DASHBOARD_PORT, C.DEBUG) == ("127.0.0.1", 5001, False)


def test_data_and_backtest_assumptions():
    assert C.MAX_STALE_DAYS == 7
    assert C.PIT_ALLOW_FALLBACK is False
    assert "%28Updated%29" in C.PIT_URL                 # the maintained membership file, not the 2019 snapshot
    assert C.COST_BPS == 10.0
    assert C.DELISTING_RETURN == -0.30


def test_new_knobs_are_neutral_at_baseline():
    assert C.HRP_LOOKBACK_MONTHS is None
    assert C.EXPOSURE_SCALAR == 1.0
    assert C.MIN_TRADE_PCT == 0.0


def test_default_broker_is_paper():
    # Safe default: no real orders unless MR_BROKER=ibkr (or strategy.yaml) says so.
    assert C.BROKER == "paper"


def test_every_regime_loads_and_validates():
    for name in C.available_regimes():
        s = C.load_settings(name)
        assert s["regime"] == name


def test_regime_overrides_apply():
    s = C.load_settings("crash")
    assert s["exposure_scalar"] == 0.6
    assert s["universe_size"] == 150          # untouched keys keep baseline values


def test_presets_keep_portfolio_size_consistent_with_clusters():
    """N_CLUSTERS x TOP_N must equal PORTFOLIO_SIZE, else live trims names the backtest keeps."""
    for name in C.available_regimes():
        s = C.load_settings(name)
        assert s["n_clusters"] * s["top_n_per_cluster"] == s["portfolio_size"], name


def test_unknown_regime_rejected():
    with pytest.raises(ValueError, match="Unknown regime"):
        C.load_settings("nope")


@pytest.fixture
def cfg_copy(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(C.CONFIG_DIR, d)
    return d


def _edit(path: Path, fn):
    data = yaml.safe_load(path.read_text())
    fn(data)
    path.write_text(yaml.safe_dump(data))


def test_override_of_unknown_key_rejected(cfg_copy):
    _edit(cfg_copy / "regimes.yaml", lambda d: d["stress"]["overrides"].update({"not_a_key": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        C.load_settings("stress", config_dir=cfg_copy)


def test_momentum_feature_must_be_a_feature(cfg_copy):
    _edit(cfg_copy / "strategy.yaml", lambda d: d["strategy"].update({"momentum_feature": "return_5m"}))
    with pytest.raises(ValueError, match="momentum_feature"):
        C.load_settings("baseline", config_dir=cfg_copy)


def test_return_lags_and_feature_cols_must_agree(cfg_copy):
    _edit(cfg_copy / "strategy.yaml", lambda d: d["features"].update({"return_lags": [1, 2, 3, 6, 9]}))
    with pytest.raises(ValueError, match="return_lags"):
        C.load_settings("baseline", config_dir=cfg_copy)


def test_bad_exposure_and_lookback_rejected(cfg_copy):
    _edit(cfg_copy / "strategy.yaml", lambda d: d["strategy"].update({"exposure_scalar": 1.5}))
    with pytest.raises(ValueError, match="exposure_scalar"):
        C.load_settings("baseline", config_dir=cfg_copy)
    _edit(cfg_copy / "strategy.yaml", lambda d: d["strategy"].update({"exposure_scalar": 1.0, "hrp_lookback_months": 6}))
    with pytest.raises(ValueError, match="hrp_lookback_months"):
        C.load_settings("baseline", config_dir=cfg_copy)


def test_duplicate_keys_across_sections_rejected(cfg_copy):
    _edit(cfg_copy / "strategy.yaml", lambda d: d["broker"].update({"n_clusters": 3}))
    with pytest.raises(ValueError, match="Duplicate config key"):
        C.load_settings("baseline", config_dir=cfg_copy)


def test_env_broker_override(monkeypatch):
    monkeypatch.setenv("MR_BROKER", "ibkr")
    assert C.load_settings()["broker"] == "ibkr"
    monkeypatch.setenv("MR_BROKER", "bogus")
    with pytest.raises(ValueError, match="broker"):
        C.load_settings()
