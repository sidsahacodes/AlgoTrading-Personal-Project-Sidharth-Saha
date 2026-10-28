"""
Run the full feature pipeline on synthetic prices with the network stubbed out.
Checks that the pipeline modules compose and produce exactly FEATURE_COLS; it says
nothing about the economic quality of the features.
"""
import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer import config as C


def _synthetic_prices(n_tickers=45, start="2017-01-02", end="2025-12-31", seed=11):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    frames = []
    for i in range(n_tickers):
        r = rng.normal(0.0004, 0.015, len(days))
        close = 50 * np.exp(np.cumsum(r))
        open_ = close * (1 + rng.normal(0, 0.003, len(days)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(days))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(days))))
        vol = rng.integers(1_000_000, 5_000_000, len(days)).astype(float) * (1 + i / 10)
        frames.append(pd.DataFrame({
            "adj close": close, "close": close, "high": high, "low": low,
            "open": open_, "volume": vol,
            "Date": days, "Ticker": f"S{i:02d}",
        }))
    df = pd.concat(frames).set_index(["Date", "Ticker"]).sort_index()
    return df


def _synthetic_ff(start="2016-01", end="2025-12", seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, end, freq="M")
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    return pd.DataFrame(rng.normal(0, 2, (len(idx), 6)), index=idx, columns=cols)


@pytest.fixture
def stubbed(monkeypatch):
    from momentum_rebalancer.features import pipeline, returns

    prices = _synthetic_prices()
    tickers = sorted(prices.index.get_level_values("Ticker").unique())
    pit = pd.DataFrame({"tickers": [tickers]}, index=pd.DatetimeIndex(["2000-01-01"], name="date"))

    monkeypatch.setattr(pipeline, "load_pit_members", lambda force_refresh=False: pit)
    monkeypatch.setattr(pipeline, "download_prices", lambda **kw: prices)
    monkeypatch.setattr(returns.web, "DataReader", lambda *a, **k: [_synthetic_ff()])
    monkeypatch.setattr(pipeline, "MODELING_DATA_PATH", C.DATA_DIR / "modeling_data_fp_test.parquet")
    yield pipeline
    (C.DATA_DIR / "modeling_data_fp_test.parquet").unlink(missing_ok=True)
    (C.DATA_DIR / "modeling_data_fp_test.meta.json").unlink(missing_ok=True)


def test_full_pipeline_produces_exact_feature_set(stubbed):
    data = stubbed.build_modeling_data(force_refresh=True)

    assert list(data.columns) == C.FEATURE_COLS
    assert data.index.names == ["Date", "Ticker"]
    assert len(data) > 0
    assert data.index.get_level_values("Ticker").nunique() > 10
    assert not data.isna().any().any()

    # cross-sectionally z-scored columns have ~zero mean per month
    zcols = ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank"]
    month_means = data[zcols].groupby("Date").mean().abs().max()
    assert (month_means < 1e-6).all()


def test_pipeline_writes_fingerprint_and_roundtrips(stubbed):
    from momentum_rebalancer.storage.fingerprint import diff_params, FEATURE_KEYS
    stubbed.build_modeling_data(force_refresh=True)
    assert diff_params(stubbed.MODELING_DATA_PATH, C.SETTINGS, FEATURE_KEYS) == []
    again = stubbed.build_modeling_data(force_refresh=False)       # served from cache
    assert list(again.columns) == C.FEATURE_COLS


def test_pit_filter_removes_non_members(stubbed, monkeypatch):
    """A ticker that is not in the index at a month-end must not appear in that month."""
    prices = stubbed.download_prices()
    tickers = sorted(prices.index.get_level_values("Ticker").unique())
    late_joiner = tickers[0]
    pit = pd.DataFrame(
        {"tickers": [tickers[1:], tickers]},
        index=pd.DatetimeIndex(["2000-01-01", "2023-01-01"], name="date"),
    )
    monkeypatch.setattr(stubbed, "load_pit_members", lambda force_refresh=False: pit)
    data = stubbed.build_modeling_data(force_refresh=True)
    dates = data.xs(late_joiner, level="Ticker").index
    assert len(dates) > 0 and dates.min() >= pd.Timestamp("2023-01-01")
