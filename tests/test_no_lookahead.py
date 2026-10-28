"""
Look-ahead detector for the feature pipeline.

Truncation invariance: features for every month up to a cut-off must be identical
whether the pipeline sees only prices up to the cut-off or all prices. If any value
differs, that feature used information from after the date it is stamped with.
"""
import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer import config as C

CUT = pd.Timestamp("2023-06-30")


def _prices(n_tickers=45, start="2015-01-02", end="2025-12-31", seed=11):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    frames = []
    for i in range(n_tickers):
        r = rng.normal(0.0004, 0.015, len(days))
        # a few fat-tailed days so winsorisation / z-scoring actually bite
        r[rng.integers(0, len(days), 6)] += rng.choice([-0.25, 0.3], 6)
        close = 50 * np.exp(np.cumsum(r))
        open_ = close * (1 + rng.normal(0, 0.003, len(days)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(days))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(days))))
        vol = rng.integers(1_000_000, 5_000_000, len(days)).astype(float) * (1 + i / 10)
        frames.append(pd.DataFrame({"adj close": close, "close": close, "high": high, "low": low,
                                    "open": open_, "volume": vol, "Date": days, "Ticker": f"S{i:02d}"}))
    return pd.concat(frames).set_index(["Date", "Ticker"]).sort_index()


def _ff(start="2014-01", end="2025-12", seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, end, freq="M")
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df = pd.DataFrame(rng.normal(0, 2, (len(idx), 6)), index=idx, columns=cols)
    df["RF"] = 0.2
    return df


def _build(monkeypatch, prices, tmp_name):
    from momentum_rebalancer.features import pipeline, returns
    tickers = sorted(prices.index.get_level_values("Ticker").unique())
    pit = pd.DataFrame({"tickers": [tickers]}, index=pd.DatetimeIndex(["2000-01-01"], name="date"))
    monkeypatch.setattr(pipeline, "load_pit_members", lambda force_refresh=False: pit)
    monkeypatch.setattr(pipeline, "download_prices", lambda **kw: prices)
    monkeypatch.setattr(returns.web, "DataReader", lambda *a, **k: [_ff()])
    monkeypatch.setattr(pipeline, "MODELING_DATA_PATH", C.DATA_DIR / f"{tmp_name}.parquet")
    for name in ("MONTHLY_RETURNS_PATH",):
        if hasattr(pipeline, name):
            monkeypatch.setattr(pipeline, name, C.DATA_DIR / f"{tmp_name}_mret.parquet")
    return pipeline.build_modeling_data(force_refresh=True)


def test_features_are_truncation_invariant(monkeypatch):
    prices = _prices()
    full = _build(monkeypatch, prices, "lookahead_full")
    part = _build(monkeypatch, prices[prices.index.get_level_values("Date") <= CUT], "lookahead_cut")

    f = full[full.index.get_level_values("Date") <= CUT]
    p = part[part.index.get_level_values("Date") <= CUT]

    assert len(p) > 200, "test needs a meaningful number of pre-cut rows"
    assert f.index.equals(p.index), (
        f"row sets differ: only in full={len(f.index.difference(p.index))}, "
        f"only in truncated={len(p.index.difference(f.index))}"
    )
    bad = {c: float((f[c] - p[c]).abs().max()) for c in f.columns
           if not np.allclose(f[c], p[c], atol=1e-9, rtol=1e-7, equal_nan=True)}
    assert not bad, f"features changed when future data was removed (look-ahead): {bad}"


def test_technical_indicators_do_not_depend_on_other_tickers():
    """A ticker's indicators must be identical whether or not other tickers are in the frame."""
    from momentum_rebalancer.features.technical import compute_all_technicals
    prices = _prices(n_tickers=4)
    subset = prices[prices.index.get_level_values("Ticker").isin(["S01", "S02"])]     # other tickers removed
    alone = compute_all_technicals(subset).xs("S01", level="Ticker", drop_level=False)
    joint = compute_all_technicals(prices).xs("S01", level="Ticker", drop_level=False)
    for col in ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd"]:
        assert np.allclose(alone[col].dropna().values, joint.loc[alone[col].dropna().index, col].values,
                           atol=1e-9, equal_nan=True), f"{col} depends on other tickers"


# ── short and odd series (index members that traded only a few days) ──────────

def _series(name, n, seed, start="2015-01-02"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n)
    r = rng.normal(0.0004, 0.015, n)
    px = 30 * np.exp(np.cumsum(r))
    return pd.DataFrame({"adj close": px, "close": px, "high": px * 1.01, "low": px * 0.99, "open": px,
                         "volume": 1e6, "Date": d, "Ticker": name})


@pytest.mark.parametrize("short_lengths", [[3], [1], [10], [3, 10, 1, 13], [14]])
def test_tickers_shorter_than_an_indicator_window_do_not_break_the_calculation(short_lengths):
    from momentum_rebalancer.features.technical import compute_all_technicals
    long_ = _series("LONG", 300, seed=1)
    shorts = [_series(f"S{i}", n, seed=10 + i) for i, n in enumerate(short_lengths)]
    prices = pd.concat([long_, *shorts]).set_index(["Date", "Ticker"]).sort_index()
    out = compute_all_technicals(prices)

    assert out.index.equals(prices.index) and out.index.is_unique
    cols = ["gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd"]
    assert set(cols) <= set(out.columns)
    # a short ticker simply has no value for an indicator it cannot compute; it must never poison others
    alone = compute_all_technicals(pd.concat([long_, _series("PAD", 300, seed=99)]).set_index(["Date", "Ticker"]).sort_index())
    for c in cols:
        a = alone.xs("LONG", level="Ticker")[c].dropna()
        b = out.xs("LONG", level="Ticker").loc[a.index, c]
        assert np.allclose(a.values, b.values, atol=1e-9, equal_nan=True), c


def test_a_frame_with_a_single_ticker_still_works():
    from momentum_rebalancer.features.technical import compute_all_technicals
    prices = _series("ONE", 120, seed=3).set_index(["Date", "Ticker"]).sort_index()
    out = compute_all_technicals(prices)
    assert out["rsi"].notna().sum() > 80 and out["macd"].notna().sum() > 60


def test_duplicate_rows_are_dropped_not_fatal():
    from momentum_rebalancer.features.technical import compute_all_technicals
    p = pd.concat([_series("A", 100, 1), _series("B", 100, 2)]).set_index(["Date", "Ticker"]).sort_index()
    dup = pd.concat([p, p.iloc[:5]])
    out = compute_all_technicals(dup)
    assert out.index.is_unique and len(out) == len(p)
