import json

import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer.data_cleaning.coverage import coverage_by_month, missing_member_months
from momentum_rebalancer.data_cleaning.universe import (
    drop_stale_bars, filter_min_history, month_end_bar_dates,
)
from momentum_rebalancer.data_ingestion import external, loader
from momentum_rebalancer.validation.checks import check_pit_freshness


# ── ticker normalisation & membership ─────────────────────────────────────────

@pytest.mark.parametrize("raw,clean", [
    ("BRK.B", "BRK-B"), ("BF.B", "BF-B"), ("AAMRQ-201312", "AAMRQ"), ("AAPL", "AAPL"), (" MSFT ", "MSFT"),
])
def test_clean_symbol(raw, clean):
    assert loader.clean_symbol(raw) == clean


CSV = 'date,tickers\n2020-01-02,"AAPL,BRK.B,AAMRQ-201312"\n2021-01-04,"AAPL,BRK.B,TSLA"\n'


@pytest.fixture
def pit_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "PIT_MEMBERS_PATH", tmp_path / "pit_members.parquet")
    return tmp_path


def test_membership_list_is_parsed_and_normalised(pit_paths, monkeypatch):
    monkeypatch.setattr(loader, "_download_text", lambda url: CSV)
    pit = loader.load_pit_members(force_refresh=True)
    assert pit["tickers"].iloc[0] == ["AAMRQ", "AAPL", "BRK-B"]
    assert pit["tickers"].iloc[1] == ["AAPL", "BRK-B", "TSLA"]
    assert (pit_paths / "pit_members.meta.json").exists()


def test_failed_download_stops_instead_of_using_todays_members(pit_paths, monkeypatch):
    monkeypatch.setattr(loader, "_download_text", lambda url: None)
    with pytest.raises(RuntimeError, match="survivorship"):
        loader.load_pit_members(force_refresh=True, allow_fallback=False)


def test_fallback_must_be_opted_into(pit_paths, monkeypatch):
    monkeypatch.setattr(loader, "_download_text", lambda url: None)
    monkeypatch.setattr(loader, "load_sp500_tickers", lambda: ["AAPL", "MSFT"])
    pit = loader.load_pit_members(force_refresh=True, allow_fallback=True)
    assert len(pit) == 1 and pit["tickers"].iloc[0] == ["AAPL", "MSFT"]


def test_window_tickers_include_members_at_the_start():
    pit = pd.DataFrame({"tickers": [["A", "B"], ["B", "C"], ["C", "D"], ["E"]]},
                       index=pd.DatetimeIndex(["2010-01-01", "2015-01-01", "2018-01-01", "2030-01-01"], name="date"))
    got = loader.get_window_tickers(pit, pd.Timestamp("2016-06-30"), pd.Timestamp("2019-06-30"))
    assert got == ["B", "C", "D"]            # B,C were members at the start; D joined; A left earlier; E is after the end


def test_pit_freshness_flags_a_frozen_list():
    pit = pd.DataFrame({"tickers": [["A"]]}, index=pd.DatetimeIndex(["2019-01-11"], name="date"))
    assert "reuses that member list" in check_pit_freshness(pit, today="2026-10-28")[0].msg
    assert check_pit_freshness(pit, today="2019-02-01") == []


# ── dates ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("today,expected", [
    ("2026-10-28", "2026-09-30"), ("2026-10-31", "2026-09-30"), ("2026-11-01", "2026-10-31"),
])
def test_last_completed_month_end(today, expected):
    assert loader.last_completed_month_end(pd.Timestamp(today)) == pd.Timestamp(expected)


# ── price download (yfinance stubbed) ─────────────────────────────────────────

def _yf_frame(tickers, start, end):
    days = pd.bdate_range(start, end, name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    rng = np.random.default_rng(0)
    base = 50 + np.cumsum(rng.normal(0, 1, (len(days), len(tickers))), axis=0)
    data = np.concatenate([base, base + 1, base - 1, base, base, np.full_like(base, 1e6)], axis=1)
    return pd.DataFrame(data, index=days, columns=cols)


@pytest.fixture
def dl(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "RAW_PRICES_PATH", tmp_path / "prices_raw.parquet")
    monkeypatch.setattr(loader, "COVERAGE_PATH", tmp_path / "price_coverage.json")
    monkeypatch.setattr(loader, "EXTERNAL_PRICES_DIR", tmp_path / "ext")
    (tmp_path / "ext").mkdir()
    calls = {}

    def fake_download(tickers, start, end, **kw):
        calls.update(tickers=tickers, start=start, end=end)
        have = [t for t in tickers if t != "GONE"]                # Yahoo has no data for GONE
        # Yahoo would stop at `end`; include a few later days to prove the loader trims them
        return _yf_frame(have, start, pd.Timestamp(end) + pd.Timedelta(days=5))

    monkeypatch.setattr(loader.yf, "download", fake_download)
    return tmp_path, calls


def test_download_end_is_the_day_after_the_last_month_end_and_data_is_trimmed(dl):
    _, calls = dl
    prices = loader.download_prices(tickers=["AAA", "BBB"], lookback_years=5, force_refresh=True,
                                    today=pd.Timestamp("2026-10-28"))
    assert pd.Timestamp(calls["end"]) == pd.Timestamp("2026-10-01")          # exclusive end -> includes 09-30
    assert prices.index.get_level_values("Date").max() <= pd.Timestamp("2026-09-30")
    assert pd.Timestamp("2026-09-30") in prices.index.get_level_values("Date")
    assert list(prices.columns) == ["adj close", "close", "high", "low", "open", "volume"] or \
           set(prices.columns) == {"adj close", "close", "high", "low", "open", "volume"}
    assert prices.index.names == ["Date", "Ticker"]


def test_missing_tickers_are_recorded(dl):
    tmp, _ = dl
    loader.download_prices(tickers=["AAA", "GONE"], lookback_years=5, force_refresh=True, today=pd.Timestamp("2026-10-28"))
    cov = json.loads((tmp / "price_coverage.json").read_text())
    assert cov["requested"] == 2 and cov["with_prices"] == 1 and cov["missing"] == ["GONE"]


def test_vendor_file_fills_only_what_yahoo_lacks(dl):
    tmp, _ = dl
    days = pd.bdate_range("2023-01-02", "2024-06-28")
    ext_rows = pd.DataFrame({"date": days, "ticker": "GONE", "open": 10.0, "high": 11.0, "low": 9.0,
                             "close": 10.0, "adj_close": 10.0, "volume": 5e5})
    other = ext_rows.assign(ticker="AAA", close=999.0, adj_close=999.0)        # must NOT override Yahoo's AAA
    pd.concat([ext_rows, other]).to_csv(tmp / "ext" / "vendor.csv", index=False)

    prices = loader.download_prices(tickers=["AAA", "GONE"], lookback_years=5, force_refresh=True,
                                    today=pd.Timestamp("2026-10-28"))
    assert "GONE" in set(prices.index.get_level_values("Ticker"))
    assert prices.xs("AAA", level="Ticker")["adj close"].max() < 500           # Yahoo's AAA kept
    cov = json.loads((tmp / "price_coverage.json").read_text())
    assert cov["from_external"] == ["GONE"] and cov["missing"] == []


def test_external_loader_validates_columns(tmp_path):
    pd.DataFrame({"date": ["2020-01-02"], "ticker": ["X"], "close": [1.0]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        external.load_external_prices(tmp_path)


def test_external_loader_normalises_symbols_and_dedupes(tmp_path):
    df = pd.DataFrame({"Date": ["2020-01-02", "2020-01-02"], "Symbol": ["BRK.B", "BRK.B"], "open": 1.0, "high": 1.0,
                       "low": 1.0, "close": [1.0, 2.0], "volume": 1.0, "Adj_Close": [1.0, 2.0]})
    df.to_csv(tmp_path / "v.csv", index=False)
    out = external.load_external_prices(tmp_path)
    assert list(out.index.get_level_values("Ticker").unique()) == ["BRK-B"]
    assert len(out) == 1 and out["adj close"].iloc[0] == 2.0                 # last duplicate wins


def test_external_loader_strips_vendor_delisting_suffixes(tmp_path):
    df = pd.DataFrame({"date": ["2018-05-01"], "ticker": ["ALOG-201806"], "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "adj close": 1.0, "volume": 1.0})
    df.to_csv(tmp_path / "norgate.csv", index=False)
    assert list(external.load_external_prices(tmp_path).index.get_level_values("Ticker")) == ["ALOG"]


def test_no_external_files_means_none(tmp_path):
    assert external.load_external_prices(tmp_path) is None
    assert external.load_external_prices(tmp_path / "does-not-exist") is None


# ── universe filters ──────────────────────────────────────────────────────────

def _daily(ticker, start, end):
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"adj close": 10.0, "close": 10.0, "Date": days, "Ticker": ticker})


def test_a_stock_that_stopped_trading_mid_month_is_not_tradable_at_month_end():
    live = _daily("LIVE", "2020-01-01", "2020-03-31")
    dead = _daily("DEAD", "2020-01-01", "2020-03-10")                         # last bar 10 March
    prices = pd.concat([live, dead]).set_index(["Date", "Ticker"]).sort_index()
    bars = month_end_bar_dates(prices)
    me = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]), ["DEAD", "LIVE"]],
                                    names=["Date", "Ticker"])
    month_end = pd.DataFrame({"x": 1}, index=me)
    kept = drop_stale_bars(month_end, bars, max_days=7)
    assert ("2020-03-31", "DEAD") not in {(str(d.date()), t) for d, t in kept.index}
    assert len(kept) == 5                                                     # only March / DEAD removed
    assert len(drop_stale_bars(month_end, bars, max_days=40)) == 6


def test_bar_dates_ignore_rows_without_a_price():
    rows = _daily("X", "2020-01-01", "2020-01-10")
    nan_tail = _daily("X", "2020-01-13", "2020-01-31").assign(**{"adj close": np.nan, "close": np.nan})
    prices = pd.concat([rows, nan_tail]).set_index(["Date", "Ticker"]).sort_index()
    assert month_end_bar_dates(prices).iloc[0] == pd.Timestamp("2020-01-10")


def test_min_history_counts_only_the_past():
    dates = pd.date_range("2018-01-31", periods=40, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["Date", "Ticker"])
    data = pd.DataFrame({"x": 1}, index=idx)
    kept = filter_min_history(data, min_months=36)
    assert kept.index.get_level_values("Date").min() == dates[35]            # the 36th month is the first eligible
    # appending the future must not change which earlier rows are eligible
    longer = filter_min_history(pd.concat([data, pd.DataFrame({"x": 1}, index=pd.MultiIndex.from_product(
        [pd.date_range("2021-05-31", periods=12, freq="ME"), ["A"]], names=["Date", "Ticker"]))]), 36)
    assert set(kept.index) <= set(longer.index)


# ── coverage ──────────────────────────────────────────────────────────────────

def test_coverage_counts_members_with_prices():
    pit = pd.DataFrame({"tickers": [["A", "B", "C"], ["A", "B"]]},
                       index=pd.DatetimeIndex(["2020-01-01", "2020-03-01"], name="date"))
    months = pd.date_range("2020-01-31", periods=4, freq="ME", name="Date")
    adj = pd.DataFrame({"A": 1.0, "B": [1.0, np.nan, np.nan, np.nan], "C": [1.0, 1.0, np.nan, np.nan]}, index=months)
    cov = coverage_by_month(pit, adj)
    assert cov["members"].tolist() == [3, 3, 2, 2]
    assert cov["with_prices"].tolist() == [3, 2, 1, 1]
    miss = missing_member_months(pit, adj)
    assert miss["B"] == 3                                # member in four months, priced only in January
    assert "A" not in miss and "C" not in miss           # C was a member only while it had prices


# ── benchmark alignment & metrics ─────────────────────────────────────────────

def test_benchmark_is_aligned_to_the_month_the_return_was_earned(monkeypatch):
    from momentum_rebalancer.strategy import backtest as bt
    days = pd.bdate_range("2019-10-01", "2020-06-30")
    month_ret = {"2020-01": 0.10, "2020-02": -0.05, "2020-03": 0.02}
    px, last = [], 100.0
    prev_m = None
    for d in days:
        m = f"{d:%Y-%m}"
        if m != prev_m and m in month_ret:
            last *= 1 + month_ret[m]                                           # the whole month's move on its first day
        prev_m = m
        px.append(last)
    spy = pd.DataFrame({"Close": px}, index=days)
    monkeypatch.setattr(bt.yf, "download", lambda *a, **k: spy)

    earned = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"])
    b = bt._load_benchmark(earned)
    assert b.tolist() == pytest.approx([0.10, -0.05, 0.02])


def test_benchmark_failure_gives_nan_not_zero(monkeypatch):
    from momentum_rebalancer.strategy import backtest as bt
    monkeypatch.setattr(bt.yf, "download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    b = bt._load_benchmark(pd.DatetimeIndex(["2020-01-31", "2020-02-29"]))
    assert b.isna().all()


def test_metrics_skip_a_missing_benchmark_and_use_overlap_otherwise():
    from momentum_rebalancer.strategy.backtest import compute_metrics
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    r = pd.Series(0.01, index=idx)
    assert "alpha" not in compute_metrics(r, pd.Series(np.nan, index=idx))
    half = pd.Series(np.where(np.arange(24) < 12, 0.005, np.nan), index=idx)
    m = compute_metrics(r, half)
    assert m["benchmark_ann_return"] == pytest.approx(1.005 ** 12 - 1, abs=1e-3)


def test_sortino_uses_downside_deviation():
    from momentum_rebalancer.strategy.backtest import compute_metrics
    r = pd.Series([0.03, -0.02, 0.04, -0.01, 0.02, 0.01, -0.03, 0.05, 0.02, -0.01, 0.01, 0.02],
                  index=pd.date_range("2020-01-31", periods=12, freq="ME"))
    m = compute_metrics(r)
    down_dev = np.sqrt(np.mean(np.minimum(r.values, 0) ** 2)) * np.sqrt(12)
    ann = (1 + r).prod() - 1
    assert m["sortino"] == pytest.approx(round(ann / down_dev, 3), abs=1e-3)
