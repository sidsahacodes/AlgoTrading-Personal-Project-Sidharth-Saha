"""CRSP source, tested against a fake WRDS connection serving CIZ-shaped tables."""
import json

import numpy as np
import pandas as pd
import pytest

from momentum_rebalancer import config as C
from momentum_rebalancer.data_ingestion import crsp, loader

CFG = C.load_crsp_settings()


# ── pure functions ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("date,expected", [
    ("2026-09-30", "2026-09-30"),
    ("2026-05-29", "2026-05-31"),      # 31 May 2026 is a Sunday: the 29th is the last trading day, May is complete
    ("2026-05-20", "2026-04-30"),      # mid-May: May is not complete
])
def test_complete_month_end(date, expected):
    assert crsp.complete_month_end(pd.Timestamp(date)) == pd.Timestamp(expected)


def test_sample_end_is_limited_by_prices_membership_and_today():
    today = pd.Timestamp("2026-09-30")
    assert crsp.sample_end(today, pd.Timestamp("2026-06-30"), pd.Timestamp("2025-12-29"), False) == pd.Timestamp("2025-12-31")
    assert crsp.sample_end(today, pd.Timestamp("2026-06-30"), pd.Timestamp("2025-12-29"), True) == pd.Timestamp("2026-06-30")
    assert crsp.sample_end(pd.Timestamp("2026-03-31"), pd.Timestamp("2026-06-30"), pd.Timestamp("2026-06-30"), False) == pd.Timestamp("2026-03-31")


def _info(rows):
    return pd.DataFrame(rows, columns=["permno", "sec_start", "sec_end", "ticker"]).astype(
        {"sec_start": "datetime64[ns]", "sec_end": "datetime64[ns]"})


def test_labels_use_the_last_ticker_so_renames_collapse_to_one_name():
    info = _info([(1, "2012-01-01", "2022-06-08", "FB"), (1, "2022-06-09", "2026-09-30", "META")])
    assert crsp.assign_labels(info, [1]) == {1: "META"}


def test_label_collisions_keep_the_plain_label_for_the_most_recent_company():
    info = _info([(10, "2010-01-01", "2014-06-30", "AAL"), (20, "2015-01-01", "2026-09-30", "AAL")])
    assert crsp.assign_labels(info, [10, 20]) == {20: "AAL", 10: "AAL_10"}


def test_share_class_tickers_are_normalised_and_missing_tickers_get_a_placeholder():
    info = _info([(1, "2010-01-01", "2026-01-01", "BRK.B"), (2, "2010-01-01", "2026-01-01", "  ")])
    labels = crsp.assign_labels(info, [1, 2, 3])
    assert labels[1] == "BRK-B" and labels[2] == "P2" and labels[3] == "P3"


def test_membership_is_expanded_to_month_ends_with_labels():
    members = pd.DataFrame({"permno": [1, 2], "mbr_start": pd.to_datetime(["2020-01-01", "2020-03-15"]),
                            "mbr_end": pd.to_datetime(["2020-12-31", "2020-04-30"])})
    me = pd.date_range("2020-02-29", "2020-05-31", freq="ME")
    pit = crsp.membership_to_pit(members, {1: "AAA", 2: "BBB"}, me)
    assert pit["tickers"].tolist() == [["AAA"], ["AAA", "BBB"], ["AAA", "BBB"], ["AAA"]]


def _daily(rows):
    return pd.DataFrame(rows, columns=["permno", "dt", "delflg", "price", "ret", "vol", "px_open", "px_high", "px_low"])


def _days(n, start="2020-01-02"):
    return pd.bdate_range(start, periods=n)


def test_adjusted_prices_are_continuous_through_a_split_and_dollar_volume_stays_raw():
    d = _days(6)
    price = [100, 102, 104, 52, 53, 54]                 # 2:1 split on day 4: raw price halves
    ret = [np.nan, 0.02, 0.0196078, 0.0, 0.0192308, 0.0188679]    # CRSP returns are split-adjusted
    vol = [1000, 1000, 1000, 2000, 2000, 2000]          # raw shares double
    rows = [(1, d[i], "N", price[i], ret[i], vol[i], price[i], price[i] * 1.01, price[i] * 0.99) for i in range(6)]
    out, _ = crsp.daily_to_prices(_daily(rows), {1: "AAA"}, -0.3)
    a = out.xs("AAA", level="Ticker")
    assert a["close"].pct_change().dropna().round(4).tolist() == pytest.approx([0.02, 0.0196, 0.0, 0.0192, 0.0189], abs=2e-4)
    assert (a["close"] == a["adj close"]).all()
    raw_dollar = np.array(price) * np.array(vol)
    assert (a["close"] * a["volume"]).tolist() == pytest.approx(raw_dollar.tolist())
    assert (a["high"] >= a["close"]).all() and (a["low"] <= a["close"]).all()


def test_the_total_return_index_starts_at_the_first_price_even_if_the_first_return_is_known():
    d = _days(3)
    rows = [(1, d[0], "N", 20.0, 0.05, 100, 20, 20, 20), (1, d[1], "N", 21.0, 0.05, 100, 21, 21, 21),
            (1, d[2], "N", 22.05, 0.05, 100, 22, 22, 22)]       # the first return refers to a day before the window
    out, _ = crsp.daily_to_prices(_daily(rows), {1: "A"}, -0.3)
    a = out.xs("A", level="Ticker")["adj close"]
    assert a.iloc[0] == pytest.approx(20.0) and a.iloc[1] / a.iloc[0] == pytest.approx(1.05)


def test_delisting_return_is_used_as_is_and_never_compounded_twice():
    d = _days(5)
    rows = [(7, d[0], "N", 10.0, np.nan, 100, 10, 10.2, 9.8),
            (7, d[1], "N", 9.0, -0.10, 100, 10, 10, 9, ),
            (7, d[2], "N", 8.1, -0.10, 100, 9, 9, 8),
            (7, d[3], "Y", 3.24, -0.60, np.nan, np.nan, np.nan, np.nan)]    # delisting row: return already includes the loss
    out, stats = crsp.daily_to_prices(_daily(rows), {7: "DEAD"}, -0.30)
    a = out.xs("DEAD", level="Ticker")["adj close"]
    assert a.iloc[-1] / a.iloc[0] == pytest.approx(0.9 * 0.9 * 0.4)             # -60% applied once
    assert stats == {"delisting_rows": 1, "missing_delisting_returns_imputed": 0}
    last = out.xs("DEAD", level="Ticker").iloc[-1]
    assert np.isnan(last["open"]) and np.isnan(last["volume"])                  # no bar, but the return is kept


def test_missing_delisting_return_gets_the_assumption_and_is_counted():
    d = _days(3)
    rows = [(7, d[0], "N", 10.0, np.nan, 100, 10, 10, 10), (7, d[1], "N", 10.0, 0.0, 100, 10, 10, 10),
            (7, d[2], "Y", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)]
    out, stats = crsp.daily_to_prices(_daily(rows), {7: "DEAD"}, -0.30)
    a = out.xs("DEAD", level="Ticker")["adj close"]
    assert a.iloc[-1] / a.iloc[-2] == pytest.approx(0.7)
    assert stats["missing_delisting_returns_imputed"] == 1


def test_no_trade_days_become_flat_bars_and_unpriced_rows_are_dropped():
    d = _days(4)
    rows = [(1, d[0], "N", 10.0, np.nan, 100, 10, 10.5, 9.5), (1, d[1], "N", 10.2, 0.02, 100, np.nan, np.nan, np.nan),
            (1, d[2], "N", np.nan, np.nan, 0, np.nan, np.nan, np.nan), (1, d[3], "N", 10.4, 0.0196, 100, 10.2, 10.5, 10.1)]
    out, _ = crsp.daily_to_prices(_daily(rows), {1: "A"}, -0.3)
    a = out.xs("A", level="Ticker")
    assert len(a) == 3                                                         # the unpriced day is gone
    assert a.iloc[1]["open"] == a.iloc[1]["high"] == a.iloc[1]["low"] == a.iloc[1]["close"]


def test_negative_legacy_prices_are_treated_as_bid_ask_averages():
    d = _days(2)
    rows = [(1, d[0], "N", -10.0, np.nan, 100, np.nan, np.nan, np.nan), (1, d[1], "N", -10.5, 0.05, 100, np.nan, np.nan, np.nan)]
    out, _ = crsp.daily_to_prices(_daily(rows), {1: "A"}, -0.3)
    assert (out["close"] > 0).all()


# ── a fake WRDS connection ────────────────────────────────────────────────────

class FakeDB:
    """Serves CIZ-shaped tables. SQL is routed by its table name; the real queries are not run."""
    COLS = {"dsf_v2": ["permno", "dlycaldt", "dlydelflg", "dlyprc", "dlyret", "dlyvol", "dlyopen", "dlyhigh", "dlylow"],
            "dsp500list_v2": ["permno", "mbrstartdt", "mbrenddt", "indno", "mbrflg", "indfam"],
            "stksecurityinfohist": ["permno", "secinfostartdt", "secinfoenddt", "ticker"]}

    def __init__(self, members, info, daily, drop_column=None):
        self.members, self.info, self.daily, self.drop = members, info, daily, drop_column
        self.queries = []

    def list_libraries(self):
        return ["crsp_m_stock", "crsp_m_indexes", "comp"]

    def list_tables(self, library):
        return {"crsp_m_stock": ["dsf_v2", "stksecurityinfohist"], "crsp_m_indexes": ["dsp500list_v2"]}.get(library, [])

    def describe_table(self, library, table):
        cols = [c for c in self.COLS[table] if c != self.drop]
        return pd.DataFrame({"name": cols})

    def raw_sql(self, sql, params=None, date_cols=None):
        s = " ".join(sql.split()).lower()
        self.queries.append(s)
        if "count(*)" in s:
            dl = self.daily[(self.daily["delflg"] == "Y") & (self.daily["dt"] >= "2015-01-01")]
            return pd.DataFrame({"n": [len(dl)], "n_null": [int(dl["ret"].isna().sum())]})
        if "max(" in s:
            return pd.DataFrame({"d": [self.daily["dt"].max()]})
        if "stksecurityinfohist" in s:
            return self.info[self.info["permno"].isin(params["p"])]
        if "dsp500list_v2" in s:
            return self.members
        if "limit 3" in s:
            a = self.daily[self.daily["permno"] == 14593].sort_values("dt", ascending=False).head(3)
            return a[["dt", "price", "ret"]]
        if "dsf_v2" in s:
            d = self.daily
            m = d["permno"].isin(params["p"]) & (d["dt"] >= params["s"]) & (d["dt"] <= params["e"])
            return d[m]
        raise AssertionError(f"unexpected SQL: {s}")


def _world(n_tickers=45, seed=4, end="2026-06-30"):
    """Members 1..n active throughout, 14593 (Apple) too, plus a delisted company and a ticker collision."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2011-01-03", end)
    permnos = list(range(1, n_tickers)) + [14593, 900, 901]
    members, info, rows = [], [], []
    delist_last_trade = pd.Timestamp("2020-03-16")
    for p in permnos:
        start = pd.Timestamp("2011-01-03")
        stop = pd.Timestamp(end)
        ticker = {14593: "AAPL", 900: "DEAD", 901: "T01"}.get(p, f"T{p:02d}")      # 901 collides with permno 1's label T01
        if p == 900:
            stop = delist_last_trade
        if p == 901:
            stop = pd.Timestamp("2014-12-31")
        members.append((p, start, stop if p in (900, 901) else pd.Timestamp("2025-12-29"), 1, "C", 1))
        info.append((p, start, stop, ticker))
        d = days[(days >= start) & (days <= stop)]
        r = rng.normal(0.0004, 0.016, len(d)); r[0] = np.nan
        price = 40 * np.cumprod(1 + np.nan_to_num(r))
        vol = rng.integers(1_000_000, 4_000_000, len(d)).astype(float) * (1 + p / 20)
        for i, dt in enumerate(d):
            rows.append((p, dt, "N", price[i], r[i], vol[i], price[i] * 0.999, price[i] * 1.01, price[i] * 0.99))
        if p == 900:                                          # the delisting row, the day after the last trade, -50%
            rows.append((p, delist_last_trade + pd.Timedelta(days=1), "Y", price[-1] * 0.5, -0.50, np.nan, np.nan, np.nan, np.nan))
    members = pd.DataFrame(members, columns=["permno", "mbr_start", "mbr_end", "indno", "mbrflg", "indfam"])
    info = pd.DataFrame(info, columns=["permno", "sec_start", "sec_end", "ticker"])
    daily = pd.DataFrame(rows, columns=["permno", "dt", "delflg", "price", "ret", "vol", "px_open", "px_high", "px_low"])
    return members, info, daily


@pytest.fixture(scope="module")
def world():
    return _world()


# ── bundle ────────────────────────────────────────────────────────────────────

def test_bundle_labels_membership_prices_and_sample_end(world):
    members, info, daily = world
    db = FakeDB(members, info, daily)
    b = crsp.build_bundle(CFG, pd.Timestamp("2026-09-30"), 15, -0.30, db=db)

    assert b.labels[1] == "T01" or b.labels[901] == "T01_901"                    # collision resolved
    assert b.labels[901].startswith("T01_") and b.labels[1] == "T01"
    assert b.labels[14593] == "AAPL"
    # membership list ends 2025-12-29 and extend_membership is false -> sample ends 2025-12-31 even though prices run on
    assert b.info["end"] == pd.Timestamp("2025-12-31")
    assert b.pit.index.max() == pd.Timestamp("2025-12-31")
    assert "DEAD" in b.pit.loc["2020-02-29", "tickers"] and "DEAD" not in b.pit.loc["2020-04-30", "tickers"]
    assert b.info["delisting_rows"] == 1 and b.info["missing"] == []
    assert b.prices.index.get_level_values("Date").max() <= pd.Timestamp("2025-12-31")
    # queries are chunked and parameterised, never string-built with data
    assert any("in %(p)s" in q for q in db.queries)


def test_extending_membership_lets_the_sample_run_to_the_price_end(world):
    members, info, daily = world
    cfg = {**CFG, "extend_membership": True}
    b = crsp.build_bundle(cfg, pd.Timestamp("2026-09-30"), 15, -0.30, db=FakeDB(members, info, daily))
    assert b.info["end"] == pd.Timestamp("2026-06-30")


# ── loader integration ────────────────────────────────────────────────────────

@pytest.fixture
def crsp_loader(tmp_path, monkeypatch, world):
    members, info, daily = world
    monkeypatch.setattr(loader, "PRICE_SOURCE", "crsp")
    monkeypatch.setattr(loader, "SETTINGS", {**C.SETTINGS, "price_source": "crsp"})
    monkeypatch.setattr(loader, "PIT_MEMBERS_PATH", tmp_path / "pit_members.parquet")
    monkeypatch.setattr(loader, "RAW_PRICES_PATH", tmp_path / "prices_raw.parquet")
    monkeypatch.setattr(loader, "COVERAGE_PATH", tmp_path / "price_coverage.json")
    monkeypatch.setattr(crsp, "connect", lambda cfg: FakeDB(members, info, daily))
    crsp.clear_bundle()
    yield tmp_path
    crsp.clear_bundle()


def test_loader_uses_crsp_for_membership_prices_and_coverage(crsp_loader):
    tmp = crsp_loader
    pit = loader.load_pit_members(force_refresh=True)
    prices = loader.download_prices(force_refresh=True, today=pd.Timestamp("2026-10-28"))
    assert "DEAD" in set(prices.index.get_level_values("Ticker"))                 # a delisted company is present
    cov = json.loads((tmp / "price_coverage.json").read_text())
    assert cov["source"] == "crsp" and cov["missing"] == [] and cov["delisting_rows"] == 1
    meta = json.loads((tmp / "pit_members.meta.json").read_text())
    assert meta["params"]["price_source"] == "crsp"
    assert len(pit) > 100


def test_switching_price_source_makes_the_cache_look_stale(crsp_loader, monkeypatch):
    from momentum_rebalancer.storage.fingerprint import diff_params
    loader.load_pit_members(force_refresh=True)
    other = {**C.SETTINGS, "price_source": "yahoo"}
    assert any("price_source" in w for w in diff_params(loader.PIT_MEMBERS_PATH, other, ["pit_url", "price_source"]))


def test_full_feature_pipeline_on_crsp_data_keeps_delisting_losses_and_drops_untradable_stocks(crsp_loader, monkeypatch):
    from momentum_rebalancer.features import pipeline, returns
    rng = np.random.default_rng(5)
    ff_idx = pd.period_range("2010-01", "2026-06", freq="M")
    ff = pd.DataFrame(rng.normal(0, 2, (len(ff_idx), 6)), index=ff_idx, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    monkeypatch.setattr(returns.web, "DataReader", lambda *a, **k: [ff])
    monkeypatch.setattr(pipeline, "MODELING_DATA_PATH", crsp_loader / "modeling_data.parquet")
    monkeypatch.setattr(pipeline, "MONTHLY_RETURNS_PATH", crsp_loader / "monthly_returns.parquet")
    monkeypatch.setattr(loader, "last_completed_month_end", lambda today=None: pd.Timestamp("2026-09-30"))

    data = pipeline.build_modeling_data(force_refresh=True)
    assert list(data.columns) == C.FEATURE_COLS

    mret = pd.read_parquet(crsp_loader / "monthly_returns.parquet")
    # DEAD: last trade 2020-03-16, delisting row 2020-03-17 with -50%. March's return must contain it.
    px = crsp_loader / "prices_raw.parquet"
    raw = pd.read_parquet(px).xs("DEAD", level="Ticker")["adj close"]
    expected = raw.loc["2020-03-17"] / raw.loc[:"2020-02-29"].iloc[-1] - 1
    assert mret.loc["2020-03-31", "DEAD"] == pytest.approx(expected)
    assert mret.loc["2020-03-31", "DEAD"] < -0.45
    assert mret["DEAD"].loc["2020-04-30":].isna().all()                              # nothing after it

    # at the March month-end DEAD's last bar is >7 days old: not tradable, not in the modelling data
    dates = data.xs("DEAD", level="Ticker").index if "DEAD" in data.index.get_level_values("Ticker") else []
    assert pd.Timestamp("2020-03-31") not in list(dates)


def test_a_stock_delisted_just_before_month_end_is_not_tradable_then():
    """Last trade 27 March, delisting row 28 March: its last bar is within 7 days of 31 March,
    but the company is gone, so it must not be selectable (and must not be charged the loss twice)."""
    from momentum_rebalancer.data_cleaning.universe import build_monthly_universe
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2017-01-02", "2020-06-30")
    rows = []
    for permno, stop in [(1, pd.Timestamp("2020-06-30")), (2, pd.Timestamp("2020-03-27"))]:
        d = days[days <= stop]
        r = rng.normal(0.0004, 0.015, len(d)); r[0] = np.nan
        px = 30 * np.cumprod(1 + np.nan_to_num(r))
        rows += [(permno, dt, "N", px[i], r[i], 2e6, px[i], px[i] * 1.01, px[i] * 0.99) for i, dt in enumerate(d)]
    rows.append((2, pd.Timestamp("2020-03-28"), "Y", 1.0, -0.40, np.nan, np.nan, np.nan, np.nan))
    prices, _ = crsp.daily_to_prices(_daily(rows), {1: "LIVE", 2: "GONE"}, -0.30)
    pit = pd.DataFrame({"tickers": [["LIVE", "GONE"]]}, index=pd.DatetimeIndex(["2017-01-01"], name="date"))
    uni = build_monthly_universe(prices, pit=pit)
    names_march = set(uni.xs(pd.Timestamp("2020-03-31"), level="Date").index)
    names_feb = set(uni.xs(pd.Timestamp("2020-02-29"), level="Date").index)
    assert "GONE" in names_feb and "GONE" not in names_march and "LIVE" in names_march


# ── crsp-check ────────────────────────────────────────────────────────────────

def test_check_passes_on_a_healthy_subscription(world):
    results = crsp.run_checks(FakeDB(*world), CFG, today_end=pd.Timestamp("2026-09-30"), expected_members=(30, 60))
    failed = [r for r in results if not r.ok]
    assert not failed, failed
    names = " ".join(r.name for r in results)
    assert "delisting rows" in names and "data coverage" in names


def test_check_reports_a_missing_column(world):
    results = crsp.run_checks(FakeDB(*world, drop_column="dlydelflg"), CFG, today_end=pd.Timestamp("2026-09-30"))
    bad = [r for r in results if not r.ok]
    assert bad and "dlydelflg" in bad[0].detail


def test_check_reports_a_missing_table(world):
    class NoIndexes(FakeDB):
        def list_tables(self, library):
            return [t for t in super().list_tables(library) if t != "dsp500list_v2"]
    results = crsp.run_checks(NoIndexes(*world), CFG)
    assert any(not r.ok and "membership" in r.name for r in results)


def test_cli_check_exits_nonzero_on_failure(world, monkeypatch, capsys):
    from momentum_rebalancer import cli
    monkeypatch.setattr(crsp, "connect", lambda cfg: FakeDB(*world, drop_column="dlyret"))
    with pytest.raises(SystemExit) as e:
        cli.main(["crsp-check"])
    assert e.value.code == 1 and "FAIL" in capsys.readouterr().out


def test_settings_loader_rejects_a_broken_crsp_yaml(tmp_path):
    import shutil, yaml
    d = tmp_path / "configs"; shutil.copytree(C.CONFIG_DIR, d)
    cfg = yaml.safe_load((d / "crsp.yaml").read_text()); del cfg["columns"]["daily"]["ret"]
    (d / "crsp.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="columns.daily.ret"):
        C.load_crsp_settings(d)
