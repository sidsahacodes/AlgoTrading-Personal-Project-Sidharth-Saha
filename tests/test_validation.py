import pandas as pd

from momentum_rebalancer.validation.checks import check_clusters, check_modeling_data, check_selection


def _frame(n_tickers=60, latest="2026-09-30"):
    dates = pd.to_datetime(["2026-07-31", "2026-08-31", latest])
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n_tickers)]], names=["Date", "Ticker"])
    return pd.DataFrame({"a": 1.0, "b": 2.0}, index=idx)


def test_healthy_modeling_data():
    assert check_modeling_data(_frame(), ["a", "b"], today="2026-10-10") == []


def test_missing_columns_is_error():
    issues = check_modeling_data(_frame(), ["a", "b", "c"], today="2026-10-10")
    assert any(i.level == "error" and "missing" in i.msg for i in issues)


def test_few_tickers_and_stale_data_warn():
    issues = check_modeling_data(_frame(n_tickers=10), ["a"], min_tickers=50, today="2026-12-31")
    msgs = " ".join(i.msg for i in issues)
    assert "only 10 tickers" in msgs and "days old" in msgs


def test_nan_in_latest_month_warns():
    df = _frame()
    df.iloc[-1, 0] = float("nan")
    assert any("NaN" in i.msg for i in check_modeling_data(df, ["a"], today="2026-10-10"))


def test_empty_is_error():
    assert check_modeling_data(pd.DataFrame(), ["a"])[0].level == "error"


def _clusters(sizes):
    rows = [{"cluster": c} for c, n in sizes.items() for _ in range(n)]
    return pd.DataFrame(rows)


def test_healthy_clusters():
    assert check_clusters(_clusters({0: 30, 1: 30, 2: 30, 3: 30, 4: 30}), 4, 5, "kmeans") == []


def test_small_cluster_and_wrong_count_warn():
    msgs = " ".join(i.msg for i in check_clusters(_clusters({0: 100, 1: 2, 2: 40}), 4, 5, "kmeans"))
    assert "smaller than top_n" in msgs and "expected 5" in msgs and "largest cluster" in msgs


def test_hdbscan_noise_warns():
    msgs = " ".join(i.msg for i in check_clusters(_clusters({-1: 60, 0: 20, 1: 20}), 4, 5, "hdbscan"))
    assert "noise" in msgs


def test_selection_check():
    assert check_selection(pd.DataFrame(), 20)[0].level == "error"
    assert check_selection(pd.DataFrame({"x": range(15)}), 20)[0].level == "warn"
    assert check_selection(pd.DataFrame({"x": range(20)}), 20) == []
