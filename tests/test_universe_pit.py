import pandas as pd

from momentum_rebalancer.data_cleaning.universe import apply_pit_filter
from momentum_rebalancer.data_ingestion.loader import get_pit_tickers_for_date


def _pit():
    return pd.DataFrame(
        {"tickers": [["A", "B"], ["A", "B", "C"], ["B", "C"]]},
        index=pd.DatetimeIndex(["2020-01-01", "2021-01-01", "2022-01-01"], name="date"),
    )


def test_membership_on_a_date_uses_latest_row_on_or_before():
    pit = _pit()
    assert get_pit_tickers_for_date(pit, pd.Timestamp("2021-06-30")) == ["A", "B", "C"]
    assert get_pit_tickers_for_date(pit, pd.Timestamp("2022-01-01")) == ["B", "C"]


def test_date_before_first_row_falls_back_to_earliest():
    assert get_pit_tickers_for_date(_pit(), pd.Timestamp("2010-01-31")) == ["A", "B"]


def test_apply_pit_filter_keeps_only_members_each_month():
    dates = pd.to_datetime(["2020-06-30", "2021-06-30", "2022-06-30"])
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["Date", "Ticker"])
    data = pd.DataFrame({"x": 1.0}, index=idx)
    out = apply_pit_filter(data, _pit())
    members = {d: set(out.xs(d, level="Date").index) for d in dates}
    assert members[dates[0]] == {"A", "B"}
    assert members[dates[1]] == {"A", "B", "C"}
    assert members[dates[2]] == {"B", "C"}
