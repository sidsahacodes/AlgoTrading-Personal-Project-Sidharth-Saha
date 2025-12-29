import pandas as pd
import yfinance as yf
from urllib.request import Request, urlopen


def get_last_month_end(today=None) -> pd.Timestamp:
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    return (today - pd.offsets.MonthEnd(1)).normalize()

def load_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    sp5 = pd.read_html(urlopen(req))[0]

    sp5["Symbol"] = (
        sp5["Symbol"]
        .str.replace(",", "-", regex=False)
        .str.replace(".", "-", regex=False)
    )
    return sp5["Symbol"].unique().tolist()

def load_price_data(lookback_years: int = 10) -> pd.DataFrame:
    end_date = get_last_month_end()
    start_date = end_date - pd.DateOffset(years=lookback_years)

    data = yf.download(
        tickers=load_sp500_tickers(),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    prices = (
        data
        .stack(future_stack=True)
        .rename_axis(["Date", "Ticker"])
    )

    prices.columns = prices.columns.str.lower()

    prices.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(prices.index.get_level_values("Date")).tz_localize(None),
            prices.index.get_level_values("Ticker"),
        ]
    )

    prices = prices.loc[:end_date]

    return prices.sort_index()
