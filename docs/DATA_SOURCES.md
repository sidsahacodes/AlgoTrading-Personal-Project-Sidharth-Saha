# Data Sources

Two price sources are supported, chosen by `price_source` in `configs/strategy.yaml` (or `MR_PRICE_SOURCE`):

| | `yahoo` (default) | `crsp` |
|---|---|---|
| Prices | Yahoo Finance | CRSP via WRDS |
| Membership | fja05680/sp500, maintained "(Updated)" file | CRSP's S&P 500 list |
| Removed / delisted companies | Often missing | Included |
| Delisting returns | None (an assumption is used) | In the data |
| Renamed companies | History only under the new symbol | One label for the whole life |
| How recent | To the last completed month | Limited by CRSP's update schedule (see below) |
| Best for | Live signals | Research: backtest and walk-forward |

**Recommended setup:** run research on CRSP in its own data folder and live signals on Yahoo in the default folder. They never share caches, because each folder is set by `MR_DATA_ROOT`.

## CRSP via WRDS

### Setup

```bat
pip install wrds                                   :: or: pip install -e .[crsp]
set WRDS_USERNAME=your_wrds_username               :: or put it in .env
set MR_PRICE_SOURCE=crsp
set MR_DATA_ROOT=C:\path\to\repo\data_research   :: a separate folder from the live data
mr crsp-check                                      :: verifies tables, columns and a few facts on YOUR subscription
mr build-features --force
mr cluster --force
mr coverage
mr backtest
mr walkforward
```

WRDS asks for your password on first use (the `wrds` package can store it in a `.pgpass` file). **Run `mr crsp-check` first.** The queries were written from WRDS and CRSP documentation and tested only against a fake connection; the check confirms the tables and columns exist on your account. Table and column names live in `configs/crsp.yaml`, so a difference on your subscription is a config edit, not a code change.

### What is used

| Data | Table (CIZ format) | Notes |
|---|---|---|
| Daily prices and returns | `dsf_v2` | `dlyprc`, `dlyret`, `dlyvol`, `dlyopen`, `dlyhigh`, `dlylow`, `dlydelflg` |
| S&P 500 membership | `dsp500list_v2` (or `msp500list_v2`) | `permno`, `mbrstartdt`, `mbrenddt` |
| Tickers over time | `stksecurityinfohist` | Used only to name each company |

The legacy tables (`dsf`, `msf`, `dsedelist`) are not used: CRSP no longer updates them. The first library on your account that holds a table is used (`crsp_m_*`, then `crsp_q_*`, `crsp_a_*`, `crsp`).

### How the data are used

- **One label per company for its whole life:** the last ticker CRSP recorded, so FB and META are one stock, and live trading symbols stay valid. If two companies share a label, the most recent keeps it and the other becomes `LABEL_permno`.
- **Membership** is CRSP's list expanded to one row per month-end.
- **Prices are rebuilt from returns.** `adj close` is a total-return index starting at the first price. Open, high, low and close are the raw prices times the same factor, so they are continuous through splits and dividends (`close` equals `adj close`). Volume is divided by that factor so that `close × volume` is the raw dollar volume that ranks liquidity.
- **Delisting returns.** In CIZ the delisting return is **already inside `dlyret`**, posted on a separate row with `dlydelflg = 'Y'`. It is used as is and never compounded with anything else: doing so double-counts the loss (a documented error in at least one widely used research pipeline). Only when that return is missing is `delisting_return` (default -30%, an assumption) imputed; the count is logged and recorded in `price_coverage.json`.
- **A delisting row carries a return, not a trading bar.** The month it falls in is untradable at that month-end, even if the last trade was a few days earlier, so the loss is realised once, in the return of the holding period it belongs to.

### What limits it

- **CRSP's S&P 500 list is updated annually, and the stock files quarterly or monthly** depending on your subscription. The research sample therefore ends at the last month both cover (`mr crsp-check` prints it). Setting `extend_membership: true` in `configs/crsp.yaml` lets months after the list's end reuse its last member list, up to the price end; that is not point-in-time.
- **The latest months are not in a CRSP research set.** For the newest signals use Yahoo in the live folder. Combining the two into one continuous series is not built.
- **A different index list.** CRSP's membership list is CRSP's record of S&P 500 membership, which can differ in small ways from the file the Yahoo path uses.

## Yahoo path

| Data | Source | Used for |
|---|---|---|
| S&P 500 membership history | [fja05680/sp500](https://github.com/fja05680/sp500), the file named `S&P 500 Historical Components & Changes (Updated).csv` (`pit_url` in `configs/strategy.yaml`) | Who was in the index at each date |
| Daily prices | Yahoo Finance via `yfinance`, `auto_adjust=False`, `adj close` used for returns | Features, returns |
| Optional vendor prices | Files you place in `<MR_DATA_ROOT>/external/prices/` | Names Yahoo does not carry |
| Factor returns | Kenneth French data library via `pandas-datareader` | FF5 betas (both paths) |
| SPY | Yahoo Finance | Benchmark only (both paths) |

### The membership file

- One row per change date: `date`, `tickers` (comma-separated). Share classes use dots (`BRK.B`); the loader converts them to Yahoo's form (`BRK-B`). An older snapshot of the same repository used `-YYYYMM` suffixes for removed names; the loader strips those as well.
- The file's last row is the last update, not necessarily today. `mr coverage`, `mr regime` and `mr build-features` warn when the list ends more than 120 days ago, because every later month would reuse its last member list.
- If the file cannot be downloaded the pipeline **stops**. Setting `pit_allow_fallback: true` makes it use today's members for every date, which is survivorship-biased.

### The gap: removed and delisted companies

Survivorship bias returns through the price source if a company that *was* in the index has no prices. Yahoo often lacks removed or delisted names, and a renamed company's history exists only under its new symbol while the membership list uses the symbol of the time.

```bash
mr build-features --force      # records which requested tickers returned nothing
mr coverage                    # per month: share of index members that have prices, and the worst offenders
```

### Closing the gap with vendor files

Put CSV or Parquet files in `<MR_DATA_ROOT>/external/prices/` (created automatically, git-ignored), then `mr build-features --force`. Required columns (case-insensitive): `date, ticker, open, high, low, close, adj close, volume`.

- `adj close` should be adjusted for splits **and dividends**. If it is missing, `close` is used and a warning is logged, which is only right for non-dividend payers.
- `ticker` must match the symbol the membership list used at the time. Dots and `-YYYYMM` suffixes are normalised. A renamed company needs its history filed under the **old** symbol for the old period.
- Vendor rows fill **only** tickers Yahoo returned nothing for; they never overwrite Yahoo.
- Include the final trading days of a delisted stock. A stock with no price at all in a month it is held gets the assumed `delisting_return`.

With CRSP available this is rarely needed. Other sources, if you ever do: Norgate Data (delisted US stocks and historical index constituents; paid subscription; delisted symbols carry a `-YYYYMM` suffix, which is stripped automatically). Verify availability, licensing and price yourself.

## Refreshing data

`mr build-features --force` always **re-downloads everything**. Yahoo prices are never appended, because Yahoo re-bases adjusted closes whenever a dividend or split occurs, so an old and a new download do not join up. Re-downloading can also change past numbers slightly. CRSP data are rebuilt in full each time too.
