# `broker/` – execution backends

`broker/__init__.py` picks a backend from `config.BROKER` and exposes three functions either way:

| Function | Returns |
|---|---|
| `broker.get_holdings()` | `{ticker: weight}` of current positions |
| `broker.get_value()` | Portfolio value ($) |
| `broker.execute(trades)` | List of fill records |

| Backend | File | How it works |
|---|---|---|
| `paper` (default) | `paper.py` | JSON ledger at `data/store/paper_ledger.json`; fills at the latest yfinance close; caps sells at shares held |
| `ibkr` | `ibkr.py` | Connects to TWS through `ib_insync`; reads NetLiquidation; prices from TWS with yfinance fallback; sends **market orders** |

Select with `broker:` in `configs/strategy.yaml` or `MR_BROKER=ibkr`.

## IBKR setup

1. Open TWS and log into the **paper** account.
2. *File → Global Configuration → API → Settings*: enable socket clients, port **7497** (paper) / 7496 (live), untick Read-Only API.
3. `MR_BROKER=ibkr`.

Start with `paper` until the trade list looks right for several months.

## Safety notes ⚠

- **Exit orders are sized from weight × NAV ÷ price, not the actual position**, so a full exit can leave a residual share or oversell by one and open a short (C1, open). Fix before real money.
- Market orders; fill records show the *estimated* price and the order status ~1 s after sending, not confirmed fills (C3).
- A fresh random client id is used on every connection.
- Holdings use `abs(marketValue)`; the strategy assumes long-only (C5).
- `ib_insync` is archived upstream; `ib_async` is the maintained successor (B7).

**In stress:** spreads widen and gaps happen. Limit orders, partial rebalances and checking fallback prices are *proposed*, not implemented. See [`../../../docs/REGIME_PLAYBOOK.md`](../../../docs/REGIME_PLAYBOOK.md) (R3).

`mr broker-check` reports whether TWS is reachable (`ibkr.is_connected()`); `mr trades` and `mr rebalance` run the same check first and stop with a clear message if it fails.

**Testing:** `tests/test_paper_broker.py` covers the paper ledger (round trip, oversell cap, cash cap). The IBKR backend is **not** covered by tests; it needs a running TWS.
