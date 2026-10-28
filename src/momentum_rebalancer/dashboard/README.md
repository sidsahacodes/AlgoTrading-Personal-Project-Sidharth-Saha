# `dashboard/` – local web UI

Flask app (`app.py`) serving one page (`templates/index.html`).

```bash
mr dashboard      # http://127.0.0.1:5001
```

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | UI |
| `/api/status` | GET | Broker, **active regime**, optimiser, cluster method, file presence, **cache-vs-config warnings** |
| `/api/portfolio` | GET | Current vs. target weights, portfolio value |
| `/api/trades` | GET | Pending trades (dry run) |
| `/api/rebalance` | POST | **Sends orders** to the configured broker |
| `/api/refresh` | POST | Rebuilds features and clusters (slow: re-downloads prices) |
| `/api/backtest` | GET | Equity curve, drawdown, monthly returns, period and cluster stats, cost drag and turnover; **cached in memory**, restart to recompute |

## Security ⚠

- **POST requests must send `Content-Type: application/json` and a same-origin `Origin` header** (when present). A web page in your browser cannot do that cross-site without a CORS preflight, which the app never approves. The page sends JSON for its two POST calls. Covered by `tests/test_dashboard_guard.py`.
- There is **no authentication**; any local process can call the API. Keep `dashboard_host: 127.0.0.1`.
- The dashboard does not show a non-baseline regime warning before `/api/rebalance`; use the CLI for regime presets.

## Notes

- The backtest tab runs the same engine as `mr backtest`: net of costs, returns labelled by the month earned. Per-cluster stats group by arbitrary cluster ids and are not meaningful (A2).
- Values that cannot be shown (no SPY, an infinite ratio) are returned as `null`, never `NaN`/`Infinity`, which browsers cannot parse (`tests/test_end_to_end.py`).
- After changing config or re-clustering, restart the dashboard so the backtest cache refreshes.
