# Regime & Stress Playbook

How to recognise a regime change and which parameters to adjust. Written for whoever runs the monthly rebalance.

> **Read this first**
> 1. **Regime *detection* is manual.** `mr regime` computes indicators and flags; it never changes parameters or sends orders. A person decides.
> 2. **The presets in `configs/regimes.yaml` are untested hypotheses.** The numbers are illustrative. Nothing here has been validated by a backtest.
> 3. **The backtest applies a preset to a whole window; it does not switch it on by a rule.** It does model HRP weights, the HRP window, the exposure scalar, the trade band, the size cap and costs, so `scripts/stress_windows.py --regime crash` shows what that preset would have done through, say, 2020. It cannot tell you whether you would have recognised the regime in time. `universe_size` and `ff_beta_window` also need the L2 rebuild first.
> 4. The sample holds only a few stress episodes (2020, 2022). Tuning to them invites overfitting; prefer few, simple, well-motivated changes, one at a time.

---

## 1. What the strategy is exposed to

| Exposure | Why it matters | Where it shows up |
|---|---|---|
| **Momentum crashes** | Winners can reverse violently in sharp rebounds after sell-offs (documented in the momentum literature, e.g. Daniel & Moskowitz, "Momentum Crashes", 2016). Long-only dampens but does not remove this | Strategy lags SPY during a rebound |
| **Correlation spike** | In a crash, cross-sectional differences collapse; HRP diversification and the cluster structure weaken | Holdings' pairwise correlation jumps; drawdown ≈ SPY |
| **Slow-moving estimates** | HRP uses all monthly history up to the decision date by default (up to `lookback_years`); FF betas use 24 months | Weights look "calm-market" in a stressed market |
| **Cluster instability** | Clusters are refit monthly; labels are arbitrary and membership can reshuffle | Turnover spikes |
| **Execution** | Market orders into wide spreads; fallback prices can be stale | Fills differ from sizing price |
| **Data failure** | yfinance gaps/outages, delisted tickers missing | Few tickers; clusters smaller than `top_n_per_cluster` |

---

## 2. Monthly indicators: `mr regime`

```bash
mr regime              # indicators + data/cluster health + cache-vs-config warnings
```

| Indicator | What it is | Default flag (in `strategy.yaml`) | Implemented |
|---|---|---|---|
| SPY drawdown | Latest close vs trailing 12-month high | caution ≤ -10%, stress ≤ -20% | Yes (`yfinance`) |
| VIX | Latest `^VIX` close | caution ≥ 25, stress ≥ 30 | Yes |
| Holdings correlation shift | Avg pairwise corr of selected names, last 12 months **minus** full history | stress if jump ≥ 0.15 | Yes |
| Dispersion | Cross-sectional std of `return_3m`, with percentile vs history | info only | Yes |
| Cluster health | Wrong cluster count, clusters smaller than `top_n`, one cluster holding > 50% of stocks, HDBSCAN noise share | warnings | Yes |
| Data health | Ticker count in latest month, NaNs, staleness | warnings | Yes |
| Cache vs config | Caches built with different parameters from the active ones | warnings | Yes |
| Turnover | Trade count and weight changes from `mr trades` | none | Manual |
| Rolling strategy vs SPY | Dashboard backtest tab / `run_backtest()` | none | Manual |

**All thresholds are illustrative.** Calibrate them against your own history before trusting a flag.

---

## 3. Regimes and what to consider

Presets can be applied with `mr --regime NAME <command>` or `MR_REGIME=NAME`. Each preset is a starting point to adapt, not a recommendation. `Re-run` follows [`PARAMETERS.md`](PARAMETERS.md).

### R1. Normal (`baseline`)
Defaults. No action beyond the monthly routine.

### R2. Volatile sell-off building (`stress`)
*Signs:* VIX elevated, SPY drawdown 10-20%.

| Key | Baseline → preset | Why it might help | Cost / risk | Re-run |
|---|---|---|---|---|
| `universe_size` | 150 → 100 | More liquid names, cheaper to trade | Fewer candidates; clusters may fall under `top_n` | L2 |
| `hrp_lookback_months` | all → 48 | Weights reflect current vol/correlation | Noisier. **With 20 assets, fewer than ~20 monthly points makes the covariance singular.** HRP still runs (no inversion) but estimates are unreliable; stay ≥ 36 | none |
| `min_trade_pct` | 0 → 0.005 | Less churn when signals flip | Weights drift | none |

### R3. Crash / liquidity event, correlations → 1 (`crash`)
| Key | Baseline → preset | Why | Cost / risk |
|---|---|---|---|
| `exposure_scalar` | 1.0 → 0.6 | Diversification across clusters isn't protecting you; cut gross risk | Misses the rebound; **you need a re-entry rule** |
| `hrp_lookback_months` | all → 48 | As above | As above |
| `min_trade_pct` | 0 → 0.01 | Trade less at bad prices | Drift |

Also manual: check data first (outages look like signals); consider partial rebalances. Order type is still `MarketOrder` in `broker/ibkr.py` (limit orders are not implemented).

### R4. Sharp rebound after a drawdown (`rebound`)
3-month winners are typically defensive names while losers rebound hardest.

| Key | Baseline → preset | Why | Cost / risk |
|---|---|---|---|
| `momentum_feature` | `return_3m` → `return_6m` | Slower signal, fewer flips | Still includes recent winners; not a crash cure |
| `top_n_per_cluster` / `portfolio_size` | 4 / 20 → 5 / 25 | More names, less single-stock risk | Dilutes the signal |
| `exposure_scalar` | 1.0 → 0.75 | The literature finds momentum crashes concentrate when markets rebound after falling | Whipsaw; needs a rule |

### R5. Leadership rotation / factor regime shift (`rotation`)
| Key | Baseline → preset | Why | Cost / risk | Re-run |
|---|---|---|---|---|
| `ff_beta_window` | 24 → 36 | More stable betas. **Shortening is riskier:** 24 points for 6 coefficients is already thin | Slower to adapt | L2 |
| `n_clusters` / `top_n_per_cluster` / `portfolio_size` | 5/4/20 → 6/3/18 | Different peer-group granularity | Smaller groups | L1 |

---

## 4. Decision guide

```
Is data healthy?  ──no──► fix data first; do not trade on it
        │yes
Is cluster structure sane (sizes ≥ top_n, none > 50%)?
        │no ──► re-cluster with --force; consider n_clusters / method
        │yes
Do `mr regime` flags show stress?
        │no  ──► baseline
        │yes
   Falling market ──► stress / crash
   Rebounding     ──► rebound
   Rotation only  ──► rotation
```

Then: preview with `mr --regime NAME trades`, rebalance with `mr --regime NAME rebalance` (you must type the regime name to confirm), and **log it** in `research_logs/`.

---

## 5. Testing a change before relying on it

1. State the hypothesis ("a 48-month HRP window reduces drawdown in sell-offs").
2. Change one thing (a preset, or one key).
3. Re-run what [`PARAMETERS.md`](PARAMETERS.md) says (`universe_size` and `ff_beta_window` need L2, `n_clusters` needs L1; everything else is immediate).
4. Compare named stress windows, not just the full sample:

```bash
python scripts/stress_windows.py                                   # baseline
python scripts/stress_windows.py --regime crash                    # a preset applied through each window
python scripts/stress_windows.py --top-n 5 --momentum return_6m    # ad hoc
python scripts/stress_windows.py --cost-bps 20                     # same, with higher costs
```

   Windows live in `configs/stress_windows.yaml`. Use cumulative return and max drawdown for short windows; annualised numbers over ~10 months mean little. Returns are net of `cost_bps`.
5. Check the change survives costs and the assumptions in [`VALIDATION.md`](VALIDATION.md): if it only helps at 0 bps, or only when `delisting_return` is generous, it is not a real improvement.
6. For choices that are in the walk-forward grid (momentum lookback, names per cluster, number of clusters), look at `mr walkforward`: a setting that wins on the validation block and then loses on the test block is overfitting.
7. Prefer robust changes: if the result flips sign when you nudge a parameter, don't adopt it.
8. Log it (`research_logs/TEMPLATE.md`).

---

## 6. Still to build (priority order)

1. **A rule-based exposure overlay** (e.g. scale exposure when SPY drawdown and VIX cross thresholds) inside the simulator, so the *rule* can be evaluated, not just a static scalar.
2. **Per-month feature scaling and a silhouette monitor** for choosing/sanity-checking `n_clusters` (limitation A4).
3. **Limit orders, fill confirmation and position-based exit sizing** in `broker/ibkr.py` (limitations C1, C3).
4. **Calibration of the monitor thresholds** and of any preset against history.
