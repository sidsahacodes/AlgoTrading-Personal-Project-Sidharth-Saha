# Cluster-Momentum Rebalancer

Monthly, long-only equity strategy on the S&P 500. Each month it groups stocks into peer clusters using technical, momentum and Fama-French-beta features, buys the strongest recent performers inside each cluster, weights them with Hierarchical Risk Parity (HRP), and rebalances through Interactive Brokers or a local paper ledger. Features are point-in-time, the backtest is net of costs and follows the live path, and hyperparameters are evaluated out of sample with a walk-forward. Parameters, and the changes to make when the market regime shifts, live in versioned config files, with a regime report and a playbook for the monthly review.

> **Status:** research / paper-trading project. Before trusting a backtest number or sending real orders, read [`docs/VALIDATION.md`](docs/VALIDATION.md) and [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md). In particular: results depend on assumed costs and delisting returns (A1, A3), on whether Yahoo has prices for removed companies (run `mr coverage`), no real-data run is recorded here (A6), IBKR exit orders are sized from weight rather than the actual position (C1), and every regime preset is an untested hypothesis.

## External Resources

| Resource | Link | Purpose |
|---|---|---|
| S&P 500 point-in-time constituents | [fja05680/sp500](https://github.com/fja05680/sp500), the maintained "(Updated)" file | Which tickers were in the index at each date |
| Prices (live) | [yfinance](https://github.com/ranaroussi/yfinance) | Daily OHLCV, SPY benchmark, `^VIX` for the regime report |
| Prices and membership (research) | CRSP via WRDS (`pip install wrds`) | Delisted companies, delisting returns, CRSP's S&P 500 list; see `docs/DATA_SOURCES.md` |
| Factor data | [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), via `pandas-datareader` | Fama-French 5-factor returns for the rolling betas |
| Broker API | [ib_insync](https://github.com/erdewit/ib_insync) (archived) / [ib_async](https://github.com/ib-api-reloaded/ib_async) (maintained successor) | TWS connection; see limitation B7 |
| HRP | López de Prado, "Building Diversified Portfolios that Outperform Out of Sample", *Journal of Portfolio Management*, 2016 | Weighting method |
| Momentum crashes | Daniel & Moskowitz, "Momentum Crashes", *Journal of Financial Economics*, 2016 | Why the regime playbook has a rebound case |

## Methods

| Module | Description |
|---|---|
| `data_ingestion.loader` | Downloads the PIT S&P 500 membership list and daily prices; records price coverage; caches parquet |
| `data_ingestion.crsp` | CRSP via WRDS as the research source: membership, prices rebuilt from returns, delisting returns, `mr crsp-check` |
| `data_ingestion.external` | Optional vendor price files for removed/delisted names Yahoo lacks |
| `data_cleaning.universe` | Month-end alignment, tradability and point-in-time filters, 60-month rolling dollar-volume ranking → top-N universe |
| `data_cleaning.coverage` | Share of index members that have prices, per month |
| `features.technical` | Garman-Klass volatility, RSI, MACD, ATR, Bollinger bands |
| `features.returns` | Point-in-time lagged returns, cross-sectional winsorising/z-scores, rolling FF5 betas |
| `features.pipeline` | Builds `modeling_data.parquet` (19 features) and `monthly_returns.parquet`, records a parameter fingerprint |
| `clustering.methods` | K-Means, GMM, PCA+K-Means, HDBSCAN, fit independently each month |
| `clustering.runner` | Runs and caches clusters; latest-month helpers |
| `strategy.signals` | Top-N by momentum within each cluster, capped at `portfolio_size` |
| `strategy.optimiser` | HRP or equal weights |
| `strategy.targets` | Selection → target weights (optional recent-window HRP, exposure scalar) → trades |
| `strategy.rebalancer` | Target vs. current weights → sells-then-buys list (dollar floor, optional % band) |
| `strategy.simulate` | Month-by-month point-in-time engine: same selection and weights as live, costs on drifted turnover, real exit returns |
| `strategy.backtest` | `run_backtest` on the engine; metrics, SPY benchmark (never a silent 0%), period slicing |
| `strategy.walkforward` | Train / validation / test evaluation of the hyperparameters, net of costs |
| `strategy.regime` | Regime indicators (SPY drawdown, VIX, correlation shift, dispersion) and the stress-window helper |
| `validation.checks` | Data, cluster and selection health checks |
| `storage.fingerprint` | Warns when a cache was built with different parameters from the active ones |
| `broker.paper` / `broker.ibkr` | Simulated ledger / Interactive Brokers TWS execution |
| `dashboard.app` | Flask UI over portfolio, trades, backtest |
| `config` | Loads `configs/strategy.yaml`, applies a regime overlay, validates |

## Quick Start

```bash
pip install -r requirements.txt && pip install -e .

mr build-features        # first run downloads ~600+ tickers (15-20 min)
mr cluster
mr coverage              # how many index members have prices (the survivorship gap)
mr crsp-check            # (research on CRSP) verify WRDS tables and columns first
mr backtest              # point-in-time backtest, net of costs (--cost-bps, --delist-return)
mr walkforward           # out-of-sample evaluation of the hyperparameters
mr signals               # this month's selected stocks + health checks
mr regime                # regime indicators; changes nothing
mr trades                # dry run (add --save to write a CSV to data/results/)
mr rebalance             # confirmation prompt; paper broker by default
```

Or use the building blocks directly (this snippet runs as written):

```python
import numpy as np
import pandas as pd
from momentum_rebalancer.strategy.signals import select_top_momentum
from momentum_rebalancer.strategy.optimiser import compute_weights
from momentum_rebalancer.strategy.rebalancer import compute_trades

# Toy inputs: one month of clustered stocks, and a history of monthly returns.
rng = np.random.default_rng(0)
tickers = [f"S{i:02d}" for i in range(60)]
month_data = pd.DataFrame(
    {"cluster": rng.integers(0, 5, 60), "return_3m": rng.normal(0.01, 0.03, 60)},
    index=pd.Index(tickers, name="Ticker"),
)
monthly_returns = pd.DataFrame(rng.normal(0, 0.05, (60, 60)), columns=tickers)

selected = select_top_momentum(month_data, "return_3m", top_n=4)          # top 4 per cluster
weights  = compute_weights(selected.index.tolist(), returns=monthly_returns, method="hrp")
trades   = compute_trades(weights, current_holdings={}, portfolio_value=1_000_000)
```

See `notebooks/` for exploration and [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for the monthly routine.

## When the Market Regime Changes

| Need | Where |
|---|---|
| What is the market doing? | `mr regime` (SPY drawdown, VIX, holdings correlation shift, dispersion, data/cluster health, stale caches) |
| What can I change, and what must I re-run? | [`docs/PARAMETERS.md`](docs/PARAMETERS.md) |
| What should I change in a sell-off / crash / rebound / rotation? | [`docs/REGIME_PLAYBOOK.md`](docs/REGIME_PLAYBOOK.md), and presets in [`configs/regimes.yaml`](configs/regimes.yaml) |
| Preview or apply a preset | `mr regimes`, then `mr --regime crash trades` / `rebalance` |
| Test a change on 2020 / 2022 | `python scripts/stress_windows.py --regime rebound` (net of costs) |
| Record the decision | `research_logs/TEMPLATE.md` |

Regime presets are **untested hypotheses**. The backtest can show what a preset would have done through a window (HRP window, exposure, band and size cap are all modelled), but not whether you would have recognised the regime in time. Detection is manual: the tool reports and a person decides.

## Repository Structure

```
.
├── configs/                          # All parameters, as data
│   ├── strategy.yaml                 # Baseline parameters (universe, features, clustering, strategy, monitor, broker)
│   ├── regimes.yaml                  # Named overlays: baseline, stress, crash, rebound, rotation (untested)
│   ├── walk_forward.yaml             # Folds, objective and hyperparameter grid for `mr walkforward`
│   ├── crsp.yaml                     # WRDS/CRSP tables, columns, options (price_source: crsp)
│   ├── stress_windows.yaml           # Named date windows for scripts/stress_windows.py
│   └── README.md                     # How configs, overlays and validation fit together
│
├── docs/
│   ├── README.md                     # Index: what to read when
│   ├── PARAMETERS.md                 # Every key: where it is read, effect, which re-run it needs
│   ├── VALIDATION.md                 # What is point-in-time, how it is tested, walk-forward, what it cannot tell you
│   ├── DATA_SOURCES.md               # Membership file, Yahoo's gaps, vendor files, measuring the survivorship gap
│   ├── REGIME_PLAYBOOK.md            # Regime indicators, what to change, how to test it
│   ├── KNOWN_LIMITATIONS.md          # Known issues and their consequences
│   ├── DATA_CONVENTIONS.md           # Cache files, index/column conventions
│   └── WORKFLOW.md                   # Monthly routine, change control, testing
│
├── notebooks/                        # Exploration (outputs stripped on commit)
├── reports/
│   ├── figures/                      # Saved charts
│   └── tables/                       # Saved summary tables
├── research_logs/
│   ├── TEMPLATE.md                   # Monthly rebalance / regime decision log
│   └── README.md
│
├── scripts/
│   ├── README.md
│   └── stress_windows.py             # Strategy vs SPY on named stress windows
│
├── src/momentum_rebalancer/          # Core package
│   ├── README.md                     # Package map and data flow
│   ├── __init__.py
│   ├── __main__.py                   # `python -m momentum_rebalancer`
│   ├── cli.py                        # `mr` command
│   ├── config.py                     # Loads/validates YAML, exposes constants
│   │
│   ├── data_ingestion/
│   │   ├── loader.py                 # PIT membership + price download + coverage record
│   │   ├── crsp.py                   # CRSP via WRDS: membership, prices, delisting returns, crsp-check
│   │   ├── external.py               # Optional vendor price files
│   │   └── symbols.py                # Ticker normalisation
│   ├── data_cleaning/
│   │   ├── universe.py               # Month-end alignment, tradability + PIT filters, liquidity universe
│   │   └── coverage.py               # Index members with prices, per month
│   ├── features/
│   │   ├── technical.py              # Daily indicators → month-end snapshot
│   │   ├── returns.py                # Lagged returns, z-scores, FF5 betas
│   │   └── pipeline.py               # Builds modeling_data.parquet
│   ├── clustering/
│   │   ├── methods.py                # kmeans / gmm / pca_kmeans / hdbscan
│   │   └── runner.py                 # Run + cache + latest month
│   ├── strategy/
│   │   ├── signals.py                # Top-N momentum per cluster
│   │   ├── optimiser.py              # HRP / equal weight
│   │   ├── targets.py                # Selection → target weights → trades
│   │   ├── rebalancer.py             # Trade list
│   │   ├── simulate.py               # Point-in-time simulation engine (costs, drift, exits)
│   │   ├── backtest.py               # Backtest + metrics
│   │   ├── walkforward.py            # Train / validation / test evaluation
│   │   └── regime.py                 # Regime indicators and report
│   ├── validation/
│   │   └── checks.py                 # Data / cluster / selection health checks
│   ├── storage/
│   │   └── fingerprint.py            # Cache parameter fingerprints
│   ├── broker/
│   │   ├── paper.py                  # Local simulated ledger
│   │   └── ibkr.py                   # Interactive Brokers TWS
│   └── dashboard/
│       ├── app.py                    # Flask API
│       └── templates/index.html      # Single-page UI
│                                     # (each package folder has its own README.md)
│
├── tests/                            # 188 tests; synthetic data, no network, no TWS (README lists coverage)
│
├── data/                             # Generated caches (contents git-ignored)
├── old/                              # Archived baseline strategy (K-Means + top-5 momentum)
│
├── .env.example                      # MR_DATA_ROOT, MR_REGIME, MR_BROKER
├── .gitignore
├── .pre-commit-config.yaml           # nbstripout
├── pyproject.toml                    # Package metadata, `mr` entry point, pytest config
├── requirements.txt
└── README.md
```

## Key Files

| File | Purpose |
|---|---|
| `configs/strategy.yaml` | The single source of truth for parameters |
| `configs/regimes.yaml` | Regime overlays (untested hypotheses except `baseline`) |
| `src/momentum_rebalancer/config.py` | Loads, overlays and validates configuration |
| `src/momentum_rebalancer/cli.py` | All `mr` commands |
| `src/momentum_rebalancer/strategy/targets.py` | The live path from clusters to trades |
| `src/momentum_rebalancer/strategy/regime.py` | Regime indicators |
| `src/momentum_rebalancer/broker/ibkr.py` | Real order placement; read limitation C1 first |
| `docs/REGIME_PLAYBOOK.md` | What to change when the market changes |

## Data Storage

Large data stays out of git. By default it lives in `<repo>/data/`; set `MR_DATA_ROOT` in `.env` to move it.

```
data/store/      parquet caches, *.meta.json fingerprints, price_coverage.json, paper_ledger.json
data/results/    CSVs written by --save
data/external/   vendor price files you provide (git-ignored)
```

### Core Tables

| Table | Description |
|---|---|
| `pit_members.parquet` | Date-indexed S&P 500 membership (list of tickers per date) |
| `prices_raw.parquet` | Daily OHLCV, `(Date, Ticker)` |
| `modeling_data.parquet` | Month-end `(Date, Ticker)` × 19 features |
| `monthly_returns.parquet` | Raw monthly total return, Date × Ticker, for every ticker with prices |
| `price_coverage.json` | Which requested tickers returned no prices |
| `clustered_<method>.parquet` | Modeling data + `cluster` |
| `paper_ledger.json` | Simulated cash, positions, fills |

### Core Variables

| Variable | Description |
|---|---|
| `return_<n>m` | Per-month-equivalent return over the last *n* months, winsorised |
| `momentum_feature` | Column used for ranking inside clusters (default `return_3m`) |
| `cluster` | Month-specific label; ids are **not** comparable across months |
| `momentum_score` | Value of the momentum feature for a selected stock |
| `target_weight` | Fraction of NAV after HRP and the exposure scalar |
| `delta_pct` | Target minus current weight |
| `exposure_scalar` | Multiplier on all target weights; remainder is cash |
| `hrp_lookback_months` | Months of return history fed to HRP (`null` = all) |
| `min_trade_pct` | Smallest weight change worth trading (exits exempt) |

## Core Formulas

Per-month-equivalent return over lag $n$ (before winsorisation):

$$r^{(n)}_{i,t} = \left(\frac{P_{i,t}}{P_{i,t-n}}\right)^{1/n} - 1$$

Selection: within cluster $c$ at month $t$, take the $N$ stocks with the largest $r^{(3)}_{i,t}$ (default $N=4$).

HRP correlation distance and recursive-bisection split between left/right sub-portfolios with variances $V_L$, $V_R$:

$$d_{ij} = \sqrt{\tfrac{1}{2}\,(1-\rho_{ij})}, \qquad \alpha_L = 1-\frac{V_L}{V_L+V_R}, \quad \alpha_R = 1-\alpha_L$$

Target weight with exposure scalar $s \in (0,1]$ (the remainder $1-s$ is cash):

$$w^{\text{target}}_i = s \cdot w^{\text{HRP}}_i$$

Net return earned over month $t+1$, with drifted weights $\tilde w$, cost rate $c = \text{cost\_bps}/10^4$ and delisting return $\delta$ for a held stock with no price:

$$R^{\text{net}}_{t+1} = \sum_i w_{i,t}\, r_{i,t+1} \;-\; c \sum_i \left| w_{i,t} - \tilde w_{i,t} \right|, \qquad \tilde w_{i,t} = \frac{w_{i,t-1}(1+r_{i,t})}{1+R^{\text{gross}}_t}$$

Trade and band, with NAV $V$ and band $b$ (exits, $w^{\text{target}}_i = 0$, always trade):

$$\Delta_i = w^{\text{target}}_i - w^{\text{current}}_i, \qquad \text{trade if } |\Delta_i V| \ge \$100 \text{ and } \big(w^{\text{target}}_i = 0 \text{ or } |\Delta_i| \ge b\big)$$

## Installation

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`, Mac/Linux: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
pip install -e .                 # required: config.py locates configs/ from the source tree
copy .env.example .env           # Windows   (cp on Mac/Linux); every variable is optional
```

Needs pandas ≥ 2.2 (and `pip install wrds` for the CRSP path); tested with pandas 2.3 and 3.0 on Python 3.12.

## Testing

```bash
pytest
```

188 tests on synthetic data: look-ahead detection (the feature set must not change when the future is removed), the simulator's causality, costs and exits, the walk-forward, the data layer and coverage, optimiser, rebalancer, signals, regime indicators, the feature pipeline end to end with the network stubbed, paper broker, dashboard POST guard. They show the code does what it says, not that the strategy makes money. The IBKR backend is not covered.

## Notebook Output Cleanup

```bash
pip install pre-commit nbstripout
pre-commit install
nbstripout --install
```

## Workflow

Monthly routine, parameter change control and branching rules: [`docs/WORKFLOW.md`](docs/WORKFLOW.md). Log every regime decision in `research_logs/`.

## Example Commands

```bash
mr regimes                                    # list presets
mr backtest --cost-bps 20 --delist-return -0.5   # sensitivity to the assumptions
mr --regime stress trades                     # preview trades under a preset
python scripts/stress_windows.py --top-n 5    # ad hoc stress-window comparison
MR_BROKER=ibkr mr rebalance                   # opt in to TWS (paper port 7497 by default)
python -m momentum_rebalancer dashboard
pytest
```

## Disclaimer

Educational / personal research project. Not investment advice. Backtests exclude transaction costs and carry the biases listed in the limitations doc.
