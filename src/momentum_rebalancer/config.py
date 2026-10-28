"""
config.py — loads configs/strategy.yaml, applies an optional regime overlay
from configs/regimes.yaml, validates, and exposes the familiar UPPER_CASE
constants used across the package.

Environment variables (all optional, may be set in a local .env):
  MR_REGIME       regime preset name from regimes.yaml (default: strategy.yaml `regime`)
  MR_BROKER       "paper" | "ibkr"  (overrides strategy.yaml)
  MR_DATA_ROOT    where data/store and data/results live (default: <repo>/data)
  MR_CONFIG_DIR   folder holding strategy.yaml / regimes.yaml (default: <repo>/configs)

Requires an editable install (`pip install -e .`) so that the repo root can be
located from this file's position (<repo>/src/momentum_rebalancer/config.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

try:                                   # optional convenience
    from dotenv import load_dotenv
except ImportError:                    # pragma: no cover
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

CONFIG_DIR = Path(os.getenv("MR_CONFIG_DIR", REPO_ROOT / "configs"))

VALID_CLUSTER_METHODS = ("kmeans", "gmm", "pca_kmeans", "hdbscan")
VALID_OPTIMISERS      = ("hrp", "equal")
VALID_BROKERS         = ("paper", "ibkr")
VALID_PRICE_SOURCES   = ("yahoo", "crsp")


# ── Loading ───────────────────────────────────────────────────────────────────

def _read_yaml(name: str, config_dir: Path) -> dict:
    path = Path(config_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run from a checkout installed with `pip install -e .`, "
            f"or set MR_CONFIG_DIR."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _flatten(base: dict) -> dict:
    """Collapse the sectioned YAML to {key: value}; keys must be unique."""
    flat: dict = {}
    for key, val in base.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if k in flat:
                    raise ValueError(f"Duplicate config key '{k}' in strategy.yaml")
                flat[k] = v
        else:
            flat[key] = val
    return flat


def available_regimes(config_dir: Path | None = None) -> dict:
    return _read_yaml("regimes.yaml", config_dir or CONFIG_DIR)


def load_settings(regime: str | None = None, config_dir: Path | None = None) -> dict:
    """Return the flat, validated settings dict for `regime` (or the active one)."""
    config_dir = Path(config_dir or CONFIG_DIR)
    base    = _read_yaml("strategy.yaml", config_dir)
    regimes = _read_yaml("regimes.yaml", config_dir)

    name = regime or os.getenv("MR_REGIME") or base.get("regime", "baseline")
    if name not in regimes:
        raise ValueError(f"Unknown regime '{name}'. Available: {sorted(regimes)}")

    flat = _flatten(base)
    overrides = dict(regimes[name].get("overrides") or {})
    unknown = sorted(set(overrides) - set(flat))
    if unknown:
        raise ValueError(f"Regime '{name}' overrides unknown keys: {unknown}")

    flat.update(overrides)
    flat["regime"] = name
    flat["regime_overrides"] = overrides
    flat["regime_description"] = regimes[name].get("description", "")

    env_source = os.getenv("MR_PRICE_SOURCE")
    if env_source:
        flat["price_source"] = env_source

    env_broker = os.getenv("MR_BROKER")
    if env_broker:
        flat["broker"] = env_broker

    validate(flat)
    return flat


def validate(s: dict) -> None:
    """Fail fast on inconsistent settings instead of failing mid-pipeline."""
    errs = []
    feats = list(s["feature_cols"])
    if s["momentum_feature"] not in feats:
        errs.append(f"momentum_feature '{s['momentum_feature']}' not in feature_cols")
    for lag in s["return_lags"]:
        if f"return_{lag}m" not in feats:
            errs.append(f"return_lags has {lag} but feature_cols lacks 'return_{lag}m'")
    for col in feats:
        if col.startswith("return_") and col.endswith("m"):
            try:
                lag = int(col[len("return_"):-1])
            except ValueError:
                continue
            if lag not in s["return_lags"]:
                errs.append(f"feature_cols has '{col}' but return_lags lacks {lag}")
    if s["cluster_method"] not in VALID_CLUSTER_METHODS:
        errs.append(f"cluster_method must be one of {VALID_CLUSTER_METHODS}")
    if s["optimiser"] not in VALID_OPTIMISERS:
        errs.append(f"optimiser must be one of {VALID_OPTIMISERS}")
    if s["broker"] not in VALID_BROKERS:
        errs.append(f"broker must be one of {VALID_BROKERS}")
    if not 0 < float(s["exposure_scalar"]) <= 1:
        errs.append("exposure_scalar must be in (0, 1]")
    if not 0 <= float(s["min_trade_pct"]) < 0.1:
        errs.append("min_trade_pct must be in [0, 0.1)")
    hl = s["hrp_lookback_months"]
    if hl is not None and int(hl) < 12:
        errs.append("hrp_lookback_months must be null or >= 12 (>= 36 advised for ~20 assets)")
    if s["price_source"] not in VALID_PRICE_SOURCES:
        errs.append(f"price_source must be one of {VALID_PRICE_SOURCES}")
    if not 0 <= float(s["cost_bps"]) <= 500:
        errs.append("cost_bps must be in [0, 500]")
    if not -1.0 <= float(s["delisting_return"]) <= 1.0:
        errs.append("delisting_return must be in [-1, 1]")
    if int(s["max_stale_days"]) < 1:
        errs.append("max_stale_days must be >= 1")
    if int(s["lookback_years"]) < 5:
        errs.append("lookback_years must be >= 5 (36m history + 24m betas leave too little otherwise)")
    if not str(s["pit_url"]).startswith("http"):
        errs.append("pit_url must be an http(s) URL")
    if int(s["top_n_per_cluster"]) < 1 or int(s["portfolio_size"]) < 1:
        errs.append("top_n_per_cluster and portfolio_size must be >= 1")
    if errs:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errs))


# ── Active settings (module-level constants) ──────────────────────────────────

SETTINGS = load_settings()

REGIME             = SETTINGS["regime"]
REGIME_OVERRIDES   = SETTINGS["regime_overrides"]
REGIME_DESCRIPTION = SETTINGS["regime_description"]

# Paths
DATA_ROOT   = Path(os.getenv("MR_DATA_ROOT", REPO_ROOT / "data"))
DATA_DIR    = DATA_ROOT / "store"
RESULTS_DIR = DATA_ROOT / "results"
EXTERNAL_PRICES_DIR = DATA_ROOT / "external" / "prices"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_PRICES_DIR.mkdir(parents=True, exist_ok=True)

# Universe
LOOKBACK_YEARS     = int(SETTINGS["lookback_years"])
MAX_STALE_DAYS     = int(SETTINGS["max_stale_days"])
PIT_URL            = str(SETTINGS["pit_url"])
PRICE_SOURCE       = str(SETTINGS["price_source"])
PIT_ALLOW_FALLBACK = bool(SETTINGS["pit_allow_fallback"])
MIN_HISTORY_MONTHS = int(SETTINGS["min_history_months"])
UNIVERSE_SIZE      = int(SETTINGS["universe_size"])

# Features
RSI_PERIOD     = int(SETTINGS["rsi_period"])
ATR_PERIOD     = int(SETTINGS["atr_period"])
BB_PERIOD      = int(SETTINGS["bb_period"])
MACD_FAST      = int(SETTINGS["macd_fast"])
MACD_SLOW      = int(SETTINGS["macd_slow"])
MACD_SIGNAL    = int(SETTINGS["macd_signal"])
FF_BETA_WINDOW = int(SETTINGS["ff_beta_window"])
RETURN_LAGS    = [int(x) for x in SETTINGS["return_lags"]]
FEATURE_COLS   = list(SETTINGS["feature_cols"])

# Clustering
CLUSTER_METHOD   = SETTINGS["cluster_method"]
N_CLUSTERS       = int(SETTINGS["n_clusters"])
PCA_COMPONENTS   = int(SETTINGS["pca_components"])
HDBSCAN_MIN_SIZE = int(SETTINGS["hdbscan_min_size"])
RANDOM_STATE     = int(SETTINGS["random_state"])

# Strategy
MOMENTUM_FEATURE    = SETTINGS["momentum_feature"]
TOP_N_PER_CLUSTER   = int(SETTINGS["top_n_per_cluster"])
PORTFOLIO_SIZE      = int(SETTINGS["portfolio_size"])
OPTIMISER           = SETTINGS["optimiser"]
HRP_LOOKBACK_MONTHS = None if SETTINGS["hrp_lookback_months"] is None else int(SETTINGS["hrp_lookback_months"])
EXPOSURE_SCALAR     = float(SETTINGS["exposure_scalar"])
MIN_TRADE_PCT       = float(SETTINGS["min_trade_pct"])

# Backtest assumptions
COST_BPS         = float(SETTINGS["cost_bps"])
DELISTING_RETURN = float(SETTINGS["delisting_return"])

# Regime monitor thresholds
SPY_DD_CAUTION    = float(SETTINGS["spy_dd_caution"])
SPY_DD_STRESS     = float(SETTINGS["spy_dd_stress"])
VIX_CAUTION       = float(SETTINGS["vix_caution"])
VIX_STRESS        = float(SETTINGS["vix_stress"])
CORR_JUMP         = float(SETTINGS["corr_jump"])
CLUSTER_MAX_SHARE = float(SETTINGS["cluster_max_share"])

# Broker
BROKER          = SETTINGS["broker"]
PORTFOLIO_VALUE = float(SETTINGS["portfolio_value"])
TWS_HOST        = SETTINGS["tws_host"]
TWS_PORT        = int(SETTINGS["tws_port"])

# Dashboard
DASHBOARD_HOST = SETTINGS["dashboard_host"]
DASHBOARD_PORT = int(SETTINGS["dashboard_port"])
DEBUG          = bool(SETTINGS["dashboard_debug"])


# ── Walk-forward settings ─────────────────────────────────────────────────────

def load_walk_forward(config_dir: Path | None = None) -> dict:
    """Load and validate configs/walk_forward.yaml."""
    wf = _read_yaml("walk_forward.yaml", Path(config_dir or CONFIG_DIR))
    sp, grid = wf["splits"], wf["grid"]
    errs = []
    for k in ("train_months", "val_months", "test_months", "step_months"):
        if int(sp[k]) < 1:
            errs.append(f"splits.{k} must be >= 1")
    if int(sp["embargo_months"]) < 0:
        errs.append("splits.embargo_months must be >= 0")
    if int(sp["step_months"]) < int(sp["test_months"]):
        errs.append("splits.step_months must be >= test_months (test blocks must not overlap)")
    for f in grid["momentum_feature"]:
        if f not in FEATURE_COLS:
            errs.append(f"grid.momentum_feature '{f}' is not in feature_cols")
    if wf.get("objective", "sharpe") not in ("sharpe", "cumulative_return"):
        errs.append("objective must be 'sharpe' or 'cumulative_return'")
    if errs:
        raise ValueError("Invalid walk_forward.yaml:\n  - " + "\n  - ".join(errs))
    return wf


# ── CRSP / WRDS settings ──────────────────────────────────────────────────────

def load_crsp_settings(config_dir: Path | None = None) -> dict:
    """Load and sanity-check configs/crsp.yaml (only needed when price_source is crsp)."""
    cfg = _read_yaml("crsp.yaml", Path(config_dir or CONFIG_DIR))
    need = {"wrds": ["username_env"], "libraries": ["stock", "indexes"],
            "tables": ["daily", "membership", "security_info"],
            "columns": ["daily", "membership", "security_info"]}
    errs = [f"crsp.yaml: missing {sec}.{k}" for sec, keys in need.items() for k in keys
            if k not in (cfg.get(sec) or {})]
    for k in ("permno", "date", "delisting_flag", "price", "ret", "volume", "open", "high", "low"):
        if k not in (cfg.get("columns", {}).get("daily") or {}):
            errs.append(f"crsp.yaml: missing columns.daily.{k}")
    if errs:
        raise ValueError("Invalid crsp.yaml:\n  - " + "\n  - ".join(errs))
    return cfg

