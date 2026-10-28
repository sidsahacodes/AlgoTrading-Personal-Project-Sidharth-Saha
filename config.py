"""
config.py — single source of truth for all project settings.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "store"
RESULTS_DIR = ROOT / "data" / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────
LOOKBACK_YEARS     = 10
MIN_HISTORY_MONTHS = 36
UNIVERSE_SIZE      = 150

# ── Features ──────────────────────────────────────────────────────────────────
RSI_PERIOD     = 14
ATR_PERIOD     = 14
BB_PERIOD      = 20
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
FF_BETA_WINDOW = 24
RETURN_LAGS    = [1, 2, 3, 6, 9, 12]

FEATURE_COLS = [
    "gkv", "rsi", "bblow", "bbmid", "bbhigh", "atr", "macd", "dv_rank",
    "return_1m", "return_2m", "return_3m", "return_6m", "return_9m", "return_12m",
    "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
]

# ── Clustering ────────────────────────────────────────────────────────────────
CLUSTER_METHOD   = "kmeans"   # "kmeans" | "gmm" | "pca_kmeans" | "hdbscan"
N_CLUSTERS       = 5
PCA_COMPONENTS   = 6
HDBSCAN_MIN_SIZE = 15
RANDOM_STATE     = 42

# ── Strategy ──────────────────────────────────────────────────────────────────
MOMENTUM_FEATURE   = "return_3m"
TOP_N_PER_CLUSTER  = 4          # 5 clusters × 4 = 20 stocks
PORTFOLIO_SIZE     = 20
OPTIMISER          = "hrp"      # "hrp" | "equal"

# ── Broker ────────────────────────────────────────────────────────────────────
# "paper" = local simulation, no TWS needed
# "ibkr"  = connects to TWS (must be open and logged in on port 7497)
BROKER           = "ibkr"
PORTFOLIO_VALUE  = 1_000_000    # your paper account value — used as fallback only

# ── TWS connection (only used when BROKER = "ibkr") ───────────────────────────
TWS_HOST  = "127.0.0.1"
TWS_PORT  = 7497                # 7497 = paper | 7496 = live
TWS_CLIENT_ID = 1

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5001
DEBUG          = False          # set True only for development
