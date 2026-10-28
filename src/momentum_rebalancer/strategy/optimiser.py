"""
strategy/optimiser.py — portfolio weight optimisation.

Supports:
  - HRP  (Hierarchical Risk Parity, Lopez de Prado 2016) — DEFAULT
  - Equal weight

HRP is preferred over MVO because:
  - No matrix inversion (avoids instability with ~20 assets)
  - Naturally diversified via hierarchical clustering on correlations
  - Does not require expected return estimates (which are noisy)
  - Robust out-of-sample (empirically outperforms MVO on Sharpe)
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from momentum_rebalancer.config import OPTIMISER

logger = logging.getLogger(__name__)


# ── Equal Weight ──────────────────────────────────────────────────────────────

def equal_weight(tickers: list[str]) -> dict[str, float]:
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


# ── HRP ───────────────────────────────────────────────────────────────────────

def _correlation_to_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-r))."""
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Recursively extract leaf order from linkage matrix (quasi-diagonalisation)."""
    return list(leaves_list(link))


def _hrp_recursive_bisection(
    cov: pd.DataFrame,
    sorted_tickers: list[str],
) -> dict[str, float]:
    """
    Recursive bisection: allocate risk equally across left/right branches.
    """
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]  # start with the full list as one cluster

    while clusters:
        clusters = [
            c[i:j]
            for c in clusters
            for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]

        for i in range(0, len(clusters), 2):
            left  = clusters[i]
            right = clusters[i + 1] if i + 1 < len(clusters) else []
            if not right:
                continue

            left_var  = _cluster_var(cov, left)
            right_var = _cluster_var(cov, right)
            total_var = left_var + right_var

            alloc_left  = 1 - left_var  / total_var
            alloc_right = 1 - right_var / total_var

            weights[left]  *= alloc_left
            weights[right] *= alloc_right

    return weights.to_dict()


def _cluster_var(cov: pd.DataFrame, tickers: list[str]) -> float:
    """Variance of an equal-weight sub-portfolio over the given tickers."""
    sub = cov.loc[tickers, tickers].values
    n   = len(tickers)
    w   = np.full(n, 1.0 / n)
    return float(w @ sub @ w)


def hrp(
    returns: pd.DataFrame,
    tickers: list[str] | None = None,
) -> dict[str, float]:
    """
    Hierarchical Risk Parity weights.

    Parameters
    ----------
    returns : DataFrame of monthly returns, columns = tickers
    tickers : optional subset of columns to use

    Returns
    -------
    dict of {ticker: weight}, weights sum to 1.0
    """
    if tickers:
        returns = returns[tickers]

    # Drop tickers with all-NaN or zero-variance
    returns = returns.dropna(axis=1, how="all")
    returns = returns.loc[:, returns.std() > 1e-8]

    if len(returns.columns) == 1:
        ticker = returns.columns[0]
        return {ticker: 1.0}

    cov  = returns.cov()
    corr = returns.corr()

    dist      = _correlation_to_distance(corr)
    condensed = squareform(dist, checks=False)
    link      = linkage(condensed, method="single")
    order     = _get_quasi_diag(link)

    sorted_tickers = [returns.columns[i] for i in order]
    weights = _hrp_recursive_bisection(cov, sorted_tickers)

    # Normalise to sum = 1
    total = sum(weights.values())
    weights = {t: w / total for t, w in weights.items()}

    logger.info(
        "HRP weights computed for %d assets. Min=%.3f, Max=%.3f",
        len(weights), min(weights.values()), max(weights.values()),
    )
    return weights


# ── Dispatcher ────────────────────────────────────────────────────────────────

def compute_weights(
    tickers: list[str],
    returns: pd.DataFrame | None = None,
    method: str = OPTIMISER,
) -> dict[str, float]:
    """
    Compute target portfolio weights.

    Parameters
    ----------
    tickers : selected tickers from signal generation
    returns : historical monthly returns DataFrame (required for HRP)
    method  : "hrp" | "equal"
    """
    if method == "equal":
        return equal_weight(tickers)

    if method == "hrp":
        if returns is None:
            logger.warning("No returns provided for HRP — falling back to equal weight.")
            return equal_weight(tickers)
        return hrp(returns, tickers=tickers)

    raise ValueError(f"Unknown optimiser '{method}'. Choose 'hrp' or 'equal'.")


# ── History → weights (shared by live trading and the backtest) ───────────────

MIN_HRP_OBS = 24      # months of returns a ticker needs before it enters HRP


def weights_from_history(
    tickers: list[str],
    history: pd.DataFrame,
    method: str = OPTIMISER,
    lookback_months: int | None = None,
    min_obs: int = MIN_HRP_OBS,
) -> tuple[dict[str, float], str]:
    """
    Weights for `tickers` from a wide frame of monthly returns (`history`, rows = months).

    The caller decides what `history` contains; for point-in-time use it must end at the
    decision date. Returns (weights, how) where `how` is "hrp", "equal", or
    "equal (insufficient history)" / "equal (hrp failed)" for fallbacks.

    * HRP uses only tickers with at least `min_obs` observations in the window; others are
      excluded from the portfolio.
    * Fewer than two usable tickers, or an HRP failure, falls back to equal weight over `tickers`.
    """
    if method == "equal" or not tickers:
        return equal_weight(tickers), "equal"

    cols = [t for t in tickers if t in history.columns]
    h = history.loc[:, cols].dropna(how="all")
    if lookback_months:
        h = h.tail(lookback_months)
    usable = [t for t in cols if h[t].notna().sum() >= min_obs]
    if len(usable) < 2:
        return equal_weight(tickers), "equal (insufficient history)"
    try:
        return hrp(h[usable], tickers=usable), "hrp"
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return equal_weight(tickers), "equal (hrp failed)"
