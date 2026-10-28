"""
clustering/methods.py — four clustering algorithms, uniform interface.

Every method accepts (data, feature_cols) and returns data with a
'cluster' column added. All clustering is done independently per month
to prevent lookahead bias.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import hdbscan

from momentum_rebalancer.config import N_CLUSTERS, PCA_COMPONENTS, HDBSCAN_MIN_SIZE, RANDOM_STATE


# ── Core helper ───────────────────────────────────────────────────────────────

def _apply_per_month(
    data: pd.DataFrame,
    feature_cols: list[str],
    fit_fn,
) -> pd.DataFrame:
    """
    Apply fit_fn independently for each month.
    fit_fn: (X: np.ndarray) → cluster_labels: np.ndarray
    """
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        X = df[feature_cols].values
        df = df.copy()
        df["cluster"] = fit_fn(X)
        return df

    return (
        data
        .dropna(subset=feature_cols)
        .groupby("Date", group_keys=False)
        .apply(_apply)
    )


# ── K-Means ───────────────────────────────────────────────────────────────────

def kmeans(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    def fit(X):
        return KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
        ).fit_predict(X)

    return _apply_per_month(data, feature_cols, fit)


# ── GMM ───────────────────────────────────────────────────────────────────────

def gmm(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    def fit(X):
        return GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        ).fit_predict(X)

    return _apply_per_month(data, feature_cols, fit)


# ── PCA + K-Means ─────────────────────────────────────────────────────────────

def pca_kmeans(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = N_CLUSTERS,
    n_components: int = PCA_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    def fit(X):
        X_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
        return KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X_pca)

    return _apply_per_month(data, feature_cols, fit)


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def hdbscan_cluster(
    data: pd.DataFrame,
    feature_cols: list[str],
    min_cluster_size: int = HDBSCAN_MIN_SIZE,
    min_samples: int | None = None,
) -> pd.DataFrame:
    """
    Density-based clustering. Noise points are labelled -1.
    Consider filtering out cluster=-1 before strategy generation.
    """
    def fit(X):
        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        ).fit_predict(X)

    return _apply_per_month(data, feature_cols, fit)


# ── Dispatcher ────────────────────────────────────────────────────────────────

METHODS = {
    "kmeans":     kmeans,
    "gmm":        gmm,
    "pca_kmeans": pca_kmeans,
    "hdbscan":    hdbscan_cluster,
}


def cluster(
    data: pd.DataFrame,
    feature_cols: list[str],
    method: str = "kmeans",
    **kwargs,
) -> pd.DataFrame:
    """Dispatch to the requested clustering method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS)}")
    return METHODS[method](data, feature_cols, **kwargs)
