import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import hdbscan 


# ==================================================
# Core helper (DO NOT TOUCH)
# ==================================================

def _cluster_by_date(
    data: pd.DataFrame,
    feature_cols: list[str],
    fit_func,
) -> pd.DataFrame:
    """
    Apply clustering independently for each month.
    Prevents lookahead bias and cross-month leakage.
    """

    def apply_cluster(df):
        X = df[feature_cols].values
        df["cluster"] = fit_func(X)
        return df

    return (
        data
        .dropna(subset=feature_cols)
        .groupby("Date", group_keys=False)
        .apply(apply_cluster)
    )


# ==================================================
# K-MEANS
# ==================================================

def kmeans_clustering(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 5,
    init_centroids: np.ndarray | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Cross-sectional K-Means clustering (hard assignment).
    """

    def fit(X):
        model = KMeans(
            n_clusters=n_clusters,
            init=init_centroids if init_centroids is not None else "k-means++",
            n_init=1 if init_centroids is not None else 10,
            random_state=random_state,
        )
        return model.fit_predict(X)

    return _cluster_by_date(data, feature_cols, fit)


# ==================================================
# GAUSSIAN MIXTURE MODEL
# ==================================================

def gmm_clustering(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Soft clustering using Gaussian Mixture Models.
    """

    def fit(X):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        )
        return model.fit_predict(X)

    return _cluster_by_date(data, feature_cols, fit)


# ==================================================
# PCA + K-MEANS
# ==================================================

def pca_kmeans_clustering(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 5,
    n_components: int = 6,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    PCA-reduced K-Means clustering.
    Useful for denoising high-dimensional features.
    """

    def fit(X):
        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X)

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
        )
        return model.fit_predict(X_pca)

    return _cluster_by_date(data, feature_cols, fit)


# ==================================================
# HDBSCAN (RECOMMENDED)
# ==================================================

def hdbscan_clustering(
    data: pd.DataFrame,
    feature_cols: list[str],
    min_cluster_size: int = 15,
    min_samples: int | None = None,
) -> pd.DataFrame:
    """
    Density-based clustering with noise detection.
    Noise points are labeled as -1.
    """

    def fit(X):
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        return model.fit_predict(X)

    return _cluster_by_date(data, feature_cols, fit)
