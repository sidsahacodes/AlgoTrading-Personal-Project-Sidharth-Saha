"""
clustering/runner.py — apply clustering to modeling data and cache results.
"""

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

from momentum_rebalancer.config import DATA_DIR, CLUSTER_METHOD, FEATURE_COLS, SETTINGS
from momentum_rebalancer.storage.fingerprint import cluster_keys, read_meta, write_meta
from momentum_rebalancer.clustering.methods import cluster

logger = logging.getLogger(__name__)


def get_clustered_path(method: str) -> Path:
    return DATA_DIR / f"clustered_{method}.parquet"


def run_clustering(
    data: pd.DataFrame,
    method: str = CLUSTER_METHOD,
    force_refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply clustering to modeling data.
    Caches results per method; set force_refresh=True to recompute.
    """
    cache_path = get_clustered_path(method)

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached clusters: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Running %s clustering …", method)
    clustered = cluster(data, FEATURE_COLS, method=method, **kwargs)

    clustered.to_parquet(cache_path)
    modeling_meta = read_meta(DATA_DIR / "modeling_data.parquet") or {}
    write_meta(cache_path, SETTINGS, cluster_keys(method),
               extra={"modeling_built_at": modeling_meta.get("built_at")})
    logger.info(
        "Saved %s clusters → %s  (%d unique clusters found)",
        method, cache_path,
        clustered["cluster"].nunique(),
    )
    return clustered


def load_clustered(method: str = CLUSTER_METHOD) -> pd.DataFrame:
    path = get_clustered_path(method)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached clusters for method='{method}'. "
            f"Run: python run.py cluster"
        )
    return pd.read_parquet(path)


def get_latest_month(clustered: pd.DataFrame) -> pd.Timestamp:
    return clustered.index.get_level_values("Date").max()


def get_current_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    """Return only the most recent month's cluster assignments."""
    latest = get_latest_month(clustered)
    return clustered.xs(latest, level="Date")
