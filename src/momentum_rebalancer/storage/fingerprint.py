"""
storage/fingerprint.py — remember which parameters built each cached file.

Problem this solves: caches (modeling_data.parquet, clustered_<method>.parquet)
are reused unless you pass --force, so changing a parameter silently keeps
stale results. We write a small `<file>.meta.json` next to each cache with the
parameters that shaped it, and warn when the active config differs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_KEYS = [
    "lookback_years", "min_history_months", "universe_size", "max_stale_days", "pit_url", "price_source",
    "rsi_period", "atr_period", "bb_period",
    "macd_fast", "macd_slow", "macd_signal",
    "ff_beta_window", "return_lags", "feature_cols",
]


def cluster_keys(method: str) -> list[str]:
    """Only the parameters that actually influence `method`."""
    keys = ["feature_cols", "cluster_method"]
    if method in ("kmeans", "gmm", "pca_kmeans"):
        keys += ["n_clusters", "random_state"]
    if method == "pca_kmeans":
        keys += ["pca_components"]
    if method == "hdbscan":
        keys += ["hdbscan_min_size"]
    return keys


def meta_path(data_path: Path) -> Path:
    data_path = Path(data_path)
    return data_path.with_name(data_path.stem + ".meta.json")


def _snapshot(settings: dict, keys: list[str]) -> dict:
    return {k: settings.get(k) for k in keys}


def write_meta(data_path: Path, settings: dict, keys: list[str], extra: dict | None = None) -> dict:
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "regime": settings.get("regime"),
        "params": _snapshot(settings, keys),
    }
    if extra:
        meta.update(extra)
    meta_path(data_path).write_text(json.dumps(meta, indent=2, default=str))
    return meta


def read_meta(data_path: Path) -> dict | None:
    p = meta_path(data_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def diff_params(data_path: Path, settings: dict, keys: list[str]) -> list[str]:
    """Human-readable differences between the cache's parameters and the active ones."""
    data_path = Path(data_path)
    if not data_path.exists():
        return []
    meta = read_meta(data_path)
    if meta is None:
        return [f"{data_path.name}: no parameter record (built before fingerprints existed); "
                f"re-run with --force to record one"]
    old, new = meta.get("params", {}), _snapshot(settings, keys)
    out = []
    for k in keys:
        if old.get(k) != new.get(k):
            out.append(f"{data_path.name}: '{k}' was {old.get(k)!r} when built, now {new.get(k)!r}")
    return out


def stale_warnings(settings: dict, data_dir: Path) -> list[str]:
    """All cache-vs-config mismatches for the active settings."""
    data_dir = Path(data_dir)
    warns: list[str] = []

    modeling = data_dir / "modeling_data.parquet"
    warns += [w + "  -> `mr build-features --force`"
              for w in diff_params(modeling, settings, FEATURE_KEYS)]

    warns += [w + "  -> `mr build-features --force`"
              for w in diff_params(data_dir / "pit_members.parquet", settings, ["pit_url"])]

    method = settings["cluster_method"]
    clustered = data_dir / f"clustered_{method}.parquet"
    warns += [w + "  -> `mr cluster --force`"
              for w in diff_params(clustered, settings, cluster_keys(method))]

    # clusters older than the features they were fitted on
    cm, mm = read_meta(clustered), read_meta(modeling)
    if cm and mm and cm.get("modeling_built_at") not in (None, mm.get("built_at")):
        warns.append(f"{clustered.name}: built from an older modeling_data.parquet  -> `mr cluster --force`")
    return warns
