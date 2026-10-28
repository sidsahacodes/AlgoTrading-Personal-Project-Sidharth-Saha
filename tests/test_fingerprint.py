import pandas as pd

from momentum_rebalancer.storage import fingerprint as fp

S = {"regime": "baseline", "n_clusters": 5, "random_state": 42, "cluster_method": "kmeans",
     "feature_cols": ["a", "b"], "hdbscan_min_size": 15, "pca_components": 6,
     **{k: 1 for k in fp.FEATURE_KEYS if k not in ("feature_cols",)}}


def _touch(path):
    pd.DataFrame({"x": [1]}).to_parquet(path)
    return path


def test_no_diff_right_after_write(tmp_path):
    p = _touch(tmp_path / "clustered_kmeans.parquet")
    fp.write_meta(p, S, fp.cluster_keys("kmeans"))
    assert fp.diff_params(p, S, fp.cluster_keys("kmeans")) == []


def test_changed_parameter_is_reported(tmp_path):
    p = _touch(tmp_path / "clustered_kmeans.parquet")
    fp.write_meta(p, S, fp.cluster_keys("kmeans"))
    changed = {**S, "n_clusters": 6}
    out = fp.diff_params(p, changed, fp.cluster_keys("kmeans"))
    assert len(out) == 1 and "n_clusters" in out[0]


def test_irrelevant_parameter_not_reported(tmp_path):
    p = _touch(tmp_path / "clustered_hdbscan.parquet")
    fp.write_meta(p, S, fp.cluster_keys("hdbscan"))
    changed = {**S, "n_clusters": 9}                 # HDBSCAN ignores n_clusters
    assert fp.diff_params(p, changed, fp.cluster_keys("hdbscan")) == []


def test_missing_meta_is_flagged_not_crashing(tmp_path):
    p = _touch(tmp_path / "modeling_data.parquet")
    out = fp.diff_params(p, S, fp.FEATURE_KEYS)
    assert len(out) == 1 and "no parameter record" in out[0]


def test_missing_data_file_gives_no_warning(tmp_path):
    assert fp.diff_params(tmp_path / "nope.parquet", S, fp.FEATURE_KEYS) == []


def test_clusters_older_than_features_are_flagged(tmp_path):
    modeling = _touch(tmp_path / "modeling_data.parquet")
    clustered = _touch(tmp_path / "clustered_kmeans.parquet")
    fp.write_meta(modeling, S, fp.FEATURE_KEYS)
    fp.write_meta(clustered, S, fp.cluster_keys("kmeans"), extra={"modeling_built_at": "2000-01-01T00:00:00"})
    warns = fp.stale_warnings(S, tmp_path)
    assert any("older modeling_data" in w for w in warns)
