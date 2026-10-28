import os
import tempfile

# Must run before momentum_rebalancer.config is imported anywhere:
# keep test artefacts (caches, paper ledger) out of the real data folder.
os.environ["MR_DATA_ROOT"] = tempfile.mkdtemp(prefix="mr_test_")
os.environ["MR_NO_DOTENV"] = "1"            # ignore any local .env file
# Tests must not depend on (or touch) whatever the shell happens to have set: a research session with
# MR_PRICE_SOURCE=crsp would otherwise send Yahoo-path tests to the real WRDS database.
for var in ("MR_REGIME", "MR_BROKER", "MR_PRICE_SOURCE", "MR_CONFIG_DIR", "WRDS_USERNAME"):
    os.environ.pop(var, None)

import numpy as np
import pandas as pd
import pytest


def make_modeling(n_months=30, n_tickers=60, seed=0, cols=None):
    """Synthetic modeling_data-like frame indexed by (Date, Ticker)."""
    from momentum_rebalancer.config import FEATURE_COLS
    cols = cols or FEATURE_COLS
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    df = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    for c in cols:
        if c.startswith("return_"):
            df[c] = df[c] * 0.04
    return df


@pytest.fixture(autouse=True)
def _tests_never_reach_wrds(monkeypatch):
    """No test may open a real WRDS connection, whatever the shell, .env or saved password say.
    Tests that need a connection replace `crsp.connect` with a fake."""
    from momentum_rebalancer.data_ingestion import crsp

    def _blocked(cfg):
        raise AssertionError("A test tried to open a real WRDS connection; use a fake via monkeypatch.")
    monkeypatch.setattr(crsp, "connect", _blocked)


@pytest.fixture
def modeling():
    return make_modeling()
