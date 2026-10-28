"""Notebooks must be valid, free of saved outputs, and still run against the files the CLI writes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_clean_and_compiles(path):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None, f"cell {i} has saved output; strip it (nbstripout)"
            compile(cell.source, f"{path.name} cell {i}", "exec")


def _demo_results(folder: Path) -> None:
    """Files shaped exactly like `mr walkforward --save` and `mr backtest --save` write them."""
    from momentum_rebalancer.strategy.walkforward import make_folds
    folds = make_folds(pd.date_range("2014-08-31", periods=145, freq="ME"), 48, 12, 12, 12, 1)
    label = lambda p: f"{p[0]:%Y-%m}..{p[1]:%Y-%m}"
    rows, wf, fb, idx = [], [], [], []
    rng = np.random.default_rng(1)
    for f in folds:
        rows.append({"fold": f.k, "train": label(f.train), "validation": label(f.val), "test": label(f.test),
                     "chosen_momentum_feature": ["return_3m", "return_6m", "return_12m"][f.k % 3],
                     "chosen_top_n_per_cluster": 3 + (f.k % 3), "chosen_n_clusters": 4 + (f.k % 2),
                     "train_sharpe": 0.8, "val_sharpe": 1.0 + 0.3 * f.k, "test_sharpe": 1.2 - 0.4 * (f.k % 3),
                     "test_return": 0.15, "baseline_test_sharpe": 0.7, "baseline_test_return": 0.1})
        start = pd.Period(label(f.test).split("..")[0], "M") + 1
        wf += list(rng.normal(0.012, 0.03, 12)); fb += list(rng.normal(0.008, 0.03, 12))
        idx += list(pd.period_range(start, periods=12, freq="M").to_timestamp(how="end").normalize())
    index = pd.DatetimeIndex(idx, name="earned_in")
    pd.DataFrame(rows).set_index("fold").to_csv(folder / "walkforward_folds_baseline_2026-01-01.csv")
    pd.DataFrame({"walk_forward": wf, "fixed_baseline": fb}, index=index).to_csv(folder / "walkforward_oos_baseline_2026-01-01.csv")
    pd.DataFrame({"net": wf, "gross": wf, "benchmark": rng.normal(0.01, 0.04, len(index)), "traded": 1.3},
                 index=index).to_csv(folder / "backtest_baseline_2026-01-01.csv")


def test_walk_forward_notebook_runs_end_to_end(tmp_path, monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    _demo_results(tmp_path)
    monkeypatch.setenv("MR_RESULTS_DIR", str(tmp_path))
    nb = nbformat.read(NB_DIR / "01_walk_forward_explained.ipynb", as_version=4)
    nbclient.NotebookClient(nb, timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_DIR)}}).execute()
    shown = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    assert "Folds          : 7" in shown and "out-of-sample months" in shown
    figures = sum("image/png" in o.get("data", {}) for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    assert figures >= 5
