import pandas as pd

from momentum_rebalancer.strategy.signals import select_top_momentum


def _month(sizes, seed=0):
    rows = []
    for cid, n in sizes.items():
        for i in range(n):
            rows.append({"Ticker": f"C{cid}_{i}", "cluster": cid, "return_3m": i * 0.01 + cid * 0.001})
    return pd.DataFrame(rows).set_index("Ticker")


def test_top_n_per_cluster_is_highest_momentum():
    sel = select_top_momentum(_month({0: 10, 1: 10}), "return_3m", top_n=3)
    assert len(sel) == 6
    assert set(sel[sel.cluster == 0].index) == {"C0_9", "C0_8", "C0_7"}


def test_noise_cluster_excluded():
    sel = select_top_momentum(_month({-1: 10, 0: 10}), "return_3m", top_n=3)
    assert set(sel.cluster) == {0}


def test_small_cluster_takes_all_members():
    sel = select_top_momentum(_month({0: 2, 1: 10}), "return_3m", top_n=4)
    assert (sel.cluster == 0).sum() == 2 and (sel.cluster == 1).sum() == 4


def test_portfolio_size_cap_trims_lowest_scores():
    sel = select_top_momentum(_month({c: 10 for c in range(6)}), "return_3m", top_n=4)   # 24 -> cap 20
    assert len(sel) == 20


def test_empty_input_returns_empty_frame():
    sel = select_top_momentum(_month({-1: 5}), "return_3m", top_n=3)
    assert sel.empty
