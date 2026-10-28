"""
strategy/signals.py — generate long-only momentum signals within clusters.

For each cluster, rank stocks by a momentum feature and select the top N.
Returns a list of selected tickers for the current month.
"""

from __future__ import annotations
import logging

import pandas as pd

from momentum_rebalancer.config import MOMENTUM_FEATURE, TOP_N_PER_CLUSTER, PORTFOLIO_SIZE

logger = logging.getLogger(__name__)


def select_top_momentum(
    month_data: pd.DataFrame,
    momentum_feature: str = MOMENTUM_FEATURE,
    top_n: int = TOP_N_PER_CLUSTER,
    exclude_noise: bool = True,
    portfolio_size: int = PORTFOLIO_SIZE,
) -> pd.DataFrame:
    """
    Within each cluster, select the top_n stocks by momentum_feature.

    Parameters
    ----------
    month_data : DataFrame indexed by Ticker (single month snapshot)
    momentum_feature : column to rank on (higher = better momentum)
    top_n : stocks to select per cluster
    exclude_noise : if True, skip cluster == -1 (HDBSCAN noise)
    portfolio_size : cap on the number of names; the lowest momentum scores are trimmed

    Returns
    -------
    DataFrame of selected stocks with columns: cluster, momentum_score, rank
    """
    if exclude_noise and "cluster" in month_data.columns:
        month_data = month_data[month_data["cluster"] != -1]

    selected = []

    for cluster_id, group in month_data.groupby("cluster"):
        if len(group) < top_n:
            logger.debug(
                "Cluster %s has only %d stocks (< top_n=%d); taking all.",
                cluster_id, len(group), top_n,
            )
            chosen = group
        else:
            chosen = group.nlargest(top_n, momentum_feature)

        chosen = chosen[[momentum_feature, "cluster"]].copy()
        chosen = chosen.rename(columns={momentum_feature: "momentum_score"})
        chosen["rank"] = range(1, len(chosen) + 1)
        selected.append(chosen)

    if not selected:
        logger.warning("No stocks selected — check clustering output.")
        return pd.DataFrame(columns=["cluster", "momentum_score", "rank"])

    result = pd.concat(selected)

    # Hard cap: if more than portfolio_size, trim by momentum score
    if len(result) > portfolio_size:
        logger.debug("Trimming from %d to %d stocks (portfolio_size cap).", len(result), portfolio_size)
        result = result.nlargest(portfolio_size, "momentum_score")

    logger.debug("Selected %d stocks across %d clusters.", len(result), result["cluster"].nunique())
    return result
