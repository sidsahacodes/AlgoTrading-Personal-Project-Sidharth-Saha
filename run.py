"""
run.py — CLI entry point for the momentum rebalancer.

Usage
-----
  python run.py build-features          # download prices + build modeling_data
  python run.py cluster                 # run clustering on modeling_data
  python run.py signals                 # print this month's selected stocks
  python run.py trades                  # print pending trades
  python run.py rebalance               # execute rebalance (paper or IBKR)
  python run.py dashboard               # start the web dashboard
  python run.py pipeline                # build-features + cluster in one shot
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_build_features(args):
    from features.pipeline import build_modeling_data
    data = build_modeling_data(force_refresh=args.force)
    print(f"\n✓  Modeling data ready: {len(data):,} rows, {data.shape[1]} features")
    print(f"   Date range: {data.index.get_level_values('Date').min().date()} → "
          f"{data.index.get_level_values('Date').max().date()}")
    print(f"   Unique tickers: {data.index.get_level_values('Ticker').nunique()}")


def cmd_cluster(args):
    from config import CLUSTER_METHOD
    from features.pipeline import load_modeling_data
    from clustering.runner import run_clustering
    data      = load_modeling_data()
    clustered = run_clustering(data, method=CLUSTER_METHOD, force_refresh=args.force)
    print(f"\n✓  Clustering complete ({CLUSTER_METHOD})")
    print(f"   Clusters found: {clustered['cluster'].nunique()}")
    latest = clustered.index.get_level_values("Date").max()
    print(f"   Latest month:   {latest.date()}")
    counts = clustered.xs(latest, level="Date")["cluster"].value_counts().sort_index()
    for cid, n in counts.items():
        print(f"   Cluster {cid}: {n} stocks")


def cmd_signals(args):
    from config import CLUSTER_METHOD, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER
    from clustering.runner import load_clustered, get_current_clusters
    from strategy.signals import select_top_momentum

    clustered  = load_clustered(CLUSTER_METHOD)
    month_data = get_current_clusters(clustered)
    selected   = select_top_momentum(month_data, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER)

    latest = clustered.index.get_level_values("Date").max()
    print(f"\n✓  Signals for {latest.date()}")
    print(f"   {len(selected)} stocks selected\n")
    print(selected.sort_values(["cluster", "momentum_score"], ascending=[True, False])
          .to_string())


def cmd_trades(args):
    import broker
    from config import CLUSTER_METHOD, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER, OPTIMISER
    from clustering.runner import load_clustered, get_current_clusters
    from features.pipeline import load_modeling_data
    from strategy.signals import select_top_momentum
    from strategy.optimiser import compute_weights
    from strategy.rebalancer import compute_trades, trades_to_dataframe

    clustered  = load_clustered(CLUSTER_METHOD)
    month_data = get_current_clusters(clustered)
    selected   = select_top_momentum(month_data, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER)
    tickers    = selected.index.tolist()

    if OPTIMISER == "hrp":
        modeling   = load_modeling_data()
        returns_df = modeling["return_1m"].unstack("Ticker").loc[:, tickers].dropna(how="all")
        target     = compute_weights(tickers, returns=returns_df, method="hrp")
    else:
        target = compute_weights(tickers, method="equal")

    current = broker.get_holdings()
    value   = broker.get_value()
    trades  = compute_trades(target, current, portfolio_value=value)

    if not trades:
        print("\n✓  Portfolio is already at target. No trades needed.")
        return

    df = trades_to_dataframe(trades)
    print(f"\n✓  {len(trades)} trades pending (portfolio value: ${value:,.0f})\n")
    print(df.to_string(index=False))


def cmd_rebalance(args):
    import broker
    from config import CLUSTER_METHOD, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER, OPTIMISER
    from clustering.runner import load_clustered, get_current_clusters
    from features.pipeline import load_modeling_data
    from strategy.signals import select_top_momentum
    from strategy.optimiser import compute_weights
    from strategy.rebalancer import compute_trades

    clustered  = load_clustered(CLUSTER_METHOD)
    month_data = get_current_clusters(clustered)
    selected   = select_top_momentum(month_data, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER)
    tickers    = selected.index.tolist()

    if OPTIMISER == "hrp":
        modeling   = load_modeling_data()
        returns_df = modeling["return_1m"].unstack("Ticker").loc[:, tickers].dropna(how="all")
        target     = compute_weights(tickers, returns=returns_df, method="hrp")
    else:
        target = compute_weights(tickers, method="equal")

    current = broker.get_holdings()
    value   = broker.get_value()
    trades  = compute_trades(target, current, portfolio_value=value)

    if not trades:
        print("\n✓  Portfolio already at target.")
        return

    print(f"\n  About to execute {len(trades)} trades on [{broker.BROKER if hasattr(broker, 'BROKER') else 'configured broker'}]")
    confirm = input("  Confirm? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    fills = broker.execute(trades)
    print(f"\n✓  {len(fills)} fills executed.")


def cmd_dashboard(args):
    from config import DASHBOARD_HOST, DASHBOARD_PORT, DEBUG
    from dashboard.app import app
    print(f"\n  Dashboard running at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DEBUG)


def cmd_pipeline(args):
    """Convenience: build features + cluster in one shot."""
    args.force = getattr(args, "force", False)
    cmd_build_features(args)
    cmd_cluster(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

COMMANDS = {
    "build-features": cmd_build_features,
    "cluster":        cmd_cluster,
    "signals":        cmd_signals,
    "trades":         cmd_trades,
    "rebalance":      cmd_rebalance,
    "dashboard":      cmd_dashboard,
    "pipeline":       cmd_pipeline,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py", description="Momentum Rebalancer")
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--force", action="store_true", help="Force refresh (ignore cache)")
    args = parser.parse_args()

    try:
        COMMANDS[args.command](args)
    except FileNotFoundError as e:
        print(f"\n✗  {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
