"""
cli.py — command-line entry point (`mr ...` or `python -m momentum_rebalancer ...`).

  mr build-features [--force]   download prices + build modeling_data
  mr cluster [--force]          cluster each month
  mr pipeline [--force]         build-features + cluster
  mr signals [--save]           this month's selected stocks (+ health checks)
  mr trades  [--save]           pending trades (dry run, no orders)
  mr rebalance                  execute trades via the configured broker
  mr backtest [--save]          point-in-time backtest, net of costs (--cost-bps, --delist-return)
  mr walkforward [--save]       out-of-sample train/validation/test evaluation of the hyperparameters
  mr coverage                   how many index members have prices (the survivorship gap)
  mr regime                     regime indicators + health report (no changes made)
  mr regimes                    list regime presets
  mr broker-check               is the configured broker reachable? (TWS for ibkr)
  mr crsp-check                 verify WRDS/CRSP tables and columns before price_source: crsp
  mr dashboard                  start the web dashboard

Global option:  mr --regime stress <command>    (same as env MR_REGIME=stress)

NOTE: nothing from momentum_rebalancer.config is imported at module level, so
that --regime can be applied before the configuration is loaded.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def C_source() -> str:
    from momentum_rebalancer import config as C
    return C.PRICE_SOURCE


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner() -> None:
    from momentum_rebalancer import config as C
    from momentum_rebalancer.storage.fingerprint import stale_warnings

    print(f"\n  regime={C.REGIME}  broker={C.BROKER}  optimiser={C.OPTIMISER}  "
          f"clusters={C.CLUSTER_METHOD}  momentum={C.MOMENTUM_FEATURE}  data={C.PRICE_SOURCE}")
    if C.REGIME != "baseline":
        print(f"  {C.REGIME_DESCRIPTION}")
    if C.REGIME_OVERRIDES:
        print(f"  overrides in effect: {C.REGIME_OVERRIDES}")
    for w in stale_warnings(C.SETTINGS, C.DATA_DIR):
        print(f"  ! {w}")


def _cluster_health(month_data) -> list:
    from momentum_rebalancer import config as C
    from momentum_rebalancer.validation.checks import check_clusters
    return check_clusters(month_data, C.TOP_N_PER_CLUSTER, C.N_CLUSTERS,
                          C.CLUSTER_METHOD, C.CLUSTER_MAX_SHARE)


def _tws_ready() -> None:
    """Raise a clear error if the broker is IBKR and TWS cannot be reached."""
    from momentum_rebalancer import config as C
    if C.BROKER != "ibkr":
        return
    from momentum_rebalancer.broker import ibkr
    if not ibkr.is_connected():
        raise ConnectionError(
            f"Cannot reach TWS at {C.TWS_HOST}:{C.TWS_PORT}. Open TWS, log in, enable the API "
            f"(docs: src/momentum_rebalancer/broker/README.md), or set MR_BROKER=paper."
        )


def _save_csv(df, name: str) -> None:
    from momentum_rebalancer import config as C
    path = C.RESULTS_DIR / name
    df.to_csv(path)
    print(f"\n  saved -> {path}")


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_build_features(args):
    from momentum_rebalancer.features.pipeline import build_modeling_data
    _banner()
    data = build_modeling_data(force_refresh=args.force)
    print(f"\n[ok] Modeling data ready: {len(data):,} rows, {data.shape[1]} features")
    print(f"   Date range: {data.index.get_level_values('Date').min().date()} -> "
          f"{data.index.get_level_values('Date').max().date()}")
    print(f"   Unique tickers: {data.index.get_level_values('Ticker').nunique()}")

    import pandas as pd
    from momentum_rebalancer.data_ingestion.loader import PIT_MEMBERS_PATH, read_price_coverage
    from momentum_rebalancer.validation.checks import check_pit_freshness
    if PIT_MEMBERS_PATH.exists():
        pit = pd.read_parquet(PIT_MEMBERS_PATH)
        if C_source() == "crsp":
            print(f"   CRSP sample ends {data.index.get_level_values('Date').max().date()} "
                  f"(limited by CRSP prices and its S&P 500 list; see docs/DATA_SOURCES.md)")
        else:
            for issue in check_pit_freshness(pit):
                print(f"   {issue}")
    cov = read_price_coverage()
    if cov:
        print(f"   Prices: {cov['with_prices']} of {cov['requested']} index members in the window "
              f"({len(cov['missing'])} missing, {len(cov['from_external'])} from vendor files). "
              f"Run `mr coverage` to see what the missing names cost.")


def cmd_cluster(args):
    from momentum_rebalancer.config import CLUSTER_METHOD
    from momentum_rebalancer.features.pipeline import load_modeling_data
    from momentum_rebalancer.clustering.runner import run_clustering
    _banner()
    data      = load_modeling_data()
    clustered = run_clustering(data, method=CLUSTER_METHOD, force_refresh=args.force)
    print(f"\n[ok] Clustering complete ({CLUSTER_METHOD})")
    print(f"   Clusters found: {clustered['cluster'].nunique()}")
    latest = clustered.index.get_level_values("Date").max()
    print(f"   Latest month:   {latest.date()}")
    counts = clustered.xs(latest, level="Date")["cluster"].value_counts().sort_index()
    for cid, n in counts.items():
        print(f"   Cluster {cid}: {n} stocks")


def cmd_signals(args):
    from momentum_rebalancer.clustering.runner import get_current_clusters, load_clustered
    from momentum_rebalancer.config import CLUSTER_METHOD, PORTFOLIO_SIZE
    from momentum_rebalancer.strategy.targets import current_selection
    from momentum_rebalancer.validation.checks import check_selection
    _banner()
    clustered = load_clustered(CLUSTER_METHOD)
    latest    = clustered.index.get_level_values("Date").max()
    selected  = current_selection()
    print(f"\n[ok] Signals for {latest.date()}")
    print(f"   {len(selected)} stocks selected\n")
    print(selected.sort_values(["cluster", "momentum_score"], ascending=[True, False]).to_string())
    issues = _cluster_health(get_current_clusters(clustered)) + check_selection(selected, PORTFOLIO_SIZE)
    if issues:
        print("\n  Health checks:")
        for i in issues:
            print(f"   {i}")
    if args.save:
        from momentum_rebalancer.config import REGIME
        _save_csv(selected.sort_values(["cluster", "momentum_score"], ascending=[True, False]),
                  f"signals_{latest:%Y-%m-%d}_{REGIME}.csv")


def cmd_trades(args):
    from momentum_rebalancer.strategy.rebalancer import trades_to_dataframe
    from momentum_rebalancer.strategy.targets import plan_rebalance
    _banner()
    _tws_ready()
    _, _, _, value, trades = plan_rebalance()
    if not trades:
        print("\n[ok] Portfolio is already at target. No trades needed.")
        return
    df = trades_to_dataframe(trades)
    print(f"\n[ok] {len(trades)} trades pending (portfolio value: ${value:,.0f})\n")
    print(df.to_string(index=False))
    if args.save:
        import pandas as pd
        from momentum_rebalancer.config import REGIME
        _save_csv(df.set_index("ticker"), f"trades_{pd.Timestamp.today():%Y-%m-%d}_{REGIME}.csv")
    print("\n   (shares are sized by the broker at execution; dry run shows weights only)")


def cmd_rebalance(args):
    from momentum_rebalancer import broker, config as C
    from momentum_rebalancer.strategy.targets import plan_rebalance
    _banner()
    _tws_ready()
    _, _, _, _, trades = plan_rebalance()
    if not trades:
        print("\n[ok] Portfolio already at target.")
        return

    print(f"\n  About to execute {len(trades)} trades on [{C.BROKER}]")
    if C.REGIME != "baseline":
        print(f"\n  !! Non-baseline regime '{C.REGIME}' is active: {C.REGIME_OVERRIDES}")
        print("  !! Regime presets are untested hypotheses (configs/regimes.yaml).")
        if input(f"  Type '{C.REGIME}' to proceed: ").strip() != C.REGIME:
            print("  Aborted.")
            return
    if input("  Confirm? [y/N] ").strip().lower() != "y":
        print("  Aborted.")
        return

    fills = broker.execute(trades)
    print(f"\n[ok] {len(fills)} fills executed.")


def cmd_regime(args):
    from momentum_rebalancer import config as C
    from momentum_rebalancer.clustering.runner import get_current_clusters, load_clustered
    from momentum_rebalancer.features.pipeline import load_modeling_data
    from momentum_rebalancer.storage.fingerprint import stale_warnings
    from momentum_rebalancer.strategy import regime as R
    from momentum_rebalancer.strategy.targets import current_selection, history_for
    from momentum_rebalancer.validation.checks import check_modeling_data, check_selection

    modeling  = load_modeling_data()
    clustered = load_clustered(C.CLUSTER_METHOD)
    selected  = current_selection()
    from momentum_rebalancer.data_ingestion.loader import PIT_MEMBERS_PATH
    from momentum_rebalancer.validation.checks import check_pit_freshness
    import pandas as pd
    pit_issues = check_pit_freshness(pd.read_parquet(PIT_MEMBERS_PATH)) if PIT_MEMBERS_PATH.exists() else []
    issues = (pit_issues
              + check_modeling_data(modeling, C.FEATURE_COLS)
              + _cluster_health(get_current_clusters(clustered))
              + check_selection(selected, C.PORTFOLIO_SIZE))

    spy, vix = R.fetch_market()
    holdings_returns = history_for(selected.index.tolist(), None, lookback_months=None)
    flags = R.build_flags(
        spy, vix, holdings_returns, modeling,
        spy_dd_caution=C.SPY_DD_CAUTION, spy_dd_stress=C.SPY_DD_STRESS,
        vix_caution=C.VIX_CAUTION, vix_stress=C.VIX_STRESS, corr_jump=C.CORR_JUMP,
    )
    print(R.format_report(C.REGIME, C.REGIME_OVERRIDES, flags, issues,
                          stale_warnings(C.SETTINGS, C.DATA_DIR)))


def cmd_regimes(args):
    from momentum_rebalancer import config as C
    print(f"\n  Active: {C.REGIME}\n")
    for name, spec in C.available_regimes().items():
        print(f"  {name}")
        print(f"     {spec.get('description', '')}")
        if spec.get("overrides"):
            print(f"     overrides: {spec['overrides']}")
        if spec.get("requires"):
            print(f"     re-run needed: {spec['requires']}")
    print("\n  Presets other than 'baseline' are untested hypotheses.")


def cmd_broker_check(args):
    from momentum_rebalancer import config as C
    if C.BROKER != "ibkr":
        print(f"\n  broker={C.BROKER}: no external connection needed.")
        return
    _tws_ready()
    print(f"\n  TWS reachable at {C.TWS_HOST}:{C.TWS_PORT} (broker=ibkr).")


def cmd_backtest(args):
    import pandas as pd
    from momentum_rebalancer import config as C
    from momentum_rebalancer.strategy.backtest import run_backtest
    _banner()
    kw = {}
    if args.cost_bps is not None:
        kw["cost_bps"] = args.cost_bps
    if args.delist_return is not None:
        kw["delisting_return"] = args.delist_return
    res = run_backtest(**kw)
    m, idx = res.metrics, res.monthly_returns.index
    cost = kw.get("cost_bps", C.COST_BPS)
    pct = lambda x: f"{x * 100:.1f}%" if x is not None else "n/a"
    print(f"\n  Backtest: {len(idx)} months earned, {idx.min():%Y-%m} -> {idx.max():%Y-%m}   (cost {cost:.0f} bps per unit traded)")
    print(f"  Net annualised return   {pct(m['annualised_return'])}    gross {pct(m['gross_annualised_return'])}    cost drag {pct(m['annual_cost_drag'])}/yr")
    print(f"  Sharpe {m['sharpe']:.2f}   Sortino {m['sortino']:.2f}   max drawdown {pct(m['max_drawdown'])}   win rate {pct(m['win_rate'])}")
    print(f"  Avg monthly traded      {pct(m['avg_monthly_traded'])} of NAV")
    if "benchmark_ann_return" in m:
        print(f"  SPY annualised return   {pct(m['benchmark_ann_return'])}   excess return {pct(m['excess_return'])}   "
              f"alpha {pct(m['alpha'])} (CAPM, rf=0)   beta {m['beta']:.2f}")
    else:
        print("  SPY benchmark unavailable (download failed): benchmark metrics omitted")
    d = res.diagnostics
    print(f"  Held names with no next-month price: {d['delisted']} (assumed {kw.get('delisting_return', C.DELISTING_RETURN):.0%}); data gaps: {d['data_gaps']}; weight fallbacks: {d['weight_fallbacks']}")
    age = (pd.Timestamp.today().normalize() - idx.max()).days
    if age > 75:
        print(f"  ! Backtest ends {idx.max():%Y-%m}, {age} days ago. Refresh with `mr build-features --force` then `mr cluster --force`.")
    if args.save:
        out = pd.DataFrame({"net": res.monthly_returns, "gross": res.gross_returns,
                            "benchmark": res.benchmark_returns, "traded": res.turnover})
        _save_csv(out.rename_axis("earned_in"), f"backtest_{C.REGIME}_{pd.Timestamp.today():%Y-%m-%d}.csv")


def cmd_walkforward(args):
    import pandas as pd
    from momentum_rebalancer import config as C
    from momentum_rebalancer.features.pipeline import load_modeling_data, load_monthly_returns
    from momentum_rebalancer.strategy.backtest import _load_benchmark
    from momentum_rebalancer.strategy.simulate import SimConfig
    from momentum_rebalancer.strategy.walkforward import format_walk_forward, run_walk_forward
    _banner()
    base = SimConfig(
        momentum_feature=C.MOMENTUM_FEATURE, top_n=C.TOP_N_PER_CLUSTER, portfolio_size=C.PORTFOLIO_SIZE,
        optimiser=C.OPTIMISER, hrp_lookback_months=C.HRP_LOOKBACK_MONTHS, exposure_scalar=C.EXPOSURE_SCALAR,
        min_trade_pct=C.MIN_TRADE_PCT,
        cost_bps=args.cost_bps if args.cost_bps is not None else C.COST_BPS,
        delisting_return=args.delist_return if args.delist_return is not None else C.DELISTING_RETURN,
    )
    baseline = {"momentum_feature": C.MOMENTUM_FEATURE, "top_n_per_cluster": C.TOP_N_PER_CLUSTER,
                "n_clusters": C.N_CLUSTERS}
    res = run_walk_forward(load_modeling_data(), load_monthly_returns(), C.load_walk_forward(), base, baseline,
                           C.CLUSTER_METHOD, C.FEATURE_COLS, benchmark_loader=_load_benchmark)
    print(format_walk_forward(res))
    if args.save:
        stamp = f"{C.REGIME}_{pd.Timestamp.today():%Y-%m-%d}"
        _save_csv(res.folds, f"walkforward_folds_{stamp}.csv")
        _save_csv(pd.DataFrame({"walk_forward": res.oos_returns, "fixed_baseline": res.baseline_returns})
                  .rename_axis("earned_in"), f"walkforward_oos_{stamp}.csv")


def cmd_coverage(args):
    import pandas as pd
    from momentum_rebalancer import config as C
    from momentum_rebalancer.data_cleaning.coverage import coverage_by_month, missing_member_months
    from momentum_rebalancer.data_ingestion.loader import PIT_MEMBERS_PATH, RAW_PRICES_PATH, read_price_coverage
    from momentum_rebalancer.features.returns import monthly_adjclose_panel
    from momentum_rebalancer.validation.checks import check_pit_freshness

    if not (PIT_MEMBERS_PATH.exists() and RAW_PRICES_PATH.exists()):
        raise FileNotFoundError("Run `mr build-features` first (needs pit_members and prices_raw).")
    _banner()
    pit = pd.read_parquet(PIT_MEMBERS_PATH)
    adj = monthly_adjclose_panel(pd.read_parquet(RAW_PRICES_PATH))
    cov = coverage_by_month(pit, adj, start=adj.index.min())
    miss = missing_member_months(pit, adj, start=adj.index.min())

    print(f"\n  Membership list: {pit.index.min():%Y-%m-%d} -> {pit.index.max():%Y-%m-%d}")
    for i in check_pit_freshness(pit):
        print(f"  {i}")
    print(f"  Index members with a price, by month: mean {cov['share'].mean():.1%}, worst {cov['share'].min():.1%} ({cov['share'].idxmin():%Y-%m})")
    low = cov[cov["share"] < 0.95]
    print(f"  Months below 95% coverage: {len(low)} of {len(cov)}")
    if len(miss):
        print(f"\n  Members with no price in the most months ({len(miss)} tickers affected):")
        print(miss.head(30).to_string())
    saved = read_price_coverage()
    if saved:
        print(f"\n  At download: {saved['with_prices']} of {saved['requested']} requested tickers had prices; "
              f"{len(saved['from_external'])} filled from vendor files.")
    print(f"\n  Every month below 100% is index membership the backtest cannot see. To close the gap, put vendor\n"
          f"  price files for the missing tickers in {C.EXTERNAL_PRICES_DIR} (see docs/DATA_SOURCES.md), then\n"
          f"  `mr build-features --force`.")
    if args.save:
        _save_csv(cov, f"coverage_{pd.Timestamp.today():%Y-%m-%d}.csv")


def cmd_crsp_check(args):
    from momentum_rebalancer import config as C
    from momentum_rebalancer.data_ingestion import crsp
    cfg = C.load_crsp_settings()
    db = crsp.connect(cfg)
    results = crsp.run_checks(db, cfg)
    print("\n  CRSP / WRDS check")
    for r in results:
        print(f"  [{'ok' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
    if not all(r.ok for r in results):
        raise ConnectionError("Some CRSP checks failed. Fix configs/crsp.yaml (names differ on some subscriptions) "
                              "or your WRDS access, then re-run.")
    print("\n  All checks passed. Use `MR_PRICE_SOURCE=crsp MR_DATA_ROOT=<research folder> mr build-features --force`.")


def cmd_dashboard(args):
    from momentum_rebalancer.config import DASHBOARD_HOST, DASHBOARD_PORT, DEBUG
    from momentum_rebalancer.dashboard.app import app
    print(f"\n  Dashboard running at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DEBUG)


def cmd_pipeline(args):
    """Convenience: build features + cluster in one shot."""
    args.force = getattr(args, "force", False)
    cmd_build_features(args)
    cmd_cluster(args)


COMMANDS = {
    "build-features": cmd_build_features,
    "cluster":        cmd_cluster,
    "signals":        cmd_signals,
    "trades":         cmd_trades,
    "rebalance":      cmd_rebalance,
    "backtest":       cmd_backtest,
    "walkforward":    cmd_walkforward,
    "coverage":       cmd_coverage,
    "regime":         cmd_regime,
    "regimes":        cmd_regimes,
    "broker-check":   cmd_broker_check,
    "crsp-check":     cmd_crsp_check,
    "dashboard":      cmd_dashboard,
    "pipeline":       cmd_pipeline,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mr", description="Cluster-momentum rebalancer")
    parser.add_argument("--regime", help="regime preset from configs/regimes.yaml (default: strategy.yaml)")
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--force", action="store_true", help="force refresh (ignore cache)")
    parser.add_argument("--save", action="store_true", help="signals/trades/backtest/walkforward/coverage: also write a CSV to data/results/")
    parser.add_argument("--cost-bps", type=float, default=None, help="backtest/walkforward: override cost per unit traded (bps)")
    parser.add_argument("--delist-return", type=float, default=None, help="backtest/walkforward: override the assumed return of a held stock with no next-month price")
    args = parser.parse_args(argv)

    if args.regime:
        os.environ["MR_REGIME"] = args.regime      # must precede any config import

    try:
        COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"\n[error] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")


if __name__ == "__main__":
    main()
