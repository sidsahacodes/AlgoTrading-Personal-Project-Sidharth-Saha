"""
dashboard/app.py — Flask dashboard for the momentum rebalancer.
"""

from __future__ import annotations
import logging
import traceback

from flask import Flask, jsonify, render_template

import broker
from clustering.runner import load_clustered, get_current_clusters
from config import (CLUSTER_METHOD, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER,
                    OPTIMISER, BROKER, PORTFOLIO_VALUE)
from features.pipeline import load_modeling_data
from strategy.optimiser import compute_weights
from strategy.rebalancer import compute_trades, trades_to_dataframe
from strategy.signals import select_top_momentum

logger = logging.getLogger(__name__)
app    = Flask(__name__)


def _target_weights() -> dict[str, float]:
    clustered  = load_clustered(CLUSTER_METHOD)
    month_data = get_current_clusters(clustered)
    selected   = select_top_momentum(month_data, MOMENTUM_FEATURE, TOP_N_PER_CLUSTER)
    tickers    = selected.index.tolist()

    if OPTIMISER == "hrp":
        modeling   = load_modeling_data()
        valid      = [t for t in tickers if t in modeling["return_1m"].unstack("Ticker").columns]
        returns_df = modeling["return_1m"].unstack("Ticker").loc[:, valid].dropna(how="all")
        return compute_weights(tickers, returns=returns_df, method="hrp")

    return compute_weights(tickers, method="equal")


# ── Portfolio routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", broker=BROKER, optimiser=OPTIMISER)


@app.route("/api/status")
def api_status():
    from config import DATA_DIR
    return jsonify({
        "broker":           BROKER,
        "optimiser":        OPTIMISER,
        "cluster_method":   CLUSTER_METHOD,
        "momentum_feature": MOMENTUM_FEATURE,
        "top_n":            TOP_N_PER_CLUSTER,
        "modeling_data":    (DATA_DIR / "modeling_data.parquet").exists(),
        "clustered_data":   (DATA_DIR / f"clustered_{CLUSTER_METHOD}.parquet").exists(),
    })


@app.route("/api/portfolio")
def api_portfolio():
    try:
        current = broker.get_holdings()
        value   = broker.get_value()
        value   = float(value) if value else float(PORTFOLIO_VALUE)
        target  = _target_weights()

        all_tickers = sorted(set(current) | set(target))
        rows = [{
            "ticker":      t,
            "current_pct": round(current.get(t, 0.0) * 100, 2),
            "target_pct":  round(target.get(t, 0.0) * 100, 2),
            "delta_pct":   round((target.get(t, 0.0) - current.get(t, 0.0)) * 100, 2),
        } for t in all_tickers]

        return jsonify({
            "portfolio_value": round(value, 2),
            "n_positions":     len(current),
            "n_target":        len(target),
            "holdings":        rows,
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades")
def api_trades():
    try:
        current = broker.get_holdings()
        value   = float(broker.get_value() or PORTFOLIO_VALUE)
        target  = _target_weights()
        trades  = compute_trades(target, current, portfolio_value=value)
        df      = trades_to_dataframe(trades)
        return jsonify({"n_trades": len(trades), "trades": df.to_dict(orient="records")})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/rebalance", methods=["POST"])
def api_rebalance():
    try:
        current = broker.get_holdings()
        value   = float(broker.get_value() or PORTFOLIO_VALUE)
        target  = _target_weights()
        trades  = compute_trades(target, current, portfolio_value=value)

        if not trades:
            return jsonify({"message": "Portfolio already at target. No trades needed.", "fills": []})

        fills = broker.execute(trades)
        return jsonify({"message": f"Submitted {len(fills)} orders to TWS.", "fills": fills})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        from features.pipeline import build_modeling_data
        from clustering.runner import run_clustering
        data = build_modeling_data(force_refresh=True)
        run_clustering(data, method=CLUSTER_METHOD, force_refresh=True)
        return jsonify({"message": "Features and clusters refreshed successfully."})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── Backtest routes ───────────────────────────────────────────────────────────

_backtest_cache = {}

@app.route("/api/backtest")
def api_backtest():
    """
    Run the full backtest and return all data needed for the analytics tab.
    Results are cached in memory after first run.
    """
    try:
        from strategy.backtest import run_backtest, slice_period, PERIODS, compute_metrics

        if _backtest_cache:
            logger.info("Returning cached backtest result.")
            return jsonify(_backtest_cache)
        import pandas as pd

        result = run_backtest()
        r      = result.monthly_returns
        b      = result.benchmark_returns

        # Build period breakdown
        period_stats = {}
        for period in PERIODS:
            r_slice = slice_period(r, period)
            b_slice = slice_period(b, period)
            m = compute_metrics(r_slice, b_slice)
            period_stats[period] = {
                "annualised_return":    pct(m.get("annualised_return")),
                "sharpe":               m.get("sharpe"),
                "sortino":              m.get("sortino"),
                "calmar":               m.get("calmar"),
                "omega":                m.get("omega"),
                "max_drawdown":         pct(m.get("max_drawdown")),
                "win_rate":             pct(m.get("win_rate")),
                "benchmark_ann_return": pct(m.get("benchmark_ann_return")),
                "alpha":                pct(m.get("alpha")),
                "beta":                 m.get("beta"),
                "n_months":             m.get("n_months"),
            }

        # Equity curve (full history, frontend slices by timeframe)
        wealth    = result.metrics["wealth_series"]
        bm_wealth = (1 + b).cumprod()

        equity_data = {
            "dates":     [d.strftime("%Y-%m-%d") for d in wealth.index],
            "strategy":  [round(float(v), 4) for v in wealth.values],
            "benchmark": [round(float(v), 4) for v in bm_wealth.reindex(wealth.index).ffill().values],
        }

        # Drawdown curve
        dd = result.metrics["drawdown_series"]
        drawdown_data = {
            "dates":  [d.strftime("%Y-%m-%d") for d in dd.index],
            "values": [round(float(v) * 100, 2) for v in dd.values],
        }

        # Monthly returns bar data
        monthly_data = {
            "dates":   [d.strftime("%Y-%m-%d") for d in r.index],
            "returns": [round(float(v) * 100, 2) for v in r.values],
        }

        # Cluster performance
        cp = result.cluster_perf
        cluster_data = []
        if not cp.empty:
            for cid, row in cp.iterrows():
                cluster_data.append({
                    "cluster":    int(cid),
                    "avg_return": round(float(row["avg_return"]) * 100, 2),
                    "sharpe":     round(float(row["sharpe"]), 3),
                    "win_rate":   round(float(row["win_rate"]) * 100, 1),
                    "n_trades":   int(row["n_trades"]),
                })

        # Avg trades per month
        avg_trades = len(result.positions) / result.metrics["n_months"] if result.metrics.get("n_months") else 0

        response = {
            "period_stats":   period_stats,
            "equity":         equity_data,
            "drawdown":       drawdown_data,
            "monthly":        monthly_data,
            "cluster_perf":   cluster_data,
            "avg_trades_per_month": round(avg_trades, 1),
            "full_metrics":   {
                k: v for k, v in result.metrics.items()
                if not hasattr(v, "values")
            },
        }
        _backtest_cache.update(response)
        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def pct(v):
    return round(float(v) * 100, 2) if v is not None else None