"""
broker/paper.py — paper trading simulation (no IBKR required).

Maintains a local JSON ledger of positions and simulates order fills
at the last known price from yfinance. Useful for testing the full
pipeline before connecting real IBKR credentials.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

import yfinance as yf

from config import DATA_DIR, PORTFOLIO_VALUE
from strategy.rebalancer import Trade

logger = logging.getLogger(__name__)

LEDGER_PATH = DATA_DIR / "paper_ledger.json"


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {
        "portfolio_value": PORTFOLIO_VALUE,
        "cash":            PORTFOLIO_VALUE,
        "positions":       {},   # {ticker: {shares, avg_cost}}
        "history":         [],
    }


def _save_ledger(ledger: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=str))


def get_current_price(ticker: str) -> float:
    """Fetch latest close price from yfinance."""
    data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"Could not fetch price for {ticker}")
    return float(data["Close"].iloc[-1])


def get_portfolio_value(ledger: dict | None = None) -> float:
    """Mark-to-market portfolio value: cash + sum(shares * current_price)."""
    ledger = ledger or _load_ledger()
    total = ledger["cash"]
    for ticker, pos in ledger["positions"].items():
        try:
            price = get_current_price(ticker)
            total += pos["shares"] * price
        except Exception:
            # Fall back to cost basis if price unavailable
            total += pos["shares"] * pos["avg_cost"]
    return total


def get_current_holdings(ledger: dict | None = None) -> dict[str, float]:
    """Return {ticker: weight} for current positions (weights sum to ~1)."""
    ledger = ledger or _load_ledger()
    total  = get_portfolio_value(ledger)
    if total == 0:
        return {}
    holdings = {}
    for ticker, pos in ledger["positions"].items():
        try:
            price = get_current_price(ticker)
        except Exception:
            price = pos["avg_cost"]
        holdings[ticker] = (pos["shares"] * price) / total
    return holdings


def execute_trades(trades: list[Trade]) -> list[dict]:
    """
    Simulate execution of a trade list.
    Fills at current market price; updates ledger.
    Returns a list of fill records.
    """
    ledger = _load_ledger()
    total_value = get_portfolio_value(ledger)
    fills = []

    for trade in trades:
        try:
            price = get_current_price(trade.ticker)
        except Exception as e:
            logger.error("Could not get price for %s: %s", trade.ticker, e)
            continue

        dollar_amount = abs(trade.delta_pct) * total_value
        shares = int(dollar_amount / price)

        if shares == 0:
            continue

        if trade.action == "SELL":
            held = ledger["positions"].get(trade.ticker, {}).get("shares", 0)
            shares = min(shares, held)   # can't sell more than we have
            if shares == 0:
                continue
            ledger["cash"] += shares * price
            if trade.ticker in ledger["positions"]:
                ledger["positions"][trade.ticker]["shares"] -= shares
                if ledger["positions"][trade.ticker]["shares"] <= 0:
                    del ledger["positions"][trade.ticker]

        else:  # BUY
            cost = shares * price
            if cost > ledger["cash"]:
                shares = int(ledger["cash"] / price)
                cost   = shares * price
            if shares == 0:
                continue
            ledger["cash"] -= cost
            if trade.ticker in ledger["positions"]:
                old_shares = ledger["positions"][trade.ticker]["shares"]
                old_cost   = ledger["positions"][trade.ticker]["avg_cost"]
                new_shares = old_shares + shares
                ledger["positions"][trade.ticker] = {
                    "shares":   new_shares,
                    "avg_cost": (old_shares * old_cost + cost) / new_shares,
                }
            else:
                ledger["positions"][trade.ticker] = {
                    "shares":   shares,
                    "avg_cost": price,
                }

        fill = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticker":    trade.ticker,
            "action":    trade.action,
            "shares":    shares,
            "price":     price,
            "value":     shares * price,
        }
        fills.append(fill)
        ledger["history"].append(fill)
        logger.info("%s %d %s @ $%.2f", trade.action, shares, trade.ticker, price)

    _save_ledger(ledger)
    return fills
