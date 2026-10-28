"""
broker/ibkr.py — IBKR broker via ib_insync connecting directly to TWS.

Requirements:
  - TWS open and logged into paper account
  - API enabled: File > Global Configuration > API > Settings
      Socket port 7497, Enable ActiveX ticked, Read-Only API unticked
  - pip install ib_insync
  - config.py: BROKER = "ibkr"
"""

from __future__ import annotations
import asyncio
import logging
import math

import yfinance as yf
from ib_insync import IB, Stock, MarketOrder

import config
from strategy.rebalancer import Trade

logger = logging.getLogger(__name__)


def _connect() -> IB:
    """Connect to TWS, creating a fresh event loop if needed (Flask threading)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    import random
    ib = IB()
    # Use a random client ID each time to avoid "already in use" errors
    # when Flask makes simultaneous requests
    client_id = random.randint(10, 999)
    ib.connect(config.TWS_HOST, config.TWS_PORT, clientId=client_id)
    logger.info("Connected to TWS %s:%s (clientId=%d)", config.TWS_HOST, config.TWS_PORT, client_id)
    return ib


def _net_liq(ib: IB) -> float:
    """Net liquidation value from an already-open IB connection."""
    for av in ib.accountValues():
        if av.tag == "NetLiquidation" and av.currency == "USD":
            val = float(av.value)
            if val > 0:
                return val
    logger.warning("NetLiquidation not found — using config fallback.")
    return float(config.PORTFOLIO_VALUE)


def _get_price(ib: IB, contract) -> float | None:
    """
    Try to get a valid price from TWS market data.
    Falls back to yfinance if TWS returns NaN (e.g. outside market hours).
    """
    md = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)

    # Try each price field in order of preference
    for field in [md.last, md.close, md.bid, md.ask]:
        if field and not math.isnan(float(field)) and float(field) > 0:
            return float(field)

    # Fallback: yfinance last close
    logger.info("TWS price unavailable for %s — falling back to yfinance.", contract.symbol)
    try:
        data = yf.download(contract.symbol, period="5d", progress=False, auto_adjust=True)
        if not data.empty:
            price = float(data["Close"].dropna().iloc[-1])
            if price > 0:
                logger.info("yfinance price for %s: $%.2f", contract.symbol, price)
                return price
    except Exception as e:
        logger.error("yfinance fallback failed for %s: %s", contract.symbol, e)

    return None


# ── Public interface ──────────────────────────────────────────────────────────

def get_portfolio_value() -> float:
    ib = _connect()
    try:
        return _net_liq(ib)
    finally:
        ib.disconnect()


def get_current_holdings() -> dict[str, float]:
    """Return {ticker: weight} based on current TWS portfolio positions."""
    ib = _connect()
    try:
        total = _net_liq(ib)
        if total == 0:
            return {}

        holdings = {}
        for item in ib.portfolio():
            symbol    = item.contract.symbol
            mkt_value = abs(item.marketValue)
            if mkt_value and not math.isnan(mkt_value) and mkt_value > 0:
                holdings[symbol] = mkt_value / total

        logger.info("Loaded %d positions from TWS.", len(holdings))
        return holdings
    finally:
        ib.disconnect()


def execute_trades(trades: list[Trade]) -> list[dict]:
    """Submit trades as market orders to TWS. Returns list of fill records."""
    ib = _connect()
    fills = []

    try:
        total = _net_liq(ib)

        for trade in trades:
            # Qualify contract
            contract  = Stock(trade.ticker, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logger.error("Cannot qualify %s — skipping.", trade.ticker)
                continue
            contract = qualified[0]

            # Get price (TWS with yfinance fallback)
            price = _get_price(ib, contract)
            if price is None:
                logger.warning("No valid price for %s — skipping.", trade.ticker)
                continue

            # Size the order
            dollar_amount = abs(trade.delta_pct) * total
            shares = int(dollar_amount / price)
            if shares == 0:
                logger.info("%s trade too small (<1 share) — skipping.", trade.ticker)
                continue

            # Place market order
            order  = MarketOrder(trade.action, shares)
            placed = ib.placeOrder(contract, order)
            ib.sleep(1)

            status = placed.orderStatus.status
            fill   = {
                "ticker": trade.ticker,
                "action": trade.action,
                "shares": shares,
                "price":  round(price, 2),
                "value":  round(shares * price, 2),
                "status": status,
            }
            fills.append(fill)
            logger.info(
                "%s %d %s @ $%.2f [%s]",
                trade.action, shares, trade.ticker, price, status,
            )

    finally:
        ib.disconnect()

    return fills


def is_connected() -> bool:
    try:
        ib = _connect()
        ib.disconnect()
        return True
    except Exception:
        return False