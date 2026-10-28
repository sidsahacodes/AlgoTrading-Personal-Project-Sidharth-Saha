"""
broker/__init__.py — swap between paper and IBKR via config.BROKER.
"""

from config import BROKER

if BROKER == "ibkr":
    from broker.ibkr import (
        get_current_holdings as get_holdings,
        get_portfolio_value  as get_value,
        execute_trades       as execute,
    )
elif BROKER == "paper":
    from broker.paper import (
        get_current_holdings as get_holdings,
        get_portfolio_value  as get_value,
        execute_trades       as execute,
    )
else:
    raise ValueError(f"Unknown broker '{BROKER}'. Set config.BROKER to 'paper' or 'ibkr'.")

__all__ = ["get_holdings", "get_value", "execute"]
