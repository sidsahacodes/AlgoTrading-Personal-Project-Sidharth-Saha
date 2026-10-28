import pytest

from momentum_rebalancer import config as C
from momentum_rebalancer.broker import paper
from momentum_rebalancer.strategy.rebalancer import Trade


@pytest.fixture(autouse=True)
def fresh_ledger(monkeypatch):
    paper.LEDGER_PATH.unlink(missing_ok=True)
    monkeypatch.setattr(paper, "get_current_price", lambda t: 100.0)
    yield
    paper.LEDGER_PATH.unlink(missing_ok=True)


def _trade(t, action, delta):
    return Trade(ticker=t, action=action, shares=0, target_pct=max(delta, 0), current_pct=0, delta_pct=delta)


def test_buy_then_full_sell_roundtrip():
    fills = paper.execute_trades([_trade("AAA", "BUY", 0.10)])
    assert fills[0]["shares"] == int(0.10 * C.PORTFOLIO_VALUE / 100)
    holdings = paper.get_current_holdings()
    assert holdings["AAA"] == pytest.approx(0.10, abs=1e-6)

    fills = paper.execute_trades([_trade("AAA", "SELL", -0.10)])
    assert fills and "AAA" not in paper.get_current_holdings()
    assert paper.get_portfolio_value() == pytest.approx(C.PORTFOLIO_VALUE)


def test_cannot_oversell():
    paper.execute_trades([_trade("AAA", "BUY", 0.05)])
    held = int(0.05 * C.PORTFOLIO_VALUE / 100)
    fills = paper.execute_trades([_trade("AAA", "SELL", -0.50)])
    assert fills[0]["shares"] == held                      # capped at shares held
    assert "AAA" not in paper.get_current_holdings()


def test_cannot_overspend_cash():
    fills = paper.execute_trades([_trade("AAA", "BUY", 5.0)])   # 500% of NAV
    assert fills[0]["shares"] * 100 <= C.PORTFOLIO_VALUE
