"""IBKR order sizing, tested against a simulated TWS connection (no TWS needed)."""
from types import SimpleNamespace

import pytest

pytest.importorskip("ib_insync")

from momentum_rebalancer.broker import ibkr
from momentum_rebalancer.strategy.rebalancer import Trade

NAV = 1_000_000.0


class FakeIB:
    """Just enough of ib_insync.IB for execute_trades."""

    def __init__(self, holdings, prices, unqualified=()):
        self.holdings, self.prices, self.unqualified = holdings, prices, set(unqualified)
        self.orders, self.cancelled = [], []

    def accountValues(self):
        return [SimpleNamespace(tag="NetLiquidation", currency="USD", value=str(NAV))]

    def portfolio(self):
        return [SimpleNamespace(contract=SimpleNamespace(symbol=s), position=q, marketValue=q * self.prices[s])
                for s, q in self.holdings.items()]

    def qualifyContracts(self, contract):
        return [] if contract.symbol in self.unqualified else [contract]

    def reqMktData(self, contract, *args):
        p = self.prices[contract.symbol]
        return SimpleNamespace(last=p, close=p, bid=p, ask=p)

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)

    def sleep(self, seconds):
        pass

    def placeOrder(self, contract, order):
        self.orders.append((contract.symbol, order.action, int(order.totalQuantity)))
        return SimpleNamespace(orderStatus=SimpleNamespace(status="Submitted"))

    def disconnect(self):
        pass


def _trade(ticker, action, target, current):
    return Trade(ticker=ticker, action=action, shares=0, target_pct=target, current_pct=current, delta_pct=target - current)


def _run(fake, trades, monkeypatch):
    monkeypatch.setattr(ibkr, "_connect", lambda: fake)
    return ibkr.execute_trades(trades)


def test_a_full_exit_sells_every_share_held_even_when_the_price_has_moved(monkeypatch):
    """Sized from the weight at a higher market price, the old rule gave 28 of 29 shares and left one behind."""
    weight = 29 * 147.43 / NAV
    fake = FakeIB({"BKNG": 29}, {"BKNG": 148.0})
    fills = _run(fake, [_trade("BKNG", "SELL", 0.0, weight)], monkeypatch)
    assert fake.orders == [("BKNG", "SELL", 29)]
    assert fills[0]["shares"] == 29


def test_a_single_share_position_can_be_cleared(monkeypatch):
    fake = FakeIB({"MRNA": 1}, {"MRNA": 216.0})
    _run(fake, [_trade("MRNA", "SELL", 0.0, 216.0 / NAV)], monkeypatch)
    assert fake.orders == [("MRNA", "SELL", 1)]


def test_a_partial_sell_never_exceeds_what_is_held(monkeypatch):
    fake = FakeIB({"AAA": 10}, {"AAA": 100.0})
    _run(fake, [_trade("AAA", "SELL", 0.001, 0.006)], monkeypatch)             # weight says 50 shares; only 10 held
    assert fake.orders == [("AAA", "SELL", 10)]


def test_a_partial_sell_uses_the_weight_when_that_is_smaller(monkeypatch):
    fake = FakeIB({"AAA": 1000}, {"AAA": 100.0})
    _run(fake, [_trade("AAA", "SELL", 0.05, 0.06)], monkeypatch)               # 1% of NAV = 10,000 dollars = 100 shares
    assert fake.orders == [("AAA", "SELL", 100)]


def test_a_symbol_that_is_not_held_is_never_sold(monkeypatch):
    fake = FakeIB({}, {"ZZZ": 50.0})
    fills = _run(fake, [_trade("ZZZ", "SELL", 0.0, 0.01)], monkeypatch)
    assert fake.orders == [] and fills == []


def test_buys_are_sized_from_the_weight_as_before(monkeypatch):
    fake = FakeIB({}, {"AAA": 50.0})
    _run(fake, [_trade("AAA", "BUY", 0.02, 0.0)], monkeypatch)                  # 2% of 1,000,000 / 50
    assert fake.orders == [("AAA", "BUY", 400)]


def test_market_data_lines_are_released_and_unqualifiable_names_skipped(monkeypatch):
    fake = FakeIB({}, {"AAA": 50.0, "BAD": 10.0}, unqualified=["BAD"])
    fills = _run(fake, [_trade("AAA", "BUY", 0.01, 0.0), _trade("BAD", "BUY", 0.01, 0.0)], monkeypatch)
    assert fake.cancelled == ["AAA"] and [f["ticker"] for f in fills] == ["AAA"]


@pytest.mark.parametrize("sized,target,held,expected", [
    (28, 0.0, 29, 29), (50, 0.001, 10, 10), (5, 0.02, 100, 5), (5, 0.0, 0, 0), (5, 0.0, -3, 0), (5, 0.02, 0.9, 0),
])
def test_sell_quantity(sized, target, held, expected):
    assert ibkr.sell_quantity(sized, target, held) == expected
