from momentum_rebalancer.strategy.rebalancer import compute_trades, trades_to_dataframe


def test_sells_come_before_buys():
    trades = compute_trades({"A": 0.5, "B": 0.5}, {"C": 0.6, "A": 0.4}, portfolio_value=1_000_000)
    actions = [t.action for t in trades]
    assert actions == sorted(actions, key=lambda a: 0 if a == "SELL" else 1)
    assert {t.ticker: t.action for t in trades} == {"C": "SELL", "B": "BUY", "A": "BUY"}


def test_dollar_floor_skips_noise():
    # 0.00005 * 1,000,000 = $50 < $100
    assert compute_trades({"A": 0.50005}, {"A": 0.5}, portfolio_value=1_000_000) == []


def test_pct_band_skips_small_resizes_but_never_exits():
    target  = {"A": 0.052, "B": 0.0}
    current = {"A": 0.050, "B": 0.003}          # A: +0.2% resize; B: tiny leftover to exit
    trades = compute_trades(target, current, portfolio_value=1_000_000, min_trade_pct=0.005)
    assert [t.ticker for t in trades] == ["B"]
    assert trades[0].action == "SELL"


def test_band_zero_is_baseline_behaviour():
    target, current = {"A": 0.052}, {"A": 0.050}
    assert len(compute_trades(target, current, portfolio_value=1_000_000)) == 1
    assert len(compute_trades(target, current, portfolio_value=1_000_000, min_trade_pct=0.0)) == 1


def test_entries_below_band_are_skipped():
    assert compute_trades({"A": 0.003}, {}, portfolio_value=1_000_000, min_trade_pct=0.005) == []


def test_dataframe_shape():
    df = trades_to_dataframe(compute_trades({"A": 1.0}, {}, portfolio_value=1_000_000))
    assert list(df.columns) == ["ticker", "action", "shares", "target_pct", "current_pct", "delta_pct"]
