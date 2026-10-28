import pytest

from momentum_rebalancer import cli


def test_regimes_command_lists_presets(capsys):
    cli.main(["regimes"])
    out = capsys.readouterr().out
    for name in ("baseline", "stress", "crash", "rebound", "rotation"):
        assert name in out
    assert "untested" in out


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])


def test_regime_flag_sets_env_before_commands_run(monkeypatch):
    monkeypatch.setenv("MR_REGIME", "baseline")        # restored on teardown
    seen = {}
    monkeypatch.setitem(cli.COMMANDS, "regimes", lambda a: seen.setdefault("env", __import__("os").environ["MR_REGIME"]))
    cli.main(["--regime", "crash", "regimes"])
    assert seen["env"] == "crash"


def test_missing_data_gives_friendly_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["signals"])
    assert e.value.code == 1
    assert "[error]" in capsys.readouterr().err


# ── broker pre-flight and --save ──────────────────────────────────────────────

def test_broker_check_paper_needs_no_connection(capsys):
    cli.main(["broker-check"])
    assert "no external connection" in capsys.readouterr().out


def _fake_ibkr(monkeypatch, reachable):
    from momentum_rebalancer import config as C
    from momentum_rebalancer.broker import ibkr
    monkeypatch.setattr(C, "BROKER", "ibkr")
    monkeypatch.setattr(ibkr, "is_connected", lambda: reachable)


def test_broker_check_reports_unreachable_tws(monkeypatch, capsys):
    _fake_ibkr(monkeypatch, reachable=False)
    with pytest.raises(SystemExit) as e:
        cli.main(["broker-check"])
    assert e.value.code == 1
    assert "Cannot reach TWS" in capsys.readouterr().err


def test_broker_check_reports_reachable_tws(monkeypatch, capsys):
    _fake_ibkr(monkeypatch, reachable=True)
    cli.main(["broker-check"])
    assert "reachable" in capsys.readouterr().out


def test_rebalance_stops_before_planning_when_tws_is_down(monkeypatch, capsys):
    _fake_ibkr(monkeypatch, reachable=False)
    from momentum_rebalancer.strategy import targets
    monkeypatch.setattr(targets, "plan_rebalance",
                        lambda: (_ for _ in ()).throw(AssertionError("planning must not run")))
    for cmd in ("rebalance", "trades"):
        with pytest.raises(SystemExit):
            cli.main([cmd])
    assert "Cannot reach TWS" in capsys.readouterr().err
