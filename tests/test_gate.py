import pytest

from insiderbot.config import Settings
from insiderbot.models import AccountState, OrderPlan, Position
from insiderbot.risk import gate


def cfg(tmp_path):
    s = Settings()
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    return s


def acct(equity=100_000.0, last=100_000.0, cash=100_000.0, blocked=False, positions=None):
    return AccountState(equity, cash, last, positions or {}, blocked)


def test_clean_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)
    assert gate.pre_trade(cfg(tmp_path), acct(), 100_000.0, True, True) == []


def test_kill_file_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)
    c = cfg(tmp_path)
    assert not gate.kill_switch_active(c)
    gate.trip(c, "manual")
    assert gate.kill_switch_active(c)
    assert gate.pre_trade(c, acct(), None, True, True)
    (tmp_path / "KILL_SWITCH").unlink()
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert gate.kill_switch_active(c)
    monkeypatch.setenv("BOT_KILL_SWITCH", "no")
    assert not gate.kill_switch_active(c)


def test_daily_loss_trips_kill(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)
    c = cfg(tmp_path)
    r = gate.pre_trade(c, acct(equity=96_500.0), 100_000.0, True, True)
    assert any("daily loss" in x for x in r) and gate.kill_switch_active(c)
    assert "daily loss" in (tmp_path / "KILL_SWITCH").read_text()


def test_drawdown_trips_kill(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)
    c = cfg(tmp_path)
    r = gate.pre_trade(c, acct(equity=84_000.0, last=84_000.0), 100_000.0, True, True)
    assert any("drawdown" in x for x in r) and gate.kill_switch_active(c)


def test_small_loss_ok_and_blockers(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)
    c = cfg(tmp_path)
    assert gate.pre_trade(c, acct(equity=98_000.0), 100_000.0, True, True) == []
    assert gate.pre_trade(c, acct(), None, False, True)
    assert gate.pre_trade(c, acct(), None, True, False)
    assert gate.pre_trade(c, acct(blocked=True), None, True, True)
    assert gate.pre_trade(c, acct(equity=0.0, last=0.0), None, True, True)
    assert not gate.kill_switch_active(c)


def buy(sym, qty, px=20.0):
    return OrderPlan(sym, "buy", qty, "t", px)


def test_check_orders(tmp_path):
    c = cfg(tmp_path)
    assert gate.check_orders(c, [buy("A", 400)], acct(), {}, {}) == []
    assert gate.check_orders(c, [buy("A", 401)], acct(), {}, {})  # > 8% weight
    assert gate.check_orders(c, [buy("A", 0)], acct(), {}, {})
    assert gate.check_orders(c, [buy("A", 10, 0.0)], acct(), {}, {})
    assert gate.check_orders(c, [buy(s, 10) for s in "ABCD"], acct(), {}, {})  # > 3 new buys
    assert gate.check_orders(c, [buy("A", 10), buy("A", 10)], acct(), {}, {})  # duplicate
    assert gate.check_orders(c, [buy("A", 400)], acct(cash=100.0), {}, {})
    pos = {"P": Position("P", 1, 1, 89_000.0, None, None)}
    assert gate.check_orders(c, [buy("A", 400)], acct(positions=pos), {}, {})  # gross > 90%
    sell = OrderPlan("P", "sell", 5, "t", 1.0)
    assert gate.check_orders(c, [sell], acct(positions=pos), {}, {})  # more than held
    assert gate.check_orders(c, [OrderPlan("A", "short", 1, "t", 1.0)], acct(), {}, {})


@pytest.fixture(autouse=True)
def _noenv(monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)


def test_check_orders_blocks_everything_when_killed(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    sell = OrderPlan("P", "sell", 1, "t", 1.0)
    pos = {"P": Position("P", 5, 1, 5.0, None, None)}
    assert gate.check_orders(c, [sell], acct(positions=pos), {}, {}) == []
    gate.trip(c, "x")
    assert gate.check_orders(c, [sell], acct(positions=pos), {}, {}) == ["kill switch active"]
    (tmp_path / "KILL_SWITCH").unlink()
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert gate.check_orders(c, [sell], acct(positions=pos), {}, {})


def test_loss_trip_writes_file_and_blocks_sells(tmp_path):
    c = cfg(tmp_path)
    assert gate.pre_trade(c, acct(equity=90_000.0), 100_000.0, True, True)
    assert (tmp_path / "KILL_SWITCH").exists()
    sell = OrderPlan("P", "sell", 1, "t", 1.0)
    pos = {"P": Position("P", 5, 1, 5.0, None, None)}
    assert gate.check_orders(c, [sell], acct(positions=pos), {}, {})


def test_baseline_never_fails_open(tmp_path):
    c = cfg(tmp_path)
    # first run: no journal peak, valid last_equity -> drawdown 0, clear
    assert gate.pre_trade(c, acct(), None, True, True) == []
    # no peak and no usable last_equity -> blocked
    r = gate.pre_trade(c, acct(last=0.0), None, True, True)
    assert "risk baseline unavailable" in r
    r = gate.pre_trade(c, acct(last=float("nan")), None, True, True)
    assert "risk baseline unavailable" in r
    r = gate.pre_trade(c, acct(equity=float("nan")), 100_000.0, True, True)
    assert "risk baseline unavailable" in r
    # last_equity unusable but journal peak present -> drawdown still checked
    r = gate.pre_trade(c, acct(equity=80_000.0, last=0.0), 100_000.0, True, True)
    assert any("drawdown" in x for x in r)


def test_peak_uses_max_of_sources(tmp_path):
    c = cfg(tmp_path)
    # journal peak stale-low (50k) but last_equity 100k: drawdown measured from 100k
    r = gate.pre_trade(c, acct(equity=84_000.0, last=100_000.0), 50_000.0, True, True)
    assert any("drawdown" in x for x in r)


def test_check_orders_pending_and_sector(tmp_path):
    c = cfg(tmp_path)
    assert gate.check_orders(c, [buy("A", 100)], acct(), {"A": 2000.0}, {})
    assert gate.check_orders(c, [buy("A", 100)], acct(), {"P": 86_000.0}, {})  # gross
    assert gate.check_orders(c, [buy("A", 100)], acct(), {f"P{i}": 1.0 for i in range(12)}, {})
    sec = {"A": "Tech", "P": "Tech"}
    assert gate.check_orders(c, [buy("A", 100)], acct(), {"P": 29_000.0}, sec)  # sector
    assert gate.check_orders(c, [buy("A", 100)], acct(), {"P": 5_000.0}, sec) == []


def test_check_sells_ignores_portfolio_limits(tmp_path):
    """Regression: 11 positions x $8.4k on $100k = 92.4% gross must not block exits."""
    c = cfg(tmp_path)
    pos = {f"P{i}": Position(f"P{i}", 100, 84.0, 8_400.0, None, None) for i in range(11)}
    a = acct(cash=7_600.0, positions=pos)
    sells = [OrderPlan("P0", "sell", 100, "exit:stop", 70.0), OrderPlan("P1", "sell", 100, "exit:stop", 70.0)]
    assert gate.check_orders(c, sells, a, {}, {})  # the aggregate check would block
    assert gate.check_sells(c, sells, a) == []


def test_check_sells_basic_violations_and_kill(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    pos = {"P": Position("P", 5, 1, 5.0, None, None)}
    a = acct(positions=pos)
    assert gate.check_sells(c, [OrderPlan("P", "sell", 6, "t", 1.0)], a)
    assert gate.check_sells(c, [OrderPlan("Q", "sell", 1, "t", 1.0)], a)
    assert gate.check_sells(c, [OrderPlan("P", "buy", 1, "t", 1.0)], a)
    assert gate.check_sells(c, [OrderPlan("P", "sell", 0, "t", 1.0)], a)
    assert gate.check_sells(c, [OrderPlan("P", "sell", 2, "t", 1.0)] * 2, a)
    ok = [OrderPlan("P", "sell", 5, "t", 1.0)]
    assert gate.check_sells(c, ok, a) == []
    gate.trip(c, "x")
    assert gate.check_sells(c, ok, a) == ["kill switch active"]
    (tmp_path / "KILL_SWITCH").unlink()
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert gate.check_sells(c, ok, a) == ["kill switch active"]


def test_kill_switch_from_env_file(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    f = tmp_path / "proj.env"
    monkeypatch.setattr(gate, "ENV_FILE", f)
    assert not gate.kill_switch_active(c)
    f.write_text("ALPACA_API_KEY=x\r\nBOT_KILL_SWITCH=yes\r\n")
    assert gate.kill_switch_active(c)
    assert gate.pre_trade(c, acct(), 100_000.0, True, True) == ["kill switch active"]
    f.write_text("BOT_KILL_SWITCH=no\n")
    assert not gate.kill_switch_active(c)
