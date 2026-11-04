"""End-to-end CLI tests: fakes for EDGAR, market, LLM; SimBroker; temp journal. No network."""
from __future__ import annotations

import fcntl
import shutil
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from insiderbot import cli
from insiderbot.agents.schemas import AnalystNote, DebateCase, PMOut, ResearchOut
from insiderbot.broker.sim import SimBroker
from insiderbot.config import ROOT, Secrets, Settings
from insiderbot.journal.store import Journal
from insiderbot.models import InsiderTxn

# Tue 2026-11-03 10:30 ET (EST, UTC-5)
NOW = datetime(2026, 11, 3, 15, 30, tzinfo=timezone.utc)
OPEN = datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)
CLOSE = datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)


def txn(cik, title="", director=False, officer=True, code="P", ticker="ABC", acc=None,
        accepted=NOW - timedelta(hours=2), ten_b5=False, shares=10_000.0):
    return InsiderTxn(
        accession=acc or f"acc-{cik}-{code}", accepted_at=accepted, tx_date=accepted.date(),
        ticker=ticker, issuer_cik="1", issuer_name=f"{ticker} Inc", insider_cik=cik,
        insider_name=f"insider {cik}", is_officer=officer, is_director=director,
        is_ten_pct=False, is_entity=False, officer_title=title, code=code, shares=shares,
        price=100.0, value=shares * 100.0, owned_after=shares * 2, is_10b5_1=ten_b5)


def cluster(ticker="ABC"):
    return [txn("c1", "Chief Executive Officer", ticker=ticker),
            txn("c2", director=True, officer=False, ticker=ticker),
            txn("c3", director=True, officer=False, ticker=ticker)]


def bars(n=260, start=100.0, end=70.0, vol=100_000):
    close = np.linspace(start, end, n)
    idx = pd.bdate_range(end=date(2026, 11, 2), periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": float(vol)}, index=idx)


class FakeMarket:
    def __init__(self, tickers=("ABC",), cal=(OPEN, CLOSE), quotes=None):
        self.tickers = set(tickers)
        self.cal = cal
        self.quotes = quotes if quotes is not None else {t: (69.9, 70.1) for t in tickers}

    holidays: set = set()
    spy_end = date(2026, 11, 2)

    def calendar_today(self, now):
        return self.cal

    def trading_days(self, start, end):
        return [d.date() for d in pd.bdate_range(start, end) if d.date() not in self.holidays]

    def bars(self, symbols, start, end, now=None):
        out = {s: bars() for s in symbols if s in self.tickers}
        if "SPY" in symbols and self.spy_end is not None:
            out["SPY"] = bars().loc[:pd.Timestamp(self.spy_end)]
        return out

    def fundamentals(self, symbol):
        return {"market_cap": 1e9, "sector": "Tech"} if symbol in self.tickers else {}

    def asset_info(self, symbols):
        return {s: {"tradable": True, "exchange": "NASDAQ", "name": f"{s} Inc",
                    "fractionable": True} for s in symbols if s in self.tickers}

    def latest_quotes(self, symbols):
        return {s: self.quotes[s] for s in symbols if s in self.quotes}

    def news(self, symbol, limit=8):
        return [{"headline": "h", "summary": "s", "created_at": "x", "source": "y"}]


class FakeEdgar:
    def __init__(self, txns, errors=0, attempted=None, fail=False):
        self.txns = txns
        self.calls = []
        self.errors, self.attempted, self.fail = errors, attempted, fail
        self.last_errors = self.last_attempted = 0

    def fetch(self, since, until):
        self.calls.append((since, until))
        if self.fail:
            raise ConnectionError("EDGAR down")
        out = [t for t in self.txns if since < t.accepted_at <= until]
        self.last_errors = self.errors
        self.last_attempted = self.attempted if self.attempted is not None else len(out)
        return out


class FakeMessages:
    def __init__(self, action="buy"):
        self.action = action
        self.calls = 0

    def parse(self, **kw):
        self.calls += 1
        schema = kw["output_format"]
        if schema is PMOut:
            out = PMOut(action=self.action, conviction=4, thesis="t", rule_flags_addressed="all",
                        what_would_change_my_mind="m")
        elif schema is AnalystNote:
            out = AnalystNote(summary="ok", positives=["p"], negatives=[], red_flags=[])
        elif schema is DebateCase:
            out = DebateCase(side="x", argument="a", key_risks=["r"])
        else:
            out = ResearchOut(change=False, rationale="r", new_weights={}, new_params={},
                              what_we_learned="w")
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=out, usage=usage, stop_reason="end_turn")


def fake_client(action="buy"):
    msgs = FakeMessages(action)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


class Recorder:
    def __init__(self):
        self.pushes = []

    def __call__(self, title, body, *, priority=3, tags=None, enabled, now=None):
        self.pushes.append({"title": title, "body": body, "priority": priority, "enabled": enabled})
        return True


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)

    def no_network_broker(*a, **kw):
        raise AssertionError("tests must inject a broker; AlpacaBroker would hit the network")

    monkeypatch.setattr(cli, "AlpacaBroker", no_network_broker)
    fdir = tmp_path / "formula"
    fdir.mkdir()
    shutil.copy(ROOT / "formula" / "v001.yaml", fdir / "v001.yaml")
    (fdir / "ACTIVE").write_text("v001\n")
    s = Settings()
    s.paths.journal_db = str(tmp_path / "data" / "journal.db")
    s.paths.reports_dir = str(tmp_path / "reports")
    s.paths.formula_dir = str(fdir)
    s.paths.cache_dir = str(tmp_path / "data" / "cache")
    s.paths.lock_file = str(tmp_path / "data" / "run.lock")
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    client, msgs = fake_client()
    push = Recorder()
    deps = cli.Deps(
        settings=s,
        secrets=Secrets(alpaca_key="k", alpaca_secret="s", anthropic_key=None, edgar_identity="me@x.com"),
        root=tmp_path, now=lambda: NOW, sleep=lambda _s: None,
        market=FakeMarket(), edgar=FakeEdgar(cluster()), broker=None,
        llm_client=client, push=push)
    return SimpleNamespace(deps=deps, s=s, tmp=tmp_path, push=push, msgs=msgs)


def journal(env):
    return Journal(env.s.paths.journal_db)


def dry_journal(env):
    return Journal(env.tmp / "data" / "journal.dry-run.db")


def last_line(capsys):
    out = capsys.readouterr().out.strip().splitlines()
    return out[-1] if out else ""


def runs(env, job="scan", j=None):
    return [dict(r) for r in (j or journal(env)).conn.execute(
        "SELECT * FROM runs WHERE job=? ORDER BY id", (job,))]


# --- scheduling ------------------------------------------------------------

def test_scan_late_skipped(env, capsys):
    env.deps.now = lambda: datetime(2026, 11, 3, 17, 5, tzinfo=timezone.utc)  # 12:05 ET
    rc = cli.main(["scan", "--slot", "am"], env.deps)
    assert rc == 0
    assert last_line(capsys).startswith("OK scan skipped_late")
    assert runs(env)[-1]["status"] == "skipped_late"
    assert env.deps.edgar.calls == []


def test_scan_late_forced_runs(env, capsys):
    env.deps.now = lambda: datetime(2026, 11, 3, 17, 5, tzinfo=timezone.utc)
    env.deps.market.cal = (OPEN, CLOSE)
    rc = cli.main(["scan", "--slot", "am", "--mode", "dry-run", "--force"], env.deps)
    assert rc == 0, capsys.readouterr()
    assert runs(env, j=dry_journal(env))[-1]["status"] == "ok"


def test_scan_closed_day_skipped(env, capsys):
    env.deps.market.cal = None
    rc = cli.main(["scan", "--slot", "am"], env.deps)
    assert rc == 0
    assert last_line(capsys).startswith("OK scan skipped_closed")
    assert runs(env)[-1]["status"] == "skipped_closed"
    assert env.deps.edgar.calls == []


# --- dry-run ---------------------------------------------------------------

def test_scan_dry_run_places_sim_order_and_stop(env, capsys):
    made = []
    orig = cli.SimBroker

    def spy(*a, **kw):
        b = orig(*a, **kw)
        made.append(b)
        return b

    cli.SimBroker = spy
    try:
        env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
        rc = cli.main(["scan", "--mode", "dry-run"], env.deps)
    finally:
        cli.SimBroker = orig
    line = last_line(capsys)
    assert rc == 0, line
    assert line.startswith("OK scan")
    (sim,) = made
    acct = sim.get_account()
    assert "ABC" in acct.positions
    stops = sim.open_orders()
    assert len(stops) == 1 and stops[0]["symbol"] == "ABC"
    assert stops[0]["stop_price"] == pytest.approx(round(70.1 * (1 - 0.12), 2))
    j = dry_journal(env)
    # dry-run never touches broker-state tables
    assert j.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    assert j.conn.execute("SELECT COUNT(*) FROM position_meta").fetchone()[0] == 0
    assert j.conn.execute("SELECT COUNT(*) FROM equity").fetchone()[0] == 0
    # research rows recorded in the dry-run journal only
    assert len(j.candidates()) == 1
    assert j.decision_actions() == {1: "buy"}
    assert runs(env, j=j)[-1]["mode"] == "dry-run"
    assert j.get_watermark() == NOW - timedelta(hours=2)
    assert list((env.tmp / "reports").glob("scan-*.md"))
    real = journal(env)
    for table in ("runs", "txns", "candidates", "decisions", "costs", "meta"):
        assert real.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_dry_run_budget_counts_both_journals(env, capsys):
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    journal(env).add_cost(0, 25.0)
    dry_journal(env).add_cost(0, 16.0)  # 41 >= 40 across both
    assert cli.main(["scan", "--mode", "dry-run"], env.deps) == 0, capsys.readouterr()
    assert env.msgs.calls == 0
    assert dry_journal(env).decision_actions() == {1: "pass"}
    # paper sees the dry-run spend too
    b, _ = paper(env)
    env.deps.edgar = FakeEdgar(cluster("XYZ"))
    env.deps.market = FakeMarket(tickers=("XYZ",))
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert env.msgs.calls == 0 and b.get_account().positions == {}
    capsys.readouterr()
    assert cli.main(["status"], env.deps) == 0
    assert "$41.00 of $40" in capsys.readouterr().out


def test_dry_run_never_builds_paper_broker(env, capsys, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("AlpacaBroker constructed in dry-run")

    monkeypatch.setattr(cli, "AlpacaBroker", boom)
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    assert cli.main(["scan", "--mode", "dry-run"], env.deps) == 0, capsys.readouterr()


# --- paper on SimBroker ----------------------------------------------------

def paper(env, price=70.0, positions=None):
    px = {"ABC": price, "XYZ": price}
    b = SimBroker(100_000, lambda s: px[s], hide_entry_info=True)
    env.deps.broker = b
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    return b, px


def test_paper_buy_places_stop_and_meta(env, capsys):
    b, _ = paper(env)
    rc = cli.main(["scan"], env.deps)
    assert rc == 0, capsys.readouterr()
    j = journal(env)
    meta = j.position_meta("ABC")
    assert meta is not None and meta["entry_price"] == 70.0 and meta["candidate_id"] == 1
    rows = [dict(r) for r in j.conn.execute("SELECT * FROM orders ORDER BY id")]
    assert [r["side"] for r in rows] == ["buy", "sell"]
    assert rows[0]["status"] == "filled" and rows[0]["fill_price"] == 70.0
    assert rows[1]["reason"] == "stop"
    assert len(b.open_orders()) == 1
    assert j.peak_equity() == 100_000
    assert any("order" in p["title"].lower() for p in env.push.pushes)
    assert all(p["enabled"] is True for p in env.push.pushes)


def test_scan_kill_switch_no_orders_but_candidates_logged(env, capsys):
    b, _ = paper(env)
    (env.tmp / "KILL_SWITCH").write_text("manual\n")
    rc = cli.main(["scan"], env.deps)
    assert rc == 0, capsys.readouterr()
    j = journal(env)
    assert len(j.candidates()) == 1
    assert j.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    assert b.get_account().positions == {}
    assert env.msgs.calls == 0  # no agent spend while blocked


def test_scan_kill_switch_env_blocks_sells_too(env, capsys, monkeypatch):
    b, px = paper(env)
    b.submit(__import__("insiderbot.models", fromlist=["OrderPlan"]).OrderPlan("XYZ", "buy", 10, "x", 70.0))
    j = journal(env)
    j.add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)  # deep below stop
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert "XYZ" in b.get_account().positions
    assert j.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_loss_limit_trips_kill_switch_and_pushes(env, capsys):
    b, _ = paper(env)
    b.last_equity = 110_000  # equity 100k: -9% day
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert (env.tmp / "KILL_SWITCH").exists()
    kills = [p for p in env.push.pushes if "kill" in p["title"].lower()]
    assert kills and kills[0]["priority"] >= 5
    assert journal(env).conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_scan_no_anthropic_key_no_buys(env, capsys):
    b, _ = paper(env)
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": None})
    env.deps.llm_client = None
    rc = cli.main(["scan"], env.deps)
    assert rc == 0, capsys.readouterr()
    j = journal(env)
    assert j.decision_actions() == {1: "pass"}
    assert b.get_account().positions == {}
    assert "agents_unavailable" in runs(env)[-1]["summary"]
    rep = next((env.tmp / "reports").glob("scan-*.md")).read_text()
    assert "Anthropic" in rep


def test_scan_budget_hit_no_buys(env, capsys):
    b, _ = paper(env)
    j = journal(env)
    j.add_cost(0, 41.0)
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    j = journal(env)
    row = j.conn.execute("SELECT action, thesis FROM decisions").fetchone()
    assert row["action"] == "pass" and "budget" in row["thesis"]
    assert b.get_account().positions == {}
    assert env.msgs.calls == 0
    budget = [p for p in env.push.pushes if "budget" in p["title"].lower()]
    assert len(budget) == 1
    # second scan same month: no second budget push
    env.deps.edgar = FakeEdgar([])
    assert cli.main(["scan"], env.deps) == 0
    assert len([p for p in env.push.pushes if "budget" in p["title"].lower()]) == 1


def test_scan_exit_before_buy(env, capsys):
    b, px = paper(env)
    from insiderbot.models import OrderPlan
    b.submit(OrderPlan("XYZ", "buy", 100, "seed", 70.0))
    stop = b.submit_stop("XYZ", 100, 60.0)
    j = journal(env)
    j.add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)  # -30%: stop rule
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ"),
                                 quotes={"ABC": (69.9, 70.1), "XYZ": (69.9, 70.1)})
    seq = []
    orig_submit, orig_cancel = b.submit, b.cancel_orders_for
    b.submit = lambda plan, client_order_id=None: (seq.append((plan.side, plan.symbol)),
                                                   orig_submit(plan, client_order_id))[1]
    b.cancel_orders_for = lambda sym: (seq.append(("cancel", sym)), orig_cancel(sym))[1]
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert seq[:3] == [("cancel", "XYZ"), ("sell", "XYZ"), ("buy", "ABC")]
    assert "XYZ" not in b.get_account().positions
    j = journal(env)
    closed = j.closed_positions()
    assert closed[0]["symbol"] == "XYZ" and closed[0]["exit_reason"] == "stop"
    assert b.order_status(stop.broker_id)["status"] == "canceled"


def test_insider_sale_exit(env, capsys):
    b, px = paper(env)
    from insiderbot.models import OrderPlan
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))
    j = journal(env)
    j.add_txns([txn("x1", "CFO", ticker="XYZ", accepted=datetime(2026, 9, 1, tzinfo=timezone.utc))], "live")
    j.add_position_meta("XYZ", date(2026, 10, 30), 70.0, "v001", None)
    sale = txn("x1", "CFO", ticker="XYZ", code="S", accepted=NOW - timedelta(hours=1))
    env.deps.edgar = FakeEdgar(cluster() + [sale])
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ"),
                                 quotes={"ABC": (69.9, 70.1), "XYZ": (69.9, 70.1)})
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert "XYZ" not in b.get_account().positions
    assert journal(env).closed_positions()[0]["exit_reason"] == "insider_sale"


@pytest.mark.parametrize("entry,exited", [(date(2026, 10, 1), True), (date(2026, 11, 2), False)])
def test_agent_exit_pm_only_respects_min_hold(env, capsys, entry, exited):
    b, px = paper(env)
    from insiderbot.models import OrderPlan
    b.submit(OrderPlan("ABC", "buy", 10, "seed", 70.0))
    j = journal(env)
    j.add_position_meta("ABC", entry, 70.0, "v001", None)
    env.msgs.action = "exit"
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert env.msgs.calls == 1  # PM only, no analysts/debate
    assert ("ABC" not in b.get_account().positions) is exited
    j = journal(env)
    assert j.conn.execute("SELECT action FROM decisions").fetchone()[0] == "exit"
    if exited:
        assert j.closed_positions()[0]["exit_reason"] == "agent"


def test_sell_rejected_after_cancel_restores_stop(env, capsys):
    b, px = paper(env)
    from insiderbot.models import OrderPlan, SubmittedOrder
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))
    b.submit_stop("XYZ", 10, 60.0)
    j = journal(env)
    j.add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ"),
                                 quotes={"ABC": (69.9, 70.1), "XYZ": (69.9, 70.1)})
    orig = b.submit
    b.submit = lambda plan, client_order_id=None: (
        SubmittedOrder(plan, None, "rejected", "nope") if plan.side == "sell" else orig(plan, client_order_id))
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    stops = [o for o in b.open_orders() if o["symbol"] == "XYZ"]
    assert len(stops) == 1 and stops[0]["stop_price"] == pytest.approx(88.0)
    assert journal(env).position_meta("XYZ") is not None


def test_unknown_buy_is_pending_never_rebought(env, capsys):
    b, _ = paper(env)
    from insiderbot.models import SubmittedOrder
    orig = b.submit
    calls = []

    def flaky(plan, client_order_id=None):
        calls.append(client_order_id)
        return SubmittedOrder(plan, None, "unknown", "timeout")

    b.submit = flaky
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert len(calls) == 1 and calls[0].startswith("ib-")
    j = journal(env)
    assert j.conn.execute("SELECT status FROM orders").fetchone()["status"] == "unknown"
    assert b.open_orders() == []  # no stop for an unseen fill
    # next scan: a new buyer arrives, ABC still must not be re-bought
    env.deps.edgar = FakeEdgar([txn("c9", "CFO", accepted=NOW - timedelta(minutes=30))])
    env.deps.now = lambda: NOW + timedelta(minutes=10)
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert len(calls) == 1


def test_market_closed_records_candidates_no_orders(env, capsys):
    b, _ = paper(env)
    env.deps.market.cal = (OPEN, NOW - timedelta(minutes=1))  # half day, already closed
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    j = journal(env)
    assert len(j.candidates()) == 1
    assert b.get_account().positions == {}


# --- errors, lock, status ---------------------------------------------------

def test_lock_busy(env, capsys):
    lock = env.tmp / "data" / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = cli.main(["scan"], env.deps)
        assert rc == 0
        assert last_line(capsys) == "OK scan skipped_locked (another job running)"
        assert cli.main(["reconcile"], env.deps) == 0
        assert cli.main(["review"], env.deps) == 1
        assert last_line(capsys) == "ERROR: lock: another job running"
    assert env.push.pushes == []  # lock contention never pushes
    assert runs(env) == []


def test_missing_identity_still_runs_exits_and_stops(env, capsys):
    b, px = paper(env)
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))  # stop rule fires (entry 100)
    px["QQQ"] = 70.0
    b.submit(OrderPlan("QQQ", "buy", 5, "seed", 70.0))   # held, no stop: must get one
    j = journal(env)
    j.add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)
    j.add_position_meta("QQQ", date(2026, 10, 30), 70.0, "v001", None)
    env.deps.market = FakeMarket(tickers=("XYZ", "QQQ"),
                                 quotes={"XYZ": (69.9, 70.1), "QQQ": (69.9, 70.1)})
    env.deps.edgar = None
    env.deps.secrets = env.deps.secrets.model_copy(update={"edgar_identity": None})
    rc = cli.main(["scan"], env.deps)
    assert rc == 0, capsys.readouterr()
    pos = b.get_account().positions
    assert "XYZ" not in pos and "QQQ" in pos
    assert any(o["symbol"] == "QQQ" for o in b.open_orders())
    r = runs(env)[-1]
    assert r["status"] == "ok" and "EDGAR_IDENTITY not set" in r["summary"]
    ident = [p for p in env.push.pushes if "EDGAR_IDENTITY" in p["title"]]
    assert len(ident) == 1 and ident[0]["priority"] >= 5
    assert journal(env).get_watermark() is None
    assert cli.main(["scan"], env.deps) == 0
    assert len([p for p in env.push.pushes if "EDGAR_IDENTITY" in p["title"]]) == 1  # daily dedupe


def test_features_failure_runs_exits_and_carries_tickers(env, capsys, monkeypatch):
    b, px = paper(env)
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))
    px["QQQ"] = 70.0
    b.submit(OrderPlan("QQQ", "buy", 5, "seed", 70.0))
    j = journal(env)
    j.add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)
    j.add_position_meta("QQQ", date(2026, 10, 30), 70.0, "v001", None)
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ", "QQQ"), quotes={
        "ABC": (69.9, 70.1), "XYZ": (69.9, 70.1), "QQQ": (69.9, 70.1)})
    real_build = cli.features.build

    def boom(*a, **kw):
        raise ValueError("bad bars frame")

    monkeypatch.setattr(cli.features, "build", boom)
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    pos = b.get_account().positions
    assert "XYZ" not in pos and "ABC" not in pos  # exit ran, no buy
    assert any(o["symbol"] == "QQQ" for o in b.open_orders())  # stop step ran
    j = journal(env)
    assert j.get_watermark() is None and j.candidates() == []
    assert "features failed" in runs(env)[-1]["summary"]
    (p,) = [p for p in env.push.pushes if "features" in p["title"]]
    assert p["priority"] >= 5
    # next scan: EDGAR has nothing new, but the carried ticker is journaled and bought
    monkeypatch.setattr(cli.features, "build", real_build)
    env.deps.now = lambda: NOW + timedelta(minutes=30)
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    j = journal(env)
    assert [c["ticker"] for c in j.candidates()] == ["ABC"]
    assert j.get_watermark() == NOW - timedelta(hours=2)
    assert "ABC" in b.get_account().positions


def test_status_works_without_identity(env, capsys):
    b, _ = paper(env)
    env.deps.edgar = None
    env.deps.secrets = env.deps.secrets.model_copy(update={"edgar_identity": None})
    rc = cli.main(["status"], env.deps)
    out = capsys.readouterr().out
    assert rc == 0
    assert "v001" in out and "100,000" in out
    assert out.strip().splitlines()[-1].startswith("OK status")


# --- reconcile / backfill / review ----------------------------------------

def test_reconcile(env, capsys):
    b, px = paper(env)
    from insiderbot.models import OrderPlan
    b.submit(OrderPlan("ABC", "buy", 10, "seed", 70.0))
    j = journal(env)
    j.add_position_meta("ABC", date(2026, 10, 30), 70.0, "v001", None)
    j.add_position_meta("GONE", date(2026, 10, 1), 50.0, "v001", None)
    run = j.start_run("scan", "paper", "v001")
    from insiderbot.models import SubmittedOrder
    oid = j.add_order(run, SubmittedOrder(OrderPlan("GONE", "sell", 5, "stop", 44.0), "brk-9", "new", None))
    b._orders["brk-9"] = {"status": "filled", "filled_qty": 5, "filled_avg_price": 43.5}
    px["ABC"] = 80.0
    env.deps.now = lambda: datetime(2026, 11, 3, 21, 15, tzinfo=timezone.utc)
    rc = cli.main(["reconcile"], env.deps)
    assert rc == 0, capsys.readouterr()
    j = journal(env)
    stops = [o for o in b.open_orders() if o["symbol"] == "ABC"]
    assert len(stops) == 1 and stops[0]["qty"] == 10 and stops[0]["stop_price"] == pytest.approx(61.6)
    closed = {p["symbol"]: p for p in j.closed_positions()}
    assert closed["GONE"]["exit_reason"] == "stop_broker" and closed["GONE"]["exit_price"] == 43.5
    assert j.position_meta("ABC")["peak_price"] == 80.0
    assert j.peak_equity() == pytest.approx(100_000 + 100)
    assert j.conn.execute("SELECT status FROM orders WHERE id=?", (oid,)).fetchone()[0] == "filled"
    assert list((env.tmp / "reports").glob("daily-*.md"))


def test_reconcile_kill_switch_places_no_stop(env, capsys):
    b, px = paper(env)
    from insiderbot.models import OrderPlan
    b.submit(OrderPlan("ABC", "buy", 10, "seed", 70.0))
    (env.tmp / "KILL_SWITCH").write_text("x\n")
    assert cli.main(["reconcile"], env.deps) == 0, capsys.readouterr()
    assert b.open_orders() == []


def test_backfill(env, capsys):
    env.deps.edgar = FakeEdgar(cluster())
    rc = cli.main(["backfill", "--days", "3"], env.deps)
    assert rc == 0, capsys.readouterr()
    j = journal(env)
    c = j.candidates()
    assert len(c) == 1 and c[0]["source"] == "backfill"
    assert len(env.deps.edgar.calls) == 3
    assert j.get_watermark() is None
    assert j.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    assert env.msgs.calls == 0


def test_review(env, capsys):
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    rc = cli.main(["review"], env.deps)
    assert rc == 0, capsys.readouterr()
    assert list((env.tmp / "reports" / "weekly").glob("*.md"))
    assert any("weekly" in p["title"].lower() for p in env.push.pushes)


def test_job_failure_pushes(env, capsys):
    class Broken(FakeMarket):
        def calendar_today(self, now):
            raise RuntimeError("calendar down")

    env.deps.market = Broken()
    rc = cli.main(["scan"], env.deps)
    assert rc == 1
    assert last_line(capsys) == "ERROR: calendar: calendar down"
    assert env.push.pushes[-1]["priority"] >= 5


# --- fix round 1 -------------------------------------------------------------

from insiderbot.models import OrderPlan, SubmittedOrder  # noqa: E402
from insiderbot.risk import gate  # noqa: E402


def stop_pushes(env):
    return [p for p in env.push.pushes if "without stop" in p["title"]]


def test_exit_not_blocked_by_gross_exposure(env, capsys):
    """Regression: 11 positions x $8.4k (92.4% gross) must not block a stop-loss exit."""
    syms = [f"P{i}" for i in range(11)]
    px = {s: 84.0 for s in syms}
    b = SimBroker(100_000, lambda s: px[s], hide_entry_info=True)
    env.deps.broker = b
    j = journal(env)
    for s in syms:
        b.submit(OrderPlan(s, "buy", 100, "seed", 84.0))
        j.add_position_meta(s, date(2026, 10, 30), 84.0, "v001", None)
    px["P0"] = 70.0  # below the 12% stop (73.92)
    a = b.get_account()
    assert sum(p.market_value for p in a.positions.values()) / a.equity > 0.90
    env.deps.edgar = FakeEdgar([])
    env.deps.market = FakeMarket(tickers=syms, quotes={s: (px[s] - 0.1, px[s] + 0.1) for s in syms})
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert "P0" not in b.get_account().positions
    assert journal(env).closed_positions()[0]["exit_reason"] == "stop"


def test_kill_switch_in_env_file_blocks_scan(env, capsys, monkeypatch):
    b, _ = paper(env)
    f = env.tmp / "project.env"
    f.write_text("BOT_KILL_SWITCH=yes\n")
    monkeypatch.setattr(gate, "ENV_FILE", f)
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    j = journal(env)
    assert len(j.candidates()) == 1
    assert j.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    assert "kill switch active" in runs(env)[-1]["summary"]


def test_blocked_scan_alerts_missing_stop_once_per_day(env, capsys, monkeypatch):
    b, _ = paper(env)
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))  # no stop at the broker
    journal(env).add_position_meta("XYZ", date(2026, 10, 30), 70.0, "v001", None)
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert b.open_orders() == []
    (p,) = stop_pushes(env)
    assert p["priority"] >= 5 and "XYZ" in p["body"]
    assert "NO GTC STOP: XYZ" in runs(env)[-1]["summary"]
    env.deps.edgar = FakeEdgar([])
    assert cli.main(["scan"], env.deps) == 0
    assert len(stop_pushes(env)) == 1  # deduped for the day
    env.deps.now = lambda: NOW + timedelta(days=1)
    env.deps.market.cal = (OPEN + timedelta(days=1), CLOSE + timedelta(days=1))
    env.deps.market.spy_end = date(2026, 11, 3)
    assert cli.main(["scan"], env.deps) == 0
    assert len(stop_pushes(env)) == 2  # next day: alert again


def test_stop_failed_after_fill_alerts(env, capsys):
    b, _ = paper(env)

    def boom(*a, **kw):
        raise RuntimeError("stop endpoint down")

    b.submit_stop = boom
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert "ABC" in b.get_account().positions
    (p,) = stop_pushes(env)
    assert p["priority"] >= 5 and "ABC" in p["body"]


def test_unblocked_scan_places_missing_stops(env, capsys):
    b, _ = paper(env)
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))
    journal(env).add_position_meta("XYZ", date(2026, 10, 30), 70.0, "v001", None)
    env.deps.edgar = FakeEdgar([])
    env.deps.market = FakeMarket(tickers=("XYZ",))
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    stops = [o for o in b.open_orders() if o["symbol"] == "XYZ"]
    assert len(stops) == 1 and stops[0]["qty"] == 10 and stops[0]["stop_price"] == pytest.approx(61.6)
    assert stop_pushes(env) == []


@pytest.mark.parametrize("errors,attempted,fail,buys", [
    (1, 20, False, True),    # 5%: proceed with what loaded
    (5, 20, False, False),   # 25%: no new buys
    (0, 0, True, False),     # whole fetch failed: no new buys
])
def test_edgar_errors_block_only_buys(env, capsys, errors, attempted, fail, buys):
    b, px = paper(env)
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))
    b.submit_stop("XYZ", 10, 60.0)
    j = journal(env)
    j.add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)  # stop rule fires
    env.deps.edgar = FakeEdgar(cluster(), errors=errors, attempted=attempted, fail=fail)
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ"),
                                 quotes={"ABC": (69.9, 70.1), "XYZ": (69.9, 70.1)})
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    pos = b.get_account().positions
    assert "XYZ" not in pos  # the exit always proceeds
    assert ("ABC" in pos) is buys
    if buys:
        assert any(o["symbol"] == "ABC" for o in b.open_orders())  # stop placed
    summary = runs(env)[-1]["summary"]
    assert "data stale" not in summary
    if errors:
        assert f"{errors} of {attempted} filing(s) failed" in summary
    if fail:
        assert "EDGAR fetch failed" in summary
        assert journal(env).get_watermark() is None  # refetch next scan
        assert env.msgs.calls == 0


@pytest.mark.parametrize("how", ["spy_old", "spy_missing", "no_quotes"])
def test_stale_data_blocks_orders(env, capsys, how):
    b, _ = paper(env)
    if how == "spy_old":
        env.deps.market.spy_end = date(2026, 10, 30)  # last session is Mon Nov 2
    elif how == "spy_missing":
        env.deps.market.spy_end = None
    else:
        env.deps.market.quotes = {}
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert b.get_account().positions == {}
    assert journal(env).conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    assert "data stale" in runs(env)[-1]["summary"]


def test_fresh_data_with_holiday_calendar(env, capsys):
    b, _ = paper(env)
    env.deps.market.holidays = {date(2026, 11, 2)}  # last session = Fri Oct 30
    env.deps.market.spy_end = date(2026, 10, 30)
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert "ABC" in b.get_account().positions


@pytest.mark.parametrize("holidays,exited",
                         [(set(), True), ({date(2026, 10, 28), date(2026, 10, 29)}, False)])
def test_days_held_uses_trading_calendar(env, capsys, holidays, exited):
    b, _ = paper(env)
    b.submit(OrderPlan("ABC", "buy", 10, "seed", 70.0))
    journal(env).add_position_meta("ABC", date(2026, 10, 27), 70.0, "v001", None)
    env.deps.market.holidays = holidays  # 5 weekdays held; minus 2 holidays = 3 < min hold 5
    env.msgs.action = "exit"
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert ("ABC" not in b.get_account().positions) is exited


def test_days_held_falls_back_to_weekdays(env, capsys, caplog):
    class NoCal(FakeMarket):
        def trading_days(self, start, end):
            raise RuntimeError("calendar api down")

    b, _ = paper(env)
    env.deps.market = NoCal()
    b.submit(OrderPlan("ABC", "buy", 10, "seed", 70.0))
    journal(env).add_position_meta("ABC", date(2026, 10, 27), 70.0, "v001", None)
    env.msgs.action = "exit"
    with caplog.at_level("WARNING"):
        assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert "ABC" not in b.get_account().positions
    assert "counting weekdays" in caplog.text


def test_exit_check_journaled_without_new_candidate(env, capsys):
    from insiderbot.models import IssuerFeatures
    b, _ = paper(env)
    b.submit(OrderPlan("ABC", "buy", 10, "seed", 70.0))
    j = journal(env)
    j.add_txns([txn("c1", "Chief Executive Officer", accepted=NOW - timedelta(days=2))], "live")
    run = j.start_run("scan", "paper", "v001")
    f = IssuerFeatures(ticker="ABC", asof=NOW - timedelta(days=2), n_buyers=1, total_value=1e6,
                       has_ceo=True, has_cfo=False, has_chair=False, n_directors=0,
                       size_vs_holdings=1.0, first_buy_months=0.0, frac_10b5_1=0.0, ret_3m=None,
                       drawdown_52w=None, price=70.0, adv20_usd=None, mcap_usd=None, sector=None,
                       exchange=None, is_fund_like=False, accessions=("a",),
                       latest_accepted_at=NOW - timedelta(days=2))
    prior = j.add_candidate(run, f, {}, [], None, "live")
    j.add_position_meta("ABC", date(2026, 10, 30), 70.0, "v001", prior)
    plan_sale = txn("c1", "Chief Executive Officer", code="S", ten_b5=True, acc="s1")
    env.deps.edgar = FakeEdgar([plan_sale])
    env.msgs.action = "pass"
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    rows = journal(env).conn.execute("SELECT candidate_id, action FROM decisions").fetchall()
    assert [tuple(r) for r in rows] == [(prior, "hold")]


def test_config_failure_pushes(monkeypatch, capsys):
    calls = []

    def bad():
        raise ValueError("bad yaml")

    monkeypatch.setattr(cli, "default_deps", bad)
    monkeypatch.setattr(cli.notify, "push", lambda *a, **k: calls.append((a, k)) or True)
    assert cli.main(["scan"]) == 1
    assert last_line(capsys) == "ERROR: config: bad yaml"
    assert calls and calls[0][1]["priority"] >= 5


def test_adopted_position_uses_last_filled_buy_date(env, capsys):
    b, _ = paper(env)
    b.submit(OrderPlan("XYZ", "buy", 10, "seed", 70.0))
    j = journal(env)
    run = j.start_run("scan", "paper", "v001")
    oid = j.add_order(run, SubmittedOrder(OrderPlan("XYZ", "buy", 10, "x", 70.0), "b1", "filled", None))
    j.conn.execute("UPDATE orders SET created_at=? WHERE id=?", ("2026-10-20T15:00:00+00:00", oid))
    j.conn.commit()
    b.submit_stop("XYZ", 10, 60.0)
    env.deps.edgar = FakeEdgar([])
    assert cli.main(["scan"], env.deps) == 0, capsys.readouterr()
    assert journal(env).position_meta("XYZ")["entry_date"] == "2026-10-20"
