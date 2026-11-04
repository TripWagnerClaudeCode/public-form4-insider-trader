"""Phase 3 final-review fixes (insider-bot): each test reproduces one finding."""
from __future__ import annotations

import datetime as dt
import stat
import sys
import time
from datetime import date
from types import SimpleNamespace

import pytest

from insiderbot import cli, events
from insiderbot.agents.llm import LLM
from insiderbot.agents.schemas import EventCall
from insiderbot.journal.store import Journal

from test_cli import FakeMarket, env, journal, last_line, paper  # noqa: F401
from test_poll import _poll_env


# finding 4: deterministic exits run before Event Reviews; a both-agree vote gets its own exit pass
def test_exits_run_before_event_reviews_then_vote_pass(env, capsys, monkeypatch):
    b, _ = paper(env)
    from insiderbot.models import OrderPlan

    for sym in ("XYZ", "ABC"):
        b.submit(OrderPlan(sym, "buy", 100, "seed", 70.0))
        b.submit_stop(sym, 100, 50.0)
    journal(env).add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)   # -30%: the stop rule
    journal(env).add_position_meta("ABC", date(2026, 10, 1), 72.0, "v001", None)    # no formula exit
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ"), quotes={"ABC": (69.9, 70.1), "XYZ": (69.9, 70.1)})
    _poll_env(env, txns=[])
    env.s.features.event_review = True
    seen = {}

    def fake_reviews(team, j, held, *a, **k):
        seen["held"] = sorted(held)
        seen["xyz_still_held"] = "XYZ" in b.get_account().positions
        seen["per_run"] = k.get("per_run")
        return {"ABC": True}

    monkeypatch.setattr(events, "run_event_reviews", fake_reviews)
    assert cli.main(["scan", "--slot", "poll"], env.deps) == 0, last_line(capsys)
    assert seen["xyz_still_held"] is False and seen["held"] == ["ABC"]    # XYZ's stop exit already ran
    assert seen["per_run"] == 2
    assert "ABC" not in b.get_account().positions                         # the vote's own exit pass sold it


class FakeTeam:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, []

    def event_review(self, ticker, trigger, kind, position, filing_text, memory=None):
        self.calls.append(ticker)
        if self.exc:
            e, self.exc = self.exc, None
            raise e
        a = EventCall(action="hold", reason="r")
        return a, {"call": a.model_dump()}, None


def _held(*syms):
    return {x: SimpleNamespace(qty=10, avg_price=10.0, market_value=93.0) for x in syms}


def _run(j, team, held, mids, filings=None, per_run=None, day=dt.date(2026, 11, 4)):
    return events.run_event_reviews(team, j, held, mids, {x: 10.0 for x in held}, {x: 0.02 for x in held}, day, 1,
                                    "paper", 6, 2.5, [], filings_fn=lambda s, d: (filings or {}).get(s, []),
                                    per_run=per_run)


# finding 8: a failed review does not mark its filing as reviewed
def test_failed_filing_review_is_retried(tmp_path):
    j = Journal(tmp_path / "j.db")
    t = FakeTeam(exc=RuntimeError("cli hiccup"))
    f = {"ABC": [{"form": "8-K", "date": "2026-11-04", "url": "https://sec.gov/a", "title": "8-K"}]}
    _run(j, t, _held("ABC"), {"ABC": 10.0}, filings=f)
    _run(j, t, _held("ABC"), {"ABC": 10.0}, filings=f)
    assert t.calls == ["ABC", "ABC"]


def test_per_poll_cap(tmp_path):
    t = FakeTeam()
    syms = [f"S{i}" for i in range(5)]
    _run(Journal(tmp_path / "j.db"), t, _held(*syms), {x: 9.3 for x in syms}, per_run=2)
    assert len(t.calls) == 2


# finding 7: the research timeout holds even when the CLI never reads a large stdin
def test_research_timeout_with_unread_large_stdin(tmp_path):
    p = tmp_path / "fake_claude"
    p.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    llm = LLM(100_000, cli_path=str(p), max_monthly_usd=40, month_cost=lambda: 0.0)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        llm.research(model="m", effort="e", system="s", user="x" * 3_000_000, schema=EventCall, timeout_s=2)
    assert time.time() - t0 < 15
