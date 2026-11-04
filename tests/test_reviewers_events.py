"""Phase 3 for insider-bot: a Risk Reviewer is a second vetoer on every PM buy (it can only remove a buy),
and intraday Event Reviews on held names can vote for an exit only when two agents agree."""
from __future__ import annotations

import datetime as dt
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from insiderbot import events
from insiderbot.agents.llm import LLM
from insiderbot.agents.schemas import (AnalystNote, BuyApproval, DebateCase, EventCall, PMOut, ResearchOut)
from insiderbot.agents.team import AgentTeam
from insiderbot.config import Agents, Features, Settings
from insiderbot.journal.store import Journal
from insiderbot.memory.scorecards import build_scorecards
from insiderbot.models import AgentDecision, IssuerFeatures


class Cfg:
    agents = Agents()
    reviewers = Settings().reviewers
    events = Settings().events


def feats():
    now = datetime(2026, 11, 3, tzinfo=timezone.utc)
    return IssuerFeatures(
        ticker="ABC", asof=now, n_buyers=2, total_value=500000.0, has_ceo=True, has_cfo=False,
        has_chair=False, n_directors=1, size_vs_holdings=0.2, first_buy_months=30.0, frac_10b5_1=0.0,
        ret_3m=-0.1, drawdown_52w=-0.3, price=10.0, adv20_usd=5e6, mcap_usd=1e9, sector="Tech",
        exchange="NYSE", is_fund_like=False, accessions=("a1",), latest_accepted_at=now)


class Msgs:
    def __init__(self, action="buy", approve=True, risk_exc=None):
        self.action, self.approve, self.risk_exc, self.calls = action, approve, risk_exc, []

    def parse(self, **kw):
        self.calls.append(kw)
        sch = kw["output_format"]
        if sch is PMOut:
            out = PMOut(action=self.action, conviction=4, thesis="t", rule_flags_addressed="a",
                        what_would_change_my_mind="m")
        elif sch is AnalystNote:
            out = AnalystNote(summary="ok", positives=[], negatives=[], red_flags=[])
        elif sch is DebateCase:
            out = DebateCase(side="x", argument="a", key_risks=[])
        elif sch is BuyApproval:
            if self.risk_exc:
                raise self.risk_exc
            out = BuyApproval(approve=self.approve, reason="dilution risk" if not self.approve else "fine")
        else:
            out = ResearchOut(change=False, rationale="r", new_weights={}, new_params={}, what_we_learned="w")
        u = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=out, usage=u, stop_reason="end_turn")


def _team(msgs):
    llm = LLM(100_000, client=SimpleNamespace(beta=SimpleNamespace(messages=msgs)), max_monthly_usd=40,
              month_cost=lambda: 0.0)
    return AgentTeam(llm, Cfg())


CTX = {"fundamentals": "widgets", "news": ["n"], "bars_summary": "up"}


def test_buy_needs_risk_approval():
    m = Msgs(approve=False)
    d = _team(m).review(feats(), 72.5, [], [], CTX, False, reviewers=True, portfolio={"held": ["XYZ"]})
    assert d.action == "pass" and d.reports["pm_raw_action"] == "buy"
    assert d.reports["risk_review"] == {"approve": False, "reason": "dilution risk"}
    rk = [c for c in m.calls if c["output_format"] is BuyApproval][0]
    assert rk["model"] == "claude-opus-5-5" and "XYZ" in rk["messages"][0]["content"]
    assert "instruction" in rk["system"][0]["text"].lower()


def test_risk_approves_buy_stands():
    d = _team(Msgs(approve=True)).review(feats(), 72.5, [], [], CTX, False, reviewers=True, portfolio={})
    assert d.action == "buy" and d.reports["risk_review"]["approve"] is True


def test_reviewer_only_on_buys_and_off_by_default():
    m = Msgs(action="pass")
    _team(m).review(feats(), 72.5, [], [], CTX, False, reviewers=True, portfolio={})
    m2 = Msgs(action="buy")
    _team(m2).review(feats(), 72.5, [], [], CTX, False)
    assert not any(c["output_format"] is BuyApproval for c in m.calls + m2.calls)


def test_risk_error_is_veto_but_not_pm_veto(tmp_path):
    d = _team(Msgs(risk_exc=RuntimeError("down"))).review(feats(), 72.5, [], [], CTX, False, reviewers=True,
                                                          portfolio={})
    assert d.action == "pass" and d.reports["risk_review"]["error"]
    j = Journal(tmp_path / "j.db")
    cid = j.add_candidate(1, feats(), {"v1": 72.5}, [], None, "live")
    j.add_decision(1, cid, d, 0.0)
    j.conn.execute("INSERT INTO outcomes VALUES (?,?,?,?)", (cid, 20, 0.05, 0.04))
    j.conn.commit()
    c = build_scorecards(j)
    assert c["agents"]["pm"]["n"] == 1 and c["agents"]["pm"]["hit_rate"] == 1.0   # the PM's raw buy, scored +1
    assert "risk_reviewer" not in c["agents"]                                        # an error is not a call


def test_scorecards_add_risk_reviewer(tmp_path):
    j = Journal(tmp_path / "j.db")
    for ex, approve in ((0.04, True), (-0.03, False), (0.02, False)):
        d = _team(Msgs(approve=approve)).review(feats(), 72.5, [], [], CTX, False, reviewers=True, portfolio={})
        cid = j.add_candidate(1, feats(), {"v1": 72.5}, [], None, "live")
        j.add_decision(1, cid, d, 0.0)
        j.conn.execute("INSERT INTO outcomes VALUES (?,?,?,?)", (cid, 20, ex, ex))
    j.conn.commit()
    rr = build_scorecards(j)["agents"]["risk_reviewer"]
    assert rr["n"] == 3 and rr["hit_rate"] == pytest.approx(2 / 3)


# ------------------------------------------------------------------ Event Review
def test_event_triggers_move_and_8k():
    tr = events.find_triggers(["ABC", "DEF", "GHI"], {"ABC": 9.3, "DEF": 9.8, "GHI": 10.0},
                              {"ABC": 10.0, "DEF": 10.0, "GHI": 10.0}, {"ABC": 0.02, "DEF": 0.02, "GHI": 0.02},
                              {"GHI": [{"form": "8-K", "date": "2026-11-04", "url": "u"}]}, 2.5)
    assert [(t.symbol, t.kind) for t in tr] == [("GHI", "filing"), ("ABC", "move")]


def test_refs_from_bars():
    idx = pd.bdate_range("2026-09-01", periods=30)
    close = pd.Series(100 * np.cumprod(1 + np.tile([0.01, -0.01], 15)), index=idx)
    prev, sig = events.refs({"ABC": pd.DataFrame({"close": close})})
    assert prev["ABC"] == pytest.approx(close.iloc[-1])
    assert sig["ABC"] == pytest.approx(close.pct_change().dropna().tail(20).std())


class FakeTeam:
    def __init__(self, a="exit", b="exit", exc=None):
        self.a, self.b, self.exc, self.calls = a, b, exc, []

    def event_review(self, ticker, trigger, kind, position, filing_text, memory=None):
        self.calls.append((ticker, kind))
        if self.exc:
            raise self.exc
        a = EventCall(action=self.a, reason="ra")
        b = None if self.a == "hold" else EventCall(action=self.b, reason="rb")
        return a, {"call": a.model_dump(), "touched": []}, b


def _held(*syms):
    return {x: SimpleNamespace(symbol=x, qty=10, avg_price=10.0, market_value=93.0) for x in syms}


def _run(j, team, held, mids, day=dt.date(2026, 11, 4), filings=None, cap=6):
    return events.run_event_reviews(
        team, j, held, mids, prev_close={x: 10.0 for x in held}, sigma={x: 0.02 for x in held},
        day=day, run_id=1, mode="paper", cap=cap, k=2.5, notes=[],
        filings_fn=lambda sym, since: (filings or {}).get(sym, []))


def test_event_exit_needs_both(tmp_path):
    j = Journal(tmp_path / "j.db")
    votes = _run(j, FakeTeam("exit", "hold"), _held("ABC"), {"ABC": 9.3})
    assert votes == {}
    row = dict(j.conn.execute("SELECT * FROM events").fetchone())
    assert row["action"] == "hold" and json.loads(row["analyst"])["call"]["action"] == "exit"


def test_event_exit_feeds_exits_check(tmp_path):
    j = Journal(tmp_path / "j.db")
    # the vote goes to the existing exits.check, which applies min_hold_days_for_agent_exit
    assert _run(j, FakeTeam("exit", "exit"), _held("ABC"), {"ABC": 9.3}) == {"ABC": True}
    assert dict(j.conn.execute("SELECT action FROM events").fetchone())["action"] == "exit"


def test_event_dedupe_and_cap(tmp_path):
    j = Journal(tmp_path / "j.db")
    t = FakeTeam("hold")
    _run(j, t, _held("ABC"), {"ABC": 9.3})
    _run(j, t, _held("ABC"), {"ABC": 9.2})
    assert len(t.calls) == 1                                       # one review per symbol, kind and day
    f = {"ABC": [{"form": "8-K", "date": "2026-11-04", "url": "https://sec.gov/x", "title": "8-K"}]}
    _run(j, t, _held("ABC"), {"ABC": 10.0}, filings=f)
    _run(j, t, _held("ABC"), {"ABC": 10.0}, day=dt.date(2026, 11, 5), filings=f)
    assert len(t.calls) == 2                                       # a filing is reviewed once, ever
    t2 = FakeTeam("hold")
    syms = [f"S{i}" for i in range(8)]
    _run(Journal(tmp_path / "j2.db"), t2, _held(*syms), {x: 9.3 for x in syms})
    assert len(t2.calls) == 6


def test_event_error_never_raises(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert _run(j, FakeTeam(exc=RuntimeError("cli down")), _held("ABC"), {"ABC": 9.3}) == {}
    assert dict(j.conn.execute("SELECT action FROM events").fetchone())["action"] == "error"


def test_team_event_review_calls(tmp_path):
    class LLMStub:
        def __init__(self):
            self.calls = []

        def research(self, **k):
            self.calls.append(("research", k))
            return EventCall(action="exit", reason="offering"), [{"tool": "WebFetch", "url": "u"}]

        def structured(self, **k):
            self.calls.append(("structured", k))
            return EventCall(action="hold", reason="noise")

    st = LLMStub()
    team = AgentTeam(st, Cfg())
    a, arec, b = team.event_review("ABC", "-7% (-3.5 sigma)", "move", {"qty": 10}, None)
    assert (a.action, b.action) == ("exit", "hold") and arec["touched"][0]["url"] == "u"
    assert st.calls[0][1]["model"] == "claude-sonnet-5-5" and st.calls[1][1]["model"] == "claude-opus-5-5"
    assert "<untrusted_data>" in st.calls[0][1]["user"]


def test_control_settings_resets_phase3_flags():
    from insiderbot import cli

    s = Settings().model_copy(update={"features": Features(research=True, reviewers=True, event_review=True)})
    f = cli.control_settings(s).features
    assert not (f.research or f.reviewers or f.event_review)


def test_live_config_enables_phase3():
    from insiderbot import config

    f = config.load().features
    assert f.research and f.reviewers and f.event_review
