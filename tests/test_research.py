from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from insiderbot import formula as fml
from insiderbot.agents.schemas import ResearchOut
from insiderbot.config import ROOT, Bounds, Settings
from insiderbot.journal.store import Journal
from insiderbot.models import IssuerFeatures
from insiderbot.research import outcomes, replay, review

TODAY = date(2026, 12, 31)
ACCEPTED = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)  # 06:00 NY, before the open


@pytest.fixture
def v001():
    return fml.load("v001", ROOT / "formula")


def bars(closes, start="2026-09-28", opens=None, lows=None, highs=None):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "open": opens if opens is not None else c, "high": highs if highs is not None else c * 1.001,
        "low": lows if lows is not None else c * 0.999, "close": c, "volume": 1e6}, index=idx)


class FakeMarket:
    def __init__(self, data):
        self.data = data

    def bars(self, symbols, start, end, now=None):
        return {s: self.data[s] for s in symbols if s in self.data}


def feats(n_buyers=1, ceo=False, accepted=ACCEPTED, ticker="ABC"):
    return IssuerFeatures(
        ticker=ticker, asof=accepted, n_buyers=n_buyers, total_value=1e5, has_ceo=ceo, has_cfo=False,
        has_chair=False, n_directors=0, size_vs_holdings=0.2, first_buy_months=12.0, frac_10b5_1=0.0,
        ret_3m=None, drawdown_52w=None, price=10.0, adv20_usd=None, mcap_usd=None, sector=None,
        exchange=None, is_fund_like=False, accessions=("a",), latest_accepted_at=accepted, tradable=True)


# --- outcomes --------------------------------------------------------------

def test_outcome_excess_vs_spy(tmp_path):
    j = Journal(tmp_path / "j.db")
    run = j.start_run("scan", "paper", "v001")
    cid = j.add_candidate(run, feats(), {}, [], None, "live")
    n = 30
    abc = bars([100 * 1.01 ** i for i in range(n)])
    spy = bars([400 * 1.002 ** i for i in range(n)])
    stored = outcomes.update(j, FakeMarket({"ABC": abc, "SPY": spy}), TODAY)
    assert stored == 3  # 5, 10, 20 matured (40 not)
    # entry = Oct 1 close = position 3
    o = j.outcomes_all()[cid]
    assert set(o) == {5, 10, 20}
    ret = 1.01 ** 5 - 1
    assert o[5][0] == pytest.approx(ret)
    assert o[5][1] == pytest.approx(ret - (1.002 ** 5 - 1))
    assert outcomes.update(j, FakeMarket({"ABC": abc, "SPY": spy}), TODAY) == 0  # idempotent


def test_outcome_skips_missing_data(tmp_path):
    j = Journal(tmp_path / "j.db")
    run = j.start_run("scan", "paper", "v001")
    j.add_candidate(run, feats(), {}, [], None, "live")
    assert outcomes.update(j, FakeMarket({"SPY": bars([400.0] * 30)}), TODAY) == 0
    assert outcomes.update(j, FakeMarket({"ABC": bars([10.0] * 30)}), TODAY) == 0


# --- replay ----------------------------------------------------------------

def path(opens, lows, closes):
    n = len(closes)
    spy = [400.0] * n
    idx = pd.bdate_range("2026-10-01", periods=n)
    return pd.DataFrame({"open": opens, "high": [max(o, c) for o, c in zip(opens, closes)], "low": lows,
                         "close": closes, "spy_open": spy, "spy_close": spy}, index=idx, dtype=float)


def test_replay_stop_gap_fill(v001):
    # stop = 88; day 1 gaps open to 80 -> fills at 80, not 88
    p = path([100, 80, 80], [99, 78, 78], [100, 82, 82])
    r = replay.simulate(p, v001, cost_bps=0)
    assert r["reason"] == "stop" and r["days"] == 1
    assert r["ret"] == pytest.approx(-0.20)
    # intraday touch without gap fills at the stop price
    p = path([100, 95, 95], [99, 87, 90], [100, 92, 92])
    r = replay.simulate(p, v001, cost_bps=0)
    assert r["reason"] == "stop" and r["ret"] == pytest.approx(-0.12)
    assert r["excess"] == pytest.approx(-0.12)


def test_replay_trailing(v001):
    # peak close 126 >= 125 (take-profit); 112 <= 126 * 0.9 -> exit at that close
    closes = [100, 110, 126, 112, 120]
    p = path(closes, [c * 0.99 for c in closes], closes)
    r = replay.simulate(p, v001, cost_bps=0)
    assert r["reason"] == "trailing" and r["days"] == 3
    assert r["ret"] == pytest.approx(0.12)
    assert r["mdd"] == pytest.approx(112 / 126 - 1)


def test_replay_time_stop_and_nodata(v001):
    closes = [100.0] * 50
    r = replay.simulate(path(closes, closes, closes), v001, cost_bps=0)
    assert r["reason"] == "time" and r["days"] == v001.time_stop_days
    assert replay.simulate(None, v001)["ret"] is None


def _cands(n):
    out = []
    for i in range(n):
        cl = i % 4
        out.append({"id": i, "features": feats(n_buyers=cl + 1, ceo=cl < 2),
                    "excess": {10: cl / 10}, "path": None})
    return out


def _pair(v001):
    base = v001.model_copy(update={"buy_threshold": 0.0, "review_threshold": 0.0})
    other = base.model_copy(update={"weights": {**base.weights, "ceo_cfo": 0.0}})
    return base, other


def test_compare_needs_min_n(v001):
    a, o = _pair(v001)
    small = replay.compare(_cands(8), a, o, 10, min_n=30)
    assert small["n"] == 8 and small["winner"] == "undecided"
    big = replay.compare(_cands(40), a, o, 10, min_n=30)
    assert big["other"]["rank_ic"] > big["active"]["rank_ic"]
    assert big["winner"] == "other"
    assert replay.compare(_cands(40), o, a, 10, min_n=30)["winner"] == "active"


# --- review ----------------------------------------------------------------

class FakeTeam:
    def __init__(self, out=None):
        self.out, self.calls = out, 0

    def research(self, stats, yaml_text):
        self.calls += 1
        return self.out


def setup_dirs(tmp_path, shadow_other=False, zero=True, shadow_update=None, base_update=None):
    fdir = tmp_path / "formula"
    upd = ({"buy_threshold": 0.0, "review_threshold": 0.0,
            "tiers": [(0.0, 0.75), (0.5, 1.0), (1.0, 1.25)]} if zero else {})
    upd.update(base_update or {})
    base = fml.load("v001", ROOT / "formula").model_copy(update=upd)
    fml.save(base, fdir)
    fml.set_active("v001", fdir)
    if shadow_other:
        if shadow_update is None:  # a legal weight change (0.06 is the max step at 0.2)
            shadow_update = {"weights": {**base.weights, "ceo_cfo": 0.14}}
        fml.save(base.model_copy(update={**shadow_update, "parent": "v001", "author": "review-agent"}), fdir)
        fml.set_shadow("v002", fdir)
    return fdir


def add_live(j, n, source="live", offset=0):
    run = j.start_run("scan", "paper", "v001")
    for i in range(n):
        cl = i % 4
        cid = j.add_candidate(run, feats(n_buyers=cl + 1, ceo=cl < 2, ticker=f"T{offset + i}"),
                              {}, [], None, source)
        j.set_outcome(cid, 10, cl / 10, cl / 10)


def go(j, team, fdir, tmp_path, max_weeks=10):
    st = Settings(bounds=Bounds(buy_threshold=(0, 95), review_threshold=(0, 90), shadow_max_weeks=max_weeks))
    return review.run(j, None, team, st, TODAY, formula_dir=fdir, reports_dir=tmp_path / "reports")


def test_review_no_proposal_below_40(tmp_path):
    j, fdir, team = Journal(tmp_path / "j.db"), setup_dirs(tmp_path), FakeTeam()
    add_live(j, 39)
    s = go(j, team, fdir, tmp_path)
    assert s["action"] == "no_proposal" and team.calls == 0
    assert (tmp_path / "reports" / "weekly" / "2026-12-31.md").exists()


def test_backfill_rows_excluded(tmp_path):
    j, fdir, team = Journal(tmp_path / "j.db"), setup_dirs(tmp_path), FakeTeam()
    add_live(j, 39)
    add_live(j, 30, source="backfill", offset=100)
    s = go(j, team, fdir, tmp_path)
    assert s["matured"] == 39 and s["action"] == "no_proposal" and team.calls == 0


def test_review_rejects_invalid_proposal(tmp_path):
    out = ResearchOut(change=True, rationale="r", new_weights={"cluster": 0.9}, new_params={},
                      what_we_learned="x")
    j, fdir, team = Journal(tmp_path / "j.db"), setup_dirs(tmp_path), FakeTeam(out)
    add_live(j, 40)
    s = go(j, team, fdir, tmp_path)
    assert team.calls == 1 and s["action"] == "rejected"
    assert fml.shadow(fdir) is None and not (fdir / "v002.yaml").exists()
    ev = j.formula_events(event="rejected")
    assert len(ev) == 1 and any("too large" in e for e in ev[0]["detail"]["errors"])


def test_review_valid_proposal_becomes_shadow(tmp_path):
    out = ResearchOut(change=True, rationale="r", new_weights={"cluster": 0.30}, new_params={},
                      what_we_learned="x")
    j, fdir, team = Journal(tmp_path / "j.db"), setup_dirs(tmp_path, zero=False), FakeTeam(out)
    add_live(j, 40)
    s = go(j, team, fdir, tmp_path)
    assert s["action"] == "shadow"
    sh = fml.shadow(fdir)
    assert sh.version == "v002" and sh.parent == "v001" and sh.author == "review-agent"
    assert sh.weights["cluster"] == 0.30
    assert [e["event"] for e in j.formula_events(version="v002")] == ["proposed", "shadow"]


def test_review_promotes_winner(tmp_path):
    j, fdir, team = Journal(tmp_path / "j.db"), setup_dirs(tmp_path, shadow_other=True), FakeTeam()
    j.add_formula_event("v002", "shadow", {"since": "2026-01-01"})
    add_live(j, 36)
    s = go(j, team, fdir, tmp_path)
    assert s["action"] == "promoted", (s["action"], s.get("errors"))
    assert fml.active(fdir).version == "v002" and fml.shadow(fdir) is None
    assert j.formula_events(event="promoted")[0]["version"] == "v002"


def test_review_keeps_or_retires_shadow(tmp_path):
    j, fdir = Journal(tmp_path / "j.db"), setup_dirs(tmp_path, shadow_other=True)
    j.add_formula_event("v002", "shadow", {"since": "2026-12-01"})
    assert go(j, FakeTeam(), fdir, tmp_path)["action"] == "keep"
    assert fml.shadow(fdir) is not None
    j.add_formula_event("v002", "shadow", {"since": "2026-01-01"})  # 10+ weeks old, no evidence
    assert go(j, FakeTeam(), fdir, tmp_path)["action"] == "retired"
    assert fml.shadow(fdir) is None


def test_weekly_and_control_book_smoke(tmp_path, v001):
    from insiderbot.research import stats
    j = Journal(tmp_path / "j.db")
    run = j.start_run("scan", "paper", "v001")
    fm = v001.model_copy(update={"buy_threshold": 0.0})
    for i in range(6):
        cid = j.add_candidate(run, feats(n_buyers=i % 4 + 1, ticker="ABC"), {}, [], None, "live")
        j.set_outcome(cid, 10, 0.01 * i, 0.01 * i)
    mk = FakeMarket({"ABC": bars([100 + i for i in range(60)]), "SPY": bars([400.0] * 60)})
    w = stats.weekly(j, fm, mk, TODAY)
    assert w["n_matured_10d"] == 6 and w["control_book"]["n"] == 6
    assert w["control_book"]["mean_excess"] > 0
    assert stats.weekly(j, fm)["real_book"]["n"] == 0


def test_param_only_shadow_wins_on_mean_excess(tmp_path):
    j = Journal(tmp_path / "j.db")
    fdir = setup_dirs(tmp_path, True, base_update={"buy_threshold": 30.0},
                      shadow_update={"buy_threshold": 33.0})
    j.add_formula_event("v002", "shadow", {"since": "2026-01-01"})
    add_live(j, 36)
    s = go(j, FakeTeam(), fdir, tmp_path)
    assert s["compare"]["primary"] == "mean_excess"
    assert s["action"] == "promoted", (s["action"], s.get("errors"))
    assert fml.active(fdir).buy_threshold == 33.0


def test_param_only_shadow_loses_and_is_retired_at_once(tmp_path):
    j = Journal(tmp_path / "j.db")
    fdir = setup_dirs(tmp_path, True, base_update={"buy_threshold": 30.0},
                      shadow_update={"buy_threshold": 37.0})
    j.add_formula_event("v002", "shadow", {"since": "2026-09-01"})
    add_live(j, 36)
    s = go(j, FakeTeam(), fdir, tmp_path, max_weeks=100)  # not old: retired for losing
    assert s["compare"]["winner"] == "active" and s["action"] == "retired"
    assert fml.active(fdir).version == "v001" and fml.shadow(fdir) is None
    assert j.formula_events(event="retired")[0]["detail"]["reason"] == "lost"


def test_exit_param_shadow_uses_replay(v001):
    a = v001
    o = v001.model_copy(update={"stop_loss_pct": 0.10})
    assert replay.primary_metric(a, o) == "replay_excess"
    assert replay.primary_metric(a, a) is None
    assert replay.primary_metric(a, a.model_copy(update={"buy_threshold": 65.0})) == "mean_excess"
    # no replay paths -> primary metric missing -> never wins
    assert replay.compare(_cands(40), a, o, 10)["winner"] == "undecided"


def test_promotion_revalidates_tampered_shadow(tmp_path):
    j = Journal(tmp_path / "j.db")
    # ceo_cfo 0.2 -> 0.0 wins the comparison but breaks the max-change bound
    fdir = setup_dirs(tmp_path, True, shadow_update={"weights": {**fml.load("v001", ROOT / "formula").weights,
                                                                 "ceo_cfo": 0.0}})
    j.add_formula_event("v002", "shadow", {"since": "2026-01-01"})
    add_live(j, 36)
    s = go(j, FakeTeam(), fdir, tmp_path)
    assert s["action"] == "retired" and any("too large" in e for e in s["errors"])
    assert fml.active(fdir).version == "v001" and fml.shadow(fdir) is None
    assert j.formula_events(event="promoted") == []
    assert "errors" in j.formula_events(event="retired")[0]["detail"]


def test_proposal_rejects_bool_and_noop(tmp_path, v001):
    out = ResearchOut(change=True, rationale="r", new_weights={}, new_params={"time_stop_days": 40.4},
                      what_we_learned="")
    assert review.build_proposal(v001, out, TODAY) == (None, [review.NOOP])
    # pydantic coerces bools in ResearchOut, but build_proposal also rejects them directly
    out = ResearchOut(change=True, rationale="r", new_weights={}, new_params={}, what_we_learned="")
    object.__setattr__(out, "new_weights", {"cluster": True})
    assert "not a finite number" in review.build_proposal(v001, out, TODAY)[1][0]
