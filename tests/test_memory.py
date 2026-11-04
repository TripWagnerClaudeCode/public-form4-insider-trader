"""Phase 2 (insider-bot): scorecards from decisions + outcomes, memory files, test guards."""
from __future__ import annotations

from datetime import date

import pytest

from insiderbot.config import Settings
from insiderbot.journal.store import Journal
from insiderbot.memory.files import Lesson, MemoryFiles
from insiderbot.memory.scorecards import build_scorecards, render_scorecards
from insiderbot.models import AgentDecision

from test_cli import NOW, FakeMarket, cluster  # noqa: F401


def _journal_with(tmp_path, rows):
    """rows: (ticker, action, conviction, excess20, red_flags)"""
    from insiderbot import features

    j = Journal(tmp_path / "j.db")
    rid = j.start_run("scan", "paper", "v001")
    for i, (tk, action, conv, ex, flags) in enumerate(rows):
        txns = cluster(tk)
        j.add_txns(txns, "live")
        (f,) = features.build(txns, FakeMarket(tickers=(tk,)), j, NOW, 14)
        cid = j.add_candidate(rid, f, {"v001": 65.0}, [], None, "live")
        j.add_decision(rid, cid, AgentDecision(tk, action, conv, "t", "m",
                                               {"analysts": {"filing": {"red_flags": flags}}}), 0.0)
        if ex is not None:
            j.set_outcome(cid, 20, ex + 0.01, ex)
    return j


def test_pm_buy_vs_pass_scoring(tmp_path):
    j = _journal_with(tmp_path, [("AAA", "buy", 4, 0.05, []), ("BBB", "buy", 2, -0.01, []),
                                 ("CCC", "pass", 1, -0.03, ["tiny"]), ("DDD", "pass", 1, 0.02, [])])
    c = build_scorecards(j)
    assert c["n_views"] == 4
    assert c["pm"]["buy"]["n"] == 2 and c["pm"]["buy"]["mean_excess"] == pytest.approx(0.02)
    assert c["pm"]["pass"]["n"] == 2 and c["pm"]["pass"]["mean_excess"] == pytest.approx(-0.005)
    assert c["pm"]["by_conviction"][4]["n"] == 1
    assert c["agents"]["pm"]["hit_rate"] == pytest.approx(0.5)     # AAA buy up, CCC pass down


def test_would_have_won_counts_positive_excess_passes(tmp_path):
    j = _journal_with(tmp_path, [("CCC", "pass", 1, -0.03, []), ("DDD", "pass", 1, 0.02, []),
                                 ("EEE", "pass", 1, 0.01, [])])
    assert build_scorecards(j)["pm"]["pass"]["would_have_won"] == pytest.approx(2 / 3)


def test_red_flag_split(tmp_path):
    j = _journal_with(tmp_path, [("AAA", "buy", 4, 0.05, []), ("CCC", "pass", 1, -0.03, ["entity buyer"])])
    rf = build_scorecards(j)["red_flags"]
    assert rf["with"]["n"] == 1 and rf["with"]["mean_excess"] == pytest.approx(-0.03)
    assert rf["without"]["n"] == 1


def test_unscored_and_exit_checks_ignored(tmp_path):
    j = _journal_with(tmp_path, [("AAA", "buy", 4, None, []), ("BBB", "hold", 3, 0.02, [])])
    assert build_scorecards(j)["n_views"] == 0


def test_empty_journal_no_track_record(tmp_path):
    c = build_scorecards(Journal(tmp_path / "j.db"))
    assert c["n_views"] == 0 and c["agents"] == {}
    assert "No track record yet" in render_scorecards(c)


def test_tickers_history(tmp_path):
    j = _journal_with(tmp_path, [("AAA", "buy", 4, 0.05, [])])
    h = build_scorecards(j)["symbols"]["AAA"]
    assert h[-1]["pm_direction"] == 1 and h[-1]["excess_univ"] == pytest.approx(0.05)


# ----------------------------------------------------------------------------- files (ported)
def _mf(tmp_path):
    return MemoryFiles(tmp_path / "mem", title="Lessons (insider)", today=lambda: "2026-11-20")


def _l(i, text="PM vetoes of CEO buys were wrong 7 of 10 times."):
    return Lesson(id=f"L-{i:04d}", text=text, added="2026-11-16", n=30, evidence="e", tags=["pm"])


def test_files_roundtrip_and_pinned(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1)])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text(p.read_text().replace("## Pinned by you\n", "## Pinned by you\nNo SPACs.\n"))
    lessons, pinned = mf.read_lessons()
    assert [x.id for x in lessons] == ["L-0001"] and "No SPACs." in pinned


def test_files_owner_deleted_never_returns(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1), _l(2, "Second.")])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text("\n".join(x for x in p.read_text().splitlines() if "L-0002" not in x
                           and not x.strip() == "evidence: e" or "L-0001" in x) + "\n")
    lessons, _ = mf.read_lessons()
    assert "L-0002" not in {x.id for x in lessons} and "L-0002" in mf.retired_ids()


def test_files_owner_added_line_gets_id(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1)])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text(p.read_text() + "- Avoid biotech insider buys before trial readouts.\n")
    new = mf.read_lessons()[0][-1]
    assert new.id == "L-0002" and new.tags == ["owner"]


def test_files_sync_conflict_ignored(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1)])
    (tmp_path / "mem" / "lessons.sync-conflict-20261120-1.md").write_text("## Active lessons\n- [L-0009] (2026-11-20, n=9) x\n")
    assert [x.id for x in mf.read_lessons()[0]] == ["L-0001"]


def test_files_staging(tmp_path):
    mf = _mf(tmp_path)
    mf.append_staging("2026-11-17", ["a"])
    mf.append_staging("2026-11-18", ["b"])
    mf.clear_staging("2026-11-17")
    assert len(mf.read_staging(None)) == 1


# ----------------------------------------------------------------------------- guards
def test_real_cli_blocked_in_tests():
    from insiderbot.agents.llm import LLM

    from insiderbot.agents.schemas import AnalystNote

    llm = LLM(100, cli_path="/usr/bin/true", max_monthly_usd=10, month_cost=lambda: 0.0)
    with pytest.raises(RuntimeError, match="disabled in tests"):
        llm._structured_cli(model="m", effort="low", system="s", user="u", schema=AnalystNote)


def test_tests_cannot_touch_real_memory_dir(tmp_path):
    from insiderbot.memory import memory_root

    s = Settings()
    assert s.memory.dir == "memory"
    assert str(tmp_path.parent.parent) in str(memory_root(s))


def test_memory_defaults_and_control_off():
    from insiderbot import cli

    s = Settings()
    assert s.features.memory is False
    s.features.memory = True
    assert cli.control_settings(s).features.memory is False
    assert date.today()  # noqa: keeps the import used
