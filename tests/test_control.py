"""Frozen v1 control for insider-bot: dry-run behavior with a persistent simulated book,
v1 prompts, its own journal, lock and kill switch, and no pushes."""
from __future__ import annotations

import json

import pytest

from insiderbot import cli
from insiderbot.journal.store import Journal

from test_cli import env, last_line, runs  # noqa: F401  (fixtures/helpers shared with the CLI tests)


def book(env):
    return env.tmp / "data" / "book.control.json"


def control_journal(env):
    return Journal(env.tmp / "data" / "journal.control.db")


def _key(env):
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})


def _init(env, capsys):
    assert cli.main(["reconcile", "--mode", "control", "--init-book"], env.deps) == 0, capsys.readouterr()


def test_control_book_persists_across_scans(env, capsys):
    _key(env)
    _init(env, capsys)
    assert cli.main(["scan", "--mode", "control"], env.deps) == 0, last_line(capsys)
    d = json.loads(book(env).read_text())
    assert "ABC" in d["positions"]
    qty = d["positions"]["ABC"]["qty"]
    assert cli.main(["scan", "--mode", "control"], env.deps) == 0, last_line(capsys)
    d2 = json.loads(book(env).read_text())
    assert d2["positions"]["ABC"]["qty"] == qty          # held, so not bought twice
    assert runs(env, j=control_journal(env))[-1]["mode"] == "control"


def test_control_uses_v1_prompts(env, capsys, monkeypatch):
    _key(env)
    _init(env, capsys)
    import insiderbot.agents.team as team

    seen = []
    orig = team.load_prompt
    monkeypatch.setattr(team, "load_prompt", lambda name, prompt_dir="prompts": seen.append(prompt_dir)
                        or orig(name, prompt_dir))
    assert cli.main(["scan", "--mode", "control"], env.deps) == 0, last_line(capsys)
    assert seen and set(seen) == {"prompts_v1"}


def test_control_never_touches_live_journal_or_paper_broker(env, capsys):
    _key(env)
    _init(env, capsys)
    assert cli.main(["scan", "--mode", "control"], env.deps) == 0, last_line(capsys)
    assert not (env.tmp / "data" / "journal.db").exists()        # live journal never opened
    assert not (env.tmp / "data" / "journal.dry-run.db").exists()
    assert env.push.pushes == []


def test_control_corrupt_book_errors(env, capsys):
    book(env).parent.mkdir(parents=True, exist_ok=True)
    book(env).write_text("{nope")
    assert cli.main(["scan", "--mode", "control"], env.deps) == 1
    assert runs(env, j=control_journal(env))[-1]["status"] == "error"


def test_control_missing_book_errors_without_init(env, capsys):
    assert cli.main(["scan", "--mode", "control"], env.deps) == 1
    assert not book(env).exists()


def test_control_settings_isolated(env):
    c = cli.control_settings(env.s)
    assert c.agents.prompt_dir == "prompts_v1"
    assert c.risk.kill_switch_file != env.s.risk.kill_switch_file
    assert c.paths.lock_file != env.s.paths.lock_file
    assert c.notify.ntfy is False
    # strategy-freeze guard: identical formula inputs, filters and risk limits apart from the kill switch file
    assert c.filters == env.s.filters and c.review == env.s.review
    assert c.risk.model_dump(exclude={"kill_switch_file"}) == env.s.risk.model_dump(exclude={"kill_switch_file"})


def test_control_reconcile_records_equity(env, capsys):
    _key(env)
    _init(env, capsys)
    assert cli.main(["scan", "--mode", "control"], env.deps) == 0, last_line(capsys)
    assert cli.main(["reconcile", "--mode", "control"], env.deps) == 0, last_line(capsys)
    rows = control_journal(env).conn.execute("SELECT equity FROM equity").fetchall()
    assert rows and rows[-1][0] > 0


def test_control_has_no_upgrade_features(env):
    env.s.features.poll = True
    assert not any(cli.control_settings(env.s).features.model_dump().values())


def test_control_cannot_poll(env, capsys):
    env.s.features.poll = True
    assert cli.main(["scan", "--mode", "control", "--slot", "poll"], env.deps) == 0
    assert "skipped_disabled" in last_line(capsys)
