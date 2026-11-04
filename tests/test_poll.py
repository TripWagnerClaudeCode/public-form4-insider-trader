"""`scan --slot poll`: every 5 minutes in market hours, read EDGAR's live feed and apply the SAME
rules as the scheduled scans, just sooner (strategy freeze)."""
from __future__ import annotations

import dataclasses
import fcntl
from datetime import date, datetime, timedelta, timezone

from insiderbot import cli

from test_cli import NOW, FakeEdgar, FakeMarket, cluster, env, journal, last_line, paper  # noqa: F401


class FeedEdgar(FakeEdgar):
    """FakeEdgar plus the live current-filings feed."""

    def __init__(self, txns, feed_fail=False, **kw):
        super().__init__(txns, **kw)
        self.feed_calls, self.feed_fail = [], feed_fail

    def fetch_current(self, since, until):
        self.feed_calls.append((since, until))
        if self.feed_fail:
            raise ConnectionError("feed down")
        return [t for t in self.txns if since < t.accepted_at <= until]


def _poll_env(env, txns=None, on=True, **kw):
    env.s.features.poll = on
    env.deps.edgar = FeedEdgar(cluster() if txns is None else txns, **kw)
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    return env.deps.edgar


def _at(env, hh, mm):   # Tue 2026-11-03 is EST (UTC-5)
    t = datetime(2026, 11, 3, hh + 5, mm, tzinfo=timezone.utc)
    env.deps.now = lambda: t


def test_poll_disabled_flag(env, capsys):
    ed = _poll_env(env, on=False)
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0
    assert "skipped_disabled" in last_line(capsys)
    assert ed.calls == [] and ed.feed_calls == []


def test_poll_outside_window_skipped(env, capsys):
    ed = _poll_env(env)
    for hh, mm in ((9, 40), (15, 50)):
        _at(env, hh, mm)
        assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0
        assert "skipped_window" in last_line(capsys)
    assert ed.feed_calls == [] and ed.calls == []


def test_poll_uses_current_feed_and_buys_by_the_same_rules(env, capsys):
    ed = _poll_env(env)
    from insiderbot.journal.store import Journal
    Journal(env.tmp / "data" / "journal.dry-run.db").set_watermark(NOW - timedelta(hours=3))
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0, last_line(capsys)
    assert ed.feed_calls and ed.calls == []
    assert "buys=1" in last_line(capsys)


def test_poll_falls_back_to_index_on_feed_error(env, capsys):
    ed = _poll_env(env, feed_fail=True)
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0, last_line(capsys)
    assert ed.feed_calls and ed.calls


def test_poll_locked_is_silent(env, capsys):
    _poll_env(env)
    lock = env.tmp / "data" / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cli.main(["scan", "--slot", "poll"], env.deps) == 0
        assert last_line(capsys) == "OK scan skipped_locked (another job running)"
    assert env.push.pushes == []
    assert journal(env).get_watermark() is None


def test_poll_no_orders_no_report_no_push(env, capsys):
    _poll_env(env, txns=[])
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0, last_line(capsys)
    assert not list((env.tmp / "reports").glob("scan-*.md"))
    assert env.push.pushes == []


def test_poll_does_not_rebuy_after_scan(env, capsys):
    b, _ = paper(env)
    _poll_env(env)
    assert cli.main(["scan", "--slot", "am", "--force"], env.deps) == 0, last_line(capsys)
    assert "ABC" in b.get_account().positions
    n_cand = len(journal(env).candidates())
    assert cli.main(["scan", "--slot", "poll"], env.deps) == 0, last_line(capsys)
    assert "buys=0" in last_line(capsys)
    assert len(journal(env).candidates()) == n_cand          # the same accession is never a new candidate


def test_poll_runs_deterministic_exits(env, capsys):
    b, _ = paper(env)
    from insiderbot.models import OrderPlan
    b.submit(OrderPlan("XYZ", "buy", 100, "seed", 70.0))
    b.submit_stop("XYZ", 100, 60.0)
    journal(env).add_position_meta("XYZ", date(2026, 10, 1), 100.0, "v001", None)   # -30%: stop rule
    env.deps.market = FakeMarket(tickers=("ABC", "XYZ"), quotes={"ABC": (69.9, 70.1), "XYZ": (69.9, 70.1)})
    _poll_env(env, txns=[])
    assert cli.main(["scan", "--slot", "poll"], env.deps) == 0, last_line(capsys)
    assert "XYZ" not in b.get_account().positions
    assert journal(env).closed_positions()[0]["exit_reason"] == "stop"


# ----------------------------------------------------------------------------- C1: feed shorter than the gap
class ShortFeed(FeedEdgar):
    """A live feed that only reaches back `hours` (EDGAR's is ~24h); older filings exist only in the index."""

    def __init__(self, txns, hours, **kw):
        super().__init__(txns, **kw)
        self.hours = hours

    def fetch_current(self, since, until):
        self.feed_calls.append((since, until))
        floor = until - timedelta(hours=self.hours)
        self.last_reached_since = since >= floor
        return [t for t in self.txns if max(since, floor) < t.accepted_at <= until]


def test_poll_after_a_weekend_also_reads_the_daily_index(env, capsys):
    # Friday after-close cluster; Monday 09:45 poll; the watermark is from Friday afternoon
    fri = datetime(2026, 10, 30, 21, 30, tzinfo=timezone.utc)                 # Fri 16:30 ET
    txns = [dataclasses.replace(t, accepted_at=fri, tx_date=fri.date()) for t in cluster()]
    ed = ShortFeed(txns, hours=24)
    env.s.features.poll = True
    env.deps.edgar = ed
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    _ = journal(env)
    from insiderbot.journal.store import Journal
    Journal(env.tmp / "data" / "journal.dry-run.db").set_watermark(datetime(2026, 10, 30, 20, 45, tzinfo=timezone.utc))
    mon = datetime(2026, 11, 2, 14, 50, tzinfo=timezone.utc)                  # Mon 09:50 ET
    env.deps.now = lambda: mon
    env.deps.market = FakeMarket(cal=(datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc),
                                      datetime(2026, 11, 2, 21, 0, tzinfo=timezone.utc)))
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0, last_line(capsys)
    assert ed.feed_calls and ed.calls                                          # the index was read too
    assert "buys=1" in last_line(capsys)                                       # Friday's cluster is not lost


def test_poll_recent_watermark_feed_only(env, capsys):
    ed = ShortFeed(cluster(), hours=24)
    env.s.features.poll = True
    env.deps.edgar = ed
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    from insiderbot.journal.store import Journal
    Journal(env.tmp / "data" / "journal.dry-run.db").set_watermark(NOW - timedelta(hours=3))
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0, last_line(capsys)
    assert ed.feed_calls and ed.calls == []


def test_poll_merge_does_not_double_count(env, capsys):
    ed = ShortFeed(cluster(), hours=24)
    ed.last_reached_since = False
    env.s.features.poll = True
    env.deps.edgar = ed
    env.deps.secrets = env.deps.secrets.model_copy(update={"anthropic_key": "x"})
    from insiderbot.journal.store import Journal
    Journal(env.tmp / "data" / "journal.dry-run.db").set_watermark(NOW - timedelta(hours=40))
    assert cli.main(["scan", "--slot", "poll", "--mode", "dry-run"], env.deps) == 0, last_line(capsys)
    j = Journal(env.tmp / "data" / "journal.dry-run.db")
    assert j.conn.execute("SELECT COUNT(*) FROM txns").fetchone()[0] == len(cluster())
    assert len(j.candidates()) == 1
