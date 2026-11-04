"""Command-line runner: scan | reconcile | review | status | backfill.

Last stdout line of every job: `OK <job> <summary>` or `ERROR: <stage>: <reason>` (exit 1).
Every job except `status` (read-only) takes a non-blocking file lock and records a run row.

Safety contracts honoured here (see the module docstrings they come from):
- Paper only. `--mode dry-run` uses an in-memory SimBroker and never constructs AlpacaBroker;
  it also never writes broker-state rows (orders, position_meta, equity) to the journal.
- risk.gate.pre_trade: ANY reason blocks ALL orders, sells included (agent reviews are skipped
  too, so nothing is spent while blocked). check_orders failing rejects the WHOLE batch.
- Orders carry decide.client_order_id. Status "unknown" = possibly filled: journaled as pending,
  never re-bought, never followed by a stop; reconcile resolves it.
- Sells: cancel_orders_for(symbol) first (tolerate failure, retry next run); a sell that fails
  after the cancel re-places the GTC stop. Buys: GTC stop only after a fill is seen.
- EdgarClient is built lazily inside scan/backfill only, so `status` works without an identity.
- LLM is wired to journal.add_cost / month_cost_usd; no key or budget cap -> recorded passes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np

from insiderbot import config, decide, exits, features, notify
from insiderbot import formula as fml
from insiderbot.agents.llm import LLM
from insiderbot.agents.team import AgentTeam
from insiderbot.broker.alpaca_broker import AlpacaBroker
from insiderbot.broker.sim import SimBroker
from insiderbot.config import ROOT, Secrets, Settings
from insiderbot.data.edgar import EdgarClient, EdgarError
from insiderbot.journal import report
from insiderbot.journal.store import Journal
from insiderbot.models import AgentDecision, InsiderTxn, OrderPlan, SubmittedOrder
from insiderbot.research import outcomes
from insiderbot.research import review as research_review
from insiderbot.risk import gate

log = logging.getLogger("insiderbot")
NY = ZoneInfo("America/New_York")
TERMINAL = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day", "replaced",
            "filled_inferred", "not_filled_inferred"}
FILL_WAIT_S = 30
CANCEL_WAIT_S = 10
EDGAR_MAX_ERROR_FRAC = 0.10  # above this share of failed filings: no new buys this scan
FETCH_OVERLAP = timedelta(hours=2)
PENDING_TICKERS_KEY = "pending_candidate_tickers"  # tickers whose candidates a failed scan lost  # EDGAR index lag: refetch a little, dedupe in the journal
STOP_TYPES = {"stop", "stop_limit", "trailing_stop"}


# --- dependencies ------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Deps:
    """Everything a job touches from the outside world. Tests replace the fields with fakes."""

    settings: Settings
    secrets: Secrets
    root: Path = ROOT
    now: Callable[[], datetime] = _utcnow
    sleep: Callable[[float], None] = time.sleep
    market: Any = None      # MarketData built lazily when None
    edgar: Any = None       # EdgarClient built lazily (scan/backfill only) when None
    broker: Any = None      # paper broker; AlpacaBroker built lazily when None
    llm_client: Any = None  # pre-built Anthropic client (tests); else api_key from secrets
    push: Callable[..., bool] = notify.push


def default_deps() -> Deps:
    s = config.load()
    return Deps(settings=s, secrets=s.secrets())


class Ctx:
    def __init__(self, deps: Deps, job: str, mode: str):
        self.deps = deps
        self.s = deps.settings
        self.job = job
        self.mode = mode
        self.stage = "init"
        self.journal: Journal | None = None
        self.cost_journals: list[Journal] = []  # active + the other (paper/dry-run) journal
        self.run_id: int | None = None
        self._market = None
        self._warned_days = False
        self.book: SimBroker | None = None   # control mode: the persistent simulated book

    def month_cost(self) -> float:
        """Month-to-date agent spend over BOTH journals (paper and dry-run), so dry-runs can
        never double the monthly budget."""
        n = datetime.now(timezone.utc)  # same clock as Journal.add_cost
        return sum(j.month_cost_usd(n.year, n.month) for j in self.cost_journals or [self.journal])

    def days_between(self, start: date, end: date) -> int:
        """Trading days in (start, end] from the Alpaca calendar; weekdays if it fails (logged)."""
        if end <= start:
            return 0
        try:
            days = self.market().trading_days(start, end)
            return sum(1 for d in days if start < d <= end)
        except Exception as e:  # noqa: BLE001
            if not self._warned_days:
                log.warning("trading calendar unavailable (%s); counting weekdays", _reason(e))
                self._warned_days = True
            return max(0, int(np.busday_count(start, end)))

    def path(self, p: str) -> Path:
        return _path(self.deps, p)

    def market(self):
        if self._market is None:
            if self.deps.market is not None:
                self._market = self.deps.market
            else:
                sec = self.deps.secrets
                if not (sec.alpaca_key and sec.alpaca_secret):
                    raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY not set")
                from insiderbot.data.market import MarketData
                self._market = MarketData(sec.alpaca_key, sec.alpaca_secret,
                                          self.path(self.s.paths.cache_dir))
        return self._market

    def edgar(self):
        if self.deps.edgar is not None:
            return self.deps.edgar
        return EdgarClient(self.deps.secrets.edgar_identity or "",
                           self.path(self.s.paths.cache_dir) / "edgar")

    def paper_broker(self):
        if self.mode != "paper":
            raise RuntimeError(f"paper broker requested in mode {self.mode}")
        if self.deps.broker is not None:
            return self.deps.broker
        sec = self.deps.secrets
        return AlpacaBroker(sec.alpaca_key or "", sec.alpaca_secret or "")

    def llm(self) -> LLM:
        j, run_id = self.journal, self.run_id
        cli = _claude_cli(self.s) if self.deps.llm_client is None else None
        return LLM(self.s.agents.max_run_output_tokens, client=self.deps.llm_client,
                   api_key=None if cli else self.deps.secrets.anthropic_key, cli_path=cli,
                   max_monthly_usd=self.s.agents.max_monthly_usd,
                   month_cost=self.month_cost, on_cost=lambda usd: j.add_cost(run_id, usd))

    def push(self, title: str, body: str, priority: int = 3, tags: list[str] | None = None) -> None:
        if self.mode == "control":   # the frozen control never notifies
            return
        _push(self.deps, title, body, priority, tags)


def _claude_cli(s: Settings) -> str | None:
    """Path of the claude CLI when agents.backend is claude-cli and it exists, else None."""
    if s.agents.backend != "claude-cli":
        return None
    p = Path(s.agents.claude_cli).expanduser()
    return str(p) if p.is_file() else None


def _path(deps: Deps, p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else Path(deps.root) / q


def control_settings(s: Settings) -> Settings:
    """Settings for the frozen v1 control (spec 2026-11-04 Phase 0): the same strategy (formula,
    filters, review and risk limits), v1 prompts, and its own kill switch, lock and reports, no pushes."""
    ks = Path(s.risk.kill_switch_file)
    lock = Path(s.paths.lock_file)
    return s.model_copy(deep=True, update={
        "agents": s.agents.model_copy(update={"prompt_dir": "prompts_v1"}),
        "risk": s.risk.model_copy(update={"kill_switch_file": str(ks.with_name(ks.name + "_CONTROL"))}),
        "paths": s.paths.model_copy(update={
            "lock_file": str(lock.with_name(f"{lock.stem}.control{lock.suffix}")),
            "reports_dir": str(Path(s.paths.reports_dir) / "control")}),
        "notify": s.notify.model_copy(update={"ntfy": False}),
        "features": type(s.features)(),   # every upgrade feature off: the control stays v1
    })


def _control_path(real: Path) -> Path:
    """Control journal next to the real one: data/journal.db -> data/journal.control.db."""
    return real.with_name(f"{real.stem}.control{real.suffix}")


def _book_path(ctx: Ctx) -> Path:
    return ctx.path(ctx.s.paths.journal_db).with_name("book.control.json")


def _load_book(ctx: Ctx, market, sim_prices: dict[str, float], init: bool = False) -> SimBroker:
    """The control's persistent SimBroker. Held names are priced at the live quote mid, falling
    back to their average entry. A missing book is an error unless init (reconcile --init-book)."""
    path = _book_path(ctx)
    if path.exists():
        try:
            d = json.loads(path.read_text())
        except ValueError as e:
            raise ValueError(f"control book {path} is not valid JSON: {e}") from None
        avg = {k: float(v.get("avg") or math.nan) for k, v in (d.get("positions") or {}).items()}
        q = _quotes(market, avg)
        sim_prices.update({k: _mid(v) for k, v in q.items()})
        broker = SimBroker.load(d, lambda sym: sim_prices.get(sym, avg.get(sym, math.nan)))
    elif init:
        broker = SimBroker(ctx.s.risk.starting_equity, lambda sym: sim_prices.get(sym, math.nan))
    else:
        raise FileNotFoundError(f"control book {path} missing; run: reconcile --mode control --init-book")
    ctx.book = broker
    return broker


def _save_book(ctx: Ctx) -> None:
    path = _book_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ctx.book.dump(), indent=1, sort_keys=True))
    os.replace(tmp, path)


POLL_FEED_SPAN = timedelta(hours=20)   # trust the live feed alone only for gaps shorter than this


def _merge_txns(a: list[InsiderTxn], b: list[InsiderTxn]) -> list[InsiderTxn]:
    """Union of two fetches of the same window (live feed + daily index), without duplicates."""
    out, seen = [], set()
    for t in [*a, *b]:
        k = (t.accession, t.insider_cik, t.code, t.tx_date, t.shares, t.price)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


MEMORY_AGENTS = ["filing", "fundamentals", "news", "technical", "bull", "bear", "pm",
                 "risk_reviewer", "event_analyst", "event_reviewer"]


def _memory_for(ctx: Ctx, ticker: str, notes: list[str]) -> dict[str, str] | None:
    """Memory blocks for one ticker's review (features.memory); any error -> None, noted once."""
    from insiderbot.memory.block import blocks_for_run

    blocks, err = blocks_for_run(ctx.s, ctx.journal, MEMORY_AGENTS, [ticker])
    if err and not any(n.startswith("memory unavailable") for n in notes):
        notes.append(f"memory unavailable ({err}); agents ran without it")
    return blocks


def memory_after_outcomes(s: Settings, j: Journal, llm_factory, mark: int, today: str) -> dict:
    """After reconcile's outcome update: rewrite scorecard.md and, when candidates were newly scored,
    run the Post-Trade Reviewer. Never raises (memory must never break reconcile)."""
    out: dict = {"observations": 0, "error": None}
    if not s.features.memory or not s.memory.dir:
        return out
    try:
        from insiderbot.memory import memory_root
        from insiderbot.memory.files import MemoryFiles
        from insiderbot.memory.reviewer import run_post_trade
        from insiderbot.memory.scorecards import build_scorecards, render_scorecards

        files = MemoryFiles(memory_root(s), title="Lessons (insider)")
        cards = build_scorecards(j)
        files.write_scorecard(render_scorecards(cards))
        from insiderbot.memory.scorecards import reviewed

        rows = [{k: r[k] for k in ("candidate_id", "ticker", "asof", "ret", "excess", "action", "conviction")}
                for r in j.conn.execute(
                    "SELECT o.candidate_id, c.ticker, c.asof, o.ret, o.excess, d.action, d.conviction, d.reports "
                    "FROM outcomes o JOIN candidates c ON c.id = o.candidate_id "
                    "JOIN decisions d ON d.id = (SELECT MAX(id) FROM decisions WHERE candidate_id = o.candidate_id "
                    "AND action IN ('buy', 'pass')) "
                    "WHERE o.rowid > ? AND o.horizon = 20 AND c.source = 'live'", (mark,))
                if reviewed(r["reports"])]
        if rows:
            out["observations"] = run_post_trade(llm_factory(), s, j, files, rows, cards, today)
    except Exception as e:  # noqa: BLE001
        log.warning("memory step failed: %s", e)
        out["error"] = _reason(e)
    return out


def _dry_path(real: Path) -> Path:
    """Dry-run journal next to the real one: data/journal.db -> data/journal.dry-run.db."""
    return real.with_name(f"{real.stem}.dry-run{real.suffix}")


def _push(deps: Deps, title: str, body: str, priority: int = 3, tags: list[str] | None = None) -> None:
    try:
        deps.push(title, body, priority=priority, tags=tags or [],
                  enabled=deps.settings.notify.ntfy)
    except Exception as e:  # noqa: BLE001 - a push must never sink a job
        log.warning("push failed: %s", type(e).__name__)


def _reason(e: BaseException) -> str:
    msg = " ".join(str(e).split()) or type(e).__name__
    return msg[:300]


# --- small journal helpers (SQL over Journal.conn; store.py stays unchanged) ----

def _ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _meta_get(j: Journal, key: str) -> str | None:
    r = j.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else None


def _take_dive(j: Journal, day: date, cap: int) -> bool:
    """Count one deep dive against today's cap (features.research); False once the cap is used up."""
    key = f"research_dives_{day.isoformat()}"
    n = int(_meta_get(j, key) or 0)
    if n >= cap:
        return False
    _meta_set(j, key, str(n + 1))
    return True


def _meta_set(j: Journal, key: str, value: str) -> None:
    j.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    j.conn.commit()


def _unseen(j: Journal, txns: list[InsiderTxn]) -> list[InsiderTxn]:
    """Txns not yet in the journal (same key as the txns UNIQUE constraint)."""
    out = []
    for t in txns:
        r = j.conn.execute(
            "SELECT 1 FROM txns WHERE accession=? AND insider_cik=? AND tx_date=? AND code=? "
            "AND shares=? AND price=?",
            (t.accession, t.insider_cik or "", t.tx_date.isoformat(), t.code,
             round(float(t.shares or 0.0), 6), round(float(t.price or 0.0), 6))).fetchone()
        if r is None:
            out.append(t)
    return out


def _buyers(j: Journal, ticker: str) -> set[str]:
    return {r[0] for r in j.conn.execute(
        "SELECT DISTINCT insider_cik FROM txns WHERE ticker=? AND code='P'", (ticker,))}


def _insider_sold(j: Journal, ticker: str, entry: date) -> bool:
    """Any non-10b5-1 sale since entry by an insider who bought this ticker (journal)."""
    buyers = _buyers(j, ticker)
    if not buyers:
        return False
    since = datetime.combine(entry, dtime(0, 0), NY)
    rows = j.conn.execute(
        "SELECT insider_cik FROM txns WHERE ticker=? AND code='S' AND is_10b5_1=0 AND accepted_at>=?",
        (ticker, _ts(since))).fetchall()
    return any(r[0] in buyers for r in rows)


def _latest_candidate_id(j: Journal, ticker: str) -> int | None:
    r = j.conn.execute("SELECT id FROM candidates WHERE ticker=? ORDER BY id DESC LIMIT 1",
                       (ticker,)).fetchone()
    return r[0] if r else None


def _last_buy_date(j: Journal, ticker: str) -> date | None:
    """ET date of the journal's latest filled buy for ticker (adoption entry date)."""
    r = j.conn.execute("SELECT created_at FROM orders WHERE symbol=? AND side='buy' AND status IN "
                       "('filled','filled_inferred','partially_filled') ORDER BY id DESC LIMIT 1",
                       (ticker,)).fetchone()
    if r is None or not r[0]:
        return None
    try:
        d = datetime.fromisoformat(r[0])
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(NY).date()


def _sectors(j: Journal, syms) -> dict[str, str | None]:
    syms = sorted(set(syms))
    out: dict[str, str | None] = {}
    if not syms:
        return out
    q = ",".join("?" * len(syms))
    for r in j.conn.execute(f"SELECT ticker, features FROM candidates WHERE ticker IN ({q}) ORDER BY id",
                            syms):
        try:
            sec = json.loads(r["features"]).get("sector")
        except (ValueError, TypeError, AttributeError):
            sec = None
        if sec:
            out[r["ticker"]] = sec
    return out


def _last_review(j: Journal, ticker: str) -> tuple[date, int] | None:
    """(run date ET, n_buyers then) of the last real agent review (not a budget/no-key pass)."""
    rows = j.conn.execute(
        "SELECT d.reports, c.features, r.started_at FROM decisions d "
        "JOIN candidates c ON c.id=d.candidate_id JOIN runs r ON r.id=d.run_id "
        "WHERE d.ticker=? AND d.action IN ('buy','pass') ORDER BY d.id DESC LIMIT 20", (ticker,))
    for r in rows:
        try:
            rep = json.loads(r["reports"] or "{}")
            if "pm" not in rep:
                continue
            nb = int(json.loads(r["features"]).get("n_buyers") or 0)
            when = datetime.fromisoformat(r["started_at"]).astimezone(NY).date()
            return when, nb
        except (ValueError, TypeError):
            continue
    return None


def _pending(j: Journal | None, open_orders: list[dict], acct, risk) -> dict[str, float]:
    """symbol -> notional of open or unresolved buys (broker open orders + journal rows whose
    status is not terminal, incl. "unknown"). Unknown notional counts as a full position."""
    pending: dict[str, float] = {}
    ref: dict[str, float] = {}
    if j is not None:
        for r in j.conn.execute("SELECT symbol, qty, ref_price, status FROM orders "
                                "WHERE side='buy' ORDER BY id"):
            n = float(r["qty"] or 0) * float(r["ref_price"] or 0)
            ref[r["symbol"]] = n
            if (r["status"] or "unknown") not in TERMINAL:
                pending[r["symbol"]] = max(pending.get(r["symbol"], 0.0), n)
    full = max(float(acct.equity or 0), 0.0) * risk.max_position_weight
    for o in open_orders:
        if str(o.get("side", "")).lower() != "buy":
            continue
        sym = o.get("symbol")
        n = float(o.get("qty") or 0) * float(o.get("stop_price") or 0) or ref.get(sym) or full
        pending[sym] = max(pending.get(sym, 0.0), n)
    for sym, v in list(pending.items()):
        if not (isinstance(v, float) and math.isfinite(v) and v > 0):
            pending[sym] = full
    return pending


def _open_exit_sells(j: Journal) -> set[str]:
    """Symbols with an exit sell that is open or unresolved ("unknown"): never sell twice."""
    return {r["symbol"] for r in j.conn.execute(
        "SELECT symbol, status FROM orders WHERE side='sell' AND reason LIKE 'exit:%'")
        if (r["status"] or "unknown") not in TERMINAL}


# --- market helpers -------------------------------------------------------------

def _quotes(market, syms) -> dict[str, tuple[float, float]]:
    syms = sorted(set(syms))
    if not syms:
        return {}
    try:
        q = market.latest_quotes(syms) or {}
    except Exception as e:  # noqa: BLE001 - no quote means skip, never guess
        log.warning("quotes failed: %s", e)
        return {}
    return {s: v for s, v in q.items() if v and v[0] > 0 and v[1] > 0}


def _mid(q) -> float:
    return (q[0] + q[1]) / 2 if q else 0.0


def _as_date(v) -> date | None:
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


# --- order helpers ----------------------------------------------------------------

def _poll(ctx: Ctx, broker, broker_id: str) -> dict | None:
    st = None
    for i in range(FILL_WAIT_S + 1):
        try:
            st = broker.order_status(broker_id)
        except Exception as e:  # noqa: BLE001
            log.warning("order_status %s failed: %s", broker_id, e)
            st = None
        if st and st.get("status") in TERMINAL:
            return st
        if i < FILL_WAIT_S:
            ctx.deps.sleep(1)
    return st


def _row(sub: SubmittedOrder, **kw) -> dict:
    p = sub.plan
    return {"symbol": p.symbol, "side": p.side, "qty": p.qty, "status": sub.status,
            "reason": p.reason, "error": sub.error, **kw}


def _record(ctx: Ctx, sub: SubmittedOrder) -> int | None:
    if ctx.mode != "paper":
        return None
    return ctx.journal.add_order(ctx.run_id, sub)


def _place_stop(ctx: Ctx, broker, sym: str, qty: int, entry: float, fm, tag: str,
                rows: list[dict]) -> SubmittedOrder | None:
    px = exits.stop_price(entry, fm)
    try:
        sub = broker.submit_stop(sym, int(qty), px, decide.client_order_id(ctx.run_id, sym, tag))
    except Exception as e:  # noqa: BLE001 - reconcile re-places missing stops
        log.error("stop for %s failed: %s", sym, e)
        rows.append({"symbol": sym, "side": "sell", "qty": qty, "status": "stop_failed",
                     "reason": "stop", "error": _reason(e)})
        return None
    _record(ctx, sub)
    rows.append(_row(sub, stop_price=px))
    return sub


def _wait_no_open(ctx: Ctx, broker, sym: str) -> None:
    for i in range(CANCEL_WAIT_S + 1):
        try:
            if not any(o.get("symbol") == sym for o in broker.open_orders()):
                return
        except Exception:  # noqa: BLE001
            return
        if i < CANCEL_WAIT_S:
            ctx.deps.sleep(1)


def _sell(ctx: Ctx, broker, plan: OrderPlan, meta: dict, fm, today: date, rows: list[dict]) -> None:
    sym = plan.symbol
    try:
        broker.cancel_orders_for(sym)
    except Exception as e:  # noqa: BLE001 - cancel is async/fallible: retry next run
        rows.append({"symbol": sym, "side": "sell", "qty": plan.qty, "status": "cancel_failed",
                     "reason": plan.reason, "error": _reason(e)})
        return
    _wait_no_open(ctx, broker, sym)
    sub = broker.submit(plan, decide.client_order_id(ctx.run_id, sym, "sell"))
    oid = _record(ctx, sub)
    if sub.status == "unknown":  # possibly filled: never re-place a stop; reconcile resolves
        rows.append(_row(sub))
        return
    entry = meta.get("entry_price") or plan.ref_price
    if sub.status == "rejected" or not sub.broker_id:
        rows.append(_row(sub))
        _place_stop(ctx, broker, sym, plan.qty, entry, fm, "stop-restore", rows)
        return
    st = _poll(ctx, broker, sub.broker_id) or {}
    status = st.get("status") or sub.status
    px = st.get("filled_avg_price")
    if oid is not None and st:
        ctx.journal.update_order(oid, status, px, st.get("filled_qty"))
    rows.append(_row(sub, status=status, fill_price=px))
    if status == "filled":
        if ctx.mode == "paper":
            ctx.journal.close_position_meta(sym, today, px, plan.reason.removeprefix("exit:"))
    elif status in TERMINAL:  # canceled/expired/rejected after the cancel: restore protection
        _place_stop(ctx, broker, sym, plan.qty, entry, fm, "stop-restore", rows)


def _buy(ctx: Ctx, broker, plan: OrderPlan, cid: int | None, fm, today: date,
         rows: list[dict]) -> None:
    sym = plan.symbol
    sub = broker.submit(plan, decide.client_order_id(ctx.run_id, sym, "buy"))
    oid = _record(ctx, sub)
    if sub.status == "unknown" or sub.status == "rejected" or not sub.broker_id:
        rows.append(_row(sub))  # unknown = pending (journal row); reconcile resolves
        return
    st = _poll(ctx, broker, sub.broker_id) or {}
    status = st.get("status") or sub.status
    px, fq = st.get("filled_avg_price"), st.get("filled_qty")
    if oid is not None and st:
        ctx.journal.update_order(oid, status, px, fq)
    rows.append(_row(sub, status=status, fill_price=px))
    if status == "filled" and px and fq and int(fq) >= 1:
        if ctx.mode == "paper":
            ctx.journal.add_position_meta(sym, today, float(px), fm.version, cid)
        _place_stop(ctx, broker, sym, int(fq), float(px), fm, "stop", rows)


# --- jobs -----------------------------------------------------------------------

def _slot_due(s: Settings, slot: str, day: date) -> datetime:
    hhmm = s.schedule.scans[0 if slot == "am" else -1]
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(day, dtime(h, m), ZoneInfo(s.schedule.tz))


def _meta_for(ctx: Ctx, sym: str, pos, fm, today: date, notes: list[str]) -> dict:
    j = ctx.journal
    if ctx.mode == "paper":
        row = j.position_meta(sym)
        if row is None:
            entry = _last_buy_date(j, sym) or today
            j.add_position_meta(sym, entry, pos.avg_price, fm.version, _latest_candidate_id(j, sym))
            notes.append(f"adopted {sym}: no position_meta, entry {entry} at avg price")
            row = j.position_meta(sym)
        return dict(row)
    return {"entry_date": pos.entry_date or today, "entry_price": pos.avg_price,
            "peak_price": pos.peak_price}


def _decision_row(d: AgentDecision) -> dict:
    return {"ticker": d.ticker, "action": d.action, "confidence": d.conviction,
            "rationale": d.thesis}


def _scan(ctx: Ctx, args, fm, sh):
    s, j, deps = ctx.s, ctx.journal, ctx.deps
    now = deps.now()
    today = now.astimezone(NY).date()
    mode = ctx.mode
    paper = mode == "paper"
    summary: dict = {"mode": mode, "slot": args.slot}
    notes: list[str] = []

    # 1. schedule
    poll = args.slot == "poll"
    if poll:
        if not s.features.poll:
            return "skipped_disabled", summary, "skipped_disabled (features.poll is off)"
        t = now.astimezone(NY).strftime("%H:%M")
        lo, hi = s.schedule.poll_window
        if not (lo <= t <= hi):
            return "skipped_window", summary, f"skipped_window ({t} ET outside {lo}-{hi})"
    elif args.slot and not args.force:
        due = _slot_due(s, args.slot, today)
        if now > due + timedelta(minutes=s.schedule.max_late_minutes):
            late = int((now - due).total_seconds() // 60)
            summary["late_minutes"] = late
            return "skipped_late", summary, f"skipped_late ({late} min late)"
    ctx.stage = "market"
    market = ctx.market()
    ctx.stage = "calendar"
    cal = market.calendar_today(now)
    if cal is None:
        return "skipped_closed", summary, "skipped_closed (not a trading day)"
    market_open = cal[0] <= now < cal[1]

    # 2. fetch
    ctx.stage = "edgar"
    wm = j.get_watermark()
    since = (wm - FETCH_OVERLAP) if wm else now - timedelta(days=3)
    # EDGAR problems (including a missing identity) never block exits or stops: those work
    # from prices and broker state. They block NEW BUYS only when the whole fetch failed or
    # > EDGAR_MAX_ERROR_FRAC of the attempted filings failed; below that the scan proceeds.
    edgar_block_buys = None
    client = None
    try:
        client = ctx.edgar()
        txns = []
        if since < now and poll and hasattr(client, "fetch_current"):
            try:   # EDGAR's live feed: same-day filings within minutes
                txns = client.fetch_current(since, now)
                # The feed only covers ~24h: after a weekend, holiday or outage, the gap back to the
                # watermark is filled from the daily indexes too, so no filing is ever skipped.
                if not getattr(client, "last_reached_since", True) or now - since > POLL_FEED_SPAN:
                    notes.append("EDGAR live feed does not reach the watermark; daily index read too")
                    txns = _merge_txns(txns, client.fetch(since, now))
            except Exception as e:  # noqa: BLE001 - fall back to the daily indexes
                log.warning("EDGAR current feed failed (%s); using the daily index", _reason(e))
                notes.append(f"EDGAR live feed failed ({_reason(e)}); used the daily index")
                txns = client.fetch(since, now)
        elif since < now:
            txns = client.fetch(since, now)
    except EdgarError as e:  # EDGAR_IDENTITY not set
        log.error("EDGAR unavailable: %s", e)
        txns = []
        edgar_block_buys = f"EDGAR unavailable ({_reason(e)})"
        _push_daily(ctx, today, "edgar_identity", f"insider-bot: {_reason(e)}",
                    "No new buys until it is set in .env; exits and stops still run.", 5)
    except Exception as e:  # noqa: BLE001
        log.error("EDGAR fetch failed: %s", e)
        txns = []
        edgar_block_buys = f"EDGAR fetch failed ({_reason(e)})"
        _push_daily(ctx, today, "edgar_fetch", "insider-bot: EDGAR fetch failed",
                    f"{_reason(e)}\nNo new buys this scan; exits and stops still run.", 4)
    errs = int(getattr(client, "last_errors", 0) or 0)
    tried = int(getattr(client, "last_attempted", 0) or 0)
    dropped = int(getattr(client, "last_dropped", 0) or 0)
    if dropped:
        notes.append(f"EDGAR: {dropped} filing(s) dropped for a malformed ticker")
    if edgar_block_buys is None and errs:
        frac = errs / max(tried, errs, 1)
        msg = f"EDGAR: {errs} of {tried} filing(s) failed ({frac:.0%})"
        if frac > EDGAR_MAX_ERROR_FRAC:
            edgar_block_buys = msg
        else:
            notes.append(msg + "; proceeding with the filings that loaded")
    if edgar_block_buys:
        notes.append(f"no new buys: {edgar_block_buys}")
        summary["edgar_block_buys"] = edgar_block_buys
    summary["edgar_errors"] = {"failed": errs, "attempted": tried}
    new = _unseen(j, txns)
    j.add_txns(txns, "live")
    # the watermark advances only after candidates are journaled (below); a failed fetch
    # leaves it alone so the next scan refetches the window
    new_wm = max([t.accepted_at for t in txns] + ([wm] if wm else []), default=now) \
        if (edgar_block_buys is None or txns) else None
    summary["txns_new"] = len(new)

    # 3. account
    ctx.stage = "broker"
    sim_prices: dict[str, float] = {}
    if paper:
        broker = ctx.paper_broker()
    elif mode == "control":
        broker = _load_book(ctx, market, sim_prices)
    else:
        broker = SimBroker(s.risk.starting_equity, lambda sym: sim_prices.get(sym, math.nan))
    acct = broker.get_account()
    if paper:
        j.add_equity(now, acct.equity)
    peak = j.peak_equity() if paper else None
    open_orders = broker.open_orders()
    pending = _pending(j if paper else None, open_orders, acct, s.risk)
    held = acct.positions

    # 5. features + candidates (journaled for everything, traded or not; before the gate,
    #    no orders happen before the gate)
    # A failure here must not abort the scan: exits and stops still run, buys are blocked,
    # and the tickers are carried (journal meta) so the next scan journals them.
    ctx.stage = "features"
    carried = set(json.loads(_meta_get(j, PENDING_TICKERS_KEY) or "[]"))
    new_p = {t.ticker for t in new if t.code == "P" and t.ticker} | carried
    activity = {t.ticker for t in new if t.ticker}
    cands: dict[str, dict] = {}
    try:
        want = new_p | (activity & set(held))
        win = [t for t in j.txns_since(now - timedelta(days=fm.window_days), ("P",)) if t.ticker in want]
        feats = features.build(win, market, j, now, fm.window_days) if want else []
        scored = []
        for f in feats:
            sc, flags = fml.score(f, fm)
            scores = {fm.version: sc}
            if sh is not None:
                scores[sh.version] = fml.score(f, sh)[0]
            scored.append((f, sc, flags, scores, features.hard_filter(f, s.filters)))
        for f, sc, flags, scores, why in scored:
            cid = j.add_candidate(ctx.run_id, f, scores, flags, why, "live") if f.ticker in new_p else None
            cands[f.ticker] = {"f": f, "score": sc, "flags": flags, "filter": why, "cid": cid,
                               "txns": [t for t in win if t.ticker == f.ticker]}
    except Exception as e:  # noqa: BLE001
        log.error("features stage failed", exc_info=True)
        cands = {}
        msg = f"features failed ({_reason(e)})"
        edgar_block_buys = edgar_block_buys or msg
        notes.append(f"no new buys: {msg}; {len(new_p)} ticker(s) carried to the next scan")
        summary["features_failed"] = _reason(e)
        _meta_set(j, PENDING_TICKERS_KEY, json.dumps(sorted(new_p)))
        _push_daily(ctx, today, "features", "insider-bot: features stage failed",
                    f"{_reason(e)}\nNo new buys; exits and stops still run.", 5)
    else:
        if carried:
            _meta_set(j, PENDING_TICKERS_KEY, "[]")
        if new_wm is not None:
            j.set_watermark(new_wm)
    cand_rows = [{"ticker": t, "score": round(c["score"], 1), "filter": c["filter"] or "",
                  "flags": " ".join(c["flags"])} for t, c in cands.items() if c["cid"]]
    summary["candidates"] = len(cand_rows)

    # 6a. review queue (reviews themselves run after the gate and the exits)
    ctx.stage = "review"
    queue = []
    for t, c in cands.items():
        if not c["cid"] or c["filter"] or c["score"] < fm.review_threshold:
            continue
        if t in held or t in pending:
            continue
        prev = _last_review(j, t)
        if prev and ctx.days_between(prev[0], today) < s.review.recent_review_days \
                and c["f"].n_buyers <= prev[1]:
            continue
        queue.append(c)
    queue.sort(key=lambda c: c["score"], reverse=True)
    queue = queue[: s.review.max_reviews_per_scan]

    # gate: data freshness + pre_trade
    ctx.stage = "risk"
    quotes = _quotes(market, set(held) | {c["f"].ticker for c in queue})
    stale = _staleness(ctx, market, client, now, today,
                       set(held) | {c["f"].ticker for c in queue}, quotes)
    if stale:
        notes.append("data stale: " + "; ".join(stale))
        summary["stale"] = stale
    was_killed = gate.kill_switch_active(s)
    reasons = gate.pre_trade(s, acct, peak, market_open, data_fresh=not stale)
    tripped = not was_killed and gate.kill_switch_active(s)
    blocked = bool(reasons)
    summary["blocked"] = reasons
    if blocked:
        notes.append("no orders this run: " + "; ".join(reasons))

    ctx.stage = "agents"
    llm = ctx.llm()
    team = AgentTeam(llm, s)
    n = datetime.now(timezone.utc)
    budget_hit = ctx.month_cost() >= s.agents.max_monthly_usd
    agents_ok = llm.available()
    if budget_hit:
        notes.append(f"monthly agent budget ${s.agents.max_monthly_usd:g} reached: no new buys")
        summary["budget_hit"] = True
        key = f"budget_push_{n.year}-{n.month:02d}"
        if _meta_get(j, key) is None:
            ctx.push("insider-bot: agent budget reached",
                     f"Monthly agent spend reached ${s.agents.max_monthly_usd:g}. "
                     "No new buys until next month; exits still run.", priority=4)
            _meta_set(j, key, _ts(now))
    if not agents_ok:
        msg = "agents unavailable: no Anthropic API key, reviewed names recorded as pass (no buys)"
        log.warning(msg)
        notes.append(msg)
        summary["agents_unavailable"] = True

    # PM exit opinion for held names that got a new insider purchase this scan, filed by one
    # of the ticker's insiders (cheap: PM only)
    dec_rows: list[dict] = []
    exit_votes: dict[str, bool] = {}
    if not blocked and agents_ok and not budget_hit:
        for sym in sorted(held):
            c = cands.get(sym)
            if c is None:
                continue
            buyers = _buyers(j, sym)
            fresh = [t for t in new if t.ticker == sym and t.insider_cik in buyers]
            if not fresh:
                continue
            before = llm.cost_usd
            mem = _memory_for(ctx, sym, notes)
            d = team.exit_review(c["f"], c["score"], c["flags"], fresh, memory=mem)
            if mem:
                d.reports["memory"] = mem
            cid = c["cid"] or _latest_candidate_id(j, sym)
            if cid is not None:
                j.add_decision(ctx.run_id, cid, d, llm.cost_usd - before)
            else:
                log.warning("exit check for %s not journaled: no candidate row", sym)
            dec_rows.append(_decision_row(d))
            exit_votes[sym] = d.action == "exit"

    # 4. exits (placed before any buy). Deterministic exits run first; Event Review exit votes
    # (Phase 3) get a second pass afterwards, so slow agent calls never delay or stale the formula's exits.
    ctx.stage = "exits"
    order_rows: list[dict] = []
    exit_rows: list[dict] = []
    metas: dict[str, dict] = {}
    exited: set[str] = set()

    def exit_pass(only: set[str] | None = None) -> None:
        nonlocal acct, pending
        sells: list[OrderPlan] = []
        open_sells = _open_exit_sells(j) if paper else set()
        for sym, pos in sorted(held.items()):
            if only is not None and sym not in only:
                continue
            meta = _meta_for(ctx, sym, pos, fm, today, notes)
            metas[sym] = meta
            last = _mid(quotes.get(sym))
            if last > 0:
                if paper:
                    j.update_peak(sym, last)
                meta["peak_price"] = max(float(meta.get("peak_price") or 0), last)
            entry_d = _as_date(meta.get("entry_date")) or today
            days = ctx.days_between(entry_d, today)
            sold = _insider_sold(j, sym, entry_d)
            p2 = dataclasses.replace(pos, entry_date=entry_d, peak_price=meta.get("peak_price"))
            why = exits.check(p2, meta, last, fm, today, days, sold, exit_votes.get(sym, False),
                              min_hold_days=s.risk.min_hold_days_for_agent_exit)
            if why is None:
                continue
            entry = meta.get("entry_price") or pos.avg_price
            exit_rows.append({"symbol": sym, "reason": why,
                              "pnl": f"{(last / entry - 1):+.1%}" if last > 0 and entry else "n/a"})
            if sym in open_sells:
                notes.append(f"exit {sym} ({why}): a sell is already pending")
                continue
            qty = int(math.floor(pos.qty))
            if qty >= 1:
                sells.append(OrderPlan(sym, "sell", qty, f"exit:{why}", last if last > 0 else pos.avg_price))
        if sells:
            if blocked:
                summary["sells_blocked"] = [p.symbol for p in sells]
            else:
                # sell-only checks: portfolio limits never block a risk-reducing exit
                errs = gate.check_sells(s, sells, acct)
                if errs:
                    notes.append("sell batch rejected: " + "; ".join(errs))
                    summary["sell_batch_rejected"] = errs
                else:
                    for p in sells:
                        _sell(ctx, broker, p, metas[p.symbol], fm, today, order_rows)
                    exited.update(p.symbol for p in sells)
                    acct = broker.get_account()
                    pending = _pending(j if paper else None, broker.open_orders(), acct, s.risk)


    exit_pass()

    # Phase 3 Event Review (poll slot, held names): both agents must say exit to add an exit vote,
    # which the existing exits.check applies (min hold days; the formula's exits are unchanged).
    if poll and s.features.event_review and held and not blocked and agents_ok and not budget_hit:
        ctx.stage = "events"
        try:
            from insiderbot import events as ev
            from insiderbot.data import sec_filings

            ident = deps.secrets.edgar_identity
            prev, sig = ev.refs(market.bars(sorted(held), today - timedelta(days=45), today, now))
            mids = {x: _mid(q) for x, q in quotes.items() if x in held}
            votes = ev.run_event_reviews(
                team, j, {k: v for k, v in held.items() if k not in exited}, mids, prev, sig, today,
                ctx.run_id, mode, s.events.max_reviews_per_day,
                s.events.k_sigma, notes,
                filings_fn=lambda sym, since: sec_filings.recent_filings(
                    sym, since, forms=("8-K", "10-Q", "10-K"), cache_dir=ctx.path(s.paths.cache_dir),
                    user_agent=ident),
                memory_fn=lambda sym: _memory_for(ctx, sym, notes),
                text_fn=lambda url: sec_filings.filing_text(url, s.research.max_filing_chars, user_agent=ident),
                per_run=s.events.max_reviews_per_poll)
            for sym in votes:
                exit_votes[sym] = True
            if votes:   # second exit pass, on fresh quotes, only for the names both agents voted out
                quotes.update(_quotes(market, set(votes)))
                exit_pass(set(votes))
        except Exception as e:  # noqa: BLE001 - Event Review never blocks exits or stops
            log.warning("event review failed: %s", e)
            notes.append(f"event review failed ({_reason(e)})")

    # 6b. agent reviews
    ctx.stage = "review"
    reviewed: list[tuple[dict, AgentDecision]] = []
    if queue and blocked:
        notes.append(f"agent reviews skipped while blocked ({len(queue)} queued)")
    elif queue and edgar_block_buys:  # no buys possible: do not spend on reviews
        notes.append(f"agent reviews skipped: new buys blocked by EDGAR ({len(queue)} queued)")
    elif queue:
        for c in queue:
            f = c["f"]
            before = llm.cost_usd
            if budget_hit:
                d = AgentDecision(f.ticker, "pass", 1, "budget: monthly agent cap reached", "",
                                  {"skipped": "budget"})
            elif not agents_ok:
                d = AgentDecision(f.ticker, "pass", 1, "agents unavailable: no Anthropic API key", "",
                                  {"skipped": "no_key"})
            else:
                mem = _memory_for(ctx, f.ticker, notes)
                rs = None
                if s.features.research:
                    if _take_dive(j, today, s.research.max_dives_per_day):
                        from insiderbot.agents.deep_research import Researcher

                        rs = Researcher(llm, s.research, s.agents.prompt_dir, cache_dir=ctx.path(s.paths.cache_dir),
                                        user_agent=deps.secrets.edgar_identity)
                    elif "deep-dive cap reached" not in notes:
                        notes.append("deep-dive cap reached")
                port = {"held": sorted(held), "n_positions": len(held), "max_positions": s.risk.max_positions,
                        "sector": f.sector, "sectors_held": sorted({x for x in _sectors(j, set(held)).values() if x})}
                d = team.review(f, c["score"], c["flags"], c["txns"], _context(market, f, c["txns"], now), held=False,
                                memory=mem, researcher=rs, reviewers=s.features.reviewers, portfolio=port)
                if mem:
                    d.reports["memory"] = mem
            j.add_decision(ctx.run_id, c["cid"], d, llm.cost_usd - before)
            dec_rows.append(_decision_row(d))
            reviewed.append((c, d))
    summary["reviewed"] = len(reviewed)

    # 7. buys
    ctx.stage = "buys"
    want_buy = [(c, d) for c, d in reviewed if d.action == "buy"]
    if want_buy and not blocked and not edgar_block_buys:
        q = _quotes(market, [c["f"].ticker for c, _ in want_buy])
        batch = []
        for c, d in want_buy:
            sym = c["f"].ticker
            if sym not in q:
                notes.append(f"buy {sym} skipped: no valid quote")
                continue
            ask = q[sym][1]
            sim_prices[sym] = ask
            batch.append((dataclasses.replace(c["f"], price=ask), c["score"], d))
        sector_of = _sectors(j, set(acct.positions) | set(pending))
        for c, _ in want_buy:
            sector_of.setdefault(c["f"].ticker, c["f"].sector)
        plans, skips = decide.buy_orders(batch, acct, fm, s.risk, sector_of, pending)
        for sk in skips:
            notes.append(f"buy {sk['symbol']} skipped: {sk['reason']}")
        errs = gate.check_orders(s, plans, acct, pending, sector_of) if plans else []
        if errs:
            notes.append("buy batch rejected: " + "; ".join(errs))
            summary["buy_batch_rejected"] = errs
            plans = []
        cid_of = {c["f"].ticker: c["cid"] for c, _ in want_buy}
        for p in plans:
            _buy(ctx, broker, p, cid_of.get(p.symbol), fm, today, order_rows)

    # missing GTC stops: placed when not blocked; alert (p5, once/day/symbol) when any remain
    ctx.stage = "stops"
    unprotected = _ensure_stops(ctx, broker, fm, today, reasons, notes, order_rows, "stop-ensure")
    _alert_unprotected(ctx, today, unprotected, notes)
    summary["unprotected"] = [u[0] for u in unprotected]

    # 8. report + pushes
    ctx.stage = "report"
    placed = [r for r in order_rows if r.get("status") not in ("rejected", "cancel_failed", "stop_failed")]
    summary.update(orders=order_rows, exits=exit_rows, notes=notes)
    stamp = now.astimezone(NY).strftime("%Y-%m-%d-%H%M")
    rpath = ctx.path(s.paths.reports_dir) / f"scan-{stamp}-{args.slot or 'manual'}" \
        f"{'' if paper else '-' + mode}.md"
    if not poll or order_rows or exit_rows or tripped:   # a quiet 5-minute poll writes no report
        report.write_scan(rpath, {"id": ctx.run_id, "mode": mode, "formula_version": fm.version,
                                  "started_at": now}, cand_rows, dec_rows, order_rows, exit_rows, notes)
        summary["report"] = str(rpath)
    if tripped:
        ctx.push("insider-bot: KILL SWITCH tripped", "\n".join(reasons), priority=5, tags=["warning"])
    if placed and paper:
        body = "\n".join(f"{r['side'].upper()} {r['qty']} {r['symbol']} {r['status']}"
                         + (f" @ {r['fill_price']:.2f}" if r.get("fill_price") else "")
                         + (f" stop {r['stop_price']:.2f}" if r.get("stop_price") else "")
                         for r in order_rows)
        ctx.push(f"insider-bot: {len(placed)} order(s) placed", body)
    buys = sum(1 for r in order_rows if r["side"] == "buy")
    sold = sum(1 for r in order_rows if r["side"] == "sell" and r.get("reason", "").startswith("exit:"))
    line = (f"mode={mode} candidates={len(cand_rows)} reviewed={len(reviewed)} buys={buys} "
            f"sells={sold} blocked={'yes' if blocked else 'no'}"
            + (" agents=unavailable" if not agents_ok else "") + (" budget=hit" if budget_hit else "")
            + (" buys_blocked=yes" if edgar_block_buys else ""))
    return "ok", summary, line


def _prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _staleness(ctx: Ctx, market, client, now: datetime, today: date, names: set[str],
               quotes: dict) -> list[str]:
    """Price-data freshness rule. Any hit makes data_fresh False, so pre_trade blocks ALL
    orders this run:
    1. the latest SPY daily bar is older than the last completed trading session (the last
       trading day before today; market.bars drops today's partial bar), or SPY is missing;
    2. there are names to trade (held + review queue) and none has a valid (bid, ask) quote.
    EDGAR errors are NOT price staleness: they only block new buys (see _scan, fetch step)."""
    out: list[str] = []
    try:
        days = [d for d in market.trading_days(today - timedelta(days=14), today) if d < today]
        last_session = days[-1] if days else _prev_weekday(today)
    except Exception as e:  # noqa: BLE001
        log.warning("trading calendar unavailable (%s); last session = previous weekday", _reason(e))
        last_session = _prev_weekday(today)
    try:
        spy = (market.bars(["SPY"], today - timedelta(days=14), today, now=now) or {}).get("SPY")
        latest = spy.index[-1].date() if spy is not None and len(spy) else None
    except Exception as e:  # noqa: BLE001
        log.warning("SPY bars failed: %s", _reason(e))
        latest = None
    if latest is None or latest < last_session:
        out.append(f"latest SPY bar {latest} older than last session {last_session}")
    if names and not any(x in quotes for x in names):
        out.append("no valid quote for any held or queued name")
    return out


def _ensure_stops(ctx: Ctx, broker, fm, today: date, reasons: list[str], notes: list[str],
                  rows: list[dict], tag: str) -> list[tuple[str, int, str]]:
    """Place a GTC stop for every held share not already covered by an open stop. Nothing is
    placed when reasons (gate) is non-empty. Returns the positions left unprotected."""
    acct = broker.get_account()
    stopped: dict[str, float] = {}
    for o in broker.open_orders():
        if str(o.get("side", "")).lower() == "sell" and str(o.get("type", "")).lower() in STOP_TYPES:
            stopped[o["symbol"]] = stopped.get(o["symbol"], 0.0) + float(o.get("qty") or 0)
    out: list[tuple[str, int, str]] = []
    for sym, pos in sorted(acct.positions.items()):
        missing = int(math.floor(pos.qty - stopped.get(sym, 0.0)))
        if missing < 1:
            continue
        if reasons:
            out.append((sym, missing, "not placed: " + "; ".join(reasons)))
            continue
        meta = _meta_for(ctx, sym, pos, fm, today, notes)
        entry = meta.get("entry_price") or pos.avg_price
        sub = _place_stop(ctx, broker, sym, missing, entry, fm, tag, rows)
        if sub is None or sub.status in ("rejected", "unknown") or not sub.broker_id:
            err = (sub.error if sub is not None else None) or "stop placement failed"
            out.append((sym, missing, f"stop {('failed' if sub is None else sub.status)}: {err}"))
    return out


def _push_daily(ctx: Ctx, today: date, key: str, title: str, body: str, priority: int) -> None:
    """Push at most once per day per key (journal meta dedupe)."""
    k = f"push_{key}_{today.isoformat()}"
    if _meta_get(ctx.journal, k) is None:
        ctx.push(title, body, priority=priority, tags=["warning"] if priority >= 5 else None)
        _meta_set(ctx.journal, k, _ts(ctx.deps.now()))


def _alert_unprotected(ctx: Ctx, today: date, items: list[tuple[str, int, str]],
                       notes: list[str]) -> None:
    """Note every unprotected position; p5 push once per day per symbol (journal-deduped)."""
    fresh = []
    for sym, qty, why in items:
        notes.append(f"NO GTC STOP: {sym} {qty} sh unprotected ({why})")
        key = f"stop_alert_{today.isoformat()}_{sym}"
        if ctx.mode == "paper" and _meta_get(ctx.journal, key) is None:
            fresh.append(f"{sym}: {qty} sh without a stop ({why})")
            _meta_set(ctx.journal, key, _ts(ctx.deps.now()))
    if fresh:
        ctx.push("insider-bot: position without stop", "\n".join(fresh), priority=5,
                 tags=["warning"])


def _context(market, f, txns=None, now: datetime | None = None) -> dict:
    """Evidence for the agents. Information only: the formula score and filters never read it."""
    ctx: dict = {}
    try:
        ctx["fundamentals"] = market.fundamentals(f.ticker) or "none"
    except Exception:  # noqa: BLE001
        ctx["fundamentals"] = "unavailable"
    try:
        ctx["news"] = market.news(f.ticker) or "none"
    except Exception:  # noqa: BLE001
        ctx["news"] = "unavailable"

    def pct(x):
        return "n/a" if x is None else f"{x:+.1%}"

    ctx["bars_summary"] = (f"last close {f.price}; 3-month return {pct(f.ret_3m)}; "
                           f"drawdown from 52w high {pct(f.drawdown_52w)}; "
                           f"20d avg dollar volume {f.adv20_usd}")
    buys = [t for t in (txns or []) if t.code == "P" and t.price and t.price > 0 and t.shares and t.shares > 0]
    if buys and f.price:
        avg = sum(t.price * t.shares for t in buys) / sum(t.shares for t in buys)
        ctx["bars_summary"] += (f"; insiders' average purchase price ${avg:,.2f}, "
                                f"price {pct(f.price / avg - 1)} since")
        latest = getattr(f, "latest_accepted_at", None)
        if latest is not None and now is not None:
            ctx["bars_summary"] += f"; latest filing filed {(now - latest).total_seconds() / 3600:.1f}h ago"
    return ctx


def _refresh_orders(j: Journal, broker) -> int:
    n = 0
    q = ",".join("?" * len(TERMINAL))
    rows = j.conn.execute(f"SELECT id, broker_id FROM orders WHERE broker_id IS NOT NULL "
                          f"AND COALESCE(status,'') NOT IN ({q})", sorted(TERMINAL)).fetchall()
    for r in rows:
        try:
            st = broker.order_status(r["broker_id"])
        except Exception as e:  # noqa: BLE001
            log.warning("order_status %s failed: %s", r["broker_id"], e)
            continue
        if st.get("status") and st["status"] != "unknown":
            j.update_order(r["id"], st["status"], st.get("filled_avg_price"), st.get("filled_qty"))
            n += 1
    return n


def _resolve_unknown(j: Journal, acct, open_orders: list[dict], market_open: bool) -> int:
    """Orders whose submit status was "unknown" and that have no broker id. Only after the
    session (DAY orders are done): infer the outcome from positions."""
    if market_open:
        return 0
    n = 0
    open_syms = {o.get("symbol") for o in open_orders}
    for r in j.conn.execute("SELECT id, symbol, side FROM orders WHERE broker_id IS NULL "
                            "AND status='unknown'").fetchall():
        if r["symbol"] in open_syms:
            continue
        held = r["symbol"] in acct.positions
        filled = held if r["side"] == "buy" else not held
        j.update_order(r["id"], "filled_inferred" if filled else "not_filled_inferred", None, None)
        n += 1
    return n


def _gone_exit(j: Journal, sym: str) -> tuple[str, float | None]:
    r = j.conn.execute("SELECT reason, fill_price FROM orders WHERE symbol=? AND side='sell' "
                       "AND status='filled' ORDER BY id DESC LIMIT 1", (sym,)).fetchone()
    if r is not None:
        reason = r["reason"] or ""
        if reason.startswith("exit:"):
            return reason[5:], r["fill_price"]
        return "stop_broker", r["fill_price"]
    has_stop = j.conn.execute("SELECT 1 FROM orders WHERE symbol=? AND reason='stop'",
                              (sym,)).fetchone()
    return ("stop_broker" if has_stop else "gone_unknown"), None


def _reconcile(ctx: Ctx, args, fm, sh):
    s, j, deps = ctx.s, ctx.journal, ctx.deps
    now = deps.now()
    today = now.astimezone(NY).date()
    summary: dict = {"mode": ctx.mode}
    notes: list[str] = []
    ctx.stage = "market"
    market = ctx.market()
    if ctx.mode == "control":
        ctx.stage = "book"
        book = _load_book(ctx, market, {}, init=getattr(args, "init_book", False))
        acct = book.get_account()
        j.add_equity(now, acct.equity)
        book.last_equity = acct.equity   # tomorrow's daily-loss check measures from today's close
        summary["equity"] = acct.equity
        ctx.stage = "outcomes"
        n = outcomes.update(j, market, today)
        summary["outcomes"] = n
        return "ok", summary, f"mode=control equity={acct.equity:.2f} outcomes={n}"
    if ctx.mode != "paper":
        ctx.stage = "outcomes"
        n = outcomes.update(j, market, today)
        summary["outcomes"] = n
        return "ok", summary, f"mode={ctx.mode} broker skipped outcomes={n}"
    ctx.stage = "broker"
    broker = ctx.paper_broker()
    ctx.stage = "calendar"
    cal = market.calendar_today(now)
    market_open = bool(cal and cal[0] <= now < cal[1])

    ctx.stage = "orders"
    summary["orders_refreshed"] = _refresh_orders(j, broker)
    acct = broker.get_account()
    summary["unknown_resolved"] = _resolve_unknown(j, acct, broker.open_orders(), market_open)
    j.add_equity(now, acct.equity)

    ctx.stage = "risk"
    was_killed = gate.kill_switch_active(s)
    # market hours do not apply to queuing GTC stops; every other reason still blocks
    reasons = gate.pre_trade(s, acct, j.peak_equity(), True, data_fresh=True)
    tripped = not was_killed and gate.kill_switch_active(s)

    ctx.stage = "positions"
    exit_rows = []
    for r in j.conn.execute("SELECT symbol, entry_price FROM position_meta WHERE exit_date IS NULL").fetchall():
        sym = r["symbol"]
        if sym in acct.positions:
            continue
        reason, px = _gone_exit(j, sym)
        j.close_position_meta(sym, today, px, reason)
        pnl = f"{(px / r['entry_price'] - 1):+.1%}" if px and r["entry_price"] else "n/a"
        exit_rows.append({"symbol": sym, "reason": reason, "pnl": pnl})
    order_rows: list[dict] = []
    pos_rows = []
    for sym, pos in sorted(acct.positions.items()):
        meta = _meta_for(ctx, sym, pos, fm, today, notes)
        px = pos.market_value / pos.qty if pos.qty else 0.0
        if px > 0:
            j.update_peak(sym, px)
        entry = meta.get("entry_price") or pos.avg_price
        pos_rows.append({"symbol": sym, "qty": pos.qty, "entry_price": entry,
                         "pnl": f"{(px / entry - 1):+.1%}" if px > 0 and entry else "n/a"})
    ctx.stage = "stops"
    unprotected = _ensure_stops(ctx, broker, fm, today, reasons, notes, order_rows, "stop-reconcile")
    _alert_unprotected(ctx, today, unprotected, notes)
    summary["unprotected"] = [u[0] for u in unprotected]

    ctx.stage = "outcomes"
    mark = j.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM outcomes").fetchone()[0]
    summary["outcomes"] = outcomes.update(j, market, today)
    if s.features.memory:
        ctx.stage = "memory"
        summary["memory"] = memory_after_outcomes(s, j, ctx.llm, mark, today.isoformat())

    ctx.stage = "report"
    start = datetime.combine(today, dtime(0, 0), NY)
    today_orders = [dict(r) for r in j.conn.execute(
        "SELECT symbol, side, qty, status, reason FROM orders WHERE created_at >= ? ORDER BY id",
        (_ts(start),))]
    rpath = ctx.path(s.paths.reports_dir) / f"daily-{today.isoformat()}.md"
    report.write_daily(rpath, {"id": ctx.run_id, "mode": ctx.mode, "formula_version": fm.version,
                               "started_at": now}, today_orders, pos_rows, exit_rows, acct.equity)
    summary.update(report=str(rpath), exits=exit_rows, stops=order_rows, notes=notes,
                   blocked=reasons)
    if tripped:
        ctx.push("insider-bot: KILL SWITCH tripped", "\n".join(reasons), priority=5, tags=["warning"])
    line = (f"equity={acct.equity:.2f} positions={len(acct.positions)} closed={len(exit_rows)} "
            f"stops_placed={sum(1 for r in order_rows if r.get('status') not in ('rejected', 'stop_failed'))} "
            f"outcomes={summary['outcomes']}")
    return "ok", summary, line


def _review(ctx: Ctx, args, fm, sh):
    s, j, deps = ctx.s, ctx.journal, ctx.deps
    today = deps.now().astimezone(NY).date()
    ctx.stage = "market"
    market = ctx.market()
    ctx.stage = "agents"
    llm = ctx.llm()
    if not llm.available():
        log.warning("agents unavailable: no Anthropic API key (research proposal will be 'no change')")
    team = AgentTeam(llm, s)
    ctx.stage = "review"
    res = research_review.run(j, market, team, s, today, formula_dir=ctx.path(s.paths.formula_dir),
                              reports_dir=ctx.path(s.paths.reports_dir))
    ctx.push("insider-bot: weekly review",
             f"Action: {res.get('action')}\nActive: {res.get('active')}"
             + (f"\nShadow: {res['shadow']}" if res.get("shadow") else "")
             + f"\nReport: {Path(res.get('report', '')).name}")
    return "ok", res, f"action={res.get('action')} active={res.get('active')}"


def _backfill(ctx: Ctx, args, fm, sh):
    s, j, deps = ctx.s, ctx.journal, ctx.deps
    now = deps.now()
    ctx.stage = "market"
    market = ctx.market()
    ctx.stage = "edgar"
    client = ctx.edgar()
    n_txns = n_cands = 0
    for i in range(args.days, 0, -1):
        since, until = now - timedelta(days=i), now - timedelta(days=i - 1)
        ctx.stage = "edgar"
        txns = client.fetch(since, until)
        new = _unseen(j, txns)
        j.add_txns(txns, "backfill")
        n_txns += len(new)
        tick = {t.ticker for t in new if t.code == "P" and t.ticker}
        print(f"backfill {until.astimezone(NY).date()}: {len(new)} new txns, {len(tick)} issuers",
              file=sys.stderr)
        if not tick:
            continue
        ctx.stage = "features"
        win = [t for t in j.txns_since(until - timedelta(days=fm.window_days), ("P",)) if t.ticker in tick]
        for f in features.build(win, market, j, until, fm.window_days):
            sc, flags = fml.score(f, fm)
            scores = {fm.version: sc}
            if sh is not None:
                scores[sh.version] = fml.score(f, sh)[0]
            j.add_candidate(ctx.run_id, f, scores, flags, features.hard_filter(f, s.filters), "backfill")
            n_cands += 1
    summary = {"days": args.days, "txns": n_txns, "candidates": n_cands}
    return "ok", summary, f"days={args.days} txns={n_txns} candidates={n_cands}"


def _curate(ctx: Ctx, args, fm, sh):
    """Weekly (Sunday 19:00): the Curator turns staged observations into lessons (features.memory)."""
    s, j = ctx.s, ctx.journal
    summary: dict = {}
    if not s.features.memory or not s.memory.dir:
        return "skipped", summary, "skipped (memory is off)"
    from insiderbot.memory import memory_root
    from insiderbot.memory.curator import curate
    from insiderbot.memory.files import MemoryFiles
    from insiderbot.memory.scorecards import build_scorecards

    files = MemoryFiles(memory_root(s), title="Lessons (insider)")
    if not files.read_staging(None):
        return "ok", summary, "nothing to curate"
    ctx.stage = "curate"
    out = curate(ctx.llm(), s, files, build_scorecards(j), ctx.deps.now().astimezone(NY).date().isoformat())
    summary.update(out)
    ctx.push("insider-bot: weekly lessons", ", ".join(f"{k}={v}" for k, v in out.items()))
    return "ok", summary, " ".join(f"{k}={v}" for k, v in out.items())


JOBS = {"scan": _scan, "reconcile": _reconcile, "review": _review, "backfill": _backfill, "curate": _curate}


# --- status (read-only: no lock, no run row) ----------------------------------------

def _status(deps: Deps) -> int:
    s = deps.settings
    out = [f"insider-bot status  {deps.now().astimezone(NY):%Y-%m-%d %H:%M %Z}",
           f"mode: {s.mode} (Alpaca paper only)",
           f"kill switch: {'ACTIVE' if gate.kill_switch_active(s) else 'off'}"]
    sec = deps.secrets
    out.append("keys: alpaca " + ("set" if sec.alpaca_key and sec.alpaca_secret else "MISSING")
               + ", anthropic " + ("set" if sec.anthropic_key else
                                   "not needed" if s.agents.backend == "claude-cli" else "MISSING")
               + ", edgar identity " + ("set" if sec.edgar_identity else "MISSING"))
    if s.agents.backend == "claude-cli":
        out.append("agents: claude CLI (subscription) "
                   + ("found" if _claude_cli(s) else f"NOT FOUND at {s.agents.claude_cli}"))
    try:
        fdir = _path(deps, s.paths.formula_dir)
        sh = fml.shadow(fdir)
        out.append(f"formula: active {fml.active(fdir).version}, shadow {sh.version if sh else 'none'}")
    except Exception as e:  # noqa: BLE001
        out.append(f"formula: unavailable ({_reason(e)})")
    try:
        broker = deps.broker if deps.broker is not None else \
            AlpacaBroker(sec.alpaca_key or "", sec.alpaca_secret or "")
        a = broker.get_account()
        out.append(f"account: equity ${a.equity:,.2f}  cash ${a.cash:,.2f}  "
                   f"prev close ${a.last_equity:,.2f}" + ("  TRADING BLOCKED" if a.trading_blocked else ""))
        out.append(f"positions ({len(a.positions)}):")
        for p in sorted(a.positions.values(), key=lambda p: p.symbol):
            out.append(f"  {p.symbol:<6} {p.qty:>8g} sh  avg {p.avg_price:,.2f}  value {p.market_value:,.2f}")
    except Exception as e:  # noqa: BLE001
        out.append(f"account: unavailable ({_reason(e)})")
    try:
        j = Journal(_path(deps, s.paths.journal_db))
        for job in ("scan", "reconcile", "review", "backfill"):
            r = j.conn.execute("SELECT * FROM runs WHERE job=? ORDER BY id DESC LIMIT 1", (job,)).fetchone()
            out.append(f"last {job}: " + (f"{r['started_at']} {r['status']} ({r['mode']})" if r else "never"))
        n = datetime.now(timezone.utc)
        jp = _path(deps, s.paths.journal_db)
        dry = _dry_path(jp)
        dry_cost = Journal(dry).month_cost_usd(n.year, n.month) if dry.exists() else 0.0
        paper_cost = j.month_cost_usd(n.year, n.month)
        out.append(f"agent cost this month: ${paper_cost + dry_cost:.2f} of "
                   f"${s.agents.max_monthly_usd:g} (dry-run ${dry_cost:.2f})")
        pk = j.peak_equity()
        out.append(f"equity peak (journal): {'n/a' if pk is None else f'${pk:,.2f}'}")
        wm = j.get_watermark()
        out.append(f"EDGAR watermark: {wm.isoformat() if wm else 'none'}")
    except Exception as e:  # noqa: BLE001
        out.append(f"journal: unavailable ({_reason(e)})")
    print("\n".join(out))
    print("OK status")
    return 0


# --- entry point ---------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="insiderbot", description="Insider-buying paper trading bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="fetch filings, exits, reviews, buys")
    sc.add_argument("--slot", choices=["am", "pm", "poll"])
    sc.add_argument("--mode", choices=["dry-run", "paper", "control"])
    sc.add_argument("--force", action="store_true", help="run even if late for the slot")
    rc = sub.add_parser("reconcile", help="order statuses, stops, equity, outcomes, daily report")
    rc.add_argument("--mode", choices=["dry-run", "paper", "control"])
    rc.add_argument("--init-book", action="store_true",
                    help="control mode: create the simulated book if it does not exist")
    sub.add_parser("review", help="weekly research review")
    sub.add_parser("curate", help="weekly lessons curation (features.memory)")
    sub.add_parser("status", help="one-screen status (read-only)")
    bf = sub.add_parser("backfill", help="journal past Form 4s as backfill candidates")
    bf.add_argument("--days", type=int, default=60)
    return p


def _lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _run(deps: Deps, args) -> int:
    s = deps.settings
    job = args.cmd
    mode = getattr(args, "mode", None) or s.mode
    if job == "backfill":
        mode = "backfill"
    ctx = Ctx(deps, job, mode)
    try:
        ctx.stage = "journal"
        real = ctx.path(s.paths.journal_db)
        dry = _dry_path(real)
        # dry-run never touches the real journal (watermark, txns, decisions, review timing)
        if mode == "control":   # the control never opens the live or dry-run journals
            ctx.journal = Journal(_control_path(real))
            ctx.cost_journals = [ctx.journal]
        else:
            ctx.journal = Journal(dry if mode == "dry-run" else real)
            other = Journal(real if mode == "dry-run" else dry)
            ctx.cost_journals = [ctx.journal, other]
        ctx.stage = "formula"
        fdir = ctx.path(s.paths.formula_dir)
        fm, sh = fml.active(fdir), fml.shadow(fdir)
        ctx.run_id = ctx.journal.start_run(job, mode, fm.version)
        status, summary, line = JOBS[job](ctx, args, fm, sh)
        if ctx.book is not None:
            ctx.stage = "book"
            _save_book(ctx)
        ctx.journal.finish_run(ctx.run_id, status, summary)
    except Exception as e:  # noqa: BLE001 - every failure becomes one ERROR line + a push
        reason = _reason(e)
        msg = f"ERROR: {ctx.stage}: {reason}"
        log.error("%s job failed at %s", job, ctx.stage, exc_info=True)
        if ctx.journal is not None and ctx.run_id is not None:
            try:
                ctx.journal.finish_run(ctx.run_id, "error", {"stage": ctx.stage, "error": reason})
            except Exception:  # noqa: BLE001
                pass
        if mode != "control":
            _push(deps, f"insider-bot: {job} failed", msg, priority=5, tags=["warning"])
        print(msg)
        return 1
    print(f"OK {job} {line}")
    return 0


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = _parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if deps is None:
        try:
            deps = default_deps()
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR: config: {_reason(e)}"
            try:  # best effort: settings are unavailable, so notify's own config decides
                notify.push(f"insider-bot: {args.cmd} failed", msg, priority=5, tags=["warning"],
                            enabled=True)
            except Exception:  # noqa: BLE001
                pass
            print(msg)
            return 1
    if args.cmd == "status":
        return _status(deps)
    if getattr(args, "mode", None) == "control":   # own lock, kill switch, reports; v1 prompts
        deps = dataclasses.replace(deps, settings=control_settings(deps.settings))
    fh = _lock(_path(deps, deps.settings.paths.lock_file))
    if fh is None:
        # no push on lock contention: a scheduled run overlapping another job is routine
        log.warning("%s not run: another job holds the lock", args.cmd)
        if args.cmd in ("scan", "reconcile"):
            print(f"OK {args.cmd} skipped_locked (another job running)")
            return 0
        print("ERROR: lock: another job running")
        return 1
    try:
        return _run(deps, args)
    finally:
        fh.close()
