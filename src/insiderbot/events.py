"""Intraday Event Review for insider-bot held names (spec 2026-11-04 Phase 3).

In the 5-minute poll, code looks for triggers on HELD names: a move of at least k x daily sigma against
the prior close, or a new 8-K/10-Q/10-K (each filing URL reviewed once, ever). Each trigger opens a
review: the Event analyst (Sonnet, web research) says hold or exit, then the Event reviewer (Opus)
answers on its own. Only when BOTH say exit does the position get an agent exit vote, which the
existing exits.check applies (min_hold_days_for_agent_exit; whole position: the strategy has no trims).
Insider-bot's exits (stop, trailing, take-profit, time stop) are unchanged.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Trigger:
    symbol: str
    kind: str          # "move" | "filing"
    detail: str
    z: float
    filing: dict | None = None


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def find_triggers(symbols, mids: dict, prev_close: dict, sigma: dict, filings: dict, k: float) -> list[Trigger]:
    out_f, out_m = [], []
    for sym in symbols:
        mid, prev, sig = mids.get(sym), prev_close.get(sym), sigma.get(sym)
        z = (mid / prev - 1) / sig if _ok(mid) and _ok(prev) and _ok(sig) else 0.0
        for f in filings.get(sym) or []:
            out_f.append(Trigger(sym, "filing", f"{f.get('form')} filed {f.get('date')}: {f.get('url')}", z, f))
        if abs(z) >= k:
            out_m.append(Trigger(sym, "move", f"{mid / prev - 1:+.2%} vs prior close ({z:+.1f} sigma)", z))
    return out_f + sorted(out_m, key=lambda t: -abs(t.z))


def refs(bars: dict) -> tuple[dict[str, float], dict[str, float]]:
    """(prior close, 20-day daily return sigma) per symbol from completed daily bars."""
    prev, sig = {}, {}
    for sym, df in (bars or {}).items():
        try:
            c = df["close"].dropna()
        except (KeyError, TypeError, AttributeError):
            continue
        if len(c) >= 2:
            prev[sym] = float(c.iloc[-1])
            s = float(c.pct_change().dropna().tail(20).std())
            if math.isfinite(s) and s > 0:
                sig[sym] = s
    return prev, sig


def run_event_reviews(team, j, held: dict, mids: dict, prev_close: dict, sigma: dict, day: dt.date,
                      run_id: int, mode: str, cap: int, k: float, notes: list[str], filings_fn,
                      memory_fn=None, text_fn=None, per_run: int | None = None) -> dict[str, bool]:
    """Event Reviews for held names; returns {symbol: True} for both-agree exits. Never raises."""
    syms = sorted(held)
    seen = {r["filing_url"] for r in j.conn.execute(   # a failed review does not count: the filing is retried
        "SELECT filing_url FROM events WHERE filing_url IS NOT NULL AND action <> 'error'")}
    filings: dict[str, list] = {}
    for sym in syms:
        try:
            fl = [f for f in filings_fn(sym, day - dt.timedelta(days=1)) if f.get("url") and f["url"] not in seen]
        except Exception as e:  # noqa: BLE001
            notes.append(f"EDGAR filing check {sym} failed: {type(e).__name__}")
            continue
        if fl:
            filings[sym] = fl[:1]
    today = j.events_today(day, mode)
    done = {(e["symbol"], e["kind"]) for e in today}
    left = cap - len(today)
    votes: dict[str, bool] = {}
    reviewed: set[str] = set()
    for t in find_triggers(syms, mids, prev_close, sigma, filings, k):
        if (t.symbol, t.kind) in done and t.kind != "filing":
            continue
        if t.symbol in reviewed:
            continue
        if left <= 0:
            notes.append("event review: daily cap reached")
            break
        if per_run is not None and len(reviewed) >= per_run:
            notes.append("event review: per-poll cap reached (the rest wait for the next poll)")
            break
        reviewed.add(t.symbol)
        left -= 1
        p = held[t.symbol]
        pos = {"qty": getattr(p, "qty", None), "avg_price": getattr(p, "avg_price", None),
               "market_value": getattr(p, "market_value", None), "mid": mids.get(t.symbol)}
        furl = (t.filing or {}).get("url")
        try:
            text = text_fn(furl) if (text_fn and furl) else None
        except Exception:  # noqa: BLE001
            text = None
        try:
            mem = memory_fn(t.symbol) if memory_fn else None
            a, arec, b = team.event_review(t.symbol, t.detail, t.kind, pos, text, memory=mem)
        except Exception as e:  # noqa: BLE001 - one failed review never stops the scan
            log.warning("event review %s failed: %s", t.symbol, e)
            j.record_event(day, t.symbol, t.kind, t.detail, {"error": f"{type(e).__name__}: {e}"[:300]}, None,
                           "error", mids.get(t.symbol), run_id, mode, filing_url=furl)
            continue
        both = a.action == "exit" and b is not None and b.action == "exit"
        if both:
            votes[t.symbol] = True
            notes.append(f"event review: both agents vote exit {t.symbol} ({t.kind})")
        j.record_event(day, t.symbol, t.kind, t.detail, arec, {"call": b.model_dump()} if b else None,
                       "exit" if both else "hold", mids.get(t.symbol), run_id, mode, filing_url=furl)
    return votes
