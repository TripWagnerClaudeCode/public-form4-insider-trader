"""Forward outcomes of every candidate vs SPY (close-to-close)."""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
HORIZONS = (5, 10, 20, 40)
MAX_GAP_DAYS = 5  # first bar must be this close to the acceptance date, else data is stale


def accepted_ny(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NY)


def entry_pos(index: pd.Index, accepted: datetime, kind: str) -> int | None:
    """Position of the first bar after `accepted`: kind 'close' = first close after it,
    'open' = first open after it. None when no bar qualifies or the data is stale."""
    a = accepted_ny(accepted)
    cutoff = time(16, 0) if kind == "close" else time(9, 30)
    d0 = a.date()
    for i, ts in enumerate(index):
        d = ts.date()
        if d > d0 or (d == d0 and a.time() < cutoff):
            return i if (d - d0).days <= MAX_GAP_DAYS else None
    return None


def _px(df: pd.DataFrame, ts, col: str = "close") -> float | None:
    try:
        v = float(df.at[ts, col])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def outcome_for(bars: pd.DataFrame, spy: pd.DataFrame, accepted: datetime, h: int):
    """(ret, excess) at horizon h trading days, or None if not matured / data missing."""
    i0 = entry_pos(bars.index, accepted, "close")
    if i0 is None:
        return None
    d0 = bars.index[i0]
    if d0 not in spy.index:
        return None
    j0 = spy.index.get_loc(d0)
    if j0 + h >= len(spy):
        return None
    dh = spy.index[j0 + h]
    c0, ch = _px(bars, d0), _px(bars, dh)
    s0, sh = _px(spy, d0), _px(spy, dh)
    if None in (c0, ch, s0, sh):
        return None
    ret = ch / c0 - 1
    return ret, ret - (sh / s0 - 1)


def update(journal, market, today: date) -> int:
    """Store ret/excess for every matured (candidate, horizon). Returns rows stored."""
    cands = {c["id"]: c for c in journal.candidates()}
    todo: dict[int, list[int]] = {}
    for h in HORIZONS:
        for r in journal.pending_outcomes(h):
            if r["candidate_id"] in cands:
                todo.setdefault(r["candidate_id"], []).append(h)
    if not todo:
        return 0
    syms = sorted({cands[i]["ticker"] for i in todo} | {"SPY"})
    start = min(accepted_ny(cands[i]["features"].latest_accepted_at).date() for i in todo) - timedelta(days=7)
    bars = market.bars(syms, start, today)
    spy = bars.get("SPY")
    if spy is None or spy.empty:
        return 0
    n = 0
    for cid, hs in todo.items():
        c = cands[cid]
        b = bars.get(c["ticker"])
        if b is None or b.empty:
            continue
        for h in hs:
            res = outcome_for(b, spy, c["features"].latest_accepted_at, h)
            if res is not None:
                journal.set_outcome(cid, h, res[0], res[1])
                n += 1
    return n
