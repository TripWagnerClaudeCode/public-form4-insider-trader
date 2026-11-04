"""Daily-bar replay of the exit rules, and shadow-vs-active comparison."""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from insiderbot.formula import Formula, score
from insiderbot.research.outcomes import accepted_ny, entry_pos

DEFAULT_COST_BPS = 10.0
TOLERANCE = 0.005  # 0.5 pp


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _none(reason: str) -> dict:
    return {"ret": None, "excess": None, "days": 0, "reason": reason, "mdd": None}


def simulate(path: pd.DataFrame, fm: Formula, cost_bps: float = DEFAULT_COST_BPS) -> dict:
    """Enter at the first row's open; exit on stop (intraday, gap fills at the open), trailing
    (close based, as in exits.check) or time stop. Needs open/low/close/spy_close columns
    (spy_open optional). Missing data returns reason 'nodata' with ret None."""
    need = {"open", "low", "close", "spy_close"}
    if path is None or len(path) == 0 or not need <= set(path.columns):
        return _none("nodata")
    entry = float(path["open"].iloc[0])
    spy0 = float(path["spy_open"].iloc[0]) if "spy_open" in path.columns else float(path["spy_close"].iloc[0])
    if not (_ok(entry) and _ok(spy0)):
        return _none("nodata")
    cost = cost_bps / 1e4
    stop = entry * (1 - fm.stop_loss_pct)
    peak = entry
    mdd = 0.0
    exit_px, exit_i, spy_x, reason = None, 0, None, "open"
    last = None
    for i in range(len(path)):
        row = path.iloc[i]
        o, lo, c, sc = (float(row["open"]), float(row["low"]), float(row["close"]), float(row["spy_close"]))
        if not (_ok(o) and _ok(lo) and _ok(c) and _ok(sc)):
            continue
        last = (i, c, sc)
        if o <= stop:
            exit_px, reason = o, "stop"
        elif lo <= stop:
            exit_px, reason = stop, "stop"
        if exit_px is not None:
            exit_i, spy_x = i, sc
            mdd = min(mdd, exit_px / peak - 1)
            break
        peak = max(peak, c)
        mdd = min(mdd, c / peak - 1)
        if peak >= entry * (1 + fm.take_profit_pct) and c <= peak * (1 - fm.trailing_pct):
            exit_px, exit_i, spy_x, reason = c, i, sc, "trailing"
            break
        if i >= fm.time_stop_days:
            exit_px, exit_i, spy_x, reason = c, i, sc, "time"
            break
    if exit_px is None:
        if last is None:
            return _none("nodata")
        exit_i, exit_px, spy_x = last
    ret = exit_px * (1 - cost) / (entry * (1 + cost)) - 1
    return {"ret": ret, "excess": ret - (spy_x / spy0 - 1), "days": exit_i, "reason": reason, "mdd": mdd}


def rank_ic(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    v = pd.Series(xs).rank().corr(pd.Series(ys).rank())
    return None if v is None or not math.isfinite(v) else float(v)


def build_cands(journal, market, today: date, horizon: int | None, since: date | None = None) -> list[dict]:
    """Live, unfiltered candidates matured at `horizon` (any horizon when None; asof strictly after `since`), with their
    stored features, outcomes and a replay path (None when bars are missing)."""
    outs = journal.outcomes_all()
    rows = [c for c in journal.candidates()
            if c["source"] == "live" and not c["filter_reason"]
            and (horizon in outs.get(c["id"], {}) if horizon else bool(outs.get(c["id"])))
            and (since is None or c["asof"].date() > since)]
    cands = [{"id": c["id"], "ticker": c["ticker"], "asof": c["asof"], "features": c["features"],
              "excess": {h: v[1] for h, v in outs[c["id"]].items()},
              "ret": {h: v[0] for h, v in outs[c["id"]].items()}, "path": None} for c in rows]
    if not cands or market is None:
        return cands
    start = min(accepted_ny(c["features"].latest_accepted_at).date() for c in cands) - timedelta(days=7)
    bars = market.bars(sorted({c["ticker"] for c in cands} | {"SPY"}), start, today)
    spy = bars.get("SPY")
    if spy is None or spy.empty:
        return cands
    for c in cands:
        b = bars.get(c["ticker"])
        if b is None or b.empty:
            continue
        i0 = entry_pos(b.index, c["features"].latest_accepted_at, "open")
        if i0 is None:
            continue
        p = b.iloc[i0:][["open", "high", "low", "close"]].join(
            spy[["open", "close"]].rename(columns={"open": "spy_open", "close": "spy_close"}), how="inner")
        if len(p):
            c["path"] = p
    return cands


def _metrics(cands: list[dict], fm: Formula, horizon: int, cost_bps: float, window: int) -> dict:
    """Metrics for one formula. Replay trades are limited to `window` trading days after entry
    (window+1 bars); candidates whose path is shorter are skipped for every formula so the
    replayed set lines up with the matured set."""
    scored = [(score(c["features"], fm)[0], c) for c in cands if horizon in c["excess"]]
    ic = rank_ic([s for s, _ in scored], [c["excess"][horizon] for _, c in scored])
    buys = [c for s, c in scored if s >= fm.buy_threshold]
    exc = [c["excess"][horizon] for c in buys]
    sims = [r for c in buys if c.get("path") is not None and len(c["path"]) >= window + 1
            for r in [simulate(c["path"].iloc[:window + 1], fm, cost_bps)] if r["excess"] is not None]
    return {
        "rank_ic": ic,
        "n_buys": len(buys),
        "mean_excess": sum(exc) / len(exc) if exc else None,
        "n_replayed": len(sims),
        "replay_excess": sum(r["excess"] for r in sims) / len(sims) if sims else None,
        "worst_drawdown": min(r["mdd"] for r in sims) if sims else None,
    }


EXIT_PARAMS = ("stop_loss_pct", "time_stop_days", "take_profit_pct", "trailing_pct")
BUY_PARAMS = ("buy_threshold", "review_threshold", "tiers")


def primary_metric(active: Formula, other: Formula) -> str | None:
    """The metric a shadow must win on, chosen by what it changes: weights -> rank_ic;
    thresholds/tiers -> mean_excess; exit params -> replay_excess. window_days (and a shadow
    identical to active) has no stored-data metric, so it returns None and never wins."""
    if active.weights != other.weights:
        return "rank_ic"
    if any(getattr(active, k) != getattr(other, k) for k in BUY_PARAMS):
        return "mean_excess"
    if any(getattr(active, k) != getattr(other, k) for k in EXIT_PARAMS):
        return "replay_excess"
    return None


def compare(cands: list[dict], active: Formula, other: Formula, horizon: int,
            min_n: int = 30, cost_bps: float = DEFAULT_COST_BPS) -> dict:
    """Decide on the shadow's own metric (see primary_metric).

    'other' wins only with n >= min_n, the primary metric present and strictly better (rank IC
    must also be positive), every other available metric no more than 0.5 pp worse, and for exit
    params the worst drawdown no worse by more than 0.5 pp. A metric missing on `other` while
    present on active blocks the win. With n >= min_n and a defined primary on both sides, a
    shadow that does not win is 'active' (it lost); otherwise 'undecided'.
    Replay window: the horizon (10 days) for weight/threshold shadows; for exit-param shadows the
    larger time stop of the two formulas (so a time-stop change is observable)."""
    prim = primary_metric(active, other)
    exit_prim = prim == "replay_excess"
    window = max(active.time_stop_days, other.time_stop_days) if exit_prim else horizon
    a = _metrics(cands, active, horizon, cost_bps, window)
    o = _metrics(cands, other, horizon, cost_bps, window)
    n = sum(1 for c in cands if horizon in c["excess"])
    winner = "undecided"
    if n >= min_n and prim is not None and a[prim] is not None and o[prim] is not None:
        def not_worse(k):
            if o[k] is None:
                return a[k] is None
            return a[k] is None or o[k] >= a[k] - TOLERANCE
        better = o[prim] > a[prim] and (prim != "rank_ic" or o[prim] > 0)
        others_ok = all(not_worse(k) for k in ("rank_ic", "mean_excess", "replay_excess") if k != prim)
        if exit_prim and not not_worse("worst_drawdown"):
            others_ok = False
        winner = "other" if (better and others_ok) else "active"
    return {"active": a, "other": o, "n": n, "winner": winner, "primary": prim, "window": window}
