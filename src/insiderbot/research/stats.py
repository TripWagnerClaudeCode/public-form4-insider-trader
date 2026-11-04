"""Weekly outcome statistics (live rows only) and the rules-only control book."""
from __future__ import annotations

from datetime import date, datetime, timezone

from insiderbot.formula import FEATURE_KEYS, Formula, score, transforms
from insiderbot.research.outcomes import accepted_ny
from insiderbot.research.replay import DEFAULT_COST_BPS, build_cands, rank_ic, simulate

BUCKETS = ((0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 101))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _live(journal) -> tuple[list[dict], dict]:
    outs = journal.outcomes_all()
    rows = [c for c in journal.candidates() if c["source"] == "live" and not c["filter_reason"]]
    return rows, outs


def control_book(journal, market, fm: Formula, start: date, today: date | None = None,
                 cost_bps: float = DEFAULT_COST_BPS, weight: float = 0.08) -> dict:
    """Replay every live candidate scoring >= buy_threshold from `start`, no agents."""
    today = today or date.today()
    cands = build_cands(journal, market, today, None)
    seen, trades = set(), []
    for c in sorted(cands, key=lambda c: c["asof"]):
        if c["id"] in seen or c["asof"].date() < start or c["path"] is None:
            continue
        seen.add(c["id"])
        if score(c["features"], fm)[0] < fm.buy_threshold:
            continue
        r = simulate(c["path"], fm, cost_bps)
        if r["ret"] is not None:
            trades.append(r)
    equity = 1.0
    for r in trades:
        equity *= 1 + weight * r["ret"]
    wins = [r["ret"] for r in trades if r["ret"] > 0]
    return {
        "n": len(trades), "mean_ret": _mean([r["ret"] for r in trades]),
        "mean_excess": _mean([r["excess"] for r in trades]),
        "hit_rate": len(wins) / len(trades) if trades else None,
        "worst_drawdown": min((r["mdd"] for r in trades), default=None),
        "equity_end": equity,
    }


def _real_book(journal, cost_bps: float, outs: dict) -> dict:
    closed = [p for p in journal.closed_positions()
              if p["entry_price"] and p["exit_price"] and p["entry_price"] > 0]
    rets = [p["exit_price"] / p["entry_price"] - 1 for p in closed]
    wins, losses = [r for r in rets if r > 0], [r for r in rets if r <= 0]
    reasons: dict[str, dict] = {}
    for p, r in zip(closed, rets):
        d = reasons.setdefault(p["exit_reason"] or "unknown", {"n": 0, "hits": 0})
        d["n"] += 1
        d["hits"] += r > 0
    mean = _mean(rets)
    return {
        "n": len(rets),
        "mean_excess_10d": _mean([outs[p["candidate_id"]][10][1] for p in closed
                                   if p["candidate_id"] in outs and 10 in outs[p["candidate_id"]]]),
        "hit_rate": len(wins) / len(rets) if rets else None,
        "avg_win": _mean(wins), "avg_loss": _mean(losses), "paper_mean_ret": mean,
        # first-pass simplification: modeled = paper return less a flat 2x cost_bps (10 bps/side)
        "modeled_mean_ret": None if mean is None else mean - 2 * cost_bps / 1e4,
        "exit_reasons": {k: {"n": v["n"], "hit_rate": v["hits"] / v["n"]} for k, v in reasons.items()},
    }


def weekly(journal, fm: Formula, market=None, today: date | None = None,
           start: date | None = None, cost_bps: float = DEFAULT_COST_BPS) -> dict:
    """Weekly stats, live rows only. First-pass simplifications: modeled-cost P&L is a flat
    2 x cost_bps off the paper return, and real-book "P&L vs SPY" is the mean 10-day excess of
    the names bought, not an equity-curve comparison."""
    rows, outs = _live(journal)
    out: dict = {"n_live_candidates": len(rows)}
    feat_ic = {}
    for h in (10, 20):
        sub = [c for c in rows if h in outs.get(c["id"], {})]
        ys = [outs[c["id"]][h][1] for c in sub]
        tx = [transforms(c["features"]) for c in sub]
        out[f"n_matured_{h}d"] = len(sub)
        feat_ic[f"{h}d"] = {k: rank_ic([t[k] for t in tx], ys) for k in FEATURE_KEYS}
        sc = [score(c["features"], fm)[0] for c in sub]
        out[f"score_ic_{h}d"] = rank_ic(sc, ys)
        out[f"score_buckets_{h}d"] = {
            f"{lo}-{min(hi, 100)}": {"n": len(b), "mean_excess": _mean(b)}
            for lo, hi in BUCKETS
            for b in [[y for s, y in zip(sc, ys) if lo <= s < hi]] if b}
    out["feature_ic"] = feat_ic
    actions = journal.decision_actions()
    agent = {}
    for act in ("buy", "pass"):
        ex = [outs[c["id"]][10][1] for c in rows if actions.get(c["id"]) == act and 10 in outs.get(c["id"], {})]
        agent[act] = {"n": len(ex), "mean_excess_10d": _mean(ex)}
    out["agent_buy_vs_pass"] = agent
    out["real_book"] = _real_book(journal, cost_bps, outs)
    if market is not None and today is not None:
        s = start or min((c["asof"].date() for c in rows), default=today)
        out["control_book"] = control_book(journal, market, fm, s, today, cost_bps)
    out["agent_cost_usd_total"] = journal.total_cost_usd()
    now = datetime.now(timezone.utc)  # same month convention as Journal.add_cost
    out["agent_cost_usd_month"] = journal.month_cost_usd(now.year, now.month)
    return out
