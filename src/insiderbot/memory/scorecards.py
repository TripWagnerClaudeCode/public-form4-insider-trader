"""insider-bot scorecards from the PM's decisions and the candidates' 20-day excess vs SPY (no LLM).

The PM's call counts as a direction: buy = +1 (long), pass = -1 (veto). hit = the direction's sign
matches the excess sign. Exit checks ("exit"/"hold") are not buy-vs-pass calls and are excluded, and so
are decisions that never were a real review: agent errors and budget/no-key skips are journaled as
"pass" but carry an "error" or "skipped" key in their reports.
Keys mirror trading-bot's scorecards (n_views, agents, symbols) so the memory block is shared.
"""
from __future__ import annotations

import datetime as dt
import json

NO_RECORD = "No track record yet: candidates are scored 20 trading days after the filing."


def reviewed(reports) -> bool:
    """True for a real PM decision (not an agent failure or a budget/no-key skip)."""
    try:
        r = json.loads(reports or "{}")
    except (TypeError, ValueError):
        return False
    return isinstance(r, dict) and "error" not in r and "skipped" not in r


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def build_scorecards(journal, horizon: int = 20) -> dict:
    rows = journal.conn.execute(
        "SELECT d.candidate_id, d.ticker, d.action, d.conviction, d.reports, o.excess, c.asof "
        "FROM decisions d JOIN outcomes o ON o.candidate_id = d.candidate_id AND o.horizon = ? "
        "JOIN candidates c ON c.id = d.candidate_id "
        "WHERE d.id IN (SELECT MAX(id) FROM decisions WHERE action IN ('buy', 'pass') GROUP BY candidate_id) "
        "ORDER BY c.asof", (horizon,)).fetchall()
    buy, pas, calls = [], [], []
    conv: dict[int, list[float]] = {}
    flagged, clean = [], []
    symbols: dict[str, list[dict]] = {}
    risk: list[tuple[int, float]] = []
    for _cid, tk, action, cv, reports, ex, asof in rows:
        if not reviewed(reports):
            continue
        try:
            rep = json.loads(reports or "{}")
        except (TypeError, ValueError):
            rep = {}
        rr = rep.get("risk_review") if isinstance(rep, dict) else None
        if isinstance(rr, dict):   # Phase 3: score the PM's own call; the Risk Reviewer gets its own row
            action = "buy" if (rep.get("pm") or {}).get("action") == "buy" else action
            if not rr.get("error"):
                risk.append((1 if rr.get("approve") else -1, ex))
        d = 1 if action == "buy" else -1
        (buy if d > 0 else pas).append(ex)
        conv.setdefault(int(cv or 0), []).append(ex)
        calls.append((d, ex))
        try:
            rf = ((json.loads(reports or "{}").get("analysts") or {}).get("filing") or {}).get("red_flags") or []
        except (TypeError, ValueError, AttributeError):
            rf = []
        (flagged if rf else clean).append(ex)
        symbols.setdefault(tk, []).append({"as_of": str(asof)[:10], "pm_direction": d, "excess_univ": ex,
                                           "action": action})
    agents = {}
    for name, cs in (("pm", calls), ("risk_reviewer", risk)):
        if cs:
            agents[name] = {"n": len(cs), "hit_rate": sum(1 for d, ex in cs if (d > 0) == (ex > 0) and ex != 0)
                            / len(cs), "mean_signed_excess": _mean([d * ex for d, ex in cs])}
    return {
        "n_views": len(calls),
        "agents": agents,
        "pm": {"buy": {"n": len(buy), "mean_excess": _mean(buy)},
               "pass": {"n": len(pas), "mean_excess": _mean(pas),
                        "would_have_won": (sum(1 for x in pas if x > 0) / len(pas)) if pas else None},
               "by_conviction": {k: {"n": len(v), "mean_excess": _mean(v)} for k, v in sorted(conv.items())}},
        "red_flags": {"with": {"n": len(flagged), "mean_excess": _mean(flagged)},
                      "without": {"n": len(clean), "mean_excess": _mean(clean)}},
        "symbols": {k: v[-3:] for k, v in symbols.items()},
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def render_scorecards(cards: dict) -> str:
    L = ["# Scorecard (insider)", "", f"*Regenerated {dt.date.today()} by the bot. Edits here are overwritten.*", ""]
    if not cards["n_views"]:
        return "\n".join(L + [NO_RECORD, ""])
    pm = cards["pm"]
    a = cards["agents"]["pm"]
    L += [f"Scored PM decisions: {cards['n_views']} (20 trading days, excess vs SPY)", "",
          f"- PM hit rate {a['hit_rate']:.0%}, mean signed excess {_pct(a['mean_signed_excess'])}",
          f"- Buys: {pm['buy']['n']}, mean excess {_pct(pm['buy']['mean_excess'])}",
          f"- Passes (vetoes): {pm['pass']['n']}, mean excess {_pct(pm['pass']['mean_excess'])}; "
          f"would have won {('n/a' if pm['pass']['would_have_won'] is None else f'{pm['pass']['would_have_won']:.0%}')}",
          f"- Filing red flags: with {cards['red_flags']['with']['n']} ({_pct(cards['red_flags']['with']['mean_excess'])}),"
          f" without {cards['red_flags']['without']['n']} ({_pct(cards['red_flags']['without']['mean_excess'])})", "",
          "| conviction | decisions | mean excess |", "|---|---|---|"]
    for k, v in pm["by_conviction"].items():
        L.append(f"| {k} | {v['n']} | {_pct(v['mean_excess'])} |")
    return "\n".join(L + [""])
