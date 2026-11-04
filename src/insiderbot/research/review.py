"""Weekly review: shadow comparison / promotion, or one bounded formula proposal."""
from __future__ import annotations

import difflib
import json
import math
from datetime import date
from pathlib import Path

import yaml

from insiderbot import formula as fml
from insiderbot.config import ROOT
from insiderbot.journal import report
from insiderbot.research import replay, stats

PARAMS = ("review_threshold", "buy_threshold", "stop_loss_pct", "time_stop_days",
          "take_profit_pct", "trailing_pct", "window_days")
INT_PARAMS = {"time_stop_days", "window_days"}
NOOP = "no effective change"


def _yaml(fm: fml.Formula) -> str:
    return yaml.safe_dump(fm.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100)


# All dates compared with `since` are UTC dates (journal timestamps are UTC-aware);
# `today` is the caller's run date.
def _adoption(journal, fm: fml.Formula) -> date | None:
    ev = journal.formula_events(version=fm.version, event="promoted")
    return ev[-1]["ts"].date() if ev else None


def _count_matured(journal, horizon: int, since: date | None) -> int:
    outs = journal.outcomes_all()
    return sum(1 for c in journal.candidates()
               if c["source"] == "live" and not c["filter_reason"] and horizon in outs.get(c["id"], {})
               and (since is None or c["asof"].date() >= since))


def build_proposal(active: fml.Formula, out, today: date) -> tuple[fml.Formula | None, list[str]]:
    """Apply a ResearchOut to a copy of active. Returns (formula or None, errors)."""
    errs: list[str] = []
    data = active.model_dump()
    changes = 0

    def num(v):
        return type(v) in (int, float) and math.isfinite(v)  # bool is rejected

    for k, v in out.new_weights.items():
        if k not in active.weights:
            errs.append(f"unknown weight key: {k}")
        elif not num(v):
            errs.append(f"weight {k} not a finite number")
        elif float(v) != active.weights[k]:
            data["weights"][k] = float(v)
            changes += 1
    for k, v in out.new_params.items():
        if k not in PARAMS:
            errs.append(f"unknown param key: {k}")
        elif not num(v):
            errs.append(f"param {k} not a finite number")
        else:
            nv = int(round(v)) if k in INT_PARAMS else float(v)  # compare after rounding
            if nv != getattr(active, k):
                data[k] = nv
                changes += 1
    if changes > 1:
        errs.append(f"more than one change proposed ({changes})")
    if changes == 0 and not errs:
        return None, [NOOP]
    if errs:
        return None, errs
    data.update(version="v000", parent=active.version, author="review-agent", created=today,
                rationale=out.rationale, evidence={"what_we_learned": out.what_we_learned})
    try:
        return fml.Formula(**data), []
    except Exception as e:  # noqa: BLE001
        return None, [f"invalid formula: {e}"]


def run(journal, market, team, settings, today: date, formula_dir: Path | None = None,
        reports_dir: Path | None = None) -> dict:
    b = settings.bounds
    fdir = Path(formula_dir) if formula_dir else ROOT / settings.paths.formula_dir
    rdir = Path(reports_dir) if reports_dir else ROOT / Path(settings.paths.reports_dir).expanduser()
    horizon = b.promotion_horizon_days
    active = fml.active(fdir)
    shadow = fml.shadow(fdir)
    summary: dict = {"active": active.version, "action": "none"}
    diff, learned, report_stats = "", "", {}

    if shadow is not None:
        ev = journal.formula_events(version=shadow.version, event="shadow")
        since = date.fromisoformat(ev[-1]["detail"]["since"]) if ev and ev[-1]["detail"].get("since") else None
        cands = replay.build_cands(journal, market, today, horizon, since=since)
        cmp_ = replay.compare(cands, active, shadow, horizon, min_n=b.min_matured_for_promotion)
        report_stats = {"shadow": shadow.version, "compare": cmp_}
        summary.update(shadow=shadow.version, compare=cmp_)
        age_days = (today - since).days if since else 0
        if cmp_["winner"] == "other":
            errs = fml.validate(shadow, active, b)  # re-validate against the hard bounds
            if errs:
                journal.add_formula_event(shadow.version, "retired", {"errors": errs, "reason": "failed validation"})
                fml.set_shadow(None, fdir)
                summary.update(action="retired", errors=errs)
                learned = f"{shadow.version} retired: failed validation at promotion."
            else:
                journal.add_formula_event(shadow.version, "promoted",
                                          {"compare": cmp_, "replaces": active.version})
                fml.set_active(shadow.version, fdir)
                fml.set_shadow(None, fdir)
                summary["action"] = "promoted"
                diff = "".join(difflib.unified_diff(_yaml(active).splitlines(True),
                                                    _yaml(shadow).splitlines(True),
                                                    active.version, shadow.version))
                learned = f"{shadow.version} promoted over {active.version} on {cmp_['primary']}."
        elif cmp_["winner"] == "active" or (since is not None and age_days >= b.shadow_max_weeks * 7):
            why = "lost" if cmp_["winner"] == "active" else "undecided after max weeks"
            journal.add_formula_event(shadow.version, "retired",
                                      {"compare": cmp_, "age_days": age_days, "reason": why})
            fml.set_shadow(None, fdir)
            summary["action"] = "retired"
            learned = f"{shadow.version} retired ({why})."
        else:
            summary["action"] = "keep"
    else:
        since = _adoption(journal, active)
        n = _count_matured(journal, horizon, since)
        summary["matured"] = n
        report_stats = {"matured_live_10d": n, "min_for_proposal": b.min_matured_for_proposal}
        if n >= b.min_matured_for_proposal:
            st = stats.weekly(journal, active, market, today, since)
            report_stats = st
            ytxt = _yaml(active) + "\nBounds (hard, not editable):\n" + json.dumps(b.model_dump(), indent=1)
            out = team.research(st, ytxt)
            learned = out.what_we_learned
            if not out.change:
                summary["action"] = "no_change"
            else:
                new, errs = build_proposal(active, out, today)
                if new is not None:
                    errs = fml.validate(new, active, b)
                if errs == [NOOP]:
                    summary["action"] = "no_change"
                elif errs:
                    journal.add_formula_event(active.version, "rejected", {
                        "errors": errs, "new_weights": out.new_weights, "new_params": out.new_params,
                        "rationale": out.rationale})
                    summary.update(action="rejected", errors=errs)
                else:
                    path = fml.save(new, fdir)
                    ver = path.stem
                    saved = fml.load(ver, fdir)
                    journal.add_formula_event(ver, "proposed", {"parent": active.version, "rationale": out.rationale})
                    journal.add_formula_event(ver, "shadow", {"since": today.isoformat()})
                    fml.set_shadow(ver, fdir)
                    summary.update(action="shadow", shadow=ver)
                    diff = "".join(difflib.unified_diff(_yaml(active).splitlines(True),
                                                        _yaml(saved).splitlines(True), active.version, ver))
        else:
            summary["action"] = "no_proposal"

    flat = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
            for k, v in report_stats.items()}
    summary["report"] = str(report.write_weekly(rdir / "weekly" / f"{today.isoformat()}.md", flat, diff, learned))
    return summary
