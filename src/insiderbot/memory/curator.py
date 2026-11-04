"""Weekly Curator (Opus): promotes well-evidenced observations into lessons, edits or retires
lessons, all under rules enforced here in code. Lessons are context for the agents, never rules.

- an add or edit needs n >= memory.min_lesson_n scored outcomes; smaller ones are carried in staging
- at most memory.max_lessons active lessons; overflow adds are rejected
- lessons the owner added (tag "owner") or edited, the owner's pinned section, and lessons the owner deleted are off limits
- the n the Curator claims is capped by the scorecards: an agent-tagged lesson by that agent's scored
  calls, any other lesson by the number of scored views
- the LLM is called before any file is touched, so a failure leaves every file unchanged
"""
from __future__ import annotations

import dataclasses
import json
import re

from insiderbot.agents.llm import load_prompt
from insiderbot.agents.schemas import LessonsUpdate
from insiderbot.agents.team import wrap
from insiderbot.memory.files import Lesson

_DATE = re.compile(r"^- \((\d{4}-\d{2}-\d{2})\) ")


def evidence_cap(cards: dict, tags: list[str]) -> int:
    """The most scored outcomes the scorecards can support for a lesson with these tags."""
    agents = cards.get("agents") or {}
    caps = [int(agents[t]["n"]) for t in tags if t in agents]
    return min(caps) if caps else int(cards.get("n_views") or 0)


def curate(llm, s, files, cards: dict, today: str) -> dict:
    lessons, pinned = files.read_lessons()
    staged = files.read_staging(None)
    retired = files.retired_texts()
    min_n, cap = s.memory.min_lesson_n, s.memory.max_lessons
    data = {"lessons": [dataclasses.asdict(x) for x in lessons], "pinned_by_owner": pinned, "staging": staged,
            "deleted_by_owner": list(retired.values()), "scorecards": cards,
            "rules": {"min_lesson_n": min_n, "max_lessons": cap}}
    upd: LessonsUpdate = llm.structured(
        model=s.memory.curator_model, effort=s.memory.curator_effort,
        system=load_prompt("curator", s.agents.prompt_dir),
        user=f"Today is {today}. Current memory and this week's observations:\n{wrap(json.dumps(data, default=str))}",
        schema=LessonsUpdate)
    by_id = {x.id: x for x in lessons}
    owner_owned = files.owner_owned_ids()
    log: list[str] = []

    def protected(x) -> bool:
        return "owner" in x.tags or x.id in owner_owned

    def capped(n: int, tags: list[str], what: str) -> int:
        cap = evidence_cap(cards, tags)
        if n > cap:
            log.append(f"{what}: claimed n={n} capped at {cap} (scored outcomes in the scorecards)")
            return cap
        return n
    carried = []
    out = {"added": 0, "edited": 0, "retired": 0, "rejected": 0}
    for r in upd.retire:
        x = by_id.get(r.id)
        if x is None or protected(x):
            log.append(f"{r.id}: retire ignored (unknown or the owner's lesson)")
            continue
        del by_id[r.id]
        out["retired"] += 1
        log.append(f"{r.id}: retired by the Curator ({r.reason})")
    for e in upd.edit:
        x = by_id.get(e.id)
        if x is not None and not protected(x):
            e = e.model_copy(update={"n": capped(e.n, x.tags, e.id)})
        if x is None or protected(x):
            log.append(f"{e.id}: edit ignored (unknown or the owner's lesson)")
        elif e.n < min_n:
            out["rejected"] += 1
            log.append(f"{e.id}: edit rejected (n={e.n} < {min_n})")
        else:
            by_id[e.id] = dataclasses.replace(x, text=e.text, evidence=e.evidence, n=e.n)
            out["edited"] += 1
            log.append(f"{e.id}: edited (n={e.n}): {e.text}")
    deleted = {t.strip().lower() for t in retired.values()}
    for a in upd.add:
        a = a.model_copy(update={"n": capped(a.n, list(a.tags), f"add '{a.text[:40]}'")})
        if a.n < min_n:
            out["rejected"] += 1
            carried.append(a)
            log.append(f"add rejected (n={a.n} < {min_n}), kept in staging: {a.text}")
        elif a.text.strip().lower() in deleted:
            out["rejected"] += 1
            log.append(f"add rejected (the owner deleted this lesson): {a.text}")
        elif len(by_id) >= cap:
            out["rejected"] += 1
            log.append(f"add rejected ({cap} active lessons is the maximum): {a.text}")
        else:
            lid = files.allocate_id()
            by_id[lid] = Lesson(lid, a.text, today, a.n, a.evidence, list(a.tags))
            out["added"] += 1
            log.append(f"{lid}: added (n={a.n}): {a.text}")
    files.write_lessons(list(by_id.values()))
    dates = [m.group(1) for ln in staged if (m := _DATE.match(ln))]
    if dates:
        files.clear_staging(max(dates))
    if carried:
        files.append_staging(today, [f"carried (n={a.n} < {min_n}): {a.text} evidence: {a.evidence}"
                                     for a in carried])
    files.append_changelog(today, log or [f"weekly curation: no changes ({upd.summary})"])
    return out
