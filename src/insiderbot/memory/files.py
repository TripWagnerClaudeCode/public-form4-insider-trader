"""The human-readable memory layer, kept in a folder the owner can read and edit (for example a synced one).

lessons.md     curated lessons; "## Pinned by you" is never edited by the bot
staging.md     dated daily observations from the Post-Trade Reviewer
changelog.md   every change to lessons, with the reason
scorecard.md   regenerated each reconcile (overwritten)
.state.json    the lessons the bot last wrote, ids the owner retired, lessons the owner edited

the owner's edits win:
- the bot always reads lessons.md as the source of truth; headings match case-insensitively
- an id the bot wrote that is now missing was deleted by the owner: it is retired and never written again
- a bot lesson whose text the owner changed becomes the owner's (the Curator can't edit or retire it)
- a bullet without an id is a lesson the owner added and gets the next id (a bullet whose text is a
  lesson the bot wrote keeps that lesson's id)
- free text, other sections and unrecognized lines are kept verbatim on every rewrite
- Syncthing conflict copies are never read
A missing file, or one without the "## Active lessons" heading, while the bot has lessons on record
is a MemoryAnomaly: nothing is retired, and callers run without memory.

Everything the bot writes is single-line (newlines collapsed), so model text can never forge a
heading, a bullet, or the owner's pinned section.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PINNED, ACTIVE = "## Pinned by you", "## Active lessons"
_PIN_RE = re.compile(r"^##\s*pinned\s+by\s+you\s*$", re.IGNORECASE)
_ACT_RE = re.compile(r"^##\s*active\s+lessons\s*$", re.IGNORECASE)
_HEAD = re.compile(r"^- \[(L-\d{4})\] \((\d{4}-\d{2}-\d{2}), n=(\d+)(?:, tags: ([^)]*))?\) (.*)$")
_STAGE = re.compile(r"^- \((\d{4}-\d{2}-\d{2})\) ")
NOTE = ("> Edit freely. Keep the two `##` headings. Bot lessons start with `- [L-NNNN]`: you can edit their "
        "text or delete them (deleted ones never come back). Add your own as `- your lesson`. Text under "
        "\"Pinned by you\" is yours and is never changed by the bot.")


class MemoryAnomaly(RuntimeError):
    """lessons.md is missing or unreadable as lessons while the bot has lessons on record."""


@dataclass
class Lesson:
    id: str
    text: str
    added: str
    n: int
    evidence: str = ""
    tags: list[str] = field(default_factory=list)


def one_line(s) -> str:
    return " ".join(str(s).split())


def _tag(t) -> str:
    return one_line(str(t).replace(",", " ").replace(")", " ").replace("(", " "))


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(5):   # Windows: an editor or Syncthing may hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)


@dataclass
class _Parsed:
    exists: bool
    has_active: bool
    lessons: list[Lesson]
    pinned: str
    untagged: list[str]
    extra: list[str]          # unrecognized lines in the active section, kept verbatim
    sections: list[str]       # other "## ..." sections, kept verbatim


class MemoryFiles:
    def __init__(self, root: Path, title: str = "Lessons", today: Callable[[], str] | None = None,
                 confirm_delay: float = 0.5):
        self.root = Path(root).expanduser()
        self.title = title
        self._today = today or (lambda: str(dt.date.today()))
        self.confirm_delay = confirm_delay

    # ------------------------------------------------------------------ state
    def _state(self) -> dict:
        p = self.root / ".state.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        return {"written": list(d.get("written", [])), "retired": list(d.get("retired", [])),
                "max_id": int(d.get("max_id", 0)), "lessons": dict(d.get("lessons", {})),
                "owner_owned": list(d.get("owner_owned", []))}

    def _save_state(self, st: dict) -> None:
        _atomic(self.root / ".state.json", json.dumps(st, indent=1, sort_keys=True))

    def retired_ids(self) -> set[str]:
        return set(self._state()["retired"])

    def retired_texts(self) -> dict[str, str]:
        """{id: last text the bot wrote} for lessons the owner deleted (so the Curator can't re-add them)."""
        st = self._state()
        return {i: st["lessons"][i]["text"] for i in st["retired"] if i in st["lessons"]}

    def owner_owned_ids(self) -> set[str]:
        """Bot lessons whose text the owner edited: the Curator may not edit or retire them."""
        return set(self._state()["owner_owned"])

    def allocate_id(self) -> str:
        st = self._state()
        p = self._parse()
        st["max_id"] = max([st["max_id"], *(int(x.id[2:]) for x in p.lessons),
                            *(int(i[2:]) for i in st["retired"])], default=0) + 1
        self._save_state(st)
        return f"L-{st['max_id']:04d}"

    # ------------------------------------------------------------------ lessons
    def _parse(self) -> _Parsed:
        p = self.root / "lessons.md"
        if not p.exists():
            return _Parsed(False, False, [], "", [], [], [])
        lessons, pinned, untagged, extra, sections = [], [], [], [], []
        section, skip_next, has_active = None, False, False
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            s = ln.strip()
            if _PIN_RE.match(s):
                section = "pinned"
                continue
            if _ACT_RE.match(s):
                section, has_active = "active", True
                continue
            if s.startswith("## "):
                section = "other"
                sections.append(ln)
                continue
            if section == "pinned":
                pinned.append(ln)
            elif section == "other":
                sections.append(ln)
            elif section == "active":
                m = _HEAD.match(ln)
                if m:
                    ev = ""
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("evidence:"):
                        ev = lines[i + 1].strip()[len("evidence:"):].strip()
                        skip_next = True
                    tags = [t.strip() for t in (m.group(4) or "").split(",") if t.strip()]
                    lessons.append(Lesson(m.group(1), m.group(5).strip(), m.group(2), int(m.group(3)), ev, tags))
                elif ln.startswith("- ") and ln[2:].strip():
                    untagged.append(ln[2:].strip())
                elif s:
                    extra.append(ln)
        return _Parsed(True, has_active, lessons, "\n".join(pinned).strip("\n"), untagged, extra,
                       sections)

    def read_lessons(self) -> tuple[list[Lesson], str]:
        st = self._state()
        p = self._parse()
        if (not p.exists or not p.has_active) and st["written"]:
            raise MemoryAnomaly(f"{self.root / 'lessons.md'} is missing or has no '{ACTIVE}' heading; "
                                "no lessons were retired (fix the file or restore the heading)")
        present = {x.id for x in p.lessons}
        gone = [i for i in st["written"] if i not in present and i not in st["retired"]]
        if gone and self.confirm_delay:      # an editor may be mid-save: confirm before retiring anything
            time.sleep(self.confirm_delay)
            p = self._parse()
            present = {x.id for x in p.lessons}
            gone = [i for i in st["written"] if i not in present and i not in st["retired"]]
        lessons, changed, notes = list(p.lessons), False, []
        for text in list(p.untagged):        # the owner stripped a lesson's header but kept its text
            for gid in list(gone):
                old = st["lessons"].get(gid)
                if old and one_line(old["text"]) == one_line(text):
                    lessons.append(Lesson(**old))
                    gone.remove(gid)
                    p.untagged.remove(text)
                    notes.append(f"{gid}: header removed by the owner; lesson kept")
                    changed = True
                    break
        for x in lessons:                    # the owner rewrote a bot lesson: it is theirs now
            old = st["lessons"].get(x.id)
            if old and one_line(old["text"]) != one_line(x.text) and x.id not in st["owner_owned"]:
                st["owner_owned"].append(x.id)
                notes.append(f"{x.id}: text edited by the owner; the Curator will not change it")
                changed = True
        max_id = max([st["max_id"], *(int(x.id[2:]) for x in lessons), *(int(i[2:]) for i in st["retired"])],
                     default=0)
        added = []
        for text in p.untagged:
            max_id += 1
            added.append(Lesson(f"L-{max_id:04d}", one_line(text), self._today(), 0, "added by you", ["owner"]))
        lessons += added
        if gone:
            st["retired"] = sorted(set(st["retired"]) | set(gone))
            notes += [f"{i}: removed from lessons.md by the owner; retired (never re-added)" for i in gone]
            changed = True
        notes += [f"{x.id}: added by you" for x in added]
        if changed or added:
            st["max_id"] = max_id
            self._save_state(st)
            self.append_changelog(self._today(), notes)
        if added or any(n.endswith("lesson kept") for n in notes):   # give the owner's lines their ids now
            self.write_lessons(lessons)
        return lessons, p.pinned

    def write_lessons(self, lessons: list[Lesson]) -> None:
        st = self._state()
        retired = set(st["retired"])
        blocked = [x.id for x in lessons if x.id in retired]
        keep = [dataclasses.replace(x, text=one_line(x.text), evidence=one_line(x.evidence),
                                    tags=[t for t in (_tag(t) for t in x.tags) if t])
                for x in lessons if x.id not in retired]
        cur = self._parse()
        out = [f"# {self.title}", "", NOTE, "", PINNED, cur.pinned, "", ACTIVE]
        for x in keep:
            tags = f", tags: {', '.join(x.tags)}" if x.tags else ""
            out.append(f"- [{x.id}] ({x.added}, n={x.n}{tags}) {x.text}")
            out.append(f"  evidence: {x.evidence}")
        if cur.extra:
            out += ["", *cur.extra]
        if cur.sections:
            out += ["", *cur.sections]
        _atomic(self.root / "lessons.md", "\n".join(out).rstrip("\n") + "\n")
        st["written"] = [x.id for x in keep]
        st["lessons"] = {**st["lessons"], **{x.id: dataclasses.asdict(x) for x in keep}}
        st["max_id"] = max([st["max_id"], *(int(x.id[2:]) for x in keep)], default=0)
        self._save_state(st)
        if blocked:
            self.append_changelog(self._today(), [f"{i}: not re-added (retired by the owner)" for i in blocked])

    # ------------------------------------------------------------------ staging / changelog / scorecard
    def append_staging(self, date: str, lines: list[str]) -> None:
        p = self.root / "staging.md"
        old = p.read_text(encoding="utf-8") if p.exists() else "# Observations (staging)\n\n"
        _atomic(p, old + "".join(f"- ({date}) {one_line(ln)}\n" for ln in lines))

    def read_staging(self, since: str | None) -> list[str]:
        p = self.root / "staging.md"
        if not p.exists():
            return []
        out = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            m = _STAGE.match(ln)
            if m and (since is None or m.group(1) >= since):
                out.append(ln)
        return out

    def clear_staging(self, upto: str) -> None:
        p = self.root / "staging.md"
        if not p.exists():
            return
        keep = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                if not ((m := _STAGE.match(ln)) and m.group(1) <= upto)]
        _atomic(p, "\n".join(keep).rstrip("\n") + "\n")

    def append_changelog(self, date: str, lines: list[str]) -> None:
        if not lines:
            return
        p = self.root / "changelog.md"
        old = p.read_text(encoding="utf-8") if p.exists() else "# Memory changelog\n\n"
        _atomic(p, old + "".join(f"- ({date}) {one_line(ln)}\n" for ln in lines))

    def write_scorecard(self, md: str) -> None:
        _atomic(self.root / "scorecard.md", md)
