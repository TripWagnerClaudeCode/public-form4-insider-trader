"""Phone-readable markdown reports (short lines, at most one table per section)."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def _when(v) -> str:
    try:
        d = datetime.fromisoformat(v) if isinstance(v, str) else v
        if d.tzinfo is None:
            d = d.replace(tzinfo=ZoneInfo("UTC"))
        return d.astimezone(NY).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return str(v) if v else "n/a"


def _cell(v) -> str:
    return str(v if v is not None else "").replace("|", "/").replace("\n", " ")[:80]


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_cell(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _cols(rows: list[dict], preferred: list[str]) -> list[str]:
    seen = [c for c in preferred if any(c in r for r in rows)]
    return seen or (list(rows[0].keys())[:4] if rows else preferred)


def _section(title: str, rows: list[dict], preferred: list[str]) -> str:
    return f"## {title} ({len(rows)})\n\n{_table(rows, _cols(rows, preferred))}\n"


def _write(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _head(title: str, run: dict) -> str:
    lines = [f"# {title}", ""]
    for key, label in (("id", "Run"), ("mode", "Mode"), ("formula_version", "Formula")):
        if run.get(key) is not None:
            lines.append(f"- {label}: {run[key]}")
    if run.get("started_at"):
        lines.append(f"- Started: {_when(run['started_at'])}")
    return "\n".join(lines) + "\n\n"


def write_scan(path: Path, run: dict, candidates: list[dict], decisions: list[dict],
               orders: list[dict], exits: list[dict], notes: list[str] | None = None) -> Path:
    text = _head("Scan report", run)
    text += (f"Candidates: {len(candidates)} | Decisions: {len(decisions)} | "
             f"Orders: {len(orders)} | Exits: {len(exits)}\n\n")
    if notes:
        text += "## Notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n\n"
    text += _section("Candidates", candidates, ["ticker", "score"])
    text += _section("Decisions", decisions, ["ticker", "action", "confidence", "rationale"])
    text += _section("Orders", orders, ["symbol", "side", "qty", "status"])
    text += _section("Exits", exits, ["symbol", "reason", "pnl"])
    return _write(path, text)


def write_daily(path: Path, run: dict, orders: list[dict], positions: list[dict],
                exits: list[dict], equity: float | None = None) -> Path:
    text = _head("Daily report", run)
    if equity is not None:
        text += f"Equity: ${equity:,.2f}\n\n"
    text += f"Orders: {len(orders)} | Exits: {len(exits)} | Open positions: {len(positions)}\n\n"
    text += _section("Orders", orders, ["symbol", "side", "qty", "status"])
    text += _section("Exits", exits, ["symbol", "reason", "pnl"])
    text += _section("Positions", positions, ["symbol", "qty", "entry_price", "pnl"])
    return _write(path, text)


def write_weekly(path: Path, stats: dict, formula_diff: str, learned: str) -> Path:
    text = "# Weekly review\n\n## Stats\n\n"
    text += "\n".join(f"- {k}: {v}" for k, v in stats.items()) or "_none_"
    text += "\n\n## Formula changes\n\n"
    text += f"```diff\n{formula_diff.strip()}\n```\n\n" if formula_diff.strip() else "_none_\n\n"
    text += f"## Learned\n\n{learned.strip() or '_nothing_'}\n"
    return _write(path, text)
