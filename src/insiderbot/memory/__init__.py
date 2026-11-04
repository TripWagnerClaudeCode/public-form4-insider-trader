"""Phase 2 memory: scorecards, lessons files, memory block, reviewer, curator."""
from __future__ import annotations

from pathlib import Path


def memory_root(s) -> Path:
    """The memory folder for these settings (the one place it is resolved, so tests can redirect it)."""
    p = Path(s.memory.dir).expanduser()
    return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p   # relative = inside the repo
