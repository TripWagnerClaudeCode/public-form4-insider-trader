"""Versioned scoring formula (YAML) and the fixed feature transforms."""
from __future__ import annotations

import math
import os
import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from insiderbot.config import ROOT, Bounds
from insiderbot.models import IssuerFeatures

FORMULA_DIR = ROOT / "formula"
PLAN_KEY = "plan_penalty"
FEATURE_KEYS = ("cluster", "dollars", "ceo_cfo", "chair", "directors", "size", "first_buy",
                "dip", "momentum_neg", "small_cap", PLAN_KEY)
FLAG_THRESHOLD = 0.66
MAX_TIER_MULT = 2.0


class Formula(BaseModel):
    version: str
    parent: str | None = None
    author: Literal["human", "review-agent"]
    created: date
    rationale: str
    evidence: dict = {}
    weights: dict[str, float]
    review_threshold: float
    buy_threshold: float
    tiers: list[tuple[float, float]]
    stop_loss_pct: float
    time_stop_days: int
    take_profit_pct: float
    trailing_pct: float
    window_days: int


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def transforms(f: IssuerFeatures) -> dict[str, float]:
    """Map features to [0,1]. Missing data gives the neutral/worst value (0)."""
    total = f.total_value
    dollars = _clip(math.log10(total / 1e4) / 3) if total and total > 0 else 0.0
    return {
        "cluster": min(max(f.n_buyers - 1, 0), 3) / 3,
        "dollars": dollars,
        "ceo_cfo": 1.0 if (f.has_ceo or f.has_cfo) else 0.0,
        "chair": 1.0 if f.has_chair else 0.0,
        "directors": min(max(f.n_directors, 0), 3) / 3,
        "size": _clip(f.size_vs_holdings / 0.5),
        "first_buy": _clip(f.first_buy_months / 24),
        "dip": _clip(f.drawdown_52w / 0.5) if f.drawdown_52w is not None else 0.0,
        "momentum_neg": _clip(-f.ret_3m / 0.3) if f.ret_3m is not None else 0.0,
        PLAN_KEY: _clip(f.frac_10b5_1),
        "small_cap": 1.0 if (f.mcap_usd is not None and f.mcap_usd < 2e9) else 0.0,
    }


def score(f: IssuerFeatures, fm: Formula) -> tuple[float, list[str]]:
    x = transforms(f)
    w = fm.weights
    denom = sum(v for k, v in w.items() if k != PLAN_KEY and v > 0)
    if denom <= 0:
        return 0.0, []
    num = sum(w.get(k, 0.0) * x[k] for k in x if k != PLAN_KEY)
    num -= w.get(PLAN_KEY, 0.0) * x[PLAN_KEY]
    s = _clip(100 * num / denom, 0.0, 100.0)
    flags = [f"+{k}" for k in x if k != PLAN_KEY and x[k] >= FLAG_THRESHOLD]
    if x[PLAN_KEY] > 0:
        flags.append("-10b5-1")
    return s, flags


def tier_weight(score_: float, fm: Formula) -> float:
    """Multiplier of the highest tier whose min_score is met; 0 below all tiers."""
    best, best_min = 0.0, -math.inf
    for lo, mult in fm.tiers:
        if score_ >= lo and lo > best_min:
            best_min, best = lo, mult
    return best


# --- persistence -----------------------------------------------------------

def _dir(directory: Path | None) -> Path:
    return Path(directory) if directory is not None else FORMULA_DIR


def load(version: str, directory: Path | None = None) -> Formula:
    if not re.fullmatch(r"v\d{3,}", version):
        raise ValueError(f"bad formula version: {version!r}")
    data = yaml.safe_load((_dir(directory) / f"{version}.yaml").read_text())
    return Formula(**data)


def active(directory: Path | None = None) -> Formula:
    return load((_dir(directory) / "ACTIVE").read_text().strip(), directory)


def shadow(directory: Path | None = None) -> Formula | None:
    p = _dir(directory) / "SHADOW"
    if not p.exists():
        return None
    v = p.read_text().strip()
    return load(v, directory) if v else None


def _next_version(d: Path) -> str:
    nums = [int(m.group(1)) for p in d.glob("v*.yaml") if (m := re.fullmatch(r"v(\d+)", p.stem))]
    return f"v{(max(nums) if nums else 0) + 1:03d}"


def save(fm: Formula, directory: Path | None = None) -> Path:
    """Write fm as the next vNNN (the version field is overwritten). Returns the path.
    Never overwrites: files are opened exclusively and the number retried on collision."""
    d = _dir(directory)
    d.mkdir(parents=True, exist_ok=True)
    n = int(_next_version(d)[1:])
    while True:
        version = f"v{n:03d}"
        path = d / f"{version}.yaml"
        out = fm.model_copy(update={"version": version})
        try:
            fh = path.open("x", newline="\n", encoding="utf-8")
        except FileExistsError:
            n += 1
            continue
        with fh:
            yaml.safe_dump(out.model_dump(mode="json"), fh, sort_keys=False,
                           allow_unicode=True, width=100)
        return path


def _write_pointer(p: Path, version: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", newline="\n", encoding="utf-8") as fh:
        fh.write(version + "\n")
    os.replace(tmp, p)


def set_active(version: str, directory: Path | None = None) -> None:
    d = _dir(directory)
    load(version, d)  # must exist and parse
    _write_pointer(d / "ACTIVE", version)


def set_shadow(version: str | None, directory: Path | None = None) -> None:
    d = _dir(directory)
    if version is None:
        (d / "SHADOW").unlink(missing_ok=True)
        return
    load(version, d)
    _write_pointer(d / "SHADOW", version)


# --- validation ------------------------------------------------------------

def _too_big(new: float, old: float, bounds: Bounds) -> bool:
    return abs(new - old) > max(bounds.max_rel_change * abs(old), bounds.min_abs_change) + 1e-12


def validate(new: Formula, old: Formula, bounds: Bounds) -> list[str]:
    errs: list[str] = []
    known = set(FEATURE_KEYS)
    for k in sorted(set(new.weights) - known):
        errs.append(f"unknown weight key: {k}")
    for k in sorted(known - set(new.weights)):
        errs.append(f"missing weight key: {k}")
    lo, hi = bounds.weight
    for k, v in new.weights.items():
        if k not in known:
            continue
        if not (lo <= v <= hi):
            errs.append(f"weight {k}={v} outside [{lo}, {hi}]")
        if k in old.weights and _too_big(v, old.weights[k], bounds):
            errs.append(f"weight {k} change {old.weights[k]} -> {v} too large")
    for name in ("review_threshold", "buy_threshold", "stop_loss_pct", "time_stop_days",
                 "take_profit_pct", "trailing_pct", "window_days"):
        v, o = getattr(new, name), getattr(old, name)
        blo, bhi = getattr(bounds, name)
        if not (blo <= v <= bhi):
            errs.append(f"{name}={v} outside [{blo}, {bhi}]")
        if _too_big(v, o, bounds):
            errs.append(f"{name} change {o} -> {v} too large")
    if new.review_threshold > new.buy_threshold:
        errs.append("review_threshold above buy_threshold")
    mins = [t[0] for t in new.tiers]
    if mins != sorted(mins) or len(set(mins)) != len(mins):
        errs.append("tiers not strictly ascending by min score")
    if new.tiers and new.tiers[0][0] > new.buy_threshold:
        errs.append("tiers[0] min score above buy_threshold")
    for i, (_, m) in enumerate(new.tiers):
        if m > MAX_TIER_MULT:
            errs.append(f"tiers[{i}] multiplier {m} above {MAX_TIER_MULT}")
    if len(new.tiers) != len(old.tiers):
        errs.append("tiers: length changed")
    else:
        for i, ((nlo, nm), (olo, om)) in enumerate(zip(new.tiers, old.tiers)):
            if _too_big(nlo, olo, bounds) or _too_big(nm, om, bounds):
                errs.append(f"tiers[{i}] change {(olo, om)} -> {(nlo, nm)} too large")
            if nm < 0:
                errs.append(f"tiers[{i}] multiplier negative")
    return errs
