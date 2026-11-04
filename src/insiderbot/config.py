"""Typed configuration: config/settings.yaml + secrets from .env."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]


class Schedule(BaseModel):
    tz: str = "America/New_York"
    scans: list[str] = ["10:00", "15:00"]
    reconcile: str = "16:15"
    review: str = "Sun 18:00"
    max_late_minutes: int = 60
    poll_window: tuple[str, str] = ("09:45", "15:45")   # ET; no polls in the first/last 15 min of the session


class Filters(BaseModel):
    min_price: float = 5
    min_adv_usd: float = 2_000_000
    min_mcap_usd: float = 300_000_000
    exchanges: list[str] = ["NYSE", "NASDAQ"]


class Review(BaseModel):
    max_reviews_per_scan: int = 6
    recent_review_days: int = 5


class Agents(BaseModel):
    # "api" = Anthropic API key (ANTHROPIC_API_KEY); "claude-cli" = headless `claude -p` on the
    # Claude subscription the local claude CLI is logged in to (no key, no $ spend).
    backend: Literal["api", "claude-cli"] = "api"
    claude_cli: str = "~/.local/bin/claude"
    analyst_model: str = "claude-sonnet-5-5"
    analyst_effort: str = "low"
    debate_model: str = "claude-sonnet-5-5"
    debate_effort: str = "medium"
    pm_model: str = "claude-opus-5-5"
    pm_effort: str = "high"
    research_model: str = "claude-opus-5-5"
    research_effort: str = "high"
    max_run_output_tokens: int = 120000
    prompt_dir: str = "prompts"  # under src/insiderbot/agents/; the frozen control uses prompts_v1
    max_monthly_usd: float = 40


class Risk(BaseModel):
    starting_equity: float = 100000
    max_position_weight: float = 0.08
    max_positions: int = 12
    max_gross_exposure: float = 0.90
    max_new_buys_per_scan: int = 3
    max_sector_weight: float = 0.30
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    min_hold_days_for_agent_exit: int = 5
    kill_switch_file: str = "KILL_SWITCH"


class Bounds(BaseModel):
    """Hard limits the bot can never edit."""

    weight: tuple[float, float] = (0, 1)
    max_rel_change: float = 0.30
    min_abs_change: float = 0.05
    review_threshold: tuple[float, float] = (30, 90)
    buy_threshold: tuple[float, float] = (40, 95)
    stop_loss_pct: tuple[float, float] = (0.06, 0.20)
    time_stop_days: tuple[int, int] = (10, 60)
    take_profit_pct: tuple[float, float] = (0.10, 0.60)
    trailing_pct: tuple[float, float] = (0.05, 0.20)
    window_days: tuple[int, int] = (7, 30)
    min_matured_for_proposal: int = 40
    min_matured_for_promotion: int = 30
    promotion_horizon_days: int = 10
    shadow_max_weeks: int = 10


class Paths(BaseModel):
    journal_db: str = "data/journal.db"
    reports_dir: str = "reports"  # relative to the repo, or absolute / ~-path (e.g. a synced folder)
    formula_dir: str = "formula"
    cache_dir: str = "data/cache"
    lock_file: str = "data/run.lock"


class Notify(BaseModel):
    ntfy: bool = True


class Secrets(BaseModel):
    alpaca_key: str | None = None
    alpaca_secret: str | None = None
    anthropic_key: str | None = None
    edgar_identity: str | None = None


def _clean(v: str | None) -> str | None:
    v = (v or "").replace("\r", "").strip()
    return v or None


class Memory(BaseModel):
    """Phase 2 memory (on/off: features.memory). Empty dir disables memory even with the flag on."""
    dir: str = "memory"   # relative = inside the repo; the owner can read and edit lessons.md there
    max_block_chars: int = 4000
    min_lesson_n: int = 20
    max_lessons: int = 15
    reviewer_model: str = "claude-sonnet-5-5"
    reviewer_effort: str = "medium"
    curator_model: str = "claude-opus-5-5"
    curator_effort: str = "high"


class Research(BaseModel):
    """Phase 3 deep dives (on/off: features.research): one per full review, before the PM."""
    max_dives_per_day: int = 6
    model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    max_turns: int = 12
    timeout_s: int = 300
    filings_days: int = 30
    max_filing_chars: int = 6000


class Reviewers(BaseModel):
    """Phase 3 second vetoer (on/off: features.reviewers): a Risk Reviewer on every PM buy."""
    risk_model: str = "claude-opus-5-5"
    risk_effort: str = "high"


class Events(BaseModel):
    """Phase 3 intraday Event Review on held names (on/off: features.event_review), in the poll slot."""
    k_sigma: float = 2.5                 # move trigger: |mid / prior close - 1| >= k x 20-day daily sigma
    max_reviews_per_day: int = 6
    max_reviews_per_poll: int = 2        # bounds how long one 5-minute poll can run
    analyst_model: str = "claude-sonnet-5-5"
    analyst_effort: str = "medium"
    reviewer_model: str = "claude-opus-5-5"
    reviewer_effort: str = "high"
    max_turns: int = 8
    timeout_s: int = 240


class Features(BaseModel):
    """Upgrade features (spec 2026-11-04). Every flag defaults to False; the frozen control sets none."""

    memory: bool = False  # Phase 2: scorecards, lessons, memory block in agent prompts
    poll: bool = False   # scan --slot poll: EDGAR live feed every 5 min in market hours (same rules, sooner)
    research: bool = False  # Phase 3: one cited deep dive (web + SEC) per full review, before the PM
    reviewers: bool = False  # Phase 3: Risk Reviewer as a second vetoer on every PM buy
    event_review: bool = False  # Phase 3: intraday Event Review on held names (both agents must say exit)


class Settings(BaseModel):
    mode: Literal["dry-run", "paper"] = "paper"
    schedule: Schedule = Schedule()
    filters: Filters = Filters()
    review: Review = Review()
    agents: Agents = Agents()
    risk: Risk = Risk()
    bounds: Bounds = Bounds()
    paths: Paths = Paths()
    notify: Notify = Notify()
    features: Features = Features()
    memory: Memory = Memory()
    research: Research = Research()
    reviewers: Reviewers = Reviewers()
    events: Events = Events()

    def secrets(self, env_path: Path | None = None) -> Secrets:
        vals = dotenv_values(env_path or ROOT / ".env")
        return Secrets(
            alpaca_key=_clean(vals.get("ALPACA_API_KEY")),
            alpaca_secret=_clean(vals.get("ALPACA_SECRET_KEY")),
            anthropic_key=_clean(vals.get("ANTHROPIC_API_KEY")),
            edgar_identity=_clean(vals.get("EDGAR_IDENTITY")),
        )


def load(path: Path = ROOT / "config/settings.yaml") -> Settings:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Settings(**data)
