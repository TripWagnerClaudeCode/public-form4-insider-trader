"""Daily Post-Trade Reviewer (Sonnet): turns newly scored candidates into dated observations in
staging.md. Observations are context only; the weekly Curator decides what becomes a lesson."""
from __future__ import annotations

import json

from insiderbot.agents.llm import load_prompt
from insiderbot.agents.schemas import PostTradeNotes
from insiderbot.agents.team import wrap

MAX_OBSERVATIONS = 5


def run_post_trade(llm, s, journal, files, newly_scored: list[dict], cards: dict, today: str) -> int:
    data = json.dumps({"newly_scored": newly_scored, "scorecards": cards}, default=str, sort_keys=True)
    user = (f"Today is {today}. These insider-buy candidates just reached their 20-trading-day horizon. Each row "
            f"has the realized excess return vs SPY and the portfolio manager's decision (buy or pass).\n{wrap(data)}")
    out = llm.structured(model=s.memory.reviewer_model, effort=s.memory.reviewer_effort,
                         system=load_prompt("post_trade", s.agents.prompt_dir), user=user, schema=PostTradeNotes)
    obs = out.observations[:MAX_OBSERVATIONS]
    files.append_staging(today, [f"{o.text} [agents: {', '.join(o.agents)}] [symbols: {', '.join(o.symbols)}] "
                                 f"evidence: {o.evidence}" for o in obs])
    return len(obs)
