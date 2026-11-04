from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalystNote(BaseModel):
    summary: str
    positives: list[str]
    negatives: list[str]
    red_flags: list[str]


class DebateCase(BaseModel):
    side: str
    argument: str
    key_risks: list[str]


class PMOut(BaseModel):
    action: Literal["buy", "pass", "exit"]
    conviction: int = Field(ge=1, le=5)
    thesis: str
    rule_flags_addressed: str
    what_would_change_my_mind: str


class ResearchOut(BaseModel):
    change: bool
    rationale: str
    new_weights: dict[str, float]
    new_params: dict[str, float]
    what_we_learned: str


class Observation(BaseModel):
    text: str
    agents: list[str]
    symbols: list[str]
    evidence: str


class PostTradeNotes(BaseModel):
    observations: list[Observation]
    summary: str


class NewLesson(BaseModel):
    text: str
    evidence: str
    n: int = Field(description="scored outcomes supporting this lesson")
    tags: list[str]


class LessonEdit(BaseModel):
    id: str
    text: str
    evidence: str
    n: int


class LessonRetire(BaseModel):
    id: str
    reason: str


class LessonsUpdate(BaseModel):
    add: list[NewLesson]
    edit: list[LessonEdit]
    retire: list[LessonRetire]
    summary: str


class Finding(BaseModel):
    claim: str
    source_url: str
    date: str
    support: Literal["strong", "moderate", "weak"]


class DiveFindings(BaseModel):
    symbol: str
    question: str
    answer: str
    findings: list[Finding]


class BuyApproval(BaseModel):
    approve: bool
    reason: str


class EventCall(BaseModel):
    action: Literal["hold", "exit"]
    reason: str
