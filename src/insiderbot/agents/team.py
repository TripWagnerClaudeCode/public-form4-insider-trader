"""Analysts -> bull/bear debate -> portfolio manager. Agents can only veto or exit, never originate."""
from __future__ import annotations

import dataclasses
import json
import logging
import re

import anthropic

from insiderbot.agents.llm import (
    LLM,
    AgentRefused,
    AgentUnavailable,
    BudgetExceeded,
    load_prompt,
)
from insiderbot.agents.schemas import AnalystNote, BuyApproval, DebateCase, EventCall, PMOut, ResearchOut
from insiderbot.models import AgentDecision, InsiderTxn, IssuerFeatures

log = logging.getLogger(__name__)

ANALYSTS = ("filing", "fundamentals", "news", "technical")
_OPEN, _CLOSE = "<untrusted_data>", "</untrusted_data>"
_TAG = re.compile(r"<\s*(/?)\s*untrusted_data\s*>", re.IGNORECASE)
_ERRORS = (BudgetExceeded, AgentRefused, AgentUnavailable, anthropic.APIError, RuntimeError)


def wrap(text) -> str:
    """Wrap third-party text so the model treats it as data. Embedded tags are defanged so the
    text cannot close the wrapper early (any case or spacing)."""
    s = text if isinstance(text, str) else json.dumps(text, default=str, indent=1)
    s = _TAG.sub(lambda m: f"[{m.group(1)}untrusted_data]", s)
    return f"{_OPEN}\n{s}\n{_CLOSE}"


def _dump(obj) -> str:
    return json.dumps(obj, default=str, indent=1)


def _txn_rows(txns: list[InsiderTxn]) -> list[dict]:
    return [{"insider": t.insider_name, "title": t.officer_title, "officer": t.is_officer,
             "director": t.is_director, "ten_pct": t.is_ten_pct, "date": str(t.tx_date),
             "shares": t.shares, "price": t.price, "value": t.value,
             "owned_after": t.owned_after, "10b5-1": t.is_10b5_1} for t in txns]


class AgentTeam:
    def __init__(self, llm: LLM, cfg):
        self.llm = llm
        self.cfg = cfg

    def _prompt(self, name: str) -> str:
        return load_prompt(name, self.cfg.agents.prompt_dir)

    @staticmethod
    def _mem(memory: dict[str, str] | None, agent: str, system: str, user: str) -> tuple[str, str]:
        """Prepend the agent's memory block (features.memory); None leaves prompts exactly as before."""
        if not memory or agent not in memory:
            return system, user
        from insiderbot.memory.block import MEMORY_SENTENCE

        return f"{system}\n\n{MEMORY_SENTENCE}", f"{memory[agent]}\n\n{user}"

    def _facts(self, f: IssuerFeatures) -> str:
        d = dataclasses.asdict(f)
        return "Computed facts (from our own data):\n" + _dump(d)

    def _analyst_inputs(self, f, txns, context) -> dict[str, str]:
        facts = self._facts(f)
        return {
            "filing": f"{facts}\n\nForm 4 purchases:\n{wrap(_txn_rows(txns))}",
            "fundamentals": f"{facts}\n\nCompany data:\n{wrap(context.get('fundamentals', 'none supplied'))}",
            "news": f"{facts}\n\nRecent news:\n{wrap(context.get('news', 'none supplied'))}",
            "technical": f"{facts}\n\nPrice summary:\n{wrap(context.get('bars_summary', 'none supplied'))}",
        }

    def review(self, f: IssuerFeatures, score: float, flags: list[str], txns: list[InsiderTxn],
               context: dict, held: bool, memory: dict[str, str] | None = None,
               researcher=None, reviewers: bool = False, portfolio: dict | None = None) -> AgentDecision:
        a = self.cfg.agents
        step = "start"
        try:
            notes: dict[str, dict] = {}
            for name, user in self._analyst_inputs(f, txns, context).items():
                step = f"analyst:{name}"
                sys_, usr = self._mem(memory, name, self._prompt(name), user)
                notes[name] = self.llm.structured(
                    model=a.analyst_model, effort=a.analyst_effort, system=sys_,
                    user=usr, schema=AnalystNote).model_dump()
            debate_user = f"{self._facts(f)}\n\nAnalyst notes:\n{wrap(notes)}"
            debate: dict[str, dict] = {}
            for side in ("bull", "bear"):
                step = f"debate:{side}"
                sys_, usr = self._mem(memory, side, self._prompt(side), debate_user)
                debate[side] = self.llm.structured(
                    model=a.debate_model, effort=a.debate_effort, system=sys_,
                    user=usr, schema=DebateCase).model_dump()
            research: dict = {}
            if researcher is not None:   # features.research: one cited deep dive before the PM
                from insiderbot.agents.deep_research import QUESTION

                step = "research"
                found, rlog = researcher.dive(f.ticker, QUESTION, {
                    "form4_purchases": _txn_rows(txns), "news": context.get("news"),
                    "company": context.get("fundamentals"), "bull": debate["bull"], "bear": debate["bear"]})
                research["research_log"] = rlog
                if found is not None:
                    research["research"] = found.model_dump()
            pm_user = (
                f"Formula score: {score}\n"
                f"Rule flags: {', '.join(flags) if flags else 'none'}\n"
                f"Currently held: {'yes' if held else 'no'}\n"
                f"Ticker: {f.ticker}\n\n"
                f"{self._facts(f)}\n\nAnalyst notes and debate:\n{wrap({'analysts': notes, 'debate': debate})}"
            )
            if research.get("research"):
                pm_user += f"\n\nDeep-dive findings:\n{wrap(research['research'])}"
            step = "pm"
            sys_, pm_user = self._mem(memory, "pm", self._prompt("pm"), pm_user)
            pm = self.llm.structured(
                model=a.pm_model, effort=a.pm_effort, system=sys_,
                user=pm_user, schema=PMOut)
        except _ERRORS as e:
            return self._fail(f, step, e)
        except Exception as e:  # last resort: malformed output etc. must never crash a scan
            return self._fail(f, step, e)
        action = pm.action
        if action == "buy" and held:
            action = "pass"
        elif action == "exit" and not held:
            action = "pass"
        if action == "buy" and reviewers:   # features.reviewers: the buy needs a second, independent yes
            ok, why, err = self.risk_review(f, pm, {"analysts": notes, "debate": debate, **research},
                                            portfolio or {}, memory)
            research["risk_review"] = {"approve": ok, "reason": why} | ({"error": True} if err else {})
            if not ok:
                action = "pass"
        return AgentDecision(
            ticker=f.ticker, action=action, conviction=pm.conviction, thesis=pm.thesis,
            change_mind=pm.what_would_change_my_mind,
            reports={"analysts": notes, "debate": debate, "pm": pm.model_dump(),
                     "pm_raw_action": pm.action, "held": held, **research})

    def risk_review(self, f: IssuerFeatures, pm: PMOut, evidence: dict, portfolio: dict,
                    memory: dict[str, str] | None = None) -> tuple[bool, str, bool]:
        """(approve, reason, failed). It can only veto; any failure is a veto (fail closed)."""
        r = self.cfg.reviewers
        user = (f"Ticker: {f.ticker}\nThe PM confirmed this rule-selected buy: {pm.thesis}\n"
                f"Conviction {pm.conviction}/5. What would change the PM's mind: {pm.what_would_change_my_mind}\n"
                f"Portfolio now: {_dump(portfolio)}\n\n{self._facts(f)}\n\n"
                f"Analyst notes, debate and research:\n{wrap(evidence)}")
        try:
            sys_, user = self._mem(memory, "risk_reviewer", self._prompt("risk_reviewer"), user)
            out = self.llm.structured(model=r.risk_model, effort=r.risk_effort, system=sys_, user=user,
                                      schema=BuyApproval)
            return out.approve, out.reason, False
        except Exception as e:  # noqa: BLE001
            log.warning("risk review failed for %s: %s", f.ticker, e)
            return False, f"error: {type(e).__name__}", True

    def event_review(self, ticker: str, trigger: str, kind: str, position: dict, filing_text: str | None,
                     memory: dict[str, str] | None = None):
        """(analyst call, analyst record, reviewer call or None). The reviewer runs only on an exit proposal.
        Errors propagate: the caller journals them and holds."""
        e = self.cfg.events
        evidence = {"trigger": trigger, "position": position}
        if filing_text:
            evidence["filing_text"] = filing_text
        user = f"Held position: {ticker}. Trigger: {kind}.\n\nEvidence:\n{wrap(evidence)}"
        sys_, user = self._mem(memory, "event_analyst", self._prompt("event_analyst"), user)
        a, touched = self.llm.research(model=e.analyst_model, effort=e.analyst_effort, system=sys_, user=user,
                                       schema=EventCall, max_turns=e.max_turns, timeout_s=e.timeout_s)
        arec = {"call": a.model_dump(), "touched": touched}
        if a.action != "exit":
            return a, arec, None
        ruser = (f"Held position: {ticker}. Trigger: {kind}. The Event analyst proposes exit.\n\n"
                 f"Evidence and the analyst's reasoning:\n{wrap(evidence | {'analyst_reason': a.reason})}")
        rsys, ruser = self._mem(memory, "event_reviewer", self._prompt("event_reviewer"), ruser)
        b = self.llm.structured(model=e.reviewer_model, effort=e.reviewer_effort, system=rsys, user=ruser,
                                schema=EventCall)
        return a, arec, b

    def exit_review(self, f: IssuerFeatures, score: float, flags: list[str],
                    txns: list[InsiderTxn], memory: dict[str, str] | None = None) -> AgentDecision:
        """Cheap exit opinion for a HELD name with new insider activity: PM only, no analysts or
        debate. Returns action "exit" or "hold" ("hold" so research stats, which compare
        buy vs pass, never count it). Any error -> "hold" (no exit on a failed call)."""
        a = self.cfg.agents
        user = (
            f"Formula score: {score}\n"
            f"Rule flags: {', '.join(flags) if flags else 'none'}\n"
            "Currently held: yes\n"
            f"Ticker: {f.ticker}\n"
            "Exit check only: there are no analyst notes for this review. Answer exit only for a "
            "concrete reason in the new activity; otherwise pass (keep holding).\n\n"
            f"{self._facts(f)}\n\nNew Form 4 activity:\n{wrap(_txn_rows(txns))}"
        )
        try:
            sys_, user = self._mem(memory, "pm", self._prompt("pm"), user)
            pm = self.llm.structured(model=a.pm_model, effort=a.pm_effort, system=sys_,
                                     user=user, schema=PMOut)
        except Exception as e:  # includes _ERRORS
            d = self._fail(f, "pm-exit", e)
            return dataclasses.replace(d, action="hold")
        return AgentDecision(
            ticker=f.ticker, action="exit" if pm.action == "exit" else "hold",
            conviction=pm.conviction, thesis=pm.thesis, change_mind=pm.what_would_change_my_mind,
            reports={"pm": pm.model_dump(), "pm_raw_action": pm.action, "held": True,
                     "kind": "exit_check"})

    @staticmethod
    def _fail(f: IssuerFeatures, step: str, e: Exception) -> AgentDecision:
        log.warning("agent team failed for %s at %s: %s", f.ticker, step, e)
        return AgentDecision(ticker=f.ticker, action="pass", conviction=1,
                             thesis=f"agent error: {type(e).__name__}", change_mind="",
                             reports={"error": f"{step}: {type(e).__name__}"})

    def research(self, stats: dict, formula_yaml: str) -> ResearchOut:
        a = self.cfg.agents
        none = ResearchOut(change=False, rationale="agent unavailable", new_weights={},
                           new_params={}, what_we_learned="")
        user = f"Outcome statistics:\n{_dump(stats)}\n\nCurrent formula YAML (with bounds):\n{formula_yaml}"
        try:
            return self.llm.structured(model=a.research_model, effort=a.research_effort,
                                       system=self._prompt("research"), user=user, schema=ResearchOut)
        except Exception as e:  # includes _ERRORS; research must never crash the review job
            log.warning("research failed: %s", e)
            none.rationale = f"agent error: {type(e).__name__}"
            return none
