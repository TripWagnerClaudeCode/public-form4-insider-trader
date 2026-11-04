"""Researcher agent (spec 2026-11-04, Phase 3), ported from trading-bot: one deep dive on one ticker.

The model gets WebSearch/WebFetch only (LLM.research) plus recent SEC filings fetched in code and
wrapped as untrusted data. Every finding must cite a page the model actually fetched or a filing we
prefetched; anything else is dropped. Never raises: a failed dive returns (None, record).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from urllib.parse import urlsplit

from insiderbot.agents.llm import load_prompt
from insiderbot.agents.schemas import DiveFindings
from insiderbot.data import sec_filings

log = logging.getLogger(__name__)
QUESTION = ("Verify the key factual claims in the bull and bear cases and find any material news or "
            "filings since the insider purchases.")


def _norm(url: str) -> str:
    u = urlsplit(url.strip())
    return f"{u.netloc.lower().removeprefix('www.')}{u.path.rstrip('/')}"


def same_source(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


class Researcher:
    def __init__(self, llm, cfg, prompt_dir: str, cache_dir=None, session=None, filings_fn=None,
                 user_agent: str | None = None):
        self.llm, self.cfg, self.prompt_dir = llm, cfg, prompt_dir
        self.cache_dir, self.session, self.filings_fn, self.user_agent = cache_dir, session, filings_fn, user_agent

    def _filings(self, symbol: str) -> list[dict]:
        since = dt.date.today() - dt.timedelta(days=self.cfg.filings_days)
        if self.filings_fn is not None:
            return self.filings_fn(symbol, since)
        return sec_filings.recent_filings(symbol, since, session=self.session, cache_dir=self.cache_dir,
                                          user_agent=self.user_agent)

    def _text(self, url: str) -> str:
        return sec_filings.filing_text(url, self.cfg.max_filing_chars, session=self.session,
                                       user_agent=self.user_agent)

    def _prefetch(self, symbol: str, rec: dict) -> list[dict]:
        try:
            fl = self._filings(symbol)
        except Exception as e:  # noqa: BLE001
            rec["prefetch_error"] = repr(e)[:300]
            return []
        fl = [f for f in fl if f["form"] == "8-K"] + [f for f in fl if f["form"] != "8-K"]
        out = []
        for f in fl[:3]:
            try:
                out.append(f | {"text": self._text(f["url"])})
            except Exception as e:  # noqa: BLE001
                rec.setdefault("prefetch_error", repr(e)[:300])
        return out

    def dive(self, symbol: str, question: str, context: dict) -> tuple[DiveFindings | None, dict]:
        from insiderbot.agents.team import wrap

        rec: dict = {"symbol": symbol, "question": question, "touched": [], "prefetched_urls": [],
                     "dropped": [], "error": None}
        try:
            filings = self._prefetch(symbol, rec)
            rec["prefetched_urls"] = [f["url"] for f in filings]
            data = json.dumps({"context": context, "sec_filings": filings}, default=str, indent=1)
            user = (f"Ticker: {symbol}\nQuestion to resolve: {question}\n\n"
                    f"Context and recent SEC filings (fetched for you):\n{wrap(data)}")
            out, touched = self.llm.research(
                model=self.cfg.model, effort=self.cfg.effort, system=load_prompt("researcher", self.prompt_dir),
                user=user, schema=DiveFindings, max_turns=self.cfg.max_turns, timeout_s=self.cfg.timeout_s)
            rec["touched"] = touched
            allowed = [t["url"] for t in touched if t.get("tool") == "WebFetch"] + rec["prefetched_urls"]
            keep = []
            for f in out.findings:
                if any(same_source(f.source_url, a) for a in allowed):
                    keep.append(f)
                else:
                    rec["dropped"].append(f.source_url)
            return out.model_copy(update={"symbol": symbol, "question": question, "findings": keep}), rec
        except Exception as e:  # noqa: BLE001 - a failed dive never blocks a review
            log.warning("deep dive %s failed: %s", symbol, e)
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            return None, rec
