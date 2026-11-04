"""Read-only SEC EDGAR filings lookup for the Researcher and Event Review (spec 2026-11-04, Phase 3).

Ported from trading-bot (bot/research/edgar.py). Fetched in code (research calls have no shell) and
handed to agents as untrusted data. SEC asks automated clients to identify themselves and stay under
10 requests/second: we send the bot's EDGAR identity and throttle to 5/s.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import threading
import time
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
MIN_INTERVAL_S = 0.2
_lock = threading.Lock()
_last = [0.0]
_memo: dict[str, tuple[float, dict[str, str]]] = {}


def _get(url: str, session: httpx.Client | None, user_agent: str | None = None) -> httpx.Response:
    with _lock:
        wait = _last[0] + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
    ua = (user_agent or "").strip()
    if not ua:
        raise RuntimeError("EDGAR_IDENTITY not set (SEC requires 'Your Name your@email')")
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    if session is not None:
        r = session.get(url, headers=headers)
    else:
        r = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
    r.raise_for_status()
    return r


def _cik_map(session, cache_dir: Path | None, user_agent: str | None) -> dict[str, str]:
    key = str(cache_dir)
    hit = _memo.get(key)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    f = Path(cache_dir) / "sec_company_tickers.json" if cache_dir else None
    raw = None
    if f and f.exists() and time.time() - f.stat().st_mtime < 86400:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            raw = None
    if raw is None:
        raw = _get(TICKERS_URL, session, user_agent).json()
        if f:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(raw), encoding="utf-8")
    m = {str(v["ticker"]).upper(): f"{int(v['cik_str']):010d}" for v in raw.values()}
    _memo[key] = (time.time(), m)
    return m


def recent_filings(symbol: str, since: dt.date, forms=("8-K", "10-Q", "10-K", "6-K"),
                   session: httpx.Client | None = None, cache_dir: Path | None = None,
                   user_agent: str | None = None) -> list[dict]:
    """Filings of the given forms on or after `since`, newest first: [{form, date, url, title}]."""
    cik = _cik_map(session, cache_dir, user_agent).get(symbol.upper().replace(".", "-"))
    if cik is None:
        return []
    rec = (_get(SUBMISSIONS_URL.format(cik=cik), session, user_agent).json().get("filings") or {}).get("recent") or {}
    cols = [rec.get(k) or [] for k in ("form", "filingDate", "accessionNumber", "primaryDocument",
                                        "primaryDocDescription")]
    out = []
    for form, date, acc, doc, desc in zip(*cols):
        if form not in forms or not doc or date < since.isoformat():
            continue
        out.append({"form": form, "date": date, "title": desc or form,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"})
    return sorted(out, key=lambda x: x["date"], reverse=True)


_DROP = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def filing_text(url: str, max_chars: int = 6000, session: httpx.Client | None = None,
                user_agent: str | None = None) -> str:
    s = _DROP.sub(" ", _get(url, session, user_agent).text)
    s = html.unescape(_TAGS.sub(" ", s))
    return _WS.sub(" ", s).strip()[:max_chars]
