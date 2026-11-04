"""SEC EDGAR Form 4 fetch and parse (the bot's only source of insider trades)."""
from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from insiderbot.models import InsiderTxn

log = logging.getLogger(__name__)

ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])(LLC|LP|L\.P\.|FUND|PARTNERS|CAPITAL|HOLDINGS|TRUST|INC|CORP|MANAGEMENT|ADVISORS)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")
MIN_INTERVAL = 0.3  # seconds per filing (~1 request: header and xml share one cached sgml download)
NY = ZoneInfo("America/New_York")


class EdgarError(Exception):
    pass


def _text(el: ET.Element | None, path: str) -> str:
    if el is None:
        return ""
    found = el.find(path)
    return (found.text or "").strip() if found is not None and found.text else ""


def _num(el: ET.Element | None, path: str) -> float:
    try:
        return float(_text(el, path))
    except ValueError:
        return 0.0


def _flag(s: str) -> bool:
    return s.strip().lower() in ("1", "true")


def parse_form4_xml(xml: str, accession: str, accepted_at: datetime,
                    stats: dict | None = None) -> list[InsiderTxn]:
    """stats (optional) counts drops: stats["bad_ticker"] for symbols failing TICKER_RE."""
    if re.search(r"<!(DOCTYPE|ENTITY)", xml, re.IGNORECASE):
        log.warning("rejecting %s: DOCTYPE/ENTITY declaration in XML", accession)
        return []
    root = ET.fromstring(xml.strip())
    if _text(root, "documentType") == "4/A":
        return []
    ticker = _text(root, "issuer/issuerTradingSymbol").strip().upper()
    if not ticker:
        return []
    if not TICKER_RE.match(ticker):
        log.warning("dropping %s: malformed ticker %r", accession, ticker[:40])
        if stats is not None:
            stats["bad_ticker"] = stats.get("bad_ticker", 0) + 1
        return []
    issuer_cik = _text(root, "issuer/issuerCik")
    issuer_name = _text(root, "issuer/issuerName")
    doc_10b5 = _flag(_text(root, "aff10b5One"))
    footnotes = {
        fn.get("id"): "".join(fn.itertext()) for fn in root.findall("footnotes/footnote")
    }

    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        name = _text(ro, "reportingOwnerId/rptOwnerName")
        is_off = _flag(_text(rel, "isOfficer"))
        is_dir = _flag(_text(rel, "isDirector"))
        owners.append(
            dict(
                cik=_text(ro, "reportingOwnerId/rptOwnerCik"),
                name=name,
                is_officer=is_off,
                is_director=is_dir,
                is_ten_pct=_flag(_text(rel, "isTenPercentOwner")),
                is_entity=not is_off and not is_dir and bool(ENTITY_RE.search(name)),
                title=_text(rel, "officerTitle"),
            )
        )

    out: list[InsiderTxn] = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(tx, "transactionCoding/transactionCode").upper()
        if code not in ("P", "S"):
            continue
        shares = _num(tx, "transactionAmounts/transactionShares/value")
        price = _num(tx, "transactionAmounts/transactionPricePerShare/value")
        if price <= 0:
            continue
        try:
            tx_date = date.fromisoformat(_text(tx, "transactionDate/value")[:10])
        except ValueError:
            continue
        fn_text = " ".join(
            footnotes.get(f.get("id"), "") for f in tx.iter("footnoteId")
        )
        is_10b5 = doc_10b5 or "10b5-1" in fn_text.lower()
        owned = _num(tx, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        for o in owners:
            out.append(
                InsiderTxn(
                    accession=accession,
                    accepted_at=accepted_at,
                    tx_date=tx_date,
                    ticker=ticker,
                    issuer_cik=issuer_cik,
                    issuer_name=issuer_name,
                    insider_cik=o["cik"],
                    insider_name=o["name"],
                    is_officer=o["is_officer"],
                    is_director=o["is_director"],
                    is_ten_pct=o["is_ten_pct"],
                    is_entity=o["is_entity"],
                    officer_title=o["title"],
                    code=code,
                    shares=shares,
                    price=price,
                    value=shares * price,
                    owned_after=owned,
                    is_10b5_1=is_10b5,
                )
            )
    return out


def dedupe_cofilers(txns: list[InsiderTxn]) -> list[InsiderTxn]:
    """Collapse identical rows reported by several owners of one accession.

    Individuals are preferred over entities; otherwise the first row wins.
    """
    chosen: dict[tuple, InsiderTxn] = {}
    order: list[tuple] = []
    for t in txns:
        key = (t.accession, t.ticker, t.tx_date, t.code, t.shares, t.price)
        if key not in chosen:
            chosen[key] = t
            order.append(key)
        elif chosen[key].is_entity and not t.is_entity:
            chosen[key] = t
    return [chosen[k] for k in order]


class EdgarClient:
    def __init__(self, identity: str, cache_dir: Path):
        if not identity or not identity.strip():
            raise EdgarError("EDGAR_IDENTITY not set")
        self.identity = identity.strip()
        self.cache_dir = Path(cache_dir)
        self._warned_fallback = False
        self.last_errors = 0  # filings skipped on error in the last fetch (data freshness)
        self.last_attempted = 0  # filings in the window (or errored) in the last fetch
        self.last_dropped = 0  # filings dropped for a malformed ticker in the last fetch
        self.last_reached_since = True  # fetch_current: False when the ~24h feed ran out before `since`

    def _accepted_at(self, filing) -> datetime:
        try:
            dt = filing.header.acceptance_datetime
        except Exception:  # noqa: BLE001 - header fetch failure falls back
            dt = None
        if dt is not None:
            # edgartools returns naive US Eastern time.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=NY)
            return dt.astimezone(timezone.utc)
        if not self._warned_fallback:
            log.warning("no acceptance time from filing header; using filing date 21:30 ET")
            self._warned_fallback = True
        fd = filing.filing_date
        if isinstance(fd, str):
            fd = date.fromisoformat(fd)
        return datetime.combine(fd, dtime(21, 30), tzinfo=NY).astimezone(timezone.utc)

    def _start(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(self.cache_dir))
        import edgar

        edgar.set_identity(self.identity)
        self.last_errors = 0
        self.last_attempted = 0
        self.last_dropped = 0

    def _ingest(self, filing, since: datetime, until: datetime, txns: list[InsiderTxn], drops: dict) -> datetime | None:
        """Parse one Form 4 into txns if accepted in (since, until]. Returns its acceptance time
        (None for non-Form-4 entries or an error), so a newest-first feed can stop paging."""
        if str(filing.form).strip() != "4":
            return None
        started = time.monotonic()
        counted = False
        accepted = None
        try:
            accepted = self._accepted_at(filing)
            if not (since < accepted <= until):
                return accepted          # out of window: no XML download, no rate-limit pause
            self.last_attempted += 1
            counted = True
            xml = filing.xml()
            if xml:
                txns.extend(parse_form4_xml(xml, filing.accession_no, accepted, drops))
            self.last_dropped = drops.get("bad_ticker", 0)
        except Exception as exc:  # noqa: BLE001 - one bad filing must not sink the run
            log.warning("skipping %s: %s", getattr(filing, "accession_no", "?"), exc)
            self.last_errors += 1
            self.last_attempted += 0 if counted else 1
        wait = MIN_INTERVAL - (time.monotonic() - started)
        if wait > 0:
            time.sleep(wait)
        return accepted

    def fetch(self, since: datetime, until: datetime) -> list[InsiderTxn]:
        """Form 4s accepted in (since, until] from EDGAR's daily filing indexes (complete, but a
        day's index is only published after that day: use fetch_current for same-day filings)."""
        self._start()
        import edgar

        d0 = (since.astimezone(timezone.utc) - timedelta(days=1)).date()
        d1 = until.astimezone(timezone.utc).date()
        drops: dict = {}
        filings = edgar.get_filings(form="4", filing_date=f"{d0:%Y-%m-%d}:{d1:%Y-%m-%d}")
        txns: list[InsiderTxn] = []
        if filings is None:
            return txns
        for filing in filings:
            self._ingest(filing, since, until, txns, drops)
        return dedupe_cofilers(txns)

    def fetch_current(self, since: datetime, until: datetime) -> list[InsiderTxn]:
        """Form 4s accepted in (since, until] from EDGAR's live current-filings feed (updated every
        few minutes, newest first, about the last 24 hours). Pages until a filing older than since."""
        self._start()
        import edgar

        drops: dict = {}
        txns: list[InsiderTxn] = []
        self.last_reached_since = False
        page = edgar.get_current_filings(form="4", owner="include", page_size=100)
        while page is not None:
            older = False
            for filing in page:
                accepted = self._ingest(filing, since, until, txns, drops)
                if accepted is not None and accepted <= since:
                    older = self.last_reached_since = True
                    break
            if older:
                break
            page = page.next()
        return dedupe_cofilers(txns)
