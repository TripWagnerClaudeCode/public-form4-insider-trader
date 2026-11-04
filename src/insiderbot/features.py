"""Build per-issuer features from Form 4 purchases and apply hard filters.

Missing-data policy: missing data never helps a candidate. Absent price/ADV/mcap/
exchange/tradable fail the hard filter; absent return/drawdown features are None
(transform to 0); size_vs_holdings ignores buyers with no usable price or inconsistent
holdings data; first_buy_months is capped at the history the journal actually covers.

Point-in-time caveat: fundamentals (market cap, sector) and asset_info (exchange, name,
tradable) are CURRENT values, not point-in-time. A replay/backtest must therefore use the
features stored with each candidate, never rebuild them from today's data.

Lookahead: prior-buy lookups drop any journal txn with accepted_at > asof.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

import pandas as pd

from insiderbot.config import Filters
from insiderbot.models import InsiderTxn, IssuerFeatures

SIZE_CAP = 10.0
NO_PRIOR_MONTHS = 24.0
_CEO = re.compile(r"chief executive|\bceo\b", re.I)
_CFO = re.compile(r"chief financial|\bcfo\b", re.I)
_CHAIR = re.compile(r"chair", re.I)
_FUND_NAME = re.compile(
    r"\bETFS?\b|\bFUNDS?\b|\b(ISHARES|SPDR|PROSHARES)\b"
    r"|\b(INDEX|COMMODITY|CURRENCY|GOLD|SILVER|BITCOIN|ETHEREUM|BULLION|ETN)\s+TRUST\b"
    r"|\bACQUISITION\s+(CORP|CORPORATION|CO|COMPANY)\b|\bSPAC\b|\bBLANK\s+CHECK\b",
    re.I)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
PRICE_TOL = 0.01  # owned_after may undershoot shares by this fraction (new-holder rounding)


class MarketLike(Protocol):
    def bars(self, symbols: list[str], start: date, end: date, now: datetime | None = None) -> dict: ...
    def fundamentals(self, symbol: str) -> dict: ...
    def asset_info(self, symbols: list[str]) -> dict: ...


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _price_features(df: pd.DataFrame | None) -> dict:
    out = {"price": None, "ret_3m": None, "drawdown_52w": None, "adv20_usd": None}
    if df is None or df.empty:
        return out
    df = df.sort_index()
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    if last > 0:
        out["price"] = last
    if len(close) >= 64 and float(close.iloc[-64]) > 0 and last > 0:
        out["ret_3m"] = last / float(close.iloc[-64]) - 1
    if len(close) >= 60 and last > 0:
        hi = float(df["high"].astype(float).tail(252).max())
        if hi > 0:
            out["drawdown_52w"] = max(0.0, 1 - last / hi)
    vol = df["volume"].astype(float).tail(20)
    if df.attrs.get("feed") != "iex" and len(vol) >= 20 and vol.notna().sum() >= 20:
        out["adv20_usd"] = float((close.tail(20) * vol.fillna(0)).mean())
    return out


def _by_buyer(buys: list[InsiderTxn]) -> dict[str, list[InsiderTxn]]:
    by: dict[str, list[InsiderTxn]] = defaultdict(list)
    for t in buys:
        by[t.insider_cik].append(t)
    return by


def _size_vs_holdings(buys: list[InsiderTxn]) -> float:
    best = 0.0
    for ts in _by_buyer(buys).values():
        first = min(ts, key=lambda t: t.accepted_at)
        if not first.price or first.price <= 0:
            continue  # cannot size without a price: contributes nothing
        own = first.owned_after
        if not own or own <= 0 or own < first.shares * (1 - PRICE_TOL):
            continue  # missing/inconsistent holdings: contributes nothing
        value = sum(t.value for t in ts)
        prior = max(1.0, (own - first.shares) * first.price)
        best = max(best, min(SIZE_CAP, value / prior))
    return best


def _coverage_start(txns: list[InsiderTxn], journal, asof: datetime) -> datetime | None:
    """Earliest accepted_at of any P txn we know about (journal plus the txns passed in).
    Months of history the journal can vouch for = first buy date - this. None = no history."""
    times = [_utc(t.accepted_at) for t in txns if t.code == "P"]
    times += [_utc(t.accepted_at) for t in journal.txns_since(_EPOCH, codes=("P",))]
    times = [x for x in times if x <= asof]
    return min(times) if times else None


def _first_buy_months(buys: list[InsiderTxn], journal, asof: datetime,
                      coverage: datetime | None) -> float:
    """Months since the insider's previous P on the ticker. With no previous buy, the
    answer is capped at the months of history we cover (0 when we have none)."""
    best = 0.0
    for cik, ts in _by_buyer(buys).items():
        first_date = min(t.tx_date for t in ts)
        prior = [p for p in journal.insider_prior_buys(cik, ts[0].ticker, first_date)
                 if _utc(p.accepted_at) <= asof]  # no lookahead
        if prior:
            months = (first_date - max(p.tx_date for p in prior)).days / 30.44
        elif coverage is None:
            months = 0.0
        else:
            covered = (first_date - coverage.date()).days / 30.44
            months = min(NO_PRIOR_MONTHS, max(0.0, covered))
        best = max(best, months)
    return best


def build(txns: list[InsiderTxn], market: MarketLike, journal, asof: datetime,
          window_days: int) -> list[IssuerFeatures]:
    """Coverage start for first_buy_months is derived from the journal's earliest P txn
    and the txns passed in."""
    asof_u = _utc(asof)
    lo = asof_u - timedelta(days=window_days)
    groups: dict[str, list[InsiderTxn]] = defaultdict(list)
    for t in txns:
        if t.code == "P" and t.ticker and lo <= _utc(t.accepted_at) <= asof_u:
            groups[t.ticker].append(t)
    if not groups:
        return []
    tickers = sorted(groups)
    end = asof_u.date()
    bars = market.bars(tickers, end - timedelta(days=370), end, now=asof_u) or {}
    info = market.asset_info(tickers) or {}
    coverage = _coverage_start(txns, journal, asof_u)
    out = []
    for tk in tickers:
        buys = groups[tk]
        buyers = _by_buyer(buys)
        total = sum(t.value for t in buys)
        if total > 0:
            frac_plan = sum(t.value for t in buys if t.is_10b5_1) / total
        else:
            frac_plan = sum(1 for t in buys if t.is_10b5_1) / len(buys)
        fund_like = all(t.is_entity and not t.is_officer and not t.is_director
                        for ts in buyers.values() for t in ts)
        directors = {c for c, ts in buyers.items() if any(t.is_director for t in ts)}
        titles = [t.officer_title or "" for t in buys]
        fund = market.fundamentals(tk) or {}
        ai = info.get(tk) or {}
        pf = _price_features(bars.get(tk))
        out.append(IssuerFeatures(
            ticker=tk, asof=asof, n_buyers=len(buyers), total_value=total,
            has_ceo=any(_CEO.search(x) for x in titles),
            has_cfo=any(_CFO.search(x) for x in titles),
            has_chair=any(_CHAIR.search(x) for x in titles),
            n_directors=len(directors),
            size_vs_holdings=_size_vs_holdings(buys),
            first_buy_months=_first_buy_months(buys, journal, asof_u, coverage),
            frac_10b5_1=frac_plan,
            ret_3m=pf["ret_3m"], drawdown_52w=pf["drawdown_52w"], price=pf["price"],
            adv20_usd=pf["adv20_usd"],
            mcap_usd=fund.get("market_cap"), sector=fund.get("sector"),
            exchange=ai.get("exchange"), is_fund_like=fund_like,
            accessions=tuple(sorted({t.accession for t in buys})),
            latest_accepted_at=max(t.accepted_at for t in buys),
            name=ai.get("name") or buys[0].issuer_name or "",
            tradable=bool(ai.get("tradable", False)),
        ))
    return out


def hard_filter(f: IssuerFeatures, cfg: Filters) -> str | None:
    """Reason string if the candidate must be dropped, else None."""
    if not f.tradable:
        return "not tradable"
    if not f.exchange or f.exchange.upper() not in {e.upper() for e in cfg.exchanges}:
        return "exchange"
    if f.price is None or f.price < cfg.min_price:
        return "price<5"
    if f.adv20_usd is None or f.adv20_usd < cfg.min_adv_usd:
        return "adv<2M"
    if f.mcap_usd is None:
        return "mcap unknown"
    if f.mcap_usd < cfg.min_mcap_usd:
        return "mcap<300M"
    if f.is_fund_like:
        return "fund-only buyers"
    if _FUND_NAME.search(f.name or ""):
        return "etf/fund"
    return None
