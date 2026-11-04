"""Plain dataclasses shared across modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class InsiderTxn:
    accession: str
    accepted_at: datetime
    tx_date: date
    ticker: str
    issuer_cik: str
    issuer_name: str
    insider_cik: str
    insider_name: str
    is_officer: bool
    is_director: bool
    is_ten_pct: bool
    is_entity: bool
    officer_title: str
    code: str
    shares: float
    price: float
    value: float
    owned_after: float
    is_10b5_1: bool


@dataclass(frozen=True)
class IssuerFeatures:
    ticker: str
    asof: datetime
    n_buyers: int
    total_value: float
    has_ceo: bool
    has_cfo: bool
    has_chair: bool
    n_directors: int
    size_vs_holdings: float
    first_buy_months: float
    frac_10b5_1: float
    ret_3m: float | None
    drawdown_52w: float | None
    price: float | None
    adv20_usd: float | None
    mcap_usd: float | None
    sector: str | None
    exchange: str | None
    is_fund_like: bool
    accessions: tuple[str, ...]
    latest_accepted_at: datetime
    # Defaults are the conservative values: unknown means untradable / unnamed.
    name: str = ""
    tradable: bool = False


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float
    entry_date: date | None
    peak_price: float | None


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    last_equity: float
    positions: dict[str, Position]
    trading_blocked: bool


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    side: str
    qty: int
    reason: str
    ref_price: float


@dataclass(frozen=True)
class SubmittedOrder:
    plan: OrderPlan
    broker_id: str | None
    status: str
    error: str | None


@dataclass(frozen=True)
class AgentDecision:
    ticker: str
    action: str  # buy | pass | exit
    conviction: int
    thesis: str
    change_mind: str
    reports: dict = field(default_factory=dict)
