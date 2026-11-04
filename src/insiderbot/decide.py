"""Buy decisions and position sizing. Rules veto first; the PM can only say no.

Runner contract: build `pending` (symbol -> notional) from open orders AND from submits whose
status is "unknown" (possibly filled; never re-buy, let reconcile resolve). Pass
client_order_id() to submit for idempotent retries. Run check_orders on the batch and
reject the whole batch if it returns anything. Exits: cancel_orders_for(symbol) before a
sell; if the sell fails after the cancel, re-place the GTC stop.
"""
from __future__ import annotations

import hashlib
import math

from insiderbot.formula import Formula, tier_weight
from insiderbot.models import AccountState, AgentDecision, IssuerFeatures, OrderPlan

CONV_MULT = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.1, 5: 1.25}
_UNKNOWN = "?unknown"
_EPS = 1e-9


def client_order_id(run_id: int | str, symbol: str, side: str) -> str:
    """Deterministic id so a retried submit is idempotent at the broker."""
    raw = f"insiderbot|{run_id}|{symbol.upper()}|{side}"
    return "ib-" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _good(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def buy_orders(cands: list[tuple[IssuerFeatures, float, AgentDecision]], acct: AccountState,
               fm: Formula, risk, sector_of: dict[str, str | None],
               pending: dict[str, float]) -> tuple[list[OrderPlan], list[dict]]:
    orders: list[OrderPlan] = []
    skips: list[dict] = []
    equity = acct.equity
    n_pos = len(set(acct.positions) | set(pending))
    gross = sum(p.market_value for p in acct.positions.values()) + sum(pending.values())
    cash = acct.cash
    sectors: dict[str, float] = {}
    for sym, p in acct.positions.items():
        key = sector_of.get(sym) or _UNKNOWN
        sectors[key] = sectors.get(key, 0.0) + p.market_value
    for sym, v in pending.items():
        if sym not in acct.positions:
            key = sector_of.get(sym) or _UNKNOWN
            sectors[key] = sectors.get(key, 0.0) + v

    def skip(sym: str, reason: str, **kw) -> None:
        skips.append({"symbol": sym, "reason": reason, **kw})

    for f, sc, d in sorted(cands, key=lambda c: c[1], reverse=True):
        sym = f.ticker
        if d.action != "buy" or d.ticker != sym:
            skip(sym, "agent_not_buy", action=d.action)
            continue
        if not isinstance(sc, (int, float)) or not math.isfinite(sc) or sc < fm.buy_threshold:
            skip(sym, "below_buy_threshold", score=sc)
            continue
        if sym in acct.positions or sym in pending:
            skip(sym, "already_held_or_pending")
            continue
        if len(orders) >= risk.max_new_buys_per_scan:
            skip(sym, "max_new_buys_per_scan")
            continue
        mult = CONV_MULT.get(d.conviction) if isinstance(d.conviction, int) else None
        if mult is None:
            skip(sym, "bad_conviction", conviction=d.conviction)
            continue
        price = f.price
        if not _good(price) or not _good(equity):
            skip(sym, "no_price_or_equity")
            continue
        weight = min(risk.max_position_weight * tier_weight(sc, fm) * mult, risk.max_position_weight)
        qty = math.floor(weight * equity / price)
        if qty < 1:
            skip(sym, "qty<1", weight=weight, price=price)
            continue
        if n_pos + len(orders) + 1 > risk.max_positions:
            skip(sym, "max_positions")
            continue
        cost = qty * price
        if (gross + cost) / equity > risk.max_gross_exposure + _EPS:
            skip(sym, "max_gross_exposure")
            continue
        sector = sector_of.get(sym) or f.sector or _UNKNOWN
        if (sectors.get(sector, 0.0) + cost) / equity > risk.max_sector_weight + _EPS:
            skip(sym, "max_sector_weight", sector=sector)
            continue
        if cost > cash:
            skip(sym, "insufficient_cash")
            continue
        orders.append(OrderPlan(sym, "buy", qty, f"score={sc:.1f} conv={d.conviction}", float(price)))
        gross += cost
        cash -= cost
        sectors[sector] = sectors.get(sector, 0.0) + cost
    return orders, skips
