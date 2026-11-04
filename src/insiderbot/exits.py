"""Exit rules. Pure functions; the runner does the broker work."""
from __future__ import annotations

import math
from datetime import date

from insiderbot.broker.alpaca_broker import round_stop
from insiderbot.formula import Formula
from insiderbot.models import Position


def stop_price(entry_price: float, fm: Formula) -> float:
    """Protective GTC stop level, rounded the way the broker accepts it."""
    return round_stop(entry_price * (1 - fm.stop_loss_pct))


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def check(pos: Position, meta: dict, last_price: float, fm: Formula, today: date,
          trading_days_held: int, insider_sold: bool, agent_exit: bool,
          min_hold_days: int) -> str | None:
    """First matching exit reason, or None.

    meta supplies entry_price/peak_price from the journal (the broker does not).
    Runner contract: cancel_orders_for(symbol) before the sell; if the sell fails after the
    cancel, re-place the GTC stop; "unknown" order status counts as pending, not exited.
    min_hold_days is risk.min_hold_days_for_agent_exit. An unusable last price
    disables the price-based rules only; the other rules still apply.
    """
    entry = meta.get("entry_price") if _ok(meta.get("entry_price")) else pos.avg_price
    if _ok(last_price) and _ok(entry):
        if last_price <= entry * (1 - fm.stop_loss_pct):
            return "stop"
        peak = max(x for x in (meta.get("peak_price"), pos.peak_price, last_price) if _ok(x))
        if peak >= entry * (1 + fm.take_profit_pct) and last_price <= peak * (1 - fm.trailing_pct):
            return "trailing"
    if trading_days_held >= fm.time_stop_days:
        return "time"
    if insider_sold:
        return "insider_sale"
    if agent_exit and trading_days_held >= min_hold_days:
        return "agent"
    return None
