"""Kill switch and risk gate.

Spec: ANY blocking reason (kill switch, stale data, market closed, broker trading block,
loss limit, missing risk baseline) means NO ORDERS AT ALL, sells included. The GTC stops
already at the broker keep protecting positions meanwhile.

Runner contract: call pre_trade first and submit nothing if it returns reasons; build the
batch; reject the WHOLE batch if check_orders returns anything; check_orders re-checks the
kill switch so it is the last gate before submit. Exit (sell-only) batches use check_sells
instead: portfolio-level limits (gross, positions, sector, cash) apply to buys only and must
never block a sell that reduces risk.

Kill switch: the KILL_SWITCH file, or BOT_KILL_SWITCH=yes in the process environment OR in
the project .env (systemd timers do not load .env into the environment).
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

from insiderbot.config import ROOT
from insiderbot.models import AccountState, OrderPlan

_EPS = 1e-9
ENV_FILE = ROOT / ".env"
_ON = ("yes", "1", "true")


def _path(cfg) -> Path:
    return ROOT / cfg.risk.kill_switch_file  # absolute paths win


def _env_file_flag() -> bool:
    """BOT_KILL_SWITCH from the project .env (only that key is read). An existing but
    unreadable .env fails closed (kill switch active)."""
    try:
        if not Path(ENV_FILE).exists():
            return False
        v = dotenv_values(ENV_FILE).get("BOT_KILL_SWITCH") or ""
    except Exception:  # noqa: BLE001
        return True
    return v.replace("\r", "").strip().lower() in _ON


def kill_switch_active(cfg) -> bool:
    if os.environ.get("BOT_KILL_SWITCH", "").strip().lower() in _ON:
        return True
    if _env_file_flag():
        return True
    return _path(cfg).exists()


def trip(cfg, reason: str) -> None:
    p = _path(cfg)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with p.open("a", newline="\n", encoding="utf-8") as fh:
        fh.write(f"{stamp} {reason}\n")


def _bad(x) -> bool:
    return not isinstance(x, (int, float)) or not math.isfinite(x)


def _valid(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def pre_trade(cfg, acct: AccountState, peak: float | None, market_open: bool,
              data_fresh: bool) -> list[str]:
    """Reasons that block ALL orders, buys and sells (empty = clear).

    Trips the KILL_SWITCH file (via trip) on a daily-loss or drawdown breach. Never fails
    open: with no usable baseline it returns "risk baseline unavailable".
    """
    r = cfg.risk
    reasons: list[str] = []
    if kill_switch_active(cfg):
        reasons.append("kill switch active")
    if acct.trading_blocked:
        reasons.append("broker account trading_blocked")
    if not market_open:
        reasons.append("market closed")
    if not data_fresh:
        reasons.append("data stale")
    if not _valid(acct.equity):
        reasons.append("invalid equity")
        reasons.append("risk baseline unavailable")
        return reasons
    if not _valid(acct.last_equity) and not _valid(peak):
        reasons.append("risk baseline unavailable")
    if _valid(acct.last_equity):
        loss = (acct.last_equity - acct.equity) / acct.last_equity
        if loss >= r.max_daily_loss_pct - _EPS:
            msg = f"daily loss {loss:.2%} >= {r.max_daily_loss_pct:.2%}"
            reasons.append(msg)
            trip(cfg, msg)
    base = max(x for x in (peak, acct.equity, acct.last_equity) if _valid(x))
    dd = (base - acct.equity) / base
    if dd >= r.max_drawdown_pct - _EPS:
        msg = f"drawdown {dd:.2%} >= {r.max_drawdown_pct:.2%}"
        reasons.append(msg)
        trip(cfg, msg)
    return reasons


def check_sells(cfg, orders: list[OrderPlan], acct: AccountState) -> list[str]:
    """Checks for an exit (sell-only) batch (empty = ok): kill switch, side, whole qty >= 1,
    no duplicate symbol, qty <= held. Deliberately NO portfolio-level limits: a breach of
    gross/positions/sector/cash must never block a risk-reducing sell."""
    if kill_switch_active(cfg):
        return ["kill switch active"]
    out: list[str] = []
    seen: set[str] = set()
    for o in orders:
        tag = f"{o.side} {o.symbol}"
        if o.side != "sell":
            out.append(f"{tag}: not a sell")
            continue
        if not isinstance(o.qty, int) or isinstance(o.qty, bool) or o.qty < 1:
            out.append(f"{tag}: qty must be a whole number >= 1")
            continue
        if o.symbol in seen:
            out.append(f"{tag}: duplicate order")
        seen.add(o.symbol)
        held = acct.positions.get(o.symbol)
        if held is None or o.qty > held.qty:
            out.append(f"{tag}: sells more than held")
    return out


def check_orders(cfg, orders: list[OrderPlan], acct: AccountState, pending: dict[str, float],
                 sector_of: dict[str, str | None]) -> list[str]:
    """Order-level violations over the whole batch (empty = ok); last check before submit.

    Any violation means submit NOTHING from this batch. Re-checks the kill switch.
    pending maps symbol -> notional of open or "unknown"-status buys; those count as held
    and toward gross, position count and sector weight.
    """
    r = cfg.risk
    if kill_switch_active(cfg):
        return ["kill switch active"]
    out: list[str] = []
    equity = acct.equity
    if not _valid(equity):
        return ["invalid equity"]
    gross = sum(p.market_value for p in acct.positions.values()) + sum(pending.values())
    cash = acct.cash
    sectors: dict[str, float] = {}
    for sym, p in acct.positions.items():
        k = sector_of.get(sym) or "?unknown"
        sectors[k] = sectors.get(k, 0.0) + p.market_value
    for sym, v in pending.items():
        if sym not in acct.positions:
            k = sector_of.get(sym) or "?unknown"
            sectors[k] = sectors.get(k, 0.0) + v
    held_or_pending = set(acct.positions) | set(pending)
    seen: set[tuple[str, str]] = set()
    n_new = 0
    for o in orders:
        tag = f"{o.side} {o.symbol}"
        if o.side not in ("buy", "sell"):
            out.append(f"{tag}: bad side")
            continue
        if not isinstance(o.qty, int) or isinstance(o.qty, bool) or o.qty < 1:
            out.append(f"{tag}: qty must be a whole number >= 1")
            continue
        if (o.symbol, o.side) in seen:
            out.append(f"{tag}: duplicate order")
        seen.add((o.symbol, o.side))
        if o.side == "sell":
            held = acct.positions.get(o.symbol)
            if held is None or o.qty > held.qty:
                out.append(f"{tag}: sells more than held")
            continue
        if _bad(o.ref_price) or o.ref_price <= 0:
            out.append(f"{tag}: no valid price")
            continue
        cost = o.qty * o.ref_price
        if o.symbol in held_or_pending:
            out.append(f"{tag}: already held or pending")
        else:
            n_new += 1
        if cost / equity > r.max_position_weight + _EPS:
            out.append(f"{tag}: position weight {cost / equity:.2%} > {r.max_position_weight:.2%}")
        gross += cost
        cash -= cost
        k = sector_of.get(o.symbol) or "?unknown"
        sectors[k] = sectors.get(k, 0.0) + cost
        if sectors[k] / equity > r.max_sector_weight + _EPS:
            out.append(f"{tag}: sector {k} weight > {r.max_sector_weight:.2%}")
    if n_new > r.max_new_buys_per_scan:
        out.append(f"{n_new} new buys > max_new_buys_per_scan {r.max_new_buys_per_scan}")
    if len(held_or_pending) + n_new > r.max_positions:
        out.append("exceeds max_positions")
    if gross / equity > r.max_gross_exposure + _EPS:
        out.append(f"gross exposure {gross / equity:.2%} > {r.max_gross_exposure:.2%}")
    if cash < -_EPS:
        out.append("insufficient cash")
    return out
