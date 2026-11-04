"""Simulated broker: instant fills at the given price. In memory by default; dump()/load()
let the frozen control persist its book between runs."""
from __future__ import annotations

import itertools
import math
from datetime import date, datetime
from typing import Callable
from zoneinfo import ZoneInfo

from insiderbot.models import AccountState, OrderPlan, Position, SubmittedOrder

NY = ZoneInfo("America/New_York")


class SimBroker:
    """hide_entry_info=False (default) reports entry_date/peak_price; True reports
    None for both, like AlpacaBroker. last_equity is settable (defaults to
    starting equity) so daily-loss guards can be tested."""

    def __init__(self, starting_equity: float, prices: Callable[[str], float],
                 hide_entry_info: bool = False):
        self.cash = float(starting_equity)
        self.last_equity = float(starting_equity)
        self.prices = prices
        self.hide_entry_info = hide_entry_info
        self._pos: dict[str, dict] = {}
        self._orders: dict[str, dict] = {}
        self._stops: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.open = True

    def _id(self) -> str:
        return f"sim-{next(self._ids)}"

    def get_account(self) -> AccountState:
        positions = {}
        for sym, p in self._pos.items():
            px = self.prices(sym)
            p["peak"] = max(p["peak"], px)
            hide = self.hide_entry_info
            positions[sym] = Position(sym, p["qty"], p["avg"], p["qty"] * px,
                                      None if hide else p["entry"],
                                      None if hide else p["peak"])
        equity = self.cash + sum(p.market_value for p in positions.values())
        return AccountState(equity, self.cash, self.last_equity, positions, False)

    def is_market_open(self) -> bool:
        return self.open

    def _reject(self, plan: OrderPlan, why: str) -> SubmittedOrder:
        return SubmittedOrder(plan, None, "rejected", why)

    def _stopped_qty(self, symbol: str) -> int:
        return sum(s["qty"] for s in self._stops.values() if s["symbol"] == symbol)

    @staticmethod
    def _whole(qty) -> bool:
        return isinstance(qty, int) and not isinstance(qty, bool) and qty > 0

    def submit(self, plan: OrderPlan, client_order_id: str | None = None) -> SubmittedOrder:
        qty = plan.qty
        if not self._whole(qty):
            return self._reject(plan, "qty must be a positive whole number")
        px = float(self.prices(plan.symbol))
        if not math.isfinite(px) or px <= 0:
            return self._reject(plan, "no valid price")
        if plan.side == "buy":
            cost = qty * px
            if cost > self.cash:
                return self._reject(plan, "insufficient cash")
            p = self._pos.get(plan.symbol)
            if p:
                tot = p["qty"] + qty
                p["avg"] = (p["avg"] * p["qty"] + cost) / tot
                p["qty"] = tot
            else:
                self._pos[plan.symbol] = {
                    "qty": qty, "avg": px, "peak": px,
                    "entry": datetime.now(NY).date()}
            self.cash -= cost
        elif plan.side == "sell":
            p = self._pos.get(plan.symbol)
            held = p["qty"] if p else 0
            if qty > held - self._stopped_qty(plan.symbol):
                return self._reject(plan, "insufficient qty available")
            p["qty"] -= qty
            self.cash += qty * px
            if p["qty"] == 0:
                del self._pos[plan.symbol]
        else:
            return self._reject(plan, f"bad side {plan.side}")
        oid = self._id()
        self._orders[oid] = {"status": "filled", "filled_qty": qty,
                             "filled_avg_price": px, "client_order_id": client_order_id}
        return SubmittedOrder(plan, oid, "filled", None)

    def submit_stop(self, symbol: str, qty: int, stop_price: float,
                    client_order_id: str | None = None) -> SubmittedOrder:
        plan = OrderPlan(symbol, "sell", qty, "stop", stop_price)
        if not self._whole(qty):
            return self._reject(plan, "qty must be a positive whole number")
        if not isinstance(stop_price, (int, float)) or not math.isfinite(stop_price) \
                or stop_price <= 0:
            return self._reject(plan, "invalid stop price")
        p = self._pos.get(symbol)
        if qty > (p["qty"] if p else 0) - self._stopped_qty(symbol):
            return self._reject(plan, "insufficient qty available")
        oid = self._id()
        self._stops[oid] = {"id": oid, "symbol": symbol, "qty": qty, "side": "sell",
                            "type": "stop", "stop_price": stop_price, "status": "new"}
        self._orders[oid] = {"status": "new", "filled_qty": 0, "filled_avg_price": None,
                             "client_order_id": client_order_id}
        return SubmittedOrder(plan, oid, "new", None)

    def cancel_orders_for(self, symbol: str) -> None:
        for oid in [k for k, v in self._stops.items() if v["symbol"] == symbol]:
            del self._stops[oid]
            self._orders[oid]["status"] = "canceled"

    def order_status(self, broker_id: str) -> dict:
        o = self._orders.get(broker_id)
        return dict(o) if o else {"status": "unknown", "filled_qty": 0,
                                  "filled_avg_price": None}

    def open_orders(self) -> list[dict]:
        return [dict(v) for v in self._stops.values()]

    # --- persistence (frozen control only) ------------------------------------------------

    def dump(self) -> dict:
        pos = {k: {**v, "entry": v["entry"].isoformat() if v.get("entry") else None}
               for k, v in self._pos.items()}
        return {"version": 1, "cash": self.cash, "last_equity": self.last_equity, "positions": pos,
                "orders": self._orders, "stops": self._stops, "next_id": next(self._ids)}

    @classmethod
    def load(cls, d: dict, prices: Callable[[str], float]) -> SimBroker:
        if not isinstance(d, dict) or d.get("version") != 1:
            raise ValueError("unknown simulated book format")
        b = cls(float(d["cash"]), prices)
        b.last_equity = float(d["last_equity"])
        b._pos = {k: {**v, "entry": date.fromisoformat(v["entry"]) if v.get("entry") else None}
                  for k, v in d.get("positions", {}).items()}
        b._orders = dict(d.get("orders", {}))
        b._stops = dict(d.get("stops", {}))
        b._ids = itertools.count(int(d.get("next_id", 1)))
        return b
