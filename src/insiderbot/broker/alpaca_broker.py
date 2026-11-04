"""Alpaca PAPER broker. Live trading is never constructed.

Order status "unknown" means the submit call failed in transport AND the order
could not be looked up by client_order_id. Callers MUST treat "unknown" as
possibly filled (do not assume no position; reconcile before acting again).
"""
from __future__ import annotations

import math
import uuid

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetOrdersRequest, MarketOrderRequest,
                                     StopOrderRequest)

from insiderbot.models import AccountState, OrderPlan, Position, SubmittedOrder


def _f(v) -> float | None:
    return None if v is None else float(v)


def _status(v) -> str:
    return str(getattr(v, "value", v))


def _valid_qty(qty) -> bool:
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return False
    return math.isfinite(q) and q > 0 and q == int(q)


def _valid_price(p) -> bool:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p) and p > 0


def round_stop(price: float) -> float:
    return round(price, 2 if price >= 1 else 4)


class AlpacaBroker:
    def __init__(self, key: str, secret: str):
        if not key or not secret:
            raise ValueError("Alpaca key/secret required")
        # paper=True is hard-wired; there is no parameter to change it.
        self._client = TradingClient(key, secret, paper=True)

    def get_account(self) -> AccountState:
        a = self._client.get_account()
        positions = {}
        for p in self._client.get_all_positions():
            positions[p.symbol] = Position(
                p.symbol, float(p.qty), float(p.avg_entry_price),
                float(p.market_value), None, None)
        blocked = bool(a.trading_blocked) or bool(getattr(a, "account_blocked", False))
        return AccountState(float(a.equity), float(a.cash), float(a.last_equity),
                            positions, blocked)

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def _send(self, plan: OrderPlan, request, cid: str) -> SubmittedOrder:
        try:
            o = self._client.submit_order(request)
            return SubmittedOrder(plan, str(o.id), _status(o.status), None)
        except Exception as e:  # noqa: BLE001
            try:
                o = self._client.get_order_by_client_id(cid)
            except Exception:  # noqa: BLE001
                return SubmittedOrder(plan, None, "unknown",
                                      f"{e} (client_order_id={cid}; lookup failed)")
            return SubmittedOrder(plan, str(o.id), _status(o.status), None)

    def submit(self, plan: OrderPlan, client_order_id: str | None = None) -> SubmittedOrder:
        if plan.side not in ("buy", "sell") or not _valid_qty(plan.qty):
            return SubmittedOrder(plan, None, "rejected", "invalid order plan")
        cid = client_order_id or uuid.uuid4().hex
        try:
            req = MarketOrderRequest(
                symbol=plan.symbol, qty=int(plan.qty),
                side=OrderSide.BUY if plan.side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY, client_order_id=cid)
        except Exception as e:  # noqa: BLE001
            return SubmittedOrder(plan, None, "rejected", str(e))
        return self._send(plan, req, cid)

    def submit_stop(self, symbol: str, qty: int, stop_price: float,
                    client_order_id: str | None = None) -> SubmittedOrder:
        plan = OrderPlan(symbol, "sell", qty, "stop", stop_price)
        if not _valid_qty(qty):
            return SubmittedOrder(plan, None, "rejected", "invalid qty")
        if not _valid_price(stop_price):
            return SubmittedOrder(plan, None, "rejected", "invalid stop price")
        cid = client_order_id or uuid.uuid4().hex
        try:
            req = StopOrderRequest(
                symbol=symbol, qty=int(qty), side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round_stop(float(stop_price)), client_order_id=cid)
        except Exception as e:  # noqa: BLE001
            return SubmittedOrder(plan, None, "rejected", str(e))
        return self._send(plan, req, cid)

    def cancel_orders_for(self, symbol: str) -> None:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        for o in self._client.get_orders(req):
            self._client.cancel_order_by_id(o.id)

    def order_status(self, broker_id: str) -> dict:
        o = self._client.get_order_by_id(broker_id)
        return {"status": _status(o.status), "filled_qty": float(o.filled_qty or 0),
                "filled_avg_price": _f(o.filled_avg_price)}

    def open_orders(self) -> list[dict]:
        out = []
        for o in self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            out.append({"id": str(o.id), "symbol": o.symbol, "qty": _f(o.qty),
                        "side": _status(o.side), "type": _status(o.order_type),
                        "stop_price": _f(o.stop_price), "status": _status(o.status)})
        return out
