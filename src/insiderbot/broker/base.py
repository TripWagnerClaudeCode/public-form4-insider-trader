"""Broker protocol."""
from __future__ import annotations

from typing import Protocol

from insiderbot.models import AccountState, OrderPlan, SubmittedOrder


class Broker(Protocol):
    def get_account(self) -> AccountState: ...

    def is_market_open(self) -> bool: ...

    def submit(self, plan: OrderPlan, client_order_id: str | None = None) -> SubmittedOrder:
        """Market order, time-in-force day."""
        ...

    def submit_stop(self, symbol: str, qty: int, stop_price: float,
                    client_order_id: str | None = None) -> SubmittedOrder:
        """Sell stop, GTC."""
        ...

    def cancel_orders_for(self, symbol: str) -> None: ...

    def order_status(self, broker_id: str) -> dict:
        """Keys: status, filled_qty, filled_avg_price."""
        ...

    def open_orders(self) -> list[dict]: ...
