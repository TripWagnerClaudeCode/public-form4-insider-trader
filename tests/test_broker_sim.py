from insiderbot.broker.sim import SimBroker
from insiderbot.models import OrderPlan


def plan(side, qty, price=50.0):
    return OrderPlan("ABC", side, qty, "t", price)


def test_sim_buy_then_sell():
    b = SimBroker(100000, lambda s: 50.0)
    o = b.submit(plan("buy", 10))
    assert o.status == "filled" and o.broker_id
    acct = b.get_account()
    assert acct.cash == 100000 - 500
    assert acct.positions["ABC"].qty == 10
    assert acct.equity == 100000
    assert b.order_status(o.broker_id)["filled_qty"] == 10
    b.submit(plan("sell", 10))
    acct = b.get_account()
    assert "ABC" not in acct.positions and acct.cash == 100000


def test_sim_rejects_bad_orders():
    b = SimBroker(1000, lambda s: 50.0)
    assert b.submit(plan("buy", 1000)).status == "rejected"
    assert b.submit(plan("sell", 1)).status == "rejected"
    assert b.submit(plan("buy", 0)).status == "rejected"
    assert b.get_account().cash == 1000


def test_sim_stop_recorded():
    b = SimBroker(100000, lambda s: 50.0)
    b.submit(plan("buy", 10))
    s = b.submit_stop("ABC", 10, 45.0)
    assert s.broker_id
    oo = b.open_orders()
    assert len(oo) == 1 and oo[0]["symbol"] == "ABC" and oo[0]["stop_price"] == 45.0
    b.cancel_orders_for("ABC")
    assert b.open_orders() == []


def test_sim_sell_blocked_by_open_stop():
    b = SimBroker(100000, lambda s: 50.0)
    b.submit(plan("buy", 10))
    b.submit_stop("ABC", 10, 45.0)
    r = b.submit(plan("sell", 5))
    assert r.status == "rejected" and "insufficient qty" in r.error
    b.cancel_orders_for("ABC")
    assert b.submit(plan("sell", 5)).status == "filled"


def test_sim_stop_needs_shares_and_valid_price():
    b = SimBroker(100000, lambda s: 50.0)
    assert b.submit_stop("ABC", 1, 45.0).status == "rejected"
    b.submit(plan("buy", 10))
    assert b.submit_stop("ABC", 11, 45.0).status == "rejected"
    assert b.submit_stop("ABC", 2.7, 45.0).status == "rejected"
    assert b.submit_stop("ABC", 5, float("nan")).status == "rejected"
    assert b.submit_stop("ABC", 5, 0).status == "rejected"


def test_sim_hide_entry_info_and_last_equity():
    b = SimBroker(1000, lambda s: 50.0, hide_entry_info=True)
    b.submit(plan("buy", 2))
    p = b.get_account().positions["ABC"]
    assert p.entry_date is None and p.peak_price is None
    b.last_equity = 1200.0
    assert b.get_account().last_equity == 1200.0
    d = SimBroker(1000, lambda s: 50.0)
    d.submit(plan("buy", 2))
    assert d.get_account().positions["ABC"].entry_date is not None


class StubClient:
    def __init__(self, fail_submit=False, lookup=None):
        self.fail_submit, self.lookup, self.sent = fail_submit, lookup, []

    def submit_order(self, req):
        self.sent.append(req)
        if self.fail_submit:
            raise TimeoutError("boom")
        return type("O", (), {"id": "A1", "status": "accepted"})()

    def get_order_by_client_id(self, cid):
        if self.lookup is None:
            raise RuntimeError("lookup failed")
        return type("O", (), {"id": "A2", "status": "filled"})()


def _broker(client):
    from insiderbot.broker.alpaca_broker import AlpacaBroker

    b = AlpacaBroker("k", "s")
    b._client = client
    return b


def test_alpaca_broker_is_paper(monkeypatch):
    import insiderbot.broker.alpaca_broker as ab

    seen = {}

    class FakeTC:
        def __init__(self, key, secret, paper=False, **kw):
            seen["paper"] = paper

    monkeypatch.setattr(ab, "TradingClient", FakeTC)
    ab.AlpacaBroker("k", "s")
    assert seen["paper"] is True


def test_alpaca_sends_client_order_id():
    c = StubClient()
    b = _broker(c)
    o = b.submit(plan("buy", 3), client_order_id="abc123")
    assert o.status == "accepted" and c.sent[0].client_order_id == "abc123"
    b.submit_stop("ABC", 3, 45.678)
    assert c.sent[1].client_order_id and c.sent[1].stop_price == 45.68


def test_alpaca_submit_error_found_by_lookup():
    o = _broker(StubClient(fail_submit=True, lookup=True)).submit(plan("buy", 3))
    assert o.status == "filled" and o.broker_id == "A2"


def test_alpaca_submit_error_lookup_fails_is_unknown():
    o = _broker(StubClient(fail_submit=True)).submit(plan("buy", 3))
    assert o.status == "unknown" and o.broker_id is None and "client_order_id=" in o.error
    s = _broker(StubClient(fail_submit=True)).submit_stop("ABC", 3, 40.0)
    assert s.status == "unknown"


def test_alpaca_validation_and_subdollar_rounding():
    c = StubClient()
    b = _broker(c)
    assert b.submit(plan("buy", float("nan"))).status == "rejected"
    assert b.submit(plan("buy", 2.7)).status == "rejected"
    assert b.submit_stop("ABC", 2.7, 10.0).status == "rejected"
    assert b.submit_stop("ABC", 2, float("nan")).status == "rejected"
    assert b.submit_stop("ABC", 2, -1).status == "rejected"
    assert not c.sent
    b.submit_stop("ABC", 2, 0.123456)
    assert c.sent[0].stop_price == 0.1235

