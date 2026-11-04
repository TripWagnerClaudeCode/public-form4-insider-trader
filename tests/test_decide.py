from datetime import datetime, timezone

import pytest

from insiderbot import formula
from insiderbot.config import ROOT, Risk
from insiderbot.decide import buy_orders, client_order_id
from insiderbot.models import AccountState, AgentDecision, IssuerFeatures, Position


def feats(ticker="ABC", price=20.0, sector="Tech", **kw):
    d = dict(ticker=ticker, asof=datetime(2026, 11, 2, tzinfo=timezone.utc), n_buyers=1, total_value=15_000.0,
             has_ceo=False, has_cfo=False, has_chair=False, n_directors=1, size_vs_holdings=0.02,
             first_buy_months=0.0, frac_10b5_1=0.0, ret_3m=0.05, drawdown_52w=0.05, price=price,
             adv20_usd=5e6, mcap_usd=5e9, sector=sector, exchange="NASDAQ", is_fund_like=False,
             accessions=("a",), latest_accepted_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
             name="X", tradable=True)
    d.update(kw)
    return IssuerFeatures(**d)


def dec(ticker="ABC", action="buy", conviction=3):
    return AgentDecision(ticker, action, conviction, "t", "c")


def acct(equity=100_000.0, cash=None, positions=None):
    return AccountState(equity, equity if cash is None else cash, equity, positions or {}, False)


@pytest.fixture
def fm():
    return formula.load("v001", ROOT / "formula")


def cand(t="ABC", score=75.0, price=20.0, action="buy", conviction=3, sector="Tech"):
    return (feats(t, price, sector), score, dec(t, action, conviction))


def test_basic_sizing(fm):
    orders, skips = buy_orders([cand(score=75)], acct(), fm, Risk(), {"ABC": "Tech"}, {})
    # tier 70 -> 1.0, conv 3 -> 1.0, weight 0.08 -> 8000 / 20
    assert [(o.symbol, o.side, o.qty) for o in orders] == [("ABC", "buy", 400)]
    assert skips == []


def test_weight_capped(fm):
    orders, _ = buy_orders([cand(score=90, conviction=5)], acct(), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders[0].qty == 400  # 1.25*1.25 capped to max_position_weight


def test_rules_veto_beats_agent_buy(fm):
    orders, _ = buy_orders([cand(score=fm.buy_threshold - 1)], acct(), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == []


@pytest.mark.parametrize("action", ["pass", "exit", "", "BUY", "error"])
def test_agent_pass_beats_rules(fm, action):
    orders, _ = buy_orders([cand(score=95, action=action)], acct(), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == []


def test_already_held_not_rebought(fm):
    pos = {"ABC": Position("ABC", 10, 20, 200, None, None)}
    orders, _ = buy_orders([cand()], acct(positions=pos), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == []


def test_whole_share_floor_skip(fm):
    orders, skips = buy_orders([cand(price=5000.0)], acct(10_000.0), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == [] and skips[0]["reason"] == "qty<1"


def test_no_price_skips(fm):
    orders, skips = buy_orders([cand(price=0.0)], acct(), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == [] and skips


def test_max_new_buys(fm):
    cs = [cand(t, score=90 - i, sector=t) for i, t in enumerate("ABCDE")]
    sec = {t: t for t in "ABCDE"}
    orders, skips = buy_orders(cs, acct(), fm, Risk(), sec, {})
    assert [o.symbol for o in orders] == ["A", "B", "C"]
    assert {s["reason"] for s in skips} == {"max_new_buys_per_scan"}


def test_ordered_by_score(fm):
    cs = [cand("A", 65, sector="a"), cand("B", 90, sector="b")]
    orders, _ = buy_orders(cs, acct(), fm, Risk(), {"A": "a", "B": "b"}, {})
    assert [o.symbol for o in orders] == ["B", "A"]


def test_sector_cap(fm):
    cs = [cand(t, score=90 - i) for i, t in enumerate("ABCDE")]
    sec = {t: "Tech" for t in "ABCDE"}
    r = Risk(max_new_buys_per_scan=10, max_sector_weight=0.20)
    orders, skips = buy_orders(cs, acct(), fm, r, sec, {})
    assert len(orders) == 2
    assert all(s["reason"] == "max_sector_weight" for s in skips)


def test_sector_cap_counts_existing_and_unknown(fm):
    pos = {"Z": Position("Z", 100, 200, 25_000, None, None)}
    orders, skips = buy_orders([cand()], acct(positions=pos), fm, Risk(), {"ABC": "Tech", "Z": "Tech"}, {})
    assert orders == [] and skips[0]["reason"] == "max_sector_weight"
    # unknown sectors share one bucket, still capped
    orders, skips = buy_orders([cand(sector=None)], acct(positions=pos), fm, Risk(), {"ABC": None}, {})
    assert orders == [] and skips[0]["reason"] == "max_sector_weight"


def test_max_positions_and_gross(fm):
    pos = {f"P{i}": Position(f"P{i}", 1, 1, 1, None, None) for i in range(12)}
    orders, skips = buy_orders([cand()], acct(positions=pos), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == [] and skips[0]["reason"] == "max_positions"
    pos = {"P": Position("P", 1, 1, 86_000, None, None)}
    orders, skips = buy_orders([cand()], acct(positions=pos), fm, Risk(), {"ABC": "Tech", "P": "Oil"}, {})
    assert orders == [] and skips[0]["reason"] == "max_gross_exposure"


def test_cash_limit(fm):
    orders, skips = buy_orders([cand()], acct(100_000.0, cash=1000.0), fm, Risk(), {"ABC": "Tech"}, {})
    assert orders == [] and skips[0]["reason"] == "insufficient_cash"


def test_ticker_mismatch_and_bad_conviction(fm):
    c = (feats("ABC"), 80.0, dec("XYZ"))
    assert buy_orders([c], acct(), fm, Risk(), {}, {})[0] == []
    assert buy_orders([cand(conviction=9)], acct(), fm, Risk(), {"ABC": "Tech"}, {})[0] == []


def test_client_order_id_deterministic():
    assert client_order_id(7, "abc", "buy") == client_order_id(7, "ABC", "buy")
    assert client_order_id(7, "ABC", "buy") != client_order_id(7, "ABC", "sell")
    assert client_order_id(7, "ABC", "buy") != client_order_id(8, "ABC", "buy")


def test_pending_counts_as_held(fm):
    orders, skips = buy_orders([cand()], acct(), fm, Risk(), {"ABC": "Tech"}, {"ABC": 100.0})
    assert orders == [] and skips[0]["reason"] == "already_held_or_pending"


def test_pending_counts_toward_gross_positions_sector(fm):
    sec = {"ABC": "Tech", "P": "Oil"}
    o, s = buy_orders([cand()], acct(), fm, Risk(), sec, {"P": 86_000.0})
    assert o == [] and s[0]["reason"] == "max_gross_exposure"
    pend = {f"P{i}": 1.0 for i in range(12)}
    o, s = buy_orders([cand()], acct(), fm, Risk(), sec, pend)
    assert o == [] and s[0]["reason"] == "max_positions"
    o, s = buy_orders([cand()], acct(), fm, Risk(), {"ABC": "Tech", "P": "Tech"}, {"P": 25_000.0})
    assert o == [] and s[0]["reason"] == "max_sector_weight"
