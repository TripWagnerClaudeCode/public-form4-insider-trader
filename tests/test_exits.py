from datetime import date

import pytest

from insiderbot import formula
from insiderbot.config import ROOT
from insiderbot.exits import check, stop_price
from insiderbot.models import Position

TODAY = date(2026, 11, 3)


@pytest.fixture
def fm():
    return formula.load("v001", ROOT / "formula")  # stop .12 tp .25 trail .10 time 40


def pos():
    return Position("ABC", 10, 100.0, 1000.0, None, None)


def meta(peak=100.0):
    return {"entry_price": 100.0, "peak_price": peak}


def run(fm, last, peak=100.0, days=3, sold=False, agent=False, pos_peak=None):
    p = Position("ABC", 10, 100.0, 1000.0, None, pos_peak)
    return check(p, meta(peak), last, fm, TODAY, days, sold, agent, 5)


def test_exit_stop(fm):
    assert run(fm, 88.0) == "stop"
    assert run(fm, 88.01) is None


def test_exit_trailing(fm):
    assert run(fm, 112.0, peak=130.0) == "trailing"
    assert run(fm, 118.0, peak=130.0) is None
    assert run(fm, 100.0, peak=120.0) is None  # peak never reached take-profit
    assert run(fm, 100.0, peak=None) is None


def test_exit_time(fm):
    assert run(fm, 100.0, days=40) == "time"
    assert run(fm, 100.0, days=39) is None


def test_insider_sale(fm):
    assert run(fm, 100.0, sold=True) == "insider_sale"


def test_agent_exit_min_hold(fm):
    assert run(fm, 100.0, days=4, agent=True) is None
    assert run(fm, 100.0, days=5, agent=True) == "agent"


def test_order_of_precedence(fm):
    assert run(fm, 80.0, days=50, sold=True, agent=True) == "stop"


def test_bad_price_no_price_exit(fm):
    assert run(fm, 0.0) is None
    assert run(fm, float("nan"), days=40) == "time"


def test_falls_back_to_position_avg_price(fm):
    assert check(pos(), {}, 85.0, fm, TODAY, 1, False, False, 5) == "stop"


def test_stop_price(fm):
    assert stop_price(100.0, fm) == 88.0
    assert stop_price(0.5, fm) == 0.44
    assert stop_price(0.5123, fm) == 0.4508


def test_trailing_uses_broker_peak_too(fm):
    assert run(fm, 112.0, peak=100.0, pos_peak=130.0) == "trailing"
    assert run(fm, 112.0, peak=float("nan"), pos_peak=130.0) == "trailing"
    assert run(fm, 140.0, peak=100.0) is None  # last itself counts as peak
