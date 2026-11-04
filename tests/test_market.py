from datetime import datetime, timezone

import insiderbot.data.market as market
from insiderbot.data.market import MarketData


def test_fundamentals_failsoft(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("net down")

    monkeypatch.setattr(market.yf, "Ticker", Boom)
    md = MarketData(None, None, tmp_path)
    assert md.fundamentals("ABC") == {}


def test_fundamentals_cached(tmp_path, monkeypatch):
    calls = []

    class T:
        def __init__(self, s):
            calls.append(s)
            self.info = {"marketCap": 5, "sector": "Tech", "trailingPE": 12.0}

    monkeypatch.setattr(market.yf, "Ticker", T)
    md = MarketData(None, None, tmp_path)
    a = md.fundamentals("ABC")
    assert a["market_cap"] == 5 and a["sector"] == "Tech"
    assert md.fundamentals("ABC") == a
    assert len(calls) == 1


def test_calendar_today_none_without_client(tmp_path):
    md = MarketData(None, None, tmp_path)
    assert md.calendar_today(datetime(2026, 11, 3, 15, tzinfo=timezone.utc)) is None


def test_fundamentals_empty_not_cached(tmp_path, monkeypatch):
    class T:
        def __init__(self, s):
            self.info = {}

    monkeypatch.setattr(market.yf, "Ticker", T)
    md = MarketData(None, None, tmp_path)
    assert md.fundamentals("ABC") == {}
    assert list(tmp_path.glob("fund_*")) == []


def test_fundamentals_symbol_sanitised(tmp_path, monkeypatch):
    class T:
        def __init__(self, s):
            self.info = {"sector": "x"}

    monkeypatch.setattr(market.yf, "Ticker", T)
    MarketData(None, None, tmp_path).fundamentals("../evil/sym")
    assert [p.name for p in tmp_path.iterdir()] == ["fund_.._EVIL_SYM.json"]


def test_bars_drops_partial_today(tmp_path):
    import pandas as pd
    from datetime import date

    idx = pd.MultiIndex.from_product(
        [["ABC"], pd.to_datetime(["2026-11-02 05:00", "2026-11-03 05:00"], utc=True)])
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                      index=idx)

    class Stub:
        def get_stock_bars(self, req):
            return type("R", (), {"df": df})()

    md = MarketData(None, None, tmp_path)
    md.data = Stub()
    during = datetime(2026, 11, 3, 15, tzinfo=timezone.utc)  # 10:00 ET
    after = datetime(2026, 11, 3, 22, tzinfo=timezone.utc)  # 17:00 ET
    assert len(md.bars(["ABC"], date(2026, 11, 1), date(2026, 11, 3), during)["ABC"]) == 1
    assert len(md.bars(["ABC"], date(2026, 11, 1), date(2026, 11, 3), after)["ABC"]) == 2


def _bars_stub(df, calls, fail_sip=False):
    class Stub:
        def get_stock_bars(self, req):
            calls.append(req)
            if fail_sip and req.feed.value == "sip":
                raise RuntimeError("subscription")
            return type("R", (), {"df": df})()
    return Stub()


def _two_day_df():
    import pandas as pd
    idx = pd.MultiIndex.from_product(
        [["ABC"], pd.to_datetime(["2026-11-02 05:00", "2026-11-03 05:00"], utc=True)])
    return pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1000}, index=idx)


def test_bars_uses_sip_and_caps_end(tmp_path):
    from datetime import date, timedelta
    calls = []
    md = MarketData(None, None, tmp_path)
    md.data = _bars_stub(_two_day_df(), calls)
    now = datetime(2026, 11, 3, 15, tzinfo=timezone.utc)
    out = md.bars(["ABC"], date(2026, 11, 1), date(2026, 11, 3), now)
    assert calls[0].feed.value == "sip"
    assert calls[0].end.replace(tzinfo=timezone.utc) == now - timedelta(minutes=16)
    assert out["ABC"]["volume"].iloc[0] == 1000
    # 16:05 ET is still inside the delay window: today's bar stays dropped
    near = datetime(2026, 11, 3, 21, 5, tzinfo=timezone.utc)
    assert len(md.bars(["ABC"], date(2026, 11, 1), date(2026, 11, 3), near)["ABC"]) == 1


def test_bars_iex_fallback_nans_volume(tmp_path):
    from datetime import date
    calls = []
    md = MarketData(None, None, tmp_path)
    md.data = _bars_stub(_two_day_df(), calls, fail_sip=True)
    after = datetime(2026, 11, 3, 22, tzinfo=timezone.utc)
    d = md.bars(["ABC"], date(2026, 11, 1), date(2026, 11, 3), after)["ABC"]
    assert [c.feed.value for c in calls] == ["sip", "iex"]
    assert d["volume"].isna().all() and d.attrs["feed"] == "iex"


def test_trading_days_cached_and_sorted(tmp_path):
    from datetime import date
    from types import SimpleNamespace

    md = MarketData(None, None, tmp_path)
    calls = []

    class T:
        def get_calendar(self, req):
            calls.append(req)
            return [SimpleNamespace(date=date(2026, 11, 3)), SimpleNamespace(date=date(2026, 11, 2)),
                    SimpleNamespace(date=date(2026, 10, 1))]

    md.trading = T()
    d = md.trading_days(date(2026, 11, 1), date(2026, 11, 3))
    assert d == [date(2026, 11, 2), date(2026, 11, 3)]
    assert md.trading_days(date(2026, 11, 1), date(2026, 11, 3)) == d and len(calls) == 1


def test_trading_days_raises_without_client(tmp_path):
    import pytest
    from datetime import date

    with pytest.raises(RuntimeError):
        MarketData(None, None, tmp_path).trading_days(date(2026, 11, 1), date(2026, 11, 3))


def test_bars_bad_symbol_does_not_lose_others(tmp_path):
    from datetime import date
    calls = []
    good = _two_day_df()

    class Stub:
        def get_stock_bars(self, req):
            calls.append(list(req.symbol_or_symbols))
            if "BAD" in req.symbol_or_symbols:
                raise RuntimeError("invalid symbol BAD")
            return type("R", (), {"df": good})()

    md = MarketData(None, None, tmp_path)
    md.data = Stub()
    after = datetime(2026, 11, 3, 22, tzinfo=timezone.utc)
    out = md.bars(["ABC", "BAD"], date(2026, 11, 1), date(2026, 11, 3), after)
    assert set(out) == {"ABC"} and out["ABC"].attrs["feed"] == "sip"
    assert ["ABC"] in calls and ["BAD"] in calls
