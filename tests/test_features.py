from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from insiderbot import features
from insiderbot.config import Filters
from insiderbot.journal.store import Journal
from insiderbot.models import InsiderTxn, IssuerFeatures

ASOF = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)


def txn(**kw):
    d = dict(accession="a1", accepted_at=ASOF - timedelta(days=2), tx_date=date(2026, 10, 30),
             ticker="ABC", issuer_cik="1", issuer_name="ABC Inc", insider_cik="9", insider_name="Jo",
             is_officer=True, is_director=False, is_ten_pct=False, is_entity=False,
             officer_title="Chief Executive Officer", code="P", shares=1000.0, price=10.0,
             value=10_000.0, owned_after=11_000.0, is_10b5_1=False)
    d.update(kw)
    return InsiderTxn(**d)


def bars(n=260, start=100.0, end=70.0, vol=100_000):
    close = np.linspace(start, end, n)
    idx = pd.bdate_range(end=date(2026, 11, 2), periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": float(vol)}, index=idx)


class FakeMarket:
    def __init__(self, bars_=None, fund=None, info=None):
        self._b = bars_ if bars_ is not None else {"ABC": bars()}
        self._f = fund if fund is not None else {"ABC": {"market_cap": 1e9, "sector": "Tech"}}
        self._i = info if info is not None else {"ABC": {"tradable": True, "exchange": "NASDAQ", "name": "ABC Inc"}}

    def bars(self, symbols, start, end, now=None):
        return {s: self._b[s] for s in symbols if s in self._b}

    def fundamentals(self, symbol):
        return self._f.get(symbol, {})

    def asset_info(self, symbols):
        return {s: self._i[s] for s in symbols if s in self._i}


def build(txns, market=None, tmp_path=None, window=14):
    j = Journal(tmp_path / "j.db")
    return features.build(txns, market or FakeMarket(), j, ASOF, window), j


def test_build_basic(tmp_path):
    t2 = txn(accession="a2", insider_cik="8", insider_name="Al", is_officer=False, is_director=True,
             officer_title="", value=20_000.0, shares=2000.0, owned_after=12_000.0)
    (f,), _ = build([t2, txn()], tmp_path=tmp_path)
    assert f.ticker == "ABC" and f.n_buyers == 2 and f.total_value == 30_000.0
    assert f.has_ceo and not f.has_cfo and f.n_directors == 1
    assert f.accessions == ("a1", "a2")
    assert f.first_buy_months == 0  # no journal history: neutral
    assert f.price == 70.0 and f.drawdown_52w > 0.3 and f.ret_3m < 0
    assert f.adv20_usd > 5e6 and f.mcap_usd == 1e9 and f.exchange == "NASDAQ" and f.tradable
    # buyer 9: value 10k / (10k holdings*10) = 0.1; buyer 8: 20k / (10k*10) = 0.2
    assert abs(f.size_vs_holdings - 0.2) < 1e-9


def test_window_and_code_filter(tmp_path):
    old = txn(accession="old", accepted_at=ASOF - timedelta(days=30))
    sale = txn(accession="s", code="S")
    future = txn(accession="f", accepted_at=ASOF + timedelta(hours=1))
    assert build([old, sale, future], tmp_path=tmp_path)[0] == []


def test_size_capped_and_no_price_ignored(tmp_path):
    big = txn(owned_after=1000.0, shares=1000.0, value=1e6)
    (f,), _ = build([big], tmp_path=tmp_path)
    assert f.size_vs_holdings == 10
    (f,), _ = build([txn(price=0.0)], tmp_path=tmp_path)
    assert f.size_vs_holdings == 0


def test_first_buy_months_from_journal(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.add_txns([txn(accession="p", tx_date=date(2026, 4, 30),
                    accepted_at=datetime(2026, 5, 1, tzinfo=timezone.utc))], "live")
    (f,) = features.build([txn()], FakeMarket(), j, ASOF, 14)
    assert 5.5 < f.first_buy_months < 6.5


def test_roles_and_fund_like(tmp_path):
    t = txn(officer_title="Chief Financial Officer and Chairman")
    (f,), _ = build([t], tmp_path=tmp_path)
    assert f.has_cfo and f.has_chair and not f.has_ceo
    e = txn(is_entity=True, is_officer=False, is_director=False, is_ten_pct=True, officer_title="")
    (f,), _ = build([e], tmp_path=tmp_path)
    assert f.is_fund_like
    (f,), _ = build([e, txn(accession="x", insider_cik="7")], tmp_path=tmp_path)
    assert not f.is_fund_like


def test_10b5_fraction_value_weighted(tmp_path):
    a = txn(is_10b5_1=True, value=10_000.0)
    b = txn(accession="b", insider_cik="8", value=30_000.0)
    (f,), _ = build([a, b], tmp_path=tmp_path)
    assert abs(f.frac_10b5_1 - 0.25) < 1e-9


def test_missing_data_is_conservative(tmp_path):
    m = FakeMarket(bars_={}, fund={}, info={})
    (f,), _ = build([txn()], m, tmp_path)
    assert f.price is None and f.adv20_usd is None and f.mcap_usd is None
    assert f.ret_3m is None and f.drawdown_52w is None
    assert not f.tradable and f.exchange is None
    assert features.hard_filter(f, Filters()) == "not tradable"


def feat(**kw):
    d = dict(ticker="ABC", asof=ASOF, n_buyers=1, total_value=1e5, has_ceo=True, has_cfo=False,
             has_chair=False, n_directors=0, size_vs_holdings=1.0, first_buy_months=24, frac_10b5_1=0.0,
             ret_3m=0.0, drawdown_52w=0.2, price=20.0, adv20_usd=5e6, mcap_usd=1e9, sector="Tech",
             exchange="NASDAQ", is_fund_like=False, accessions=("a",), latest_accepted_at=ASOF,
             name="ABC Inc", tradable=True)
    d.update(kw)
    return IssuerFeatures(**d)


def test_hard_filter_reasons():
    c = Filters()
    assert features.hard_filter(feat(), c) is None
    assert features.hard_filter(feat(price=4.0), c) == "price<5"
    assert features.hard_filter(feat(price=None), c) == "price<5"
    assert features.hard_filter(feat(adv20_usd=1e6), c) == "adv<2M"
    assert features.hard_filter(feat(adv20_usd=None), c) == "adv<2M"
    assert features.hard_filter(feat(mcap_usd=1e8), c) == "mcap<300M"
    assert features.hard_filter(feat(mcap_usd=None), c) == "mcap unknown"
    assert features.hard_filter(feat(exchange="OTC"), c) == "exchange"
    assert features.hard_filter(feat(exchange=None), c) == "exchange"
    assert features.hard_filter(feat(tradable=False), c) == "not tradable"
    assert features.hard_filter(feat(is_fund_like=True), c) == "fund-only buyers"
    assert features.hard_filter(feat(sector=None, name="Foo Growth Fund"), c) == "etf/fund"
    assert features.hard_filter(feat(sector=None, name="Bar Acquisition Corp"), c) == "etf/fund"
    assert features.hard_filter(feat(sector="Tech", name="Foo Trust"), c) is None


def test_first_buy_capped_by_coverage(tmp_path):
    j = Journal(tmp_path / "j.db")
    # journal history starts 2026-08-01 (some other insider); our buyer has no prior buy
    j.add_txns([txn(accession="o", insider_cik="5", tx_date=date(2026, 7, 31),
                    accepted_at=datetime(2026, 8, 1, tzinfo=timezone.utc), ticker="ZZZ")], "live")
    (f,) = features.build([txn()], FakeMarket(), j, ASOF, 14)
    assert 2.8 < f.first_buy_months < 3.2  # ~3 months of coverage, not 24


def test_first_buy_ignores_future_prior(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.add_txns([txn(accession="p", tx_date=date(2026, 4, 30),
                    accepted_at=ASOF + timedelta(days=1))], "live")
    (f,) = features.build([txn()], FakeMarket(), j, ASOF, 14)
    assert f.first_buy_months == 0  # the only prior is in the future, so no history


def test_size_bad_holdings_contribute_zero(tmp_path):
    for bad in (0.0, -5.0, 500.0):  # missing, negative, below shares
        (f,), _ = build([txn(owned_after=bad, shares=1000.0)], tmp_path=tmp_path)
        assert f.size_vs_holdings == 0, bad


def test_size_new_holder_keeps_cap(tmp_path):
    (f,), _ = build([txn(owned_after=995.0, shares=1000.0, value=1e5)], tmp_path=tmp_path)
    assert f.size_vs_holdings == 10


def test_exchange_nyse_nasdaq_only():
    c = Filters()
    for ex in ("NYSE", "NASDAQ"):
        assert features.hard_filter(feat(exchange=ex), c) is None
    for ex in ("ARCA", "AMEX", "BATS"):
        assert features.hard_filter(feat(exchange=ex), c) == "exchange"


def test_etf_spac_name_checks():
    c = Filters()
    assert features.hard_filter(feat(name="NETFLIX INC"), c) is None
    assert features.hard_filter(feat(name="Prologis Trust"), c) is None
    assert features.hard_filter(feat(name="Foo Acquisition Corp II"), c) == "etf/fund"
    assert features.hard_filter(feat(name="SPDR S&P 500 ETF Trust"), c) == "etf/fund"
    assert features.hard_filter(feat(sector="Financial", name="Bar Income Fund"), c) == "etf/fund"


def test_adv_none_when_volume_unusable(tmp_path):
    b = bars()
    b["volume"] = float("nan")
    (f,), _ = build([txn()], FakeMarket(bars_={"ABC": b}), tmp_path)
    assert f.adv20_usd is None
    b2 = bars()
    b2.attrs["feed"] = "iex"
    (f,), _ = build([txn()], FakeMarket(bars_={"ABC": b2}), tmp_path)
    assert f.adv20_usd is None
