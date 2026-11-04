from datetime import datetime, timezone
from pathlib import Path

import pytest

from insiderbot.data.edgar import EdgarClient, EdgarError, dedupe_cofilers, parse_form4_xml

FIX = Path(__file__).parent / "fixtures"
ACC = datetime(2026, 10, 31, 14, 0, tzinfo=timezone.utc)


def load(name):
    return parse_form4_xml((FIX / name).read_text(), "0001-26-000001", ACC)


def test_officer_buy_parsed():
    (t,) = load("form4_officer_buy.xml")
    assert t.code == "P" and t.is_officer and not t.is_entity
    assert t.ticker == "ACME" and t.officer_title == "Chief Executive Officer"
    assert t.shares == 1000 and t.price == 10.5 and t.value == 10500
    assert t.owned_after == 5000 and t.tx_date.isoformat() == "2026-10-30"
    assert t.accepted_at == ACC and not t.is_10b5_1


def test_10b5_1_flag():
    (t,) = load("form4_director_10b5_1.xml")
    assert t.is_director and t.is_10b5_1


def test_entity_owner_flag():
    (t,) = load("form4_entity_10pct.xml")
    assert t.is_ten_pct and t.is_entity


def test_sale_kept_as_S():
    (t,) = load("form4_sale.xml")
    assert t.code == "S"


def test_derivative_only_empty():
    assert load("form4_derivative_only.xml") == []


def test_zero_price_dropped():
    assert load("form4_zero_price.xml") == []


def test_amendment_ignored():
    assert load("form4_amendment.xml") == []


def test_cofiler_dedupe():
    rows = load("form4_cofiler.xml")
    assert len(rows) == 2
    out = dedupe_cofilers(rows[::-1])
    assert len(out) == 1 and not out[0].is_entity


def test_missing_identity_raises(tmp_path):
    with pytest.raises(EdgarError, match="EDGAR_IDENTITY not set"):
        EdgarClient("", tmp_path)


def test_doctype_rejected():
    assert load("form4_doctype_bomb.xml") == []


class _Hdr:
    def __init__(self, dt):
        self.acceptance_datetime = dt


class _Filing:
    def __init__(self, dt, form="4"):
        self.header = _Hdr(dt)
        self.form = form
        self.accession_no = "0001-26-000002"
        self.filing_date = dt.date()

    def xml(self):
        return (FIX / "form4_officer_buy.xml").read_text()


def _client(tmp_path):
    return EdgarClient("test test@example.invalid", tmp_path)


@pytest.mark.parametrize(
    "naive,expected",
    [
        (datetime(2026, 7, 1, 16, 30), datetime(2026, 7, 1, 20, 30, tzinfo=timezone.utc)),
        (datetime(2026, 12, 1, 16, 30), datetime(2026, 12, 1, 21, 30, tzinfo=timezone.utc)),
    ],
)
def test_accepted_at_eastern_to_utc(tmp_path, naive, expected):
    assert _client(tmp_path)._accepted_at(_Filing(naive)) == expected


def test_fetch_window_uses_eastern(tmp_path, monkeypatch):
    import edgar

    f = _Filing(datetime(2026, 7, 1, 16, 30))  # 20:30 UTC
    monkeypatch.setattr(edgar, "set_identity", lambda i: None)
    monkeypatch.setattr(edgar, "get_filings", lambda **kw: [f])
    monkeypatch.setattr("insiderbot.data.edgar.MIN_INTERVAL", 0)
    since = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    got = _client(tmp_path).fetch(since, datetime(2026, 7, 1, 21, 0, tzinfo=timezone.utc))
    assert len(got) == 1 and got[0].accepted_at.hour == 20
    assert _client(tmp_path).fetch(since, datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)) == []


@pytest.mark.parametrize("sym,ok", [("ACME", True), (" brk.b ", True), ("BF-A", True),
                                    ("ACME CORP", False), ("N/A", False), ("TOOLONG", False),
                                    ("AB1", False), ("A.BCD", False)])
def test_malformed_ticker_dropped(sym, ok):
    xml = (FIX / "form4_officer_buy.xml").read_text().replace(
        "<issuerTradingSymbol>ACME</issuerTradingSymbol>", f"<issuerTradingSymbol>{sym}</issuerTradingSymbol>")
    assert sym in xml
    stats = {}
    out = parse_form4_xml(xml, "0001-26-000001", ACC, stats)
    assert bool(out) is ok
    assert stats.get("bad_ticker", 0) == (0 if ok else 1)
    if ok:
        assert out[0].ticker == sym.strip().upper()


# ----------------------------------------------------------------------------- current (live) feed
class _Page:
    """A fake edgartools CurrentFilings page: iterable, newest first, .next() -> next page or None."""

    def __init__(self, filings, nxt=None):
        self.filings, self.nxt, self.next_calls = filings, nxt, 0

    def __iter__(self):
        return iter(self.filings)

    def next(self):
        self.next_calls += 1
        return self.nxt


def _f(dt, acc, form="4"):
    f = _Filing(dt, form)
    f.accession_no = acc
    return f


def _current(monkeypatch, first_page):
    import edgar

    calls = []
    monkeypatch.setattr(edgar, "set_identity", lambda i: None)
    monkeypatch.setattr(edgar, "get_current_filings", lambda **kw: calls.append(kw) or first_page)
    monkeypatch.setattr("insiderbot.data.edgar.MIN_INTERVAL", 0)
    return calls


def test_fetch_current_stops_at_since(tmp_path, monkeypatch):
    # naive ET times: 11:00, 10:30 | 10:00, 09:00 (older than since) | never reached
    p3 = _Page([_f(datetime(2026, 11, 4, 8, 0), "a5")])
    p2 = _Page([_f(datetime(2026, 11, 4, 10, 0), "a3"), _f(datetime(2026, 11, 4, 9, 0), "a4")], p3)
    p1 = _Page([_f(datetime(2026, 11, 4, 11, 0), "a1"), _f(datetime(2026, 11, 4, 10, 30), "a2")], p2)
    calls = _current(monkeypatch, p1)
    since = datetime(2026, 11, 4, 14, 45, tzinfo=timezone.utc)   # 09:45 ET
    until = datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc)
    got = _client(tmp_path).fetch_current(since, until)
    assert sorted({t.accession for t in got}) == ["a1", "a2", "a3"]
    assert calls and calls[0]["form"] == "4"
    assert p2.next_calls == 0                                     # stopped at the first filing older than since


def test_fetch_current_same_txns_as_fetch_for_same_filing(tmp_path, monkeypatch):
    import edgar

    dt = datetime(2026, 11, 4, 11, 0)
    _current(monkeypatch, _Page([_f(dt, "0001-26-000002")]))
    monkeypatch.setattr(edgar, "get_filings", lambda **kw: [_f(dt, "0001-26-000002")])
    since = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
    until = datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc)
    c = _client(tmp_path)
    assert c.fetch_current(since, until) == c.fetch(since, until) != []


def test_fetch_current_skips_amendments_and_future(tmp_path, monkeypatch):
    _current(monkeypatch, _Page([_f(datetime(2026, 11, 4, 12, 30), "fut"),
                                 _f(datetime(2026, 11, 4, 11, 0), "amd", form="4/A"),
                                 _f(datetime(2026, 11, 4, 10, 0), "ok")]))
    got = _client(tmp_path).fetch_current(datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc),
                                          datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc))
    assert {t.accession for t in got} == {"ok"}


def test_out_of_window_filings_do_not_pause(tmp_path, monkeypatch):
    """The daily index spans thousands of Form 4s: only in-window ones may pay the rate-limit pause."""
    import edgar

    slept = []
    monkeypatch.setattr(edgar, "set_identity", lambda i: None)
    monkeypatch.setattr(edgar, "get_filings", lambda **kw: [_f(datetime(2026, 11, 3, 9, 0), f"old{i}")
                                                            for i in range(50)])
    monkeypatch.setattr("insiderbot.data.edgar.MIN_INTERVAL", 1.0)
    monkeypatch.setattr("insiderbot.data.edgar.time.sleep", lambda s: slept.append(s))
    got = _client(tmp_path).fetch(datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc),
                                  datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc))
    assert got == [] and slept == []


def test_fetch_current_reports_whether_it_reached_since(tmp_path, monkeypatch):
    p1 = _Page([_f(datetime(2026, 11, 4, 11, 0), "a1")])                   # feed ends before since
    _current(monkeypatch, p1)
    c = _client(tmp_path)
    c.fetch_current(datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc), datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc))
    assert c.last_reached_since is False
    p2 = _Page([_f(datetime(2026, 11, 4, 11, 0), "a1"), _f(datetime(2026, 11, 4, 6, 0), "old")])
    _current(monkeypatch, p2)
    c.fetch_current(datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc), datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc))
    assert c.last_reached_since is True
