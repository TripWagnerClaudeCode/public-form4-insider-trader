from datetime import date, datetime, timedelta, timezone

from insiderbot.journal.store import Journal
from insiderbot.models import AgentDecision, InsiderTxn, IssuerFeatures


def txn(**kw):
    d = dict(accession="0001-23-1", accepted_at=datetime(2026, 11, 2, 10, 0), tx_date=date(2026, 10, 30),
             ticker="ABC", issuer_cik="1", issuer_name="ABC Inc", insider_cik="9", insider_name="Jo",
             is_officer=True, is_director=False, is_ten_pct=False, is_entity=False, officer_title="CEO",
             code="P", shares=100.0, price=10.0, value=1000.0, owned_after=500.0, is_10b5_1=False)
    d.update(kw)
    return InsiderTxn(**d)


def feats():
    return IssuerFeatures(ticker="ABC", asof=datetime(2026, 11, 2), n_buyers=1, total_value=1000.0, has_ceo=True,
                          has_cfo=False, has_chair=False, n_directors=0, size_vs_holdings=0.2, first_buy_months=12.0,
                          frac_10b5_1=0.0, ret_3m=None, drawdown_52w=None, price=10.0, adv20_usd=None,
                          mcap_usd=None, sector=None, exchange=None, is_fund_like=False,
                          accessions=("0001-23-1",), latest_accepted_at=datetime(2026, 11, 2, 10, 0))


def test_txn_dedupe(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.add_txns([txn()], "live") == 1
    assert j.add_txns([txn()], "live") == 0
    assert len(j.txns_since(datetime(2026, 1, 1))) == 1
    assert len(j.insider_prior_buys("9", "ABC", date(2026, 12, 1))) == 1


def test_watermark_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.get_watermark() is None
    ts = datetime(2026, 11, 2, 10, 30, tzinfo=timezone.utc)
    j.set_watermark(ts)
    assert j.get_watermark() == ts


def test_candidate_and_outcome(tmp_path):
    j = Journal(tmp_path / "j.db")
    run = j.start_run("scan", "paper", "v1")
    cid = j.add_candidate(run, feats(), {"v1": 55.0}, ["x"], None, "live")
    j.add_decision(run, cid, AgentDecision("ABC", "buy", 70, "t", "c", {}), 0.5)
    assert cid in [r["candidate_id"] for r in j.pending_outcomes(10)]
    j.set_outcome(cid, 10, 0.05, 0.02)
    assert cid not in [r["candidate_id"] for r in j.pending_outcomes(10)]


def test_month_cost(tmp_path):
    j = Journal(tmp_path / "j.db")
    run = j.start_run("scan", "paper", "v1")
    j.add_cost(run, 1.5)
    j.add_cost(run, 2.0)
    now = datetime.now(timezone.utc)
    assert j.month_cost_usd(now.year, now.month) == 3.5
    assert j.month_cost_usd(2000, 1) == 0.0


def test_datetime_normalised_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.db")
    est = timezone(timedelta(hours=-5))
    aware_est = datetime(2026, 11, 2, 5, 0, tzinfo=est)  # 10:00 UTC
    naive_utc = datetime(2026, 11, 2, 10, 0)
    utc = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    j.set_watermark(aware_est)
    assert j.get_watermark() == utc and j.get_watermark().tzinfo is not None
    j.set_watermark(naive_utc)
    assert j.get_watermark() == utc
    j.add_txns([txn(accession="a1", accepted_at=aware_est), txn(accession="a2", accepted_at=naive_utc)], "live")
    got = j.txns_since(utc)
    assert len(got) == 2 and all(t.accepted_at == utc for t in got)
    assert j.txns_since(utc + timedelta(seconds=1)) == []
    assert len(j.txns_since(datetime(2026, 11, 2, 5, 0, tzinfo=est))) == 2
    j.add_txns([txn(accession="a3", accepted_at=datetime(2026, 11, 2, 4, 30, tzinfo=est))], "live")  # 09:30Z
    assert len(j.txns_since(utc)) == 2
    assert len(j.txns_since(datetime(2026, 11, 2, 9, 0))) == 3


def test_dedupe_float_drift_and_none(tmp_path):
    j = Journal(tmp_path / "j.db")
    assert j.add_txns([txn(price=10.0000001, shares=100.0)], "live") == 1
    assert j.add_txns([txn(price=10.0000002, shares=100.00000001)], "live") == 0
    assert j.add_txns([txn(accession="n1", officer_title=None)], "live") == 1
    assert j.add_txns([txn(accession="n1", officer_title=None)], "live") == 0


def test_decision_actions_ignore_hold_and_exit(tmp_path):
    j = Journal(tmp_path / "j.db")
    run = j.start_run("scan", "paper", "v001")
    cid = j.add_candidate(run, feats(), {}, [], None, "live")
    j.add_decision(run, cid, AgentDecision("ABC", "buy", 4, "t", "m"), 0.0)
    j.add_decision(run, cid, AgentDecision("ABC", "hold", 3, "t", "m"), 0.0)
    j.add_decision(run, cid, AgentDecision("ABC", "exit", 3, "t", "m"), 0.0)
    assert j.decision_actions() == {cid: "buy"}
