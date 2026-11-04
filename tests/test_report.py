from insiderbot.journal import report


def test_scan_report_contains_counts(tmp_path):
    run = {"id": 7, "job": "scan", "mode": "paper", "formula_version": "v1",
           "started_at": "2026-11-03T15:00:00+00:00"}
    cands = [{"ticker": "ABC", "score": 71.5}, {"ticker": "XYZ", "score": 44.0}]
    decs = [{"ticker": "ABC", "action": "buy", "confidence": 80, "rationale": "cluster"}]
    orders = [{"symbol": "ABC", "side": "buy", "qty": 10, "status": "filled"}]
    p = report.write_scan(tmp_path / "r" / "scan.md", run, cands, decs, orders, [])
    text = p.read_bytes().decode()
    assert "Candidates: 2" in text and "Decisions: 1" in text and "Orders: 1" in text
    assert "ABC" in text and "\r" not in text
    assert "10:00 EST" in text  # rendered in New York time


def test_daily_and_weekly(tmp_path):
    d = report.write_daily(tmp_path / "d.md", {"job": "reconcile"}, [], [],
                           [{"symbol": "Q", "reason": "stop"}])
    assert "Exits: 1" in d.read_text()
    w = report.write_weekly(tmp_path / "w.md", {"matured": 12}, "- a\n+ b", "lesson")
    t = w.read_text()
    assert "matured" in t and "lesson" in t and "+ b" in t


def test_daily_empty_and_equity(tmp_path):
    t = report.write_daily(tmp_path / "e.md", {"job": "reconcile"}, [], [], []).read_text()
    assert t.count("_none_") == 3 and "Equity" not in t
    t2 = report.write_daily(tmp_path / "e2.md", {}, [], [], [], equity=101234.5).read_text()
    assert "Equity: $101,234.50" in t2


def test_write_atomic_leaves_no_temp(tmp_path):
    p = report.write_weekly(tmp_path / "w.md", {}, "", "")
    assert [f.name for f in tmp_path.iterdir()] == ["w.md"] and p.exists()
