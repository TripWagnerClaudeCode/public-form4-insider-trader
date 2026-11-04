"""SQLite journal (WAL). All timestamps are stored as ISO-8601 strings."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from insiderbot.models import AgentDecision, InsiderTxn, IssuerFeatures, SubmittedOrder

_FEATURE_DTS = ("asof", "latest_accepted_at")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, job TEXT, mode TEXT, formula_version TEXT,
  started_at TEXT, finished_at TEXT, status TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS txns(
  id INTEGER PRIMARY KEY, accession TEXT, accepted_at TEXT, tx_date TEXT, ticker TEXT,
  issuer_cik TEXT, issuer_name TEXT, insider_cik TEXT, insider_name TEXT,
  is_officer INT, is_director INT, is_ten_pct INT, is_entity INT, officer_title TEXT,
  code TEXT, shares REAL, price REAL, value REAL, owned_after REAL, is_10b5_1 INT,
  source TEXT,
  UNIQUE(accession, insider_cik, tx_date, code, shares, price));
CREATE INDEX IF NOT EXISTS ix_txns_accepted ON txns(accepted_at);
CREATE INDEX IF NOT EXISTS ix_txns_insider ON txns(insider_cik, ticker, tx_date);
CREATE TABLE IF NOT EXISTS candidates(
  id INTEGER PRIMARY KEY, run_id INT, ticker TEXT, asof TEXT, features TEXT, scores TEXT,
  flags TEXT, filter_reason TEXT, source TEXT, price REAL);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY, run_id INT, candidate_id INT, ticker TEXT, action TEXT,
  conviction INT, thesis TEXT, change_mind TEXT, reports TEXT, cost_usd REAL);
CREATE TABLE IF NOT EXISTS orders(
  id INTEGER PRIMARY KEY, run_id INT, symbol TEXT, side TEXT, qty INT, reason TEXT,
  ref_price REAL, broker_id TEXT, status TEXT, error TEXT, fill_price REAL, filled_qty REAL,
  created_at TEXT);
CREATE TABLE IF NOT EXISTS position_meta(
  symbol TEXT PRIMARY KEY, entry_date TEXT, entry_price REAL, formula_version TEXT,
  candidate_id INT, peak_price REAL, exit_date TEXT, exit_price REAL, exit_reason TEXT);
CREATE TABLE IF NOT EXISTS equity(ts TEXT PRIMARY KEY, equity REAL);
CREATE TABLE IF NOT EXISTS outcomes(
  candidate_id INT, horizon INT, ret REAL, excess REAL, PRIMARY KEY(candidate_id, horizon));
CREATE TABLE IF NOT EXISTS costs(
  id INTEGER PRIMARY KEY, run_id INT, ts TEXT, year INT, month INT, usd REAL);
CREATE TABLE IF NOT EXISTS formula_events(
  id INTEGER PRIMARY KEY, ts TEXT, version TEXT, event TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, day TEXT, ts TEXT, symbol TEXT, kind TEXT, detail TEXT, analyst TEXT,
  reviewer TEXT, action TEXT, price REAL, run_id INT, mode TEXT, filing_url TEXT);
"""

TXN_COLS = (
    "accession accepted_at tx_date ticker issuer_cik issuer_name insider_cik insider_name "
    "is_officer is_director is_ten_pct is_entity officer_title code shares price value "
    "owned_after is_10b5_1"
).split()
FLOATS = {"shares", "price", "value", "owned_after"}
BOOLS = {"is_officer", "is_director", "is_ten_pct", "is_entity", "is_10b5_1"}
FORMULA_EVENTS = {"proposed", "shadow", "promoted", "retired", "rejected"}

_INSERT_TXN = (
    f"INSERT OR IGNORE INTO txns({','.join(TXN_COLS)}, source) "
    f"VALUES({','.join('?' * (len(TXN_COLS) + 1))})"
)


def _utc(dt: datetime) -> datetime:
    """Aware UTC datetime; naive input is treated as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ts(dt: datetime) -> str:
    return _utc(dt).isoformat(timespec="seconds")


def _parse(s: str) -> datetime:
    return _utc(datetime.fromisoformat(s))


def _now() -> str:
    return _ts(datetime.now(timezone.utc))


def _row_to_txn(r: sqlite3.Row) -> InsiderTxn:
    d = {c: r[c] for c in TXN_COLS}
    d["accepted_at"] = _parse(d["accepted_at"])
    d["tx_date"] = date.fromisoformat(d["tx_date"])
    for b in BOOLS:
        d[b] = bool(d[b])
    return InsiderTxn(**d)


class Journal:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    # runs
    def start_run(self, job: str, mode: str, formula_version: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(job, mode, formula_version, started_at, status) VALUES(?,?,?,?,?)",
            (job, mode, formula_version, _now(), "running"),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, summary: dict) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, summary=? WHERE id=?",
            (_now(), status, json.dumps(summary, default=str), run_id),
        )
        self.conn.commit()

    def last_run(self, job: str, status: str = "ok") -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE job=? AND status=? ORDER BY id DESC LIMIT 1", (job, status)
        ).fetchone()

    # watermark
    def get_watermark(self) -> datetime | None:
        r = self.conn.execute("SELECT value FROM meta WHERE key='watermark'").fetchone()
        return _parse(r["value"]) if r else None

    def set_watermark(self, ts: datetime) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('watermark', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_ts(ts),),
        )
        self.conn.commit()

    # txns (accepted_at must use a consistent tz convention: string comparison)
    def add_txns(self, txns: list[InsiderTxn], source: str) -> int:
        n = 0
        for t in txns:
            vals = []
            for c in TXN_COLS:
                v = getattr(t, c)
                if c == "accepted_at":
                    v = _ts(v)
                elif isinstance(v, date):
                    v = v.isoformat()
                elif c in BOOLS:
                    v = int(bool(v))
                elif c in FLOATS:
                    v = round(float(v or 0.0), 6)
                elif v is None:
                    v = ""
                vals.append(v)
            n += self.conn.execute(_INSERT_TXN, (*vals, source)).rowcount
        self.conn.commit()
        return n

    def txns_since(self, since: datetime, codes: tuple[str, ...] = ("P",)) -> list[InsiderTxn]:
        q = ",".join("?" * len(codes))
        rows = self.conn.execute(
            f"SELECT * FROM txns WHERE accepted_at >= ? AND code IN ({q}) ORDER BY accepted_at",
            (_ts(since), *codes),
        ).fetchall()
        return [_row_to_txn(r) for r in rows]

    def insider_prior_buys(self, insider_cik: str, ticker: str, before: date) -> list[InsiderTxn]:
        rows = self.conn.execute(
            "SELECT * FROM txns WHERE insider_cik=? AND ticker=? AND code='P' AND tx_date < ? "
            "ORDER BY tx_date",
            (insider_cik, ticker, before.isoformat()),
        ).fetchall()
        return [_row_to_txn(r) for r in rows]

    # candidates / decisions / orders
    def add_candidate(self, run_id: int, f: IssuerFeatures, scores: dict[str, float],
                      flags: list[str], filter_reason: str | None, source: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO candidates(run_id, ticker, asof, features, scores, flags, filter_reason, source, price) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, f.ticker, _ts(f.asof), json.dumps(dataclasses.asdict(f), default=str),
             json.dumps(scores), json.dumps(flags), filter_reason, source, f.price),
        )
        self.conn.commit()
        return cur.lastrowid

    def record_event(self, day, symbol: str, kind: str, detail: str, analyst, reviewer, action: str,
                     price: float | None, run_id: int, mode: str, filing_url: str | None = None) -> int:
        """Phase 3 intraday Event Review row (held names)."""
        cur = self.conn.execute(
            "INSERT INTO events(day, ts, symbol, kind, detail, analyst, reviewer, action, price, run_id, mode, "
            "filing_url) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(day), _now(), symbol, kind, detail, None if analyst is None else json.dumps(analyst, default=str),
             None if reviewer is None else json.dumps(reviewer, default=str), action, price, run_id, mode,
             filing_url))
        self.conn.commit()
        return int(cur.lastrowid)

    def events_today(self, day, mode: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM events WHERE day=? AND mode=? ORDER BY id",
                                                   (str(day), mode))]

    def add_decision(self, run_id: int, candidate_id: int, d: AgentDecision, cost_usd: float) -> None:
        self.conn.execute(
            "INSERT INTO decisions(run_id, candidate_id, ticker, action, conviction, thesis, "
            "change_mind, reports, cost_usd) VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, candidate_id, d.ticker, d.action, d.conviction, d.thesis, d.change_mind,
             json.dumps(d.reports, default=str), cost_usd),
        )
        self.conn.commit()

    def add_order(self, run_id: int, o: SubmittedOrder) -> int:
        p = o.plan
        cur = self.conn.execute(
            "INSERT INTO orders(run_id, symbol, side, qty, reason, ref_price, broker_id, status, "
            "error, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (run_id, p.symbol, p.side, p.qty, p.reason, p.ref_price, o.broker_id, o.status,
             o.error, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_order(self, order_id: int, status: str, fill_price: float | None,
                     filled_qty: float | None) -> None:
        self.conn.execute(
            "UPDATE orders SET status=?, fill_price=?, filled_qty=? WHERE id=?",
            (status, fill_price, filled_qty, order_id),
        )
        self.conn.commit()

    # position meta
    def add_position_meta(self, symbol: str, entry_date: date, entry_price: float,
                          formula_version: str, candidate_id: int | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO position_meta(symbol, entry_date, entry_price, formula_version, "
            "candidate_id, peak_price) VALUES(?,?,?,?,?,?)",
            (symbol, entry_date.isoformat(), entry_price, formula_version, candidate_id, entry_price),
        )
        self.conn.commit()

    def position_meta(self, symbol: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM position_meta WHERE symbol=? AND exit_date IS NULL", (symbol,)
        ).fetchone()

    def close_position_meta(self, symbol: str, exit_date: date, exit_price: float, reason: str) -> None:
        self.conn.execute(
            "UPDATE position_meta SET exit_date=?, exit_price=?, exit_reason=? "
            "WHERE symbol=? AND exit_date IS NULL",
            (exit_date.isoformat(), exit_price, reason, symbol),
        )
        self.conn.commit()

    def update_peak(self, symbol: str, price: float) -> None:
        self.conn.execute(
            "UPDATE position_meta SET peak_price=MAX(COALESCE(peak_price, 0), ?) "
            "WHERE symbol=? AND exit_date IS NULL",
            (price, symbol),
        )
        self.conn.commit()

    # equity
    def add_equity(self, ts: datetime, equity: float) -> None:
        self.conn.execute("INSERT OR REPLACE INTO equity(ts, equity) VALUES(?,?)", (_ts(ts), equity))
        self.conn.commit()

    def peak_equity(self) -> float | None:
        return self.conn.execute("SELECT MAX(equity) AS m FROM equity").fetchone()["m"]

    # outcomes
    def pending_outcomes(self, horizon: int) -> list[sqlite3.Row]:
        """Candidates with no outcome yet at this horizon (caller checks maturity)."""
        return self.conn.execute(
            "SELECT c.id AS candidate_id, c.ticker, c.asof, c.price, c.source FROM candidates c "
            "WHERE NOT EXISTS (SELECT 1 FROM outcomes o WHERE o.candidate_id=c.id AND o.horizon=?) "
            "ORDER BY c.id",
            (horizon,),
        ).fetchall()

    def set_outcome(self, candidate_id: int, horizon: int, ret: float, excess: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO outcomes(candidate_id, horizon, ret, excess) VALUES(?,?,?,?)",
            (candidate_id, horizon, ret, excess),
        )
        self.conn.commit()

    # cost
    def add_cost(self, run_id: int, usd: float) -> None:
        now = datetime.now(timezone.utc)
        self.conn.execute(
            "INSERT INTO costs(run_id, ts, year, month, usd) VALUES(?,?,?,?,?)",
            (run_id, now.isoformat(), now.year, now.month, usd),
        )
        self.conn.commit()

    def month_cost_usd(self, year: int, month: int) -> float:
        r = self.conn.execute(
            "SELECT COALESCE(SUM(usd), 0) AS s FROM costs WHERE year=? AND month=?", (year, month)
        ).fetchone()
        return float(r["s"])

    # formula events
    def add_formula_event(self, version: str, event: str, detail: dict) -> None:
        if event not in FORMULA_EVENTS:
            raise ValueError(f"bad formula event: {event}")
        self.conn.execute(
            "INSERT INTO formula_events(ts, version, event, detail) VALUES(?,?,?,?)",
            (_now(), version, event, json.dumps(detail, default=str)),
        )
        self.conn.commit()

    # research reads
    def candidates(self) -> list[dict]:
        """All candidates with parsed features/scores/flags. Rows whose features cannot be
        parsed are skipped (never fabricated)."""
        out = []
        for r in self.conn.execute("SELECT * FROM candidates ORDER BY id").fetchall():
            try:
                d = json.loads(r["features"])
                for k in _FEATURE_DTS:
                    d[k] = _parse(d[k])
                d["accessions"] = tuple(d.get("accessions") or ())
                f = IssuerFeatures(**d)
                out.append({
                    "id": r["id"], "run_id": r["run_id"], "ticker": r["ticker"],
                    "asof": _parse(r["asof"]), "features": f,
                    "scores": json.loads(r["scores"] or "{}"), "flags": json.loads(r["flags"] or "[]"),
                    "filter_reason": r["filter_reason"] or "", "source": r["source"] or "",
                    "price": r["price"],
                })
            except (ValueError, TypeError, KeyError):
                continue
        return out

    def outcomes_all(self) -> dict[int, dict[int, tuple[float, float]]]:
        res: dict[int, dict[int, tuple[float, float]]] = {}
        for r in self.conn.execute("SELECT candidate_id, horizon, ret, excess FROM outcomes"):
            res.setdefault(r["candidate_id"], {})[r["horizon"]] = (r["ret"], r["excess"])
        return res

    def decision_actions(self) -> dict[int, str]:
        """Latest buy/pass agent action per candidate id (held-name exit checks, recorded as
        hold/exit, never overwrite the entry review)."""
        return {r["candidate_id"]: r["action"] for r in self.conn.execute(
            "SELECT candidate_id, action FROM decisions WHERE action IN ('buy','pass') ORDER BY id")}

    def closed_positions(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM position_meta WHERE exit_date IS NOT NULL ORDER BY exit_date")]

    def formula_events(self, version: str | None = None, event: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM formula_events WHERE 1=1", []
        if version is not None:
            q += " AND version=?"
            args.append(version)
        if event is not None:
            q += " AND event=?"
            args.append(event)
        rows = self.conn.execute(q + " ORDER BY id", args).fetchall()
        return [{"id": r["id"], "ts": _parse(r["ts"]), "version": r["version"], "event": r["event"],
                 "detail": json.loads(r["detail"] or "{}")} for r in rows]

    def total_cost_usd(self) -> float:
        return float(self.conn.execute("SELECT COALESCE(SUM(usd), 0) AS s FROM costs").fetchone()["s"])
