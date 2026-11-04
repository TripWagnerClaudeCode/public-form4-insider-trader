"""Market data: Alpaca (IEX) bars/quotes/news/assets, yfinance fundamentals."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

NY = ZoneInfo("America/New_York")
CACHE_SECONDS = 86400
SIP_DELAY = timedelta(minutes=16)  # free plan serves delayed SIP only
log = logging.getLogger(__name__)


class MarketData:
    def __init__(self, key: str | None, secret: str | None, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if key and secret:
            self.data = StockHistoricalDataClient(key, secret)
            self.news_client = NewsClient(key, secret)
            self.trading = TradingClient(key, secret, paper=True)
        else:
            self.data = self.news_client = self.trading = None

    def bars(self, symbols: list[str], start: date, end: date,
             now: datetime | None = None) -> dict[str, pd.DataFrame]:
        """Daily bars per symbol. Symbols with no data are absent from the dict.
        Today's (partial) bar is dropped unless `now` is after 16:00 ET.
        If a multi-symbol request fails on both feeds, each symbol is retried alone and
        the ones that still fail are skipped (absent), so one bad symbol never loses the rest.
        A single-symbol request that fails on both feeds raises."""
        if not symbols:
            return {}
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        end_dt = min(datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc),
                     now_utc - SIP_DELAY)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)

        def fetch(feed, syms):
            return self.data.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=list(syms), timeframe=TimeFrame.Day,
                start=start_dt, end=end_dt, feed=feed)).df

        def fetch_set(syms):
            try:
                return fetch(DataFeed.SIP, syms), "sip"
            except Exception as e:  # noqa: BLE001
                log.warning("SIP bars failed (%s); falling back to IEX, volume unusable", e)
                df = fetch(DataFeed.IEX, syms)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["volume"] = float("nan")  # IEX volume understates: ADV must fail safe
                return df, "iex"

        syms = list(symbols)
        try:
            parts = [fetch_set(syms)]
        except Exception as e:  # noqa: BLE001
            if len(syms) == 1:
                raise
            log.warning("bars for %d symbols failed (%s); retrying one by one", len(syms), e)
            parts = []
            for s in syms:
                try:
                    parts.append(fetch_set([s]))
                except Exception as e1:  # noqa: BLE001
                    log.warning("bars for %s failed, skipped: %s", s, e1)
        out: dict[str, pd.DataFrame] = {}
        now_ny = now_utc.astimezone(NY)
        today_ny = now_ny.date()
        # the SIP feed lags SIP_DELAY, so today's bar is only complete after 16:00 + delay
        after_close = (now_ny.hour, now_ny.minute) >= (16, SIP_DELAY.seconds // 60)
        for df, feed_name in parts:
            if df is None or df.empty:
                continue
            for sym in df.index.get_level_values(0).unique():
                d = df.xs(sym, level=0)[["open", "high", "low", "close", "volume"]].copy()
                d.index = pd.to_datetime(d.index).tz_convert(NY).tz_localize(None).normalize()
                d.index.name = "date"
                d.attrs["feed"] = feed_name
                if not after_close:
                    d = d[d.index.date < today_ny]
                if d.empty:
                    continue
                out[str(sym)] = d
        return out

    def latest_quotes(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        """(bid, ask) per symbol. Either side may be 0.0 when no quote exists;
        symbols Alpaca returns nothing for are absent."""
        if not symbols:
            return {}
        q = self.data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX))
        return {s: (float(v.bid_price), float(v.ask_price)) for s, v in q.items()}

    def news(self, symbol: str, limit: int = 8) -> list[dict]:
        res = self.news_client.get_news(NewsRequest(symbols=symbol, limit=limit))
        items = res.data.get("news", []) if hasattr(res, "data") else []
        return [{"headline": n.headline, "summary": n.summary,
                 "created_at": n.created_at, "source": n.source} for n in items]

    def asset_info(self, symbols: list[str]) -> dict[str, dict]:
        """Asset details per symbol; unknown or errored symbols are absent."""
        out = {}
        for s in symbols:
            try:
                a = self.trading.get_asset(s)
            except Exception:  # noqa: BLE001
                continue
            out[s] = {"tradable": bool(a.tradable),
                      "exchange": str(getattr(a.exchange, "value", a.exchange)),
                      "fractionable": bool(a.fractionable), "name": a.name}
        return out

    def fundamentals(self, symbol: str) -> dict:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol.upper())
        path = self.cache_dir / f"fund_{safe}.json"
        try:
            if path.exists() and time.time() - path.stat().st_mtime < CACHE_SECONDS:
                return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
        try:
            info = yf.Ticker(symbol).info or {}
            res = {k: info[src] for k, src in [
                ("market_cap", "marketCap"), ("sector", "sector"),
                ("industry", "industry"),
                ("long_business_summary", "longBusinessSummary"),
                ("trailing_pe", "trailingPE")] if info.get(src) is not None}
        except Exception:  # noqa: BLE001
            return {}
        if not res:
            return {}
        try:
            path.write_text(json.dumps(res))
        except Exception:  # noqa: BLE001
            pass
        return res

    def trading_days(self, start: date, end: date) -> list[date]:
        """Exchange trading days in [start, end] (Alpaca calendar), sorted. Cached in memory.
        Raises when no trading client or the API fails (callers fall back and log)."""
        if self.trading is None:
            raise RuntimeError("no Alpaca trading client")
        cache = self.__dict__.setdefault("_tdays", {})
        key = (start, end)
        if key not in cache:
            cal = self.trading.get_calendar(GetCalendarRequest(start=start, end=end))
            cache[key] = sorted({c.date for c in cal if start <= c.date <= end})
        return list(cache[key])

    def calendar_today(self, now: datetime) -> tuple[datetime, datetime] | None:
        """Today's (open, close) as aware UTC datetimes, or None if closed/unavailable."""
        if self.trading is None:
            return None
        today = now.astimezone(NY).date()
        cal = self.trading.get_calendar(GetCalendarRequest(start=today, end=today))
        for c in cal:
            if c.date == today:
                o = c.open.replace(tzinfo=NY).astimezone(timezone.utc)
                cl = c.close.replace(tzinfo=NY).astimezone(timezone.utc)
                return o, cl
        return None
