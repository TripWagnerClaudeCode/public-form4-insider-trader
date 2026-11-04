"""OpenBB access, the same pinned version as trading-bot (openbb 5.0.0).

Loaded lazily: the first `from openbb import obb` builds static assets (about a minute), so
scans that never ask for OpenBB data don't pay for it. Use `obb().yfinance...`, `obb().news...` etc.
Upstream ownership moved to the `openbq-org` GitHub org in 2026-09 and is unverified: keep the
versions pinned exactly in pyproject.toml and only upgrade after checking who publishes them.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def obb():
    from openbb import obb as app  # heavy import, deferred

    return app
