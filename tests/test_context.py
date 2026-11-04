"""Agents see how far the price moved since the insiders bought (information only; scoring unchanged)."""
from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from insiderbot import cli, features
from insiderbot import formula as fml
from insiderbot.config import ROOT, Settings

from test_cli import NOW, FakeMarket, cluster


@pytest.fixture
def feats(tmp_path):
    from insiderbot.journal.store import Journal

    j = Journal(tmp_path / "j.db")
    txns = cluster()
    j.add_txns(txns, "live")
    (f,) = features.build(txns, FakeMarket(), j, NOW, 14)
    return f, txns


def test_context_includes_move_since_insider_buy(feats):
    f, txns = feats                       # insiders bought at $100; FakeMarket last close is $70
    f = dataclasses.replace(f, price=70.0)
    ctx = cli._context(FakeMarket(), f, txns, now=NOW)
    assert "insiders' average purchase price $100.00" in ctx["bars_summary"]
    assert "-30.0% since" in ctx["bars_summary"]
    assert "filed 2.0h ago" in ctx["bars_summary"]


def test_context_without_txns_unchanged_shape(feats):
    f, _ = feats
    assert "insiders' average purchase price" not in cli._context(FakeMarket(), f)["bars_summary"]


def test_score_unchanged_by_context_fields(feats):
    f, txns = feats
    fm = fml.active(ROOT / "formula")
    before = fml.score(f, fm)
    snapshot = dataclasses.asdict(f)
    cli._context(FakeMarket(), f, txns)
    assert fml.score(f, fm) == before and dataclasses.asdict(f) == snapshot
    assert features.hard_filter(f, Settings().filters) == features.hard_filter(f, Settings().filters)


def test_context_ignores_zero_price_txns(feats):
    f, txns = feats
    zero = [dataclasses.replace(t, price=0.0) for t in txns]
    assert "average purchase price" not in cli._context(FakeMarket(), f, zero)["bars_summary"]
    _ = timedelta  # noqa: F841
