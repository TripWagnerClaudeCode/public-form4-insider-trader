from datetime import datetime, timezone

import pytest

from insiderbot import formula
from insiderbot.config import Bounds, ROOT
from insiderbot.models import IssuerFeatures


def feats(**kw):
    d = dict(ticker="ABC", asof=datetime(2026, 11, 2, tzinfo=timezone.utc), n_buyers=1, total_value=15_000.0,
             has_ceo=False, has_cfo=False, has_chair=False, n_directors=1, size_vs_holdings=0.02,
             first_buy_months=0.0, frac_10b5_1=0.0, ret_3m=0.05, drawdown_52w=0.05, price=20.0,
             adv20_usd=5e6, mcap_usd=5e9, sector="Tech", exchange="NASDAQ", is_fund_like=False,
             accessions=("a",), latest_accepted_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
             name="ABC Inc", tradable=True)
    d.update(kw)
    return IssuerFeatures(**d)


@pytest.fixture
def fm():
    return formula.load("v001", ROOT / "formula")


def test_active_is_v001():
    assert formula.active(ROOT / "formula").version == "v001"
    assert formula.shadow(ROOT / "formula") is None


def test_cluster_ceo_scores_high(fm):
    f = feats(n_buyers=3, total_value=2e6, has_ceo=True, n_directors=2, size_vs_holdings=0.6,
              first_buy_months=24, drawdown_52w=0.30, ret_3m=-0.1, mcap_usd=1e9)
    s, flags = formula.score(f, fm)
    assert s >= 70
    assert "+ceo_cfo" in flags


def test_single_director_small_scores_low(fm):
    s, _ = formula.score(feats(), fm)
    assert s < 40


def test_10b5_1_penalised(fm):
    base = feats(n_buyers=3, total_value=2e6, has_ceo=True)
    s0, f0 = formula.score(base, fm)
    s1, f1 = formula.score(feats(n_buyers=3, total_value=2e6, has_ceo=True, frac_10b5_1=1.0), fm)
    assert s1 < s0 - 20
    assert "-10b5-1" in f1 and "-10b5-1" not in f0


def test_missing_data_is_neutral(fm):
    s_none, _ = formula.score(feats(ret_3m=None, drawdown_52w=None, mcap_usd=None), fm)
    s_good, _ = formula.score(feats(ret_3m=-0.3, drawdown_52w=0.5, mcap_usd=1e9), fm)
    assert s_none < s_good


def test_score_clipped(fm):
    s, _ = formula.score(feats(frac_10b5_1=1.0, total_value=0.0, n_buyers=1), fm)
    assert s == 0.0


def test_tier_weight(fm):
    assert formula.tier_weight(55, fm) == 0
    assert formula.tier_weight(60, fm) == 0.75
    assert formula.tier_weight(75, fm) == 1.0
    assert formula.tier_weight(99, fm) == 1.25


def test_validate_rejects_big_jump(fm):
    new = fm.model_copy(update={"weights": {**fm.weights, "cluster": 0.45}})
    errs = formula.validate(new, fm, Bounds())
    assert any("cluster" in e for e in errs)


def test_validate_allows_zero_to_005(fm):
    old = fm.model_copy(update={"weights": {**fm.weights, "chair": 0.0}})
    new = fm.model_copy(update={"weights": {**fm.weights, "chair": 0.05}})
    assert formula.validate(new, old, Bounds()) == []


def test_validate_allows_small_change(fm):
    new = fm.model_copy(update={"weights": {**fm.weights, "cluster": 0.30}})
    assert formula.validate(new, fm, Bounds()) == []


def test_validate_rejects_out_of_bounds_stop(fm):
    new = fm.model_copy(update={"stop_loss_pct": 0.25})
    errs = formula.validate(new, fm, Bounds())
    assert any("stop_loss_pct" in e for e in errs)
    assert formula.validate(fm.model_copy(update={"stop_loss_pct": 0.05}), fm, Bounds())


def test_validate_weight_keys_and_range(fm):
    w = dict(fm.weights)
    del w["chair"]
    w["bogus"] = 0.1
    errs = formula.validate(fm.model_copy(update={"weights": w}), fm, Bounds())
    assert any("missing" in e and "chair" in e for e in errs)
    assert any("unknown" in e and "bogus" in e for e in errs)
    errs = formula.validate(fm.model_copy(update={"weights": {**fm.weights, "cluster": 1.5}}), fm, Bounds())
    assert any("outside" in e for e in errs)


def test_save_next_version(tmp_path, fm):
    (tmp_path / "v001.yaml").write_text((ROOT / "formula/v001.yaml").read_text())
    p = formula.save(fm.model_copy(update={"parent": "v001", "author": "review-agent"}), tmp_path)
    assert p.name == "v002.yaml"
    loaded = formula.load("v002", tmp_path)
    assert loaded.version == "v002" and loaded.parent == "v001" and loaded.weights == fm.weights
    assert formula.save(fm, tmp_path).name == "v003.yaml"
    assert b"\r" not in p.read_bytes()


def test_active_and_shadow_pointers(tmp_path, fm):
    (tmp_path / "v001.yaml").write_text((ROOT / "formula/v001.yaml").read_text())
    formula.set_active("v001", tmp_path)
    assert formula.active(tmp_path).version == "v001"
    assert formula.shadow(tmp_path) is None
    formula.set_shadow("v001", tmp_path)
    assert formula.shadow(tmp_path).version == "v001"
    formula.set_shadow(None, tmp_path)
    assert formula.shadow(tmp_path) is None
    with pytest.raises(ValueError):
        formula.load("../x", tmp_path)


def test_validate_tiers_rules(fm):
    b = Bounds()
    assert any("ascending" in e for e in formula.validate(
        fm.model_copy(update={"tiers": [(70, 1.0), (60, 0.75), (80, 1.25)]}), fm, b))
    assert any("multiplier" in e and "above" in e for e in formula.validate(
        fm.model_copy(update={"tiers": [(60, 0.75), (70, 1.0), (80, 2.5)]}), fm, b))
    assert any("buy_threshold" in e for e in formula.validate(
        fm.model_copy(update={"tiers": [(65, 0.75), (70, 1.0), (80, 1.25)]}), fm, b))


def test_save_never_overwrites(tmp_path, fm, monkeypatch):
    (tmp_path / "v001.yaml").write_text((ROOT / "formula/v001.yaml").read_text())
    monkeypatch.setattr(formula, "_next_version", lambda d: "v001")  # stale view
    assert formula.save(fm, tmp_path).name == "v002.yaml"
    assert "v001" in (tmp_path / "v001.yaml").read_text()
