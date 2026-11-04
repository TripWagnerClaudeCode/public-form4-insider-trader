import pytest
from pydantic import ValidationError

from insiderbot import config


def test_defaults_load():
    s = config.load()
    assert s.mode == "paper"
    assert s.risk.max_position_weight == 0.08
    assert s.bounds.max_rel_change == 0.30


def test_live_mode_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("mode: live\n")
    with pytest.raises(ValidationError):
        config.load(p)


def test_secrets_strip_cr(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"ALPACA_API_KEY=abc\r\nALPACA_SECRET_KEY=def\r\n")
    s = config.load()
    sec = s.secrets(env)
    assert sec.alpaca_key == "abc"
    assert sec.alpaca_secret == "def"
