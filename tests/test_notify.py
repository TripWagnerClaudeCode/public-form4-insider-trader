import datetime as dt
import json
import urllib.request
from zoneinfo import ZoneInfo

import pytest

from insiderbot import notify


def test_push_no_config_false(tmp_path, monkeypatch):
    monkeypatch.setenv("NTFY_CONFIG", str(tmp_path / "secrets" / "ntfy.json"))
    assert notify.push("t", "b", enabled=True) is False


def test_push_disabled_false(tmp_path, monkeypatch):
    monkeypatch.setenv("NTFY_CONFIG", str(tmp_path / "secrets" / "ntfy.json"))
    called = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: called.append(1))
    assert notify.push("t", "b", enabled=False) is False
    assert not called


def _cfg(tmp_path):
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "ntfy.json").write_text(
        json.dumps({"url": "https://ntfy.example/", "topic": "x", "token": "SECRET"}))


def test_push_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NTFY_CONFIG", str(tmp_path / "secrets" / "ntfy.json"))
    _cfg(tmp_path)

    def boom(*a, **k):
        raise OSError("down SECRET")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert notify.push("t", "b", enabled=True) is False
    assert "SECRET" not in capsys.readouterr().err


def test_push_sends_json_with_delay_in_quiet_hours(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NTFY_CONFIG", str(tmp_path / "secrets" / "ntfy.json"))
    _cfg(tmp_path)
    seen = {}

    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data)
        seen["auth"] = req.get_header("Authorization")
        return R()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    night = dt.datetime(2026, 11, 3, 5, 0, tzinfo=dt.timezone.utc)  # 00:00 ET
    assert notify.push("T", "B", priority=4, tags=["a"], enabled=True, now=night) is True
    assert seen["body"]["topic"] == "x" and seen["body"]["priority"] == 4
    assert "delay" in seen["body"]
    assert seen["auth"] == "Bearer SECRET"
    noon = dt.datetime(2026, 11, 3, 17, 0, tzinfo=dt.timezone.utc)
    notify.push("T", "B", enabled=True, now=noon)
    assert "delay" not in seen["body"]
    assert "SECRET" not in capsys.readouterr().err


ET = ZoneInfo("America/New_York")


def _capture(tmp_path, monkeypatch):
    monkeypatch.setenv("NTFY_CONFIG", str(tmp_path / "secrets" / "ntfy.json"))
    _cfg(tmp_path)
    seen = {}

    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return R()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


def test_priority_5_not_delayed(tmp_path, monkeypatch):
    seen = _capture(tmp_path, monkeypatch)
    night = dt.datetime(2026, 11, 3, 1, 0, tzinfo=ET)
    assert notify.push("HALT", "x", priority=5, enabled=True, now=night) is True
    assert "delay" not in seen["body"]


def test_exact_delay_value(tmp_path, monkeypatch):
    seen = _capture(tmp_path, monkeypatch)
    notify.push("T", "B", enabled=True, now=dt.datetime(2026, 11, 3, 1, 0, tzinfo=ET))
    want = int(dt.datetime(2026, 11, 3, 8, 0, tzinfo=ET).timestamp())
    assert seen["body"]["delay"] == str(want)


def test_enabled_is_required():
    with pytest.raises(TypeError):
        notify.push("t", "b")


def test_quiet_hour_boundaries():
    q = notify.quiet_until
    assert q(dt.datetime(2026, 11, 3, 22, 59, tzinfo=ET)) is None
    assert q(dt.datetime(2026, 11, 3, 23, 0, tzinfo=ET)) == dt.datetime(2026, 11, 4, 8, 0, tzinfo=ET)
    assert q(dt.datetime(2026, 11, 3, 7, 59, tzinfo=ET)) == dt.datetime(2026, 11, 3, 8, 0, tzinfo=ET)
    assert q(dt.datetime(2026, 11, 3, 8, 0, tzinfo=ET)) is None
    # EDT date (UTC-4): 03:00 UTC is 23:00 ET the evening before
    edt = q(dt.datetime(2026, 7, 2, 3, 0, tzinfo=dt.timezone.utc))
    assert edt == dt.datetime(2026, 7, 2, 8, 0, tzinfo=ET)
    assert edt.utcoffset() == dt.timedelta(hours=-4)


def test_push_env_config(monkeypatch):
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example/")
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.delenv("NTFY_TOKEN", raising=False)
    seen = {}

    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["topic"] = json.loads(req.data)["topic"]
        seen["auth"] = req.get_header("Authorization")
        return R()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert notify.push("t", "b", enabled=True, priority=5) is True
    assert seen == {"url": "https://ntfy.example/", "topic": "t", "auth": None}
