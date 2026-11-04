"""OpenBB accessor: lazy (the first `from openbb import obb` builds static assets for ~1 min)."""
import subprocess
import sys
import types


def test_importing_the_accessor_does_not_load_openbb():
    code = "import sys, insiderbot.data.openbb; print('openbb' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_obb_returns_the_openbb_app_and_caches_it(monkeypatch):
    from insiderbot.data import openbb as acc

    sentinel = object()
    monkeypatch.setitem(sys.modules, "openbb", types.SimpleNamespace(obb=sentinel))
    acc.obb.cache_clear()
    try:
        assert acc.obb() is sentinel
        monkeypatch.setitem(sys.modules, "openbb", types.SimpleNamespace(obb=object()))
        assert acc.obb() is sentinel        # built once per process
    finally:
        acc.obb.cache_clear()
