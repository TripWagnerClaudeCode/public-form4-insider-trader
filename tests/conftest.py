from pathlib import Path

import pytest

from insiderbot.risk import gate


@pytest.fixture(autouse=True)
def _isolated_env_file(tmp_path, monkeypatch):
    """Tests never read the real project .env for BOT_KILL_SWITCH."""
    monkeypatch.setattr(gate, "ENV_FILE", tmp_path / "test.env")


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch, tmp_path_factory):
    """Tests must never reach the real claude CLI (your Claude subscription) or bill an API key. A fake CLI
    script a test writes under pytest's temp folder may still run."""
    from insiderbot.agents import llm

    base = tmp_path_factory.getbasetemp().resolve()
    real = llm.LLM._structured_cli

    def guarded(self, **k):
        try:
            Path(self._cli_path or "").resolve().relative_to(base)
        except ValueError:
            raise RuntimeError("real claude CLI disabled in tests") from None
        return real(self, **k)

    def no_api(*a, **k):
        raise RuntimeError("Anthropic API client disabled in tests (it would bill ANTHROPIC_API_KEY)")

    monkeypatch.setattr(llm.LLM, "_structured_cli", guarded)
    real_research = llm.LLM.research

    def guarded_research(self, **k):
        try:
            Path(self._cli_path or "").resolve().relative_to(base)
        except ValueError:
            raise RuntimeError("real claude CLI disabled in tests") from None
        return real_research(self, **k)

    monkeypatch.setattr(llm.LLM, "research", guarded_research)
    from insiderbot.data import sec_filings

    real_get = sec_filings._get

    def no_sec(url, session, user_agent=None):   # tests never reach SEC without a mock session
        if session is None:
            raise RuntimeError("SEC network disabled in tests")
        return real_get(url, session, user_agent)

    monkeypatch.setattr(sec_filings, "_get", no_sec)
    monkeypatch.setattr(llm.anthropic, "Anthropic", no_api)


@pytest.fixture(autouse=True)
def _no_real_memory_dir(monkeypatch, tmp_path_factory):
    """Settings default memory to a real memory folder; tests must never write there."""
    import insiderbot.memory as memory

    base = tmp_path_factory.getbasetemp()
    real = memory.memory_root

    def guarded(s):
        r = real(s)
        try:
            r.resolve().relative_to(base.resolve())
            return r
        except ValueError:
            return base / "redirected-memory" / r.name

    monkeypatch.setattr(memory, "memory_root", guarded)
