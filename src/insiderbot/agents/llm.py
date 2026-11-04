"""Thin wrapper over the Anthropic SDK: structured output, budgets, cost tracking.

Mirrors trading-bot's LLM. Additions: available() / AgentUnavailable (no API key -> fail safe),
an optional on_cost hook (journal.add_cost) and an optional monthly USD cap.

Second backend: cli_path set -> each call runs headless `claude -p` on the user's Claude
subscription (no API key, no $ cost recorded; the per-run output-token budget still applies).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import anthropic
from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

PROMPT_DIR = Path(__file__).parent / "prompts"
CLI_TIMEOUT_S = 600
RESEARCH_TOOLS = "WebSearch,WebFetch"   # the only tools a research call ever gets (read-only web)

# $ per 1M tokens (input, output). Cached on 2026-10-06; verify against current pricing.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class BudgetExceeded(RuntimeError):
    pass


class AgentRefused(RuntimeError):
    pass


class AgentUnavailable(RuntimeError):
    """No Anthropic API key configured."""


def load_prompt(name: str, prompt_dir: str = "prompts") -> str:
    return (Path(__file__).parent / prompt_dir / f"{name}.md").read_text(encoding="utf-8").strip()


def prompt_version(prompt_dir: str = "prompts") -> str:
    files = sorted((Path(__file__).parent / prompt_dir).glob("*.md"))
    h = hashlib.sha1()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


class LLM:
    def __init__(
        self,
        max_output_tokens: int,
        client: anthropic.Anthropic | None = None,
        *,
        api_key: str | None = None,
        max_monthly_usd: float,
        month_cost: Callable[[], float],
        on_cost: Callable[[float], None] | None = None,
        cli_path: str | None = None,
    ):
        """api_key: from Settings.secrets().anthropic_key. A pre-built client (tests) also counts
        as available. on_cost(usd) is called after every call (wire to Journal.add_cost).
        month_cost() returns month-to-date USD already recorded (Journal.month_cost_usd).
        Both max_monthly_usd and month_cost are required so the cap can never be silently off.
        Cap check = month_cost() + this process's spend that on_cost did NOT record (no hook, or the
        hook raised). Spend recorded by on_cost is already in month_cost(), so it is not double counted."""
        self._client = client
        self._api_key = api_key
        self._cli_path = cli_path
        self.max_output_tokens = max_output_tokens
        self.on_cost = on_cost
        self.max_monthly_usd = max_monthly_usd
        self.month_cost = month_cost
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._unrecorded_usd = 0.0
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self._client is not None or bool(self._api_key) or bool(self._cli_path)

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise AgentUnavailable("no Anthropic API key")
            self._client = anthropic.Anthropic(api_key=self._api_key, max_retries=4)
        return self._client

    def _account(self, model: str, usage) -> float:
        pin, pout = PRICES.get(model, (4.00, 20.00))
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost = (usage.input_tokens * pin + cache_write * pin * 1.25 + cache_read * pin * 0.1
                + usage.output_tokens * pout) / 1e6
        with self._lock:
            self.input_tokens += usage.input_tokens + cache_read + cache_write
            self.output_tokens += usage.output_tokens
            self.cost_usd += cost
        recorded = False
        if self.on_cost is not None:
            try:
                self.on_cost(cost)
                recorded = True
            except Exception as e:  # money is already spent; never crash on bookkeeping
                print(f"agents: cost hook failed ({type(e).__name__}: {e})", file=sys.stderr)
                log.error("cost hook failed: %s", e)
        if not recorded:
            with self._lock:
                self._unrecorded_usd += cost
        return cost

    def _check_budget(self) -> None:
        if self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(f"run output-token budget {self.max_output_tokens} exhausted")
        if self.month_cost() + self._unrecorded_usd >= self.max_monthly_usd:
            raise BudgetExceeded(f"monthly budget ${self.max_monthly_usd} reached")

    def structured(self, *, model: str, effort: str, system: str, user: str, schema: type[T],
                   max_tokens: int = 16000) -> T:
        if not self.available():
            raise AgentUnavailable("no Anthropic API key")
        self._check_budget()
        if self._client is None and not self._api_key:
            return self._structured_cli(model=model, effort=effort, system=system, user=user,
                                        schema=schema)
        resp = self.client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
            output_format=schema,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self._account(model, resp.usage)
        if resp.stop_reason == "refusal":
            raise AgentRefused(f"{schema.__name__}: model declined ({getattr(resp, 'stop_details', None)})")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"{schema.__name__}: output truncated at max_tokens={max_tokens}")
        parsed = resp.parsed_output
        if parsed is None:
            raise RuntimeError(f"{schema.__name__}: no parsed output (stop_reason={resp.stop_reason})")
        return parsed

    def _structured_cli(self, *, model: str, effort: str, system: str, user: str,
                        schema: type[T]) -> T:
        """One headless `claude -p` call: no tools, no user settings/hooks/CLAUDE.md, no saved
        session, empty temp cwd. ANTHROPIC_* env is dropped so it bills the subscription.
        Every failure becomes RuntimeError/AgentRefused, which the team treats as 'pass'."""
        cmd = [self._cli_path, "-p", "--output-format", "json",
               "--json-schema", json.dumps(schema.model_json_schema()),
               "--model", model, "--effort", effort, "--system-prompt", system,
               "--tools", "", "--setting-sources", "", "--no-session-persistence"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
        with tempfile.TemporaryDirectory(prefix="insiderbot-claude-") as cwd:
            try:
                p = subprocess.run(cmd, input=user, capture_output=True, text=True,
                                   timeout=CLI_TIMEOUT_S, cwd=cwd, env=env)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise RuntimeError(f"{schema.__name__}: claude CLI failed ({type(e).__name__}: {e})") from e
        try:
            d = json.loads(p.stdout)
        except ValueError:
            tail = (p.stderr or p.stdout or "").strip()[-300:]
            raise RuntimeError(f"{schema.__name__}: claude CLI exit {p.returncode}: {tail}") from None
        if not isinstance(d, dict):
            raise RuntimeError(f"{schema.__name__}: claude CLI returned non-object JSON")
        u = d.get("usage") or {}
        out_tok = int(u.get("output_tokens") or 0)
        in_tok = sum(int(u.get(k) or 0) for k in
                     ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        with self._lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
        if d.get("stop_reason") == "refusal":
            raise AgentRefused(f"{schema.__name__}: model declined")
        if d.get("is_error") or d.get("subtype") != "success":
            raise RuntimeError(f"{schema.__name__}: claude CLI error ({d.get('subtype')}): "
                               f"{str(d.get('result'))[:300]}")
        out = d.get("structured_output")
        if out is None:
            raise RuntimeError(f"{schema.__name__}: claude CLI returned no structured output")
        try:
            return schema.model_validate(out)
        except ValueError as e:
            raise RuntimeError(f"{schema.__name__}: invalid structured output ({e})") from e

    def research(self, *, model: str, effort: str, system: str, user: str, schema: type[T],
                 max_turns: int = 12, timeout_s: int = 300) -> tuple[T, list[dict]]:
        """One headless `claude -p` call that may use ONLY WebSearch and WebFetch (no shell, files, MCP
        or settings), streamed so every query and fetched URL is returned for the journal. Ported from
        trading-bot. Fails closed: timeout, garbage, a missing or error result raise RuntimeError."""
        if not self._cli_path:
            raise AgentUnavailable("research calls need the claude-cli backend")
        self._check_budget()
        cmd = [self._cli_path, "-p", "--output-format", "stream-json", "--verbose",
               "--json-schema", json.dumps(schema.model_json_schema()),
               "--model", model, "--effort", effort, "--system-prompt", system,
               "--tools", RESEARCH_TOOLS, "--allowedTools", "WebSearch", "WebFetch",
               "--max-turns", str(max_turns), "--setting-sources", "", "--no-session-persistence"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
        touched: list[dict] = []
        result: dict | None = None
        lines: queue.Queue = queue.Queue()
        errbuf: list[str] = []
        with tempfile.TemporaryDirectory(prefix="insiderbot-research-") as cwd:
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", cwd=cwd, env=env)
            except OSError as e:
                raise RuntimeError(f"{schema.__name__}: claude CLI failed ({e})") from e

            def pump():
                for line in p.stdout:
                    lines.put(line)
                lines.put(None)

            def feed():   # in a thread: a CLI that never reads stdin must not block the deadline
                try:
                    p.stdin.write(user)
                    p.stdin.close()
                except (OSError, ValueError):
                    pass

            deadline = time.monotonic() + timeout_s
            threading.Thread(target=pump, daemon=True).start()
            threading.Thread(target=lambda: errbuf.append(p.stderr.read()), daemon=True).start()
            threading.Thread(target=feed, daemon=True).start()
            try:
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise RuntimeError(f"{schema.__name__}: research call timed out after {timeout_s}s")
                    try:
                        line = lines.get(timeout=min(left, 1.0))
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(d, dict):
                        continue
                    if d.get("type") == "assistant":
                        for c in (d.get("message") or {}).get("content") or []:
                            if isinstance(c, dict) and c.get("type") == "tool_use":
                                inp = c.get("input") or {}
                                if c.get("name") == "WebFetch":
                                    touched.append({"tool": "WebFetch", "url": str(inp.get("url", ""))})
                                elif c.get("name") == "WebSearch":
                                    touched.append({"tool": "WebSearch", "query": str(inp.get("query", ""))})
                    elif d.get("type") == "result":
                        result = d
            finally:
                if p.poll() is None:
                    p.kill()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if result is None:
            raise RuntimeError(f"{schema.__name__}: research call returned no result (exit {p.returncode}): "
                               f"{''.join(errbuf).strip()[-300:]}")
        u = result.get("usage") or {}
        with self._lock:
            self.input_tokens += sum(int(u.get(k) or 0) for k in
                                     ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            self.output_tokens += int(u.get("output_tokens") or 0)
        if result.get("stop_reason") == "refusal":
            raise AgentRefused(f"{schema.__name__}: model declined")
        if result.get("is_error") or result.get("subtype") != "success":
            raise RuntimeError(f"{schema.__name__}: claude CLI error ({result.get('subtype')}): "
                               f"{str(result.get('result'))[:300]}")
        out = result.get("structured_output")
        if out is None:
            raise RuntimeError(f"{schema.__name__}: claude CLI returned no structured output")
        try:
            return schema.model_validate(out), touched
        except ValueError as e:
            raise RuntimeError(f"{schema.__name__}: invalid structured output ({e})") from e
