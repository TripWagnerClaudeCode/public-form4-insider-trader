"""Optional push notifications via ntfy (ntfy.sh or self-hosted). Never raises; never logs secrets.

Config, first match wins:
  NTFY_URL + NTFY_TOPIC (+ NTFY_TOKEN) environment variables, or
  a JSON file {"url": ..., "topic": ..., "token": ...} at $NTFY_CONFIG
  (default ~/.config/insider-bot/ntfy.json, keep it mode 0600).
On public ntfy.sh anyone who knows the topic can read it: use a long random topic name."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

TIMEOUT = 5.0
QUIET_FROM = 23
QUIET_UNTIL = 8
ZONE = ZoneInfo("America/New_York")


def _log(message: str) -> None:
    try:
        print(f"notify: {message}", file=sys.stderr)
    except Exception:
        pass


def _config_path() -> Path:
    return Path(os.environ.get("NTFY_CONFIG") or (Path.home() / ".config/insider-bot/ntfy.json")).expanduser()


def _ntfy_config() -> dict | None:
    url, topic = os.environ.get("NTFY_URL", "").strip(), os.environ.get("NTFY_TOPIC", "").strip()
    if url and topic:
        config = {"url": url, "topic": topic, "token": os.environ.get("NTFY_TOKEN", "").strip()}
    else:
        try:
            config = json.loads(_config_path().read_text())
        except Exception:
            return None
    if isinstance(config, dict) and isinstance(config.get("url"), str) and isinstance(config.get("topic"), str):
        return config
    return None


def quiet_until(now: dt.datetime) -> dt.datetime | None:
    """Next 8 AM New York time if `now` falls in quiet hours (11 PM-8 AM), else None."""
    local = now.astimezone(ZONE) if now.tzinfo else now.replace(tzinfo=ZONE)
    if not (local.hour >= QUIET_FROM or local.hour < QUIET_UNTIL):
        return None
    day = local.date() + (dt.timedelta(days=1) if local.hour >= QUIET_FROM else dt.timedelta())
    return dt.datetime(day.year, day.month, day.day, QUIET_UNTIL, tzinfo=ZONE)


def push(title: str, body: str, *, priority: int = 3, tags: list[str] | None = None,
         enabled: bool, now: dt.datetime | None = None) -> bool:
    """Send a push. enabled (required) is settings.notify.ntfy.

    ntfy holds it until 8 AM in quiet hours, except priority >= 5 (alerts), sent at once."""
    try:
        if not enabled:
            return False
        config = _ntfy_config()
        if not config:
            _log("skipped: no ntfy config (NTFY_URL/NTFY_TOPIC or $NTFY_CONFIG)")
            return False
        if not config["url"].lower().startswith(("http://", "https://")):
            _log("skipped: ntfy url is not http(s)")
            return False
        payload = {
            "topic": config["topic"],
            "title": (title or "insider-bot")[:200],
            "message": (body or "")[:3000],
            "priority": priority,
            "tags": tags or [],
        }
        hold = None if priority >= 5 else quiet_until(now or dt.datetime.now(dt.timezone.utc))
        if hold is not None:
            payload["delay"] = str(int(hold.timestamp()))
        headers = {"content-type": "application/json"}
        if config.get("token"):
            headers["authorization"] = f"Bearer {config['token']}"
        req = urllib.request.Request(config["url"], data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            if not 200 <= resp.status < 300:
                _log(f"failed: ntfy answered {resp.status}")
                return False
        return True
    except urllib.error.HTTPError as e:
        _log(f"failed: ntfy answered {e.code}")
        return False
    except Exception as e:  # never raise; never include the token
        _log(f"failed: {type(e).__name__}")
        return False
