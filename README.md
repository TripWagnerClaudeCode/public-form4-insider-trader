# form4-insider-trader

A bot that follows **SEC Form 4 insider purchases**. When company insiders buy their own stock on the open market, it reviews the filing with Claude agents. Buys that pass hard risk rules are placed on an [Alpaca](https://alpaca.markets) **paper** account. It then manages stops and exits, and reports on itself.

> **Paper trading only. Not financial advice.** The Alpaca broker is hard-wired to paper (`paper=True`); there is no live mode. Past insider buying does not predict returns. This is an experiment.

```
SEC EDGAR Form 4 ──► filters ──► agent review ───────────────────► risk gate ──► Alpaca paper
(open-market buys:   (price,     Analyst · Researcher (web + SEC)    (plain code:   (orders +
 who, how much,       size,      → Portfolio Manager                 limits, kill   protective
 cluster buys)        liquidity)  → Risk Reviewer (veto only)        switch)        stops)
                                         │
                      SQLite journal · scan/daily/weekly Markdown reports · optional ntfy push
```

The agents can say yes, no or exit. Position size, limits and stops come from code and `config/settings.yaml`, never from the model.

## What it does

- **Scan** (twice a day, plus an optional 5-minute live-feed poll during market hours): pulls new Form 4s, keeps open-market insider buys, journals every candidate, and runs the agent review and risk gate before any order.
- **Manage:** places protective stops, handles exits, and runs an intraday Event Review on held names. An exit needs both agents to agree.
- **Reconcile** (after the close): updates order status, stops, equity and outcomes, then writes the daily report.
- **Learn** (weekly): a research review of its own results, plus a Curator that keeps a short `lessons.md`. You can edit that file. Your edits and the `## Pinned by you` section always win.
- **Control:** a frozen copy of the original strategy (`--mode control`) runs on a simulated book, so you can see whether the upgrades help.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An Alpaca **paper** account (free)
- An SEC identity (`Your Name your@email`). SEC fair-access rules require one.
- For the agents, one of:
  - **Claude Code CLI** (`agents.backend: claude-cli`, what `config/settings.yaml` ships with). It runs headless `claude -p` on your Claude subscription. No API key is needed. If a usage limit is hit, that call fails safe: no buy. The Researcher's deep dives need this backend.
  - **Anthropic API** (`agents.backend: api` plus `ANTHROPIC_API_KEY`). You pay per token, and a monthly cost cap is enforced from the settings. The Researcher's deep dives aren't available in this mode.
- Linux with systemd for the timers (any OS works for manual runs)

## Setup

```bash
git clone https://github.com/TripWagnerClaudeCode/public-form4-insider-trader.git form4-insider-trader
cd form4-insider-trader
uv sync
cp .env.example .env      # then fill it in
uv run python -m insiderbot status
```

`.env` (git-ignored):

| Variable | |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | **paper** keys |
| `EDGAR_IDENTITY` | `Your Name your@email`. Scans stop buying, with an alert, until it's set. |
| `ANTHROPIC_API_KEY` | only with `agents.backend: api` |
| `NTFY_URL`, `NTFY_TOPIC`, `NTFY_TOKEN` | optional phone pushes via [ntfy](https://ntfy.sh). On public ntfy.sh, use a long random topic. |
| `BOT_KILL_SWITCH` | `yes` halts all trading |

## Commands

Run from the repo root as `uv run python -m insiderbot <cmd>`:

| Command | Purpose |
|---|---|
| `status` | Read-only status: account, positions, kill switch |
| `scan --slot am\|pm` | Fetch filings, handle exits and reviews, place buys. Skipped if more than 60 min late or the market is closed (`--force` overrides). |
| `scan --mode dry-run` | Same, but with no orders (in-memory simulator) and a separate journal |
| `reconcile` | Order status, stops, equity, outcomes, daily report |
| `review` | Weekly research review |
| `curate` | Weekly lessons curation |
| `backfill --days N` | Journal past Form 4s as candidates (slow, takes hours; run it before installing the timers) |

## Schedule (systemd user timers)

```bash
bash deploy/install.sh          # installs units with this repo's path and your uv, enables the timers
loginctl enable-linger "$USER"  # so timers run without a login session
systemctl --user list-timers | grep insider-bot
journalctl --user -u "insider-bot-*" --since today
bash deploy/uninstall.sh        # remove
```

All times are America/New_York:

| When | What |
|---|---|
| Mon–Fri 10:00, 15:00 | scan (am / pm) |
| Mon–Fri 09:45–15:45, every 5 min | live-feed poll |
| Mon–Fri 16:15 | reconcile |
| Sun 18:00 / 19:00 | review / curate |

The frozen control runs its own scan and reconcile timers.

## Kill switch

- Running `touch KILL_SWITCH` in the repo root halts all trading. Delete the file to resume.
- `BOT_KILL_SWITCH=yes` in `.env` does the same, and every check reads it.
- The bot creates the file itself on a daily-loss or drawdown breach, and sends a priority alert.

## Where things live

| | |
|---|---|
| `config/settings.yaml` | mode, filters, risk limits, features, paths |
| `formula/` | versioned strategy parameters (`ACTIVE` points to the live one) |
| `data/journal.db` | SQLite: candidates, decisions, orders, outcomes (dry-runs use their own file) |
| `reports/` | `scan-*.md`, `daily-*.md`, `weekly/*.md` |
| `memory/` | `lessons.md`, scorecards and changelog (editable) |

Set `paths.reports_dir` or `memory.dir` to a synced folder if you want to read them on another device.

## Tests

```bash
uv run --all-groups python -m pytest -q
```

The tests never reach SEC, the Claude CLI, the Anthropic API or a broker.

## License

MIT, see [LICENSE](LICENSE).
