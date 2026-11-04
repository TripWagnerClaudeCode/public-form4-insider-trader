#!/usr/bin/env bash
# Install insider-bot systemd user units and enable the timers (live + frozen v1 control). Idempotent.
# Never touches .env or any other unit. Use --uninstall to remove.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="$HOME/.config/systemd/user"
timers=(insider-bot-scan-am.timer insider-bot-scan-pm.timer insider-bot-reconcile.timer insider-bot-review.timer
        insider-bot-poll.timer insider-bot-curate.timer
        insider-bot-control-scan-am.timer insider-bot-control-scan-pm.timer insider-bot-control-reconcile.timer)
units=("${timers[@]}" "insider-bot-scan@.service" insider-bot-reconcile.service insider-bot-review.service
       "insider-bot-control-scan@.service" insider-bot-control-reconcile.service insider-bot-curate.service)

if [[ "${1:-}" == "--uninstall" ]]; then
  for t in "${timers[@]}"; do systemctl --user disable --now "$t" 2>/dev/null || true; done
  for u in "${units[@]}"; do rm -f "${dest:?}/${u:?}"; done
  systemctl --user daemon-reload
  echo "insider-bot units removed"
  exit 0
fi

repo="$(cd "$here/.." && pwd)"
uv_bin="$(command -v uv || true)"
[[ -n "$uv_bin" ]] || { echo "uv not found on PATH (https://docs.astral.sh/uv/)"; exit 1; }
mkdir -p "$dest"
for u in "${units[@]}"; do
  sed -e "s#__REPO__#$repo#g" -e "s#__UV__#$uv_bin#g" "$here/$u" > "$dest/$u"
  chmod 0644 "$dest/$u"
done
systemctl --user daemon-reload
for t in "${timers[@]}"; do systemctl --user enable --now "$t"; done
systemctl --user list-timers --no-pager | grep insider-bot || true
