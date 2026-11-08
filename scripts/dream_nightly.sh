#!/usr/bin/env bash
# The nightly Dream: ingest each session not yet dreamed, write its dream in order, and print the
# morning review.
#
# PROPOSED, NOT SCHEDULED, NEVER RUN BY AN AGENT. The first step is a live GDELT download. Until
# the owner has seen the spine ingest run live, run this by hand; see docs/DREAM_SCHEDULE.md.
#
# usage: scripts/dream_nightly.sh [YYYY-MM-DD]
#   Without a date, it catches up: every session after the last one dreamed (the `last_session`
#   in $DREAM_OUT/manifest.json) through the last session before today, in order. On a first run
#   it dreams only the last session before today. With a date it dreams only that session; the
#   dream command refuses a session out of order, or after a gap.
#   DREAM_OUT overrides the output root (default stages/52_dream_report/output).
#   DREAM_PRICES_DIR and DREAM_MARKET (shell variables, both or neither) pass --prices and
#   --market to the dream, which turns text_vs_price on; without them every dream says so under
#   "Not run".
# Each step stops the script on failure, so a failed ingest never writes a dream and a session is
# never skipped.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${DREAM_OUT:-stages/52_dream_report/output}"
PRICE_ARGS=()
if [ -n "${DREAM_PRICES_DIR:-}" ] && [ -n "${DREAM_MARKET:-}" ]; then
  PRICE_ARGS=(--prices "$DREAM_PRICES_DIR" --market "$DREAM_MARKET")
fi

sessions_to_dream() {
  uv run python - "$OUT" "${1:-}" <<'PY'
import datetime as dt
import json
import pathlib
import sys

from precedent.spine.session_calendar import load_nyse_calendar, weekday_sessions

out, given = pathlib.Path(sys.argv[1]), sys.argv[2]
if given:
    print(dt.date.fromisoformat(given).isoformat())
    sys.exit(0)
calendar = load_nyse_calendar()
today = dt.date.today()
last_before_today = weekday_sessions(today - dt.timedelta(days=10), today - dt.timedelta(days=1), calendar)[-1].date()
manifest = out / "manifest.json"
start = last_before_today
if manifest.exists():
    start = dt.date.fromisoformat(json.loads(manifest.read_text())["last_session"]) + dt.timedelta(days=1)
if start <= last_before_today:
    for day in weekday_sessions(start, last_before_today, calendar):
        print(day.date().isoformat())
PY
}

SESSIONS="$(sessions_to_dream "${1:-}")"
if [ -z "$SESSIONS" ]; then
  echo "dream_nightly: nothing to dream, output ${OUT}"
fi
FIRST=""
for SESSION in $SESSIONS; do
  FIRST="${FIRST:-$SESSION}"
  echo "dream_nightly: session ${SESSION}, output ${OUT}"
  uv run precedent ingest gdelt --since "$SESSION" --until "$SESSION"
  uv run precedent dream --date "$SESSION" --out "$OUT" ${PRICE_ARGS[@]+"${PRICE_ARGS[@]}"}
done
uv run precedent dream review --out "$OUT" ${FIRST:+--since "$FIRST"}
