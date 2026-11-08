#!/usr/bin/env bash
# The nightly Dream: ingest the previous session, write its dream, print the morning review.
#
# PROPOSED, NOT SCHEDULED, NEVER RUN BY AN AGENT. The first step is a live GDELT download. Until
# the owner has seen the spine ingest run live, run this by hand; see docs/DREAM_SCHEDULE.md.
#
# usage: scripts/dream_nightly.sh [YYYY-MM-DD]
#   The date is a session to (re)build; without it, the last session before today.
#   DREAM_OUT overrides the output root (default stages/52_dream_report/output).
# Each step stops the script on failure, so a failed ingest never writes a dream.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${DREAM_OUT:-stages/52_dream_report/output}"

previous_session() {
  uv run python -c '
import datetime as dt
from precedent.spine.session_calendar import load_nyse_calendar, weekday_sessions
today = dt.date.today()
days = weekday_sessions(today - dt.timedelta(days=10), today - dt.timedelta(days=1), load_nyse_calendar())
print(days[-1].date().isoformat())
'
}

SESSION="${1:-$(previous_session)}"
echo "dream_nightly: session ${SESSION}, output ${OUT}"

uv run precedent ingest gdelt --since "$SESSION" --until "$SESSION"
uv run precedent dream --date "$SESSION" --out "$OUT"
uv run precedent dream review --out "$OUT"
