"""`scripts/dream_nightly.sh` is checked, never run: its first step is a live ingest."""

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dream_nightly.sh"


def test_the_script_parses_and_is_executable():
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)  # syntax only; nothing runs


def test_the_script_stops_on_error_and_has_no_personal_path():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "stages/52_dream_report/output" in text
    assert "/Users/" not in text and "/home/" not in text


def test_the_script_names_ingest_then_the_dream_then_the_review_in_order():
    commands = [
        line.strip()
        for line in SCRIPT.read_text().splitlines()
        if re.match(r"\s*uv run precedent ", line)
    ]
    assert len(commands) == 3
    assert commands[0].startswith("uv run precedent ingest gdelt")
    assert commands[1].startswith('uv run precedent dream --date "$SESSION"')
    assert commands[2].startswith("uv run precedent dream review")


def test_the_script_catches_up_in_order_and_reviews_from_the_first_caught_up_session():
    text = SCRIPT.read_text()
    # D70: every session after the manifest's last one, ingested then dreamed one at a time
    assert "manifest.json" in text and '"last_session"' in text
    assert "for SESSION in $SESSIONS; do" in text
    assert text.index("precedent ingest gdelt") < text.index("precedent dream --date")
    assert text.index("done") < text.index("precedent dream review")
    assert '${FIRST:+--since "$FIRST"}' in text
    assert "--allow-gap" not in text  # a gap is the owner's choice, never the script's


def test_the_schedule_doc_is_proposed_not_installed_and_has_no_personal_path():
    text = (ROOT / "docs" / "DREAM_SCHEDULE.md").read_text()
    assert "proposed" in text.lower() and "03:30" in text
    assert "/Users/" not in text and "/home/" not in text


def test_the_script_passes_prices_only_when_both_shell_variables_are_set():
    text = SCRIPT.read_text()
    assert 'if [ -n "${DREAM_PRICES_DIR:-}" ] && [ -n "${DREAM_MARKET:-}" ]; then' in text
    assert '--prices "$DREAM_PRICES_DIR" --market "$DREAM_MARKET"' in text
    assert "${PRICE_ARGS[@]" in text
    doc = (ROOT / "docs" / "DREAM_SCHEDULE.md").read_text()
    assert "DREAM_PRICES_DIR" in doc and "DREAM_MARKET" in doc
