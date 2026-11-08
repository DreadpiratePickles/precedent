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


def test_the_schedule_doc_is_proposed_not_installed_and_has_no_personal_path():
    text = (ROOT / "docs" / "DREAM_SCHEDULE.md").read_text()
    assert "proposed" in text.lower() and "03:30" in text
    assert "/Users/" not in text and "/home/" not in text
