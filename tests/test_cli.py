import subprocess
import sys

from precedent import cli
from precedent.cli import TRACK_COMMANDS, main, resolve_command


def test_bare_text_is_sugar_for_ask():
    assert resolve_command(["Fed cuts rates unexpectedly"]) == (
        "ask",
        ["Fed cuts rates unexpectedly"],
    )
    assert resolve_command(["ingest", "gdelt", "--since", "2024-01-01"]) == (
        "ingest gdelt",
        ["--since", "2024-01-01"],
    )


def test_an_explicit_command_keeps_its_arguments():
    assert resolve_command(["ask", "--dry-run", "text"]) == ("ask", ["--dry-run", "text"])


def test_an_unbuilt_track_reports_not_built_yet(capsys, monkeypatch):
    monkeypatch.setitem(TRACK_COMMANDS, "ripple", "precedent.ripple._nonexistent:run")
    assert main(["ripple", "--date", "2026-11-07"]) == 2
    assert "not built yet: ripple" in capsys.readouterr().err


def test_a_missing_research_dependency_points_at_the_extra(capsys, monkeypatch):
    def run_needing_scipy(argv):
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(cli, "_load_target", lambda target: run_needing_scipy)
    assert main(["ripple"]) == 2
    assert "pip install 'precedent-engine[research]'" in capsys.readouterr().err


def test_a_target_is_called_with_the_remaining_arguments(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_load_target", lambda target: lambda argv: seen.append(argv) or 0)
    assert main(["ingest", "edgar", "--cik", "1"]) == 0
    assert seen == [["--cik", "1"]]


def test_an_unknown_ingest_source_is_a_usage_error():
    assert main(["ingest", "nowhere"]) == 2


def test_cli_imports_no_track_at_startup():
    code = (
        "import sys, precedent.cli; "
        "print(sorted(m for m in sys.modules if m.startswith("
        "('precedent.engine', 'precedent.dream', 'precedent.ripple', 'precedent.mechanism'))))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
