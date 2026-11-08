import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "## Objective",
    "## Inputs",
    "## Process",
    "## Outputs",
    "## Verify",
    "## Approval",
    "## Failure Behavior",
]
SPINE_STAGES = {
    "01_ingest",
    "02_triage",
    "03_parse",
    "04_align",
    "05_factors",
    "06_dataset",
    "07_train",
    "08_evaluate",
    "09_backtest",
}
BANNERS = ("SYNTHETIC —", "LIVE —", "NO LIVE RUN —")
FORBIDDEN_OUTSIDE_CONFIG = re.compile(r"https?://|\bclaude-[a-z0-9]|\.cache/precedent")
SECRET_SHAPES = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def _contracts() -> list[Path]:
    return sorted((ROOT / "stages").glob("[0-9][0-9]_*/CONTEXT.md"))


def test_every_stage_contract_has_all_sections():
    contracts = _contracts()
    assert {c.parent.name for c in contracts} >= SPINE_STAGES
    for contract in contracts:
        assert [h for h in REQUIRED if h not in contract.read_text()] == [], contract


def test_every_contract_declares_a_loop_or_says_single_pass():
    for contract in _contracts():
        text = contract.read_text()
        assert "## Loop" in text or "single pass" in text.lower(), contract
    for stage in ("03_parse", "07_train"):
        assert "## Loop" in (ROOT / "stages" / stage / "CONTEXT.md").read_text()


def test_every_example_file_starts_with_an_evidence_banner():
    for path in (ROOT / "docs" / "examples").rglob("*"):
        if not path.is_file() or path.name == "README.md":
            continue
        text_file = path.with_suffix(".md") if path.suffix == ".png" else path
        assert text_file.read_text(encoding="utf-8").startswith(BANNERS), path


def test_urls_model_ids_and_data_dir_defaults_live_only_in_config():
    config = ROOT / "src" / "precedent" / "config.py"
    offenders = [
        p
        for p in (ROOT / "src" / "precedent").rglob("*.py")
        if p != config and FORBIDDEN_OUTSIDE_CONFIG.search(p.read_text())
    ]
    assert offenders == []


def test_no_secret_shaped_strings_are_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.split()
    leaky = [
        p
        for p in tracked
        if (ROOT / p).is_file() and SECRET_SHAPES.search((ROOT / p).read_text(errors="ignore"))
    ]
    assert leaky == []


def test_ci_pins_actions_by_sha_and_runs_the_offline_checks():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    uses = re.findall(r"uses:\s*(\S+)\s*(#.*)?", ci)
    assert uses
    for ref, comment in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref
        assert comment.startswith("# v"), ref
    assert "secrets." not in ci
    assert "contents: read" in ci
    for step in (
        "uv sync --frozen --all-extras",
        "uv run ruff check .",
        "uv run pytest -q",
        "uv run python scripts/ci_ask_lifecycle.py",
        'test -z "$(git status --porcelain)"',
    ):
        assert step in ci


def test_wheel_ships_the_references():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["force-include"]["references"] == "precedent/references"
    assert "addopts" not in pyproject["tool"]["pytest"]["ini_options"]
