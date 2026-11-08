import importlib

import pytest

from precedent.dream import cli
from precedent.dream.cli import run as dream_run


def test_dream_cli_defaults_to_report_and_rejects_unknown(tmp_path, populated_stores, capsys):
    assert dream_run(["--date", "2024-06-03", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "dreams" / "dream_2024-06-03.md").exists()
    assert dream_run(["nonsense"]) == 2
    err = capsys.readouterr().err
    assert "nonsense" in err and "report" in err and "brief" in err  # the list is printed


def test_the_report_word_and_the_bare_flags_reach_the_same_command(tmp_path, populated_stores):
    assert dream_run(["report", "--date", "2024-06-03", "--out", str(tmp_path / "a")]) == 0
    assert dream_run(["--date", "2024-06-03", "--out", str(tmp_path / "b")]) == 0
    name = "dreams/dream_2024-06-03.md"
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert dream_run([]) == 2  # no argument at all: report, which needs --date


@pytest.mark.parametrize("word", ["promote", "review"])
def test_a_subcommand_that_is_not_built_yet_says_so_and_returns_two(word, monkeypatch, capsys):
    real = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "precedent.dream.promote":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    assert dream_run([word]) == 2
    assert capsys.readouterr().err.strip() == f"not built yet: dream {word}"


def test_a_missing_dependency_inside_a_subcommand_is_not_reported_as_not_built(monkeypatch):
    real = importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "precedent.dream.brief":
            raise ModuleNotFoundError("No module named 'nothing_here'", name="nothing_here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="nothing_here"):
        dream_run(["brief"])


def test_the_remaining_arguments_reach_the_subcommand(monkeypatch):
    seen = []

    class FakeModule:
        @staticmethod
        def run_review(argv):
            seen.append(argv)
            return 7

    real = importlib.import_module
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name, *a, **k: (
            FakeModule if name == "precedent.dream.promote" else real(name, *a, **k)
        ),
    )
    assert dream_run(["review", "--tension", "abc"]) == 7
    assert seen == [["--tension", "abc"]]


@pytest.mark.parametrize(
    ("word", "module", "function"),
    [
        ("promote", "promote", "run_promote"),
        ("review", "promote", "run_review"),
        ("brief", "brief", "run"),
        ("report", "report", "run"),
    ],
)
def test_each_built_subcommand_reaches_its_function(word, module, function, monkeypatch):
    seen = []
    target = importlib.import_module(f"precedent.dream.{module}")
    monkeypatch.setattr(target, function, lambda argv: seen.append(argv) or 5)
    assert dream_run([word, "--x", "1"]) == 5
    assert seen == [["--x", "1"]]
