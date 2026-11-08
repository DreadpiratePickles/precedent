"""Tests for choosing a typer from the environment."""
import pytest

from precedent.config import ConfigError
from precedent.spine.typers.rules import RulesTyper
from precedent.spine.typers.select import typer_from_env


def test_default_typer_is_rules(monkeypatch):
    monkeypatch.delenv("PRECEDENT_TYPER", raising=False)
    assert isinstance(typer_from_env(), RulesTyper)


def test_rules_ignores_max_calls_and_dry_run(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    assert isinstance(typer_from_env(max_calls=3, dry_run=True), RulesTyper)


def test_an_unknown_typer_is_a_config_error(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "gpt")
    with pytest.raises(ConfigError):
        typer_from_env()


def test_a_model_typer_whose_module_is_not_built_is_a_config_error_naming_it(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "ollama")
    import importlib.util

    if importlib.util.find_spec("precedent.spine.typers.ollama_typer") is not None:
        pytest.skip("ollama typer module exists; covered by its own tests")
    with pytest.raises(ConfigError, match="ollama_typer"):
        typer_from_env(max_calls=1)
