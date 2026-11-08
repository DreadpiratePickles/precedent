"""The selector builds the model typers once their modules exist."""
from precedent.spine.typers.anthropic_typer import AnthropicTyper
from precedent.spine.typers.ollama_typer import OllamaTyper
from precedent.spine.typers.select import typer_from_env


def test_selector_builds_the_anthropic_typer(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    typer = typer_from_env()
    assert isinstance(typer, AnthropicTyper) and typer.calls_made == 0


def test_selector_passes_max_calls_and_dry_run_through(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    typer = typer_from_env(max_calls=0, dry_run=True)
    assert typer.type_text("Fed cuts rates").reason == "dry-run"
    live = typer_from_env(max_calls=0)
    assert live.type_text("Fed cuts rates").reason == "max_calls reached"


def test_selector_builds_the_ollama_typer(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "ollama")
    monkeypatch.setenv("PRECEDENT_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("PRECEDENT_OLLAMA_MODEL", "a-local-model")
    assert isinstance(typer_from_env(max_calls=5), OllamaTyper)
