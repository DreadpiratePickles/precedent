"""Choose a typer by name; `PRECEDENT_TYPER` picks one for callers that read the environment."""

from __future__ import annotations

import importlib
import os

from precedent.config import ConfigError
from precedent.spine.typers.base import Typer
from precedent.spine.typers.rules import RulesTyper

_MODEL_MODULES = {
    "anthropic": "precedent.spine.typers.anthropic_typer",
    "ollama": "precedent.spine.typers.ollama_typer",
}
TYPER_NAMES = ("rules", *_MODEL_MODULES)


def typer_by_name(name: str, max_calls: int | None = None, dry_run: bool = False) -> Typer:
    """`rules` ignores both arguments; model typers get them in their own `from_env`."""
    if name == "rules":
        return RulesTyper()
    module_name = _MODEL_MODULES.get(name)
    if module_name is None:
        raise ConfigError(f"unknown typer {name!r}; choose one of {', '.join(TYPER_NAMES)}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            message = f"typer {name!r} needs {module_name}, which is not built yet"
            raise ConfigError(message) from error
        raise
    return module.from_env(max_calls=max_calls, dry_run=dry_run)


def typer_from_env(max_calls: int | None = None, dry_run: bool = False) -> Typer:
    return typer_by_name(os.environ.get("PRECEDENT_TYPER") or "rules", max_calls, dry_run)
