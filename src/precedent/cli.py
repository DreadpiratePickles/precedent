"""Console script `precedent`. Tracks are imported lazily and never edit this file."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence

TRACK_COMMANDS: dict[str, str] = {
    "ask": "precedent.engine.ask:run",
    "fetch": "precedent.engine.snapshot:run",
    "build": "precedent.libraries.build:run",
    "evaluate": "precedent.engine.evaluate:run",
    "mcp": "precedent.mcp_server:run",
    "ingest gdelt": "precedent.spine.ingest.gdelt:run",
    "ingest polymarket": "precedent.spine.ingest.polymarket:run",
    "ingest kalshi": "precedent.spine.ingest.kalshi:run",
    "ingest edgar": "precedent.spine.ingest.edgar:run",
    "prices": "precedent.spine.prices_cli:run",
    "typers": "precedent.spine.typers.cli:run",
    "dream": "precedent.dream.cli:run",
    "ripple": "precedent.ripple.report:run",
    "mechanism": "precedent.mechanism.cli:run",
}
RESEARCH_PACKAGES = frozenset({"qlib", "lightgbm", "scipy", "yaml", "networkx"})
USAGE_ERROR = 2

Runner = Callable[[list[str]], int]


def resolve_command(argv: list[str]) -> tuple[str, list[str]]:
    """Map argv to (command, remaining arguments); a non-command first word means `ask`."""
    if not argv:
        return "ask", []
    first = argv[0]
    if first == "ingest":
        if len(argv) >= 2 and f"ingest {argv[1]}" in TRACK_COMMANDS:
            return f"ingest {argv[1]}", argv[2:]
        return "ingest", argv[1:]
    if first in TRACK_COMMANDS:
        return first, argv[1:]
    return "ask", argv


def _load_target(target: str) -> Runner:
    module_name, _, function_name = target.partition(":")
    return getattr(importlib.import_module(module_name), function_name)


def _ingest_usage() -> str:
    sources = sorted(c.split()[1] for c in TRACK_COMMANDS if c.startswith("ingest "))
    return f"usage: precedent ingest {{{'|'.join(sources)}}} [options]"


def _is_missing_target(error: ModuleNotFoundError, target: str) -> bool:
    """True when the module that failed to import is the target module or one of its parents."""
    missing = error.name or ""
    return target.partition(":")[0] == missing or target.startswith(missing + ".")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = resolve_command(args)
    if command == "ingest":
        print(_ingest_usage(), file=sys.stderr)
        return USAGE_ERROR
    target = TRACK_COMMANDS[command]
    try:
        return _load_target(target)(rest)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] in RESEARCH_PACKAGES:
            print(
                "needs the research extra: pip install 'precedent-engine[research]'",
                file=sys.stderr,
            )
            return USAGE_ERROR
        if _is_missing_target(error, target):
            print(f"not built yet: {command}", file=sys.stderr)
            return USAGE_ERROR
        raise


if __name__ == "__main__":
    raise SystemExit(main())
