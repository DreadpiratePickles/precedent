"""`precedent dream`: route to the Dream subcommands, importing each only when it is asked for.

`report` is the default, so `precedent dream --date 2024-06-03` is the nightly dream. The other
words belong to later tasks; a subcommand whose module does not exist yet says so and exits 2.
"""

from __future__ import annotations

import importlib
import sys

USAGE_ERROR = 2
DEFAULT_SUBCOMMAND = "report"
# subcommand -> (module under precedent.dream, function taking the remaining arguments)
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "report": ("report", "run"),
    "promote": ("promote", "run_promote"),
    "review": ("promote", "run_review"),
    "brief": ("brief", "run"),
}


def run(argv: list[str]) -> int:
    """Run the subcommand named by the first argument; 2 for an unknown or unbuilt one."""
    if not argv or argv[0].startswith("-"):
        word, rest = DEFAULT_SUBCOMMAND, argv
    else:
        word, rest = argv[0], argv[1:]
    if word not in SUBCOMMANDS:
        print(
            f"unknown dream subcommand {word!r}; use one of: {', '.join(SUBCOMMANDS)}",
            file=sys.stderr,
        )
        return USAGE_ERROR
    module_stem, function_name = SUBCOMMANDS[word]
    module_name = f"precedent.dream.{module_stem}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            print(f"not built yet: dream {word}", file=sys.stderr)
            return USAGE_ERROR
        raise
    return getattr(module, function_name)(rest)
