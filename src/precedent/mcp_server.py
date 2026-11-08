"""Stub for the `precedent-mcp` console script until the Precedent track builds the server.

It exists so the installed script prints one line and exits 2 instead of a traceback.
The Precedent track replaces this module.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

NOT_BUILT_EXIT_CODE = 2


def run(argv: list[str]) -> int:
    """Report that the MCP server is not built yet; `argv` is accepted and ignored."""
    print("not built yet: mcp", file=sys.stderr)
    return NOT_BUILT_EXIT_CODE


def main(argv: Sequence[str] | None = None) -> int:
    return run(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
