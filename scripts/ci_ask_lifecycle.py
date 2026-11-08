"""CI check for the product command: `precedent ask --dry-run` on a fixed headline.

While `precedent.engine.ask` does not exist the command must exit 2 with `not built yet` on
stderr. Once it exists the command must exit 0 and print the not-a-forecast line on stdout.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

HEADLINE = "Fed unexpectedly cuts rates in emergency move"
NOT_A_FORECAST_LINE = "These are past outcomes of similar events, not a forecast."
NOT_BUILT_MESSAGE = "ask: not built yet (expected before the Precedent track)"


def _ask_module_exists() -> bool:
    try:
        return importlib.util.find_spec("precedent.engine.ask") is not None
    except ModuleNotFoundError:
        return False


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "precedent.cli", "ask", "--dry-run", HEADLINE],
        capture_output=True,
        text=True,
        check=False,
    )
    if not _ask_module_exists():
        if result.returncode != 2 or "not built yet" not in result.stderr:
            print(
                f"expected exit 2 and 'not built yet' on stderr; got exit {result.returncode}, "
                f"stderr: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        print(NOT_BUILT_MESSAGE)
        return 0
    if result.returncode != 0 or NOT_A_FORECAST_LINE not in result.stdout.splitlines():
        print(
            f"expected exit 0 and the not-a-forecast line on stdout; got exit "
            f"{result.returncode}, stdout: {result.stdout.strip()}",
            file=sys.stderr,
        )
        return 1
    print("ask: dry run ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
