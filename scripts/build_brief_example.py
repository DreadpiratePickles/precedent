"""Rebuild `docs/examples/brief_2024-06-03.md` from the synthetic library (Ruling D58).

Usage: `uv run python scripts/build_brief_example.py [--out docs/examples]`. The data directory is
built in a temporary folder from `tests/library_root_synthetic.py`, the typer is the deterministic
rules typer, and no model or network is used, so a fresh clone reproduces the file byte for byte.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from library_root_synthetic import write_library_root  # noqa: E402

from precedent.dream import brief  # noqa: E402

SESSION = "2024-06-03"
FIXTURES = ROOT / "tests" / "dream" / "fixtures"
README_LINE = "SYNTHETIC — generated library for the brief example; not market data.\n"


def build(out_dir: Path) -> int:
    """Write `brief_2024-06-03.md` under `out_dir`; returns the exit code of `brief.run`."""
    previous = os.environ.get("PRECEDENT_TYPER")
    os.environ["PRECEDENT_TYPER"] = "rules"
    try:
        with tempfile.TemporaryDirectory() as scratch:
            data_dir = write_library_root(Path(scratch))
            (data_dir / "README.md").write_text(README_LINE, encoding="utf-8")
            return brief.run(
                [
                    "--date", SESSION,
                    "--headlines", str(FIXTURES / f"headlines_{SESSION}.txt"),
                    "--dream-dir", str(FIXTURES),
                    "--data-dir", str(data_dir),
                    "--out", str(out_dir),
                ]
            )  # fmt: skip
    finally:
        if previous is None:
            os.environ.pop("PRECEDENT_TYPER", None)
        else:
            os.environ["PRECEDENT_TYPER"] = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "examples")
    return build(parser.parse_args().out)


if __name__ == "__main__":
    raise SystemExit(main())
