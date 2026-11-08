"""Build the committed sample library from a built library: `data/sample/` (owner gate).

Usage: `uv run python scripts/build_sample_library.py --since 2024-07-01 --until 2024-08-31`.
Reads `<data dir>/library/`, writes `<out>/library/` and `<out>/README.md` labelled LIVE, and
prints the README's first line. Exit 2 with one stderr line when the library is missing or
altered or the sample is over the size limit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from precedent.config import data_dir
from precedent.libraries.bundle import LibraryMissingError
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.libraries.sample import (
    GDELT_CITATION,
    MAX_SAMPLE_BYTES,
    SampleTooLargeError,
    build_sample,
)

USAGE_ERROR = 2
DEFAULT_OUT = "data/sample"
EVIDENCE_LABEL = "LIVE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the committed sample library.")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PRECEDENT_DATA_DIR")
    parser.add_argument("--since", type=date.fromisoformat, required=True, help="ISO date")
    parser.add_argument("--until", type=date.fromisoformat, required=True, help="ISO date")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--citation", default=GDELT_CITATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.since > args.until:
        parser.error(f"--since {args.since} is after --until {args.until}")
    source_dir = (args.data_dir if args.data_dir is not None else data_dir()) / "library"
    try:
        build_sample(
            source_dir,
            args.out,
            args.since,
            args.until,
            EVIDENCE_LABEL,
            args.citation,
            max_bytes=MAX_SAMPLE_BYTES,
        )
    except (LibraryMissingError, LibraryIntegrityError, SampleTooLargeError) as error:
        print(f"build_sample_library: {error}", file=sys.stderr)
        return USAGE_ERROR
    print((args.out / "README.md").read_text(encoding="utf-8").splitlines()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
