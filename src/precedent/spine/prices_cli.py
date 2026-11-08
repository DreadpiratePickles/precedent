"""`precedent prices import <csv>`: load a CSV of daily prices into the data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from precedent.config import ConfigError, data_dir
from precedent.spine.price_import import PriceImportError, import_prices_csv

USAGE_ERROR = 2
FAILURE = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precedent prices")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="import a date,symbol,open,close CSV")
    importer.add_argument("csv", type=Path)
    return parser


def run(argv: list[str]) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else USAGE_ERROR
    try:
        report = import_prices_csv(args.csv, data_dir() / "prices")
    except (PriceImportError, ConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return FAILURE
    print(f"rows: {report.rows}")
    print(f"symbols: {report.symbols}")
    print(f"first_date: {report.first_date}")
    print(f"last_date: {report.last_date}")
    print(f"source_sha256: {report.source_sha256}")
    return 0
