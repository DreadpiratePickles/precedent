"""Import a `date,symbol,open,close` CSV into one Parquet file per symbol, atomically."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["date", "symbol", "open", "close"]
MANIFEST_NAME = "MANIFEST.json"
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._^-]*$")


class PriceImportError(ValueError):
    """The CSV is not an acceptable price file; the message names the line."""


@dataclass(frozen=True)
class PriceImportReport:
    rows: int
    symbols: int
    first_date: str
    last_date: str
    source_sha256: str


def _parse_price(raw: str, field: str, line: int) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise PriceImportError(f"line {line}: {field} {raw!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise PriceImportError(f"line {line}: {field} {raw!r} must be a positive price")
    return value


def _parse_rows(csv_path: Path, text: str) -> dict[str, list[tuple[date, float, float]]]:
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header is None or [name.strip() for name in header] != EXPECTED_COLUMNS:
        raise PriceImportError(
            f"{csv_path.name}: line 1: columns must be exactly {','.join(EXPECTED_COLUMNS)}"
        )
    by_symbol: dict[str, list[tuple[date, float, float]]] = {}
    seen: set[tuple[date, str]] = set()
    for line, fields in enumerate(reader, start=2):
        if not fields:
            continue
        if len(fields) != len(EXPECTED_COLUMNS):
            raise PriceImportError(f"line {line}: expected 4 fields, found {len(fields)}")
        raw_date, raw_symbol, raw_open, raw_close = (field.strip() for field in fields)
        try:
            day = date.fromisoformat(raw_date)
        except ValueError:
            raise PriceImportError(f"line {line}: date {raw_date!r} is not YYYY-MM-DD") from None
        symbol = raw_symbol.upper()
        if not _SYMBOL_PATTERN.match(symbol):
            raise PriceImportError(f"line {line}: symbol {raw_symbol!r} is not a valid symbol")
        open_price = _parse_price(raw_open, "open", line)
        close_price = _parse_price(raw_close, "close", line)
        if (day, symbol) in seen:
            raise PriceImportError(f"line {line}: duplicate row for {symbol} on {day}")
        seen.add((day, symbol))
        by_symbol.setdefault(symbol, []).append((day, open_price, close_price))
    if not by_symbol:
        raise PriceImportError(f"{csv_path.name}: no data rows")
    return by_symbol


def _write_atomically(path: Path, write: object) -> None:
    """Write through a temporary sibling and `os.replace`, so a reader never sees a partial file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)  # type: ignore[operator]
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest_symbols(prices_dir: Path) -> dict[str, dict[str, object]]:
    path = prices_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("symbols", {})


def import_prices_csv(csv_path: Path, prices_dir: Path) -> PriceImportReport:
    """Validate the whole CSV, then write `<SYMBOL>.parquet` per symbol and `MANIFEST.json`."""
    try:
        raw = csv_path.read_bytes()
    except OSError as error:
        raise PriceImportError(f"cannot read {csv_path}: {error}") from error
    source_sha256 = hashlib.sha256(raw).hexdigest()
    by_symbol = _parse_rows(csv_path, raw.decode("utf-8-sig"))

    prices_dir.mkdir(parents=True, exist_ok=True)
    manifest_symbols = _read_manifest_symbols(prices_dir)
    all_days: list[date] = []
    total_rows = 0
    for symbol, records in sorted(by_symbol.items()):
        frame = pd.DataFrame(records, columns=["date", "open", "close"]).sort_values("date")
        frame["date"] = pd.to_datetime(frame["date"])
        _write_atomically(prices_dir / f"{symbol}.parquet", lambda p, f=frame: f.to_parquet(p))
        first, last = frame["date"].min().date(), frame["date"].max().date()
        all_days += [first, last]
        total_rows += len(frame)
        manifest_symbols[symbol] = {
            "rows": len(frame),
            "first_date": first.isoformat(),
            "last_date": last.isoformat(),
            "source_sha256": source_sha256,
        }
    report = PriceImportReport(
        rows=total_rows,
        symbols=len(by_symbol),
        first_date=min(all_days).isoformat(),
        last_date=max(all_days).isoformat(),
        source_sha256=source_sha256,
    )
    manifest = {"source_sha256": source_sha256, "symbols": manifest_symbols}
    _write_atomically(
        prices_dir / MANIFEST_NAME,
        lambda p: p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"),
    )
    return report
