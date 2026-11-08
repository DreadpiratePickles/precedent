"""CSV import: strict columns, line-numbered errors, atomic per-symbol Parquet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from precedent.cli import main
from precedent.spine.price_import import PriceImportError, import_prices_csv

TINY_ROWS = [
    "2024-01-02,AAA,100,101",
    "2024-01-03,AAA,101,102",
    "2024-01-04,AAA,102,103",
    "2024-01-05,AAA,103,104",
    "2024-01-02,BBB,50,51",
    "2024-01-04,BBB,52,53",
]


def write_csv(tmp_path: Path, rows: list[str], header: str = "date,symbol,open,close") -> Path:
    path = tmp_path / "prices.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_import_writes_one_parquet_per_symbol_and_a_manifest(tmp_path):
    csv_path = write_csv(tmp_path, TINY_ROWS)
    report = import_prices_csv(csv_path, tmp_path / "prices")
    assert (report.rows, report.symbols) == (6, 2)
    assert (report.first_date, report.last_date) == ("2024-01-02", "2024-01-05")
    assert report.source_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    frame = pd.read_parquet(tmp_path / "prices" / "AAA.parquet")
    assert list(frame.columns) == ["date", "open", "close"]
    assert len(frame) == 4
    manifest = json.loads((tmp_path / "prices" / "MANIFEST.json").read_text())
    assert manifest["source_sha256"] == report.source_sha256
    assert sorted(manifest["symbols"]) == ["AAA", "BBB"]
    assert not list((tmp_path / "prices").glob("*.tmp"))


def test_import_rejects_a_duplicate_row_by_line(tmp_path):
    with pytest.raises(PriceImportError, match="line 3"):  # header is line 1
        import_prices_csv(write_csv(tmp_path, [TINY_ROWS[0], TINY_ROWS[0]]), tmp_path / "prices")


def test_import_rejects_unexpected_columns(tmp_path):
    with pytest.raises(PriceImportError):
        import_prices_csv(
            write_csv(tmp_path, TINY_ROWS, header="date,symbol,open,close,adj_close"),
            tmp_path / "prices",
        )


@pytest.mark.parametrize(
    ("bad_row", "reason"),
    [
        ("not-a-date,AAA,1,2", "date"),
        ("2024-01-03,AAA,,2", "open"),
        ("2024-01-03,AAA,1,", "close"),
        ("2024-01-03,AAA,0,2", "open"),
        ("2024-01-03,AAA,1,-2", "close"),
        ("2024-01-03,AAA,abc,2", "open"),
    ],
)
def test_import_rejects_a_bad_value_naming_the_line(tmp_path, bad_row, reason):
    rows = [TINY_ROWS[0], bad_row]
    with pytest.raises(PriceImportError, match=rf"line 3.*{reason}"):
        import_prices_csv(write_csv(tmp_path, rows), tmp_path / "prices")
    assert not (tmp_path / "prices" / "AAA.parquet").exists()


def test_reimporting_a_symbol_replaces_it(tmp_path):
    prices_dir = tmp_path / "prices"
    import_prices_csv(write_csv(tmp_path, TINY_ROWS), prices_dir)
    import_prices_csv(write_csv(tmp_path, ["2024-02-01,AAA,10,11"]), prices_dir)
    frame = pd.read_parquet(prices_dir / "AAA.parquet")
    assert list(frame["close"]) == [11.0]
    assert (prices_dir / "BBB.parquet").exists()


def test_cli_import_writes_parquet_under_the_data_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    assert main(["prices", "import", str(write_csv(tmp_path, TINY_ROWS))]) == 0
    assert (tmp_path / "prices" / "AAA.parquet").exists()
    assert "rows" in capsys.readouterr().out


def test_cli_import_reports_a_bad_csv_as_exit_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    bad = write_csv(tmp_path, [TINY_ROWS[0], TINY_ROWS[0]])
    assert main(["prices", "import", str(bad)]) == 1
    assert "line 3" in capsys.readouterr().err


def test_cli_without_a_subcommand_is_a_usage_error(capsys):
    assert main(["prices"]) == 2
    assert "usage" in capsys.readouterr().err


@pytest.mark.parametrize("raw_date", ["20240102", "2024-W01-1", "2024-1-2", "2024/01/02"])
def test_only_yyyy_mm_dd_dates_are_accepted(tmp_path: Path, raw_date: str):
    csv_path = tmp_path / "dates.csv"
    csv_path.write_text(f"date,symbol,open,close\n{raw_date},AAA,10,11\n", encoding="utf-8")
    with pytest.raises(PriceImportError, match="line 2.*YYYY-MM-DD"):
        import_prices_csv(csv_path, tmp_path / "prices")
