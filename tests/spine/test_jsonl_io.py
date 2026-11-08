"""Tests for atomic JSON Lines writing."""
import os
from pathlib import Path

import pytest

from precedent.spine.jsonl_io import (
    AppendReport,
    atomic_write_lines,
    read_json_state,
    write_json_state,
)


def test_a_write_creates_the_file_with_one_line_per_item(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "x.jsonl"
    atomic_write_lines(target, iter(["a", "b"]))
    assert target.read_text(encoding="utf-8").splitlines() == ["a", "b"]


def test_a_failed_write_leaves_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "x.jsonl"
    atomic_write_lines(target, ["a"])

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_lines(target, ["b"])
    assert target.read_text().splitlines() == ["a"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.jsonl"]


def test_a_failing_line_iterator_leaves_no_temporary_file(tmp_path: Path) -> None:
    def lines():
        yield "a"
        raise RuntimeError("bad source")

    with pytest.raises(RuntimeError):
        atomic_write_lines(tmp_path / "x.jsonl", lines())
    assert list(tmp_path.iterdir()) == []


def test_append_report_is_a_frozen_pair_of_counts() -> None:
    report = AppendReport(added=1, identical_skipped=2)
    with pytest.raises(AttributeError):
        report.added = 5  # type: ignore[misc]


def test_state_files_round_trip_and_a_missing_one_is_empty(tmp_path):
    path = tmp_path / "state" / "x.json"
    assert read_json_state(path) == {}
    write_json_state(path, {"cursor": "c1", "exhausted": False})
    assert read_json_state(path) == {"cursor": "c1", "exhausted": False}


def test_a_corrupt_state_file_is_a_value_error_naming_the_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match=r"x\.json"):
        read_json_state(path)
    path.write_text("[1]")
    with pytest.raises(ValueError, match="JSON object"):
        read_json_state(path)
