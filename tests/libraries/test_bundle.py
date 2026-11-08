"""Tests for writing and loading the verified library bundle."""

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from library_synthetic import CUTOFFS, SESSIONS

from precedent.libraries.bundle import (
    LibraryBundle,
    LibraryMissingError,
    load_bundle,
    write_bundle,
)
from precedent.libraries.common import LibrarySchemaError, validate_library_rows
from precedent.libraries.manifest import LibraryIntegrityError, Manifest

SORT = ["outcome_type", "event_id", "outcome_key", "horizon"]
BUNDLE_FILES = [
    "build_report.json",
    "events.parquet",
    "markets.parquet",
    "placebo_asset.parquet",
    "rows_asset.parquet",
    "rows_event_sequence.parquet",
    "rows_market_resolution.parquet",
    "sessions.parquet",
]


def test_bundle_round_trips_and_lists_every_file(tmp_path, bundle_parts):
    manifest = write_bundle(tmp_path / "library", **bundle_parts)
    loaded = load_bundle(tmp_path / "library")
    pd.testing.assert_frame_equal(
        loaded.rows.sort_values(SORT).reset_index(drop=True),
        validate_library_rows(bundle_parts["rows"]).sort_values(SORT).reset_index(drop=True),
    )
    assert sorted(f.path for f in manifest.files) == sorted(BUNDLE_FILES)


def test_a_tampered_file_is_refused_before_it_is_parsed(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    (tmp_path / "library" / "rows_event_sequence.parquet").write_bytes(b"not parquet")
    with pytest.raises(LibraryIntegrityError, match="rows_event_sequence.parquet"):
        load_bundle(tmp_path / "library")


def test_an_unlisted_file_is_refused(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    (tmp_path / "library" / "extra.parquet").write_bytes(b"x")
    with pytest.raises(LibraryIntegrityError, match="extra.parquet"):
        load_bundle(tmp_path / "library")


def test_missing_library_is_its_own_error(tmp_path):
    with pytest.raises(LibraryMissingError):
        load_bundle(tmp_path / "nothing")


def test_missing_library_error_is_a_file_not_found_error(tmp_path):
    assert issubclass(LibraryMissingError, FileNotFoundError)
    (tmp_path / "library").mkdir()
    with pytest.raises(FileNotFoundError, match="MANIFEST.json"):
        load_bundle(tmp_path / "library")


def test_every_table_comes_back_as_written(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    loaded = load_bundle(tmp_path / "library")
    assert isinstance(loaded, LibraryBundle)
    pd.testing.assert_frame_equal(loaded.events, bundle_parts["events"])
    pd.testing.assert_frame_equal(loaded.placebo_asset, bundle_parts["placebo_asset"])
    pd.testing.assert_frame_equal(loaded.markets, bundle_parts["markets"])
    pd.testing.assert_index_equal(
        loaded.sessions, SESSIONS.as_unit("ns"), check_names=False, check_freq=False
    )
    assert loaded.sessions.tz is None and loaded.sessions.is_monotonic_increasing
    pd.testing.assert_series_equal(
        loaded.session_cutoffs_utc, CUTOFFS, check_freq=False, check_index_type=False
    )
    assert loaded.session_cutoffs_utc.index.equals(loaded.sessions)
    assert loaded.session_cutoffs_utc.index.name == "session_date"


def test_linked_countries_come_back_as_python_lists(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    countries = load_bundle(tmp_path / "library").events["linked_countries"]
    assert all(type(item) is list for item in countries)
    assert ["USA", "CHN"] in list(countries)


def test_rows_split_by_type_and_an_empty_type_still_has_a_file(tmp_path, bundle_parts):
    rows = bundle_parts["rows"]
    parts = {**bundle_parts, "rows": rows[rows["outcome_type"] == "event_sequence"]}
    manifest = write_bundle(tmp_path / "library", **parts)
    counts = {f.path: f.rows for f in manifest.files}
    assert counts["rows_event_sequence.parquet"] == 3
    assert counts["rows_asset.parquet"] == 0 and counts["rows_market_resolution.parquet"] == 0
    loaded = load_bundle(tmp_path / "library")
    assert set(loaded.rows["outcome_type"]) == {"event_sequence"}


def test_manifest_counts_hashes_and_metadata(tmp_path, bundle_parts):
    library = tmp_path / "library"
    manifest = write_bundle(library, **bundle_parts)
    by_path = {f.path: f for f in manifest.files}
    assert by_path["events.parquet"].rows == len(bundle_parts["events"])
    assert by_path["sessions.parquet"].rows == len(SESSIONS)
    assert by_path["markets.parquet"].rows == 3 and by_path["placebo_asset.parquet"].rows == 3
    assert by_path["build_report.json"].rows is None
    for entry in manifest.files:
        assert entry.sha256 == hashlib.sha256((library / entry.path).read_bytes()).hexdigest()
    assert manifest.built_from_commit == "0" * 40
    assert manifest.sources == bundle_parts["sources"]
    assert manifest.created_utc.tzinfo is not None
    assert abs(datetime.now(UTC) - manifest.created_utc).total_seconds() < 60
    assert manifest.precedent_version
    assert Manifest.model_validate_json((library / "MANIFEST.json").read_text()) == manifest
    assert load_bundle(library).manifest == manifest


def test_build_report_is_written_as_sorted_json(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    text = (tmp_path / "library" / "build_report.json").read_text()
    assert json.loads(text) == bundle_parts["build_report"]
    assert text.index('"events"') < text.index('"note"') and '\n  "events"' in text


def test_writing_over_an_existing_library_replaces_it_atomically(tmp_path, bundle_parts):
    library = tmp_path / "library"
    write_bundle(library, **bundle_parts)
    (library / "stale.txt").write_text("old")  # not listed: the old library is now invalid
    write_bundle(library, **{**bundle_parts, "build_report": {"second": True}})
    assert not (library / "stale.txt").exists()
    assert json.loads((library / "build_report.json").read_text()) == {"second": True}
    assert [p.name for p in tmp_path.iterdir()] == ["library"]
    load_bundle(library)


def test_a_failed_write_leaves_the_existing_library_and_no_temp_dir(tmp_path, bundle_parts):
    library = tmp_path / "library"
    write_bundle(library, **bundle_parts)
    before = sorted((p.name, p.read_bytes()) for p in library.iterdir())
    with pytest.raises(ValueError, match="built_from_commit"):
        write_bundle(library, **{**bundle_parts, "commit": "abc123"})
    assert sorted((p.name, p.read_bytes()) for p in library.iterdir()) == before
    assert [p.name for p in tmp_path.iterdir()] == ["library"]


def test_invalid_rows_are_refused_before_anything_is_written(tmp_path, bundle_parts):
    bad = bundle_parts["rows"].copy()
    bad.loc[bad.index[0], "value"] = 0.5  # a binary outcome that is neither 0 nor 1
    with pytest.raises(LibrarySchemaError):
        write_bundle(tmp_path / "library", **{**bundle_parts, "rows": bad})
    assert list(tmp_path.iterdir()) == []


def test_sessions_and_cutoffs_must_line_up(tmp_path, bundle_parts):
    with pytest.raises(ValueError, match="session"):
        write_bundle(
            tmp_path / "library", **{**bundle_parts, "session_cutoffs_utc": CUTOFFS.iloc[:-1]}
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("table", ["events", "placebo_asset", "markets"])
def test_a_table_with_the_wrong_columns_is_refused(tmp_path, bundle_parts, table):
    broken = bundle_parts[table].drop(columns=[bundle_parts[table].columns[0]])
    with pytest.raises(ValueError, match="columns"):
        write_bundle(tmp_path / "library", **{**bundle_parts, table: broken})
    assert list(tmp_path.iterdir()) == []


def test_a_manifest_that_is_not_valid_is_an_integrity_error(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    (tmp_path / "library" / "MANIFEST.json").write_text("{not json")
    with pytest.raises(LibraryIntegrityError, match="MANIFEST.json"):
        load_bundle(tmp_path / "library")


def test_a_missing_listed_file_is_an_integrity_error(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    (tmp_path / "library" / "markets.parquet").unlink()
    with pytest.raises(LibraryIntegrityError, match="markets.parquet"):
        load_bundle(tmp_path / "library")


def test_an_empty_events_table_round_trips(tmp_path, bundle_parts):
    empty = bundle_parts["events"].iloc[0:0]
    write_bundle(tmp_path / "library", **{**bundle_parts, "events": empty})
    loaded = load_bundle(tmp_path / "library")
    assert loaded.events.empty and list(loaded.events.columns) == list(empty.columns)
