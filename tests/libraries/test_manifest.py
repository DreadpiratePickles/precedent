"""Tests for the library manifest and its file verification."""

import hashlib
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from precedent.libraries.manifest import (
    LibraryIntegrityError,
    Manifest,
    ManifestFile,
    ManifestSource,
    sha256_file,
    verify_files,
)

SHA_ABC = hashlib.sha256(b"abc").hexdigest()
MANIFEST_DICT = {
    "files": [{"path": "a.parquet", "sha256": SHA_ABC, "rows": 3}],
    "sources": [
        {
            "name": "gdelt2",
            "first_date": date(2021, 1, 4),
            "last_date": date(2021, 3, 1),
            "rows": 30,
            "citation": "synthetic citation",
        }
    ],
    "built_from_commit": "0" * 40,
    "precedent_version": "0.1.0",
    "created_utc": datetime(2026, 11, 7, 12, 0, tzinfo=UTC),
}


def _manifest(**changes) -> Manifest:
    return Manifest(**{**MANIFEST_DICT, **changes})


def test_manifest_rejects_a_short_commit():
    with pytest.raises(ValidationError):
        Manifest(**{**MANIFEST_DICT, "built_from_commit": "abc123"})


@pytest.mark.parametrize("commit", ["unknown", "0123456789abcdef" * 2 + "01234567"])
def test_manifest_accepts_unknown_or_a_full_lowercase_hash(commit):
    assert _manifest(built_from_commit=commit).built_from_commit == commit


@pytest.mark.parametrize("commit", ["A" * 40, "g" * 40, "0" * 41, "Unknown", ""])
def test_manifest_rejects_other_commit_spellings(commit):
    with pytest.raises(ValidationError):
        _manifest(built_from_commit=commit)


def test_manifest_requires_a_timezone_aware_creation_time():
    with pytest.raises(ValidationError):
        _manifest(created_utc=datetime(2026, 11, 7, 12, 0))


def test_manifest_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Manifest(**{**MANIFEST_DICT, "extra": 1})
    with pytest.raises(ValidationError):
        ManifestFile(path="a", sha256=SHA_ABC, rows=None, extra=1)


@pytest.mark.parametrize("sha", ["abc", "G" * 64, SHA_ABC.upper(), SHA_ABC + "0"])
def test_file_entry_needs_a_lowercase_sha256(sha):
    with pytest.raises(ValidationError):
        ManifestFile(path="a.parquet", sha256=sha, rows=None)


@pytest.mark.parametrize("path", ["", "/etc/passwd", "../x", "a/../../x", "a\\b", "MANIFEST.json"])
def test_file_entry_refuses_paths_that_could_leave_the_library(path):
    with pytest.raises(ValidationError):
        ManifestFile(path=path, sha256=SHA_ABC, rows=None)


def test_manifest_refuses_a_file_listed_twice():
    entry = {"path": "a.parquet", "sha256": SHA_ABC, "rows": 1}
    with pytest.raises(ValidationError):
        _manifest(files=[entry, entry])


def test_source_dates_must_be_ordered_and_rows_not_negative():
    good = MANIFEST_DICT["sources"][0]
    with pytest.raises(ValidationError):
        ManifestSource(**{**good, "first_date": date(2021, 3, 2)})
    with pytest.raises(ValidationError):
        ManifestSource(**{**good, "rows": -1})


def test_sha256_file_hashes_the_bytes(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    assert sha256_file(tmp_path / "f") == SHA_ABC


def _library(tmp_path, files: dict[str, bytes]) -> Manifest:
    entries = []
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append({"path": name, "sha256": hashlib.sha256(content).hexdigest(), "rows": None})
    return _manifest(files=entries)


def test_verify_files_accepts_a_matching_library_and_ignores_the_manifest_file(tmp_path):
    manifest = _library(tmp_path, {"a.parquet": b"one", "b.json": b"{}"})
    (tmp_path / "MANIFEST.json").write_text("{}")
    verify_files(tmp_path, manifest)


def test_verify_files_names_a_missing_file(tmp_path):
    manifest = _library(tmp_path, {"a.parquet": b"one"})
    (tmp_path / "a.parquet").unlink()
    with pytest.raises(LibraryIntegrityError, match="a.parquet") as caught:
        verify_files(tmp_path, manifest)
    assert caught.value.path == "a.parquet" and "missing" in caught.value.reason


def test_verify_files_names_a_changed_file(tmp_path):
    manifest = _library(tmp_path, {"a.parquet": b"one"})
    (tmp_path / "a.parquet").write_bytes(b"two")
    with pytest.raises(LibraryIntegrityError, match="a.parquet") as caught:
        verify_files(tmp_path, manifest)
    assert "sha256" in caught.value.reason


def test_verify_files_refuses_an_unlisted_file_at_any_depth(tmp_path):
    manifest = _library(tmp_path, {"a.parquet": b"one"})
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "extra.bin").write_bytes(b"x")
    with pytest.raises(LibraryIntegrityError, match="extra.bin") as caught:
        verify_files(tmp_path, manifest)
    assert caught.value.path == "sub/extra.bin"


def test_verify_files_refuses_a_symlink(tmp_path):
    manifest = _library(tmp_path, {"a.parquet": b"one"})
    outside = tmp_path / "outside"
    outside.write_bytes(b"one")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "a.parquet").symlink_to(outside)
    with pytest.raises(LibraryIntegrityError, match="a.parquet"):
        verify_files(tmp_path / "lib", manifest)
