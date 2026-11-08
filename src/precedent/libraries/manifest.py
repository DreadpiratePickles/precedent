"""The library manifest: every file of a bundle with its hash, and the check that they match."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "MANIFEST_NAME",
    "LibraryIntegrityError",
    "Manifest",
    "ManifestFile",
    "ManifestSource",
    "sha256_file",
    "verify_files",
]

MANIFEST_NAME = "MANIFEST.json"
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|unknown)$")
_READ_CHUNK_BYTES = 1 << 20

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class LibraryIntegrityError(Exception):
    """A library file is missing, changed since it was listed, or not listed at all."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class ManifestFile(BaseModel):
    """One file of the bundle; `rows` is None for a file that is not a table."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: Sha256Hex
    rows: int | None = Field(ge=0)

    @field_validator("path")
    @classmethod
    def _path_stays_inside_the_library(cls, value: str) -> str:
        parts = PurePosixPath(value).parts
        if not value or "\\" in value or PurePosixPath(value).is_absolute() or ".." in parts:
            raise ValueError(f"a manifest path must stay inside the library: {value!r}")
        if value == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} does not list itself")
        return value


class ManifestSource(BaseModel):
    """Where some of the library's records came from, and the citation it requires."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    first_date: date
    last_date: date
    rows: int = Field(ge=0)
    citation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _dates_are_ordered(self) -> ManifestSource:
        if self.first_date > self.last_date:
            raise ValueError(f"source {self.name}: first_date is after last_date")
        return self


class Manifest(BaseModel):
    """The files, sources and provenance of one built library."""

    model_config = ConfigDict(extra="forbid")

    files: list[ManifestFile]
    sources: list[ManifestSource]
    built_from_commit: str
    precedent_version: str = Field(min_length=1)
    created_utc: AwareDatetime

    @field_validator("built_from_commit")
    @classmethod
    def _commit_is_a_full_hash_or_unknown(cls, value: str) -> str:
        if not _COMMIT_PATTERN.match(value):
            raise ValueError(
                f"built_from_commit must be 40 lowercase hex digits or 'unknown', got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _each_file_is_listed_once(self) -> Manifest:
        paths = [entry.path for entry in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("a file is listed more than once in the manifest")
        return self


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, manifest: Manifest) -> None:
    """Raise `LibraryIntegrityError` unless `root` holds exactly the listed files, unchanged.

    A listed file that is absent, is a symbolic link, or has another hash fails; so does any
    file under `root` (at any depth) that the manifest does not list, `MANIFEST.json` excepted.
    Nothing is parsed: this is the check that runs before a parquet file is opened.
    """
    listed = {entry.path for entry in manifest.files}
    for entry in sorted(manifest.files, key=lambda item: item.path):
        target = root / entry.path
        if target.is_symlink():
            raise LibraryIntegrityError(entry.path, "is a symbolic link")
        if not target.is_file():
            raise LibraryIntegrityError(entry.path, "listed in the manifest but missing")
        if sha256_file(target) != entry.sha256:
            raise LibraryIntegrityError(entry.path, "sha256 differs from the manifest")
    for found in sorted(root.rglob("*")):
        if found.is_dir() and not found.is_symlink():
            continue
        relative = found.relative_to(root).as_posix()
        if relative not in listed and relative != MANIFEST_NAME:
            raise LibraryIntegrityError(relative, "present in the library but not in the manifest")
