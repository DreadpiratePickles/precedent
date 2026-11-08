"""The library snapshot: build a reproducible zip of a bundle, and install a downloaded one.

`build_snapshot` verifies the bundle against its manifest and packs it into
`precedent-library-<tag>.zip` (entries sorted, fixed 1980-01-01 timestamps, fixed permissions,
stored uncompressed, so the same bundle always gives the same bytes), with `MANIFEST.json` inside
the zip and beside it. `fetch_snapshot` downloads the manifest and the zip, extracts into a
temporary directory inside the destination, and renames it to `<dest>/library` only after every
byte has been checked against the manifest. Nothing is parsed, and nothing remains on failure.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

import httpx
from pydantic import ValidationError

from precedent import config
from precedent.libraries.manifest import (
    MANIFEST_NAME,
    LibraryIntegrityError,
    Manifest,
    verify_files,
)

__all__ = [
    "SnapshotFetchError",
    "build_snapshot",
    "fetch_snapshot",
    "run",
    "snapshot_name",
]

USAGE_ERROR = 2
_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_FILE_MODE = 0o100644
_REQUEST_TIMEOUT_S = 60.0
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_SNAPSHOT_BYTES = 1 << 31
_LIBRARY_DIR = "library"


class SnapshotFetchError(Exception):
    """A snapshot could not be downloaded: a non-200 status, a timeout or a transport failure."""


def snapshot_name(tag: str) -> str:
    return f"precedent-library-{tag}.zip"


def _checked_tag(tag: str) -> str:
    if not _TAG_PATTERN.fullmatch(tag) or ".." in tag:
        raise ValueError(f"not a valid release tag: {tag!r}")
    return tag


def _read_manifest_bytes(raw: bytes, label: str) -> Manifest:
    try:
        return Manifest.model_validate_json(raw)
    except ValidationError as error:
        raise LibraryIntegrityError(MANIFEST_NAME, f"{label} is not a valid manifest") from error


def build_snapshot(library_dir: Path, out_dir: Path, tag: str) -> Path:
    """Verify `library_dir`, write `<out_dir>/precedent-library-<tag>.zip` and `MANIFEST.json`.

    Returns the zip's path. A bundle that fails verification raises `LibraryIntegrityError`
    and nothing is written.
    """
    _checked_tag(tag)
    manifest_path = library_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise LibraryIntegrityError(MANIFEST_NAME, f"absent from {library_dir}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_manifest_bytes(manifest_bytes, "the library's manifest")
    verify_files(library_dir, manifest)
    names = sorted([entry.path for entry in manifest.files] + [MANIFEST_NAME])

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / snapshot_name(tag)
    temp = out_dir / f".{target.name}.tmp-{os.getpid()}"
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = _FILE_MODE << 16
                archive.writestr(info, (library_dir / name).read_bytes())
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    (out_dir / MANIFEST_NAME).write_bytes(manifest_bytes)
    return target


def _get(client: httpx.Client, url: str, limit_bytes: int) -> bytes:
    try:
        response = client.get(url)
    except httpx.TimeoutException as error:
        raise SnapshotFetchError(f"{url}: timed out") from error
    except httpx.HTTPError as error:
        raise SnapshotFetchError(f"{url}: {type(error).__name__}") from error
    if response.status_code != 200:
        raise SnapshotFetchError(f"{url}: HTTP {response.status_code}")
    if len(response.content) > limit_bytes:
        raise SnapshotFetchError(f"{url}: response larger than {limit_bytes} bytes")
    return response.content


def _safe_member_path(name: str) -> str:
    parts = PurePosixPath(name).parts
    if not name or "\\" in name or name.startswith("/") or ".." in parts or "." in parts:
        raise LibraryIntegrityError(name, "zip entry path leaves the library directory")
    return name


def _extract(archive_bytes: bytes, into: Path) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as error:
        raise LibraryIntegrityError("snapshot zip", "is not a valid zip file") from error
    with archive:
        infos = archive.infolist()
        names = [_safe_member_path(info.filename.removesuffix("/")) for info in infos]
        if len(set(names)) != len(names):
            raise LibraryIntegrityError("snapshot zip", "contains a repeated entry name")
        for info, name in zip(infos, names, strict=True):
            if info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise LibraryIntegrityError(name, "is a symbolic link in the zip")
            destination = into / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))


def _swap_into_place(temp: Path, target: Path) -> None:
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    replaced = target.exists()
    if replaced:
        target.rename(old)
    try:
        temp.rename(target)
    except BaseException:
        if replaced:
            old.rename(target)
        raise
    if replaced:
        shutil.rmtree(old)


def fetch_snapshot(
    tag: str, dest: Path, transport: httpx.BaseTransport, base_url: str
) -> Manifest:
    """Download release `tag`, verify it and install it as `<dest>/library`; return its manifest.

    Raises `SnapshotFetchError` for a failed download and `LibraryIntegrityError` for any
    entry that leaves the library directory, an inner manifest that differs from the downloaded
    one, or a file that does not match the manifest. On failure an existing `<dest>/library` is
    untouched and no temporary directory remains (`dest` itself may be left as an empty directory).
    """
    _checked_tag(tag)
    release_url = f"{base_url.rstrip('/')}/{tag}"
    with httpx.Client(transport=transport, timeout=_REQUEST_TIMEOUT_S) as client:
        manifest_bytes = _get(client, f"{release_url}/{MANIFEST_NAME}", _MAX_MANIFEST_BYTES)
        manifest = _read_manifest_bytes(manifest_bytes, "the downloaded manifest")
        archive_bytes = _get(client, f"{release_url}/{snapshot_name(tag)}", _MAX_SNAPSHOT_BYTES)

    dest.mkdir(parents=True, exist_ok=True)
    temp = dest / f".{_LIBRARY_DIR}.tmp-{os.getpid()}"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    try:
        _extract(archive_bytes, temp)
        inner_path = temp / MANIFEST_NAME
        if not inner_path.is_file():
            raise LibraryIntegrityError(MANIFEST_NAME, "absent from the snapshot zip")
        inner = _read_manifest_bytes(inner_path.read_bytes(), "the zip's manifest")
        if inner != manifest:
            raise LibraryIntegrityError(
                MANIFEST_NAME, "the manifest inside the zip differs from the downloaded MANIFEST"
            )
        verify_files(temp, manifest)
        _swap_into_place(temp, dest / _LIBRARY_DIR)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precedent fetch")
    parser.add_argument("--version", default=None, metavar="TAG")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="PATH")
    return parser


def run(argv: list[str]) -> int:
    """Install the published library; print its sources and return 0, or 2 on any refusal."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else USAGE_ERROR
    base_url = config.release_base_url()
    if base_url is None:
        print("error: no release repository configured", file=sys.stderr)
        return USAGE_ERROR
    tag: str = args.version or f"v{importlib.metadata.version('precedent-engine')}"
    dest: Path = args.data_dir if args.data_dir is not None else config.data_dir()
    try:
        with contextlib.closing(httpx.HTTPTransport()) as transport:
            manifest = fetch_snapshot(tag, dest, transport, base_url)
    except (SnapshotFetchError, LibraryIntegrityError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_ERROR
    for source in manifest.sources:
        print(
            f"{source.name}: {source.first_date} to {source.last_date}, "
            f"{source.rows} rows. {source.citation}"
        )
    return 0
