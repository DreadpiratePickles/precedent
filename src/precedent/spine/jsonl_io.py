"""Atomic JSON Lines writing, shared by the event store and the market store."""

import contextlib
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppendReport:
    """What an append did: records written, and records already stored identically."""

    added: int
    identical_skipped: int


def atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    """Replace `path` with `lines` (one per line) so a reader sees the old file or the new one.

    The temporary file lives in the same directory so `os.replace` is atomic; it is removed if
    anything fails, and the original file is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
