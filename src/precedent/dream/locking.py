"""One advisory lock per output root, so two Dream commands never interleave their store writes.

A store append or update reads the whole file and replaces it atomically. That keeps a reader from
seeing a torn file, but two writers can still lose one write: `precedent dream promote` between
the nightly's read and its replace, say. `report.run`, `run_promote` and `run_review` hold an
exclusive, non-blocking `flock` on `<out>/.lock` for the whole command (D74). A command that finds
it held exits 1 instead of waiting. macOS and Linux only: `fcntl` does not exist on Windows.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE = ".lock"


class DreamLockedError(Exception):
    """Another Dream command holds the lock on this output root."""


@contextmanager
def exclusive_lock(out: Path, *, create: bool = True) -> Iterator[None]:
    """Hold the exclusive lock of `out` for the block; raise DreamLockedError if it is held.

    With `create=False` an output root that does not exist yet is not created and not locked: a
    read-only command has nothing to protect there.
    """
    if not out.is_dir():
        if not create:
            yield
            return
        out.mkdir(parents=True, exist_ok=True)
    with (out / LOCK_FILE).open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DreamLockedError(f"another dream command is running on {out}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
