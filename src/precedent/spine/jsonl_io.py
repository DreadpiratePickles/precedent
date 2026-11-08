"""Atomic JSON and JSON Lines helpers shared by the event and market stores and ingesters."""

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

_Model = TypeVar("_Model", bound=BaseModel)


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


def read_models(path: Path, model: type[_Model]) -> list[_Model]:
    """Every record in a JSON Lines file, in file order; a missing file reads as empty.

    A malformed line raises ValueError naming the file and the line number.
    """
    if not path.exists():
        return []
    records: list[_Model] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(f"{path.name} line {line_number}: {error}") from error
    return records


def merge_keyed(
    stored: list[_Model],
    incoming: Iterable[_Model],
    key_of: Callable[[_Model], Hashable],
    on_conflict: Callable[[_Model], Exception],
) -> tuple[list[_Model], int, int]:
    """Append `incoming` records whose key is new to a copy of `stored`.

    Returns (merged, added, identical_skipped). A record that repeats a stored key with
    different content raises `on_conflict(record)`; equality is the model's own `==`.
    """
    merged = list(stored)
    index = {key_of(record): record for record in stored}
    added = skipped = 0
    for record in incoming:
        key = key_of(record)
        existing = index.get(key)
        if existing is None:
            index[key] = record
            merged.append(record)
            added += 1
        elif existing == record:
            skipped += 1
        else:
            raise on_conflict(record)
    return merged, added, skipped


def read_json_state(path: Path) -> dict[str, Any]:
    """A saved state object, or an empty one when the file does not exist yet."""
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name}: state file is not valid JSON: {error}") from error
    if not isinstance(state, dict):
        raise ValueError(f"{path.name}: state file must hold a JSON object")
    return state


def write_json_state(path: Path, state: dict[str, Any]) -> None:
    """Replace the state file atomically."""
    atomic_write_lines(path, [json.dumps(state, sort_keys=True)])
