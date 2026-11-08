"""Fixtures shared by the engine tests.

`library_root` and `sparse_library_root` are data dirs whose `library/` was written by
`write_bundle` from synthetic events (see `library_root_synthetic`); the ask, MCP and evaluation
tests reuse them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from engine_synthetic import events_frame as _events_frame
from library_root_synthetic import write_library_root, write_sparse_library_root


@pytest.fixture
def events_frame() -> Callable[..., pd.DataFrame]:
    return _events_frame


@pytest.fixture
def library_root(tmp_path: Path) -> Path:
    """A data dir with 20 independent `central_bank / US / high` events; no asset rows."""
    return write_library_root(tmp_path)


@pytest.fixture
def sparse_library_root(tmp_path: Path) -> Path:
    """A data dir with 3 `cyber_incident / company / US / high` events."""
    return write_sparse_library_root(tmp_path)
