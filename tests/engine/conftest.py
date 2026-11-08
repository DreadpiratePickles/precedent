"""Fixtures shared by the engine tests."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest
from engine_synthetic import events_frame as _events_frame


@pytest.fixture
def events_frame() -> Callable[..., pd.DataFrame]:
    return _events_frame
