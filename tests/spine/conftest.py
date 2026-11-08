"""Fixtures shared by the price tests: a generated Qlib provider (research)."""

from __future__ import annotations

import pandas as pd
import pytest
from fixtures.tiny_provider import write_tiny_provider

DAYS = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])


@pytest.fixture
def tiny_provider_dir(tmp_path):
    """A four-session provider holding AAA, BBB (missing 2024-01-03) and SPY."""
    prices = {
        "AAA": pd.DataFrame(
            {"open": [100.0, 101.0, 102.0, 103.0], "close": [101, 102, 103, 104]}, DAYS
        ),
        "BBB": pd.DataFrame({"open": [50.0, 52.0], "close": [51.0, 53.0]}, DAYS[[0, 2]]),
        "SPY": pd.DataFrame(
            {"open": [470.0, 471, 472, 473], "close": [471.0, 472, 473, 474]}, DAYS
        ),
    }
    provider_dir = tmp_path / "provider"
    write_tiny_provider(provider_dir, prices)
    return provider_dir


@pytest.fixture
def tiny_provider(tiny_provider_dir, tmp_path):
    """The tiny provider, with Qlib initialised on it (research: skips without qlib)."""
    pytest.importorskip("qlib")
    from precedent.spine.qlib_provider import init_qlib

    init_qlib(tiny_provider_dir, tmp_path / "work")
    return tiny_provider_dir
