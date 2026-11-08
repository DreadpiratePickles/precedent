"""Parquet price provider and the return convention (indexed by the session it became known)."""

from __future__ import annotations

import pandas as pd
import pytest
from test_price_import import TINY_ROWS, write_csv

from precedent.config import ConfigError
from precedent.spine.price_import import import_prices_csv
from precedent.spine.price_provider import (
    ParquetPriceProvider,
    default_price_provider,
    load_returns,
    load_sessions,
)


@pytest.fixture
def provider(tmp_path):
    import_prices_csv(write_csv(tmp_path, TINY_ROWS), tmp_path / "prices")
    return ParquetPriceProvider(tmp_path / "prices")


def test_open_to_open_return_is_indexed_when_known(provider):
    returns = load_returns(
        ["AAA"], "2024-01-02", "2024-01-05", kind="open_to_open", provider=provider
    )
    assert returns.loc["2024-01-03", "AAA"] == pytest.approx(0.01)
    assert pd.isna(returns.loc["2024-01-02", "AAA"])


def test_close_to_close_return(provider):
    returns = load_returns(
        ["AAA"], "2024-01-02", "2024-01-05", kind="close_to_close", provider=provider
    )
    assert returns.loc["2024-01-04", "AAA"] == pytest.approx(103 / 102 - 1)


def test_sessions_come_from_the_provider(provider):
    sessions = load_sessions("2024-01-02", "2024-01-05", provider=provider)
    assert list(sessions.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_sessions_respect_the_window(provider):
    sessions = load_sessions("2024-01-03", "2024-01-04", provider=provider)
    assert list(sessions.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04"]


def test_prices_have_nan_where_a_symbol_has_no_row(provider):
    frame = provider.prices(["AAA", "BBB"], "2024-01-02", "2024-01-05", "close")
    assert list(frame.columns) == ["AAA", "BBB"]
    assert pd.isna(frame.loc["2024-01-03", "BBB"])
    assert frame.loc["2024-01-04", "BBB"] == 53.0


def test_unknown_symbol_is_a_config_error_naming_it(provider):
    with pytest.raises(ConfigError, match="ZZZ"):
        provider.prices(["AAA", "ZZZ"], "2024-01-02", "2024-01-05", "open")


def test_return_is_not_bridged_across_a_missing_price(provider):
    returns = load_returns(
        ["BBB"], "2024-01-02", "2024-01-05", kind="close_to_close", provider=provider
    )
    assert returns["BBB"].isna().all()


def test_no_prices_is_a_config_error_naming_the_import(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    with pytest.raises(ConfigError, match="precedent prices import"):
        default_price_provider()


def test_default_provider_reads_under_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    import_prices_csv(write_csv(tmp_path, TINY_ROWS), tmp_path / "prices")
    sessions = load_sessions("2024-01-02", "2024-01-05")
    assert len(sessions) == 4
