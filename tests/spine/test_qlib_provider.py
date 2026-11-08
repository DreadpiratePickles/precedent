"""The Qlib bridge: same returns as the Parquet provider, plus the provider health check."""

import pytest

pytest.importorskip("qlib")

import pandas as pd  # noqa: E402

from precedent.config import ConfigError  # noqa: E402
from precedent.spine.price_provider import load_returns, load_sessions  # noqa: E402
from precedent.spine.qlib_provider import QlibPriceProvider, check_provider, main  # noqa: E402


def test_qlib_provider_gives_the_same_returns(tiny_provider):
    returns = load_returns(
        ["AAA"], "2024-01-02", "2024-01-05", kind="open_to_open", provider=QlibPriceProvider()
    )
    assert returns.loc["2024-01-03", "AAA"] == pytest.approx(0.01)
    assert pd.isna(returns.loc["2024-01-02", "AAA"])


def test_qlib_sessions_and_nan_for_a_missing_row(tiny_provider):
    provider = QlibPriceProvider()
    sessions = load_sessions("2024-01-02", "2024-01-05", provider=provider)
    assert len(sessions) == 4
    frame = provider.prices(["AAA", "BBB"], "2024-01-02", "2024-01-05", "close")
    assert list(frame.columns) == ["AAA", "BBB"]
    assert pd.isna(frame.loc["2024-01-03", "BBB"])
    assert frame.loc["2024-01-04", "BBB"] == pytest.approx(53.0)


def test_qlib_unknown_symbol_is_a_config_error(tiny_provider):
    with pytest.raises(ConfigError, match="ZZZ"):
        QlibPriceProvider().prices(["ZZZ"], "2024-01-02", "2024-01-05", "open")


def test_check_provider_reports_sessions_instruments_and_spy(tiny_provider_dir):
    report = check_provider(tiny_provider_dir)
    assert report.sessions == 4
    assert report.instruments == 3
    assert report.last_session.isoformat() == "2024-01-05"
    assert report.has_spy is True


def test_check_provider_on_an_empty_directory_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="provider"):
        check_provider(tmp_path)


def test_check_entry_point_prints_the_report(tiny_provider_dir, capsys):
    assert main(["--check", str(tiny_provider_dir)]) == 0
    out = capsys.readouterr().out
    assert "sessions: 4" in out
    assert "SPY last session: 2024-01-05" in out
