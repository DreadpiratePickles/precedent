import numpy as np
import pandas as pd
import pytest
from library_synthetic import SESSIONS, table

from precedent.libraries.asset import (
    AssetLibraryReport,
    build_asset_library,
    sample_asset_placebo,
)
from precedent.libraries.common import LIBRARY_COLUMNS

_SHOCK_POSITION = 302
_SHOCK = 0.03
_TINY_NOISE = 1e-12  # the first test pins the CAR to 1e-9, so the asset noise must stay below it


def _returns(symbols: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    market = rng.normal(0.0, 0.01, len(SESSIONS))
    frame = {"SPY": market}
    for symbol in symbols:
        asset = market + rng.normal(0.0, _TINY_NOISE, len(SESSIONS))
        asset[_SHOCK_POSITION] += _SHOCK
        frame[symbol] = asset
    return pd.DataFrame(frame, index=SESSIONS)


def _inputs(position: int, symbols: tuple[str, ...], returns: pd.DataFrame) -> dict:
    return {
        "events": table([("e1", "armed_conflict", ["MLI"], position)]),
        "returns": returns,
        "market": "SPY",
        "sessions": SESSIONS,
        "assets_for_event": lambda row: list(symbols),
    }


@pytest.fixture
def asset_inputs():
    return _inputs(300, ("AAA",), _returns())


@pytest.fixture
def asset_inputs_near_end():
    return _inputs(len(SESSIONS) - 4, ("AAA",), _returns())


@pytest.fixture
def asset_inputs_new_fund():
    returns = _returns()
    returns["NEWFUND"] = returns["AAA"]
    returns.iloc[:250, returns.columns.get_loc("NEWFUND")] = np.nan
    return _inputs(300, ("AAA", "NEWFUND"), returns)


def placebo_args(asset_inputs):
    returns = asset_inputs["returns"]
    return returns["AAA"], returns["SPY"], asset_inputs["sessions"]


def test_response_window_starts_after_the_trade_open(asset_inputs):
    library, _ = build_asset_library(**asset_inputs)
    row = library[library.horizon == 1].iloc[0]
    assert row.value == pytest.approx(0.03, abs=1e-9)
    assert row.response_complete_session == asset_inputs["sessions"][302]
    assert (row.outcome_type, row.is_binary) == ("asset", False)


def test_rows_carry_the_event_identity_and_the_library_schema(asset_inputs):
    library, report = build_asset_library(**asset_inputs)
    assert list(library.columns) == list(LIBRARY_COLUMNS)
    assert sorted(library.horizon) == [1, 5, 20]
    assert set(library.outcome_key) == {"AAA"} and set(library.source_ref) == {"e1"}
    assert set(library.effective_session) == {SESSIONS[300]}
    assert report == AssetLibraryReport(rows=3, skipped_incomplete_window=0, skipped_short_history=0)


def test_unfinished_windows_are_absent_and_counted(asset_inputs_near_end):  # 3 sessions before the last
    library, report = build_asset_library(**asset_inputs_near_end)
    assert set(library.horizon) == {1} and report.skipped_incomplete_window == 2


def test_short_history_assets_are_counted_not_zeroed(asset_inputs_new_fund):
    library, report = build_asset_library(**asset_inputs_new_fund)
    assert "NEWFUND" not in set(library.outcome_key) and report.skipped_short_history == 3


def test_a_symbol_absent_from_returns_is_counted_not_raised(asset_inputs):
    asset_inputs["assets_for_event"] = lambda row: ["AAA", "GHOST"]
    library, report = build_asset_library(**asset_inputs)
    assert set(library.outcome_key) == {"AAA"} and report.skipped_short_history == 3


def test_an_event_that_is_not_a_first_report_yields_no_rows(asset_inputs):
    asset_inputs["events"] = table([("e1", "armed_conflict", ["MLI"], 300, False)])
    library, report = build_asset_library(**asset_inputs)
    assert library.empty and list(library.columns) == list(LIBRARY_COLUMNS)
    assert report == AssetLibraryReport(rows=0, skipped_incomplete_window=0, skipped_short_history=0)


def test_returns_index_that_differs_from_sessions_raises(asset_inputs):
    asset_inputs["returns"] = asset_inputs["returns"].iloc[:-1]
    with pytest.raises(ValueError, match="sessions"):
        build_asset_library(**asset_inputs)


def test_returns_without_the_market_column_raises(asset_inputs):
    asset_inputs["returns"] = asset_inputs["returns"].drop(columns=["SPY"])
    with pytest.raises(ValueError, match="SPY"):
        build_asset_library(**asset_inputs)


def test_placebo_avoids_event_positions_and_is_seeded(asset_inputs):
    a = sample_asset_placebo(*placebo_args(asset_inputs), horizon=5, n_samples=50, rng=np.random.default_rng(0), exclude_positions={300})
    b = sample_asset_placebo(*placebo_args(asset_inputs), horizon=5, n_samples=50, rng=np.random.default_rng(0), exclude_positions={300})
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50


def test_placebo_shape_and_timing(asset_inputs):
    placebo = sample_asset_placebo(*placebo_args(asset_inputs), horizon=5, n_samples=20, rng=np.random.default_rng(1), exclude_positions=set())
    assert list(placebo.columns) == ["value", "response_complete_session"]
    assert placebo.value.dtype == "float64" and placebo.response_complete_session.dtype == "datetime64[ns]"
    assert np.isfinite(placebo.value).all()
    assert placebo.response_complete_session.isin(SESSIONS[6:]).all()
    assert placebo.response_complete_session.max() <= SESSIONS[-1]


def test_placebo_excludes_the_event_window_it_is_told_to(asset_inputs):
    # every eligible position except 300 excluded: only p = 300 remains, whose h=1 window holds the shock
    returns = asset_inputs["returns"]
    exclude = set(range(len(SESSIONS))) - {300}
    placebo = sample_asset_placebo(returns["AAA"], returns["SPY"], SESSIONS, 1, 5, np.random.default_rng(0), exclude)
    assert len(placebo) == 5 and (placebo.value.sub(0.03).abs() < 1e-9).all()
    assert (placebo.response_complete_session == SESSIONS[302]).all()


def test_placebo_returns_fewer_rows_when_candidates_run_out(asset_inputs):
    returns = asset_inputs["returns"]
    short = returns["AAA"].copy()
    short.iloc[:] = np.nan  # no finite CAR anywhere
    placebo = sample_asset_placebo(short, returns["SPY"], SESSIONS, 5, 10, np.random.default_rng(0), set())
    assert placebo.empty and list(placebo.columns) == ["value", "response_complete_session"]
