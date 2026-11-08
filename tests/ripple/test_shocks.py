"""Price shocks (abnormal returns on event sessions) and text shocks (signed news)."""

import numpy as np
import pandas as pd
import pytest

from precedent.ripple.shocks import price_shocks, text_shocks
from precedent.spine.event_store import FRAME_COLUMNS

PLANTED_POS = 300
PLANTED_SHOCK = 0.04


def event_frame_of(pairs):
    """An event frame with the event store's columns for (effective_session, instrument) pairs."""
    rows = [
        {
            "effective_session": session,
            "instrument": instrument,
            "severity": 5,
            "direction": 1.0,
            "confidence_weight": 1.0,
            "novelty": "new",
            "event_type": "earnings",
            "article_id": f"article-{i}",
            "available_utc": pd.Timestamp("2020-01-01", tz="UTC"),
        }
        for i, (session, instrument) in enumerate(pairs)
    ]
    frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    frame["effective_session"] = pd.to_datetime(frame["effective_session"]).astype("datetime64[ns]")
    return frame


@pytest.fixture
def shock_inputs():
    sessions = pd.bdate_range("2020-01-02", periods=400)
    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.01, len(sessions))
    returns = pd.DataFrame(
        {
            "MKT": market,
            "AAA": 0.0005 + 1.3 * market,  # exact market model, so the planted shock is recovered exactly
            "BBB": 0.8 * market,
        },
        index=sessions,
    )
    returns.iloc[PLANTED_POS, returns.columns.get_loc("AAA")] += PLANTED_SHOCK
    return {
        "event_frame": event_frame_of([(sessions[PLANTED_POS], "AAA")]),
        "returns": returns,
        "market": "MKT",
        "sessions": sessions,
    }


def test_price_shock_is_the_abnormal_return_on_the_effective_session(shock_inputs):   # +4% planted for AAA at position 300
    shocks, dropped = price_shocks(**shock_inputs)
    assert shocks.loc[(shock_inputs["sessions"][300], "AAA")] == pytest.approx(0.04, abs=1e-9)
    assert list(shocks.index.get_level_values("instrument").unique()) == ["AAA"] and dropped == 0


def test_price_shock_has_the_documented_shape(shock_inputs):
    shocks, _ = price_shocks(**shock_inputs)
    assert shocks.name == "shock" and shocks.dtype == "float64"
    assert list(shocks.index.names) == ["datetime", "instrument"]
    assert shocks.index.get_level_values("datetime").dtype == "datetime64[ns]"


def test_two_events_on_the_same_session_and_instrument_give_one_shock(shock_inputs):
    session = shock_inputs["sessions"][PLANTED_POS]
    shock_inputs["event_frame"] = event_frame_of([(session, "AAA"), (session, "AAA")])
    shocks, dropped = price_shocks(**shock_inputs)
    assert len(shocks) == 1 and dropped == 0


def test_an_instrument_without_returns_is_dropped_and_counted(shock_inputs):
    session = shock_inputs["sessions"][PLANTED_POS]
    shock_inputs["event_frame"] = event_frame_of([(session, "AAA"), (session, "ZZZ")])
    shocks, dropped = price_shocks(**shock_inputs)
    assert list(shocks.index.get_level_values("instrument")) == ["AAA"] and dropped == 1


def test_an_event_too_early_for_the_estimation_window_is_dropped_and_counted(shock_inputs):
    sessions = shock_inputs["sessions"]
    shock_inputs["event_frame"] = event_frame_of([(sessions[10], "AAA"), (sessions[PLANTED_POS], "BBB")])
    shocks, dropped = price_shocks(**shock_inputs)
    assert list(shocks.index.get_level_values("instrument")) == ["BBB"] and dropped == 1


def test_returns_index_must_equal_the_sessions(shock_inputs):
    shock_inputs["returns"] = shock_inputs["returns"].iloc[1:]
    with pytest.raises(ValueError, match="returns.index"):
        price_shocks(**shock_inputs)


def test_the_market_column_is_required(shock_inputs):
    shock_inputs["market"] = "NOPE"
    with pytest.raises(ValueError, match="NOPE"):
        price_shocks(**shock_inputs)


def test_an_event_session_outside_the_sessions_raises(shock_inputs):
    shock_inputs["event_frame"] = event_frame_of([(pd.Timestamp("2030-01-01"), "AAA")])
    with pytest.raises(ValueError, match="not in sessions"):
        price_shocks(**shock_inputs)


def test_sessions_must_be_naive_unique_and_increasing(shock_inputs):
    sessions = shock_inputs["sessions"]
    shock_inputs["sessions"] = sessions[::-1]
    shock_inputs["returns"] = shock_inputs["returns"].iloc[::-1]
    with pytest.raises(ValueError, match="sessions"):
        price_shocks(**shock_inputs)


def test_an_empty_event_frame_gives_an_empty_series(shock_inputs):
    shock_inputs["event_frame"] = event_frame_of([])
    shocks, dropped = price_shocks(**shock_inputs)
    assert shocks.empty and dropped == 0
    assert list(shocks.index.names) == ["datetime", "instrument"]
    assert shocks.name == "shock" and shocks.dtype == "float64"


def panel_of(values):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(day), instrument) for day, instrument, _ in values],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"news_signed_1d": [value for *_, value in values]}, index=index)


def test_text_shocks_keep_only_non_zero_signed_news():
    panel = panel_of(
        [
            ("2020-01-03", "BBB", 0.3),
            ("2020-01-02", "AAA", -0.2),
            ("2020-01-02", "BBB", 0.0),
            ("2020-01-03", "AAA", np.nan),
        ]
    )
    shocks = text_shocks(panel)
    assert shocks.name == "shock" and shocks.dtype == "float64"
    assert shocks.index.is_monotonic_increasing
    assert shocks.to_dict() == {
        (pd.Timestamp("2020-01-02"), "AAA"): -0.2,
        (pd.Timestamp("2020-01-03"), "BBB"): 0.3,
    }


def test_text_shocks_raise_without_the_signed_column_or_index_names():
    panel = panel_of([("2020-01-02", "AAA", 0.1)])
    with pytest.raises(ValueError, match="news_signed_1d"):
        text_shocks(panel.rename(columns={"news_signed_1d": "other"}))
    with pytest.raises(ValueError, match="index"):
        text_shocks(panel.rename_axis(index=["when", "instrument"]))
