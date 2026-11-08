"""Tests for the sparse (datetime, instrument) news factor panel and its no-lookahead property."""
import numpy as np
import pandas as pd
import pytest

from news_factors import PANEL_COLUMNS, build_news_factor_panel

SESSIONS = pd.bdate_range("2024-01-02", periods=60)


def make_events(rows: list[dict]) -> pd.DataFrame:
    base = {"instrument": "AAA", "severity": 5, "direction": 0.5, "confidence_weight": 0.8, "novelty": "new"}
    frame = pd.DataFrame([base | row for row in rows])
    frame["effective_session"] = pd.to_datetime(frame["effective_session"])
    return frame[["effective_session", "instrument", "severity", "direction", "confidence_weight", "novelty"]]


def one_event(position: int = 5, **overrides: object) -> pd.DataFrame:
    return make_events([{"effective_session": SESSIONS[position]} | overrides])


def test_single_event_halves_after_the_half_life_and_disappears_after_max_age() -> None:
    panel = build_news_factor_panel(one_event(5), SESSIONS, half_life_sessions=3.0, max_age_sessions=20)
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert list(panel.columns) == PANEL_COLUMNS
    assert len(panel) == 21
    dates = panel.index.get_level_values("datetime")
    assert dates.min() == SESSIONS[5] and dates.max() == SESSIONS[25]
    signed, attention = 0.5 * 0.5 * 0.8, 0.5 * 0.8
    day0 = panel.loc[(SESSIONS[5], "AAA")]
    assert day0["news_cnt_1d"] == 1 and day0["news_sev_max_1d"] == 5 and day0["news_novel_share_1d"] == 1
    assert day0["news_signed_1d"] == pytest.approx(signed)
    assert day0["news_signed_decay"] == pytest.approx(signed)
    assert day0["news_attention_decay"] == pytest.approx(attention)
    day3 = panel.loc[(SESSIONS[8], "AAA")]
    assert day3["news_signed_decay"] == pytest.approx(signed / 2)
    assert day3["news_attention_decay"] == pytest.approx(attention / 2)
    assert panel.loc[(SESSIONS[11], "AAA"), "news_signed_decay"] == pytest.approx(signed / 4)
    assert (SESSIONS[26], "AAA") not in panel.index


def test_one_day_columns_are_zero_after_the_effective_session() -> None:
    panel = build_news_factor_panel(one_event(5), SESSIONS)
    later = panel.loc[(SESSIONS[7], "AAA")]
    assert later[["news_cnt_1d", "news_sev_max_1d", "news_signed_1d", "news_novel_share_1d"]].eq(0).all()
    assert later["news_signed_decay"] > 0


def test_no_row_exists_before_the_effective_session() -> None:
    panel = build_news_factor_panel(one_event(5), SESSIONS)
    assert panel.index.get_level_values("datetime").min() >= SESSIONS[5]


def test_event_near_the_end_of_the_calendar_is_truncated_not_extrapolated() -> None:
    panel = build_news_factor_panel(one_event(len(SESSIONS) - 2), SESSIONS)
    assert len(panel) == 2


def test_two_same_day_events_aggregate() -> None:
    events = make_events(
        [
            {"effective_session": SESSIONS[3], "severity": 4, "direction": 1.0, "confidence_weight": 0.5, "novelty": "new"},
            {"effective_session": SESSIONS[3], "severity": 8, "direction": -0.5, "confidence_weight": 1.0, "novelty": "recap"},
        ]
    )
    row = build_news_factor_panel(events, SESSIONS).loc[(SESSIONS[3], "AAA")]
    first, second = 1.0 * 0.4 * 0.5, -0.5 * 0.8 * 1.0
    assert row["news_cnt_1d"] == 2
    assert row["news_sev_max_1d"] == 8
    assert row["news_signed_1d"] == pytest.approx(first + second)
    assert row["news_novel_share_1d"] == 0.5
    assert row["news_signed_decay"] == pytest.approx(first + second)
    assert row["news_attention_decay"] == pytest.approx(0.4 * 0.5 + 0.8 * 1.0)


def test_events_on_different_days_and_instruments_decay_independently() -> None:
    events = make_events(
        [
            {"effective_session": SESSIONS[0], "instrument": "AAA"},
            {"effective_session": SESSIONS[3], "instrument": "AAA"},
            {"effective_session": SESSIONS[3], "instrument": "BBB"},
        ]
    )
    panel = build_news_factor_panel(events, SESSIONS, half_life_sessions=3.0)
    single = 0.5 * 0.5 * 0.8
    assert panel.loc[(SESSIONS[3], "AAA"), "news_signed_decay"] == pytest.approx(single / 2 + single)
    assert panel.loc[(SESSIONS[3], "BBB"), "news_signed_decay"] == pytest.approx(single)
    assert (SESSIONS[2], "BBB") not in panel.index
    assert panel.index.is_monotonic_increasing


def test_unknown_session_raises_and_names_the_count() -> None:
    saturday = pd.Timestamp("2024-01-06")
    events = make_events([{"effective_session": SESSIONS[2]}, {"effective_session": saturday}])
    with pytest.raises(ValueError, match=r"1 of 2"):
        build_news_factor_panel(events, SESSIONS)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("severity", 11), ("severity", 0), ("direction", 1.5), ("confidence_weight", -0.1), ("novelty", "rumour")],
)
def test_out_of_range_values_raise(column: str, bad_value: object) -> None:
    events = one_event(2)
    events[column] = events[column].astype(object)
    events.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=column):
        build_news_factor_panel(events, SESSIONS)


def test_missing_column_and_missing_values_raise() -> None:
    with pytest.raises(ValueError, match="severity"):
        build_news_factor_panel(one_event(2).drop(columns="severity"), SESSIONS)
    events = one_event(2).astype({"direction": float})
    events.loc[0, "direction"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        build_news_factor_panel(events, SESSIONS)


def test_bad_parameters_raise() -> None:
    with pytest.raises(ValueError, match="half_life_sessions"):
        build_news_factor_panel(one_event(2), SESSIONS, half_life_sessions=0.0)
    with pytest.raises(ValueError, match="max_age_sessions"):
        build_news_factor_panel(one_event(2), SESSIONS, max_age_sessions=-1)


def test_no_events_gives_an_empty_panel_with_the_right_schema() -> None:
    panel = build_news_factor_panel(one_event(2).iloc[0:0], SESSIONS)
    assert panel.empty and list(panel.columns) == PANEL_COLUMNS
    assert panel.index.levels[0].dtype == pd.DatetimeIndex([]).as_unit("ns").dtype
    assert list(panel.index.names) == ["datetime", "instrument"]


def random_events(rng: np.random.Generator, n_events: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_session": SESSIONS[rng.integers(0, len(SESSIONS), size=n_events)],
            "instrument": rng.choice(["AAA", "BBB", "CCC", "DDD"], size=n_events),
            "severity": rng.integers(1, 11, size=n_events),
            "direction": rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n_events),
            "confidence_weight": rng.choice([0.3, 0.6, 0.9], size=n_events),
            "novelty": rng.choice(["new", "update", "recap", "opinion"], size=n_events),
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_lookahead_rows_up_to_t_depend_only_on_events_up_to_t(seed: int) -> None:
    rng = np.random.default_rng(seed)
    events = random_events(rng, n_events=150)
    full = build_news_factor_panel(events, SESSIONS, half_life_sessions=2.5, max_age_sessions=12)
    for cut_position in (0, 7, 25, 40, 59):
        cut = SESSIONS[cut_position]
        known = events[events["effective_session"] <= cut]
        future = events[events["effective_session"] > cut]
        from_known_only = build_news_factor_panel(known, SESSIONS, half_life_sessions=2.5, max_age_sessions=12)
        rewritten = pd.concat([known, random_events(rng, len(future)).assign(effective_session=future["effective_session"].to_numpy())])
        from_rewritten_future = build_news_factor_panel(rewritten, SESSIONS, half_life_sessions=2.5, max_age_sessions=12)
        expected = full[full.index.get_level_values("datetime") <= cut]
        pd.testing.assert_frame_equal(expected, from_known_only[from_known_only.index.get_level_values("datetime") <= cut], check_exact=True)
        pd.testing.assert_frame_equal(
            expected, from_rewritten_future[from_rewritten_future.index.get_level_values("datetime") <= cut], check_exact=True
        )
