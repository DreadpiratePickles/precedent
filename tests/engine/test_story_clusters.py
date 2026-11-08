import pandas as pd
import pytest
from engine_synthetic import events_frame

from precedent.engine.story_clusters import assign_story_clusters


def test_first_report_is_the_event():
    events = events_frame(
        [
            ("e1", "k1", "ACME", 10),
            ("e2", "k1", "ACME", 12),
            ("e3", "k1", "ACME", 20),
            ("e4", "k2", "ACME", 12),
        ]
    )
    out = assign_story_clusters(events, window_sessions=3)
    assert out["is_first_report"].tolist() == [True, False, True, True]
    assert out["story_cluster_id"].tolist() == ["e1", "e1", "e3", "e4"]


def test_clusters_chain_across_the_window():
    out = assign_story_clusters(
        events_frame([("a", "k1", "X", 0), ("b", "k1", "X", 3), ("c", "k1", "X", 6)]),
        window_sessions=3,
    )
    assert out["story_cluster_id"].tolist() == ["a", "a", "a"]


def test_chain_breaks_when_the_gap_exceeds_the_window():
    out = assign_story_clusters(
        events_frame([("a", "k1", "X", 0), ("b", "k1", "X", 3), ("c", "k1", "X", 7)]),
        window_sessions=3,
    )
    assert out["story_cluster_id"].tolist() == ["a", "a", "c"]
    assert out["is_first_report"].tolist() == [True, False, True]


def test_different_primary_actors_never_merge():
    out = assign_story_clusters(
        events_frame([("a", "k1", "X", 5), ("b", "k1", "Y", 5), ("c", "k1", "X", 6)]),
        window_sessions=3,
    )
    assert out["story_cluster_id"].tolist() == ["a", "b", "a"]
    assert out["is_first_report"].tolist() == [True, True, False]


def test_input_row_order_and_columns_are_preserved():
    events = events_frame([("late", "k1", "X", 9), ("early", "k1", "X", 8), ("z", "k2", "X", 1)])
    out = assign_story_clusters(events)
    assert out["event_id"].tolist() == ["late", "early", "z"]
    assert out["story_cluster_id"].tolist() == ["early", "early", "z"]
    assert out["is_first_report"].tolist() == [False, True, True]
    assert list(out.columns) == [*events.columns, "story_cluster_id", "is_first_report"]
    assert out.index.equals(events.index)


def test_ties_break_on_available_utc_then_event_id():
    events = events_frame([("b", "k1", "X", 4), ("a", "k1", "X", 4), ("c", "k1", "X", 4)])
    events.loc[2, "available_utc"] = events.loc[2, "available_utc"] - pd.Timedelta(minutes=5)
    out = assign_story_clusters(events)
    # c is earliest in available_utc; a and b share a timestamp and sort by event_id
    assert out["story_cluster_id"].tolist() == ["c", "c", "c"]
    assert out["is_first_report"].tolist() == [False, False, True]
    same_time = events_frame([("b", "k1", "X", 4), ("a", "k1", "X", 4)])
    assert assign_story_clusters(same_time)["story_cluster_id"].tolist() == ["a", "a"]


def test_does_not_mutate_the_input():
    events = events_frame([("a", "k1", "X", 0)])
    before = events.copy()
    assign_story_clusters(events)
    pd.testing.assert_frame_equal(events, before)


def test_empty_input_returns_the_two_columns():
    out = assign_story_clusters(events_frame([]))
    assert out.empty
    assert list(out.columns)[-2:] == ["story_cluster_id", "is_first_report"]
    assert out["is_first_report"].dtype == bool


def test_missing_columns_are_named():
    events = events_frame([("a", "k1", "X", 0)]).drop(columns=["key", "available_utc"])
    with pytest.raises(ValueError, match="available_utc.*key|key.*available_utc"):
        assign_story_clusters(events)


def test_duplicate_index_with_permuted_sort_keeps_index_and_clusters():
    events = events_frame([("b", "k1", "X", 5), ("a", "k1", "X", 4), ("c", "k2", "X", 1)])
    events.index = pd.Index([0, 0, 1])
    out = assign_story_clusters(events)
    assert out["story_cluster_id"].tolist() == ["a", "a", "c"]
    assert out["is_first_report"].tolist() == [False, True, True]
    assert out.index.tolist() == [0, 0, 1]


@pytest.mark.parametrize("column", ["key", "primary_actor", "effective_position"])
def test_null_in_a_required_column_raises_naming_it(column):
    events = events_frame([("a", "k1", "X", 0), ("b", "k1", "X", 1)])
    events[column] = events[column].astype(object)
    events.loc[1, column] = None
    with pytest.raises(ValueError, match=column):
        assign_story_clusters(events)


def test_negative_window_raises():
    with pytest.raises(ValueError, match="window_sessions"):
        assign_story_clusters(events_frame([("a", "k1", "X", 0)]), window_sessions=-1)


def test_zero_window_merges_only_the_same_position():
    out = assign_story_clusters(
        events_frame([("a", "k1", "X", 2), ("b", "k1", "X", 2), ("c", "k1", "X", 3)]),
        window_sessions=0,
    )
    assert out["story_cluster_id"].tolist() == ["a", "a", "c"]


def test_events_frame_fixture_matches_the_helper(events_frame):
    out = assign_story_clusters(events_frame([("a", "k1", "X", 0), ("b", "k1", "X", 1)]))
    assert out["story_cluster_id"].tolist() == ["a", "a"]
