import numpy as np
import pandas as pd
import pytest
from engine_synthetic import CUTOFFS, SESSIONS

from precedent.engine.aggregate import effective_sample_size
from precedent.engine.keys import key_from_flags
from precedent.engine.retrieval import (
    PrecedentQuery,
    exact_key_similarity,
    find_precedents,
    select_precedent_events,
)
from precedent.libraries.common import (
    LIBRARY_COLUMNS,
    OUTCOME_TYPES,
    validate_library_rows,
)
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS

_ONE_HOUR = pd.Timedelta(hours=1)

KEY_ANY_DIRECTION = key_from_flags("central_bank", "central_bank", "high", False, "US")
KEY_DOWN = key_from_flags("central_bank", "central_bank", "high", False, "US", direction_sign=-1)
KEY_UP = key_from_flags("central_bank", "central_bank", "high", False, "US", direction_sign=1)
KEY_OTHER_COUNTRY = key_from_flags(
    "central_bank", "central_bank", "high", False, "DM", direction_sign=-1
)
KEY_EARNINGS = key_from_flags("earnings", "company", "mid", False, "US")
KEY_GUIDANCE = key_from_flags("guidance", "company", "mid", True, "US")


def events_table(rows):
    """Tuples (event_id, key, position[, story_cluster_id[, is_first_report]]) -> events table."""
    records = []
    for event_id, key, position, *extra in rows:
        story = extra[0] if len(extra) > 0 and extra[0] is not None else event_id
        is_first = extra[1] if len(extra) > 1 else True
        records.append(
            {
                "event_id": event_id,
                "key": key.as_string(),
                "event_type": key.event_type,
                "country_group": key.country_group,
                "available_utc": CUTOFFS.iloc[position] - _ONE_HOUR,
                "effective_session": SESSIONS[position].date(),
                "effective_position": position,
                "primary_actor": "actor",
                "linked_countries": [],
                "story_cluster_id": story,
                "is_first_report": is_first,
                "one_line": f"{event_id} line",
                "source": "synthetic",
                "source_ref": event_id,
            }
        )
    return pd.DataFrame(records, columns=list(EVENTS_TABLE_COLUMNS))


def library_rows(rows):
    """Dicts -> validated library rows; sessions are given as positions in SESSIONS."""
    records = []
    for row in rows:
        outcome_type = row["outcome_type"]
        records.append(
            {
                "event_id": row["event_id"],
                "outcome_type": outcome_type,
                "outcome_key": row.get("outcome_key", "outcome"),
                "horizon": row.get("horizon", 5),
                "value": row.get("value", 0.01 if outcome_type == "asset" else 1.0),
                "is_binary": outcome_type != "asset",
                "effective_session": SESSIONS[row["effective"]],
                "response_complete_session": row.get("complete_session")
                or SESSIONS[row["complete"]],
                "source_ref": row["event_id"],
            }
        )
    return validate_library_rows(pd.DataFrame(records))


def query_for(key, position_offset_hours=1, position=120, story_cluster_id=None):
    return PrecedentQuery(
        key=key,
        available_utc=CUTOFFS.iloc[position] + pd.Timedelta(hours=position_offset_hours),
        story_cluster_id=story_cluster_id,
    )


def _sequence_row(event_id, effective, complete, horizon=5):
    return {
        "event_id": event_id,
        "outcome_type": "event_sequence",
        "effective": effective,
        "complete": complete,
        "horizon": horizon,
    }


def random_query_and_library(seed):
    rng = np.random.default_rng(seed)
    keys = [KEY_EARNINGS, KEY_GUIDANCE, KEY_ANY_DIRECTION][: int(rng.integers(2, 4))]
    positions = rng.integers(0, 250, size=40)
    events = events_table(
        [
            (f"E{i:02d}", keys[int(rng.integers(0, len(keys)))], int(position))
            for i, position in enumerate(positions)
        ]
    )
    rows = []
    for event_id, position in zip(events["event_id"], events["effective_position"], strict=True):
        p = int(position)
        rows.append(_sequence_row(event_id, p, p + 5))
        rows.append(
            {"event_id": event_id, "outcome_type": "asset", "effective": p, "complete": p + 6}
        )
        rows.append(
            {
                "event_id": event_id,
                "outcome_type": "market_resolution",
                "effective": p,
                "complete": p + 1 + int(rng.integers(1, 80)),
                "horizon": int(rng.choice([3, 5, 40])),
            }
        )
    query_position = int(rng.integers(60, 260))
    query = PrecedentQuery(
        key=keys[int(rng.integers(0, len(keys)))], available_utc=CUTOFFS.iloc[query_position]
    )
    return query, events, library_rows(rows), query_position


def events_with_keys(keys):
    return events_table([(f"K{i}", key, 10 + i) for i, key in enumerate(keys)])


def test_an_unfinished_analog_is_never_returned():
    # analog A completes on session 110, analog B on session 130; the query follows session 120
    events = events_table([("A", KEY_ANY_DIRECTION, 100), ("B", KEY_ANY_DIRECTION, 100)])
    rows = library_rows([_sequence_row("A", 100, 110), _sequence_row("B", 100, 130)])
    found = find_precedents(
        query_for(KEY_ANY_DIRECTION),
        events,
        rows,
        CUTOFFS,
        horizon=5,
        outcome_type="event_sequence",
    )
    assert found["event_id"].unique().tolist() == ["A"]


def _reference_precedents(query, events, rows, otype, horizon, top_k, query_position):
    """The P-42 rule, written naively: guard first, similarity then recency, skip overlaps."""
    similarity = exact_key_similarity(query, events)
    pool = events.assign(similarity=similarity)
    pool = pool[(pool["similarity"] > 0) & (pool["available_utc"] <= query.available_utc)]
    pool = pool.sort_values(
        ["similarity", "effective_position", "event_id"], ascending=[False, False, True]
    )
    own = rows[rows["outcome_type"] == otype].set_index("event_id")
    taken: dict[object, list[int]] = {}
    out = []
    for event in pool.itertuples(index=False):
        if event.event_id not in own.index:
            continue
        for row in own.loc[[event.event_id]].itertuples():
            if SESSIONS.get_loc(row.response_complete_session) > query_position:
                continue  # unfinished
            if otype != "market_resolution" and row.horizon != horizon:
                continue
            slot = (row.outcome_key, row.horizon) if otype == "market_resolution" else row.outcome_key
            mine = taken.setdefault(slot, [])
            if len(mine) >= top_k or any(
                abs(event.effective_position - o) < row.horizon for o in mine
            ):
                continue
            mine.append(event.effective_position)
            out.append((event.event_id, slot))
    return out


def test_time_guard_holds_for_random_libraries():
    returned = withheld = 0
    for seed in range(50):
        query, events, rows, query_position = random_query_and_library(seed)
        position_of = events.set_index("event_id")["effective_position"]
        for otype in OUTCOME_TYPES:
            found = find_precedents(query, events, rows, CUTOFFS, horizon=5, outcome_type=otype)
            assert (
                CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= query.available_utc
            ).all()
            expected = _reference_precedents(query, events, rows, otype, 5, 30, query_position)
            slots = [
                (r.outcome_key, r.horizon) if otype == "market_resolution" else r.outcome_key
                for r in found.itertuples()
            ]
            assert list(zip(found["event_id"], slots, strict=True)) == expected
            # pairwise non-overlap per instrument
            for slot in set(slots):
                mask = [x == slot for x in slots]
                chosen = found.loc[mask]
                starts = np.sort(position_of.loc[chosen["event_id"]].to_numpy())
                assert (np.diff(starts) >= chosen["horizon"].iloc[0]).all()
                assert len(chosen) <= 30
            returned += len(found)
            of_type = rows[rows["outcome_type"] == otype]
            withheld += len(of_type) - len(found)
    assert returned > 0
    assert withheld > 0


def test_a_story_is_not_its_own_precedent():
    events = events_table(
        [
            ("S1a", KEY_ANY_DIRECTION, 100, "S1"),
            ("S2a", KEY_ANY_DIRECTION, 101, "S2"),
        ]
    )
    rows = library_rows([_sequence_row("S1a", 100, 105), _sequence_row("S2a", 101, 106)])
    found = find_precedents(
        query_for(KEY_ANY_DIRECTION, story_cluster_id="S1"),
        events,
        rows,
        CUTOFFS,
        horizon=5,
        outcome_type="event_sequence",
    )
    stories = set(found.merge(events, on="event_id")["story_cluster_id"])
    assert "S1" not in stories
    assert stories == {"S2"}


def test_exact_similarity_uses_direction_only_when_the_query_sets_it():
    events = events_with_keys([KEY_DOWN, KEY_UP, KEY_OTHER_COUNTRY])
    assert exact_key_similarity(query_for(KEY_ANY_DIRECTION), events).tolist() == [1.0, 1.0, 0.0]
    assert exact_key_similarity(query_for(KEY_DOWN), events).tolist() == [1.0, 0.0, 0.0]


def test_exact_similarity_parses_each_distinct_key_once(monkeypatch):
    from precedent.engine import retrieval

    calls = []
    original = retrieval.EventKey.from_string

    def counting(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(retrieval.EventKey, "from_string", counting)
    events = events_table([(f"R{i}", KEY_DOWN, i) for i in range(20)])
    exact_key_similarity(query_for(KEY_DOWN), events)
    assert calls == [KEY_DOWN.as_string()]


def test_exact_similarity_is_aligned_to_the_events_index():
    events = events_with_keys([KEY_DOWN, KEY_UP]).set_axis([7, 3])
    similarity = exact_key_similarity(query_for(KEY_ANY_DIRECTION), events)
    assert similarity.index.tolist() == [7, 3]
    assert str(similarity.dtype) == "float64"


def test_top_k_keeps_the_latest_completed_non_overlapping_matches():
    # P-42 (documented requirement change): top_k is applied after the completion guard and to
    # events whose outcome windows do not overlap, most recent first. 40 events spaced one
    # horizon (5 sessions) apart; the 30 latest are kept, not the 30 latest of an unguarded pool.
    events = events_table([(f"M{i:02d}", KEY_ANY_DIRECTION, 5 * i) for i in range(40)])
    rows = library_rows(
        [
            {"event_id": f"M{i:02d}", "outcome_type": "asset", "effective": 5 * i,
             "complete": 5 * i + 6}
            for i in range(40)
        ]
    )
    found = find_precedents(
        query_for(KEY_ANY_DIRECTION, position=250), events, rows, CUTOFFS, 5, "asset", top_k=30
    )
    assert len(found) == 30
    assert found["event_id"].iloc[0] == "M39" and "M09" not in set(found["event_id"])
    assert "M10" in set(found["event_id"])  # the 30 latest are M10..M39


def test_selection_order_ties_break_on_position_then_event_id():
    events = events_table(
        [
            ("b", KEY_ANY_DIRECTION, 50),
            ("a", KEY_ANY_DIRECTION, 50),
            ("c", KEY_ANY_DIRECTION, 60),
        ]
    )
    picked = select_precedent_events(query_for(KEY_ANY_DIRECTION), events)
    assert picked["event_id"].tolist() == ["c", "a", "b"]
    assert picked.index.tolist() == [0, 1, 2]
    assert list(picked.columns) == [*EVENTS_TABLE_COLUMNS, "similarity"]
    assert str(picked["similarity"].dtype) == "float64"


def test_select_drops_events_not_yet_available_and_non_matches():
    events = events_table(
        [
            ("past", KEY_ANY_DIRECTION, 100),
            ("future", KEY_ANY_DIRECTION, 121),
            ("other", KEY_EARNINGS, 100),
        ]
    )
    picked = select_precedent_events(query_for(KEY_ANY_DIRECTION), events)
    assert picked["event_id"].tolist() == ["past"]


def test_a_non_first_report_match_is_never_selected():
    events = events_table(
        [
            ("first", KEY_ANY_DIRECTION, 100, "G1", True),
            ("repeat", KEY_ANY_DIRECTION, 101, "G1", False),
        ]
    )
    picked = select_precedent_events(query_for(KEY_ANY_DIRECTION), events)
    assert picked["event_id"].tolist() == ["first"]


def test_a_custom_similarity_replaces_the_exact_one():
    events = events_table([("x", KEY_EARNINGS, 100), ("y", KEY_GUIDANCE, 101)])

    def only_guidance(query, frame):
        return (frame["event_type"] == "guidance").astype("float64") * 0.5

    picked = select_precedent_events(query_for(KEY_ANY_DIRECTION), events, similarity=only_guidance)
    assert picked["event_id"].tolist() == ["y"] and picked["similarity"].tolist() == [0.5]


def test_top_k_below_one_is_rejected():
    events = events_table([("x", KEY_ANY_DIRECTION, 100)])
    with pytest.raises(ValueError, match="top_k"):
        select_precedent_events(query_for(KEY_ANY_DIRECTION), events, top_k=0)


def test_market_rows_keep_their_own_horizons():
    events = events_table([("EM", KEY_ANY_DIRECTION, 5)])
    rows = library_rows(
        [
            {
                "event_id": "EM",
                "outcome_type": "market_resolution",
                "effective": 5,
                "complete": 10,
                "horizon": 3,
            },
            {
                "event_id": "EM",
                "outcome_type": "market_resolution",
                "effective": 5,
                "complete": 50,
                "horizon": 40,
            },
        ]
    )
    found = find_precedents(
        query_for(KEY_ANY_DIRECTION, position=100),
        events,
        rows,
        CUTOFFS,
        horizon=5,
        outcome_type="market_resolution",
    )
    assert sorted(found["horizon"]) == [3, 40]


def test_other_outcome_types_are_filtered_to_the_horizon():
    events = events_table([("EA", KEY_ANY_DIRECTION, 5)])
    rows = library_rows(
        [
            _sequence_row("EA", 5, 10, horizon=5),
            _sequence_row("EA", 5, 12, horizon=10),
            {"event_id": "EA", "outcome_type": "asset", "effective": 5, "complete": 10},
        ]
    )
    found = find_precedents(
        query_for(KEY_ANY_DIRECTION, position=100),
        events,
        rows,
        CUTOFFS,
        horizon=5,
        outcome_type="event_sequence",
    )
    assert found["horizon"].tolist() == [5]
    assert found["outcome_type"].tolist() == ["event_sequence"]


def test_found_rows_have_library_columns_plus_similarity_and_none_is_empty():
    events = events_table([("EA", KEY_ANY_DIRECTION, 5)])
    rows = library_rows([_sequence_row("EA", 5, 10)])
    query = query_for(KEY_ANY_DIRECTION, position=100)
    found = find_precedents(query, events, rows, CUTOFFS, horizon=5, outcome_type="event_sequence")
    assert list(found.columns) == [*LIBRARY_COLUMNS, "similarity"]
    assert found["similarity"].tolist() == [1.0]
    none = find_precedents(query, events, rows, CUTOFFS, horizon=5, outcome_type="asset")
    assert len(none) == 0
    assert list(none.columns) == [*LIBRARY_COLUMNS, "similarity"]
    assert str(none["similarity"].dtype) == "float64"


def test_an_unknown_outcome_type_is_rejected():
    events = events_table([("EA", KEY_ANY_DIRECTION, 5)])
    rows = library_rows([_sequence_row("EA", 5, 10)])
    with pytest.raises(ValueError, match="outcome_type"):
        find_precedents(
            query_for(KEY_ANY_DIRECTION), events, rows, CUTOFFS, horizon=5, outcome_type="nope"
        )


def test_a_naive_query_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        PrecedentQuery(key=KEY_ANY_DIRECTION, available_utc=pd.Timestamp("2020-06-01 12:00"))


def test_a_row_whose_session_is_missing_from_the_cutoffs_raises():
    events = events_table([("EA", KEY_ANY_DIRECTION, 5)])
    saturday = pd.Timestamp("2020-02-01")
    rows = library_rows(
        [
            {
                "event_id": "EA",
                "outcome_type": "event_sequence",
                "effective": 5,
                "complete_session": saturday,
            }
        ]
    )
    with pytest.raises(KeyError, match="2020-02-01"):
        find_precedents(
            query_for(KEY_ANY_DIRECTION, position=100),
            events,
            rows,
            CUTOFFS,
            horizon=5,
            outcome_type="event_sequence",
        )


def _dense_world(per_session, n_sessions, horizon):
    """The Opus probe_i3 setup: one key, events every session, asset rows completing 1+h later."""
    ev, rows, k = [], [], 0
    for position in range(10, 10 + n_sessions):
        for _ in range(per_session):
            event_id = f"e{k:04d}"
            k += 1
            ev.append((event_id, KEY_ANY_DIRECTION, position))
            rows.append({"event_id": event_id, "outcome_type": "asset", "outcome_key": "SPY",
                         "horizon": horizon, "effective": position,
                         "complete": position + 1 + horizon})
    query = PrecedentQuery(
        KEY_ANY_DIRECTION, CUTOFFS.iloc[10 + n_sessions] - pd.Timedelta(hours=1)
    )
    return events_table(ev), library_rows(rows), query


def test_a_frequent_key_at_a_long_horizon_returns_all_the_independent_precedents_there_are():
    # probe_i3 setup, h=20: 300 daily events leave 14 completed, independent ones (the old rule
    # returned 9 and a refused band); every returned precedent counts toward n_eff.
    events, rows, query = _dense_world(per_session=1, n_sessions=300, horizon=20)
    found = find_precedents(query, events, rows, CUTOFFS, 20, "asset", top_k=30)
    assert len(found) == 14
    starts = events.set_index("event_id").loc[found["event_id"], "effective_position"]
    assert (np.diff(np.sort(starts.to_numpy())) >= 20).all()  # non-overlapping windows
    assert (CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= query.available_utc).all()
    assert effective_sample_size(np.ones(14), starts.to_numpy(), 20) == 14


def test_a_frequent_key_returns_top_k_independent_precedents_when_history_allows():
    events, rows, query = _dense_world(per_session=1, n_sessions=340, horizon=10)
    found = find_precedents(query, events, rows, CUTOFFS, 10, "asset", top_k=30)
    starts = events.set_index("event_id").loc[found["event_id"], "effective_position"]
    assert len(found) == 30
    assert effective_sample_size(np.ones(30), starts.to_numpy(), 10) == 30


def test_several_events_per_session_still_give_non_overlapping_precedents():
    events, rows, query = _dense_world(per_session=5, n_sessions=200, horizon=5)
    found = find_precedents(query, events, rows, CUTOFFS, 5, "asset", top_k=30)
    starts = events.set_index("event_id").loc[found["event_id"], "effective_position"]
    assert len(found) == 30 and (np.diff(np.sort(starts.to_numpy())) >= 5).all()


def test_nothing_unfinished_ever_enters_the_selection():
    # the 10 most recent events are still unfinished at the query; they must not take slots
    events = events_table([(f"U{i:02d}", KEY_ANY_DIRECTION, 100 + 6 * i) for i in range(20)])
    rows = library_rows(
        [
            {"event_id": f"U{i:02d}", "outcome_type": "asset", "effective": 100 + 6 * i,
             "complete": 100 + 6 * i + 6}
            for i in range(20)
        ]
    )
    query = query_for(KEY_ANY_DIRECTION, position=100 + 6 * 10 + 6)  # only U00..U10 are complete
    found = find_precedents(query, events, rows, CUTOFFS, 5, "asset", top_k=30)
    assert sorted(found["event_id"]) == [f"U{i:02d}" for i in range(11)]
    assert (CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= query.available_utc).all()


def test_slots_are_counted_per_instrument():
    # 12 old events carry SPY and QQQ rows; 40 newer events carry QQQ only. top_k=5 must still
    # give SPY its own five slots (the newer events do not take them).
    old = [(f"O{i:02d}", KEY_ANY_DIRECTION, 6 * i) for i in range(12)]
    new = [(f"N{i:02d}", KEY_ANY_DIRECTION, 100 + 6 * i) for i in range(40)]
    events = events_table(old + new)
    records = []
    for event_id, _, position in old:
        for instrument in ("SPY", "QQQ"):
            records.append({"event_id": event_id, "outcome_type": "asset",
                            "outcome_key": instrument, "effective": position,
                            "complete": position + 6})
    for event_id, _, position in new:
        records.append({"event_id": event_id, "outcome_type": "asset", "outcome_key": "QQQ",
                        "effective": position, "complete": position + 6})
    found = find_precedents(
        query_for(KEY_ANY_DIRECTION, position=380), events, library_rows(records), CUTOFFS, 5,
        "asset", top_k=5,
    )
    counts = found["outcome_key"].value_counts().to_dict()
    assert counts == {"SPY": 5, "QQQ": 5}
    assert set(found.loc[found["outcome_key"] == "SPY", "event_id"]) == {
        f"O{i:02d}" for i in range(7, 12)
    }
