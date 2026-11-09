import datetime as dt
import hashlib

import numpy as np
import pandas as pd
import pytest
from engine_synthetic import CUTOFFS, SESSIONS

from precedent.engine.embedder import TextEmbedder, embedding_similarity
from precedent.engine.evaluate import score_precedent_ladder
from precedent.engine.keys import key_from_flags
from precedent.engine.params import load_engine_params
from precedent.engine.retrieval import (
    PrecedentQuery,
    exact_key_similarity,
    find_precedents,
)
from precedent.libraries.common import validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS

_ONE_HOUR = pd.Timedelta(hours=1)
_DIM = 32

KEY_A = key_from_flags("central_bank", "central_bank", "high", False, "US")
KEY_B = key_from_flags("armed_conflict", "government", "high", False, "EM")
TEXT_A = "central_bank by [CENTRAL_BANK] in US severity high unscheduled"
TEXT_B = "armed_conflict by [GOVERNMENT] in EM severity high unscheduled"


class HashingEmbedder:
    """A deterministic bag-of-words embedder: each word hashes to one of `_DIM` slots."""

    release_date = dt.date(2025, 1, 1)

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), _DIM))
        for row, text in enumerate(texts):
            for word in text.split():
                slot = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % _DIM
                vectors[row, slot] += 1.0
        return vectors


@pytest.fixture
def fake_embedder():
    return HashingEmbedder()


def _events_with_text(placements):
    """Tuples (event_id, key, position, text) -> events table plus a canonical_text column."""
    records = [
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
            "story_cluster_id": event_id,
            "is_first_report": True,
            "one_line": f"{event_id} line",
            "source": "synthetic",
            "source_ref": event_id,
            "canonical_text": text,
        }
        for event_id, key, position, text in placements
    ]
    return pd.DataFrame(records, columns=[*EVENTS_TABLE_COLUMNS, "canonical_text"])


def _sequence_rows(events, horizon=5, value=1.0):
    records = [
        {
            "event_id": event_id,
            "outcome_type": "event_sequence",
            "outcome_key": "outcome",
            "horizon": horizon,
            "value": value,
            "is_binary": True,
            "effective_session": SESSIONS[position],
            "response_complete_session": SESSIONS[position + horizon],
            "source_ref": event_id,
        }
        for event_id, position in zip(events["event_id"], events["effective_position"], strict=True)
    ]
    return validate_library_rows(pd.DataFrame(records))


PLACEMENTS = [
    ("a0", KEY_A, 20, TEXT_A),
    ("a1", KEY_A, 30, TEXT_A),
    ("b0", KEY_B, 40, TEXT_A),  # wrong key, but the same canonical text as the query
    ("b1", KEY_B, 50, TEXT_B),
]
EVENTS_WITH_TEXT = _events_with_text(PLACEMENTS)
ROWS_WITH_TEXT = _sequence_rows(EVENTS_WITH_TEXT)
QUERY_WITH_TEXT = PrecedentQuery(
    key=KEY_A, available_utc=CUTOFFS.iloc[120] + _ONE_HOUR, canonical_text=TEXT_A
)


def _found(similarity):
    return find_precedents(
        QUERY_WITH_TEXT,
        EVENTS_WITH_TEXT,
        ROWS_WITH_TEXT,
        CUTOFFS,
        horizon=5,
        outcome_type="event_sequence",
        similarity=similarity,
    )


def test_similarity_uses_the_embedder_through_the_guarded_path(fake_embedder):
    found = find_precedents(
        QUERY_WITH_TEXT,
        EVENTS_WITH_TEXT,
        ROWS_WITH_TEXT,
        CUTOFFS,
        horizon=5,
        outcome_type="event_sequence",
        similarity=embedding_similarity(fake_embedder, weight=0.5),
    )
    assert not found.empty
    assert (
        CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= QUERY_WITH_TEXT.available_utc
    ).all()


def test_the_guard_still_hides_an_unfinished_analog_whatever_the_text_says(fake_embedder):
    early = PrecedentQuery(
        key=KEY_A, available_utc=CUTOFFS.iloc[34] + _ONE_HOUR, canonical_text=TEXT_A
    )
    found = find_precedents(
        early,
        EVENTS_WITH_TEXT,
        ROWS_WITH_TEXT,
        CUTOFFS,
        horizon=5,
        outcome_type="event_sequence",
        similarity=embedding_similarity(fake_embedder, weight=1.0),
    )
    # a1 completes on session 35, after the query; b0 and b1 are not yet available events
    assert list(found["event_id"]) == ["a0"]


def test_weight_zero_equals_exact_key_similarity(fake_embedder):
    similarity = embedding_similarity(fake_embedder, weight=0.0)
    got = similarity(QUERY_WITH_TEXT, EVENTS_WITH_TEXT)
    expected = exact_key_similarity(QUERY_WITH_TEXT, EVENTS_WITH_TEXT)
    pd.testing.assert_series_equal(got, expected)


def test_a_wrong_key_event_with_the_same_text_enters_through_the_embedding(fake_embedder):
    mixed = _found(embedding_similarity(fake_embedder, weight=0.5))
    exact_only = _found(exact_key_similarity)
    assert "b0" in set(mixed["event_id"]) and "b0" not in set(exact_only["event_id"])
    scores = dict(zip(mixed["event_id"], mixed["similarity"], strict=True))
    assert scores["a0"] == pytest.approx(1.0)  # right key, identical text
    assert scores["b0"] == pytest.approx(0.5)  # wrong key, identical text: only the text share
    assert scores["a0"] > scores["b0"] > scores["b1"] > 0.0  # b1: wrong key, partial text overlap


def test_similarity_is_clipped_to_the_unit_interval_and_aligned(fake_embedder):
    got = embedding_similarity(fake_embedder, weight=0.7)(QUERY_WITH_TEXT, EVENTS_WITH_TEXT)
    assert got.index.equals(EVENTS_WITH_TEXT.index)
    assert got.dtype == "float64"
    assert ((got >= 0.0) & (got <= 1.0)).all()


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan")])
def test_weight_outside_the_unit_interval_is_rejected(fake_embedder, weight):
    with pytest.raises(ValueError, match="weight"):
        embedding_similarity(fake_embedder, weight=weight)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_weight_at_or_inside_the_bounds_is_accepted(fake_embedder, weight):
    assert callable(embedding_similarity(fake_embedder, weight=weight))


def test_one_embed_call_per_similarity_evaluation(fake_embedder):
    similarity = embedding_similarity(fake_embedder, weight=0.5)
    similarity(QUERY_WITH_TEXT, EVENTS_WITH_TEXT)
    assert len(fake_embedder.calls) == 1
    assert fake_embedder.calls[0] == [TEXT_A, *EVENTS_WITH_TEXT["canonical_text"].tolist()]
    similarity(QUERY_WITH_TEXT, EVENTS_WITH_TEXT)
    assert len(fake_embedder.calls) == 2


def test_no_query_text_or_no_text_column_falls_back_to_exact_without_embedding(fake_embedder):
    similarity = embedding_similarity(fake_embedder, weight=0.9)
    no_text_query = PrecedentQuery(key=KEY_A, available_utc=QUERY_WITH_TEXT.available_utc)
    expected = exact_key_similarity(no_text_query, EVENTS_WITH_TEXT)
    pd.testing.assert_series_equal(similarity(no_text_query, EVENTS_WITH_TEXT), expected)
    plain = EVENTS_WITH_TEXT.drop(columns=["canonical_text"])
    pd.testing.assert_series_equal(
        similarity(QUERY_WITH_TEXT, plain), exact_key_similarity(QUERY_WITH_TEXT, plain)
    )
    assert fake_embedder.calls == []


def test_a_null_canonical_text_is_an_error_not_an_empty_string(fake_embedder):
    broken = EVENTS_WITH_TEXT.assign(canonical_text=[TEXT_A, None, TEXT_A, TEXT_A])
    with pytest.raises(ValueError, match="canonical_text"):
        embedding_similarity(fake_embedder, weight=0.5)(QUERY_WITH_TEXT, broken)


def test_an_embedder_returning_the_wrong_shape_is_rejected():
    class Wrong:
        release_date = dt.date(2025, 1, 1)

        def embed(self, texts):
            return np.ones((len(texts) - 1, 4))

    with pytest.raises(ValueError, match="shape"):
        embedding_similarity(Wrong(), weight=0.5)(QUERY_WITH_TEXT, EVENTS_WITH_TEXT)


def test_no_candidates_means_no_embed_call(fake_embedder):
    empty = EVENTS_WITH_TEXT.iloc[0:0]
    got = embedding_similarity(fake_embedder, weight=0.5)(QUERY_WITH_TEXT, empty)
    assert got.empty and fake_embedder.calls == []


def test_the_fake_satisfies_the_protocol(fake_embedder):
    embedder: TextEmbedder = fake_embedder
    assert embedder.release_date == dt.date(2025, 1, 1)


def _ladder_inputs():
    placements = [
        (f"e{i:03d}", KEY_A if i % 2 == 0 else KEY_B, 10 + 6 * i, TEXT_A if i % 2 == 0 else TEXT_B)
        for i in range(40)
    ]
    events = _events_with_text(placements)
    rows = _sequence_rows(events)
    rows["value"] = [1.0 if i % 2 == 0 else 0.0 for i in range(len(rows))]
    return {"events": events, "rows": rows, "session_cutoffs_utc": CUTOFFS}


def test_rung_b2_equals_b1_on_the_same_inputs():
    inputs = _ladder_inputs()
    both = score_precedent_ladder(**inputs, outcome_type="event_sequence", rungs=("B1", "B2"))
    b1 = both.loc[both["rung"] == "B1"].drop(columns="rung").reset_index(drop=True)
    b2 = both.loc[both["rung"] == "B2"].drop(columns="rung").reset_index(drop=True)
    assert len(b1) > 0
    pd.testing.assert_frame_equal(b1, b2)


def test_rung_b3_without_an_embedder_raises():
    with pytest.raises(ValueError, match="embedder"):
        score_precedent_ladder(**_ladder_inputs(), outcome_type="event_sequence", rungs=("B3",))


def test_rung_b3_at_weight_zero_equals_b1(fake_embedder):
    inputs = _ladder_inputs()
    b3 = score_precedent_ladder(
        **inputs,
        outcome_type="event_sequence",
        rungs=("B3",),
        embedder=fake_embedder,
        embedding_weight=0.0,
    )
    b1 = score_precedent_ladder(**inputs, outcome_type="event_sequence", rungs=("B1",))
    pd.testing.assert_frame_equal(
        b3.drop(columns="rung").reset_index(drop=True),
        b1.drop(columns="rung").reset_index(drop=True),
    )
    assert fake_embedder.calls  # the embedder was consulted, then given zero weight


def test_rung_b3_scores_with_the_events_own_text(fake_embedder):
    inputs = _ladder_inputs()
    table = score_precedent_ladder(
        **inputs,
        outcome_type="event_sequence",
        rungs=("B3",),
        embedder=fake_embedder,
        embedding_weight=0.5,
    )
    assert set(table["rung"]) == {"B3"} and table["n_scored"].sum() > 0
    # the first element of every call is some event's own canonical text
    assert {call[0] for call in fake_embedder.calls} == {TEXT_A, TEXT_B}


def test_the_default_rungs_are_unchanged_and_need_no_embedder():
    table = score_precedent_ladder(**_ladder_inputs(), outcome_type="event_sequence")
    assert set(table["rung"]) == {"B0", "B1"}


def _text_driven_inputs():
    """One key throughout; the outcome follows the text, so only text similarity can separate.

    P-42/P-48: events are spaced one horizon apart, so past the first top_k there are more
    independent completed events than slots and the embedding weight changes which are chosen.
    """
    placements = [
        (f"t{i:03d}", KEY_A, 10 + 6 * i, TEXT_A if i % 2 == 0 else TEXT_B) for i in range(60)
    ]
    events = _events_with_text(placements)
    rows = _sequence_rows(events)
    rows["value"] = [1.0 if i % 2 == 0 else 0.0 for i in range(len(rows))]
    return {"events": events, "rows": rows, "session_cutoffs_utc": CUTOFFS}


def test_rung_b3_at_half_weight_changes_scores_relative_to_b1(fake_embedder):
    inputs = _text_driven_inputs()
    b1 = score_precedent_ladder(**inputs, outcome_type="event_sequence", rungs=("B1",))
    b3 = score_precedent_ladder(
        **inputs,
        outcome_type="event_sequence",
        rungs=("B3",),
        embedder=fake_embedder,
        embedding_weight=0.5,
    )
    left = b1.drop(columns="rung").reset_index(drop=True)
    right = b3.drop(columns="rung").reset_index(drop=True)
    assert len(left) == len(right) > 0
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(left, right)
    assert b3["brier"].dropna().iloc[0] < b1["brier"].dropna().iloc[0]


def test_rung_b3_default_weight_is_the_params_value(fake_embedder):
    inputs = _text_driven_inputs()
    default = score_precedent_ladder(
        **inputs, outcome_type="event_sequence", rungs=("B3",), embedder=fake_embedder
    )
    explicit = score_precedent_ladder(
        **inputs,
        outcome_type="event_sequence",
        rungs=("B3",),
        embedder=fake_embedder,
        embedding_weight=load_engine_params().embedding_weight,
    )
    pd.testing.assert_frame_equal(default, explicit)


def test_rung_b3_without_a_canonical_text_column_raises(fake_embedder):
    inputs = _ladder_inputs()
    inputs["events"] = inputs["events"].drop(columns=["canonical_text"])
    with pytest.raises(ValueError, match="canonical_text"):
        score_precedent_ladder(
            **inputs, outcome_type="event_sequence", rungs=("B3",), embedder=fake_embedder
        )


def test_rung_b3_with_a_null_query_text_raises(fake_embedder):
    inputs = _ladder_inputs()
    events = inputs["events"].copy()
    events.loc[events.index[-1], "canonical_text"] = None
    inputs["events"] = events
    with pytest.raises(ValueError, match="canonical_text"):
        score_precedent_ladder(
            **inputs, outcome_type="event_sequence", rungs=("B3",), embedder=fake_embedder
        )


def test_rungs_without_b3_tolerate_a_missing_text_column():
    inputs = _ladder_inputs()
    inputs["events"] = inputs["events"].drop(columns=["canonical_text"])
    table = score_precedent_ladder(**inputs, outcome_type="event_sequence")
    assert set(table["rung"]) == {"B0", "B1"}


@pytest.mark.parametrize("weight", [True, False, "0.5", None])
def test_a_bool_or_non_number_weight_is_rejected(fake_embedder, weight):
    with pytest.raises(ValueError, match="weight"):
        embedding_similarity(fake_embedder, weight=weight)


@pytest.mark.parametrize("weight", [np.float64(0.5), np.float32(0.25), np.int64(1)])
def test_numpy_real_weights_are_accepted(fake_embedder, weight):
    assert callable(embedding_similarity(fake_embedder, weight=weight))
