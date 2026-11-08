"""`score_only` separates the scored events from the precedent pool in `score_precedent_ladder`."""

import numpy as np
import pandas as pd
import pytest
from engine_synthetic import CUTOFFS, SESSIONS

from precedent.engine.evaluate import score_precedent_ladder
from precedent.engine.keys import key_from_flags
from precedent.libraries.common import validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS

_ONE_HOUR = pd.Timedelta(hours=1)
_HORIZON = 5
KEY_A = key_from_flags("central_bank", "central_bank", "high", False, "US")
KEY_B = key_from_flags("armed_conflict", "government", "high", False, "EM")
_KEYS = {"a": KEY_A, "b": KEY_B}


def _inputs():
    rng = np.random.default_rng(3)
    events, outcomes = [], []
    for i in range(60):
        name = "a" if i % 2 == 0 else "b"
        key = _KEYS[name]
        position = 10 + 5 * i
        event_id = f"{name}{i:03d}"
        events.append(
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
            }
        )
        outcomes.append(
            {
                "event_id": event_id,
                "outcome_type": "asset",
                "outcome_key": "SPY",
                "horizon": _HORIZON,
                "value": (0.02 if name == "a" else -0.02) + rng.normal(0.0, 0.005),
                "is_binary": False,
                "effective_session": SESSIONS[position],
                "response_complete_session": SESSIONS[position + _HORIZON],
                "source_ref": event_id,
            }
        )
    return {
        "events": pd.DataFrame(events, columns=list(EVENTS_TABLE_COLUMNS)),
        "rows": validate_library_rows(pd.DataFrame(outcomes)),
        "session_cutoffs_utc": CUTOFFS,
    }


def _sorted(table):
    return table.sort_values(["rung", "event_type", "quantile"]).reset_index(drop=True)


def test_scoring_one_event_type_matches_the_full_run_restricted_to_it():
    inputs = _inputs()
    full = score_precedent_ladder(**inputs, outcome_type="asset")
    only_a = set(
        inputs["events"].loc[inputs["events"]["event_type"] == KEY_A.event_type, "event_id"]
    )
    subset = score_precedent_ladder(**inputs, outcome_type="asset", score_only=only_a)
    expected = full.loc[full["event_type"] == KEY_A.event_type]
    assert set(subset["event_type"]) == {KEY_A.event_type}
    pd.testing.assert_frame_equal(_sorted(subset), _sorted(expected))


def test_boolean_mask_and_id_set_give_the_same_table():
    inputs = _inputs()
    events = inputs["events"]
    late = events["effective_position"] > 200
    by_mask = score_precedent_ladder(**inputs, outcome_type="asset", score_only=late)
    by_ids = score_precedent_ladder(
        **inputs, outcome_type="asset", score_only=set(events.loc[late, "event_id"])
    )
    pd.testing.assert_frame_equal(by_mask, by_ids)


def test_precedent_pool_stays_full_when_only_late_events_are_scored():
    inputs = _inputs()
    events = inputs["events"]
    late = events["effective_position"] > 200
    pooled = score_precedent_ladder(**inputs, outcome_type="asset", score_only=late)
    restricted = score_precedent_ladder(events.loc[late], inputs["rows"], CUTOFFS, "asset")
    assert pooled["n_refused"].sum() < restricted["n_refused"].sum()
    assert pooled["n_scored"].sum() > restricted["n_scored"].sum()


def test_empty_score_only_scores_nothing():
    inputs = _inputs()
    table = score_precedent_ladder(**inputs, outcome_type="asset", score_only=set())
    assert table.empty


def test_misaligned_mask_is_rejected():
    inputs = _inputs()
    short = pd.Series([True, False])
    with pytest.raises(ValueError, match="score_only"):
        score_precedent_ladder(**inputs, outcome_type="asset", score_only=short)
