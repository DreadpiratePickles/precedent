"""D71: `direction_drift` reads the claim history, not the claims still valid, so it can fire in
the pipeline on the 3-session claims the rules parser builds."""

import pandas as pd
from fakes import make_event_record
from test_shifts import CUTOFFS, PARAMS, SESSIONS, feed, make_stores

from precedent.dream.claim_store import claims_asof, claims_history_asof
from precedent.dream.rule_claims import claims_from_events
from precedent.dream.shifts import run_shift_detectors
from precedent.spine.event_record import ResolvedImpact

BEFORE_ALL = CUTOFFS.iloc[0].to_pydatetime() - pd.Timedelta(days=1)
SPACING = 4  # sessions between reports, so each is its own story (window 3) and its own claim


def direction_record(position: int, direction: float):
    cutoff = CUTOFFS[SESSIONS[position]]
    return make_event_record(
        article_id=f"gdelt:{position}",
        source="gdelt2",
        available_utc=(cutoff - pd.Timedelta(hours=1)).to_pydatetime(),
        resolved_impacts=[
            ResolvedImpact(
                instrument="ACME", kind="company", direction=direction, match_method="alias"
            )
        ],
    )


def rule_built_claims(positives: int, negatives: int):
    positions = [SPACING * n for n in range(positives + negatives)]
    records = [direction_record(p, 1.0 if n < positives else -1.0) for n, p in enumerate(positions)]
    return positions, claims_from_events(records, PARAMS["story_window_sessions"], CUTOFFS)


def test_the_claims_still_valid_are_at_most_one_story_window_so_asof_cannot_hold_the_history(
    tmp_path,
):
    positions, claims = rule_built_claims(30, 12)
    stores = make_stores(tmp_path, claims=claims)
    cutoff = CUTOFFS.iloc[positions[-1]].to_pydatetime()
    assert len(claims_asof(stores.claim_store, cutoff, attribute="event_direction")) <= 1
    history = claims_history_asof(
        stores.claim_store, cutoff, BEFORE_ALL, attribute="event_direction"
    )
    assert len(history) == 42


def test_run_fires_direction_drift_on_rule_built_three_session_claims(tmp_path):
    positions, claims = rule_built_claims(30, 12)
    here = positions[-1]
    stores = make_stores(tmp_path, records=feed(range(0, here + 1)), claims=claims)
    shifts, report = run_shift_detectors(stores, SESSIONS[here].date(), CUTOFFS, PARAMS)
    [drift] = [s for s in shifts if s.kind == "direction_drift"]
    assert drift.subject == "ACME" and drift.statistic < -PARAMS["direction_drift"]["h"]
    assert report.by_kind["direction_drift"] == 1


def test_a_steady_direction_history_is_quiet_in_the_pipeline(tmp_path):
    positions, claims = rule_built_claims(42, 0)
    here = positions[-1]
    stores = make_stores(tmp_path, records=feed(range(0, here + 1)), claims=claims)
    shifts, _ = run_shift_detectors(stores, SESSIONS[here].date(), CUTOFFS, PARAMS)
    assert not [s for s in shifts if s.kind == "direction_drift"]


def test_the_history_is_read_as_of_the_cutoff_and_bounded_by_the_lookback(tmp_path):
    positions, claims = rule_built_claims(30, 12)
    stores = make_stores(tmp_path, claims=claims)
    early = CUTOFFS.iloc[positions[10]].to_pydatetime()
    assert (
        len(claims_history_asof(stores.claim_store, early, BEFORE_ALL)) == 11
    )  # later ones unknown
    recent = claims_history_asof(
        stores.claim_store,
        early,
        CUTOFFS.iloc[positions[5]].to_pydatetime() - pd.Timedelta(hours=2),
    )
    assert len(recent) == 6  # the lookback start is inclusive
