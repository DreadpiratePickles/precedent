"""D73: a stored shift is re-reported on growth, in a new episode, or when a drift reverses."""

import pandas as pd
import pytest
from test_shifts import (
    CUTOFFS,
    PARAMS,
    RUN_POSITION,
    RUN_SESSION,
    SESSIONS,
    feed,
    make_stores,
    planted_records,
)

from precedent.dream.claim_schema import Claim, Shift, ShiftKind
from precedent.dream.shifts import run_shift_detectors

WINDOW = PARAMS["event_rate"]["window_sessions"]


def stored_at(fresh: Shift, detected_position: int, statistic: float) -> Shift:
    return Shift(
        kind=fresh.kind,
        subject=fresh.subject,
        detected_session=SESSIONS[detected_position].date(),
        statistic=statistic,
        threshold=fresh.threshold,
        window_sessions=fresh.window_sessions,
        detector_version=fresh.detector_version,
        recorded_utc=CUTOFFS.iloc[detected_position].to_pydatetime(),
    )


def fresh_rate_shift(tmp_path):
    stores = make_stores(tmp_path, records=planted_records())
    [fresh], _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    return stores, fresh


def test_a_later_episode_of_the_same_size_is_reported_again(tmp_path):
    stores, fresh = fresh_rate_shift(tmp_path)
    # the stored report is as large as today's and more than one window old: a new episode
    stores.shift_store.append([stored_at(fresh, RUN_POSITION - WINDOW - 1, fresh.statistic)])
    [again], _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert again.shift_id == fresh.shift_id


def test_the_same_episode_within_one_window_stays_quiet(tmp_path):
    stores, fresh = fresh_rate_shift(tmp_path)
    stores.shift_store.append([stored_at(fresh, RUN_POSITION - WINDOW, fresh.statistic)])
    assert run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)[0] == []  # exactly one window


def test_a_re_fire_restarts_the_episode_clock(tmp_path):
    stores, fresh = fresh_rate_shift(tmp_path)
    old = stored_at(fresh, RUN_POSITION - 3 * WINDOW, fresh.statistic)
    stores.shift_store.append([old])
    stores.shift_store.refire(
        fresh.shift_id, fresh.statistic, RUN_SESSION, CUTOFFS.iloc[RUN_POSITION].to_pydatetime()
    )
    assert run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)[0] == []


def direction_claims(values: list[str]) -> list[Claim]:
    cutoff = CUTOFFS.iloc[RUN_POSITION]
    return [
        Claim(
            subject="ACME",
            attribute="event_direction",
            period="unspecified",
            value=value,
            valid_from_utc=(cutoff - pd.Timedelta(days=60 - n)).to_pydatetime(),
            recorded_utc=(cutoff - pd.Timedelta(days=60 - n)).to_pydatetime(),
            source_doc_id=f"gdelt:{n}",
            evidence_span=None,
            extractor_version="rules-test-v1",
        )
        for n, value in enumerate(values)
    ]


@pytest.mark.parametrize(
    ("flip", "stored_sign", "reported"),
    [("negative", +1.0, True), ("negative", -1.0, False), ("positive", -1.0, True)],
    ids=["reversed_to_negative", "same_direction_quiet", "reversed_to_positive"],
)
def test_a_drift_that_reverses_at_the_same_magnitude_is_reported(
    tmp_path, flip, stored_sign, reported
):
    steady = "negative" if flip == "positive" else "positive"
    claims = direction_claims([steady] * 20 + [flip] * 4)
    stores = make_stores(tmp_path, records=feed(range(0, RUN_POSITION + 1)), claims=claims)
    [fresh], _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert fresh.kind is ShiftKind.direction_drift
    # a recent stored report of the same size and the given sign (today's is negative when the
    # history flipped to negative, positive when it flipped to positive)
    stores.shift_store.append(
        [stored_at(fresh, RUN_POSITION - 1, abs(fresh.statistic) * stored_sign)]
    )
    shifts, _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert bool(shifts) is reported
