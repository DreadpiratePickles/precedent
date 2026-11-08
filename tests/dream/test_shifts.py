import time as wall_clock
from datetime import UTC, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import make_event_record

from precedent.dream.claim_schema import Claim, Shift, ShiftKind, shift_id_for
from precedent.dream.claim_store import (
    ClaimStore,
    ResolutionStore,
    ShiftStore,
    TensionStore,
)
from precedent.dream.detectors import DETECTOR_VERSION
from precedent.dream.inputs import DreamStores, load_detector_params
from precedent.dream.shifts import (
    SHIFT_DETECTOR_VERSION,
    SKIP_REASONS,
    ShiftReport,
    _event_counts,
    _rate_z,
    direction_drift,
    event_rate_shift,
    link_churn,
    run_shift_detectors,
)
from precedent.ripple.link_store import GRAPH_COLUMNS, Link, LinkStore
from precedent.spine.event_record import ResolvedImpact
from precedent.spine.pit_alignment import build_session_cutoffs

SESSIONS = pd.bdate_range("2023-01-02", periods=620)  # naive session dates
CUTOFFS = build_session_cutoffs(SESSIONS, time(16, 0))
PARAMS = load_detector_params()


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def events_at(positions, subject: str = "ACME") -> pd.DataFrame:
    """One event per entry in `positions` (a position may repeat for several events)."""
    return pd.DataFrame(
        {"subject": subject, "session": [SESSIONS[p] for p in positions]},
        columns=["subject", "session"],
    )


def planted_positions() -> list[int]:
    """0.1 events per session for 250 sessions, then 0.3 per session for 60; no RNG."""
    return list(range(0, 250, 10)) + list(range(250, 310, 3))


# --- event_rate_shift --------------------------------------------------------------------


def test_rate_shift_fires_on_a_planted_tripling():
    shift = event_rate_shift(events_at(planted_positions()), "ACME", SESSIONS[310], SESSIONS)
    assert shift is not None and shift.kind == "event_rate" and shift.statistic > 3
    assert shift.subject == "ACME" and shift.detected_session == SESSIONS[310].date()
    assert shift.threshold == 3.0 and shift.window_sessions == 60
    assert shift.shift_id == shift_id_for(ShiftKind.event_rate, "ACME")
    assert shift.detector_version == SHIFT_DETECTOR_VERSION == "dream-shift-v1"
    assert shift.recorded_utc == datetime(2024, 3, 11, tzinfo=UTC)  # the session's midnight UTC


def test_rate_shift_value_is_the_poisson_z_of_the_window_against_the_baseline():
    # baseline positions 1..250 hold 25 events; window positions 251..310 hold 20
    # lambda_b = 25 / 250 = 0.1, expected = 6, z = (20 - 6) / sqrt(6)
    positions = list(range(1, 251, 10)) + list(range(251, 311, 3))
    shift = event_rate_shift(events_at(positions), "ACME", SESSIONS[310], SESSIONS)
    assert shift is not None
    assert shift.statistic == pytest.approx((20 - 6) / np.sqrt(6))


def test_rate_shift_takes_recorded_utc_from_the_keyword():
    stamped = utc("2024-03-08T20:00:00Z")
    shift = event_rate_shift(
        events_at(planted_positions()), "ACME", SESSIONS[310], SESSIONS, recorded_utc=stamped
    )
    assert shift is not None and shift.recorded_utc == stamped


POOLED_SEEDS = range(1000, 1040)  # declared before the first run (D48); never re-seeded


def test_rate_shift_is_quiet_on_poisson_noise_pooled_over_forty_paths():
    # One 200-session path has overlapping windows and measures little (seed 11 alone fired on
    # 0.030 of its sessions); the 0.02 bound applies to the pool of 40 paths x 200 sessions.
    scored = range(310, 510)
    fired = 0
    started = wall_clock.perf_counter()
    for seed in POOLED_SEEDS:
        counts = np.random.default_rng(seed).poisson(0.1, size=len(SESSIONS))
        events = events_at(np.repeat(np.arange(len(SESSIONS)), counts))
        per_position = _event_counts(events, "ACME", SESSIONS)
        no_gap = np.zeros(len(SESSIONS), dtype=bool)
        for here in scored:
            z, reason = _rate_z(per_position, here, 60, 250, no_gap)
            fired += reason is None and z is not None and z > 3.0
    total = len(POOLED_SEEDS) * len(scored)
    print(f"pooled quiet-case event-rate fire rate {fired / total:.4f} ({fired}/{total})")
    assert fired / total < 0.02
    assert wall_clock.perf_counter() - started < 10


def test_the_scoring_helper_agrees_with_event_rate_shift_on_a_sample_of_sessions():
    counts = np.random.default_rng(POOLED_SEEDS[0]).poisson(0.1, size=len(SESSIONS))
    events = events_at(np.repeat(np.arange(len(SESSIONS)), counts))
    per_position = _event_counts(events, "ACME", SESSIONS)
    no_gap = np.zeros(len(SESSIONS), dtype=bool)
    for here in range(310, 510, 7):
        z, _ = _rate_z(per_position, here, 60, 250, no_gap)
        shift = event_rate_shift(events, "ACME", SESSIONS[here], SESSIONS)
        assert (shift is not None) == (z is not None and z > 3.0)
        if shift is not None:
            assert shift.statistic == z


def test_rate_shift_is_quiet_when_the_rate_is_unchanged():
    positions = list(range(0, 310, 10))
    assert event_rate_shift(events_at(positions), "ACME", SESSIONS[310], SESSIONS) is None


def test_a_feed_gap_inside_the_window_is_excluded_not_counted_as_zero():
    # 0.1 events per session throughout, but nothing arrives for 10 sessions of the window
    positions = [p for p in range(0, 311, 10) if not 270 <= p < 280]
    coverage = pd.Series(1, index=SESSIONS)
    coverage.iloc[270:280] = 0
    with_gap = event_rate_shift(
        events_at(positions), "ACME", SESSIONS[310], SESSIONS, coverage=coverage
    )
    assert with_gap is None  # the gap must not read as a rate drop


def test_a_gap_does_not_hide_a_real_rise_in_the_remaining_window_sessions():
    # 30 of the 60 window sessions are a feed gap; the 30 live ones run at 0.3 against 0.1
    positions = list(range(1, 251, 10)) + list(range(281, 311, 3))
    coverage = pd.Series(1, index=SESSIONS)
    coverage.iloc[251:281] = 0
    kwargs = {"coverage": coverage}
    shift = event_rate_shift(events_at(positions), "ACME", SESSIONS[310], SESSIONS, **kwargs)
    # window: 10 events in 30 live sessions; expected 0.1 * 30 = 3; z = 7 / sqrt(3)
    assert shift is not None and shift.statistic == pytest.approx(7 / np.sqrt(3))
    # counting the gap as 60 sessions of zeros would expect 6 and read z = 4 / sqrt(6) < 3
    assert event_rate_shift(events_at(positions), "ACME", SESSIONS[310], SESSIONS) is None


def test_a_gap_in_the_baseline_does_not_lower_the_baseline_rate():
    # 200 of the 250 baseline sessions are a feed gap; the 50 live ones hold 5 events (0.1)
    # and the window carries the same 0.1 rate
    baseline = [p for p in range(1, 251, 10) if p > 200]
    window = list(range(251, 311, 10))
    coverage = pd.Series(1, index=SESSIONS)
    coverage.iloc[1:201] = 0
    events = events_at(baseline + window)
    assert event_rate_shift(events, "ACME", SESSIONS[310], SESSIONS, coverage=coverage) is None
    # counted as zeros the baseline rate reads 0.02 and the unchanged window looks like a rise
    assert event_rate_shift(events, "ACME", SESSIONS[310], SESSIONS) is not None


def test_rate_shift_is_none_without_a_baseline_or_enough_sessions():
    window_only = events_at(range(251, 311, 3))
    assert event_rate_shift(window_only, "ACME", SESSIONS[310], SESSIONS) is None  # no baseline
    assert event_rate_shift(events_at([1, 2, 3]), "ACME", SESSIONS[40], SESSIONS) is None
    assert event_rate_shift(events_at([]), "ACME", SESSIONS[310], SESSIONS) is None
    other = events_at(planted_positions(), subject="OTHER")
    assert event_rate_shift(other, "ACME", SESSIONS[310], SESSIONS) is None


def test_rate_shift_clips_the_baseline_at_the_start_of_the_axis():
    # session 100: window 41..100, baseline 0..40 (41 sessions) holds 4 events (0.0976 a session)
    positions = [5, 15, 25, 35] + list(range(41, 101, 3))
    shift = event_rate_shift(events_at(positions), "ACME", SESSIONS[100], SESSIONS)
    assert shift is not None and shift.statistic > 3


def test_rate_shift_rejects_a_session_off_the_axis():
    with pytest.raises(ValueError, match="not on the sessions axis"):
        event_rate_shift(events_at([1]), "ACME", pd.Timestamp("2023-01-07").date(), SESSIONS)


def test_rate_shift_ignores_events_after_the_session():
    base = events_at(planted_positions())
    later = pd.concat([base, events_at(range(311, 400))], ignore_index=True)
    first = event_rate_shift(base, "ACME", SESSIONS[310], SESSIONS)
    second = event_rate_shift(later, "ACME", SESSIONS[310], SESSIONS)
    assert first is not None and second is not None and first.statistic == second.statistic


# --- direction_drift ---------------------------------------------------------------------


def direction_claims(values: list[str], subject: str = "ACME", start: int = 0) -> pd.DataFrame:
    """event_direction claims as `claims_asof` shows them, one per session from `start`."""
    rows = []
    for offset, value in enumerate(values):
        valid_from = pd.Timestamp(SESSIONS[start + offset], tz="UTC") + pd.Timedelta(hours=21)
        claim = Claim(
            subject=subject,
            attribute="event_direction",
            period="unspecified",
            value=value,
            valid_from_utc=valid_from.to_pydatetime(),
            recorded_utc=valid_from.to_pydatetime(),
            source_doc_id=f"gdelt:{subject}:{start + offset}",
            evidence_span=None,
            extractor_version="rules-test-v1",
        )
        rows.append(claim.model_dump(mode="json"))
    frame = pd.DataFrame(rows, columns=list(Claim.model_fields))
    for column in ("valid_from_utc", "valid_to_utc", "recorded_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


@pytest.fixture
def planted_claims() -> pd.DataFrame:
    return direction_claims(["positive"] * 30 + ["negative"] * 12)


def test_cusum_fires_on_a_sign_flip_and_is_recomputable(planted_claims):
    first = next(
        s
        for s in SESSIONS
        if direction_drift(planted_claims, "ACME", s.date(), SESSIONS) is not None
    )
    shift = direction_drift(planted_claims, "ACME", first.date(), SESSIONS)
    again = direction_drift(planted_claims, "ACME", first.date(), SESSIONS)
    assert shift is not None and again is not None and again.statistic == shift.statistic
    assert shift.kind == "direction_drift" and shift.subject == "ACME"
    assert shift.shift_id == shift_id_for(ShiftKind.direction_drift, "ACME")
    assert shift.threshold == 4.0 and shift.window_sessions > 0


def visible_at(claims: pd.DataFrame, position: int) -> pd.DataFrame:
    cutoff = pd.Timestamp(SESSIONS[position], tz="UTC") + pd.Timedelta(hours=21)
    return claims[claims["valid_from_utc"] <= cutoff]


FLIP = ["positive"] * 30 + ["negative"] * 12


def test_cusum_first_firing_session_is_at_or_after_the_first_negative_and_is_signed_negative():
    claims = direction_claims(FLIP)
    seen = {
        p: direction_drift(visible_at(claims, p), "ACME", SESSIONS[p].date(), SESSIONS)
        for p in range(len(claims))
    }
    first = next(p for p, shift in seen.items() if shift is not None)
    assert first >= 30  # nothing fires while the direction is steady (D49)
    assert first == 32  # the third negative; deviations from the earlier mean are -2, -60/31, -15/8
    shift = seen[first]
    assert shift is not None and shift.statistic < 0
    assert shift.statistic == pytest.approx(-(1.5 + 60 / 31 - 0.5 + 15 / 8 - 0.5))
    # a recomputation from the same visible claims gives the same statistic
    again = direction_drift(visible_at(claims, first), "ACME", SESSIONS[first].date(), SESSIONS)
    assert again is not None and again.statistic == shift.statistic


@pytest.mark.parametrize("value", ["positive", "negative", "neutral"])
def test_cusum_never_fires_on_a_steady_direction(value):
    claims = direction_claims([value] * 40)
    fired = [
        p
        for p in range(len(claims))
        if direction_drift(visible_at(claims, p), "ACME", SESSIONS[p].date(), SESSIONS)
    ]
    assert fired == []


def test_cusum_is_quiet_on_alternating_signs():
    claims = direction_claims(["positive", "negative"] * 20)
    assert direction_drift(claims, "ACME", SESSIONS[39].date(), SESSIONS) is None


def test_cusum_value_is_the_hand_computed_sum_of_deviations_from_the_earlier_mean():
    # deviations 0, 0, -2, -4/3, -1, -4/5; S- after each: 0, 0, 1.5, 7/3, 17/6, 47/15
    values = ["positive", "positive"] + ["negative"] * 4
    shift = direction_drift(
        direction_claims(values), "ACME", SESSIONS[5].date(), SESSIONS, k=0.5, h=3.0
    )
    assert shift is not None and shift.statistic == pytest.approx(-47 / 15)
    assert shift.threshold == 3.0
    assert direction_drift(direction_claims(values), "ACME", SESSIONS[5].date(), SESSIONS) is None
    shuffled = direction_claims(values).sample(frac=1.0, random_state=3)
    again = direction_drift(shuffled, "ACME", SESSIONS[5].date(), SESSIONS, k=0.5, h=3.0)
    assert again is not None and again.statistic == pytest.approx(-47 / 15)


def test_cusum_a_positive_flip_gives_a_positive_statistic():
    shift = direction_drift(
        direction_claims(["negative"] * 30 + ["positive"] * 3),
        "ACME",
        SESSIONS[32].date(),
        SESSIONS,
    )
    assert shift is not None and shift.statistic > 4.0


def test_cusum_ignores_other_subjects_and_attributes():
    other_subject = direction_claims(FLIP, subject="OTHER")
    assert direction_drift(other_subject, "ACME", SESSIONS[41].date(), SESSIONS) is None
    other_attribute = direction_claims(FLIP)
    other_attribute["attribute"] = "capex_direction"
    assert direction_drift(other_attribute, "ACME", SESSIONS[41].date(), SESSIONS) is None
    assert direction_drift(direction_claims([]), "ACME", SESSIONS[8].date(), SESSIONS) is None


def test_cusum_rejects_a_session_off_the_axis_and_an_unknown_value():
    with pytest.raises(ValueError, match="not on the sessions axis"):
        direction_drift(
            direction_claims(["positive"]), "ACME", pd.Timestamp("2023-01-07").date(), SESSIONS
        )
    bad = direction_claims(["positive"])
    bad["value"] = "up"
    with pytest.raises(ValueError, match="event_direction"):
        direction_drift(bad, "ACME", SESSIONS[0].date(), SESSIONS)


# --- link_churn --------------------------------------------------------------------------


def graph(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["source_node", "receiver_node", "relation"])
    for column in GRAPH_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame[GRAPH_COLUMNS]


def test_link_churn_fires_when_most_of_the_link_set_changed():
    then = graph([("ACME", "B1", "supplier_of"), ("ACME", "B2", "supplier_of")])
    now = graph([("ACME", "B2", "supplier_of"), ("ACME", "B3", "supplier_of")])
    # intersection 1, union 3: distance 2/3
    shift = link_churn(now, then, "ACME", SESSIONS[300].date())
    assert shift is not None and shift.kind == "link_churn"
    assert shift.statistic == pytest.approx(2 / 3) and shift.threshold == 0.5
    assert shift.shift_id == shift_id_for(ShiftKind.link_churn, "ACME")
    assert shift.window_sessions == 250


def test_link_churn_is_quiet_at_or_below_the_threshold():
    then = graph([("ACME", f"B{i}", "supplier_of") for i in range(3)])
    same = graph([("ACME", f"B{i}", "supplier_of") for i in range(3)])
    assert link_churn(same, then, "ACME", SESSIONS[300].date()) is None
    swapped_one = graph(
        [
            ("ACME", "B0", "supplier_of"),
            ("ACME", "B1", "supplier_of"),
            ("ACME", "B9", "supplier_of"),
        ]
    )
    # intersection 2, union 4: a distance of exactly 0.5 does not exceed the threshold
    assert link_churn(swapped_one, then, "ACME", SESSIONS[300].date()) is None
    assert link_churn(swapped_one, then, "ACME", SESSIONS[300].date(), threshold=0.4) is not None


def test_link_churn_counts_links_in_both_directions_and_by_relation():
    then = graph([("ACME", "B1", "supplier_of"), ("C1", "ACME", "customer_of")])
    now = graph([("ACME", "B1", "customer_of"), ("C1", "ACME", "customer_of")])
    # (B1, supplier_of) -> (B1, customer_of): intersection 1, union 3
    shift = link_churn(now, then, "ACME", SESSIONS[300].date(), baseline_sessions=120)
    assert shift is not None and shift.statistic == pytest.approx(2 / 3)
    assert shift.window_sessions == 120


def test_link_churn_ignores_other_subjects_and_is_none_when_both_sets_are_empty():
    then = graph([("OTHER", "B1", "supplier_of")])
    now = graph([("OTHER", "B2", "supplier_of")])
    assert link_churn(now, then, "ACME", SESSIONS[300].date()) is None
    assert link_churn(graph([]), graph([]), "ACME", SESSIONS[300].date()) is None


def test_link_churn_when_every_link_appeared_or_vanished_is_full_distance():
    shift = link_churn(
        graph([("ACME", "B1", "supplier_of")]), graph([]), "ACME", SESSIONS[300].date()
    )
    assert shift is not None and shift.statistic == 1.0


# --- run_shift_detectors -----------------------------------------------------------------

RUN_POSITION = 310
RUN_SESSION = SESSIONS[RUN_POSITION].date()


def record_at(position: int, article_id: str, instruments: tuple[str, ...] = ()):
    cutoff = CUTOFFS[SESSIONS[position]]
    return make_event_record(
        article_id=article_id,
        available_utc=(cutoff - pd.Timedelta(hours=1)).to_pydatetime(),
        resolved_impacts=[
            ResolvedImpact(instrument=i, kind="event", direction=0.1, match_method="alias")
            for i in instruments
        ],
    )


def feed(positions, skip=()) -> list:
    """One impact-less record per session, so no session reads as a feed gap, except `skip`."""
    return [record_at(p, f"filler:{p}") for p in positions if p not in skip]


def make_stores(
    tmp_path: Path, *, records=(), claims=(), links=(), with_link_store: bool = True
) -> DreamStores:
    root = tmp_path / "dream"
    root.mkdir(exist_ok=True)
    claim_store = ClaimStore(root)
    claim_store.append(claims)
    link_store = LinkStore(root) if with_link_store else None
    if link_store is not None:
        link_store.append(links)
    return DreamStores(
        claim_store=claim_store,
        tension_store=TensionStore(root),
        shift_store=ShiftStore(root),
        resolution_store=ResolutionStore(root),
        event_records=list(records),
        link_store=link_store,
        returns=None,
        market=None,
        sessions=SESSIONS,
        session_cutoffs_utc=CUTOFFS,
    )


def planted_records(skip=()) -> list:
    impacted = [record_at(p, f"acme:{p}", ("ACME",)) for p in planted_positions()]
    return impacted + feed(range(0, RUN_POSITION + 1), skip=skip)


def test_run_fires_the_planted_rate_shift_and_stamps_it_with_the_cutoff(tmp_path):
    stores = make_stores(tmp_path, records=planted_records())
    shifts, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert isinstance(report, ShiftReport)
    [shift] = shifts
    assert shift.kind == "event_rate" and shift.subject == "ACME" and shift.statistic > 3
    assert shift.detected_session == RUN_SESSION
    assert shift.recorded_utc == CUTOFFS[pd.Timestamp(RUN_SESSION)].to_pydatetime()
    assert shift.detector_version == SHIFT_DETECTOR_VERSION != DETECTOR_VERSION
    assert report.by_kind == {"event_rate": 1, "direction_drift": 0, "link_churn": 0}
    assert report.subjects_checked == 1 and report.feed_gap_sessions_excluded == 0
    assert stores.shift_store.read_all() == []  # the detectors never append


def test_run_counts_the_zero_document_sessions_in_baseline_and_window_and_excludes_them(tmp_path):
    # sessions 270-279 receive no document at all, except where a planted ACME event lands
    outage = set(range(270, 280))
    live = outage & set(planted_positions())
    stores = make_stores(tmp_path, records=planted_records(skip=outage))
    shifts, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert report.feed_gap_sessions_excluded == len(outage - live)
    assert [s.kind for s in shifts] == ["event_rate"]


def stored_shift(fresh: Shift, statistic: float) -> Shift:
    return Shift(
        kind=fresh.kind,
        subject=fresh.subject,
        detected_session=SESSIONS[RUN_POSITION - 1].date(),
        statistic=statistic,
        threshold=fresh.threshold,
        window_sessions=fresh.window_sessions,
        detector_version=fresh.detector_version,
        recorded_utc=utc("2024-03-07T21:00:00Z"),
    )


def test_run_does_not_re_report_a_shift_that_grew_by_no_more_than_the_growth_parameter(tmp_path):
    stores = make_stores(tmp_path, records=planted_records())
    [fresh], _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert PARAMS["shift_rereport_growth"] == 0.5
    stores.shift_store.append([stored_shift(fresh, fresh.statistic / 1.5 + 0.01)])
    assert run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)[0] == []


def test_run_re_reports_a_shift_whose_statistic_grew_past_the_growth_parameter(tmp_path):
    stores = make_stores(tmp_path, records=planted_records())
    [fresh], _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    stores.shift_store.append([stored_shift(fresh, fresh.statistic / 1.5 - 0.01)])
    again, _ = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert [s.shift_id for s in again] == [fresh.shift_id]
    # the caller records it with ShiftStore.refire; the next identical run is then quiet
    stores.shift_store.refire(fresh.shift_id, fresh.statistic, RUN_SESSION, fresh.recorded_utc)
    assert run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)[0] == []


def test_run_fires_direction_drift_from_the_claims_visible_at_the_cutoff(tmp_path):
    cutoff = CUTOFFS[pd.Timestamp(RUN_SESSION)]

    def direction_claim(doc: str, value: str, valid_from, recorded) -> Claim:
        return Claim(
            subject="ACME",
            attribute="event_direction",
            period="unspecified",
            value=value,
            valid_from_utc=valid_from.to_pydatetime(),
            recorded_utc=recorded.to_pydatetime(),
            source_doc_id=doc,
            evidence_span=None,
            extractor_version="rules-test-v1",
        )

    values = ["positive"] * 20 + ["negative"] * 3
    claims = [
        direction_claim(f"gdelt:{n}", value, *[cutoff - pd.Timedelta(days=30 - n)] * 2)
        for n, value in enumerate(values)
    ]
    # a fourth negative recorded after the cutoff is invisible to the run
    late = direction_claim("gdelt:late", "negative", cutoff, cutoff + pd.Timedelta(days=1))
    stores = make_stores(tmp_path, records=feed(range(0, RUN_POSITION + 1)), claims=[*claims, late])
    shifts, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    [shift] = shifts
    # deviations -2, -40/21, -20/11 against k = 0.5
    assert shift.kind == "direction_drift"
    assert shift.statistic == pytest.approx(-(1.5 + 40 / 21 - 0.5 + 20 / 11 - 0.5))
    assert report.by_kind["direction_drift"] == 1 and report.subjects_checked == 1


def churn_link(receiver: str, valid_from: str, valid_to: str) -> Link:
    return Link(
        source_node="ACME",
        receiver_node=receiver,
        relation="supplier_of",
        weight=0.5,
        valid_from_utc=utc(valid_from),
        valid_to_utc=utc(valid_to),
        source_doc_id=f"10K-{receiver}",
        evidence_span=None,
        extractor_version="rules-test-v1",
    )


def test_run_reads_link_churn_between_now_and_baseline_sessions_ago(tmp_path):
    then_cutoff = CUTOFFS.iloc[RUN_POSITION - PARAMS["event_rate"]["baseline_sessions"]]
    now_cutoff = CUTOFFS.iloc[RUN_POSITION]
    handover = (then_cutoff + pd.Timedelta(days=30)).isoformat()
    far = "2030-01-01T00:00:00Z"
    links = [
        churn_link("B1", "2022-01-01T00:00:00Z", handover),  # visible then, gone now
        churn_link("B2", "2022-01-01T00:00:00Z", handover),
        churn_link("B3", handover, far),  # visible now, not then
        churn_link("B4", handover, far),
    ]
    assert then_cutoff < pd.Timestamp(handover) < now_cutoff
    stores = make_stores(tmp_path, records=feed(range(0, RUN_POSITION + 1)), links=links)
    shifts, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    churn = [s for s in shifts if s.kind == "link_churn"]
    assert {s.subject for s in churn} == {"ACME", "B1", "B2", "B3", "B4"}
    acme = next(s for s in churn if s.subject == "ACME")
    assert acme.statistic == 1.0 and acme.window_sessions == 250
    assert report.by_kind["link_churn"] == 5 and report.subjects_checked == 5


def test_run_reports_every_skip_reason_with_zero_when_nothing_was_skipped(tmp_path):
    stores = make_stores(tmp_path, records=planted_records())
    _, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert report.skipped == {"no_link_store": 0, "short_history": 0, "no_baseline_events": 0}
    assert set(report.skipped) == set(SKIP_REASONS)


def test_run_without_a_link_store_skips_link_churn_and_counts_it(tmp_path):
    stores = make_stores(tmp_path, records=feed(range(0, RUN_POSITION + 1)), with_link_store=False)
    shifts, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert shifts == [] and report.by_kind["link_churn"] == 0
    assert report.skipped == {"no_link_store": 1, "short_history": 0, "no_baseline_events": 0}


def test_run_skips_link_churn_when_fewer_than_baseline_sessions_precede_and_counts_it(tmp_path):
    links = [churn_link("B1", "2022-01-01T00:00:00Z", "2030-01-01T00:00:00Z")]
    stores = make_stores(tmp_path, records=feed(range(0, 101)), links=links)
    shifts, report = run_shift_detectors(stores, SESSIONS[100].date(), CUTOFFS, PARAMS)
    assert shifts == [] and report.by_kind["link_churn"] == 0
    assert report.skipped["short_history"] == 1  # churn only: 100 sessions precede, window is 60


def test_run_counts_short_history_for_the_rate_test_and_for_churn(tmp_path):
    links = [churn_link("B1", "2022-01-01T00:00:00Z", "2030-01-01T00:00:00Z")]
    stores = make_stores(tmp_path, records=feed(range(0, 41)), links=links)
    shifts, report = run_shift_detectors(stores, SESSIONS[40].date(), CUTOFFS, PARAMS)
    assert shifts == [] and report.skipped["short_history"] == 2


def test_run_counts_subjects_whose_baseline_held_no_event(tmp_path):
    newcomer = [record_at(p, f"new:{p}", ("NEWCO",)) for p in (300, 305)]
    stores = make_stores(tmp_path, records=feed(range(0, RUN_POSITION + 1)) + newcomer)
    shifts, report = run_shift_detectors(stores, RUN_SESSION, CUTOFFS, PARAMS)
    assert shifts == []  # no baseline rate, so no test; the skip is what says so
    assert report.skipped["no_baseline_events"] == 1 and report.subjects_checked == 1
