from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from precedent.dream.claim_schema import Claim, Resolution, Shift, Tension
from precedent.dream.claim_store import (
    ClaimStore,
    ConflictError,
    ResolutionStore,
    ShiftStore,
    TensionStore,
    claims_asof,
    supersede,
)


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def claim(
    subject="ACME",
    attribute="capex_direction",
    value="up",
    valid_from="2024-03-01T21:30:00Z",
    recorded="2024-03-01T22:00:00Z",
    source_doc_id="doc-1",
    extractor="rules-8k-v1",
    evidence="capex will rise",
) -> Claim:
    return Claim(
        subject=subject,
        attribute=attribute,
        period="current_quarter",
        value=value,
        valid_from_utc=valid_from,
        recorded_utc=recorded,
        source_doc_id=source_doc_id,
        evidence_span=evidence,
        extractor_version=extractor,
    )


def tension(claim_ids=("a", "b"), kind="value_conflict", detector="dream-detect-v1") -> Tension:
    return Tension(
        kind=kind,
        group_key="ACME|capex_direction|unspecified",
        claim_ids=list(claim_ids),
        subject="ACME",
        opened_session=date(2024, 3, 4),
        detector_version=detector,
        recorded_utc=utc("2024-03-04T21:30:00"),
        details={"z": 2.5},
    )


def shift(statistic=3.4, detector="dream-detect-v1") -> Shift:
    return Shift(
        kind="event_rate",
        subject="ACME",
        detected_session=date(2024, 3, 4),
        statistic=statistic,
        threshold=3.0,
        window_sessions=60,
        detector_version=detector,
        recorded_utc=utc("2024-03-04T21:30:00"),
    )


def resolution(hypothesis="one_is_stale", claim_id=None) -> Resolution:
    return Resolution(
        resolution_id="r1",
        tension_id="t1",
        hypothesis=hypothesis,
        evidence_span=None,
        proposer_version="dream-resolve-v1",
        status="proposed",
        proposed_utc=utc("2024-03-05T00:00:00"),
        recorded_utc=utc("2024-03-05T00:00:00"),
        claim_id=claim_id,
    )


def line_count(path: Path) -> int:
    return path.read_text().count("\n")


def test_claim_is_invisible_before_its_document_and_after_expiry(tmp_path):
    store = ClaimStore(tmp_path)
    original = claim()
    store.append([original])
    assert claims_asof(store, utc("2024-03-01T21:00:00")).empty  # accepted after that cutoff
    assert len(claims_asof(store, utc("2024-03-04T21:00:00"))) == 1
    store.supersede(
        original.claim_id, new_claim_id="x", at_utc=utc("2024-06-01T21:30:00"), reason="superseded"
    )
    assert claims_asof(store, utc("2024-06-03T21:00:00")).empty


def test_recorded_later_is_invisible_to_a_backfilled_cutoff(tmp_path):
    store = ClaimStore(tmp_path)
    store.append([claim(recorded="2024-09-01T00:00:00Z")])  # backfilled months later
    assert claims_asof(store, utc("2024-03-04T21:00:00")).empty
    assert len(claims_asof(store, utc("2024-09-02T21:00:00"))) == 1


def test_a_cutoff_before_an_expiry_still_sees_the_claim(tmp_path):
    store = ClaimStore(tmp_path)
    original = claim()
    store.append([original])
    store.supersede(original.claim_id, "x", utc("2024-06-01T21:30:00"), "superseded")
    assert len(claims_asof(store, utc("2024-05-01T21:00:00"))) == 1  # rewind: not yet expired
    assert claims_asof(store, utc("2024-06-01T21:30:00")).empty  # valid_to is exclusive


def test_supersede_appends_and_never_rewrites(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    store.append([c1])
    path = tmp_path / "claims.jsonl"
    lines_before = line_count(path)
    first_line = path.read_text().splitlines()[0]
    result = supersede(store, c1.claim_id, "c2", utc("2024-06-01T21:30:00"), "restated")
    assert line_count(path) == lines_before + 1
    assert path.read_text().splitlines()[0] == first_line
    assert result.valid_to_utc == utc("2024-06-01T21:30:00")
    assert result.superseded_by == "c2"
    assert result.expiry_reason == "restated"
    assert result.recorded_utc == utc("2024-06-01T21:30:00")
    assert len(store.read_all()) == 2
    assert store.latest() == [result]


def test_supersede_allows_a_dated_expiry_with_no_successor(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    store.append([c1])
    expired = store.supersede(c1.claim_id, None, utc("2025-06-01T00:00:00"), "annual_expiry")
    assert expired.superseded_by is None
    assert expired.expiry_reason == "annual_expiry"


def test_supersede_refuses_unknown_and_already_expired_claims(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    store.append([c1])
    with pytest.raises(KeyError):
        store.supersede("missing", None, utc("2024-06-01T00:00:00"), "x")
    store.supersede(c1.claim_id, None, utc("2024-06-01T00:00:00"), "x")
    with pytest.raises(ConflictError):
        store.supersede(c1.claim_id, None, utc("2024-07-01T00:00:00"), "again")
    assert len(store.read_all()) == 2


def test_supersede_shortens_a_bounded_claim(tmp_path):
    store = ClaimStore(tmp_path)
    bounded = claim().model_copy(update={"valid_to_utc": utc("2024-09-01T00:00:00")})
    store.append([bounded])
    assert len(claims_asof(store, utc("2024-06-15T21:00:00"))) == 1
    shortened = store.supersede(bounded.claim_id, "x", utc("2024-06-01T21:30:00"), "restated")
    assert shortened.valid_to_utc == utc("2024-06-01T21:30:00")
    assert claims_asof(store, utc("2024-07-01T21:00:00")).empty  # between new and old expiry
    assert len(claims_asof(store, utc("2024-05-01T21:00:00"))) == 1  # rewind: still valid


def test_supersede_refuses_a_claim_already_expired_at_that_time(tmp_path):
    store = ClaimStore(tmp_path)
    bounded = claim().model_copy(update={"valid_to_utc": utc("2024-09-01T00:00:00")})
    store.append([bounded])
    with pytest.raises(ConflictError):
        store.supersede(bounded.claim_id, None, utc("2024-09-01T00:00:00"), "x")
    with pytest.raises(ConflictError):
        store.supersede(bounded.claim_id, None, utc("2024-10-01T00:00:00"), "x")
    assert len(store.read_all()) == 1


def test_supersede_refuses_a_time_before_the_claim_starts(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    store.append([c1])
    with pytest.raises(ValueError, match="valid_from_utc"):
        store.supersede(c1.claim_id, None, utc("2024-02-01T00:00:00"), "x")
    assert len(store.read_all()) == 1


def test_append_is_idempotent_and_refuses_changed_content(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    assert store.append([c1]).added == 1
    report = store.append([c1, claim(source_doc_id="doc-2")])
    assert (report.added, report.identical_skipped) == (1, 1)
    assert line_count(tmp_path / "claims.jsonl") == 2
    changed = c1.model_copy(update={"evidence_span": "different words"})
    with pytest.raises(ConflictError):
        store.append([changed])
    assert line_count(tmp_path / "claims.jsonl") == 2


def test_append_after_supersede_is_identical_even_though_lifecycle_changed(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    store.append([c1])
    store.supersede(c1.claim_id, None, utc("2024-06-01T00:00:00"), "x")
    report = store.append([c1])
    assert (report.added, report.identical_skipped) == (0, 1)
    assert len(store.read_all()) == 2


def test_a_failed_append_writes_nothing(tmp_path):
    store = ClaimStore(tmp_path)
    c1 = claim()
    store.append([c1])
    changed = c1.model_copy(update={"evidence_span": "other"})
    with pytest.raises(ConflictError):
        store.append([claim(source_doc_id="doc-9"), changed])
    assert len(store.read_all()) == 1


def test_restated_claim_coexists_and_extends_visibility(tmp_path):
    store = ClaimStore(tmp_path)
    first = claim(source_doc_id="doc-1")
    later = claim(
        source_doc_id="doc-2", valid_from="2024-05-01T21:30:00Z", recorded="2024-05-01T22:00:00Z"
    )
    store.append([first])
    assert store.append([first]).identical_skipped == 1
    assert line_count(tmp_path / "claims.jsonl") == 1
    assert later.claim_id != first.claim_id
    store.append([later])
    store.supersede(first.claim_id, later.claim_id, utc("2024-06-01T21:30:00"), "restated")
    overlap = claims_asof(store, utc("2024-05-15T21:00:00"))
    assert set(overlap["claim_id"]) == {first.claim_id, later.claim_id}
    after_first_expiry = claims_asof(store, utc("2024-07-01T21:00:00"))
    assert list(after_first_expiry["claim_id"]) == [later.claim_id]
    assert list(after_first_expiry["value"]) == ["up"]


def test_claims_asof_filters_and_keeps_the_full_schema_when_empty(tmp_path):
    store = ClaimStore(tmp_path)
    empty = claims_asof(store, utc("2024-03-04T21:00:00"))
    assert empty.empty
    assert list(empty.columns) == list(Claim.model_fields)
    for column in ("valid_from_utc", "valid_to_utc", "recorded_utc"):
        assert str(empty[column].dtype) == "datetime64[ns, UTC]"
    store.append(
        [
            claim(subject="ACME"),
            claim(subject="ACME", attribute="headcount_direction", value="down"),
            claim(subject="BETA"),
        ]
    )
    cutoff = utc("2024-03-04T21:00:00")
    frame = claims_asof(store, cutoff)
    assert len(frame) == 3
    assert list(frame.columns) == list(Claim.model_fields)
    assert str(frame["valid_from_utc"].dtype) == "datetime64[ns, UTC]"
    assert pd.isna(frame["valid_to_utc"]).all()
    assert set(frame["value"]) == {"up", "down"}
    assert set(claims_asof(store, cutoff, subject="ACME")["subject"]) == {"ACME"}
    only = claims_asof(store, cutoff, subject="ACME", attribute="headcount_direction")
    assert list(only["value"]) == ["down"]
    assert claims_asof(store, cutoff, subject="NOPE").empty


def test_claims_asof_rejects_a_naive_cutoff(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        claims_asof(ClaimStore(tmp_path), datetime(2024, 3, 4, 21, 0))


def test_tension_store_lifecycle_is_append_only(tmp_path):
    store = TensionStore(tmp_path)
    t1 = tension()
    assert store.append([t1]).added == 1
    assert store.append([t1]).identical_skipped == 1
    closed = store.close(t1.tension_id, date(2024, 3, 6), utc("2024-03-06T21:30:00"))
    assert closed.status == "closed" and closed.closed_session == date(2024, 3, 6)
    assert line_count(tmp_path / "tensions.jsonl") == 2
    assert store.append([t1]).identical_skipped == 1  # first-record identity, lifecycle ignored
    assert [t.status for t in store.latest()] == ["closed"]
    assert [t.status for t in store.read_all()] == ["open", "closed"]
    with pytest.raises(ConflictError):
        store.close(t1.tension_id, date(2024, 3, 7), utc("2024-03-07T21:30:00"))
    with pytest.raises(KeyError):
        store.acknowledge("missing", utc("2024-03-07T21:30:00"))
    refound = t1.model_copy(update={"detector_version": "dream-detect-v2"})
    lines_before = line_count(tmp_path / "tensions.jsonl")
    report = store.append([refound])  # a later detector version re-finding it is provenance
    assert (report.added, report.identical_skipped) == (0, 1)
    assert line_count(tmp_path / "tensions.jsonl") == lines_before


def test_tension_acknowledge_marks_status_without_a_session(tmp_path):
    store = TensionStore(tmp_path)
    t1 = tension()
    store.append([t1])
    acknowledged = store.acknowledge(t1.tension_id, utc("2024-03-05T00:00:00"))
    assert acknowledged.status == "acknowledged" and acknowledged.closed_session is None
    assert acknowledged.recorded_utc == utc("2024-03-05T00:00:00")


def test_shift_store_refire_updates_latest_only(tmp_path):
    store = ShiftStore(tmp_path)
    s1 = shift()
    store.append([s1])
    fired = store.refire(s1.shift_id, 4.1, date(2024, 3, 5), utc("2024-03-05T21:30:00"))
    assert fired.statistic == 4.1 and fired.detected_session == date(2024, 3, 5)
    assert [s.statistic for s in store.read_all()] == [3.4, 4.1]
    assert [s.statistic for s in store.latest()] == [4.1]
    assert store.append([s1]).identical_skipped == 1
    lines_before = line_count(tmp_path / "shifts.jsonl")
    report = store.append([shift(detector="dream-detect-v2")])
    assert (report.added, report.identical_skipped) == (0, 1)
    assert line_count(tmp_path / "shifts.jsonl") == lines_before
    with pytest.raises(KeyError):
        store.refire("missing", 1.0, date(2024, 3, 5), utc("2024-03-05T21:30:00"))
    assert (tmp_path / "shifts.jsonl").exists()


def test_resolution_store_decide_and_conflict(tmp_path):
    store = ResolutionStore(tmp_path)
    r1 = resolution()
    store.append([r1])
    decided = store.decide("r1", "accepted", utc("2024-03-06T00:00:00"))
    assert decided.status == "accepted" and decided.decided_utc == utc("2024-03-06T00:00:00")
    assert decided.recorded_utc == decided.decided_utc  # D61: the decision is recorded when made
    assert store.read_all()[0].recorded_utc == utc("2024-03-05T00:00:00")
    assert [r.status for r in store.read_all()] == ["proposed", "accepted"]
    assert [r.status for r in store.latest()] == ["accepted"]
    assert store.append([r1]).identical_skipped == 1
    with pytest.raises(ConflictError):
        store.append([resolution(hypothesis="genuine_conflict")])
    with pytest.raises(ValueError):
        store.decide("r1", "proposed", utc("2024-03-07T00:00:00"))
    with pytest.raises(KeyError):
        store.decide("missing", "rejected", utc("2024-03-07T00:00:00"))
    assert (tmp_path / "resolutions.jsonl").exists()


def test_a_resolution_renaming_its_claim_is_a_conflict_not_a_skip(tmp_path):
    """M6: `claim_id` is identity — a second proposal under the same id naming a different
    claim must conflict, not read as identical_skipped."""
    store = ResolutionStore(tmp_path)
    store.append([resolution(claim_id="claim-A")])
    assert store.append([resolution(claim_id="claim-A")]).identical_skipped == 1
    with pytest.raises(ConflictError):
        store.append([resolution(claim_id="claim-B")])


def test_stores_read_missing_files_as_empty_and_keep_first_seen_order(tmp_path):
    assert ClaimStore(tmp_path).read_all() == []
    assert TensionStore(tmp_path).latest() == []
    store = ClaimStore(tmp_path)
    a, b = claim(source_doc_id="a"), claim(source_doc_id="b")
    store.append([a, b])
    store.supersede(a.claim_id, None, utc("2024-06-01T00:00:00"), "x")
    assert [c.source_doc_id for c in store.latest()] == ["a", "b"]
