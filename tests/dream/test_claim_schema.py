import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from precedent.dream.claim_schema import (
    Attribute,
    Claim,
    Hypothesis,
    Period,
    Resolution,
    Shift,
    ShiftKind,
    Tension,
    TensionKind,
    Value,
    claim_id_for,
    claim_json_schema,
    shift_id_for,
    tension_id_for,
)

ROOT = Path(__file__).resolve().parents[2]
UTC_2024_03_01 = datetime(2024, 3, 1, 21, 30, tzinfo=UTC)


def make_claim(
    subject="ACME",
    attribute="capex_direction",
    value="up",
    period="current_quarter",
    source_doc_id="doc-1",
    extractor="rules-8k-v1",
    evidence="capex will rise",
    **overrides,
) -> Claim:
    fields = {
        "subject": subject,
        "attribute": attribute,
        "period": period,
        "value": value,
        "valid_from_utc": UTC_2024_03_01,
        "recorded_utc": UTC_2024_03_01,
        "source_doc_id": source_doc_id,
        "evidence_span": evidence,
        "extractor_version": extractor,
    }
    fields.update(overrides)
    return Claim(**fields)


def make_tension(claim_ids=("a", "b"), kind="value_conflict", **overrides) -> Tension:
    fields = {
        "kind": kind,
        "group_key": "ACME|capex_direction|unspecified",
        "claim_ids": list(claim_ids),
        "subject": "ACME",
        "opened_session": date(2024, 3, 4),
        "detector_version": "dream-detect-v1",
        "recorded_utc": UTC_2024_03_01,
        "details": {"z": 2.5, "note": "x"},
    }
    fields.update(overrides)
    return Tension(**fields)


def make_shift(**overrides) -> Shift:
    fields = {
        "kind": "event_rate",
        "subject": "ACME",
        "detected_session": date(2024, 3, 4),
        "statistic": 3.4,
        "threshold": 3.0,
        "window_sessions": 60,
        "detector_version": "dream-detect-v1",
        "recorded_utc": UTC_2024_03_01,
    }
    fields.update(overrides)
    return Shift(**fields)


def test_model_claim_needs_evidence():
    with pytest.raises(ValidationError):
        make_claim(extractor="llm-claims-v1", evidence=None)
    with pytest.raises(ValidationError):
        make_claim(extractor="llm-claims-v1", evidence="  ")
    assert make_claim(extractor="rules-8k-v1", evidence=None).evidence_span is None


def test_claim_rejects_naive_timestamps_and_unknown_fields():
    with pytest.raises(ValidationError):
        make_claim(valid_from_utc=datetime(2024, 3, 1, 21, 30))
    with pytest.raises(ValidationError):
        make_claim(recorded_utc="2024-03-01T22:00:00")
    with pytest.raises(ValidationError):
        make_claim(surprise="x")


def test_claim_normalises_offsets_to_utc():
    claim = make_claim(valid_from_utc="2024-03-01T16:30:00-05:00")
    assert claim.valid_from_utc == UTC_2024_03_01
    assert claim.valid_from_utc.utcoffset().total_seconds() == 0


def test_claim_rejects_an_expiry_before_its_start():
    with pytest.raises(ValidationError):
        make_claim(valid_to_utc=datetime(2024, 2, 1, tzinfo=UTC))


def test_claim_id_is_filled_and_a_wrong_one_rejected():
    claim = make_claim()
    expected = claim_id_for(
        "ACME", Attribute.capex_direction, Period.current_quarter, Value.up, "doc-1"
    )
    assert claim.claim_id == expected
    assert len(expected) == 64
    assert claim_id_for("ACME", "capex_direction", "current_quarter", "up", "doc-1") == expected
    with pytest.raises(ValidationError):
        make_claim(claim_id="not-the-hash")
    assert make_claim(claim_id=expected).claim_id == expected
    assert make_claim(source_doc_id="doc-2").claim_id != expected


def test_claim_rejects_values_outside_the_vocabulary():
    with pytest.raises(ValidationError):
        make_claim(attribute="mood")
    with pytest.raises(ValidationError):
        make_claim(value="sideways")
    with pytest.raises(ValidationError):
        make_claim(period="someday")


def test_tension_id_is_the_group_and_not_its_members_and_is_validated():
    # D72: the id is the kind and the group key; the members are lifecycle
    group = "ACME|capex_direction|unspecified"
    tension = make_tension(claim_ids=("b", "a"))
    assert tension.tension_id == tension_id_for(TensionKind.value_conflict, group)
    assert tension.tension_id == make_tension(claim_ids=("a", "b", "c")).tension_id
    assert (
        tension.tension_id != make_tension(group_key="ACME|capex_direction|fiscal_year").tension_id
    )
    assert tension.tension_id != make_tension(kind="stale_reliance").tension_id
    with pytest.raises(ValidationError):
        make_tension(tension_id="wrong")


def test_tension_claim_count_rules_and_status_consistency():
    with pytest.raises(ValidationError):
        make_tension(claim_ids=())
    with pytest.raises(ValidationError):
        make_tension(claim_ids=("a",), kind="value_conflict")
    assert make_tension(claim_ids=("a",), kind="stale_reliance").status == "open"
    with pytest.raises(ValidationError):
        make_tension(status="closed")
    with pytest.raises(ValidationError):
        make_tension(status="open", closed_session=date(2024, 3, 5))
    closed = make_tension(status="closed", closed_session=date(2024, 3, 5))
    assert closed.closed_session == date(2024, 3, 5)


def test_shift_id_and_validation():
    shift = make_shift()
    assert shift.shift_id == shift_id_for(ShiftKind.event_rate, "ACME")
    with pytest.raises(ValidationError):
        make_shift(shift_id="wrong")
    with pytest.raises(ValidationError):
        make_shift(window_sessions=0)
    with pytest.raises(ValidationError):
        make_shift(recorded_utc=datetime(2024, 3, 1))


def test_resolution_status_and_decision_time_agree():
    base = {
        "resolution_id": "r1",
        "tension_id": "t1",
        "hypothesis": "one_is_stale",
        "evidence_span": None,
        "proposer_version": "dream-resolve-v1",
        "proposed_utc": UTC_2024_03_01,
        "recorded_utc": UTC_2024_03_01,
    }
    assert Resolution(**base, status="proposed").decided_utc is None
    assert Resolution(**base, status="proposed").claim_id is None  # D47: optional
    named = Resolution(**base, status="proposed", claim_id="c1")
    assert named.claim_id == "c1"
    with pytest.raises(ValidationError):  # D61: recorded_utc is required
        Resolution(**{k: v for k, v in base.items() if k != "recorded_utc"}, status="proposed")
    with pytest.raises(ValidationError):
        Resolution(**{**base, "recorded_utc": datetime(2024, 3, 1)}, status="proposed")
    with pytest.raises(ValidationError):
        Resolution(**base, status="accepted")
    with pytest.raises(ValidationError):
        Resolution(**base, status="proposed", decided_utc=UTC_2024_03_01)
    accepted = Resolution(**base, status="accepted", decided_utc=UTC_2024_03_01)
    assert accepted.hypothesis is Hypothesis.one_is_stale
    with pytest.raises(ValidationError):
        Resolution(**{**base, "hypothesis": "bad"}, status="proposed")


def test_reference_schema_file_matches_the_code():
    path = ROOT / "references" / "dream" / "claim_schema_v1.json"
    assert json.loads(path.read_text()) == claim_json_schema()


def test_vocabulary_file_matches_the_enums():
    vocabulary = json.loads((ROOT / "references" / "dream" / "claim_vocabulary.json").read_text())
    assert vocabulary["attributes"] == [a.value for a in Attribute]
    assert vocabulary["periods"] == [p.value for p in Period]
    assert vocabulary["values"] == [v.value for v in Value]
    for pair in vocabulary["conflicting_value_pairs"]:
        assert len(pair) == 2
        assert {Value(x) for x in pair}
