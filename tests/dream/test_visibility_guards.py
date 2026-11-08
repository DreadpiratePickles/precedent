"""D65: each bi-temporal guard is tested alone, so removing it fails a test (verifier F-2, and
the T5 re-review minor on tension replay)."""

from datetime import date

from dream_synthetic import at, gdelt_record

from precedent.dream.claim_schema import Claim, Shift, Tension
from precedent.dream.claim_store import ClaimStore, claims_asof
from precedent.dream.report import _visible_at
from precedent.dream.report_render import build_sections

CUTOFF = at("2024-06-05T20:00:00Z")


def claim(doc: str, valid_from: str, recorded: str, **fields) -> Claim:
    return Claim(
        subject="ACME",
        attribute="capex_direction",
        period="current_quarter",
        value="up",
        valid_from_utc=at(valid_from),
        recorded_utc=at(recorded),
        source_doc_id=doc,
        evidence_span="Acme will raise capex",
        extractor_version="rules-test-v1",
        **fields,
    )


def test_a_claim_recorded_before_it_becomes_valid_is_invisible_until_then(tmp_path):
    # `recorded_utc` alone would show it at the cutoff; only the `valid_from_utc` guard hides it
    store = ClaimStore(tmp_path)
    store.append(
        [claim("early", valid_from="2024-06-10T14:00:00Z", recorded="2024-06-01T14:00:00Z")]
    )
    assert claims_asof(store, CUTOFF).empty
    assert list(claims_asof(store, at("2024-06-10T14:00:00Z"))["source_doc_id"]) == ["early"]
    assert list(claims_asof(store, at("2024-06-11T00:00:00Z"))["source_doc_id"]) == ["early"]


def test_a_record_available_after_the_cutoff_is_invisible_whatever_its_ingest_time():
    # D69: `available_utc` is the only gate, so a record ingested early but available late is out
    available_late = gdelt_record(1, "2024-06-05T21:00:00Z", ingested="2024-06-05T10:00:00Z")
    available_early = gdelt_record(2, "2024-06-05T19:00:00Z", ingested="2024-06-09T10:00:00Z")
    at_cutoff = gdelt_record(3, "2024-06-05T20:00:00Z")
    kept = _visible_at([available_late, available_early, at_cutoff], CUTOFF)
    assert [r.article_id for r in kept] == ["gdelt:2", "gdelt:3"]


def tension(**fields) -> Tension:
    base = dict(
        kind="value_conflict",
        group_key="ACME|capex_direction|current_quarter",
        claim_ids=["a", "b"],
        subject="ACME",
        opened_session=date(2024, 6, 4),
        detector_version="test-1",
        details={"attribute": "capex_direction", "period": "current_quarter", "values": "down,up"},
        recorded_utc=at("2024-06-04T20:00:00Z"),
    )
    return Tension(**{**base, **fields})


def sections_at(tensions, session=date(2024, 6, 5), claims=()):
    return build_sections(
        list(claims), tensions, [], [], [], session, CUTOFF, at("2024-06-04T20:00:00Z")
    )


def test_a_tension_recorded_after_the_cutoff_is_not_opened_by_that_session():
    later = tension(opened_session=date(2024, 6, 5), recorded_utc=at("2024-06-09T20:00:00Z"))
    assert sections_at([later])["opened_tension"] == []


def test_a_closing_recorded_after_the_cutoff_leaves_the_tension_open_as_of_it():
    opened = tension()
    closed_later = tension(
        status="closed", closed_session=date(2024, 6, 5), recorded_utc=at("2024-06-09T20:00:00Z")
    )
    visible = sections_at([opened, closed_later])
    assert visible["closed_tension"] == []
    # without the guard the later record wins and the session would list the closing
    closed_in_time = tension(
        status="closed", closed_session=date(2024, 6, 5), recorded_utc=at("2024-06-05T20:00:00Z")
    )
    assert len(sections_at([opened, closed_in_time])["closed_tension"]) == 1


def test_a_shift_recorded_after_the_cutoff_is_not_listed():
    shift = Shift(
        kind="event_rate",
        subject="ACME",
        detected_session=date(2024, 6, 5),
        statistic=4.0,
        threshold=3.0,
        window_sessions=60,
        detector_version="test-1",
        recorded_utc=at("2024-06-09T20:00:00Z"),
    )
    shown = build_sections([], [], [shift], [], [], date(2024, 6, 5), CUTOFF, None)
    assert shown["shift"] == []
