"""D72: a tension is its group, not its members. A restated link or a third document joining a
conflict is the same tension; a recurrence after closing reopens it; each kind closes by its
own test."""

from datetime import date

import pytest
from dream_synthetic import (
    at,
    customer_link,
    gdelt_record,
    plant_capex_conflict,
    write_events,
    write_links,
)

from precedent.dream.claim_schema import Claim, TensionKind, claim_id_for, link_id_for
from precedent.dream.claim_store import ClaimStore, ConflictError, TensionStore, tensions_asof
from precedent.dream.detectors import detect_value_conflicts
from precedent.dream.report import run

SESSIONS = [
    "2024-06-03",
    "2024-06-04",
    "2024-06-05",
    "2024-06-06",
    "2024-06-07",
    "2024-06-10",
    "2024-06-11",
]


def dream_through(out, last):
    """Dream every session from 2024-06-03 to `last`, in order, as the nightly would."""
    for day in SESSIONS[: SESSIONS.index(last) + 1]:
        assert run(["--date", day, "--out", str(out)]) == 0


def tensions(out, kind=None):
    found = TensionStore(out / "tensions").latest()
    return [t for t in found if kind is None or t.kind.value == kind]


def markdown(out, day):
    return (out / "dreams" / f"dream_{day}.md").read_text(encoding="utf-8")


def link_claim_id(doc, source="ACME", receiver="BIGCO"):
    subject = link_id_for(source, receiver, "customer_of", doc)
    return claim_id_for(subject, "customer_relationship", "unspecified", "ongoing", doc)


@pytest.fixture
def restated_link(tmp_path, monkeypatch):
    """F-1: ACME is BIGCO's customer from the 2023 10-K on 06-03; the restating filing is 06-10."""
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    write_events(
        tmp_path / "data" / "events",
        [gdelt_record(100, "2024-05-31T15:00:00Z", instrument="BETA", event_type="product")],
    )
    write_links(
        tmp_path / "data" / "ripple" / "links",
        [
            customer_link(),
            customer_link(
                valid_from="2024-06-10T12:00:00Z",
                valid_to="2025-06-10T12:00:00Z",
                doc="10K-ACME-2024-RESTATED",
            ),
        ],
    )
    return tmp_path / "out"


def test_a_restated_link_is_one_open_tension_whose_members_move(restated_link):
    out = restated_link
    dream_through(out, "2024-06-11")
    [only] = tensions(out, "asymmetric_link")
    assert only.group_key == only.subject == "ACME|customer_of|BIGCO"
    assert only.status == "open" and only.opened_session == date(2024, 6, 3)
    assert only.claim_ids == [link_claim_id("10K-ACME-2024-RESTATED")]
    # the restatement is neither an opening nor a closing in the note
    for day in ("2024-06-10", "2024-06-11"):
        text = markdown(out, day)
        assert "## Opened tensions (0)" in text and "## Closed tensions (0)" in text
    # the first session's note, replayed, still cites the original document
    assert "10K-ACME-2023" in markdown(out, "2024-06-03")


def test_the_restating_run_appends_one_member_update_and_a_replay_appends_nothing(restated_link):
    out = restated_link
    dream_through(out, "2024-06-10")
    stored = TensionStore(out / "tensions").read_all()
    assert [t.claim_ids for t in stored] == [
        [link_claim_id("10K-ACME-2023")],
        [link_claim_id("10K-ACME-2024-RESTATED")],
    ]
    assert run(["--date", "2024-06-10", "--out", str(out)]) == 0
    assert len(TensionStore(out / "tensions").read_all()) == 2


def test_a_third_document_joining_a_conflict_is_the_same_tension(tmp_path, conflicted_stores):
    assert run(["--date", "2024-06-03", "--out", str(tmp_path)]) == 0
    [first] = tensions(tmp_path, "value_conflict")
    assert len(first.claim_ids) == 2
    third = Claim(
        subject="ACME",
        attribute="capex_direction",
        period="current_quarter",
        value="down",
        valid_from_utc=at("2024-06-04T14:00:00Z"),
        valid_to_utc=at("2024-06-06T20:00:00Z"),
        recorded_utc=at("2024-06-04T14:00:00Z"),
        source_doc_id="8K-ACME-C",
        evidence_span="Acme confirms the capex cut",
        extractor_version="rules-test-v1",
    )
    ClaimStore(tmp_path / "claims").append([third])
    assert run(["--date", "2024-06-04", "--out", str(tmp_path)]) == 0
    [only] = tensions(tmp_path, "value_conflict")
    assert only.tension_id == first.tension_id and len(only.claim_ids) == 3
    assert only.status == "open" and only.opened_session == date(2024, 6, 3)
    text = markdown(tmp_path, "2024-06-04")
    assert "## Opened tensions (0)" in text and "## Closed tensions (0)" in text


def test_a_conflict_closes_when_one_side_expires_and_reopens_when_it_recurs(
    tmp_path, populated_stores
):
    plant_capex_conflict(tmp_path, valid_to="2024-06-05T20:00:00Z")
    run(["--date", "2024-06-03", "--out", str(tmp_path)])
    run(["--date", "2024-06-04", "--out", str(tmp_path)])
    run(["--date", "2024-06-05", "--out", str(tmp_path)])  # both sides have ended
    [closed] = tensions(tmp_path, "value_conflict")
    assert closed.status == "closed" and closed.closed_session == date(2024, 6, 5)
    again = [
        Claim(
            subject="ACME",
            attribute="capex_direction",
            period="current_quarter",
            value=value,
            valid_from_utc=at("2024-06-06T14:00:00Z"),
            recorded_utc=at("2024-06-06T14:00:00Z"),
            source_doc_id=doc,
            evidence_span=span,
            extractor_version="rules-test-v1",
        )
        for value, doc, span in (
            ("up", "8K-ACME-D", "Acme raises capex again"),
            ("down", "8K-ACME-E", "Acme cuts capex again"),
        )
    ]
    ClaimStore(tmp_path / "claims").append(again)
    run(["--date", "2024-06-06", "--out", str(tmp_path)])
    [reopened] = tensions(tmp_path, "value_conflict")
    assert reopened.tension_id == closed.tension_id
    assert reopened.status == "open" and reopened.opened_session == date(2024, 6, 6)
    assert reopened.closed_session is None
    assert "## Opened tensions (1)" in markdown(tmp_path, "2024-06-06")
    # the history stays: opened, closed, reopened
    history = [
        t.status
        for t in TensionStore(tmp_path / "tensions").read_all()
        if t.kind is TensionKind.value_conflict
    ]
    assert history == ["open", "closed", "open"]
    # and the earlier notes are what they were as of their own cutoffs
    assert "## Closed tensions (1)" in markdown(tmp_path, "2024-06-05")


def test_an_asymmetric_tension_closes_when_the_reverse_supplier_link_appears(tmp_path, monkeypatch):
    from precedent.ripple.link_store import Link

    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    write_events(
        tmp_path / "data" / "events",
        [gdelt_record(100, "2024-05-31T15:00:00Z", instrument="BETA", event_type="product")],
    )
    back = Link(
        source_node="BIGCO",
        receiver_node="ACME",
        relation="supplier_of",
        weight=0.5,
        valid_from_utc=at("2024-06-05T12:00:00Z"),
        valid_to_utc=at("2025-06-05T12:00:00Z"),
        source_doc_id="10K-BIGCO",
        evidence_span="ACME is a supplier",
        extractor_version="rules-links-v1",
    )
    write_links(tmp_path / "data" / "ripple" / "links", [customer_link(), back])
    out = tmp_path / "out"
    for day in SESSIONS[:3]:
        run(["--date", day, "--out", str(out)])
    [tension] = tensions(out, "asymmetric_link")
    assert tension.status == "closed" and tension.closed_session == date(2024, 6, 5)


def test_tensions_asof_reads_the_last_record_recorded_by_the_cutoff(tmp_path, conflicted_stores):
    run(["--date", "2024-06-03", "--out", str(tmp_path)])
    run(["--date", "2024-06-06", "--out", str(tmp_path)])  # closes the conflict at the 06-06 cutoff
    store = TensionStore(tmp_path / "tensions")
    assert {
        t.status
        for t in tensions_asof(store, at("2024-06-05T20:00:00Z"))
        if t.kind is TensionKind.value_conflict
    } == {"open"}
    assert {
        t.status
        for t in tensions_asof(store, at("2024-06-06T20:00:00Z"))
        if t.kind is TensionKind.value_conflict
    } == {"closed"}


def test_store_restate_members_keeps_status_and_refuses_a_closed_tension(
    tmp_path, conflicted_stores
):
    run(["--date", "2024-06-03", "--out", str(tmp_path)])
    store = TensionStore(tmp_path / "tensions")
    [conflict] = [t for t in store.latest() if t.kind is TensionKind.value_conflict]
    updated = store.restate_members(
        conflict.tension_id, ["x", "y", "z"], at("2024-06-04T20:00:00Z")
    )
    assert updated.claim_ids == ["x", "y", "z"] and updated.status == "open"
    assert updated.opened_session == conflict.opened_session
    store.close(conflict.tension_id, date(2024, 6, 5), at("2024-06-05T20:00:00Z"))
    with pytest.raises(ConflictError):
        store.restate_members(conflict.tension_id, ["x"], at("2024-06-06T20:00:00Z"))
    with pytest.raises(ConflictError):  # only a closed tension reopens
        store.reopen(
            [t for t in store.latest() if t.kind is TensionKind.asymmetric_link][0].tension_id,
            date(2024, 6, 6),
            ["x"],
            at("2024-06-06T20:00:00Z"),
        )


def test_value_conflict_detection_ignores_event_direction_in_the_pipeline(
    tmp_path, populated_stores
):
    # the populated stores hold ACME earnings positive against labor negative (GDELT-style news)
    run(["--date", "2024-06-03", "--out", str(tmp_path)])
    assert tensions(tmp_path, "value_conflict") == []
    assert detect_value_conflicts.__doc__ and "event_direction" in detect_value_conflicts.__doc__
