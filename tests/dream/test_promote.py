"""The owner's decisions on proposed resolutions: `dream promote` and `dream review`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from dream_synthetic import at, count_lines, hash_tree

from precedent.dream import promote
from precedent.dream.claim_schema import Claim, Resolution, Tension
from precedent.dream.claim_store import ClaimStore, ResolutionStore, TensionStore, claims_asof
from precedent.dream.promote import run_promote, run_review

OLDER_FROM = at("2024-03-01T21:30:00Z")
NEWER_FROM = at("2024-05-01T21:30:00Z")
PROPOSED = at("2024-06-04T12:00:00Z")
DECIDED = "2024-06-05T14:00:00+00:00"  # a Wednesday, before that session's cutoff
AFTER = at("2024-06-05T15:00:00Z")


@dataclass(frozen=True)
class World:
    """Stores under `root` holding one conflict between two claims and one proposed resolution."""

    root: Path
    claim_store: ClaimStore
    tension_store: TensionStore
    resolution_store: ResolutionStore
    older_claim_id: str
    newer_claim_id: str
    tension_id: str
    resolution_id: str


def tension_status(store: TensionStore, tension_id: str) -> str:
    return next(t.status for t in store.latest() if t.tension_id == tension_id)


def resolution_status(store: ResolutionStore, resolution_id: str) -> str:
    return next(r.status for r in store.latest() if r.resolution_id == resolution_id)


def make_world(root: Path, hypothesis: str = "one_is_stale", names: str | None = None) -> World:
    """`names` is None, "older" or "newer": which claim the resolution's `claim_id` points at."""

    def claim(value: str, valid_from: datetime, doc: str, span: str) -> Claim:
        return Claim(
            subject="ACME",
            attribute="capex_direction",
            period="current_quarter",
            value=value,
            valid_from_utc=valid_from,
            recorded_utc=valid_from,
            source_doc_id=doc,
            evidence_span=span,
            extractor_version="rules-8k-v1",
        )

    older = claim("up", OLDER_FROM, "doc-old", "capex will rise")
    newer = claim("down", NEWER_FROM, "doc-new", "capex will fall")
    tension = Tension(
        kind="value_conflict",
        group_key="ACME|capex_direction|current_quarter",
        claim_ids=[older.claim_id, newer.claim_id],
        subject="ACME",
        opened_session=date(2024, 5, 2),
        detector_version="dream-detect-v1",
        details={"attribute": "capex_direction", "period": "current_quarter"},
        recorded_utc=NEWER_FROM,
    )
    claim_id = {None: None, "older": older.claim_id, "newer": newer.claim_id}[names]
    resolution = Resolution(
        resolution_id="res-1",
        tension_id=tension.tension_id,
        hypothesis=hypothesis,
        evidence_span="the newer filing restates the plan",
        proposer_version="resolve-test-v1",
        status="proposed",
        proposed_utc=PROPOSED,
        recorded_utc=PROPOSED,
        claim_id=claim_id,
    )
    claim_store = ClaimStore(root / "claims")
    tension_store = TensionStore(root / "tensions")
    resolution_store = ResolutionStore(root / "resolutions")
    claim_store.append([older, newer])
    tension_store.append([tension])
    resolution_store.append([resolution])
    return World(
        root,
        claim_store,
        tension_store,
        resolution_store,
        older.claim_id,
        newer.claim_id,
        tension.tension_id,
        resolution.resolution_id,
    )


@pytest.fixture
def promote_fixture(tmp_path) -> World:
    return make_world(tmp_path)


def decide(world: World, *flags: str) -> int:
    return run_promote(["--resolution", world.resolution_id, "--out", str(world.root), *flags])


def visible_ids(world: World) -> set[str]:
    return set(claims_asof(world.claim_store, AFTER).claim_id)


def latest_claim(world: World, claim_id: str) -> Claim:
    return next(c for c in world.claim_store.latest() if c.claim_id == claim_id)


# --- the brief's three tests --------------------------------------------------------------------


def test_accepting_stale_supersedes_the_older_claim_and_closes_the_tension(promote_fixture):
    world = promote_fixture
    assert decide(world, "--accept", "--at", DECIDED) == 0
    ids = visible_ids(world)
    assert world.older_claim_id not in ids and world.newer_claim_id in ids
    assert tension_status(world.tension_store, world.tension_id) == "closed"
    older = latest_claim(world, world.older_claim_id)
    assert older.expiry_reason == f"resolution:{world.resolution_id}"
    assert older.superseded_by == world.newer_claim_id
    assert older.valid_to_utc == at(DECIDED)
    assert resolution_status(world.resolution_store, world.resolution_id) == "accepted"


def test_rejecting_changes_nothing_but_the_decision(promote_fixture):
    world = promote_fixture

    def snapshot():
        return (
            world.claim_store.read_all(),
            world.tension_store.read_all(),
            hash_tree(world.root / "claims"),
            hash_tree(world.root / "tensions"),
        )

    before = snapshot()
    assert decide(world, "--reject", "--at", DECIDED) == 0
    assert snapshot() == before
    assert resolution_status(world.resolution_store, world.resolution_id) == "rejected"
    assert [r.status for r in world.resolution_store.read_all()] == ["proposed", "rejected"]


def test_promote_is_idempotent(promote_fixture):
    world = promote_fixture
    assert decide(world, "--accept", "--at", DECIDED) == 0
    lines = count_lines(world.root)
    assert decide(world, "--accept", "--at", "2024-06-07T10:00:00+00:00") == 0
    assert count_lines(world.root) == lines


# --- the four hypotheses ------------------------------------------------------------------------


def test_one_is_stale_with_a_named_claim_supersedes_that_claim(tmp_path):
    world = make_world(tmp_path, "one_is_stale", names="newer")
    assert decide(world, "--accept", "--at", DECIDED) == 0
    assert visible_ids(world) == {world.older_claim_id}
    assert latest_claim(world, world.newer_claim_id).superseded_by == world.older_claim_id
    assert tension_status(world.tension_store, world.tension_id) == "closed"


def test_different_scope_closes_the_tension_and_leaves_the_claims(tmp_path):
    world = make_world(tmp_path, "different_scope")
    claims_before = hash_tree(world.root / "claims")
    assert decide(world, "--accept", "--at", DECIDED) == 0
    assert hash_tree(world.root / "claims") == claims_before
    assert tension_status(world.tension_store, world.tension_id) == "closed"


def test_genuine_conflict_is_acknowledged_and_stays_open(tmp_path):
    world = make_world(tmp_path, "genuine_conflict")
    claims_before = hash_tree(world.root / "claims")
    assert decide(world, "--accept", "--at", DECIDED) == 0
    assert hash_tree(world.root / "claims") == claims_before
    [tension] = world.tension_store.latest()
    assert tension.status == "acknowledged" and tension.closed_session is None
    assert resolution_status(world.resolution_store, world.resolution_id) == "accepted"


def test_extraction_error_expires_the_named_claim_and_closes_the_tension(tmp_path):
    world = make_world(tmp_path, "extraction_error", names="newer")
    assert decide(world, "--accept", "--at", DECIDED) == 0
    wrong = latest_claim(world, world.newer_claim_id)
    assert wrong.expiry_reason == f"extraction_error:{world.resolution_id}"
    assert wrong.superseded_by is None
    assert visible_ids(world) == {world.older_claim_id}
    assert tension_status(world.tension_store, world.tension_id) == "closed"


def test_extraction_error_without_a_named_claim_exits_one_and_writes_nothing(tmp_path, capsys):
    world = make_world(tmp_path, "extraction_error")
    before = hash_tree(world.root)
    assert decide(world, "--accept", "--at", DECIDED) == 1
    assert hash_tree(world.root) == before
    assert "claim_id" in capsys.readouterr().err


def test_a_named_claim_outside_the_tension_exits_one_and_writes_nothing(tmp_path, capsys):
    world = make_world(tmp_path, "one_is_stale", names="older")
    stored = world.resolution_store.latest()[0].model_copy(update={"claim_id": "not-in-tension"})
    (world.root / "resolutions" / "resolutions.jsonl").write_text(stored.model_dump_json() + "\n")
    before = hash_tree(world.root)
    assert decide(world, "--accept", "--at", DECIDED) == 1
    assert hash_tree(world.root) == before
    assert "not-in-tension" in capsys.readouterr().err


def test_the_default_older_claim_is_the_earliest_valid_from_with_ties_to_the_smallest_id(tmp_path):
    world = make_world(tmp_path, "one_is_stale")
    # Rebuild the newer claim as simultaneous with the older one: the smaller id is the older.
    tied = latest_claim(world, world.newer_claim_id).model_copy(
        update={"valid_from_utc": OLDER_FROM, "recorded_utc": OLDER_FROM}
    )
    path = world.root / "claims" / "claims.jsonl"
    older = latest_claim(world, world.older_claim_id)
    path.write_text(older.model_dump_json() + "\n" + tied.model_dump_json() + "\n")
    assert decide(world, "--accept", "--at", DECIDED) == 0
    smaller = min(world.older_claim_id, world.newer_claim_id)
    assert latest_claim(world, smaller).expiry_reason == f"resolution:{world.resolution_id}"


# --- decisions, ids and times -------------------------------------------------------------------


def test_a_different_second_decision_exits_one_and_writes_nothing(promote_fixture, capsys):
    world = promote_fixture
    assert decide(world, "--accept", "--at", DECIDED) == 0
    before = hash_tree(world.root)
    assert decide(world, "--reject", "--at", DECIDED) == 1
    assert hash_tree(world.root) == before
    assert "accepted" in capsys.readouterr().err


def test_an_unknown_resolution_id_exits_two(promote_fixture, capsys):
    before = hash_tree(promote_fixture.root)
    code = run_promote(
        ["--resolution", "nope", "--accept", "--out", str(promote_fixture.root), "--at", DECIDED]
    )
    assert code == 2
    assert hash_tree(promote_fixture.root) == before
    assert "nope" in capsys.readouterr().err


def test_a_naive_at_is_rejected_and_nothing_is_written(promote_fixture):
    before = hash_tree(promote_fixture.root)
    assert decide(promote_fixture, "--accept", "--at", "2024-06-05T14:00:00") == 2
    assert decide(promote_fixture, "--accept", "--at", "yesterday") == 2
    assert hash_tree(promote_fixture.root) == before


def test_accept_and_reject_are_required_and_exclusive(promote_fixture):
    assert decide(promote_fixture) == 2
    assert decide(promote_fixture, "--accept", "--reject") == 2


def test_an_at_with_an_offset_is_stored_in_utc(promote_fixture):
    assert decide(promote_fixture, "--accept", "--at", "2024-06-05T10:00:00-04:00") == 0
    [decision] = [r for r in promote_fixture.resolution_store.read_all() if r.decided_utc]
    assert decision.decided_utc == at("2024-06-05T14:00:00Z")
    assert decision.recorded_utc == decision.decided_utc  # D61


def test_without_at_the_decision_is_stamped_by_now(promote_fixture, monkeypatch):
    monkeypatch.setattr(promote, "_now", lambda: datetime(2024, 6, 5, 16, 0, tzinfo=UTC))
    assert decide(promote_fixture, "--accept") == 0
    assert promote_fixture.resolution_store.latest()[0].decided_utc == at("2024-06-05T16:00:00Z")


def test_a_decision_before_the_proposal_was_recorded_is_refused(promote_fixture):
    before = hash_tree(promote_fixture.root)
    assert decide(promote_fixture, "--accept", "--at", "2024-06-04T11:00:00+00:00") == 2
    assert hash_tree(promote_fixture.root) == before


def test_the_tension_closes_in_the_next_session_whose_cutoff_is_after_the_decision(tmp_path):
    friday_evening = "2024-06-07T23:30:00+00:00"  # after Friday's 20:00 UTC cutoff
    world = make_world(tmp_path / "late", "different_scope")
    assert decide(world, "--accept", "--at", friday_evening) == 0
    assert world.tension_store.latest()[0].closed_session == date(2024, 6, 10)
    world = make_world(tmp_path / "early", "different_scope")
    assert decide(world, "--accept", "--at", DECIDED) == 0
    assert world.tension_store.latest()[0].closed_session == date(2024, 6, 5)


def test_a_claim_that_already_expired_is_left_alone_and_the_decision_is_still_recorded(
    promote_fixture,
):
    world = promote_fixture
    world.claim_store.supersede(
        world.older_claim_id, None, at("2024-06-05T09:00:00Z"), "rule:annual_15_months"
    )
    assert decide(world, "--accept", "--at", DECIDED) == 0
    assert latest_claim(world, world.older_claim_id).expiry_reason == "rule:annual_15_months"
    assert resolution_status(world.resolution_store, world.resolution_id) == "accepted"
    assert tension_status(world.tension_store, world.tension_id) == "closed"


def test_a_tension_the_nightly_run_already_closed_is_left_alone(promote_fixture):
    world = promote_fixture
    world.tension_store.close(world.tension_id, date(2024, 6, 4), at("2024-06-04T20:00:00Z"))
    assert decide(world, "--accept", "--at", DECIDED) == 0
    assert world.tension_store.latest()[0].closed_session == date(2024, 6, 4)
    assert resolution_status(world.resolution_store, world.resolution_id) == "accepted"


# --- review -------------------------------------------------------------------------------------


def review(world: World, *flags: str) -> int:
    return run_review(["--out", str(world.root), *flags])


def test_review_shows_both_evidence_spans_side_by_side(promote_fixture, capsys):
    assert review(promote_fixture) == 0
    out = capsys.readouterr().out
    assert promote_fixture.resolution_id in out and "one_is_stale" in out
    side_by_side = [line for line in out.splitlines() if "capex will rise" in line]
    assert len(side_by_side) == 1 and "capex will fall" in side_by_side[0]
    assert side_by_side[0].index("capex will rise") < side_by_side[0].index("capex will fall")
    assert "doc-old" in out and "doc-new" in out
    assert "the newer filing restates the plan" in out
    assert f"precedent dream promote --resolution {promote_fixture.resolution_id} --accept" in out
    assert "not forecasts" in out


def test_review_is_read_only_and_lists_only_what_is_still_proposed(promote_fixture, capsys):
    before = hash_tree(promote_fixture.root)
    assert review(promote_fixture) == 0
    assert hash_tree(promote_fixture.root) == before
    capsys.readouterr()
    assert decide(promote_fixture, "--reject", "--at", DECIDED) == 0
    capsys.readouterr()
    assert review(promote_fixture) == 0
    out = capsys.readouterr().out
    assert "no proposed resolutions" in out and promote_fixture.resolution_id not in out


def test_review_since_keeps_resolutions_proposed_on_or_after_the_date(promote_fixture, capsys):
    assert review(promote_fixture, "--since", "2024-06-04") == 0
    assert promote_fixture.resolution_id in capsys.readouterr().out
    assert review(promote_fixture, "--since", "2024-06-05") == 0
    assert "no proposed resolutions" in capsys.readouterr().out


def test_review_with_a_bad_since_exits_two(promote_fixture):
    assert review(promote_fixture, "--since", "June 4") == 2


def test_review_of_an_empty_out_directory_says_so(tmp_path, capsys):
    assert run_review(["--out", str(tmp_path)]) == 0
    assert "no proposed resolutions" in capsys.readouterr().out


def test_review_on_a_malformed_store_line_is_exit_1_naming_the_file(tmp_path, capsys):
    """M4 (Opus final review / D76): a malformed store line is a store fault — exit 1 naming
    the file, not a traceback."""
    (tmp_path / "resolutions").mkdir(parents=True)
    (tmp_path / "resolutions" / "resolutions.jsonl").write_text("not json\n", encoding="utf-8")
    assert run_review(["--out", str(tmp_path)]) == 1
    assert "resolutions.jsonl" in capsys.readouterr().err
