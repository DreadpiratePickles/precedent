"""D70: a session is dreamed once, in order; a gap needs --allow-gap; a replay appends nothing;
the output root records what built it."""

import json

import pytest
from dream_synthetic import count_lines, gdelt_record, hash_tree, write_events

from precedent.dream.report import run

SESSIONS = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"]


def dream(out, day, *extra):
    return run(["--date", day, "--out", str(out), *extra])


def dream_files(out):
    return sorted(path.name for path in (out / "dreams").glob("dream_*"))


def test_the_first_run_writes_a_manifest_with_the_versions_that_built_the_stores(
    tmp_path, conflicted_stores
):
    assert dream(tmp_path, "2024-06-03") == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["last_session"] == "2024-06-03"
    assert manifest["parser_version"] == "v1" and manifest["story_window_sessions"] == 3
    assert len(manifest["claim_vocabulary_sha256"]) == len(manifest["detector_params_sha256"]) == 64
    assert dream(tmp_path, "2024-06-04") == 0
    assert json.loads((tmp_path / "manifest.json").read_text())["last_session"] == "2024-06-04"
    assert dream(tmp_path, "2024-06-03") == 0  # a replay does not move it back
    assert json.loads((tmp_path / "manifest.json").read_text())["last_session"] == "2024-06-04"


def test_a_session_before_the_last_one_with_no_dream_exits_two_and_changes_nothing(
    tmp_path, conflicted_stores, capsys
):
    assert dream(tmp_path, "2024-06-04", "--allow-gap") == 0
    before = hash_tree(tmp_path)
    assert dream(tmp_path, "2024-06-03") == 2
    error = capsys.readouterr().err
    assert "2024-06-04" in error and "before the last session" in error
    assert hash_tree(tmp_path) == before


def test_a_gap_exits_two_naming_the_last_session_and_changes_nothing(
    tmp_path, conflicted_stores, capsys
):
    assert dream(tmp_path, "2024-06-03") == 0
    before = hash_tree(tmp_path)
    assert dream(tmp_path, "2024-06-06") == 2
    error = capsys.readouterr().err
    assert "2024-06-03" in error and "skip 2 session" in error and "--allow-gap" in error
    assert hash_tree(tmp_path) == before and dream_files(tmp_path) == [
        "dream_2024-06-03.jsonl",
        "dream_2024-06-03.md",
    ]


def test_allow_gap_dreams_after_the_gap_and_says_how_many_sessions_were_skipped(
    tmp_path, conflicted_stores
):
    assert dream(tmp_path, "2024-06-03") == 0
    assert dream(tmp_path, "2024-06-06", "--allow-gap") == 0
    text = (tmp_path / "dreams" / "dream_2024-06-06.md").read_text()
    assert "gap: 2 sessions skipped" in text.splitlines()[0]
    header = json.loads(
        (tmp_path / "dreams" / "dream_2024-06-06.jsonl").read_text().splitlines()[0]
    )
    assert header["gap_sessions_skipped"] == 2
    # a replay of the session says the same
    first = (tmp_path / "dreams" / "dream_2024-06-06.md").read_bytes()
    assert dream(tmp_path, "2024-06-06") == 0
    assert (tmp_path / "dreams" / "dream_2024-06-06.md").read_bytes() == first
    # the ordinary next session says nothing about a gap
    assert "sessions skipped" not in (tmp_path / "dreams" / "dream_2024-06-03.md").read_text()


def test_catching_up_in_order_after_a_refused_gap_equals_an_every_session_run(
    tmp_path, conflicted_stores
):
    every, caught_up = tmp_path / "every", tmp_path / "caught_up"
    from dream_synthetic import plant_capex_conflict

    plant_capex_conflict(every)
    plant_capex_conflict(caught_up)
    for day in SESSIONS:
        assert dream(every, day) == 0
    assert dream(caught_up, SESSIONS[0]) == 0
    assert dream(caught_up, SESSIONS[3]) == 2  # the Mac was off: the last session is refused
    for day in SESSIONS[1:]:
        assert dream(caught_up, day) == 0
    assert hash_tree(every) == hash_tree(caught_up)


def test_a_replay_that_would_append_exits_one_and_writes_nothing(
    tmp_path, populated_stores, capsys
):
    out = tmp_path / "out"
    assert dream(out, "2024-06-03") == 0
    before = hash_tree(out)
    lines = count_lines(out)
    # a document that was available before the cutoff arrives in the event store afterwards
    write_events(
        populated_stores.event_root,
        [gdelt_record(777, "2024-06-03T10:00:00Z", event_type="legal_regulatory", direction=-1.0)],
    )
    assert dream(out, "2024-06-03") == 1
    error = capsys.readouterr().err
    assert "claims/claims.jsonl" in error and "fresh --out" in error
    assert hash_tree(out) == before and count_lines(out) == lines


@pytest.mark.parametrize(
    "key",
    [
        "parser_version",
        "claim_vocabulary_sha256",
        "detector_params_sha256",
        "story_window_sessions",
    ],
)
def test_a_store_root_built_by_other_versions_exits_two_naming_what_changed(
    tmp_path, populated_stores, capsys, key
):
    assert dream(tmp_path, "2024-06-03") == 0
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = "something else" if key != "story_window_sessions" else 9
    path.write_text(json.dumps(manifest))
    before = hash_tree(tmp_path)
    assert dream(tmp_path, "2024-06-04") == 2
    error = capsys.readouterr().err
    assert key in error and "fresh --out" in error
    assert hash_tree(tmp_path) == before


def test_a_malformed_manifest_exits_one(tmp_path, populated_stores, capsys):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "manifest.json").write_text("{not json")
    assert dream(tmp_path, "2024-06-03") == 1
    assert "manifest.json" in capsys.readouterr().err
