"""D74: one exclusive lock per output root for dream, promote and review."""

import fcntl
from contextlib import contextmanager

import pytest
from dream_synthetic import hash_tree
from test_promote import make_world

from precedent.dream.locking import LOCK_FILE, DreamLockedError, exclusive_lock
from precedent.dream.promote import run_promote, run_review
from precedent.dream.report import run


@contextmanager
def held(out):
    """Another command holding the lock: a separate open file description on `<out>/.lock`."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / LOCK_FILE).open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_a_dream_while_the_lock_is_held_exits_one_and_writes_nothing(
    tmp_path, populated_stores, capsys
):
    out = tmp_path / "out"
    with held(out):
        assert run(["--date", "2024-06-03", "--out", str(out)]) == 1
    assert f"another dream command is running on {out}" in capsys.readouterr().err
    assert not (out / "dreams").exists() and not (out / "claims").exists()
    assert not (out / "manifest.json").exists()


def test_promote_and_review_while_the_lock_is_held_exit_one(tmp_path, capsys):
    world = make_world(tmp_path)
    before = hash_tree(tmp_path)
    with held(tmp_path):
        assert (
            run_promote(["--resolution", world.resolution_id, "--accept", "--out", str(tmp_path)])
            == 1
        )
        assert run_review(["--out", str(tmp_path)]) == 1
    assert capsys.readouterr().err.count("another dream command is running") == 2
    assert hash_tree(tmp_path) == before


def test_the_lock_is_released_after_a_command_and_after_a_failed_one(tmp_path, populated_stores):
    out = tmp_path / "out"
    assert run(["--date", "2024-06-01", "--out", str(out)]) == 2  # not a session: a usage error
    assert run(["--date", "2024-06-03", "--out", str(out)]) == 0
    assert run(["--date", "2024-06-03", "--out", str(out)]) == 0  # a replay takes it again
    with exclusive_lock(out):  # and nothing holds it now
        pass


def test_the_lock_refuses_a_second_holder_in_the_same_process(tmp_path):
    with exclusive_lock(tmp_path), pytest.raises(DreamLockedError), exclusive_lock(tmp_path):
        pass


def test_a_read_only_command_does_not_create_a_missing_output_root(tmp_path):
    missing = tmp_path / "nowhere"
    assert run_review(["--out", str(missing)]) == 0
    assert not missing.exists()
