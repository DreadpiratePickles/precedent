"""Tests for purged walk-forward segments in Qlib's segment format."""
import pandas as pd
import pytest

from purged_splits import purged_walk_forward_segments

SESSIONS = pd.bdate_range("2024-01-01", periods=30)


def positions(segment: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[int, int]:
    return SESSIONS.get_loc(segment[0]), SESSIONS.get_loc(segment[1])


def test_hand_worked_boundaries() -> None:
    # gap = 2 + 1 = 3. Fold 0: test_start = 10 + 3 + 4 + 3 = 20, valid = [13, 16], train = [0, 9].
    folds = purged_walk_forward_segments(
        SESSIONS, train_len=10, valid_len=4, test_len=3, step=3, label_horizon=2, embargo=1
    )
    got = [{name: positions(segment) for name, segment in fold.items()} for fold in folds]
    assert got == [
        {"train": (0, 9), "valid": (13, 16), "test": (20, 22)},
        {"train": (0, 12), "valid": (16, 19), "test": (23, 25)},
        {"train": (0, 15), "valid": (19, 22), "test": (26, 28)},
    ]  # a fourth fold would need test = [29, 31] and is dropped


def test_segments_are_timestamp_pairs_with_inclusive_ends() -> None:
    folds = purged_walk_forward_segments(SESSIONS, 10, 4, 3, 3, label_horizon=2)
    assert set(folds[0]) == {"train", "valid", "test"}
    start, end = folds[0]["test"]
    assert isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp)
    assert (SESSIONS[(SESSIONS >= start) & (SESSIONS <= end)]).size == 3


@pytest.mark.parametrize("label_horizon", [1, 2, 5])
@pytest.mark.parametrize("embargo", [0, 2])
def test_no_label_window_crosses_into_the_next_segment(label_horizon: int, embargo: int) -> None:
    sessions = pd.bdate_range("2020-01-01", periods=400)
    folds = purged_walk_forward_segments(sessions, 100, 30, 20, 20, label_horizon=label_horizon, embargo=embargo)
    assert len(folds) > 3
    for fold in folds:
        train_end = sessions.get_loc(fold["train"][1])
        valid_start, valid_end = (sessions.get_loc(t) for t in fold["valid"])
        test_start = sessions.get_loc(fold["test"][0])
        assert train_end + label_horizon < valid_start
        assert valid_end + label_horizon < test_start
        assert train_end + label_horizon + embargo < valid_start


def test_expanding_keeps_the_first_session_and_rolling_keeps_the_train_length() -> None:
    expanding = purged_walk_forward_segments(SESSIONS, 10, 4, 3, 3, label_horizon=2, expanding=True)
    rolling = purged_walk_forward_segments(SESSIONS, 10, 4, 3, 3, label_horizon=2, expanding=False)
    assert all(fold["train"][0] == SESSIONS[0] for fold in expanding)
    for fold in rolling:
        start, end = positions(fold["train"])
        assert end - start + 1 == 10
    assert [f["test"] for f in expanding] == [f["test"] for f in rolling]


def test_consecutive_test_windows_advance_by_step() -> None:
    folds = purged_walk_forward_segments(SESSIONS, 10, 4, 3, step=2, label_horizon=1)
    starts = [positions(fold["test"])[0] for fold in folds]
    assert len(starts) > 2 and {b - a for a, b in zip(starts, starts[1:])} == {2}


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_len": 0},
        {"valid_len": 0},
        {"test_len": 0},
        {"step": 0},
        {"label_horizon": 0},
        {"embargo": -1},
        {"train_len": 100},  # no fold fits in 30 sessions
    ],
)
def test_invalid_arguments_and_no_fitting_fold_raise(overrides: dict) -> None:
    arguments = {"train_len": 10, "valid_len": 4, "test_len": 3, "step": 3, "label_horizon": 2} | overrides
    with pytest.raises(ValueError):
        purged_walk_forward_segments(SESSIONS, **arguments)


def test_unsorted_sessions_raise() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        purged_walk_forward_segments(SESSIONS[::-1], 10, 4, 3, 3, label_horizon=2)
