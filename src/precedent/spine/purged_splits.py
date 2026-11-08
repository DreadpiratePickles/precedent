"""Walk-forward train/valid/test segments with a purge gap, in Qlib's segment format."""
import pandas as pd

Segment = tuple[pd.Timestamp, pd.Timestamp]


def purged_walk_forward_segments(
    sessions: pd.DatetimeIndex,
    train_len: int,
    valid_len: int,
    test_len: int,
    step: int,
    label_horizon: int,
    embargo: int = 0,
    expanding: bool = True,
) -> list[dict[str, Segment]]:
    """Return folds of inclusive (start, end) segments separated by
    label_horizon + embargo sessions."""
    lengths = {"train_len": train_len, "valid_len": valid_len, "test_len": test_len, "step": step}
    too_small = {name: value for name, value in lengths.items() if value < 1}
    if too_small:
        raise ValueError(f"lengths must be >= 1, got {too_small}")
    if label_horizon < 1:
        raise ValueError(
            f"label_horizon must be >= 1 (a label always looks forward), got {label_horizon}"
        )
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be strictly increasing")

    gap = label_horizon + embargo
    folds: list[dict[str, Segment]] = []
    fold_index = 0
    while True:
        test_start = train_len + gap + valid_len + gap + fold_index * step
        test_end = test_start + test_len - 1
        if test_end > len(sessions) - 1:
            break
        valid_end = test_start - gap - 1
        valid_start = valid_end - valid_len + 1
        train_end = valid_start - gap - 1
        train_start = 0 if expanding else train_end - train_len + 1
        folds.append(
            {
                "train": (sessions[train_start], sessions[train_end]),
                "valid": (sessions[valid_start], sessions[valid_end]),
                "test": (sessions[test_start], sessions[test_end]),
            }
        )
        fold_index += 1
    if not folds:
        raise ValueError(
            f"no fold fits: the first test window ends at position {test_end} "
            f"but there are {len(sessions)} sessions"
        )
    return folds
