"""Research: leak checks (shifted panel, cutoff audit, lag curve) and the planted-lead control."""

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("qlib")

from dataclasses import dataclass  # noqa: E402

import pandas as pd  # noqa: E402
from fixtures.synthetic_market import N_SESSIONS, START_DAY, write_planted_market  # noqa: E402
from qlib.config import C  # noqa: E402
from qlib.data.dataset.loader import DataLoader, QlibDataLoader  # noqa: E402

from precedent.spine.ablation import fit_fold_predictions  # noqa: E402
from precedent.spine.leak_checks import cutoff_audit, lag_curve, shift_panel  # noqa: E402
from precedent.spine.purged_splits import purged_walk_forward_segments  # noqa: E402
from precedent.spine.qlib_dataset import DEFAULT_LABEL_EXPR, make_alpha_handler  # noqa: E402
from precedent.spine.qlib_provider import init_qlib  # noqa: E402
from precedent.spine.rank_ic import daily_rank_ic  # noqa: E402

# Qlib 0.9.7 still passes `inplace=` to pandas, which pandas 3 deprecates. The warning comes from
# third-party code we do not control; it is filtered for this module only.
pytestmark = pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame:Warning")

SESSIONS = pd.bdate_range(START_DAY, periods=N_SESSIONS)
FOLDS = purged_walk_forward_segments(
    SESSIONS, train_len=100, valid_len=30, test_len=60, step=60, label_horizon=2
)
LGB_PARAMS = {
    "n_estimators": 40,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 20,
    "n_jobs": 1,
    "random_state": 7,
    "verbose": -1,
}


@dataclass(frozen=True)
class SyntheticMarket:
    sessions: pd.DatetimeIndex
    panel: pd.DataFrame
    price_loader: DataLoader


@pytest.fixture
def synthetic_market(tmp_path, monkeypatch) -> SyntheticMarket:
    monkeypatch.chdir(tmp_path)
    planted = write_planted_market(tmp_path / "provider", seed=11)
    init_qlib(tmp_path / "provider", tmp_path / "work")
    # Qlib loads features through a worker pool; one kernel loads in-process, which is faster here.
    monkeypatch.setitem(C, "kernels", 1)
    loader = QlibDataLoader(
        config={
            "feature": (["$close/$open", "Ref($close, 1)/$open"], ["close_over_open", "prev_close"]),
            "label": ([DEFAULT_LABEL_EXPR], ["LABEL0"]),
        }
    )
    return SyntheticMarket(sessions=planted.sessions, panel=planted.panel, price_loader=loader)


def fit_one_fold(market: SyntheticMarket, panel: pd.DataFrame) -> pd.DataFrame:
    fold = FOLDS[0]
    handler, _ = make_alpha_handler(
        market.price_loader, {"news_": panel}, fold["train"][0], fold["test"][1]
    )
    return fit_fold_predictions(handler, fold, LGB_PARAMS)


def mean_rank_ic(predictions: pd.DataFrame) -> float:
    return float(daily_rank_ic(predictions["pred"], predictions["label"]).mean())


def test_positive_control_detects_a_planted_lead(synthetic_market):
    # the label at t is driven by the news factor dated t+1, so a correctly aligned panel has no signal
    aligned = mean_rank_ic(fit_one_fold(synthetic_market, synthetic_market.panel))
    leaked = mean_rank_ic(
        fit_one_fold(
            synthetic_market, shift_panel(synthetic_market.panel, synthetic_market.sessions, -1)
        )
    )
    assert abs(aligned) < 0.05 and leaked > 0.2, f"aligned={aligned:.4f} leaked={leaked:.4f}"


def small_panel() -> pd.DataFrame:
    sessions = pd.bdate_range("2024-01-02", periods=5)
    index = pd.MultiIndex.from_tuples(
        [(sessions[0], "AAA"), (sessions[2], "AAA"), (sessions[4], "BBB")],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"news_x": [1.0, 2.0, 3.0]}, index=index)


SMALL_SESSIONS = pd.bdate_range("2024-01-02", periods=5)


def test_shift_panel_positive_is_later_and_negative_is_earlier():
    later = shift_panel(small_panel(), SMALL_SESSIONS, 1)
    earlier = shift_panel(small_panel(), SMALL_SESSIONS, -1)
    assert list(later.index) == [(SMALL_SESSIONS[1], "AAA"), (SMALL_SESSIONS[3], "AAA")]
    assert later["news_x"].tolist() == [1.0, 2.0]  # the row that fell off the calendar is dropped
    assert list(earlier.index) == [
        (SMALL_SESSIONS[1], "AAA"),
        (SMALL_SESSIONS[3], "BBB"),
    ]
    assert earlier["news_x"].tolist() == [2.0, 3.0]
    assert list(earlier.index.names) == ["datetime", "instrument"]


def test_shift_panel_by_zero_is_the_identity():
    pd.testing.assert_frame_equal(shift_panel(small_panel(), SMALL_SESSIONS, 0), small_panel())


def test_shift_panel_rejects_a_row_outside_the_calendar():
    with pytest.raises(ValueError, match="not in sessions"):
        shift_panel(small_panel(), SMALL_SESSIONS[1:], 1)


def test_lag_curve_shifts_the_panel_by_each_lag():
    panel = small_panel()

    def first_position(shifted: pd.DataFrame) -> float:
        return float(SMALL_SESSIONS.get_loc(shifted.index.get_level_values("datetime").min()))

    curve = lag_curve(first_position, panel, SMALL_SESSIONS, lags=(1, 2))
    assert curve.to_dict("list") == {"lag_sessions": [1, 2], "mean_rank_ic": [1.0, 2.0]}


def test_lag_curve_default_lags_are_one_two_and_five():
    curve = lag_curve(lambda shifted: 0.0, small_panel(), SMALL_SESSIONS)
    assert curve["lag_sessions"].tolist() == [1, 2, 5]


CUTOFFS = pd.Series(
    pd.to_datetime(["2024-01-02 21:00", "2024-01-03 21:00", "2024-01-04 21:00"], utc=True),
    index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="session_date"),
    name="cutoff_utc",
)


def audit_events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_session": pd.to_datetime([day for day, _ in rows]),
            "available_utc": pd.to_datetime([when for _, when in rows], utc=True),
        }
    )


def test_cutoff_audit_passes_events_assigned_to_the_first_session_they_could_make():
    events = audit_events(
        [
            ("2024-01-02", "2024-01-01 08:00"),  # first session: nothing earlier to exceed
            ("2024-01-03", "2024-01-03 15:00"),
            ("2024-01-03", "2024-01-03 21:00"),  # exactly at the cutoff still makes the session
        ]
    )
    assert cutoff_audit(events, CUTOFFS).empty


def test_cutoff_audit_flags_events_assigned_too_early_or_too_late():
    events = audit_events(
        [
            ("2024-01-03", "2024-01-04 15:00"),  # after the 01-03 cutoff: used too early
            ("2024-01-03", "2024-01-02 15:00"),  # made the 01-02 cutoff: assigned a session late
            ("2024-01-03", "2024-01-02 21:00"),  # exactly at the 01-02 cutoff: belongs to 01-02
            ("2024-01-04", "2024-01-04 15:00"),  # fine
        ]
    )
    violations = cutoff_audit(events, CUTOFFS)
    assert violations.index.tolist() == [0, 1, 2]
    assert violations["cutoff_utc"].iloc[0] == CUTOFFS.loc["2024-01-03"]
    assert violations["previous_cutoff_utc"].iloc[0] == CUTOFFS.loc["2024-01-02"]


def test_cutoff_audit_flags_a_session_outside_the_calendar():
    events = audit_events([("2024-01-09", "2024-01-04 15:00")])
    assert cutoff_audit(events, CUTOFFS).index.tolist() == [0]


def test_cutoff_audit_requires_aware_available_time():
    events = audit_events([("2024-01-03", "2024-01-03 15:00")])
    events["available_utc"] = events["available_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        cutoff_audit(events, CUTOFFS)
