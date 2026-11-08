"""Tests for the Dream factor panel: prefixes, decay, bi-temporal reads and truncation invariance."""

import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from precedent.dream.claim_schema import Shift, Tension
from precedent.dream.claim_store import ShiftStore, TensionStore
from precedent.dream.dream_factors import PANEL_COLUMNS, build_dream_factor_panel
from precedent.spine.pit_alignment import build_session_cutoffs

SESSIONS = pd.bdate_range("2024-01-02", periods=40)
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))
HOUR = timedelta(hours=1)


def cutoff(position: int) -> datetime:
    return CUTOFFS.iloc[position].to_pydatetime()


def session_date(position: int) -> dt.date:
    return SESSIONS[position].date()


def make_tension(
    position: int,
    subject: str = "ACME",
    kind: str = "text_vs_price",
    claim_value: str | None = "positive",
    recorded: datetime | None = None,
    claim_id: str = "c1",
) -> Tension:
    details = {} if claim_value is None else {"claim_value": claim_value}
    return Tension(
        kind=kind,
        claim_ids=[claim_id, claim_id + "b"] if kind == "value_conflict" else [claim_id],
        subject=subject,
        opened_session=session_date(position),
        detector_version="test-1",
        details=details,
        recorded_utc=recorded or cutoff(position) - HOUR,
    )


def make_shift(
    position: int,
    subject: str = "ACME",
    kind: str = "event_rate",
    statistic: float = 2.0,
    recorded: datetime | None = None,
) -> Shift:
    return Shift(
        kind=kind,
        subject=subject,
        detected_session=session_date(position),
        statistic=statistic,
        threshold=1.0,
        window_sessions=10,
        detector_version="test-1",
        recorded_utc=recorded or cutoff(position) - HOUR,
    )


def make_stores(
    root: Path, tensions: list[Tension] = (), shifts: list[Shift] = ()
) -> tuple[TensionStore, ShiftStore]:
    root.mkdir(parents=True, exist_ok=True)
    tension_store, shift_store = TensionStore(root), ShiftStore(root)
    tension_store.append(tensions)
    shift_store.append(shifts)
    return tension_store, shift_store


def panel_of(tension_store: TensionStore, shift_store: ShiftStore, **kwargs) -> pd.DataFrame:
    return build_dream_factor_panel(tension_store, shift_store, SESSIONS, CUTOFFS, **kwargs)


@pytest.fixture
def busy_stores(tmp_path: Path) -> tuple[TensionStore, ShiftStore]:
    """Several instruments, kinds and signs, a close and a re-fire recorded mid-history."""
    tensions = [
        make_tension(3, "ACME", "text_vs_price", "positive", claim_id="a"),
        make_tension(5, "ACME", "value_conflict", None, claim_id="b"),
        make_tension(8, "BOLT", "text_vs_price", "negative", claim_id="c"),
        make_tension(12, "BOLT", "stale_reliance", None, claim_id="d"),
        make_tension(14, "ACME", "text_vs_price", "negative", claim_id="e"),
    ]
    shifts = [
        make_shift(6, "ACME", "event_rate", 3.0),
        make_shift(9, "BOLT", "direction_drift", -1.5),
        make_shift(10, "CYAN", "event_rate", 1.2),
        make_shift(11, "CYAN", "link_churn", 4.0),
    ]
    tension_store, shift_store = make_stores(tmp_path / "busy", tensions, shifts)
    tension_store.close(tensions[0].tension_id, session_date(7), cutoff(7) - HOUR)
    tension_store.close(tensions[2].tension_id, session_date(16), cutoff(16) - HOUR)
    rate = next(s for s in shift_store.read_all() if s.kind == "event_rate" and s.subject == "ACME")
    shift_store.refire(rate.shift_id, 5.0, session_date(13), cutoff(13) - HOUR)
    return tension_store, shift_store


def truncate_inputs(
    tension_store: TensionStore, shift_store: ShiftStore, cut_position: int, root: Path
) -> tuple[TensionStore, ShiftStore]:
    """Stores holding only the lines recorded at or before the cut session's cutoff."""
    root.mkdir(parents=True, exist_ok=True)
    kept = []
    for store in (tension_store, shift_store):
        lines = [
            record.model_dump_json()
            for record in store.read_all()
            if record.recorded_utc <= cutoff(cut_position)
        ]
        copy = type(store)(root)
        copy.path.write_text("".join(line + "\n" for line in lines))
        kept.append(copy)
    return kept[0], kept[1]


@pytest.mark.parametrize("cut_position", [4, 7, 9, 12, 13, 15, 17, 25])
def test_dream_panel_passes_truncation_invariance(
    busy_stores: tuple[TensionStore, ShiftStore], tmp_path: Path, cut_position: int
) -> None:
    full = panel_of(*busy_stores).loc[: SESSIONS[cut_position]]
    cut_stores = truncate_inputs(*busy_stores, cut_position, tmp_path / f"cut{cut_position}")
    truncated = panel_of(*cut_stores).loc[: SESSIONS[cut_position]]
    assert len(full) > 0
    pd.testing.assert_frame_equal(full, truncated)


def test_every_column_carries_the_prefix(busy_stores: tuple[TensionStore, ShiftStore]) -> None:
    panel = panel_of(*busy_stores)
    assert all(c.startswith("dream_") for c in panel.columns)
    assert list(panel.columns) == PANEL_COLUMNS
    assert list(panel.columns) == [
        "dream_open_tensions",
        "dream_text_vs_price",
        "dream_rate_shift",
        "dream_direction_drift",
        "dream_days_since_tension",
    ]
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert (panel.dtypes == "float64").all()
    assert panel.index.get_level_values("datetime").dtype == "datetime64[ns]"


def test_a_tension_recorded_after_the_cutoff_is_not_in_that_session(tmp_path: Path) -> None:
    late = 10
    stores = make_stores(
        tmp_path / "late",
        [make_tension(late, "ACME", recorded=cutoff(late) + HOUR)],
        [make_shift(late - 2, "ACME")],  # keeps an ACME row alive on the late session
    )
    panel = panel_of(*stores)
    assert panel.loc[(SESSIONS[late], "ACME"), "dream_open_tensions"] == 0
    assert panel.loc[(SESSIONS[late], "ACME"), "dream_text_vs_price"] == 0
    assert panel.loc[(SESSIONS[late + 1], "ACME"), "dream_open_tensions"] == 1
    # age counts from the opened session, not from when the record became visible
    assert panel.loc[(SESSIONS[late + 1], "ACME"), "dream_days_since_tension"] == 1


def test_text_vs_price_decays_by_hand_computed_weights(tmp_path: Path) -> None:
    stores = make_stores(tmp_path / "decay", [make_tension(3, claim_value="positive")])
    panel = panel_of(*stores, max_age_sessions=1, half_life_sessions=1.0)
    expected = pd.DataFrame(
        {
            "dream_open_tensions": [1.0, 1.0],
            "dream_text_vs_price": [1.0, 0.5],
            "dream_rate_shift": [0.0, 0.0],
            "dream_direction_drift": [0.0, 0.0],
            "dream_days_since_tension": [0.0, 1.0],
        },
        index=pd.MultiIndex.from_arrays(
            [SESSIONS[[3, 4]].as_unit("ns"), pd.Index(["ACME", "ACME"])],
            names=["datetime", "instrument"],
        ),
    )
    pd.testing.assert_frame_equal(panel, expected)


@pytest.mark.parametrize(
    ("claim_value", "sign"), [("positive", 1.0), ("negative", -1.0), (None, 0.0), ("neutral", 0.0)]
)
def test_text_vs_price_sign_follows_the_claimed_direction(
    tmp_path: Path, claim_value: str | None, sign: float
) -> None:
    stores = make_stores(tmp_path / "sign", [make_tension(3, claim_value=claim_value)])
    panel = panel_of(*stores)
    assert panel.loc[(SESSIONS[3], "ACME"), "dream_text_vs_price"] == sign
    assert panel.loc[(SESSIONS[3], "ACME"), "dream_open_tensions"] == 1


def test_only_text_vs_price_tensions_feed_the_signed_column(tmp_path: Path) -> None:
    stores = make_stores(tmp_path / "kind", [make_tension(3, kind="stale_reliance")])
    panel = panel_of(*stores)
    row = panel.loc[(SESSIONS[3], "ACME")]
    assert row["dream_text_vs_price"] == 0 and row["dream_open_tensions"] == 1


def test_a_closed_tension_leaves_the_open_count(tmp_path: Path) -> None:
    tension = make_tension(3)
    tension_store, shift_store = make_stores(tmp_path / "closed", [tension])
    tension_store.close(tension.tension_id, session_date(6), cutoff(6) - HOUR)
    panel = panel_of(tension_store, shift_store)
    assert panel.loc[(SESSIONS[5], "ACME"), "dream_open_tensions"] == 1
    assert panel.loc[(SESSIONS[6], "ACME"), "dream_open_tensions"] == 0
    # the tension still ages and still counts for days-since and for the signed column
    assert panel.loc[(SESSIONS[6], "ACME"), "dream_days_since_tension"] == 3


def test_a_close_recorded_after_the_cutoff_is_still_open_that_session(tmp_path: Path) -> None:
    tension = make_tension(3)
    tension_store, shift_store = make_stores(tmp_path / "late_close", [tension])
    tension_store.close(tension.tension_id, session_date(6), cutoff(6) + HOUR)
    panel = panel_of(tension_store, shift_store)
    assert panel.loc[(SESSIONS[6], "ACME"), "dream_open_tensions"] == 1
    assert panel.loc[(SESSIONS[7], "ACME"), "dream_open_tensions"] == 0


def test_days_since_tension_caps_at_max_age(tmp_path: Path) -> None:
    stores = make_stores(
        tmp_path / "cap",
        [make_tension(2, "ACME")],
        [make_shift(5, "ACME"), make_shift(5, "SHIFTONLY")],
    )
    panel = panel_of(*stores, max_age_sessions=3)
    assert [panel.loc[(SESSIONS[p], "ACME"), "dream_days_since_tension"] for p in range(2, 6)] == [
        0,
        1,
        2,
        3,
    ]
    # the tension (age 4) has left the window, a live shift keeps the row, days-since stays capped
    assert panel.loc[(SESSIONS[6], "ACME"), "dream_open_tensions"] == 0
    assert panel.loc[(SESSIONS[6], "ACME"), "dream_days_since_tension"] == 3
    # a row with no tension at all also reads max_age
    assert panel.loc[(SESSIONS[5], "SHIFTONLY"), "dream_days_since_tension"] == 3


def test_empty_stores_give_an_empty_frame_with_the_five_columns(tmp_path: Path) -> None:
    panel = panel_of(*make_stores(tmp_path / "empty"))
    assert panel.empty
    assert list(panel.columns) == PANEL_COLUMNS
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert (panel.dtypes == "float64").all()


def test_direction_drift_lands_in_its_own_column(tmp_path: Path) -> None:
    stores = make_stores(
        tmp_path / "drift",
        shifts=[
            make_shift(4, "ACME", "direction_drift", -1.5),
            make_shift(4, "BOLT", "event_rate"),
        ],
    )
    panel = panel_of(*stores, half_life_sessions=1.0)
    acme = panel.loc[(SESSIONS[4], "ACME")]
    assert acme["dream_direction_drift"] == -1.5 and acme["dream_rate_shift"] == 0
    assert panel.loc[(SESSIONS[5], "ACME"), "dream_direction_drift"] == pytest.approx(-0.75)
    bolt = panel.loc[(SESSIONS[4], "BOLT")]
    assert bolt["dream_rate_shift"] == 2.0 and bolt["dream_direction_drift"] == 0


def test_link_churn_shifts_do_not_create_rows(tmp_path: Path) -> None:
    stores = make_stores(tmp_path / "churn", shifts=[make_shift(4, "A-B", "link_churn")])
    assert panel_of(*stores).empty


def test_a_refired_shift_restarts_its_age_once_the_refire_is_recorded(tmp_path: Path) -> None:
    shift = make_shift(3, "ACME", statistic=2.0)
    tension_store, shift_store = make_stores(tmp_path / "refire", shifts=[shift])
    shift_store.refire(shift.shift_id, 4.0, session_date(8), cutoff(8) - HOUR)
    panel = panel_of(tension_store, shift_store, half_life_sessions=1.0)
    assert panel.loc[(SESSIONS[4], "ACME"), "dream_rate_shift"] == pytest.approx(1.0)
    assert panel.loc[(SESSIONS[7], "ACME"), "dream_rate_shift"] == pytest.approx(2.0 / 16)
    assert panel.loc[(SESSIONS[8], "ACME"), "dream_rate_shift"] == pytest.approx(4.0)
    assert panel.loc[(SESSIONS[9], "ACME"), "dream_rate_shift"] == pytest.approx(2.0)


def test_a_session_off_the_calendar_moves_to_the_next_session_and_one_past_the_end_drops(
    tmp_path: Path,
) -> None:
    saturday = dt.date(2024, 1, 6)
    on_weekend = Tension(
        kind="stale_reliance",
        claim_ids=["w"],
        subject="ACME",
        opened_session=saturday,
        detector_version="test-1",
        recorded_utc=cutoff(3) - HOUR,
    )
    after_end = Tension(
        kind="stale_reliance",
        claim_ids=["z"],
        subject="BOLT",
        opened_session=session_date(39) + timedelta(days=3),
        detector_version="test-1",
        recorded_utc=cutoff(3) - HOUR,
    )
    panel = panel_of(*make_stores(tmp_path / "edge", [on_weekend, after_end]), max_age_sessions=1)
    dates = panel.index.get_level_values("datetime")
    assert (dates.min(), dates.max()) == (SESSIONS[4], SESSIONS[5])  # Saturday -> Monday 01-08
    assert set(panel.index.get_level_values("instrument")) == {"ACME"}


def test_invalid_arguments_are_rejected(tmp_path: Path) -> None:
    stores = make_stores(tmp_path / "bad")
    with pytest.raises(ValueError, match="half_life_sessions"):
        panel_of(*stores, half_life_sessions=0.0)
    with pytest.raises(ValueError, match="max_age_sessions"):
        panel_of(*stores, max_age_sessions=-1)
    with pytest.raises(ValueError, match="sessions"):
        build_dream_factor_panel(*stores, SESSIONS[::-1], CUTOFFS)
    with pytest.raises(ValueError, match="sessions"):
        build_dream_factor_panel(*stores, SESSIONS.tz_localize("UTC"), CUTOFFS)
    with pytest.raises(ValueError, match="session_cutoffs_utc"):
        build_dream_factor_panel(*stores, SESSIONS, CUTOFFS.iloc[:-1])
    with pytest.raises(ValueError, match="session_cutoffs_utc"):
        build_dream_factor_panel(*stores, SESSIONS, CUTOFFS.dt.tz_localize(None))
