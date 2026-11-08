"""Tests for macro factors built from world events."""

from datetime import date

import pandas as pd
import pytest
from fakes import assert_truncation_invariant, make_event_record

from precedent.spine.event_record import EventRecord
from precedent.spine.macro_factors import build_macro_factor_panel

SESSIONS_10 = pd.bdate_range("2024-01-02", periods=10)
BETAS = pd.Series({"AAA": 1.5, "BBB": -0.5})


def event_on(day: str | None, macro_direction_step: int = 0) -> EventRecord:
    event = make_event_record().event.model_copy(
        update={"macro_direction_step": macro_direction_step}
    )
    effective = date.fromisoformat(day) if day is not None else None
    return make_event_record(effective_session=effective, event=event)


def test_macro_panel_decays_by_half_life():
    panel = build_macro_factor_panel(
        [event_on("2024-01-02", macro_direction_step=-2)],
        pd.Series({"AAA": 1.5}),
        SESSIONS_10,
        half_life_sessions=3.0,
    )
    assert panel.loc[("2024-01-02", "AAA"), "macro_direction_decay"] == pytest.approx(-1.0)
    assert panel.loc[("2024-01-05", "AAA"), "macro_direction_decay"] == pytest.approx(-0.5)
    assert panel.loc[("2024-01-02", "AAA"), "macro_x_beta"] == pytest.approx(-1.5)


def test_panel_covers_every_session_and_instrument_with_named_index():
    panel = build_macro_factor_panel([event_on("2024-01-04", 2)], BETAS, SESSIONS_10)
    assert panel.index.names == ["datetime", "instrument"]
    assert len(panel) == len(SESSIONS_10) * len(BETAS)
    assert list(panel.columns) == ["macro_direction_decay", "macro_x_beta"]
    assert panel.index.get_level_values("datetime").dtype == "datetime64[ns]"
    before = panel.loc[("2024-01-03", "AAA")]
    assert before["macro_direction_decay"] == 0.0
    assert before["macro_x_beta"] == 0.0


def test_events_sum_and_betas_scale_per_instrument():
    panel = build_macro_factor_panel(
        [event_on("2024-01-02", 2), event_on("2024-01-02", 1), event_on("2024-01-03", -2)],
        BETAS,
        SESSIONS_10,
        half_life_sessions=1.0,
    )
    decay = panel.loc[("2024-01-03", "BBB"), "macro_direction_decay"]
    assert decay == pytest.approx(1.5 * 0.5 + -1.0)
    assert panel.loc[("2024-01-03", "BBB"), "macro_x_beta"] == pytest.approx(decay * -0.5)
    assert panel.loc[("2024-01-03", "AAA"), "macro_x_beta"] == pytest.approx(decay * 1.5)


def test_events_without_or_outside_sessions_are_ignored():
    panel = build_macro_factor_panel(
        [event_on(None, 2), event_on("2023-12-01", 2), event_on("2024-02-01", 2)],
        BETAS,
        SESSIONS_10,
    )
    assert (panel["macro_direction_decay"] == 0.0).all()


@pytest.mark.parametrize("half_life", [0.0, -1.0])
def test_non_positive_half_life_is_rejected(half_life):
    with pytest.raises(ValueError, match="half_life_sessions"):
        build_macro_factor_panel([], BETAS, SESSIONS_10, half_life_sessions=half_life)


def test_macro_panel_passes_truncation_invariance():
    records = [
        event_on("2024-01-02", 2),
        event_on("2024-01-04", -1),
        event_on("2024-01-04", -2),
        event_on("2024-01-09", 1),
    ]
    frame = pd.DataFrame(
        {
            "record": records,
            "effective_session": [pd.Timestamp(r.effective_session) for r in records],
        }
    )

    def builder(events: pd.DataFrame) -> pd.DataFrame:
        return build_macro_factor_panel(list(events["record"]), BETAS, SESSIONS_10)

    assert_truncation_invariant(builder, frame, SESSIONS_10, [0, 2, 4, 6, 8])
