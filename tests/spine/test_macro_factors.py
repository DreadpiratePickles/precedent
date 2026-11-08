"""Tests for macro factors built from world events."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fakes import assert_truncation_invariant, make_event_record

from precedent.spine.abnormal_returns import estimate_betas
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


LONG_SESSIONS = pd.bdate_range("2024-01-02", periods=80)
EST_WINDOW = (-30, -6)


def synthetic_returns() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    market = pd.Series(rng.normal(0, 0.01, len(LONG_SESSIONS)), index=LONG_SESSIONS)
    assets = pd.DataFrame(
        {
            "AAA": 1.5 * market + rng.normal(0, 0.002, len(market)),
            "BBB": -0.5 * market + rng.normal(0, 0.002, len(market)),
        }
    )
    return assets, market


def beta_frame(assets, market, sessions) -> pd.DataFrame:
    return estimate_betas(assets, market, sessions, est_window=EST_WINDOW, min_obs=10)


def test_a_beta_frame_supplies_the_beta_at_each_session():
    betas = pd.DataFrame(
        {"AAA": np.arange(1.0, 11.0), "BBB": np.arange(11.0, 21.0)}, index=SESSIONS_10
    )
    panel = build_macro_factor_panel(
        [event_on("2024-01-02", 2)], betas, SESSIONS_10, half_life_sessions=1.0
    )
    for position, session in enumerate(SESSIONS_10[:3]):
        decay = 1.0 * 0.5**position  # direction 1.0 (step 2)
        assert panel.loc[(session, "AAA"), "macro_x_beta"] == pytest.approx(decay * (1 + position))
        assert panel.loc[(session, "BBB"), "macro_x_beta"] == pytest.approx(decay * (11 + position))


def test_a_nan_or_missing_beta_gives_nan_and_is_never_filled():
    betas = pd.DataFrame({"AAA": [np.nan, 2.0, 3.0]}, index=SESSIONS_10[1:4])
    panel = build_macro_factor_panel([event_on("2024-01-02", 2)], betas, SESSIONS_10)
    x_beta = panel["macro_x_beta"].droplevel("instrument")
    assert x_beta.isna().loc[[SESSIONS_10[0], SESSIONS_10[1], *SESSIONS_10[4:]]].all()
    assert x_beta.loc[SESSIONS_10[2]] == pytest.approx(2.0 * 0.5 ** (2 / 3))
    assert panel["macro_direction_decay"].notna().all()


def test_a_beta_change_at_session_t_leaves_the_rows_before_t_unchanged():
    base = pd.DataFrame({"AAA": 1.0, "BBB": -1.0}, index=SESSIONS_10)
    changed = base.copy()
    changed.iloc[5:] = 9.0
    events = [event_on("2024-01-02", 2), event_on("2024-01-05", -1), event_on("2024-01-10", 2)]
    before = build_macro_factor_panel(events, base, SESSIONS_10)
    after = build_macro_factor_panel(events, changed, SESSIONS_10)
    cut = SESSIONS_10[5]
    early = before.index.get_level_values("datetime") < cut
    pd.testing.assert_frame_equal(before[early], after[early])
    assert not before[~early].equals(after[~early])


def test_macro_panel_with_estimated_betas_passes_truncation_invariance():
    assets, market = synthetic_returns()
    records = [event_on("2024-02-01", 2), event_on("2024-02-12", -1), event_on("2024-03-04", 2)]
    full = build_macro_factor_panel(
        records, beta_frame(assets, market, LONG_SESSIONS), LONG_SESSIONS
    )
    assert full["macro_x_beta"].notna().any() and full["macro_x_beta"].isna().any()
    for position in (20, 35, 50, 65):
        cut_sessions = LONG_SESSIONS[: position + 1]
        kept = [r for r in records if pd.Timestamp(r.effective_session) <= cut_sessions[-1]]
        betas = beta_frame(assets.loc[cut_sessions], market.loc[cut_sessions], cut_sessions)
        truncated = build_macro_factor_panel(kept, betas, cut_sessions)
        pd.testing.assert_frame_equal(
            full[full.index.get_level_values("datetime") <= cut_sessions[-1]], truncated
        )


def test_unsorted_or_duplicated_sessions_are_rejected():
    shuffled = SESSIONS_10[::-1]
    with pytest.raises(ValueError, match="sorted"):
        build_macro_factor_panel([], BETAS, shuffled)
    with pytest.raises(ValueError, match="unique"):
        build_macro_factor_panel([], BETAS, SESSIONS_10.append(SESSIONS_10[:1]).sort_values())
