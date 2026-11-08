import numpy as np
import pandas as pd
import pytest
from engine_synthetic import CUTOFFS, SESSIONS
from fakes import assert_truncation_invariant

from precedent.engine.keys import key_from_flags
from precedent.engine.precedent_factors import PANEL_COLUMNS, build_precedent_factor_panel
from precedent.libraries.common import validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS

_ONE_HOUR = pd.Timedelta(hours=1)
KEY = key_from_flags("central_bank", "central_bank", "high", False, "US")
OTHER_KEY = key_from_flags("earnings", "company", "mid", False, "US")
INSTRUMENTS = ("SPY", "QQQ")
FIRST_POSITION = 30
SPACING = 12
N_EVENTS = 12
HORIZON = 5
CUT_SESSIONS = [SESSIONS[100], SESSIONS[130], SESSIONS[170]]
DENSE_SPACING = 4  # below HORIZON + 1: responses complete between later events' query and session
DENSE_N_EVENTS = 24
DENSE_CUT_SESSIONS = [SESSIONS[55], SESSIONS[70], SESSIONS[85], SESSIONS[110]]


def event_record(event_id, key, position, *, story=None, is_first=True):
    return {
        "event_id": event_id,
        "key": key.as_string(),
        "event_type": key.event_type,
        "country_group": key.country_group,
        "available_utc": CUTOFFS.iloc[position] - _ONE_HOUR,
        "effective_session": SESSIONS[position].date(),
        "effective_position": position,
        "primary_actor": "actor",
        "linked_countries": [],
        "story_cluster_id": story or event_id,
        "is_first_report": is_first,
        "one_line": f"{event_id} line",
        "source": "synthetic",
        "source_ref": event_id,
    }


def events_of(records):
    return pd.DataFrame(records, columns=list(EVENTS_TABLE_COLUMNS))


def asset_rows(event_id, position, values, horizon=HORIZON):
    """One asset row per (instrument, value); complete session is position + 1 + horizon."""
    return [
        {
            "event_id": event_id,
            "outcome_type": "asset",
            "outcome_key": instrument,
            "horizon": horizon,
            "value": value,
            "is_binary": False,
            "effective_session": SESSIONS[position],
            "response_complete_session": SESSIONS[position + 1 + horizon],
            "source_ref": event_id,
        }
        for instrument, value in values.items()
    ]


def rows_of(records):
    return validate_library_rows(pd.DataFrame(records))


def placebo_of(records):
    """Placebo rows: (outcome_key, horizon, value, complete_position) tuples."""
    return pd.DataFrame(
        {
            "outcome_key": [r[0] for r in records],
            "horizon": [r[1] for r in records],
            "value": [float(r[2]) for r in records],
            "response_complete_session": [SESSIONS[r[3]] for r in records],
        }
    )


@pytest.fixture
def panel_inputs():
    rng = np.random.default_rng(11)
    events, rows = [], []
    for number in range(N_EVENTS):
        position = FIRST_POSITION + SPACING * number
        event_id = f"e{number:02d}"
        events.append(event_record(event_id, KEY, position))
        rows += asset_rows(
            event_id, position, {name: rng.normal(0.0, 0.02) for name in INSTRUMENTS}
        )
    return {
        "events": events_of(events),
        "rows": rows_of(rows),
        "session_cutoffs_utc": CUTOFFS,
        "sessions": SESSIONS,
    }


def truncate_inputs(inputs, cut):
    effective = pd.to_datetime(inputs["events"]["effective_session"])
    cut_inputs = {
        **inputs,
        "events": inputs["events"].loc[effective <= cut].reset_index(drop=True),
        "rows": inputs["rows"]
        .loc[inputs["rows"]["response_complete_session"] <= cut]
        .reset_index(drop=True),
    }
    placebo = inputs.get("placebo_asset")
    if placebo is not None:
        done = placebo["response_complete_session"] <= cut
        cut_inputs["placebo_asset"] = placebo.loc[done].reset_index(drop=True)
    return cut_inputs


def dense_inputs(spacing=DENSE_SPACING, n_events=DENSE_N_EVENTS, with_placebo=True):
    """Events closer together than `HORIZON + 1` sessions: precedent responses complete between
    a later event's query time and its session, so a bypassed time guard changes the panel."""
    rng = np.random.default_rng(23)
    events, rows = [], []
    for number in range(n_events):
        position = FIRST_POSITION + spacing * number
        event_id = f"d{number:02d}"
        events.append(event_record(event_id, KEY, position))
        rows += asset_rows(
            event_id, position, {name: rng.normal(0.0, 0.02) for name in INSTRUMENTS}
        )
    inputs = {
        "events": events_of(events),
        "rows": rows_of(rows),
        "session_cutoffs_utc": CUTOFFS,
        "sessions": SESSIONS,
    }
    if with_placebo:
        records = [
            (name, HORIZON, rng.normal(0.0, 0.01), 20 + 3 * k)
            for k in range(60)
            for name in INSTRUMENTS
        ]
        inputs["placebo_asset"] = placebo_of(records)
    return inputs


def test_precedent_panel_passes_truncation_invariance(panel_inputs):
    for cut in CUT_SESSIONS:
        full = build_precedent_factor_panel(**panel_inputs).loc[:cut]
        truncated = build_precedent_factor_panel(**truncate_inputs(panel_inputs, cut)).loc[:cut]
        assert not full.empty
        pd.testing.assert_frame_equal(full, truncated)


def test_truncation_invariance_with_responses_completing_inside_the_event_gap():
    """Spacing 4 < horizon + 1: a bypassed `response_complete_session` guard changes the panel."""
    inputs = dense_inputs(with_placebo=False)
    for cut in DENSE_CUT_SESSIONS:
        full = build_precedent_factor_panel(**inputs).loc[:cut]
        truncated = build_precedent_factor_panel(**truncate_inputs(inputs, cut)).loc[:cut]
        assert not full.empty
        pd.testing.assert_frame_equal(full, truncated)


def test_truncation_invariance_with_the_placebo_input_included():
    inputs = dense_inputs()
    for cut in DENSE_CUT_SESSIONS:
        full = build_precedent_factor_panel(**inputs).loc[:cut]
        truncated = build_precedent_factor_panel(**truncate_inputs(inputs, cut)).loc[:cut]
        pd.testing.assert_frame_equal(full, truncated)
    assert (full["prec_placebo_z"] != 0.0).any()  # the last cut has summarised cells with placebo


def test_every_column_carries_the_prefix(panel_inputs):
    assert all(c.startswith("prec_") for c in build_precedent_factor_panel(**panel_inputs).columns)


def test_the_shared_truncation_helper_accepts_the_builder(panel_inputs):
    """`assert_truncation_invariant` cuts only the events; the time guard hides the rows."""
    events = panel_inputs["events"].assign(
        effective_session=lambda frame: pd.to_datetime(frame["effective_session"])
    )

    def builder(frame):
        return build_precedent_factor_panel(**{**panel_inputs, "events": frame})

    assert_truncation_invariant(builder, events, SESSIONS, [60, 100, 130, 170, 200])


def test_the_panel_has_the_declared_index_and_float_columns(panel_inputs):
    panel = build_precedent_factor_panel(**panel_inputs)
    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert list(PANEL_COLUMNS) == [
        "prec_car5_med",
        "prec_car5_iqr",
        "prec_neff",
        "prec_hit_rate",
        "prec_placebo_z",
    ]
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert str(panel.index.get_level_values("datetime").dtype) == "datetime64[ns]"
    assert set(panel.index.get_level_values("instrument")) == set(INSTRUMENTS)
    assert (panel.dtypes == "float64").all()
    assert panel.index.is_monotonic_increasing
    assert (panel["prec_placebo_z"] == 0.0).all()


def _last_cell(panel):
    position = FIRST_POSITION + (N_EVENTS - 1) * SPACING
    return panel.loc[(SESSIONS[position], "SPY")]


def test_placebo_z_is_the_median_less_the_placebo_mean_over_the_placebo_sd(panel_inputs):
    placebo = placebo_of(
        [("SPY", HORIZON, value, 100) for value in (0.00, 0.02, 0.04, 0.06)]
        # other horizon, other instrument: ignored for (SPY, 5)
        + [("SPY", 20, 9.0, 100), ("QQQ", HORIZON, 7.0, 100), ("QQQ", HORIZON, 8.0, 101)]
    )
    plain = build_precedent_factor_panel(**panel_inputs)
    panel = build_precedent_factor_panel(**panel_inputs, placebo_asset=placebo)
    mean, sd = 0.03, np.sqrt((2 * 0.03**2 + 2 * 0.01**2) / 3)  # sample sd, ddof=1
    cell, plain_cell = _last_cell(panel), _last_cell(plain)
    assert plain_cell["prec_placebo_z"] == 0.0
    assert cell["prec_placebo_z"] == pytest.approx((cell["prec_car5_med"] - mean) / sd)
    assert cell["prec_placebo_z"] != 0.0
    qqq = panel.loc[(SESSIONS[FIRST_POSITION + (N_EVENTS - 1) * SPACING], "QQQ")]
    q_mean, q_sd = 7.5, np.sqrt(0.5)
    assert qqq["prec_placebo_z"] == pytest.approx((qqq["prec_car5_med"] - q_mean) / q_sd)
    # nothing else moves
    others = [name for name in PANEL_COLUMNS if name != "prec_placebo_z"]
    pd.testing.assert_frame_equal(panel[others], plain[others])


def test_a_refused_cell_keeps_a_zero_placebo_z_even_with_placebo(panel_inputs):
    placebo = placebo_of([("SPY", HORIZON, v, 100) for v in (0.0, 0.02, 0.04)])
    panel = build_precedent_factor_panel(**panel_inputs, placebo_asset=placebo)
    assert panel.loc[(SESSIONS[FIRST_POSITION + 2 * SPACING], "SPY")]["prec_placebo_z"] == 0.0


@pytest.mark.parametrize(
    "records",
    [
        [],
        [("SPY", HORIZON, 0.01, 100)],  # fewer than 2 values
        [("SPY", HORIZON, 0.02, 100), ("SPY", HORIZON, 0.02, 101)],  # sd 0
        [("SPY", 20, 0.01, 100), ("SPY", 20, 0.03, 100)],  # wrong horizon only
    ],
)
def test_placebo_z_is_zero_without_a_usable_placebo(panel_inputs, records):
    panel = build_precedent_factor_panel(**panel_inputs, placebo_asset=placebo_of(records))
    assert (panel["prec_placebo_z"] == 0.0).all()


def test_a_placebo_row_completing_after_the_event_is_ignored(panel_inputs):
    """Rows completing after an event's query time must not move that event's cell."""
    last = FIRST_POSITION + (N_EVENTS - 1) * SPACING
    visible = [("SPY", HORIZON, v, 100) for v in (0.00, 0.02, 0.04, 0.06)]
    late = [("SPY", HORIZON, v, last) for v in (5.0, -5.0)]  # cutoff of `last` is after the query
    just_after = [("SPY", HORIZON, 9.0, last + 3)]
    base = build_precedent_factor_panel(**panel_inputs, placebo_asset=placebo_of(visible))
    guarded = build_precedent_factor_panel(
        **panel_inputs, placebo_asset=placebo_of(visible + late + just_after)
    )
    pd.testing.assert_frame_equal(base, guarded)
    assert (base["prec_placebo_z"] != 0.0).any()


def test_a_placebo_row_completing_before_the_event_counts(panel_inputs):
    last = FIRST_POSITION + (N_EVENTS - 1) * SPACING
    base = placebo_of([("SPY", HORIZON, v, 100) for v in (0.00, 0.02, 0.04, 0.06)])
    more = pd.concat([base, placebo_of([("SPY", HORIZON, 0.5, last - 1)])], ignore_index=True)
    a = build_precedent_factor_panel(**panel_inputs, placebo_asset=base)
    b = build_precedent_factor_panel(**panel_inputs, placebo_asset=more)
    cell = (SESSIONS[last], "SPY")
    assert a.loc[cell, "prec_placebo_z"] != b.loc[cell, "prec_placebo_z"]


def test_a_malformed_placebo_is_rejected(panel_inputs):
    with pytest.raises(ValueError, match="placebo_asset"):
        build_precedent_factor_panel(
            **panel_inputs,
            placebo_asset=placebo_of([("SPY", HORIZON, 0.1, 100)]).drop(columns=["value"]),
        )
    with pytest.raises(ValueError, match="placebo_asset"):
        build_precedent_factor_panel(
            **panel_inputs,
            placebo_asset=placebo_of([("SPY", HORIZON, float("nan"), 100)]),
        )


def test_empty_inputs_give_an_empty_panel_with_the_same_shape(panel_inputs):
    empty = build_precedent_factor_panel(
        panel_inputs["events"].iloc[0:0],
        panel_inputs["rows"].iloc[0:0],
        CUTOFFS,
        SESSIONS,
    )
    reference = build_precedent_factor_panel(**panel_inputs)
    assert empty.empty
    assert list(empty.columns) == list(PANEL_COLUMNS)
    assert list(empty.index.names) == ["datetime", "instrument"]
    assert (empty.dtypes == "float64").all()
    assert empty.index.dtypes.to_list() == reference.index.dtypes.to_list()


def test_a_refused_cell_carries_the_effective_size_and_zeros(panel_inputs):
    panel = build_precedent_factor_panel(**panel_inputs)
    session_of_third = SESSIONS[FIRST_POSITION + 2 * SPACING]
    third = panel.loc[(session_of_third, "SPY")]
    assert third["prec_neff"] == 2.0  # two precedents so far; below min_neff
    assert third["prec_car5_med"] == 0.0
    assert third["prec_car5_iqr"] == 0.0
    assert third["prec_hit_rate"] == 0.0
    assert third["prec_placebo_z"] == 0.0


def test_a_cell_with_enough_precedents_is_summarised(panel_inputs):
    panel = build_precedent_factor_panel(**panel_inputs)
    last = panel.loc[(SESSIONS[FIRST_POSITION + (N_EVENTS - 1) * SPACING], "SPY")]
    assert last["prec_neff"] >= 8.0
    assert last["prec_car5_iqr"] > 0.0
    assert 0.0 <= last["prec_hit_rate"] <= 1.0


def test_an_event_with_no_visible_precedent_has_no_cell(panel_inputs):
    panel = build_precedent_factor_panel(**panel_inputs)
    assert (SESSIONS[FIRST_POSITION], "SPY") not in panel.index
    assert (SESSIONS[FIRST_POSITION + SPACING], "SPY") in panel.index


def _overlap_inputs(second_position, ids=("a", "b")):
    """Ten old precedents, then two events (`ids`) that both cover session `second_position`."""
    events, rows = [], []
    for number in range(10):
        position = 20 + 7 * number
        events.append(event_record(f"p{number}", KEY, position))
        rows += asset_rows(f"p{number}", position, {"SPY": 0.01 * (number + 1)})
    # a precedent whose response completes after the first event's query but before the second's
    events.append(event_record("late", KEY, 140))
    rows += asset_rows("late", 140, {"SPY": 0.5})
    rows[-1]["response_complete_session"] = SESSIONS[second_position - 1]
    first, second = second_position - 2, second_position
    events.append(event_record(ids[0], KEY, first))
    events.append(event_record(ids[1], KEY, second))
    for event_id, position in zip(ids, (first, second), strict=True):
        rows += asset_rows(event_id, position, {"SPY": 0.0})
    return {
        "events": events_of(events),
        "rows": rows_of(rows),
        "session_cutoffs_utc": CUTOFFS,
        "sessions": SESSIONS,
    }


def test_the_most_recent_event_wins_an_overlapping_cell():
    inputs = _overlap_inputs(second_position=150)
    panel = build_precedent_factor_panel(**inputs, max_age_sessions=5)
    only_older = panel.loc[(SESSIONS[148], "SPY")]
    older = panel.loc[(SESSIONS[149], "SPY")]
    assert older.equals(only_older)  # still the older event's query
    newer_query = build_precedent_factor_panel(
        inputs["events"],
        inputs["rows"],
        CUTOFFS,
        SESSIONS,
        max_age_sessions=0,
    ).loc[(SESSIONS[150], "SPY")]
    # at 150 both events cover the cell; the later one (position 150) must win
    assert panel.loc[(SESSIONS[150], "SPY")].equals(newer_query)
    assert panel.loc[(SESSIONS[152], "SPY")].equals(newer_query)
    # the newer event also sees the late-completing precedent, so its summary differs
    assert not newer_query.equals(only_older)


def test_a_tie_on_position_goes_to_the_larger_event_id():
    inputs = _overlap_inputs(second_position=150)
    events = inputs["events"].copy()
    is_a = events["event_id"] == "a"
    # "a" ties with "b" on position but is queried earlier, so it sees fewer precedents
    events.loc[is_a, "effective_position"] = 150
    events.loc[is_a, "effective_session"] = SESSIONS[150].date()
    events.loc[is_a, "available_utc"] = CUTOFFS.iloc[149] - _ONE_HOUR

    def cell(frame):
        panel = build_precedent_factor_panel(
            frame, inputs["rows"], CUTOFFS, SESSIONS, max_age_sessions=0
        )
        return panel.loc[(SESSIONS[150], "SPY")]

    only_a, only_b = cell(events[~(events["event_id"] == "b")]), cell(events[~is_a])
    assert not only_a.equals(only_b)
    assert cell(events).equals(only_b)


def test_max_age_zero_keeps_only_the_event_session(panel_inputs):
    panel = build_precedent_factor_panel(**panel_inputs, max_age_sessions=0)
    sessions_in_panel = set(panel.index.get_level_values("datetime"))
    event_sessions = {SESSIONS[FIRST_POSITION + SPACING * number] for number in range(1, N_EVENTS)}
    assert sessions_in_panel == event_sessions


def test_a_cell_lives_for_max_age_sessions_after_the_event(panel_inputs):
    panel = build_precedent_factor_panel(**panel_inputs, max_age_sessions=3)
    position = FIRST_POSITION + SPACING
    present = [(SESSIONS[position + age], "SPY") in panel.index for age in range(6)]
    assert present == [True, True, True, True, False, False]


def test_a_non_first_report_creates_no_cell(panel_inputs):
    events = panel_inputs["events"].copy()
    events.loc[events["event_id"] == "e05", "is_first_report"] = False
    panel = build_precedent_factor_panel(**{**panel_inputs, "events": events})
    assert (SESSIONS[FIRST_POSITION + 5 * SPACING], "SPY") not in panel.index


def test_other_outcome_types_and_horizons_are_ignored(panel_inputs):
    other = asset_rows("e11", FIRST_POSITION + 11 * SPACING, {"SPY": 5.0}, horizon=20)
    rows = pd.concat([panel_inputs["rows"], rows_of(other)], ignore_index=True)
    baseline = build_precedent_factor_panel(**panel_inputs)
    assert build_precedent_factor_panel(**{**panel_inputs, "rows": rows}).equals(baseline)


def test_an_event_near_the_end_of_the_calendar_is_cut_not_extrapolated():
    last = len(SESSIONS) - 2
    events, rows = [], []
    for number in range(9):
        position = last - 60 + 5 * number
        events.append(event_record(f"p{number}", KEY, position))
        rows += asset_rows(f"p{number}", position, {"SPY": 0.01})
    events.append(event_record("tail", KEY, last))
    panel = build_precedent_factor_panel(
        events_of(events), rows_of(rows), CUTOFFS, SESSIONS, max_age_sessions=5
    )
    assert panel.index.get_level_values("datetime").max() == SESSIONS[-1]


def test_a_different_key_is_not_a_precedent():
    events, rows = [], []
    for number in range(10):
        position = 20 + 7 * number
        events.append(event_record(f"p{number}", OTHER_KEY, position))
        rows += asset_rows(f"p{number}", position, {"SPY": 0.01})
    events.append(event_record("q", KEY, 120))
    panel = build_precedent_factor_panel(events_of(events), rows_of(rows), CUTOFFS, SESSIONS)
    assert (SESSIONS[120], "SPY") not in panel.index


def test_an_event_position_outside_the_calendar_is_rejected(panel_inputs):
    events = panel_inputs["events"].copy()
    events.loc[0, "effective_position"] = len(SESSIONS)
    with pytest.raises(ValueError, match="effective_position"):
        build_precedent_factor_panel(**{**panel_inputs, "events": events})


def test_empty_inputs_still_check_the_columns(panel_inputs):
    with pytest.raises(ValueError, match="missing columns"):
        build_precedent_factor_panel(
            panel_inputs["events"].iloc[0:0].drop(columns=["key"]),
            panel_inputs["rows"],
            CUTOFFS,
            SESSIONS,
        )
    with pytest.raises(ValueError, match="columns"):
        build_precedent_factor_panel(
            panel_inputs["events"],
            panel_inputs["rows"].iloc[0:0].drop(columns=["value"]),
            CUTOFFS,
            SESSIONS,
        )
    nulls = panel_inputs["events"].copy().astype({"is_first_report": "object"})
    nulls.loc[0, "is_first_report"] = None
    with pytest.raises(ValueError, match="is_first_report"):
        build_precedent_factor_panel(nulls, panel_inputs["rows"].iloc[0:0], CUTOFFS, SESSIONS)


def test_bad_arguments_are_rejected(panel_inputs):
    with pytest.raises(ValueError, match="horizon"):
        build_precedent_factor_panel(**panel_inputs, horizon=0)
    with pytest.raises(ValueError, match="max_age_sessions"):
        build_precedent_factor_panel(**panel_inputs, max_age_sessions=-1)
    with pytest.raises(ValueError, match="sessions"):
        build_precedent_factor_panel(**{**panel_inputs, "sessions": SESSIONS[::-1]})
    with pytest.raises(ValueError, match="missing columns"):
        build_precedent_factor_panel(
            **{**panel_inputs, "events": panel_inputs["events"].drop(columns=["key"])}
        )
