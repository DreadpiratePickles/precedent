import numpy as np
import pandas as pd
import pytest
from engine_synthetic import CUTOFFS, SESSIONS

from precedent.engine.evaluate import brier_score, pinball_loss, run, score_precedent_ladder
from precedent.engine.keys import key_from_flags
from precedent.libraries.common import validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS

_ONE_HOUR = pd.Timedelta(hours=1)
_HORIZON = 5
_OUTPUT_COLUMNS = [
    "rung",
    "outcome_type",
    "event_type",
    "quantile",
    "pinball_loss",
    "coverage_80",
    "brier",
    "n_scored",
    "n_refused",
]

KEY_A = key_from_flags("central_bank", "central_bank", "high", False, "US")
KEY_B = key_from_flags("armed_conflict", "government", "high", False, "EM")


def _events_table(placements):
    """Tuples (event_id, key, position) -> events table, every event a first report."""
    records = [
        {
            "event_id": event_id,
            "key": key.as_string(),
            "event_type": key.event_type,
            "country_group": key.country_group,
            "available_utc": CUTOFFS.iloc[position] - _ONE_HOUR,
            "effective_session": SESSIONS[position].date(),
            "effective_position": position,
            "primary_actor": "actor",
            "linked_countries": [],
            "story_cluster_id": event_id,
            "is_first_report": True,
            "one_line": f"{event_id} line",
            "source": "synthetic",
            "source_ref": event_id,
        }
        for event_id, key, position in placements
    ]
    return pd.DataFrame(records, columns=list(EVENTS_TABLE_COLUMNS))


def _library_rows(outcomes):
    """Tuples (event_id, position, outcome_type, outcome_key, value, horizon) -> rows."""
    records = [
        {
            "event_id": event_id,
            "outcome_type": outcome_type,
            "outcome_key": outcome_key,
            "horizon": horizon,
            "value": value,
            "is_binary": outcome_type != "asset",
            "effective_session": SESSIONS[position],
            "response_complete_session": SESSIONS[position + horizon],
            "source_ref": event_id,
        }
        for event_id, position, outcome_type, outcome_key, value, horizon in outcomes
    ]
    return validate_library_rows(pd.DataFrame(records))


def _ladder_inputs(placements, outcomes):
    return {
        "events": _events_table(placements),
        "rows": _library_rows(outcomes),
        "session_cutoffs_utc": CUTOFFS,
    }


def _alternating(count, spacing):
    """Events alternating A and B from position 10; returns placements."""
    return [(f"e{i:03d}", KEY_A if i % 2 == 0 else KEY_B, 10 + spacing * i) for i in range(count)]


@pytest.fixture
def separable_asset_events():
    rng = np.random.default_rng(0)
    placements = _alternating(60, 6)
    outcomes = [
        (
            event_id,
            position,
            "asset",
            "SPY",
            (0.02 if key is KEY_A else -0.02) + rng.normal(0.0, 0.001),
            _HORIZON,
        )
        for event_id, key, position in placements
    ]
    return _ladder_inputs(placements, outcomes)


@pytest.fixture
def separable_sequence_events():
    placements = _alternating(60, 6)
    outcomes = [
        (
            event_id,
            position,
            "event_sequence",
            "trade_or_sanctions/linked_country",
            1.0 if key is KEY_A else 0.0,
            _HORIZON,
        )
        for event_id, key, position in placements
    ]
    return _ladder_inputs(placements, outcomes)


@pytest.fixture
def stationary_asset_events():
    rng = np.random.default_rng(0)
    placements = [(f"s{i:03d}", KEY_A, 10 + 3 * i) for i in range(120)]
    outcomes = [
        (event_id, position, "asset", "SPY", rng.normal(0.0, 0.02), _HORIZON)
        for event_id, _, position in placements
    ]
    return _ladder_inputs(placements, outcomes)


def test_loss_hand_values():
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)
    assert brier_score(np.array([1.0, 0.0]), np.array([0.8, 0.4])) == pytest.approx(0.10)


def test_typed_rung_beats_untyped_when_types_differ(separable_asset_events):
    s = (
        score_precedent_ladder(**separable_asset_events, outcome_type="asset")
        .query("quantile == 0.5")
        .groupby("rung")["pinball_loss"]
        .mean()
    )
    assert s["B1"] < s["B0"]


def test_typed_rung_beats_untyped_on_binary_outcomes(separable_sequence_events):
    s = (
        score_precedent_ladder(**separable_sequence_events, outcome_type="event_sequence")
        .groupby("rung")["brier"]
        .mean()
    )
    assert s["B1"] < s["B0"]


def test_coverage_is_near_nominal_on_stationary_data(stationary_asset_events):
    c = (
        score_precedent_ladder(**stationary_asset_events, outcome_type="asset")
        .query("rung == 'B1'")["coverage_80"]
        .mean()
    )
    assert 0.7 <= c <= 0.9


def test_evaluate_command_exits_2_without_a_library(tmp_path, capsys):
    assert run(["--data-dir", str(tmp_path)]) == 2 and "precedent build" in capsys.readouterr().err


def test_rungs_agree_when_every_event_has_the_same_key(stationary_asset_events):
    table = score_precedent_ladder(**stationary_asset_events, outcome_type="asset")
    b0 = table.query("rung == 'B0'").drop(columns="rung").reset_index(drop=True)
    b1 = table.query("rung == 'B1'").drop(columns="rung").reset_index(drop=True)
    pd.testing.assert_frame_equal(b0, b1)


def test_event_without_earlier_precedents_is_counted_as_refused():
    placements = [("only", KEY_A, 10)]
    outcomes = [("only", 10, "asset", "SPY", 0.01, _HORIZON)]
    table = score_precedent_ladder(**_ladder_inputs(placements, outcomes), outcome_type="asset")
    assert set(table["n_scored"]) == {0}
    assert set(table["n_refused"]) == {1}
    assert table["pinball_loss"].isna().all() and table["coverage_80"].isna().all()


def test_output_has_exactly_the_nine_columns_with_nan_where_inapplicable(
    separable_asset_events, separable_sequence_events
):
    continuous = score_precedent_ladder(**separable_asset_events, outcome_type="asset")
    binary = score_precedent_ladder(**separable_sequence_events, outcome_type="event_sequence")
    assert list(continuous.columns) == _OUTPUT_COLUMNS
    assert list(binary.columns) == _OUTPUT_COLUMNS
    assert continuous["brier"].isna().all()
    assert binary["quantile"].isna().all()
    assert binary["pinball_loss"].isna().all() and binary["coverage_80"].isna().all()
    assert set(continuous["quantile"]) == {0.1, 0.5, 0.9}
    assert len(continuous) == len({"B0", "B1"}) * len({"central_bank", "armed_conflict"}) * 3
    assert len(binary) == len({"B0", "B1"}) * len({"central_bank", "armed_conflict"})


def test_market_resolution_rows_are_scored_at_their_own_horizon():
    placements = [(f"m{i:03d}", KEY_A, 10 + 6 * i) for i in range(30)]
    outcomes = [
        (event_id, position, "market_resolution", "market_key", float(i % 2), 3 + i % 3)
        for i, (event_id, _, position) in enumerate(placements)
    ]
    table = score_precedent_ladder(
        **_ladder_inputs(placements, outcomes), outcome_type="market_resolution", horizon=5
    )
    assert table["n_scored"].sum() > 0
    assert table["brier"].notna().all()


def test_unknown_rung_and_unsupported_quantile_are_rejected(separable_asset_events):
    with pytest.raises(ValueError, match="rung"):
        score_precedent_ladder(**separable_asset_events, outcome_type="asset", rungs=("B9",))
    with pytest.raises(ValueError, match="quantile"):
        score_precedent_ladder(**separable_asset_events, outcome_type="asset", quantiles=(0.3,))


def _rare_key_inputs():
    """40 early key-A events, then 3 key-B events: B1 refuses the B events, B0 would not."""
    placements = [(f"a{i:03d}", KEY_A, 10 + 4 * i) for i in range(40)]
    placements += [(f"b{i:03d}", KEY_B, 180 + 4 * i) for i in range(3)]
    outcomes = [
        (event_id, position, "event_sequence", "trade_or_sanctions/linked_country", 1.0, _HORIZON)
        for event_id, _, position in placements
    ]
    return _ladder_inputs(placements, outcomes)


def test_rungs_are_scored_on_paired_units_only():
    """P-46 (I1): a unit refused by any compared rung is scored by none, so B1 - B0 is paired."""
    table = score_precedent_ladder(**_rare_key_inputs(), outcome_type="event_sequence")
    scored = table.pivot(index="event_type", columns="rung", values="n_scored")
    assert (scored["B0"] == scored["B1"]).all()
    assert scored.loc["armed_conflict"].tolist() == [0, 0]


def test_refusals_are_reported_per_rung_even_for_units_dropped_from_the_pairing():
    """P-46 (I1): n_refused stays each rung's own count; B0 refuses none of the B events."""
    table = score_precedent_ladder(**_rare_key_inputs(), outcome_type="event_sequence")
    refused = table.pivot(index="event_type", columns="rung", values="n_refused")
    assert refused.loc["armed_conflict", "B1"] == 3 and refused.loc["armed_conflict", "B0"] == 0
