"""P-43: the card header and the REFUSED decision come from the guarded rows shown."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from engine_synthetic import CUTOFFS, SESSIONS  # noqa: E402
from test_retrieval import (  # noqa: E402
    KEY_ANY_DIRECTION,
    _sequence_row,
    events_table,
    library_rows,
)

from precedent.engine import card_build as ask  # noqa: E402
from precedent.engine.card import render_text  # noqa: E402
from precedent.engine.params import load_engine_params  # noqa: E402
from precedent.engine.retrieval import PrecedentQuery  # noqa: E402
from precedent.libraries.common import LIBRARY_COLUMNS  # noqa: E402

QUERY_POSITION = 150
QUERY = PrecedentQuery(KEY_ANY_DIRECTION, CUTOFFS.iloc[QUERY_POSITION] - pd.Timedelta(hours=1))


def bundle_of(events, rows):
    placebo = pd.DataFrame(
        columns=["outcome_key", "horizon", "response_complete_session", "value"]
    )
    return SimpleNamespace(
        events=events,
        rows=rows,
        placebo_asset=placebo,
        markets=pd.DataFrame(),
        sessions=SESSIONS,
        session_cutoffs_utc=CUTOFFS,
        manifest=SimpleNamespace(sources=[]),
    )


def seven_finished_plus_one_pending():
    """V-2 probe: 7 independent finished events (positions 100..136) and one in the query's own
    session whose outcome window has not closed."""
    placed = [(f"f{i}", 100 + 6 * i) for i in range(7)] + [("pending", QUERY_POSITION - 1)]
    events = events_table([(eid, KEY_ANY_DIRECTION, pos) for eid, pos in placed])
    rows = library_rows([_sequence_row(eid, pos, pos + 5) for eid, pos in placed])
    assert list(rows.columns) == list(LIBRARY_COLUMNS)
    return events, rows


def card_for(events, rows, horizon=5, outcome_type="event_sequence"):
    return ask._precedent_card(
        QUERY, "q", None, horizon, outcome_type, bundle_of(events, rows), load_engine_params()
    )


def test_header_counts_only_completed_precedents_and_refuses_with_the_rows():
    events, rows = seven_finished_plus_one_pending()
    card = card_for(events, rows)
    assert card.n == 7 and card.n_eff == 7.0  # was n=8 independent=8, not refused
    assert card.refused is True  # 7 < min_neff 8, the same verdict the row would give
    assert card.latest == SESSIONS[136].date()


def test_header_agrees_with_the_row_when_not_refused():
    events = events_table([(f"g{i}", KEY_ANY_DIRECTION, 60 + 6 * i) for i in range(10)])
    rows = library_rows([_sequence_row(f"g{i}", 60 + 6 * i, 60 + 6 * i + 5) for i in range(10)])
    card = card_for(events, rows)
    assert card.refused is False
    assert card.rows and card.rows[0].n == card.n == 10
    assert card.rows[0].n_eff == card.n_eff


def test_a_section_with_no_completed_row_says_so():
    events = events_table([(f"f{i}", KEY_ANY_DIRECTION, 60 + 6 * i) for i in range(10)])
    rows = library_rows(
        [_sequence_row(f"f{i}", 60 + 6 * i, 60 + 6 * i + 5) for i in range(10)]
        + [
            {"event_id": "f0", "outcome_type": "asset", "effective": 60, "complete": 300,
             "horizon": 5}
        ]
    )
    card = card_for(events, rows, outcome_type="all")
    text = render_text(card)
    assert "no completed outcomes at horizon 5" in text
    asset_at = text.index("asset")
    assert text.index("no completed outcomes at horizon 5", asset_at) > asset_at
    assert "asset" not in card.not_built  # the library has asset rows; none is completed yet


@pytest.mark.parametrize(
    ("outcome_type", "horizon"),
    [("asset", 60), ("event_sequence", 1)],
)
def test_horizon_is_validated_per_outcome_type(outcome_type, horizon):
    with pytest.raises(ValueError, match="horizon"):
        ask._check_request(outcome_type, horizon, load_engine_params())


def test_the_union_of_horizons_is_still_allowed_for_all():
    ask._check_request("all", 1, load_engine_params())
    ask._check_request("all", 60, load_engine_params())
