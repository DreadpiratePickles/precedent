import datetime as dt
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from precedent.config import ConfigError, references_dir
from precedent.engine.card import (
    DEFAULT_FONT_PATH,
    DISCLAIMER,
    KEY_FLAGS_HINT,
    CardModel,
    CardRow,
    NearestPrecedent,
    SourceCoverage,
    format_outcome_row,
    render_json,
    render_png,
    render_text,
)

EXPECTED = "\n".join([
    "PRECEDENT  ·  a central bank raises rates unexpectedly",
    "typed as       central_bank / central_bank / US / severity high / unscheduled",
    "precedents     n=12  independent=12  earliest 2016-12-14  latest 2023-07-26",
    "",
    "event_sequence                      horizon   rate|median   80% band        n   vs ordinary",
    "election_or_political/linked_country     20   9/12 (75%)    [57%, 87%]     12   outside",
    "market_resolution                   horizon   rate|median   80% band        n   vs ordinary",
    "central_bank/scheduled/US               res   refused: independent=5, need 8",
    "asset                               NOT BUILT: no prices",
    "",
    "nearest independent precedents (non-overlapping outcome windows)",
    "2023-07-26  Federal Reserve raises policy rate  similarity 1.00  gdelt:1112223334",
    "library covers  gdelt2: 2015-02-19 to 2024-03-31, 41233 events; kalshi: 2021-07-01 to 2024-03-31, 812 markets",
    "These are past outcomes of similar events, not a forecast.",
])

REFUSED_EXPECTED = "\n".join([
    "PRECEDENT  ·  a cyber attack on a mid-size bank",
    "typed as       cyber_incident / company / US / severity high / unscheduled",
    "precedents     n=3  independent=3  earliest 2019-03-04  latest 2023-11-09",
    "",
    "REFUSED        3 usable precedents, need 8. No band is drawn below that.",
    "nearest independent precedents (non-overlapping outcome windows)",
    "2023-11-09  8-K item 1.05 cybersecurity incident  similarity 1.00  0000950170-23-061234:1.05",
    "library covers  edgar_8k: 2015-01-02 to 2024-03-29, 9120 events",
    "These are past outcomes of similar events, not a forecast.",
])

ASOF = dt.datetime(2024, 4, 1, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def golden_card() -> CardModel:
    return CardModel(
        headline="a central bank raises rates unexpectedly",
        typed_as="central_bank / central_bank / US / severity high / unscheduled",
        typed_reason=None,
        key="central_bank/US",
        asof_utc=ASOF,
        horizon=20,
        outcome_type="all",
        n=12,
        n_eff=12.0,
        earliest=dt.date(2016, 12, 14),
        latest=dt.date(2023, 7, 26),
        refused=False,
        min_neff=8.0,
        rows=[
            CardRow(
                outcome_type="event_sequence", outcome_key="election_or_political/linked_country",
                horizon=20, is_binary=True, n=12, n_eff=12.0, refused=False, k=9, rate=0.75,
                interval=(0.5667, 0.8731), median=None, centre_shrunk=None, band=None,
                vs_ordinary="outside",
            ),
            CardRow(
                outcome_type="market_resolution", outcome_key="central_bank/scheduled/US",
                horizon=None, is_binary=True, n=5, n_eff=5.0, refused=True, k=None, rate=None,
                interval=None, median=None, centre_shrunk=None, band=None, vs_ordinary=None,
            ),
        ],
        not_built={"asset": "no prices"},
        nearest=[
            NearestPrecedent(
                date=dt.date(2023, 7, 26), one_line="Federal Reserve raises policy rate",
                similarity=1.0, source_ref="gdelt:1112223334",
            )
        ],
        covers=[
            SourceCoverage(
                name="gdelt2", first_date=dt.date(2015, 2, 19), last_date=dt.date(2024, 3, 31),
                rows=41233,
            ),
            SourceCoverage(
                name="kalshi", first_date=dt.date(2021, 7, 1), last_date=dt.date(2024, 3, 31),
                rows=812,
            ),
        ],
        hint=None,
    )


@pytest.fixture
def refused_card() -> CardModel:
    return CardModel(
        headline="a cyber attack on a mid-size bank",
        typed_as="cyber_incident / company / US / severity high / unscheduled",
        typed_reason=None,
        key="cyber_incident/US",
        asof_utc=ASOF,
        horizon=20,
        outcome_type="all",
        n=3,
        n_eff=3.0,
        earliest=dt.date(2019, 3, 4),
        latest=dt.date(2023, 11, 9),
        refused=True,
        min_neff=8.0,
        rows=[],
        not_built={},
        nearest=[
            NearestPrecedent(
                date=dt.date(2023, 11, 9), one_line="8-K item 1.05 cybersecurity incident",
                similarity=1.0, source_ref="0000950170-23-061234:1.05",
            )
        ],
        covers=[
            SourceCoverage(
                name="edgar_8k", first_date=dt.date(2015, 1, 2), last_date=dt.date(2024, 3, 29),
                rows=9120,
            )
        ],
        hint=None,
    )


@pytest.fixture
def untyped_card() -> CardModel:
    return CardModel(
        headline="something nobody can type",
        typed_as="UNTYPED",
        typed_reason="no rule matched the text",
        key=None,
        asof_utc=ASOF,
        horizon=20,
        outcome_type="all",
        n=0,
        n_eff=0.0,
        earliest=None,
        latest=None,
        refused=True,
        min_neff=8.0,
        rows=[],
        not_built={},
        nearest=[],
        covers=[],
        hint=KEY_FLAGS_HINT,
    )


def test_card_text_is_exact(golden_card):
    assert render_text(golden_card) == EXPECTED


def test_refused_card_text_is_exact_and_draws_no_band(refused_card):
    text = render_text(refused_card)
    assert text == REFUSED_EXPECTED and "80% band" not in text and "%" not in text.split("REFUSED")[1].split("library covers")[0]


def test_continuous_row_format():
    row = CardRow(outcome_type="asset", outcome_key="SPY", horizon=5, is_binary=False, n=19, n_eff=19.0, refused=False, k=None, rate=None,
                  interval=None, median=-0.008, centre_shrunk=-0.0052, band=(-0.031, 0.014), vs_ordinary="inside")
    assert format_outcome_row(row, 8.0) == "SPY                                       5   -0.8%         [-3.1%, +1.4%]  19   inside"


def test_every_card_ends_with_covers_then_disclaimer(golden_card, refused_card, untyped_card):
    for card in (golden_card, refused_card, untyped_card):
        lines = render_text(card).splitlines()
        assert lines[-2].startswith("library covers  ") and lines[-1] == DISCLAIMER


def test_untyped_card_says_so(untyped_card):
    assert "typed as       UNTYPED" in render_text(untyped_card) and untyped_card.hint == KEY_FLAGS_HINT


def test_disclaimer_cannot_be_changed(golden_card):
    with pytest.raises(ValidationError):
        CardModel(**{**golden_card.model_dump(), "disclaimer": "trust me"})


def test_json_round_trips_with_schema_version(golden_card):
    data = json.loads(render_json(golden_card))
    assert data["schema_version"] == "1.0" and data["disclaimer"] == DISCLAIMER
    assert CardModel.model_validate_json(render_json(golden_card)) == golden_card


def test_png_is_written_and_deterministic(tmp_path, golden_card):
    a = render_png(golden_card, tmp_path / "a.png", font_path=None).read_bytes()
    b = render_png(golden_card, tmp_path / "b.png", font_path=None).read_bytes()
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and a == b
    assert len(Image.open(tmp_path / "a.png").getcolors(maxcolors=1_000_000)) > 1


def test_png_with_a_missing_font_is_a_config_error(tmp_path, golden_card):
    with pytest.raises(ConfigError, match="card font not bundled"):
        render_png(golden_card, tmp_path / "a.png", font_path=tmp_path / "missing.ttf")


def test_single_outcome_type_card_prints_only_that_section(golden_card):
    card = golden_card.model_copy(update={"outcome_type": "event_sequence"})
    text = render_text(card)
    assert "event_sequence      " in text
    assert "market_resolution" not in text and "NOT BUILT" not in text
    assert "central_bank/scheduled/US" not in text


def test_a_row_without_a_vs_ordinary_verdict_prints_a_dash(golden_card):
    row = golden_card.rows[0].model_copy(update={"vs_ordinary": None})
    assert format_outcome_row(row, 8.0).endswith("12   -")


def test_empty_covers_says_no_library_loaded(golden_card):
    card = golden_card.model_copy(update={"covers": []})
    assert render_text(card).splitlines()[-2] == "library covers  nothing (no library loaded)"


def test_empty_nearest_prints_none(golden_card):
    card = golden_card.model_copy(update={"nearest": []})
    lines = render_text(card).splitlines()
    assert lines[lines.index("nearest independent precedents (non-overlapping outcome windows)") + 1] == "(none)"


def test_the_caller_trims_rows_so_the_model_has_no_row_cap_field():
    assert "card_rows_per_type" not in CardModel.model_fields


def test_default_font_path_is_inside_the_references_dir_and_not_bundled_yet():
    expected = references_dir() / "fonts" / "DejaVuSansMono.ttf"
    assert expected == DEFAULT_FONT_PATH


def test_png_with_the_default_font_path_follows_the_bundle(tmp_path, golden_card):
    if DEFAULT_FONT_PATH.exists():
        assert render_png(golden_card, tmp_path / "a.png").exists()
    else:
        with pytest.raises(ConfigError, match="card font not bundled"):
            render_png(golden_card, tmp_path / "a.png")
        assert not (tmp_path / "a.png").exists()


def test_a_naive_asof_is_rejected(golden_card):
    with pytest.raises(ValidationError, match="timezone-aware"):
        CardModel(**{**golden_card.model_dump(), "asof_utc": dt.datetime(2024, 4, 1, 12, 0)})


def test_an_unknown_outcome_type_is_rejected(golden_card):
    with pytest.raises(ValidationError, match="outcome_type"):
        CardModel(**{**golden_card.model_dump(), "outcome_type": "crypto"})


def test_an_unrefused_row_must_carry_the_values_it_prints():
    with pytest.raises(ValidationError, match="lacks k, rate, interval"):
        CardRow(outcome_type="event_sequence", outcome_key="x", horizon=5, is_binary=True, n=9, n_eff=9.0,
                refused=False, k=None, rate=None, interval=None, median=None, centre_shrunk=None, band=None,
                vs_ordinary=None)
