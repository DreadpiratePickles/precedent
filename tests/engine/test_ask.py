import datetime as dt
import json
import socket
from pathlib import Path

import pandas as pd
import pytest
from library_root_synthetic import (
    ASSET_SYMBOL,
    EVENT_SPACING_SESSIONS,
    FIRST_EVENT_POSITION,
    MARKET_OUTCOME_KEY,
    ROOT_CUTOFFS,
    ROOT_SESSIONS,
    THIN_MARKET_OUTCOME_KEY,
    write_library_root,
)

from precedent import config
from precedent.config import ConfigError
from precedent.engine import ask
from precedent.engine.ask import run, run_card
from precedent.engine.card import DISCLAIMER, KEY_FLAGS_HINT, CardModel, render_png
from precedent.engine.keys import key_from_flags
from precedent.engine.retrieval import PrecedentQuery
from precedent.libraries.bundle import load_bundle
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.spine.typers.base import TypedText
from precedent.spine.typers.fake import FakeTyper

KEY_FLAGS = [
    "--event-type", "central_bank", "--actor-kind", "central_bank",
    "--severity-band", "high", "--unscheduled", "--country-group", "US",
]
CENTRAL_BANK_US_HIGH_KEY = key_from_flags("central_bank", "central_bank", "high", False, "US")
NOW = pd.Timestamp("2024-06-03", tz="UTC")


def typed(key):
    return TypedText(
        status="typed", key=key, reason="rule", evidence_span="rates", typer_version="fake-v1"
    )


def forbid_socket(*args, **kwargs):
    raise AssertionError("the dry run opened a socket")


def raising_config_error(message):
    def raise_it(*args, **kwargs):
        raise ConfigError(message)

    return raise_it


# --- the eleven tests of the brief ------------------------------------------------------------


def test_key_query_prints_card_and_disclaimer(capsys, library_root):
    assert run(KEY_FLAGS + ["--asof", "2024-06-03", "--data-dir", str(library_root)]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith(DISCLAIMER) and "independent=" in out and "event_sequence" in out


def test_too_few_precedents_is_a_refusal_not_a_band(capsys, sparse_library_root):
    assert run(["--event-type", "cyber_incident", "--actor-kind", "company", "--severity-band", "high", "--unscheduled", "--data-dir", str(sparse_library_root)]) == 0
    out = capsys.readouterr().out
    assert "REFUSED" in out and "80% band" not in out


def test_untyped_text_is_a_refusal_with_the_key_flags(capsys, library_root, monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    assert run(["zzzz qqqq", "--data-dir", str(library_root)]) == 0
    captured = capsys.readouterr()
    assert "typed as       UNTYPED" in captured.out and "--event-type" in captured.err


def test_typer_config_error_exits_2(capsys, library_root, monkeypatch):
    monkeypatch.setattr(ask, "typer_from_env", raising_config_error("ANTHROPIC_API_KEY is not set"))
    assert run(["a central bank raises rates unexpectedly", "--data-dir", str(library_root)]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "--event-type" in err


def test_missing_library_exits_2_and_says_what_to_do(capsys, tmp_path):
    assert run(KEY_FLAGS + ["--data-dir", str(tmp_path)]) == 2
    assert "precedent fetch" in capsys.readouterr().err


def test_dry_run_reads_the_sample_and_opens_no_socket(capsys, monkeypatch, library_root):
    monkeypatch.setattr(config, "sample_dir", lambda: library_root)  # a synthetic data dir, never data/sample
    monkeypatch.setattr(socket, "socket", forbid_socket)
    assert run(["a central bank raises rates unexpectedly", "--dry-run"]) == 0
    assert capsys.readouterr().out.rstrip().endswith(DISCLAIMER)


def test_dry_run_without_a_sample_still_renders_a_refusal(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "sample_dir", lambda: tmp_path)
    assert run(KEY_FLAGS + ["--dry-run"]) == 0
    assert "library covers  nothing (no library loaded)" in capsys.readouterr().out


def test_json_flag_emits_the_card_object(capsys, library_root):
    assert run(KEY_FLAGS + ["--json", "--data-dir", str(library_root)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "1.0"


def test_run_card_with_a_fake_typer_matches_the_key_query(library_root):
    typer = FakeTyper({"rates up": typed(CENTRAL_BANK_US_HIGH_KEY)})
    by_text = run_card("rates up", 5, "all", library_root, typer)
    by_key = run_card(PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, by_text.asof_utc), 5, "all", library_root, typer)
    assert (by_text.n, by_text.rows) == (by_key.n, by_key.rows) and by_text.headline == "rates up"


def test_query_before_any_completion_has_no_precedents(library_root):
    card = run_card(PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, pd.Timestamp("2000-01-01", tz="UTC")), 5, "all", library_root, FakeTyper({}))
    assert card.n == 0 and card.refused is True


def test_asset_without_prices_is_not_built(library_root):
    assert run_card(PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, NOW), 5, "all", library_root, FakeTyper({})).not_built == {"asset": "no prices"}


# --- the rest ---------------------------------------------------------------------------------


def _card(root, outcome_type="all", horizon=5, asof=NOW) -> CardModel:
    query = PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, asof)
    return run_card(query, horizon, outcome_type, root, FakeTyper({}))


def test_a_key_card_has_the_expected_event_level_numbers(library_root):
    card = _card(library_root)
    assert (card.n, card.n_eff, card.refused) == (20, 20.0, False)
    assert card.headline == "central_bank / central_bank / US / severity high / unscheduled"
    assert card.typed_as == card.headline and card.typed_reason is None
    assert card.key == "central_bank/central_bank/none/US/high/unscheduled/0"
    assert (card.earliest, card.latest) == (dt.date(2020, 1, 16), dt.date(2021, 11, 11))
    assert len(card.nearest) == 5 and card.nearest[0].date == card.latest
    assert [c.name for c in card.covers] == ["gdelt2", "kalshi"]
    assert card.asof_utc == NOW.to_pydatetime()


def test_the_event_sequence_row_pools_the_selected_events_at_the_horizon(library_root):
    row = next(r for r in _card(library_root).rows if r.outcome_type == "event_sequence")
    assert (row.outcome_key, row.horizon, row.is_binary, row.n) == (
        "election_or_political/linked_country", 5, True, 20,
    )
    assert row.refused is False and row.k is not None and row.k <= 20 and row.interval is not None
    assert row.vs_ordinary in ("inside", "outside", None)


def test_the_market_row_has_no_horizon_and_pools_every_horizon(library_root):
    row = next(r for r in _card(library_root).rows if r.outcome_type == "market_resolution")
    assert (row.outcome_key, row.horizon, row.n) == (MARKET_OUTCOME_KEY, None, 20)


def test_asof_date_means_the_end_of_that_day_in_utc(capsys, library_root):
    assert run(KEY_FLAGS + ["--asof", "2024-06-03", "--json", "--data-dir", str(library_root)]) == 0
    asof = json.loads(capsys.readouterr().out)["asof_utc"]
    assert pd.Timestamp(asof) == pd.Timestamp("2024-06-03T23:59:59", tz="UTC")


def test_asof_datetime_naive_is_utc_and_aware_is_converted(capsys, library_root):
    for text in ("2024-06-03T12:30:00", "2024-06-03T14:30:00+02:00"):
        assert run(KEY_FLAGS + ["--asof", text, "--json", "--data-dir", str(library_root)]) == 0
        asof = json.loads(capsys.readouterr().out)["asof_utc"]
        assert pd.Timestamp(asof) == pd.Timestamp("2024-06-03T12:30:00", tz="UTC")


def test_a_bad_asof_is_a_usage_error(capsys, library_root):
    assert run(KEY_FLAGS + ["--asof", "last tuesday", "--data-dir", str(library_root)]) == 2
    assert "--asof" in capsys.readouterr().err


def test_text_and_key_flags_together_is_a_usage_error(capsys, library_root):
    assert run(["rates up", *KEY_FLAGS, "--data-dir", str(library_root)]) == 2
    assert capsys.readouterr().out == ""


def test_neither_text_nor_key_flags_is_a_usage_error(capsys, library_root):
    assert run(["--data-dir", str(library_root)]) == 2
    assert "TEXT" in capsys.readouterr().err


def test_a_partial_key_is_a_usage_error(capsys, library_root):
    assert run(["--event-type", "central_bank", "--data-dir", str(library_root)]) == 2
    err = capsys.readouterr().err
    assert "--actor-kind" in err and "--severity-band" in err


def test_an_unknown_key_value_is_a_usage_error(capsys, library_root):
    flags = ["--event-type", "nonsense", "--actor-kind", "central_bank", "--severity-band", "high", "--scheduled"]
    assert run(flags + ["--data-dir", str(library_root)]) == 2
    assert "nonsense" in capsys.readouterr().err


def test_scheduled_and_unscheduled_together_is_a_usage_error(capsys, library_root):
    assert run([*KEY_FLAGS, "--scheduled", "--data-dir", str(library_root)]) == 2


def test_an_unsupported_horizon_is_a_usage_error_and_a_value_error(capsys, library_root):
    assert run([*KEY_FLAGS, "--horizon", "7", "--data-dir", str(library_root)]) == 2
    with pytest.raises(ValueError, match="horizon"):
        _card(library_root, horizon=7)


def test_an_unknown_outcome_type_is_a_value_error(library_root):
    with pytest.raises(ValueError, match="outcome_type"):
        _card(library_root, outcome_type="nonsense")


def test_outcome_type_event_sequence_prints_only_that_section(capsys, library_root):
    assert run([*KEY_FLAGS, "--outcome-type", "event_sequence", "--data-dir", str(library_root)]) == 0
    out = capsys.readouterr().out
    assert "event_sequence" in out and "market_resolution" not in out and "NOT BUILT" not in out


def test_a_market_key_with_few_precedents_is_a_refused_row(tmp_path):
    root = write_library_root(tmp_path, with_thin_market_key=True)
    rows = {r.outcome_key: r for r in _card(root, "market_resolution").rows}
    thin = rows[THIN_MARKET_OUTCOME_KEY]
    assert thin.refused is True and (thin.n, thin.n_eff) == (3, 3.0)
    assert thin.band is None and thin.interval is None and thin.rate is None
    assert rows[MARKET_OUTCOME_KEY].refused is False


def test_rows_are_sorted_by_count_then_key_within_a_type(tmp_path):
    root = write_library_root(tmp_path, with_thin_market_key=True)
    keys = [r.outcome_key for r in _card(root, "market_resolution").rows]
    assert keys == [MARKET_OUTCOME_KEY, THIN_MARKET_OUTCOME_KEY]


def test_asset_rows_are_continuous_with_a_band_and_a_time_guarded_placebo(tmp_path):
    root = write_library_root(tmp_path, with_assets=True)
    late = _card(root, "asset")
    row = late.rows[0]
    assert (row.outcome_key, row.horizon, row.is_binary, row.refused) == (ASSET_SYMBOL, 5, False, False)
    assert row.band is not None and row.band[0] <= row.median <= row.band[1]
    assert row.vs_ordinary in ("inside", "outside")
    early = _card(root, "asset", asof=ROOT_CUTOFFS.iloc[262])  # before any placebo response is complete
    assert early.n >= 8 and early.rows[0].vs_ordinary is None


def test_a_card_never_shows_an_outcome_that_completed_after_the_query(library_root):
    asof = ROOT_CUTOFFS.iloc[212]  # events at positions 10 .. 210 (9) are available
    card = _card(library_root, "event_sequence", asof=asof)
    assert (card.n, card.refused) == (9, False)
    assert card.rows[0].n == 8  # the 9th event's horizon-5 response completes at position 215


def test_a_missing_horizon_in_the_rows_leaves_the_section_empty(library_root):
    card = _card(library_root, "event_sequence", horizon=1)
    assert card.rows == [] and card.refused is False


def test_an_untyped_text_card_carries_the_typer_reason_and_the_hint(library_root):
    card = run_card("zzzz qqqq", 5, "all", library_root, FakeTyper({}), asof_utc=NOW)
    assert (card.typed_as, card.typed_reason, card.key) == ("UNTYPED", "not in mapping", None)
    assert (card.n, card.n_eff, card.refused, card.rows, card.nearest) == (0, 0.0, True, [], [])
    assert card.hint == KEY_FLAGS_HINT and card.not_built == {} and len(card.covers) == 2
    assert (card.earliest, card.latest) == (None, None) and card.asof_utc == NOW.to_pydatetime()


def test_a_refused_typing_is_the_untyped_card(library_root):
    refused = TypedText(status="refused_invalid", key=None, reason="bad json", evidence_span=None, typer_version="v")
    card = run_card("x", 5, "all", library_root, FakeTyper({"x": refused}))
    assert card.typed_as == "UNTYPED" and card.typed_reason == "bad json" and card.refused


def test_the_bundle_is_loaded_before_the_text_is_typed(tmp_path):
    class ExplodingTyper:
        version = "x"

        def type_text(self, text):
            raise AssertionError("typed before loading the library")

    with pytest.raises(FileNotFoundError):
        run_card("rates up", 5, "all", tmp_path, ExplodingTyper())


def test_a_corrupt_library_exits_2(capsys, library_root):
    (library_root / "library" / "events.parquet").write_bytes(b"corrupt")
    assert run([*KEY_FLAGS, "--data-dir", str(library_root)]) == 2
    assert "events.parquet" in capsys.readouterr().err
    with pytest.raises(LibraryIntegrityError):
        _card(library_root)


def test_a_naive_asof_is_rejected_by_run_card(library_root):
    with pytest.raises(ValueError, match="timezone"):
        run_card("rates up", 5, "all", library_root, FakeTyper({}), asof_utc=pd.Timestamp("2024-06-03"))


def test_the_dry_run_data_dir_defaults_to_the_sample_dir(capsys, monkeypatch, library_root):
    monkeypatch.setattr(config, "sample_dir", lambda: library_root)
    assert run([*KEY_FLAGS, "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 20


def test_the_data_dir_defaults_to_the_configured_data_dir(capsys, monkeypatch, library_root):
    monkeypatch.setattr(config, "data_dir", lambda: library_root)
    assert run([*KEY_FLAGS, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 20


def test_a_dry_run_text_without_a_library_is_the_untyped_card_and_hint(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "sample_dir", lambda: tmp_path)
    assert run(["zzzz qqqq", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "typed as       UNTYPED" in captured.out and "--event-type" in captured.err
    assert "library covers  nothing (no library loaded)" in captured.out


def test_a_missing_library_message_names_the_three_ways_out(capsys, tmp_path):
    assert run([*KEY_FLAGS, "--data-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "precedent fetch" in err and "precedent build" in err and "--dry-run" in err


def _png_with_the_default_pillow_font(card, path):
    return render_png(card, path, font_path=None)


def test_a_card_png_under_docs_examples_gets_a_sidecar(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(ask, "render_png", _png_with_the_default_pillow_font)
    library_root = write_library_root(tmp_path / "live", synthetic=False)
    png = tmp_path / "docs" / "examples" / "x.png"
    png.parent.mkdir(parents=True)
    assert run([*KEY_FLAGS, "--card", str(png), "--data-dir", str(library_root)]) == 0
    assert png.is_file()
    lines = png.with_suffix(".md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "LIVE — precedent ask card"
    assert str(library_root) in lines[1] and "central_bank" in lines[1]


def test_the_sidecar_banner_is_synthetic_for_a_fixture_library(monkeypatch, library_root, tmp_path):
    monkeypatch.setattr(ask, "render_png", _png_with_the_default_pillow_font)
    png = tmp_path / "docs" / "examples" / "f.png"
    png.parent.mkdir(parents=True)
    assert run([*KEY_FLAGS, "--card", str(png), "--data-dir", str(library_root)]) == 0
    assert png.with_suffix(".md").read_text(encoding="utf-8").startswith("SYNTHETIC — ")


def test_the_sidecar_writes_the_home_directory_as_a_tilde(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert ask._display_path(tmp_path / "data" / "x") == "~/data/x"
    assert ask._display_path(Path("/elsewhere/x")) == "/elsewhere/x"


def test_a_typer_config_error_while_typing_exits_2(capsys, library_root, monkeypatch):
    class KeylessTyper:
        version = "x"

        def type_text(self, text):
            raise ConfigError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(ask, "typer_from_env", lambda: KeylessTyper())
    assert run(["rates up", "--data-dir", str(library_root)]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "--event-type" in err


def _sidecar_banner(argv, tmp_path, name):
    png = tmp_path / "docs" / "examples" / f"{name}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    assert run([*argv, "--card", str(png)]) == 0
    return png.with_suffix(".md").read_text(encoding="utf-8").splitlines()[0]


def test_the_banner_follows_the_library_evidence_not_the_sample_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ask, "render_png", _png_with_the_default_pillow_font)
    live = write_library_root(tmp_path / "live", synthetic=False)
    synthetic = write_library_root(tmp_path / "synthetic", synthetic=True)
    monkeypatch.setattr(config, "sample_dir", lambda: live)
    at_the_sample_dir = _sidecar_banner([*KEY_FLAGS, "--data-dir", str(live)], tmp_path, "a")
    assert at_the_sample_dir == "LIVE — precedent ask card"
    monkeypatch.setattr(config, "sample_dir", lambda: synthetic)
    assert _sidecar_banner([*KEY_FLAGS, "--data-dir", str(synthetic)], tmp_path, "b").startswith(
        "SYNTHETIC — "
    )


def test_a_dry_run_on_a_non_synthetic_library_is_live_and_a_synthetic_one_is_not(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ask, "render_png", _png_with_the_default_pillow_font)
    monkeypatch.setattr(config, "sample_dir", lambda: write_library_root(tmp_path / "l", synthetic=False))
    assert _sidecar_banner([*KEY_FLAGS, "--dry-run"], tmp_path, "c").startswith("LIVE — ")
    monkeypatch.setattr(config, "sample_dir", lambda: write_library_root(tmp_path / "s"))
    assert _sidecar_banner([*KEY_FLAGS, "--dry-run"], tmp_path, "d").startswith("SYNTHETIC — ")


def test_a_dry_run_with_no_library_loaded_is_labelled_no_live_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ask, "render_png", _png_with_the_default_pillow_font)
    monkeypatch.setattr(config, "sample_dir", lambda: tmp_path / "nowhere")
    banner = _sidecar_banner([*KEY_FLAGS, "--dry-run"], tmp_path, "e")
    assert banner == "NO LIVE RUN — precedent ask card"


@pytest.mark.parametrize("text", ["[1]", "null", '"synthetic"', "42", '{"synthetic": "true"}', "{}"])
def test_a_build_report_that_is_not_a_synthetic_marker_means_live(tmp_path, text):
    (tmp_path / "library").mkdir()
    (tmp_path / "library" / "build_report.json").write_text(text, encoding="utf-8")
    assert ask._evidence_banner(tmp_path, library_loaded=True) == "LIVE"


def test_the_evidence_banner_reads_the_synthetic_marker(tmp_path):
    (tmp_path / "library").mkdir()
    (tmp_path / "library" / "build_report.json").write_text('{"synthetic": true}', encoding="utf-8")
    assert ask._evidence_banner(tmp_path, library_loaded=True) == "SYNTHETIC"
    assert ask._evidence_banner(tmp_path, library_loaded=False) == "NO LIVE RUN"


def test_a_card_png_elsewhere_gets_no_sidecar(capsys, monkeypatch, library_root, tmp_path):
    monkeypatch.setattr(ask, "render_png", _png_with_the_default_pillow_font)
    png = tmp_path / "out" / "x.png"
    png.parent.mkdir()
    assert run([*KEY_FLAGS, "--card", str(png), "--data-dir", str(library_root)]) == 0
    assert png.is_file() and not png.with_suffix(".md").exists()


def test_a_missing_card_font_exits_2_after_printing_the_card(capsys, library_root, tmp_path):
    png = tmp_path / "docs" / "examples" / "x.png"
    png.parent.mkdir(parents=True)
    assert run([*KEY_FLAGS, "--card", str(png), "--data-dir", str(library_root)]) == 2
    captured = capsys.readouterr()
    assert captured.out.rstrip().endswith(DISCLAIMER) and "font" in captured.err
    assert not png.with_suffix(".md").exists()


# --- the placebo time guards (ordinary rates may use only what had completed by `asof`) --------

_ONE_MINUTE = pd.Timedelta(minutes=1)
_SEQUENCE_KEY = "central_bank/linked_country"  # the events of the fixture are central bank events
_EVENT_POSITIONS = [FIRST_EVENT_POSITION + n * EVENT_SPACING_SESSIONS for n in range(20)]


def _placebo_at(root, kind, outcome_key, asof, horizon=5):
    bundle = load_bundle(root / "library")
    query = PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, asof)
    return ask._placebo(kind, outcome_key, horizon, query, bundle)


def _hand_sequence_rate(end_position, horizon):
    """Share of start sessions 0..end-horizon followed by a first report within the horizon."""
    starts = range(0, end_position - horizon + 1)
    hits = sum(any(s < p <= s + horizon for p in _EVENT_POSITIONS) for s in starts)
    return hits / len(starts)


def test_the_sequence_placebo_counts_only_sessions_closed_by_asof(library_root):
    last_event = _EVENT_POSITIONS[8]  # session 210
    at_close = _placebo_at(library_root, "event_sequence", _SEQUENCE_KEY, ROOT_CUTOFFS.iloc[last_event])
    before_close = _placebo_at(
        library_root, "event_sequence", _SEQUENCE_KEY, ROOT_CUTOFFS.iloc[last_event] - _ONE_MINUTE
    )
    assert at_close == pytest.approx(_hand_sequence_rate(last_event, 5))
    assert before_close == pytest.approx(_hand_sequence_rate(last_event - 1, 5))
    assert at_close != pytest.approx(before_close)  # the event at 210 is visible only once it closed
    assert at_close != pytest.approx(_hand_sequence_rate(len(ROOT_SESSIONS) - 1, 5))  # not the whole library


def test_the_sequence_placebo_is_none_before_a_horizon_of_sessions_has_closed(library_root):
    assert _placebo_at(library_root, "event_sequence", _SEQUENCE_KEY, pd.Timestamp("2000-01-01", tz="UTC")) is None


def test_the_market_placebo_counts_only_resolutions_closed_by_asof(library_root):
    # markets resolve at sessions 30, 70, 110, ... and resolve YES when their index is odd
    def rate(asof):
        return _placebo_at(library_root, "market_resolution", MARKET_OUTCOME_KEY, asof)

    assert rate(ROOT_CUTOFFS.iloc[30] - _ONE_MINUTE) is None
    assert rate(ROOT_CUTOFFS.iloc[30]) == 0.0
    assert rate(ROOT_CUTOFFS.iloc[70] - _ONE_MINUTE) == 0.0
    assert rate(ROOT_CUTOFFS.iloc[70]) == 0.5
    assert rate(ROOT_CUTOFFS.iloc[110]) == pytest.approx(1 / 3)
    assert rate(NOW) == 0.5  # all ten resolved


def test_the_asset_placebo_counts_only_responses_closed_by_asof(tmp_path):
    root = write_library_root(tmp_path, with_assets=True)
    # placebo responses complete at sessions 300 .. 399
    assert _placebo_at(root, "asset", ASSET_SYMBOL, ROOT_CUTOFFS.iloc[300] - _ONE_MINUTE) is None
    one = _placebo_at(root, "asset", ASSET_SYMBOL, ROOT_CUTOFFS.iloc[300])
    three = _placebo_at(root, "asset", ASSET_SYMBOL, ROOT_CUTOFFS.iloc[302])
    assert (len(one), len(three)) == (1, 3)
    assert len(_placebo_at(root, "asset", ASSET_SYMBOL, NOW)) == 100
