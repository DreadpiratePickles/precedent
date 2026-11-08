"""Tests for `build_libraries` and `precedent build`."""

import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from precedent.engine.params import EngineParams, load_engine_params
from precedent.libraries import build as build_module
from precedent.libraries.build import (
    BuildReport,
    NoEventsError,
    build_libraries,
    current_commit,
    run,
)
from precedent.libraries.bundle import load_bundle
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.spine.edgar_8k_events import SOURCE as EDGAR_SOURCE
from precedent.spine.edgar_8k_events import edgar_parser_version
from precedent.spine.event_record import ResolvedImpact
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import SOURCE_NAME as GDELT_SOURCE
from precedent.spine.gdelt_events import gdelt_parser_version
from precedent.spine.session_calendar import load_nyse_calendar, weekday_sessions

PARAMS = load_engine_params()
COMMIT = "0" * 40


def test_build_without_prices_builds_two_libraries_and_counts_untyped(tmp_path, populated_stores):
    report = build_libraries(populated_stores, PARAMS, commit="0" * 40)
    assert report.asset is None and report.asset_not_built_reason == "no prices"
    assert report.market_resolution.markets_untyped == 2 and report.event_sequence.rows > 0
    bundle = load_bundle(populated_stores / "library")
    assert set(bundle.rows["outcome_type"]) == {"market_resolution", "event_sequence"}
    assert (
        json.loads((populated_stores / "library" / "build_report.json").read_text())[
            "market_resolution"
        ]["markets_untyped"]
        == 2
    )


def test_build_report_counts_match_the_written_bundle(populated_stores):
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT)
    bundle = load_bundle(populated_stores / "library")
    assert isinstance(report, BuildReport)
    assert (report.events.records, report.events.kept, report.events.unaligned) == (30, 30, 0)
    assert len(bundle.events) == 30
    counts = bundle.rows["outcome_type"].value_counts()
    assert counts["event_sequence"] == report.event_sequence.rows
    assert counts["market_resolution"] == report.market_resolution.rows > 0
    assert len(bundle.markets) == 4  # the typed markets only
    assert bundle.manifest.built_from_commit == COMMIT
    assert bundle.placebo_asset.empty
    assert list(bundle.placebo_asset.columns) == [
        "outcome_key",
        "horizon",
        "value",
        "response_complete_session",
    ]


def test_sessions_cover_the_events_and_markets_with_a_lookahead(populated_stores):
    build_libraries(populated_stores, PARAMS, commit=COMMIT)
    bundle = load_bundle(populated_stores / "library")
    assert bundle.sessions[0] == pd.Timestamp("2021-02-01")
    # the last resolution is the untyped market of 2021-03-24; the lookahead is 120 days
    assert bundle.sessions[-1] == pd.Timestamp("2021-07-22")
    assert bundle.session_cutoffs_utc.index.equals(bundle.sessions)
    assert set(bundle.events["effective_session"]) <= {s.date() for s in bundle.sessions}


def test_manifest_sources_are_named_dated_and_cited_without_links(populated_stores):
    build_libraries(populated_stores, PARAMS, commit=COMMIT)
    manifest = load_bundle(populated_stores / "library").manifest
    sources = {source.name: source for source in manifest.sources}
    assert list(sources) == ["gdelt2", "polymarket", "kalshi"]
    assert sources["gdelt2"].rows == 30
    assert sources["gdelt2"].first_date == dt.date(2021, 2, 1)
    assert sources["gdelt2"].last_date == dt.date(2021, 3, 15)
    assert (sources["polymarket"].rows, sources["kalshi"].rows) == (2, 2)
    assert sources["polymarket"].first_date == dt.date(2021, 3, 10)
    assert sources["polymarket"].last_date == dt.date(2021, 3, 17)
    assert sources["kalshi"].first_date == dt.date(2021, 3, 15)
    assert sources["kalshi"].last_date == dt.date(2021, 3, 22)
    assert "citation and link required" in sources["gdelt2"].citation
    assert all("://" not in source.citation for source in manifest.sources)
    assert manifest.precedent_version


def test_an_empty_event_store_is_refused_and_names_ingest(tmp_path):
    with pytest.raises(NoEventsError, match="precedent ingest"):
        build_libraries(tmp_path, PARAMS, commit=COMMIT)
    assert not (tmp_path / "library").exists()


def test_events_that_never_aligned_are_refused(tmp_path, populated_stores):
    store = EventStore(populated_stores / "events")
    unaligned = [
        record.model_copy(update={"effective_session": None, "available_utc": None})
        for record in store.read(gdelt_parser_version())
    ]
    other = tmp_path / "other"
    EventStore(other / "events").append(unaligned)
    with pytest.raises(NoEventsError, match="aligned"):
        build_libraries(other, PARAMS, commit=COMMIT)


def test_stored_gdelt_duplicates_are_collapsed_before_tabulating(populated_stores):
    store = EventStore(populated_stores / "events")
    first = store.read(gdelt_parser_version())[0]
    store.append([first.model_copy(update={"article_id": "gdelt:999", "body_sha256": "9" * 64})])
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT)
    assert report.events.records == 30 and report.events.kept == 30


def test_another_typer_version_reads_no_markets(populated_stores):
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT, typer_version="other-v1")
    assert report.market_resolution.markets_read == 0 and report.market_resolution.rows == 0
    manifest = load_bundle(populated_stores / "library").manifest
    assert [source.name for source in manifest.sources] == ["gdelt2"]


def test_the_gdelt_source_is_listed_with_its_citation_under_the_spine_source_name(
    populated_stores,
):
    records = EventStore(populated_stores / "events").read(gdelt_parser_version())
    assert {record.source for record in records} == {GDELT_SOURCE} == {"gdelt2"}
    build_libraries(populated_stores, PARAMS, commit=COMMIT)
    manifest = load_bundle(populated_stores / "library").manifest
    gdelt = next(source for source in manifest.sources if source.name == "gdelt2")
    assert "GDELT Project" in gdelt.citation and gdelt.rows == 30


def test_an_edgar_record_gives_an_edgar_8k_source(populated_stores):
    store = EventStore(populated_stores / "events")
    base = store.read(gdelt_parser_version())[3]
    store.append(
        [
            base.model_copy(
                update={
                    "article_id": "0000000000-21-000001:2.02",
                    "body_sha256": "6" * 64,
                    "source": EDGAR_SOURCE,
                    "parser_version": edgar_parser_version(),
                }
            )
        ]
    )
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT)
    assert report.events.kept == 31
    sources = {s.name: s for s in load_bundle(populated_stores / "library").manifest.sources}
    assert list(sources) == ["gdelt2", "edgar_8k", "polymarket", "kalshi"]
    assert sources["edgar_8k"].rows == 1 and sources["gdelt2"].rows == 30
    assert "Securities and Exchange Commission" in sources["edgar_8k"].citation
    assert sources["edgar_8k"].first_date == sources["edgar_8k"].last_date


def _write_prices(root: Path, symbols: list[str]) -> None:
    days = weekday_sessions(dt.date(2020, 1, 2), dt.date(2021, 12, 31), load_nyse_calendar())
    rng = np.random.default_rng(7)
    market = rng.normal(0.0, 0.01, len(days))
    (root / "prices").mkdir()
    for symbol in symbols:
        returns = market if symbol == "SPY" else 0.8 * market + rng.normal(0.0, 0.005, len(days))
        close = 100.0 * np.cumprod(1.0 + returns)
        opens = close * (1.0 + rng.normal(0.0, 0.001, len(days)))
        frame = pd.DataFrame({"date": days, "open": opens, "close": close})
        frame.to_parquet(root / "prices" / f"{symbol}.parquet")


def _add_impacting_event(root: Path) -> None:
    """A record on 2021-04-01 whose parsed impact resolves to the instrument ZZZ."""
    store = EventStore(root / "events")
    record = store.read(gdelt_parser_version())[20]
    store.append(
        [
            record.model_copy(
                update={
                    "article_id": "gdelt:901",
                    "body_sha256": "7" * 64,
                    "first_seen_utc": dt.datetime(2021, 4, 1, 14, tzinfo=dt.UTC),
                    "available_utc": dt.datetime(2021, 4, 1, 15, tzinfo=dt.UTC),
                    "effective_session": dt.date(2021, 4, 1),
                    "resolved_impacts": [
                        ResolvedImpact(
                            instrument="ZZZ", kind="company", direction=1.0, match_method="x"
                        )
                    ],
                }
            )
        ]
    )


def test_with_prices_for_the_market_and_a_panel_symbol_the_asset_library_is_built(populated_stores):
    _write_prices(populated_stores, ["SPY", "QQQ", "ZZZ"])
    _add_impacting_event(populated_stores)
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT)
    assert report.asset is not None and report.asset_not_built_reason is None
    assert report.asset.rows > 0
    bundle = load_bundle(populated_stores / "library")
    asset = bundle.rows[bundle.rows["outcome_type"] == "asset"]
    assert len(asset) == report.asset.rows
    assert set(asset["outcome_key"]) == {"QQQ", "ZZZ"}  # the market is not its own asset
    assert set(asset.loc[asset["outcome_key"] == "ZZZ", "event_id"]) <= {"gdelt:901"}
    assert not asset.loc[asset["outcome_key"] == "ZZZ"].empty
    assert set(asset["horizon"]) <= set(PARAMS.asset_horizons)
    placebo = bundle.placebo_asset
    assert set(placebo["outcome_key"]) == {"QQQ"}
    assert set(placebo["horizon"]) == set(PARAMS.asset_horizons)
    assert (placebo.groupby(["outcome_key", "horizon"]).size() == PARAMS.placebo_samples).all()
    assert placebo["value"].map(np.isfinite).all()


def test_prices_move_the_session_start_back_so_early_events_have_history(populated_stores):
    _write_prices(populated_stores, ["SPY", "QQQ"])  # from 2020-01-02, a year before the events
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT)
    assert report.asset is not None
    assert report.asset.skipped_short_history == 0
    assert report.asset.rows == 30 * len(PARAMS.asset_horizons)  # every event, QQQ, every horizon
    bundle = load_bundle(populated_stores / "library")
    first_event = bundle.events["effective_session"].min()
    assert first_event == dt.date(2021, 2, 1)
    assert bundle.sessions[0] == pd.Timestamp("2019-12-30")  # 400 days before, next session
    asset = bundle.rows[bundle.rows["outcome_type"] == "asset"]
    assert asset["effective_session"].min() == pd.Timestamp("2021-02-01")


def test_without_prices_the_sessions_start_at_the_earliest_date(populated_stores):
    build_libraries(populated_stores, PARAMS, commit=COMMIT)
    assert load_bundle(populated_stores / "library").sessions[0] == pd.Timestamp("2021-02-01")


def test_the_placebo_is_reproducible_and_avoids_event_sessions(populated_stores):
    _write_prices(populated_stores, ["SPY", "QQQ"])
    _add_impacting_event(populated_stores)
    build_libraries(populated_stores, PARAMS, commit=COMMIT)
    first = load_bundle(populated_stores / "library")
    build_libraries(populated_stores, PARAMS, commit=COMMIT)
    second = load_bundle(populated_stores / "library")
    pd.testing.assert_frame_equal(first.placebo_asset, second.placebo_asset)
    positions = first.sessions.get_indexer(
        pd.DatetimeIndex(first.placebo_asset["response_complete_session"])
    )
    event_positions = set(first.events.loc[first.events["is_first_report"], "effective_position"])
    horizons = first.placebo_asset["horizon"].to_numpy()
    assert not (
        {int(p) - 1 - int(h) for p, h in zip(positions, horizons, strict=True)} & event_positions
    )


def test_only_the_market_symbol_is_not_enough_for_the_asset_library(populated_stores):
    _write_prices(populated_stores, ["SPY"])
    report = build_libraries(populated_stores, PARAMS, commit=COMMIT)
    assert report.asset is None and report.asset_not_built_reason == "no prices"


def test_current_commit_is_a_full_hash_or_unknown():
    commit = current_commit()
    assert commit == "unknown" or (len(commit) == 40 and commit == commit.lower())


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (subprocess.CompletedProcess([], 0, stdout=f"{'ab' * 20}\n", stderr=""), "ab" * 20),
        (subprocess.CompletedProcess([], 128, stdout="", stderr="not a repo"), "unknown"),
        (subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""), "unknown"),
        (subprocess.CompletedProcess([], 0, stdout=f"{'AB' * 20}\n", stderr=""), "unknown"),
        (FileNotFoundError("git"), "unknown"),
    ],
)
def test_current_commit_uses_git_without_a_shell(monkeypatch, outcome, expected):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(build_module.subprocess, "run", fake_run)
    assert current_commit() == expected
    command, kwargs = calls[0]
    assert command == ["git", "rev-parse", "HEAD"]
    assert not kwargs.get("shell") and kwargs["check"] is False and kwargs["capture_output"]


def test_run_builds_prints_one_json_line_and_returns_zero(populated_stores, capsys, monkeypatch):
    monkeypatch.delenv("PRECEDENT_TYPER", raising=False)
    assert run(["--data-dir", str(populated_stores)]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 1
    counts = json.loads(out)
    assert counts["events"] == {"records": 30, "kept": 30, "unaligned": 0}
    assert counts["market_resolution"]["markets_untyped"] == 2
    assert counts["asset"] is None and counts["asset_not_built_reason"] == "no prices"
    assert load_bundle(populated_stores / "library").manifest.built_from_commit == current_commit()


def test_run_defaults_to_the_data_dir_from_the_environment(populated_stores, capsys, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(populated_stores))
    monkeypatch.delenv("PRECEDENT_TYPER", raising=False)
    assert run([]) == 0
    assert (populated_stores / "library" / "MANIFEST.json").is_file()


def test_run_on_an_empty_data_dir_returns_two_and_names_ingest(tmp_path, capsys):
    assert run(["--data-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "precedent ingest" in captured.err and captured.out == ""


def test_run_reports_a_config_error_and_returns_two(populated_stores, capsys, monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "no-such-typer")
    assert run(["--data-dir", str(populated_stores)]) == 2
    assert capsys.readouterr().err.startswith("config error:")
    assert not (populated_stores / "library").exists()


def test_run_rejects_an_unknown_flag_with_two(capsys):
    assert run(["--bogus"]) == 2
    assert "--bogus" in capsys.readouterr().err


def test_run_reports_invalid_engine_parameters_in_one_line(populated_stores, capsys, monkeypatch):
    def broken_params():
        return EngineParams.model_validate({})

    monkeypatch.setattr(build_module, "load_engine_params", broken_params)
    assert run(["--data-dir", str(populated_stores)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: engine parameters are invalid") and len(err.splitlines()) == 1
    with pytest.raises(ValidationError):
        broken_params()


def test_run_reports_an_integrity_error_without_a_traceback(populated_stores, capsys, monkeypatch):
    def refuse(*args, **kwargs):
        raise LibraryIntegrityError("events.parquet", "sha256 differs from the manifest")

    monkeypatch.setattr(build_module, "build_libraries", refuse)
    assert run(["--data-dir", str(populated_stores)]) == 2
    assert "events.parquet" in capsys.readouterr().err


def test_current_commit_treats_any_os_error_as_unknown(monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("git")

    monkeypatch.setattr(build_module.subprocess, "run", refuse)
    assert current_commit() == "unknown"
