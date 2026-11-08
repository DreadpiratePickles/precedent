"""The sample library builder: a date window of a built library with no prices and no text."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from precedent.libraries.bundle import LibraryMissingError, load_bundle, write_bundle
from precedent.libraries.manifest import MANIFEST_NAME
from precedent.libraries.sample import SampleTooLargeError, build_sample

CITATION = "The GDELT Project, GDELT 2.0 Event Database; citation required on redistribution."
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_sample_library.py"
EARLY = (date(2020, 1, 1), date(2020, 4, 30))
MARKET_COLUMNS = [
    "market_id",
    "outcome_key",
    "created_utc",
    "resolved_utc",
    "value",
    "response_complete_session",
]


def _sample(root, tmp_path, window=(date(2021, 1, 1), date(2021, 2, 28)), **kwargs):
    since, until = window
    out = tmp_path / "s"
    manifest = build_sample(root / "library", out, since, until, "SYNTHETIC", CITATION, **kwargs)
    return out, manifest


def _load_script():
    spec = importlib.util.spec_from_file_location("build_sample_library_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_keeps_the_window_and_ships_no_prices(tmp_path, bundle_root):
    build_sample(
        bundle_root / "library",
        tmp_path / "s",
        date(2021, 1, 1),
        date(2021, 2, 28),
        "SYNTHETIC",
        CITATION,
    )
    bundle = load_bundle(tmp_path / "s" / "library")
    assert bundle.events["effective_session"].between(date(2021, 1, 1), date(2021, 2, 28)).all()
    assert bundle.rows.query("outcome_type == 'asset'").empty and bundle.placebo_asset.empty


def test_sample_ships_no_question_text(tmp_path, bundle_root):
    build_sample(
        bundle_root / "library",
        tmp_path / "s",
        date(2021, 1, 1),
        date(2021, 2, 28),
        "SYNTHETIC",
        CITATION,
    )
    markets = load_bundle(tmp_path / "s" / "library").markets
    assert list(markets.columns) == [
        "market_id",
        "outcome_key",
        "created_utc",
        "resolved_utc",
        "value",
        "response_complete_session",
    ]


def test_readme_banner_and_citation(tmp_path, bundle_root):
    build_sample(
        bundle_root / "library",
        tmp_path / "s",
        date(2021, 1, 1),
        date(2021, 2, 28),
        "SYNTHETIC",
        CITATION,
    )
    lines = (tmp_path / "s" / "README.md").read_text().splitlines()
    assert lines[0].startswith("SYNTHETIC — GDELT 2.0 events") and CITATION in lines


def test_oversize_sample_is_refused_and_leaves_nothing(tmp_path, bundle_root):
    with pytest.raises(SampleTooLargeError):
        build_sample(
            bundle_root / "library",
            tmp_path / "s",
            date(2020, 1, 1),
            date(2022, 12, 31),
            "SYNTHETIC",
            CITATION,
            max_bytes=1000,
        )
    assert not (tmp_path / "s" / "library").exists()


def test_oversize_refusal_leaves_no_directory_at_all(tmp_path, bundle_root):
    with pytest.raises(SampleTooLargeError, match=r"sample is \d+ bytes, limit 1000"):
        _sample(
            bundle_root, tmp_path, window=(date(2020, 1, 1), date(2022, 12, 31)), max_bytes=1000
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["library"]


def test_readme_layout_and_source_lines(tmp_path, bundle_root):
    out, manifest = _sample(bundle_root, tmp_path, window=EARLY)
    lines = (out / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "SYNTHETIC — GDELT 2.0 events and resolved Polymarket/Kalshi market outcomes "
        "(ids, typed keys and dates; no question text), 2020-01-01 to 2020-04-30"
    )
    assert lines[1] == "" and lines[2] == CITATION
    assert lines[3:] == [
        f"- {s.name}: {s.first_date} to {s.last_date}, {s.rows} rows" for s in manifest.sources
    ]
    assert lines[3:] == ["- kalshi: 2020-02-10 to 2020-04-30, 1 rows"]


def test_gdelt_source_is_recounted_and_clipped(tmp_path, bundle_root, bundle_parts):
    parts = {**bundle_parts, "events": bundle_parts["events"].assign(source="gdelt2")}
    write_bundle(tmp_path / "src" / "library", **parts)
    manifest = build_sample(
        tmp_path / "src" / "library",
        tmp_path / "s",
        date(2020, 1, 1),
        date(2020, 4, 30),
        "SYNTHETIC",
        CITATION,
    )
    bundle = load_bundle(tmp_path / "s" / "library")
    by_name = {source.name: source for source in manifest.sources}
    assert by_name["gdelt2"].rows == len(bundle.events)
    assert by_name["gdelt2"].first_date >= date(2020, 1, 1)
    assert by_name["gdelt2"].last_date <= date(2020, 4, 30)
    assert by_name["gdelt2"].citation == "synthetic gdelt citation"
    assert "gdelt2" in (tmp_path / "s" / "README.md").read_text(encoding="utf-8")


def test_window_keeps_sequence_and_market_rows_of_kept_events_only(tmp_path, bundle_root):
    out, manifest = _sample(bundle_root, tmp_path, window=EARLY)
    bundle = load_bundle(out / "library")
    assert set(bundle.rows["outcome_type"]) == {"event_sequence", "market_resolution"}
    assert set(bundle.rows["event_id"]) <= set(bundle.events["event_id"])
    assert len(bundle.rows) == 5
    assert list(bundle.markets["market_id"]) == ["polymarket:0xa", "kalshi:KX-1"]
    assert list(bundle.markets.columns) == MARKET_COLUMNS
    assert manifest.built_from_commit == "0" * 40
    assert list(bundle.placebo_asset.columns) == [
        "outcome_key",
        "horizon",
        "value",
        "response_complete_session",
    ]


def test_market_referenced_by_a_kept_row_survives_outside_the_window(tmp_path, bundle_parts):
    rows = bundle_parts["rows"].copy()
    target = (rows["event_id"] == "gdelt:2") & (rows["outcome_type"] == "market_resolution")
    rows.loc[target, "source_ref"] = "kalshi:KX-2"
    write_bundle(tmp_path / "src" / "library", **{**bundle_parts, "rows": rows})
    out = tmp_path / "s"
    build_sample(tmp_path / "src" / "library", out, *EARLY, "SYNTHETIC", CITATION)
    markets = load_bundle(out / "library").markets
    assert "kalshi:KX-2" in set(markets["market_id"])
    assert list(markets.columns) == MARKET_COLUMNS


def test_sessions_are_trimmed_and_positions_still_index_them(tmp_path, bundle_root):
    out, _ = _sample(bundle_root, tmp_path, window=EARLY)
    bundle = load_bundle(out / "library")
    assert bundle.sessions[0] >= pd.Timestamp("2020-01-01")
    assert bundle.sessions[-1] <= pd.Timestamp("2020-04-30")
    assert len(bundle.session_cutoffs_utc) == len(bundle.sessions)
    last_needed = max(
        bundle.rows["response_complete_session"].max(),
        pd.Timestamp(max(bundle.events["effective_session"])),
    )
    assert bundle.sessions[-1] == last_needed
    assert bundle.sessions[0] == pd.Timestamp("2020-01-01") + pd.offsets.BDay(1)
    positions = bundle.events["effective_position"]
    assert bundle.sessions[positions].date.tolist() == bundle.events["effective_session"].tolist()


def test_since_after_until_is_a_value_error_and_writes_nothing(tmp_path, bundle_root):
    with pytest.raises(ValueError, match="since"):
        _sample(bundle_root, tmp_path, window=(date(2021, 3, 1), date(2021, 2, 1)))
    assert not (tmp_path / "s").exists()


def test_unknown_evidence_label_is_refused(tmp_path, bundle_root):
    with pytest.raises(ValueError, match="evidence_label"):
        build_sample(
            bundle_root / "library",
            tmp_path / "s",
            *EARLY,
            "MAYBE",
            CITATION,  # type: ignore[arg-type]
        )


def test_window_without_events_gives_an_empty_valid_bundle(tmp_path, bundle_root):
    out, manifest = _sample(bundle_root, tmp_path, window=(date(2019, 1, 1), date(2019, 3, 31)))
    bundle = load_bundle(out / "library")
    assert bundle.events.empty and bundle.rows.empty and bundle.markets.empty
    assert bundle.placebo_asset.empty
    assert manifest.sources == []
    lines = (out / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines == [lines[0], "", CITATION]


def test_missing_source_library_raises(tmp_path):
    with pytest.raises(LibraryMissingError):
        build_sample(tmp_path / "nope", tmp_path / "s", *EARLY, "SYNTHETIC", CITATION)
    assert not (tmp_path / "s").exists()


def test_rebuilding_replaces_the_previous_sample(tmp_path, bundle_root):
    _sample(bundle_root, tmp_path, window=EARLY)
    out, _ = _sample(bundle_root, tmp_path, window=(date(2021, 1, 1), date(2021, 2, 28)))
    assert load_bundle(out / "library").rows.empty
    assert sorted(path.name for path in out.iterdir()) == ["README.md", "library"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["library", "s"]
    assert MANIFEST_NAME in {path.name for path in (out / "library").iterdir()}


def test_script_builds_with_the_live_label(tmp_path, bundle_root, capsys):
    script = _load_script()
    out = tmp_path / "s"
    code = script.main(
        [
            "--data-dir", str(bundle_root),
            "--since", "2020-01-01",
            "--until", "2020-04-30",
            "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    first_line = (out / "README.md").read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("LIVE — GDELT 2.0 events")  # the label comes from the script
    assert capsys.readouterr().out.strip() == first_line
    readme = (out / "README.md").read_text(encoding="utf-8").splitlines()
    assert readme[2] == script.GDELT_CITATION


def test_script_exits_2_with_one_stderr_line_on_a_missing_library(tmp_path, capsys):
    script = _load_script()
    code = script.main(
        [
            "--data-dir", str(tmp_path),
            "--since", "2020-01-01",
            "--until", "2020-04-30",
            "--out", str(tmp_path / "s"),
        ]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    assert not (tmp_path / "s").exists()


def test_script_exits_2_when_the_sample_is_too_large(tmp_path, bundle_root, capsys, monkeypatch):
    script = _load_script()
    monkeypatch.setattr(script, "MAX_SAMPLE_BYTES", 1000)
    code = script.main(
        [
            "--data-dir", str(bundle_root),
            "--since", "2020-01-01",
            "--until", "2022-12-31",
            "--out", str(tmp_path / "s"),
        ]
    )  # fmt: skip
    assert code == 2 and "1000" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_script_refuses_since_after_until(tmp_path, bundle_root, capsys):
    script = _load_script()
    with pytest.raises(SystemExit) as stopped:
        script.main(
            [
                "--data-dir", str(bundle_root),
                "--since", "2020-05-01",
                "--until", "2020-04-30",
                "--out", str(tmp_path / "s"),
            ]
        )  # fmt: skip
    assert stopped.value.code == 2 and "--since" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_a_late_window_moves_effective_positions_with_the_trimmed_sessions(tmp_path, bundle_root):
    out, _ = _sample(bundle_root, tmp_path, window=(date(2020, 3, 1), date(2020, 7, 31)))
    bundle = load_bundle(out / "library")
    assert not bundle.events.empty
    assert bundle.events["effective_position"].min() >= 0
    assert bundle.sessions[0] >= pd.Timestamp("2020-03-01")
    assert (
        bundle.sessions[bundle.events["effective_position"]].date.tolist()
        == bundle.events["effective_session"].tolist()
    )
