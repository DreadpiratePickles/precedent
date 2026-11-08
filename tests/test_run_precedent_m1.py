"""The P-M1 run script, on a small synthetic bundle (never a live library)."""

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from precedent.engine.keys import key_from_flags
from precedent.libraries.bundle import PLACEBO_COLUMNS, write_bundle
from precedent.libraries.common import validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS
from precedent.libraries.manifest import ManifestSource
from precedent.libraries.market_resolution import MARKETS_FRAME_COLUMNS
from precedent.spine.pit_alignment import build_session_cutoffs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_precedent_m1.py"
SESSIONS = pd.bdate_range("2020-01-02", periods=400)
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))
_ONE_HOUR = pd.Timedelta(hours=1)
KEY_A = key_from_flags("central_bank", "central_bank", "high", False, "US")
KEY_B = key_from_flags("armed_conflict", "government", "high", False, "EM")
SEQUENCE_KEY = "armed_conflict/government/none/EM/high/unscheduled/0"
MARKET_KEY = "central_bank/central_bank/bond/US/high/scheduled/-1"
HOLDOUT_DAYS = 200
LAST_POSITION = 399


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("run_precedent_m1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positions():
    return [10 + 12 * n for n in range(30)]


def _events():
    records = []
    for n, position in enumerate(_positions()):
        key = KEY_A if n % 2 == 0 else KEY_B
        records.append(
            {
                "event_id": f"gdelt:{n}",
                "key": key.as_string(),
                "event_type": key.event_type,
                "country_group": key.country_group,
                "available_utc": CUTOFFS.iloc[position] - _ONE_HOUR,
                "effective_session": SESSIONS[position].date(),
                "effective_position": position,
                "primary_actor": "actor",
                "linked_countries": ["USA"],
                "story_cluster_id": f"gdelt:{n}",
                "is_first_report": True,
                "one_line": f"event {n}",
                "source": "gdelt",
                "source_ref": f"gdelt:{n}",
            }
        )
    frame = pd.DataFrame(records, columns=list(EVENTS_TABLE_COLUMNS))
    frame["available_utc"] = frame["available_utc"].astype("datetime64[ns, UTC]")
    return frame


def _row(n, outcome_type, outcome_key, horizon, value):
    position = _positions()[n]
    return {
        "event_id": f"gdelt:{n}",
        "outcome_type": outcome_type,
        "outcome_key": outcome_key,
        "horizon": horizon,
        "value": value,
        "is_binary": True,
        "effective_session": SESSIONS[position].date(),
        "response_complete_session": SESSIONS[position + horizon].date(),
        "source_ref": f"gdelt:{n}",
    }


def _rows():
    records = [_row(n, "event_sequence", SEQUENCE_KEY, 5, float(n % 2 == 0)) for n in range(30)]
    records += [_row(n, "market_resolution", MARKET_KEY, 3, float(n % 2)) for n in range(0, 30, 3)]
    return validate_library_rows(pd.DataFrame(records))


def _markets():
    created = pd.to_datetime(["2020-01-02T10:00:00Z"] * 2, utc=True).as_unit("ns")
    resolved = pd.to_datetime(["2020-02-10T15:00:00Z", "2020-03-02T15:00:00Z"], utc=True).as_unit(
        "ns"
    )
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["polymarket:0xa", "kalshi:KX-1"], dtype="str"),
            "outcome_key": pd.Series([MARKET_KEY] * 2, dtype="str"),
            "created_utc": pd.Series(created, dtype="datetime64[ns, UTC]"),
            "resolved_utc": pd.Series(resolved, dtype="datetime64[ns, UTC]"),
            "value": pd.Series([1.0, 0.0], dtype="float64"),
            "response_complete_session": pd.Series(
                [SESSIONS[27].as_unit("ns"), SESSIONS[44].as_unit("ns")], dtype="datetime64[ns]"
            ),
        }
    )
    return frame[list(MARKETS_FRAME_COLUMNS)]


def _write_library(root, rows):
    placebo = pd.DataFrame(
        {
            "outcome_key": pd.Series([], dtype="str"),
            "horizon": pd.Series([], dtype="int64"),
            "value": pd.Series([], dtype="float64"),
            "response_complete_session": pd.Series([], dtype="datetime64[ns]"),
        }
    )[list(PLACEBO_COLUMNS)]
    write_bundle(
        root / "library",
        events=_events(),
        rows=rows,
        placebo_asset=placebo,
        markets=_markets(),
        sessions=SESSIONS,
        session_cutoffs_utc=CUTOFFS,
        build_report={
            "events": {"records": 30, "kept": 30, "unaligned": 0},
            "markets": {"typed": 2, "untyped": 4},
            "note": "synthetic",
        },
        sources=[
            ManifestSource(
                name="gdelt2",
                first_date=dt.date(2020, 1, 16),
                last_date=dt.date(2021, 4, 1),
                rows=30,
                citation="synthetic gdelt citation",
            )
        ],
        commit="a" * 40,
    )
    return root


@pytest.fixture
def data_dir(tmp_path):
    return _write_library(tmp_path, _rows())


def _run(script, data_dir, tmp_path, run_id="t", extra=()):
    argv = [
        "--run-id",
        run_id,
        "--data-dir",
        str(data_dir),
        "--label",
        "SYNTHETIC",
        "--out-root",
        str(tmp_path / "out"),
        "--holdout-days",
        str(HOLDOUT_DAYS),
        *extra,
    ]
    return script.main(argv)


def _expected_holdout_ids():
    last = SESSIONS[LAST_POSITION]
    start = last - pd.Timedelta(days=HOLDOUT_DAYS)
    return {f"gdelt:{n}" for n, p in enumerate(_positions()) if SESSIONS[p] > start}


def test_run_writes_report_with_header_ladder_and_build_report(script, data_dir, tmp_path):
    assert _run(script, data_dir, tmp_path) == 0
    report = (tmp_path / "out" / "t" / "report.md").read_text(encoding="utf-8")
    lines = report.splitlines()
    assert lines[0] == f"SYNTHETIC — P-M1 report, library {'a' * 12}, run t"
    assert "## Build report" in lines
    assert "- events.records: 30" in lines
    assert "- markets.untyped: 4" in lines
    assert "## Event types where B1 improves most and least on B0" in lines
    assert "## Refusal counts per rung" in lines
    assert "### event_sequence, horizon 5" in lines
    assert "### market_resolution, horizon any" in lines
    assert "| B0 | " in report
    assert "| B1 | " in report
    assert "asset" not in "\n".join(line for line in lines if line.startswith("###"))


def test_only_holdout_events_are_scored_and_every_one_is_counted(script, data_dir, tmp_path):
    assert _run(script, data_dir, tmp_path) == 0
    results = json.loads((tmp_path / "out" / "t" / "results.json").read_text(encoding="utf-8"))
    holdout = _expected_holdout_ids()
    assert 0 < len(holdout) < 30
    assert results["holdout_events"] == len(holdout)
    sequence = next(
        t for t in results["tables"] if t["outcome_type"] == "event_sequence" and t["horizon"] == 5
    )
    for rung in ("B0", "B1"):
        total = sum(
            r["n_scored"] + r["n_refused"] for r in sequence["records"] if r["rung"] == rung
        )
        assert total == len(holdout)
    assert results["build_report"]["events"]["kept"] == 30


def test_asset_rows_get_quantile_tables_the_exploratory_note_and_a_coverage_line(script, tmp_path):
    asset = [
        {**_row(n, "asset", "SPY", 5, 0.01 * (n % 5) - 0.02), "is_binary": False} for n in range(30)
    ]
    root = _write_library(
        tmp_path / "with_assets", pd.concat([_rows(), validate_library_rows(pd.DataFrame(asset))])
    )
    assert _run(script, root, tmp_path) == 0
    report = (tmp_path / "out" / "t" / "report.md").read_text(encoding="utf-8")
    assert "### asset, horizon 5" in report
    assert "EXPLORATORY (survivorship-biased prices)" in report
    assert (
        "| rung | event_type | quantile | pinball_loss | coverage_80 | n_scored | n_refused |"
        in report
    )
    assert "B1 80% band coverage (weighted by n_scored): " in report
    assert "### asset, horizon 5 (median pinball loss)" in report
    results = json.loads((tmp_path / "out" / "t" / "results.json").read_text(encoding="utf-8"))
    table = next(t for t in results["tables"] if t["outcome_type"] == "asset")
    assert {r["quantile"] for r in table["records"]} == {0.1, 0.5, 0.9}
    assert all(r["brier"] is None for r in table["records"])


def test_horizons_without_rows_produce_no_table(script, data_dir, tmp_path):
    assert _run(script, data_dir, tmp_path) == 0
    results = json.loads((tmp_path / "out" / "t" / "results.json").read_text(encoding="utf-8"))
    shown = {(t["outcome_type"], t["horizon"]) for t in results["tables"]}
    assert shown == {("event_sequence", 5), ("market_resolution", "any")}


def test_missing_library_exits_2_with_one_stderr_line(script, tmp_path, capsys):
    code = script.main(
        [
            "--run-id",
            "t",
            "--data-dir",
            str(tmp_path / "empty"),
            "--label",
            "SYNTHETIC",
            "--out-root",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert len(captured.err.strip().splitlines()) == 1
    assert not (tmp_path / "out").exists()


def test_label_is_required_and_run_id_cannot_escape_the_output_root(script, data_dir, tmp_path):
    base = ["--data-dir", str(data_dir), "--out-root", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as no_label:
        script.main(["--run-id", "t", *base])
    assert no_label.value.code == 2
    with pytest.raises(SystemExit) as bad_id:
        script.main(["--run-id", "../x", "--label", "SYNTHETIC", *base])
    assert bad_id.value.code == 2
    assert not (tmp_path / "out").exists()


def test_script_names_no_url_model_or_cache_path():
    text = SCRIPT.read_text(encoding="utf-8")
    for needle in ("http://", "https://", ".cache/precedent", "claude-"):
        assert needle not in text
