"""P-M1: score the baseline ladder (B0, B1) on the most recent year of a built library.

    uv run python scripts/run_precedent_m1.py --run-id <id> --label LIVE [--data-dir PATH]

Loads the bundle under `<data dir>/library` and holds out the events whose effective session
falls within the last `--holdout-days` days before the library's last session. Every event
still serves as a precedent through the time guard; only hold-out events are scored. B0 and B1
are scored for every outcome type present, at each of its horizons, on paired units only: a unit
either rung refuses is scored by neither, and refusals are reported per rung (P-46).
Writes `report.md` and `results.json` under `<out-root>/<run-id>/`. Exit 0 on success, 2 when
there is no library.

This script reads local files only; it fetches nothing and calls no model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from precedent import config
from precedent.engine.evaluate import RUNGS, score_precedent_ladder
from precedent.engine.params import load_engine_params
from precedent.libraries.bundle import LibraryBundle, LibraryMissingError, load_bundle
from precedent.libraries.common import OUTCOME_TYPES
from precedent.libraries.manifest import LibraryIntegrityError

LABELS = ("SYNTHETIC", "LIVE")
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = _REPO_ROOT / "stages" / "21_precedent_evaluate" / "output"
ANY_HORIZON = "any"
TOP_N = 10
COVERAGE_BAND = (0.70, 0.90)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BUILD_REPORT_FILE = "build_report.json"
_EXPLORATORY = "EXPLORATORY (survivorship-biased prices)"
_CONTINUOUS_COLUMNS = ("rung", "event_type", "quantile", "pinball_loss", "coverage_80")
_BINARY_COLUMNS = ("rung", "event_type", "brier")
_COUNT_COLUMNS = ("n_scored", "n_refused")


def _run_id(text: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(text) is None or ".." in text:
        raise argparse.ArgumentTypeError(
            f"run id {text!r} must be letters, digits, '.', '_' or '-', starting with a letter "
            "or digit"
        )
    return text


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_precedent_m1",
        description="Score B0 and B1 on the held-out last year of a built library (P-M1).",
    )
    parser.add_argument("--run-id", required=True, type=_run_id)
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PRECEDENT_DATA_DIR")
    parser.add_argument("--holdout-days", type=int, default=365)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args(argv)
    if args.holdout_days < 1:
        parser.error("--holdout-days must be at least 1")
    return args


def _holdout_mask(bundle: LibraryBundle, holdout_days: int) -> pd.Series:
    """Events with `effective_session` after `last session - holdout_days`, aligned to events."""
    start = bundle.sessions[-1] - pd.Timedelta(days=holdout_days)
    effective = pd.to_datetime(bundle.events["effective_session"])
    return (effective > start).astype(bool)


def _horizons(outcome_type: str) -> list[int | str]:
    params = load_engine_params()
    if outcome_type == "asset":
        return list(params.asset_horizons)
    if outcome_type == "event_sequence":
        return list(params.sequence_horizons)
    return [ANY_HORIZON]


def score_holdout(bundle: LibraryBundle, holdout: pd.Series) -> list[dict[str, Any]]:
    """One entry per (outcome type, horizon) with a non-empty table; full pool, hold-out scored."""
    scored: list[dict[str, Any]] = []
    for outcome_type in OUTCOME_TYPES:
        if not (bundle.rows["outcome_type"] == outcome_type).any():
            continue
        for horizon in _horizons(outcome_type):
            call_horizon = horizon if isinstance(horizon, int) else 5
            table = score_precedent_ladder(
                bundle.events,
                bundle.rows,
                bundle.session_cutoffs_utc,
                outcome_type,
                rungs=RUNGS,
                horizon=call_horizon,
                score_only=holdout,
            )
            if table.empty:
                continue
            scored.append({"outcome_type": outcome_type, "horizon": horizon, "table": table})
    return scored


def _units(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (rung, event_type): the count columns repeat across quantiles."""
    return table.drop_duplicates(["rung", "event_type"])


def _improvement(table: pd.DataFrame, is_continuous: bool) -> pd.DataFrame:
    """`B1 - B0` on median pinball loss or Brier per event type; negative means B1 is better."""
    if is_continuous:
        metric_table, metric = table.loc[table["quantile"] == 0.5], "pinball_loss"
    else:
        metric_table, metric = table, "brier"
    wide = metric_table.pivot(index="event_type", columns="rung", values=metric)
    if not {"B0", "B1"} <= set(wide.columns):
        return pd.DataFrame(columns=["event_type", "B0", "B1", "delta"])
    wide = wide.dropna(subset=["B0", "B1"])
    out = wide[["B0", "B1"]].assign(delta=wide["B1"] - wide["B0"]).reset_index()
    return out.sort_values(["delta", "event_type"], kind="stable").reset_index(drop=True)


def _b1_weighted_coverage(table: pd.DataFrame) -> float | None:
    b1 = _units(table.loc[(table["rung"] == "B1") & table["coverage_80"].notna()])
    total = int(b1["n_scored"].sum())
    if total == 0:
        return None
    return float((b1["coverage_80"] * b1["n_scored"]).sum() / total)


def _cell(value: object) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    shown = (*columns, *_COUNT_COLUMNS)
    lines = ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    for record in frame.loc[:, list(shown)].itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in record) + " |")
    return lines


def _flatten(prefix: str, value: object) -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _flatten(f"{prefix}.{key}" if prefix else str(key), value[key])
    elif isinstance(value, list):
        if all(not isinstance(item, Mapping | list) for item in value):
            yield prefix, value
        else:
            for index, item in enumerate(value):
                yield from _flatten(f"{prefix}[{index}]", item)
    else:
        yield prefix, value


def _heading(entry: Mapping[str, Any]) -> str:
    return f"{entry['outcome_type']}, horizon {entry['horizon']}"


def _ladder_section(entry: Mapping[str, Any]) -> list[str]:
    table: pd.DataFrame = entry["table"]
    is_continuous = entry["outcome_type"] == "asset"
    lines = [f"### {_heading(entry)}", ""]
    if is_continuous:
        lines += [_EXPLORATORY, ""]
    columns = _CONTINUOUS_COLUMNS if is_continuous else _BINARY_COLUMNS
    lines += _markdown_table(table, columns)
    if is_continuous:
        coverage = _b1_weighted_coverage(table)
        low, high = COVERAGE_BAND
        if coverage is None:
            verdict = "no scored B1 rows"
        else:
            inside = "inside" if low <= coverage <= high else "OUTSIDE"
            verdict = f"{coverage:.3f}, {inside} the {low:.0%} to {high:.0%} band"
        lines += ["", f"B1 80% band coverage (weighted by n_scored): {verdict}"]
    return [*lines, ""]


def _refusal_section(scored: list[dict[str, Any]]) -> list[str]:
    lines = ["## Refusal counts per rung", ""]
    lines.append("| outcome_type | horizon | rung | n_scored | n_refused |")
    lines.append("|---|---|---|---|---|")
    for entry in scored:
        for rung, group in _units(entry["table"]).groupby("rung", sort=True):
            n_scored, n_refused = int(group["n_scored"].sum()), int(group["n_refused"].sum())
            label = f"{entry['outcome_type']} | {entry['horizon']} | {rung}"
            lines.append(f"| {label} | {n_scored} | {n_refused} |")
    return [*lines, ""]


def _build_report_section(build_report: Mapping[str, Any]) -> list[str]:
    lines = ["## Build report", "", f"Every entry of `{_BUILD_REPORT_FILE}`.", ""]
    for name, value in _flatten("", build_report):
        lines.append(f"- {name}: {json.dumps(value, default=str)}")
    return [*lines, ""]


def _improvement_section(scored: list[dict[str, Any]]) -> list[str]:
    lines = ["## Event types where B1 improves most and least on B0", ""]
    lines.append(
        "Difference B1 minus B0 in median pinball loss (asset) or Brier (binary types); "
        "negative means B1 is better. Computed from the tables above."
    )
    lines.append("")
    for entry in scored:
        is_continuous = entry["outcome_type"] == "asset"
        metric = "median pinball loss" if is_continuous else "Brier"
        diff = _improvement(entry["table"], is_continuous)
        lines += [f"### {_heading(entry)} ({metric})", ""]
        if diff.empty:
            lines += ["No event type was scored on both rungs.", ""]
            continue
        for title, part in (
            ("Most improved", diff.head(TOP_N)),
            ("Least improved", diff.iloc[::-1].head(TOP_N)),
        ):
            lines += [f"**{title}**", "", "| event_type | B0 | B1 | B1 minus B0 |"]
            lines.append("|---|---|---|---|")
            for row in part.itertuples(index=False):
                cells = (row.event_type, _cell(row.B0), _cell(row.B1), _cell(row.delta))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return lines


def render_report(
    header: str,
    context: Mapping[str, object],
    scored: list[dict[str, Any]],
    build_report: Mapping[str, Any],
) -> str:
    lines = [header, "", "History of past outcomes, not a forecast.", ""]
    lines += ["## Run", ""]
    lines += [f"- {key}: {value}" for key, value in context.items()]
    lines += ["", "## Ladder scores on the hold-out set", ""]
    if not scored:
        lines += ["No outcome type had a scored hold-out event.", ""]
    for entry in scored:
        lines += _ladder_section(entry)
    lines += _refusal_section(scored)
    lines += _build_report_section(build_report)
    lines += _improvement_section(scored)
    return "\n".join(lines).rstrip("\n") + "\n"


def _json_safe(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(table: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: _json_safe(item) for key, item in record.items()}
        for record in table.to_dict(orient="records")
    ]


def _write_atomic(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    data_dir = args.data_dir if args.data_dir is not None else config.data_dir()
    library_dir = data_dir / "library"
    try:
        bundle = load_bundle(library_dir)
        build_report = json.loads((library_dir / _BUILD_REPORT_FILE).read_text(encoding="utf-8"))
    except LibraryMissingError:
        print(f"no library under {library_dir}; run precedent build first", file=sys.stderr)
        return 2
    except LibraryIntegrityError as error:
        print(f"library under {library_dir} failed its integrity check: {error}", file=sys.stderr)
        return 1
    holdout = _holdout_mask(bundle, args.holdout_days)
    scored = score_holdout(bundle, holdout)
    commit = bundle.manifest.built_from_commit
    context: dict[str, object] = {
        "run_id": args.run_id,
        "label": args.label,
        "library_commit": commit,
        "library_dir": library_dir,
        "last_session": bundle.sessions[-1].date(),
        "holdout_days": args.holdout_days,
        "holdout_events": int(holdout.sum()),
        "total_events": len(bundle.events),
    }
    header = f"{args.label} — P-M1 report, library {commit[:12]}, run {args.run_id}"
    report = render_report(header, context, scored, build_report)
    results = {
        "run_id": args.run_id,
        "label": args.label,
        "library_commit": commit,
        "holdout_days": args.holdout_days,
        "holdout_events": int(holdout.sum()),
        "tables": [
            {
                "outcome_type": entry["outcome_type"],
                "horizon": entry["horizon"],
                "records": _records(entry["table"]),
            }
            for entry in scored
        ],
        "build_report": build_report,
    }
    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(out_dir / "report.md", report)
    _write_atomic(out_dir / "results.json", json.dumps(results, indent=2, default=str) + "\n")
    print(f"wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
