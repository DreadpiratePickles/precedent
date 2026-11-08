"""`precedent mechanism`: fit a causal graph for an event key and print or save the what-if.

This module needs only the spine. The data step (`_fit_and_band`) is the one place that reaches
for the Precedent engine, and it imports it lazily, so a checkout without the engine still
reports `not built yet` instead of failing at import.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from precedent.config import ConfigError, data_dir, references_dir
from precedent.mechanism.cockpit import (
    BandRow,
    MechanismResult,
    build_result,
    render_html,
    render_json,
    render_text,
)
from precedent.mechanism.fit_sem import FittedSEM
from precedent.mechanism.graph_schema import CausalGraph, _default_vocabulary, load_params
from precedent.mechanism.simulate import Intervention
from precedent.mechanism.templates import load_templates, template_for
from precedent.spine.event_key import EventKey
from precedent.spine.typers.select import typer_from_env

USAGE_ERROR = 2
DEFAULT_HORIZON = 5
_SEVERITY_LEVEL = {"low": 0.0, "mid": 1.0, "high": 2.0}
_MISSING_ENGINE = "not built yet: mechanism needs the Precedent engine (Precedent Tasks 2 to 8)"
_PRECEDENT_PACKAGES = ("precedent.engine", "precedent.libraries")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent mechanism",
        description="A fitted model of past co-movement under an assumed structure.",
    )
    parser.add_argument("text", nargs="?", help="event text; typed by PRECEDENT_TYPER (rules)")
    parser.add_argument("--event-type")
    parser.add_argument("--actor-kind")
    parser.add_argument("--severity-band", choices=("low", "mid", "high"), default="mid")
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--scheduled", dest="scheduled", action="store_true")
    schedule.add_argument("--unscheduled", dest="scheduled", action="store_false")
    parser.set_defaults(scheduled=False)
    parser.add_argument("--direction", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--country-group", choices=("US", "DM", "EM", "none"), default="none")
    parser.add_argument("--impacted-kinds", default="", help="comma-separated impact kinds")
    parser.add_argument("--asof", help="YYYY-MM-DD; default today (UTC)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--do", action="append", default=[], metavar="NODE=VALUE")
    parser.add_argument("--html", type=Path, metavar="PATH")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-paths", type=int)
    return parser


def parse_do_flags(flags: list[str], graph: CausalGraph) -> tuple[Intervention, ...]:
    """`NODE=VALUE` strings to interventions on intermediate or asset nodes with a finite value."""
    node_ids = {node.id for node in graph.nodes}
    event_ids = set(graph.event_node_ids)
    parsed = []
    for flag in flags:
        node, separator, raw = flag.rpartition("=")
        if not separator or not node:
            raise ValueError(f"--do {flag!r}: expected NODE=VALUE")
        if node not in node_ids:
            raise ValueError(f"--do {flag!r}: {node!r} is not a node of the graph")
        if node in event_ids:
            raise ValueError(
                f"--do {flag!r}: {node!r} is an event variable, which is set by its key flag "
                "(for example --direction), not by do()"
            )
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"--do {flag!r}: {raw!r} is not a number") from error
        if not math.isfinite(value):
            raise ValueError(f"--do {flag!r}: {raw!r} is not a finite number")
        parsed.append(Intervention(node, value))
    return tuple(parsed)


def _key_from_flags(args: argparse.Namespace) -> EventKey:
    kinds = tuple(sorted(k for k in args.impacted_kinds.split(",") if k))
    key = EventKey(
        event_type=args.event_type,
        actor_kind=args.actor_kind or "other",
        impacted_kinds=kinds,
        country_group=args.country_group,
        severity_band=args.severity_band,
        is_scheduled=args.scheduled,
        direction_sign=args.direction,
    )
    return EventKey.from_string(key.as_string())  # rejects an unknown type, actor or kind


def _key_from_text(text: str) -> EventKey | None:
    typed = typer_from_env().type_text(text)
    if typed.status == "typed" and typed.key is not None:
        return typed.key
    print(
        f"cannot type that text ({typed.status}: {typed.reason}).\n"
        "Pass the key yourself: precedent mechanism --event-type <type> --actor-kind <kind> "
        "--severity-band <band> --unscheduled --direction <sign>",
        file=sys.stderr,
    )
    return None


def event_values_for(key: EventKey, graph: CausalGraph) -> dict[str, float]:
    """A number for each event node of the graph, read off the key."""
    values: dict[str, float] = {}
    for node_id in graph.event_node_ids:
        field = node_id.rsplit(":", 1)[-1]
        if field == "direction_sign":
            values[node_id] = float(key.direction_sign)
        elif field == "severity_band":
            values[node_id] = _SEVERITY_LEVEL[key.severity_band]
        elif field == "is_scheduled":
            values[node_id] = 1.0 if key.is_scheduled else 0.0
        else:
            raise ValueError(f"event node {node_id!r} has no known value encoding")
    return values


def _fit_and_band(
    key: EventKey, graph: CausalGraph, asof_utc: datetime, horizon: int, data_dir: Path
) -> tuple[FittedSEM, list[BandRow]]:
    """Fit the graph on the guarded panel and read Precedent's band for the same key.

    Task 7b fills this body: `build_guarded_panel`, `fit_sem`, and Precedent's
    `summarize_responses` into `BandRow`s. Below Precedent's `min_neff` the command refuses
    (7b). The imports stay inside the function so this module imports with the spine only.
    """
    from precedent.engine import retrieval  # noqa: F401  (7b: replaced by the real imports)

    raise NotImplementedError("Task 7b: the data step is not written yet")


def _code_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
            cwd=Path(__file__).resolve().parent,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def _parse_asof(raw: str | None) -> datetime:
    day = date.fromisoformat(raw) if raw else datetime.now(UTC).date()
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _resolve_graph(key: EventKey) -> CausalGraph | None:
    templates = load_templates(references_dir() / "mechanism" / "templates", _default_vocabulary())
    return template_for(key, templates)


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return USAGE_ERROR


def _result_for(args: argparse.Namespace, key: EventKey, graph: CausalGraph) -> MechanismResult:
    interventions = parse_do_flags(args.do, graph)
    asof_utc = _parse_asof(args.asof)
    directory = args.data_dir or data_dir()
    fitted, bands = _fit_and_band(key, graph, asof_utc, args.horizon, directory)
    return build_result(
        graph=graph,
        fitted=fitted,
        event_values=event_values_for(key, graph),
        interventions=interventions,
        precedent=bands,
        seed=args.seed,
        n_paths=load_params().n_paths if args.n_paths is None else args.n_paths,
        asof_utc=asof_utc,
        code_commit=_code_commit(),
    )


def run(argv: list[str]) -> int:
    """The `precedent mechanism` entry point; returns the process exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    if (args.text is None) == (args.event_type is None):
        return _usage_error(
            "precedent mechanism: give either the event text or --event-type, not both"
        )
    try:
        key = _key_from_text(args.text) if args.text is not None else _key_from_flags(args)
        if key is None:
            return USAGE_ERROR
        graph = _resolve_graph(key)
        if graph is None:
            return _usage_error(f"no template graph for event type {key.event_type}")
        parse_do_flags(args.do, graph)  # fail on a bad --do before any data is read
        result = _result_for(args, key, graph)
    except ConfigError as error:
        return _usage_error(str(error))
    except NotImplementedError as error:
        return _usage_error(str(error))
    except ValueError as error:
        return _usage_error(f"precedent mechanism: {error}")
    except ModuleNotFoundError as error:
        if (error.name or "").startswith(_PRECEDENT_PACKAGES):
            return _usage_error(_MISSING_ENGINE)
        raise
    if args.html is not None:
        _write_atomically(args.html, render_html(result))
    print(render_json(result) if args.as_json else render_text(result))
    return 0
