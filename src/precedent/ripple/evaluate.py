"""Research: the Ripple graph ladder. Each graph's ripple factors are scored beside a baseline.

For every graph and every horizon this runs the spine ablation (`run_ablation`) on two rungs, the
baseline (price features alone, or price plus news) and the baseline plus that graph's `ripple_`
panel, and reports the rank IC of each. The caller passes the degree-preserving placebo of every
graph as a graph of its own (`<name>_placebo`), so a graph is never reported without its shuffle.
The result is history on past data, not a forecast.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
from qlib.data.dataset.loader import DataLoader

from precedent.ripple.graph_builders import (
    EVIDENCE_LABELS,
    GraphForCutoff,
    write_text_atomically,
)
from precedent.ripple.params import load_ripple_params
from precedent.ripple.ripple_factors import build_ripple_factor_panel
from precedent.spine.ablation import FoldSegments, Rung, load_rungs, run_ablation

RIPPLE_PANEL_KEY = "ripple_"
PLACEBO_SUFFIX = "_placebo"
PRICE_ONLY_RUNG = "m0_price"
NEWS_RUNG = "m1_news"
GRAPH_RUNG_PREFIX = "m3_ripple_"
LADDER_COLUMNS = [
    "graph",
    "horizon",
    "rank_ic_mean",
    "icir",
    "incremental_rank_ic_vs_m1",
    "baseline",
]
FOLD_COLUMNS = ["graph", "horizon", "fold", "rank_ic_mean"]
PAIR_COLUMNS = [
    "graph",
    "horizon",
    "rank_ic_mean",
    "placebo_rank_ic_mean",
    "margin_over_placebo",
    "tolerance",
    "beats_placebo",
]
FOLD_FILE = "ladder_folds.csv"
LADDER_FILE = "ladder.csv"
DEFAULT_HORIZONS = (1, 5, 20, 60)
LEAD_HORIZON = 20
_GRAPH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RESEARCH_CHECK_NOTE = (
    "Research check: published link effects are monthly; expect no effect at 1 and 5 sessions "
    f"in large caps. This report leads with {LEAD_HORIZON} sessions."
)

FoldsSource = Callable[[int], Sequence[FoldSegments]] | Sequence[FoldSegments]


def label_expr_for(horizon: int) -> str:
    """Decide at the close of t, enter at the open of t+1, exit at the open of t+horizon+1."""
    return f"Ref($open, -{horizon + 1})/Ref($open, -1) - 1"


def _validate_request(
    graphs: Mapping[str, GraphForCutoff],
    horizons: Sequence[int],
    baseline_panels: Mapping[str, pd.DataFrame],
) -> None:
    if not graphs:
        raise ValueError("evaluate_graph_ladder needs at least one graph")
    for name in graphs:
        if not _GRAPH_NAME.fullmatch(name):
            raise ValueError(
                f"graph name {name!r} must be letters, digits, '_', '.' or '-' (it names a "
                "directory)"
            )
    if not horizons:
        raise ValueError("evaluate_graph_ladder needs at least one horizon")
    bad = [h for h in horizons if isinstance(h, bool) or not isinstance(h, Integral) or h < 1]
    if bad:
        raise ValueError(f"horizons must be positive integers, got {bad}")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"horizons must be distinct, got {list(horizons)}")
    if RIPPLE_PANEL_KEY in baseline_panels:
        raise ValueError(
            f"baseline_panels may not use the key {RIPPLE_PANEL_KEY!r}; it is the graph's panel"
        )


def _graph_rung(
    name: str, baseline: Rung, panel_keys: Sequence[str], shipped: Mapping[str, Rung]
) -> Rung:
    """The shipped `m3_ripple_<name>` rung when this run can supply every prefix it names,
    otherwise the baseline's prefixes plus `ripple_`, built in memory under the same name."""
    rung_name = f"{GRAPH_RUNG_PREFIX}{name}"
    candidate = shipped.get(rung_name)
    if candidate is not None and all(
        any(key.startswith(prefix) for key in panel_keys) for prefix in candidate.panel_prefixes
    ):
        return candidate
    return Rung(name=rung_name, panel_prefixes=[*baseline.panel_prefixes, RIPPLE_PANEL_KEY])


def evaluate_graph_ladder(
    graphs: Mapping[str, GraphForCutoff],
    shocks: pd.Series,
    price_loader_for_horizon: Callable[[int], DataLoader],
    folds: FoldsSource,
    out_dir: Path,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    *,
    session_cutoffs_utc: pd.Series,
    sessions: pd.DatetimeIndex,
    abnormal_returns: pd.DataFrame,
    lgb_params: Mapping[str, Any],
    rungs_dir: Path,
    parser_version: str,
    baseline_panels: Mapping[str, pd.DataFrame] | None = None,
    alpha: float = 0.5,
    hops: int = 2,
    half_life_sessions: float = 3.0,
    max_age_sessions: int = 20,
) -> pd.DataFrame:
    """Score every graph's ripple factors at every horizon; one row per (graph, horizon).

    `graphs` maps a name to a `graph_for_cutoff` (the only way links are read). The ripple panel
    of each graph is built once (`build_ripple_factor_panel`) and reused across horizons. For
    horizon `h` the label is `Ref($open, -(h+1))/Ref($open, -1) - 1`; `price_loader_for_horizon(h)`
    must return a loader carrying that label, and the folds for it are `folds(h + 1)` (a plain
    sequence of folds is reused for every horizon). `run_ablation` raises when a fold's purge gap
    is shorter than the label's look-ahead.

    The baseline rung is `m1_news` when `baseline_panels` is given (its panels are then part of
    every run) and `m0_price` otherwise; the graph rung is `m3_ripple_<name>` (see `_graph_rung`).
    Every rung and trial is recorded in `<out_dir>/<name>_h<h>/run_manifest.json`. `rank_ic_mean`
    and `icir` are the means over folds; `incremental_rank_ic_vs_m1` is the graph rung's
    `rank_ic_mean` minus the baseline rung's, and `baseline` names the rung subtracted. The table
    is also written to `<out_dir>/ladder.csv`, and each graph rung's per-fold rank ICs to
    `<out_dir>/ladder_folds.csv` (see `load_fold_ics`). Raises ValueError for an empty graph set,
    a bad horizon or graph name, or a graph whose ripple panel is empty.
    """
    panels = dict(baseline_panels or {})
    _validate_request(graphs, horizons, panels)
    shipped = {rung.name: rung for rung in load_rungs(rungs_dir)}
    baseline_name = NEWS_RUNG if panels else PRICE_ONLY_RUNG
    if baseline_name not in shipped:
        raise ValueError(f"no rung named {baseline_name!r} in {rungs_dir}")
    baseline = shipped[baseline_name]
    ripple_panels: dict[str, pd.DataFrame] = {}
    for name, graph_for_cutoff in graphs.items():
        panel = build_ripple_factor_panel(
            shocks,
            graph_for_cutoff,
            session_cutoffs_utc,
            sessions,
            abnormal_returns,
            alpha,
            hops,
            half_life_sessions,
            max_age_sessions,
        )
        if panel.empty:
            raise ValueError(
                f"graph {name!r} produced no ripple factor rows (no shock reached a linked "
                "receiver); an empty panel would be scored as if the graph carried nothing"
            )
        ripple_panels[name] = panel
    rows: list[dict[str, Any]] = []
    fold_rows: list[pd.DataFrame] = []
    for horizon in horizons:
        label_expr = label_expr_for(horizon)
        loader = price_loader_for_horizon(horizon)
        fold_list = list(folds(horizon + 1) if callable(folds) else folds)
        for name, panel in ripple_panels.items():
            factor_panels = {**panels, RIPPLE_PANEL_KEY: panel}
            graph_rung = _graph_rung(name, baseline, list(factor_panels), shipped)
            result = run_ablation(
                [baseline, graph_rung],
                loader,
                factor_panels,
                fold_list,
                out_dir / f"{name}_h{horizon}",
                lgb_params,
                parser_version=parser_version,
                label_expr=label_expr,
                rungs_dir=rungs_dir,
            )
            means = result.groupby("rung")[["rank_ic_mean", "icir"]].mean()
            graph_folds = result.loc[result["rung"] == graph_rung.name, ["fold", "rank_ic_mean"]]
            fold_rows.append(graph_folds.assign(graph=name, horizon=horizon)[FOLD_COLUMNS])
            rows.append(
                {
                    "graph": name,
                    "horizon": horizon,
                    "rank_ic_mean": float(means.loc[graph_rung.name, "rank_ic_mean"]),
                    "icir": float(means.loc[graph_rung.name, "icir"]),
                    "incremental_rank_ic_vs_m1": float(
                        means.loc[graph_rung.name, "rank_ic_mean"]
                        - means.loc[baseline.name, "rank_ic_mean"]
                    ),
                    "baseline": baseline.name,
                }
            )
    table = pd.DataFrame(rows, columns=LADDER_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomically(out_dir / LADDER_FILE, table.to_csv(index=False))
    fold_table = pd.concat(fold_rows, ignore_index=True)
    write_text_atomically(out_dir / FOLD_FILE, fold_table.to_csv(index=False))
    return table


def load_fold_ics(out_dir: Path) -> pd.DataFrame:
    """The per-fold rank ICs (`graph, horizon, fold, rank_ic_mean`) a ladder run wrote."""
    path = out_dir / FOLD_FILE
    if not path.is_file():
        raise ValueError(f"no {FOLD_FILE} in {out_dir}; run evaluate_graph_ladder first")
    table = pd.read_csv(path)
    if list(table.columns) != FOLD_COLUMNS:
        raise ValueError(f"{path.name} columns must be {FOLD_COLUMNS}, got {list(table.columns)}")
    return table


def _report_horizon_order(horizons: Sequence[int]) -> list[int]:
    """The lead horizon first when present, then the rest from longest to shortest."""
    distinct = sorted(set(int(h) for h in horizons), reverse=True)
    return ([LEAD_HORIZON] if LEAD_HORIZON in distinct else []) + [
        h for h in distinct if h != LEAD_HORIZON
    ]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(headers))) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _number(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def diffusion_table(ladder: pd.DataFrame) -> str:
    """The ladder as two Markdown tables, one row per horizon (20 first), one column per graph.

    The first holds `rank_ic_mean`, the second `incremental_rank_ic_vs_m1`, each to 4 decimals.
    """
    if list(ladder.columns) != LADDER_COLUMNS:
        raise ValueError(f"ladder columns must be {LADDER_COLUMNS}, got {list(ladder.columns)}")
    graph_names = list(dict.fromkeys(ladder["graph"]))
    baselines = ", ".join(sorted(set(ladder["baseline"])))
    blocks = []
    for column, caption in (
        ("rank_ic_mean", "Rank IC, mean over folds, by horizon in sessions."),
        (
            "incremental_rank_ic_vs_m1",
            f"Rank IC added over the baseline ({baselines}): graph rung minus baseline rung.",
        ),
    ):
        wide = ladder.pivot(index="horizon", columns="graph", values=column)
        rows = [
            [str(horizon), *[_number(wide.loc[horizon].get(name)) for name in graph_names]]
            for horizon in _report_horizon_order(list(wide.index))
        ]
        blocks.append("\n".join([caption, "", *_markdown_table(["horizon", *graph_names], rows)]))
    return "\n\n".join(blocks) + "\n"


def _noise_parameters(margin_floor: float | None, se_multiple: float | None) -> tuple[float, float]:
    """The floor and standard-error multiple, from `references/ripple_params.json` unless given."""
    if margin_floor is not None and se_multiple is not None:
        return margin_floor, se_multiple
    params = load_ripple_params()
    return (
        params["placebo_margin_floor"] if margin_floor is None else margin_floor,
        params["placebo_margin_se_multiple"] if se_multiple is None else se_multiple,
    )


def _fold_difference(
    fold_ics: pd.DataFrame, name: str, placebo: str, horizon: int
) -> pd.Series:
    """Per-fold rank IC of `name` minus that of `placebo`, indexed by fold."""

    def folds_of(graph: str) -> pd.Series:
        rows = fold_ics[(fold_ics["graph"] == graph) & (fold_ics["horizon"] == horizon)]
        return rows.set_index("fold")["rank_ic_mean"]

    graph_ics, placebo_ics = folds_of(name), folds_of(placebo)
    if graph_ics.empty or set(graph_ics.index) != set(placebo_ics.index):
        raise ValueError(
            f"fold ICs of {name!r} and {placebo!r} at horizon {horizon} do not cover the same folds"
        )
    return (graph_ics - placebo_ics).dropna()


def placebo_pairs(
    ladder: pd.DataFrame,
    fold_ics: pd.DataFrame,
    *,
    margin_floor: float | None = None,
    se_multiple: float | None = None,
) -> pd.DataFrame:
    """Each graph beside its placebo (`<name>_placebo`), one row per (graph, horizon).

    `margin_over_placebo` is the graph's mean rank IC minus its placebo's. The `tolerance` is the
    larger of `margin_floor` and `se_multiple` times the standard error of the per-fold
    difference (graph minus placebo; standard deviation with ddof 1 over the square root of the
    fold count), and the floor alone with a single fold. `beats_placebo` is margin > tolerance.
    Both numbers come from `references/ripple_params.json` unless passed. `fold_ics` is the
    `graph, horizon, fold, rank_ic_mean` table a ladder run writes (`load_fold_ics`).

    Raises ValueError naming any graph that has no placebo row at some horizon: a graph is never
    reported without its shuffle. Graphs keep their order in `ladder`; horizons lead with 20.
    """
    floor, multiple = _noise_parameters(margin_floor, se_multiple)
    names = list(dict.fromkeys(ladder["graph"]))
    scores = ladder.set_index(["graph", "horizon"])["rank_ic_mean"]
    rows = []
    for name in (n for n in names if not n.endswith(PLACEBO_SUFFIX)):
        placebo = f"{name}{PLACEBO_SUFFIX}"
        horizons = ladder.loc[ladder["graph"] == name, "horizon"]
        missing = [h for h in horizons if (placebo, h) not in scores.index]
        if missing:
            raise ValueError(f"graph {name!r} has no {placebo!r} row at horizons {missing}")
        for horizon in _report_horizon_order(list(horizons)):
            differences = _fold_difference(fold_ics, name, placebo, horizon)
            standard_error = (
                float(differences.std(ddof=1)) / math.sqrt(len(differences))
                if len(differences) >= 2
                else 0.0
            )
            margin = float(scores[(name, horizon)] - scores[(placebo, horizon)])
            tolerance = max(floor, multiple * standard_error)
            rows.append(
                {
                    "graph": name,
                    "horizon": horizon,
                    "rank_ic_mean": float(scores[(name, horizon)]),
                    "placebo_rank_ic_mean": float(scores[(placebo, horizon)]),
                    "margin_over_placebo": margin,
                    "tolerance": tolerance,
                    "beats_placebo": margin > tolerance,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _verdict_lines(pairs: pd.DataFrame, floor: float, multiple: float) -> list[str]:
    """The headline, then one line per (graph, horizon) with its margin, tolerance and verdict."""
    winners = pairs[pairs["beats_placebo"]]
    if winners.empty:
        headline = ["No graph beats its placebo at any horizon."]
    else:
        headline = []
        for name in dict.fromkeys(pairs["graph"]):
            won = [str(h) for h in winners.loc[winners["graph"] == name, "horizon"]]
            headline.append(
                f"{name} beats its placebo at horizons {', '.join(won)}."
                if won
                else f"{name} does not beat its placebo at any horizon."
            )
    lines = [
        *headline,
        "",
        f"A graph beats its placebo at a horizon only when its margin exceeds the tolerance: the "
        f"larger of {floor:.4f} rank IC and {multiple:g} standard errors of the per-fold "
        "difference (the floor alone with one fold).",
        "",
    ]
    for row in pairs.itertuples():
        if row.beats_placebo:
            verdict = "beats its placebo"
        elif row.margin_over_placebo < -row.tolerance:
            verdict = "within noise (placebo ahead of the graph by more than the tolerance)"
        else:
            verdict = "within noise"
        lines.append(
            f"- {row.graph} at {row.horizon} sessions: margin {row.margin_over_placebo:+.4f}, "
            f"tolerance {row.tolerance:.4f}: {verdict}."
        )
    return lines


def render_report(
    ladder: pd.DataFrame,
    *,
    fold_ics: pd.DataFrame,
    evidence_label: str,
    counts: Mapping[str, Any],
    notes: Sequence[str] = (),
    margin_floor: float | None = None,
    se_multiple: float | None = None,
) -> str:
    """The R-M1 report: evidence line, research check, the tables, placebo pairs, counts.

    The first line begins `LIVE —` for `LIVE-DATA` and `SYNTHETIC —` for `SYNTHETIC`. Raises
    ValueError for another label or when a graph has no placebo row. The verdict per graph and
    horizon follows `placebo_pairs`: a win needs a margin above the noise tolerance.
    """
    if evidence_label not in EVIDENCE_LABELS:
        raise ValueError(f"evidence_label must be one of {EVIDENCE_LABELS}, got {evidence_label!r}")
    floor, multiple = _noise_parameters(margin_floor, se_multiple)
    pairs = placebo_pairs(ladder, fold_ics, margin_floor=floor, se_multiple=multiple)
    first_line = (
        "LIVE — graph ladder on live prices and 8-K events; exploratory, derived statistics only."
        if evidence_label == "LIVE-DATA"
        else "SYNTHETIC — graph ladder on generated data; not evidence about real markets."
    )
    pair_rows = [
        [
            row.graph,
            str(row.horizon),
            _number(row.rank_ic_mean),
            _number(row.placebo_rank_ic_mean),
            _number(row.margin_over_placebo),
        ]
        for row in pairs.itertuples()
    ]
    lines = [
        first_line,
        "",
        "# Ripple graph ladder (milestone R-M1)",
        "",
        f"Label: {evidence_label}. EXPLORATORY (survivorship-biased prices)",
        *(
            [
                "",
                "G1 uses one static sector map for the whole run, so sector membership looks "
                "ahead (it is not dated).",
            ]
            if "g1" in set(ladder["graph"])
            else []
        ),
        "",
        RESEARCH_CHECK_NOTE,
        "",
        "## Rank IC by horizon",
        "",
        diffusion_table(ladder).rstrip("\n"),
        "",
        "## Each graph beside its placebo",
        "",
        *_markdown_table(
            ["graph", "horizon", "rank IC", "placebo rank IC", "margin"], pair_rows
        ),
        "",
        *_verdict_lines(pairs, floor, multiple),
        "",
        "## Counts",
        "",
        *[f"- {name}: {value}" for name, value in counts.items()],
    ]
    if notes:
        lines.extend(["", "## Run", "", *[f"- {note}" for note in notes]])
    return "\n".join(lines) + "\n"
