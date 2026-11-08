"""The R-M1 report: the rank-IC table, each graph beside its placebo, and the verdict lines.

Split from `precedent.ripple.evaluate` (which runs the ladder and needs Qlib and LightGBM): this
half only turns the ladder and per-fold ICs into Markdown, so it uses core dependencies only
(pandas). A graph is never reported without its degree-preserving placebo: `placebo_pairs`
raises for a graph with no `<name>_placebo` row, and `render_report` calls it. The result is
history on past data, not a forecast.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from precedent.ripple.graph_builders import EVIDENCE_LABELS
from precedent.ripple.params import load_ripple_params

PLACEBO_SUFFIX = "_placebo"
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
LEAD_HORIZON = 20
RESEARCH_CHECK_NOTE = (
    "Research check: published link effects are monthly; expect no effect at 1 and 5 sessions "
    f"in large caps. This report leads with {LEAD_HORIZON} sessions."
)


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
        "difference (the floor alone with one fold). "
        f"{len(pairs)} graph-versus-placebo comparisons were made at this rule; a single win "
        "among many comparisons is weak evidence.",
        "",
    ]
    for row in pairs.itertuples():
        if row.beats_placebo:
            verdict = "beats its placebo"
        elif row.margin_over_placebo < -row.tolerance:
            verdict = "below its placebo (beyond the tolerance)"
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
