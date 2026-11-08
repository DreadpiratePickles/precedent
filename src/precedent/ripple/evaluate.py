"""Research: the Ripple graph ladder. Each graph's ripple factors are scored beside a baseline.

For every graph and every horizon this runs the spine ablation (`run_ablation`) on two rungs, the
baseline (price features alone, or price plus news) and the baseline plus that graph's `ripple_`
panel, and reports the rank IC of each. The caller passes the degree-preserving placebo of every
graph as a graph of its own (`<name>_placebo`), so a graph is never reported without its shuffle.
The result is history on past data, not a forecast. The report half (tables, placebo pairs,
verdict lines) lives in `precedent.ripple.ladder_report`, which needs no research package.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
from qlib.data.dataset.loader import DataLoader

from precedent.ripple.graph_builders import GraphForCutoff, write_text_atomically
from precedent.ripple.ladder_report import FOLD_COLUMNS, LADDER_COLUMNS
from precedent.ripple.ripple_factors import build_ripple_factor_panel
from precedent.spine.ablation import FoldSegments, Rung, load_rungs, run_ablation

RIPPLE_PANEL_KEY = "ripple_"
PRICE_ONLY_RUNG = "m0_price"
NEWS_RUNG = "m1_news"
GRAPH_RUNG_PREFIX = "m3_ripple_"
FOLD_FILE = "ladder_folds.csv"
LADDER_FILE = "ladder.csv"
DEFAULT_HORIZONS = (1, 5, 20, 60)
_GRAPH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

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
