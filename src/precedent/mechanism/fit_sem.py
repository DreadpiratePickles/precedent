"""Structural equation fitting: the numbers on a causal graph, read from a guarded panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from precedent.mechanism.graph_schema import CausalGraph, NodeKind, descendants

_ZERO_TOL = 1e-12
_MIN_ROWS_BEYOND_PARENTS = 2


@dataclass(frozen=True)
class GuardedPanel:
    """The per-precedent observations a fit may see: event variables and abnormal returns."""

    event_values: pd.DataFrame
    asset_returns: pd.DataFrame
    horizon: int
    asof_utc: pd.Timestamp

    def __post_init__(self) -> None:
        if self.asof_utc.tzinfo is None:
            raise ValueError("asof_utc must be tz-aware UTC; got a naive timestamp")
        if self.asof_utc.utcoffset() != timedelta(0):
            raise ValueError(f"asof_utc must be in UTC; got offset {self.asof_utc.utcoffset()}")
        if not self.event_values.index.equals(self.asset_returns.index):
            raise ValueError("event_values and asset_returns must have the same index")
        if self.event_values.index.has_duplicates:
            raise ValueError("duplicate precedent ids in the panel index")
        if len(self.event_values) < 1:
            raise ValueError("a guarded panel needs at least one row")

    @property
    def n_obs(self) -> int:
        """Number of precedent events in the panel."""
        return len(self.event_values)


@dataclass(frozen=True)
class FittedSEM:
    """Fitted equations for one graph.

    `residual_cov` is over `assets`, in that order, with each asset's variance on n - slopes - 1
    degrees of freedom. It is estimated pairwise, so it may not be positive definite; the
    simulator owns the Cholesky-or-diagonal fallback.

    An identified latent (intermediate) node is evaluated like any other, as
    `intercepts[node] + sum(coefficients[node][p] * parent_p)`. Its intercept is minus the
    weighted panel-mean parent value, so that value is the centred score its children's
    equations were fitted on; an unidentified latent has intercept 0.0 and no coefficients.

    `node_ranges` maps every event and intermediate node to the 5th and 95th percentile of its
    values over the fitted panel (a latent's is its centred score); the cockpit's sliders run
    over it. A node with no observed value is absent.
    """

    coefficients: dict[str, dict[str, float]]
    intercepts: dict[str, float]
    residual_cov: np.ndarray
    assets: list[str]
    n_obs: int
    underdetermined: frozenset[str]
    sign_disagreements: tuple[tuple[str, str], ...]
    graph_id: str
    graph_version: str
    fitted_asof_utc: pd.Timestamp
    horizon: int
    node_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Plain floats, strings and nested lists, ready for `json.dumps`."""
        return {
            "coefficients": {n: dict(c) for n, c in self.coefficients.items()},
            "intercepts": dict(self.intercepts),
            "residual_cov": self.residual_cov.tolist(),
            "assets": list(self.assets),
            "n_obs": self.n_obs,
            "underdetermined": sorted(self.underdetermined),
            "sign_disagreements": [list(pair) for pair in self.sign_disagreements],
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "fitted_asof_utc": self.fitted_asof_utc.isoformat(),
            "horizon": self.horizon,
            "node_ranges": {node: list(pair) for node, pair in self.node_ranges.items()},
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> FittedSEM:
        """Rebuild a fit from `to_json_dict` output; a naive timestamp or bad shape is an error."""
        asof = pd.Timestamp(payload["fitted_asof_utc"])
        if asof.tzinfo is None:
            raise ValueError("fitted_asof_utc must be tz-aware")
        assets = [str(a) for a in payload["assets"]]
        cov = np.array(payload["residual_cov"], dtype=float).reshape(len(assets), len(assets))
        return cls(
            coefficients={n: dict(c) for n, c in payload["coefficients"].items()},
            intercepts=dict(payload["intercepts"]),
            residual_cov=cov,
            assets=assets,
            n_obs=int(payload["n_obs"]),
            underdetermined=frozenset(payload["underdetermined"]),
            sign_disagreements=tuple((s, t) for s, t in payload["sign_disagreements"]),
            graph_id=payload["graph_id"],
            graph_version=payload["graph_version"],
            fitted_asof_utc=asof.tz_convert("UTC"),
            horizon=int(payload["horizon"]),
            node_ranges={
                node: (float(pair[0]), float(pair[1]))
                for node, pair in payload.get("node_ranges", {}).items()
            },
        )


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge slopes (p x d) and free intercepts (d) of y on x; the penalty skips the intercept."""
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_centred, y_centred = x - x_mean, y - y_mean
    gram = x_centred.T @ x_centred + ridge * np.eye(x.shape[1])
    slopes = np.linalg.lstsq(gram, x_centred.T @ y_centred, rcond=None)[0]
    return slopes, y_mean - x_mean @ slopes


def _anchor_flips_weights(
    graph: CausalGraph, node: str, weights: np.ndarray, child_loadings: dict[str, float]
) -> bool:
    """Whether the weights must flip so the template's edge signs read the right way round."""
    first_in = next(edge for edge in graph.edges if edge.target == node)
    if abs(weights[0]) > _ZERO_TOL:  # parents() follows edge order, so the first parent is index 0
        return (weights[0] > 0) != (first_in.sign > 0)
    for edge in graph.edges:
        loading = child_loadings.get(edge.target, 0.0)
        if edge.source == node and abs(loading) > _ZERO_TOL:
            return (loading > 0) != (edge.sign > 0)
    return False


def _identify_latent(
    graph: CausalGraph,
    node: str,
    series: dict[str, np.ndarray],
    observed_assets: list[str],
    ridge: float,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Weights, intercept and per-row score of an unobserved node; None when not identifiable.

    A row joins the identification when every parent is observed and at least one child is;
    requiring all children on one row would throw away history. Each child is regressed on
    the rows where that child is observed, and a child seen on fewer than parents + 2 rows
    takes no part in the identification. Because each child is fitted on its own rows, the
    ridge shrinkage differs between children and the SVD weighs their slope columns fitted
    on different sample sizes; accepted as a disclosed approximation. The score is centred
    on the identifying rows' parent mean, and the intercept is -(parent mean @ weights), so
    intercept + sum(weight * parent) reproduces the score exactly.
    """
    parents = graph.parents(node)
    reached = descendants(graph, node)
    children = [a for a in observed_assets if a in reached]
    if not parents or not children:
        return None
    parent_matrix = np.column_stack([series[p] for p in parents])
    parent_ok = ~np.isnan(parent_matrix).any(axis=1)
    min_rows = len(parents) + _MIN_ROWS_BEYOND_PARENTS
    columns: list[np.ndarray] = []
    identified_children: list[str] = []
    identifying = np.zeros(parent_matrix.shape[0], dtype=bool)
    for child in children:
        mask = parent_ok & ~np.isnan(series[child])
        if mask.sum() < min_rows:
            continue  # this child takes no part in the identification
        slopes, _ = _ridge(parent_matrix[mask], series[child][mask, None], ridge)
        columns.append(slopes[:, 0])
        identified_children.append(child)
        identifying |= mask
    if not columns:
        return None
    slope_matrix = np.column_stack(columns)
    left, singular, right_t = np.linalg.svd(slope_matrix, full_matrices=False)
    if singular[0] == 0.0:
        return None
    weights = left[:, 0]
    loadings = dict(zip(identified_children, singular[0] * right_t[0], strict=True))
    if _anchor_flips_weights(graph, node, weights, loadings):
        weights = -weights
    intercept = -float(parent_matrix[identifying].mean(axis=0) @ weights)
    score = parent_matrix @ weights + intercept
    return weights, intercept, score


def _fit_asset(
    parents: list[str], series: dict[str, np.ndarray], node: str, ridge: float
) -> tuple[dict[str, float], float, np.ndarray, bool]:
    """Coefficients, intercept, residuals (NaN where unobserved) and the underdetermined flag."""
    target = series[node]
    design = np.column_stack([series[p] for p in parents]) if parents else None
    usable = ~np.isnan(target)
    if design is not None:
        usable &= ~np.isnan(design).any(axis=1)
    residuals = np.full(len(target), np.nan)
    count = int(usable.sum())
    if parents and count >= len(parents) + _MIN_ROWS_BEYOND_PARENTS:
        slopes, intercept = _ridge(design[usable], target[usable, None], ridge)
        residuals[usable] = target[usable] - design[usable] @ slopes[:, 0] - intercept[0]
        coefs = {p: float(b) for p, b in zip(parents, slopes[:, 0], strict=True)}
        return coefs, float(intercept[0]), residuals, False
    intercept = float(target[usable].mean()) if count else 0.0
    residuals[usable] = target[usable] - intercept
    return {}, intercept, residuals, bool(parents) or count == 0


def _dof_scale(residuals: np.ndarray, slope_count: int) -> float:
    """Factor turning the n - 1 denominator into n - slopes - 1; 1.0 when no dof remain."""
    observed = int(np.count_nonzero(~np.isnan(residuals)))
    remaining = observed - slope_count - 1
    return float(np.sqrt((observed - 1) / remaining)) if remaining >= 1 else 1.0


def _residual_covariance(residuals: np.ndarray) -> np.ndarray:
    """Pairwise-complete sample covariance (a x a); a pair with under two common rows is 0."""
    size = residuals.shape[0]
    cov = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            both = ~np.isnan(residuals[i]) & ~np.isnan(residuals[j])
            if both.sum() < 2:
                continue
            a, b = residuals[i, both], residuals[j, both]
            cov[i, j] = cov[j, i] = float(
                ((a - a.mean()) * (b - b.mean())).sum() / (both.sum() - 1)
            )
    return cov


def _column(frame: pd.DataFrame, name: str, what: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"panel has no {what} column for node {name!r}")
    return frame[name].to_numpy(dtype=float)


def fit_sem(graph: CausalGraph, panel: GuardedPanel, ridge: float = 1.0) -> FittedSEM:
    """Fit every non-event node on the panel, in topological order.

    The ridge penalty is scale-free: event columns are unit-scale enums and latent weights are
    unit-norm. A slope is shrunk by the factor n*var/(n*var + ridge) of its centred column, so
    the default 1.0 shrinks a unit-variance column's slope by n/(n + 1).

    `underdetermined` holds every node whose equation could not be fitted (too little data,
    a singular latent, no observations) and, propagated downstream, every node with an
    underdetermined parent: an asset below an unidentified latent is fitted against a zero
    score and must be read as uninformative. The propagation passes through observed
    intermediates and also flags a node that has a second, identified path from the event:
    that is conservative for rollouts, where the event's effect through the broken path is
    lost, at the cost of over-flagging the fit itself.
    """
    if ridge < 0:
        raise ValueError(f"ridge must not be negative; got {ridge}")
    kinds = {node.id: node.kind for node in graph.nodes}
    for edge in graph.edges:
        if kinds[edge.target] is NodeKind.EVENT:
            raise ValueError(
                f"edge {edge.source!r} -> {edge.target!r}: "
                "an event node cannot have parents in a fitted graph"
            )
    assets = [node.id for node in graph.nodes if node.kind is NodeKind.ASSET]
    series = {a: _column(panel.asset_returns, a, "asset return") for a in assets}
    observed_assets = [a for a in assets if not np.isnan(series[a]).all()]
    coefficients: dict[str, dict[str, float]] = {}
    intercepts: dict[str, float] = {}
    residuals: dict[str, np.ndarray] = {}
    underdetermined: set[str] = set()
    for node in graph.topological_order():
        kind = kinds[node]
        if kind is NodeKind.EVENT:
            series[node] = _column(panel.event_values, node, "event")
        elif kind is NodeKind.INTERMEDIATE:
            latent = _identify_latent(graph, node, series, observed_assets, ridge)
            if latent is None:
                coefficients[node], series[node] = {}, np.zeros(panel.n_obs)
                intercepts[node] = 0.0
                underdetermined.add(node)
            else:
                weights, intercepts[node], series[node] = latent
                coefficients[node] = dict(
                    zip(graph.parents(node), map(float, weights), strict=True)
                )
        else:
            parents = graph.parents(node)
            coefs, intercepts[node], residuals[node], flagged = _fit_asset(
                parents, series, node, ridge
            )
            coefficients[node] = coefs
            residuals[node] *= _dof_scale(residuals[node], len(coefs))
            if flagged:
                underdetermined.add(node)
    disagreements = tuple(
        (edge.source, edge.target)
        for edge in graph.edges
        if (coef := coefficients[edge.target].get(edge.source, 0.0)) != 0.0
        and (coef > 0) != (edge.sign > 0)
    )
    # Propagate the flag downstream: parents come first in topological order, so one pass
    # flags every node with an underdetermined parent (an asset below an unidentified latent
    # is fitted against a zero score and is uninformative).
    for node in graph.topological_order():
        if node not in underdetermined and any(
            parent in underdetermined for parent in graph.parents(node)
        ):
            underdetermined.add(node)
    node_ranges = {
        node: (float(np.nanpercentile(series[node], 5)), float(np.nanpercentile(series[node], 95)))
        for node in graph.topological_order()
        if kinds[node] is not NodeKind.ASSET and not np.isnan(series[node]).all()
    }
    residual_matrix = np.vstack([residuals[a] for a in assets]) if assets else np.empty((0, 0))
    return FittedSEM(
        coefficients=coefficients,
        intercepts=intercepts,
        residual_cov=_residual_covariance(residual_matrix),
        assets=assets,
        n_obs=panel.n_obs,
        underdetermined=frozenset(underdetermined),
        sign_disagreements=disagreements,
        graph_id=graph.graph_id,
        graph_version=graph.graph_version,
        fitted_asof_utc=panel.asof_utc,
        horizon=panel.horizon,
        node_ranges=node_ranges,
    )
