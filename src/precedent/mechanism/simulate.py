"""Monte Carlo rollouts of a fitted causal graph, with do-interventions and counterfactual pairs.

Every number is fitted or sampled from fitted residuals; the simulator invents none. The draws
come from a counter-based hash (`hash_uniform`, `hash_normals`) rather than numpy's Generator so
the browser cockpit can reproduce them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from precedent.mechanism.fit_sem import FittedSEM
from precedent.mechanism.graph_schema import CausalGraph, NodeKind, descendants

DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_MASK32 = 0xFFFFFFFF
_SEED_MULTIPLIER = 0x9E3779B1
_STREAM_XOR = 0x6A09E667
_MAX_UNIFORM_INDEX = 1 << 32


def _mix32(x: np.ndarray) -> np.ndarray:
    """Murmur3 fmix32 on a uint32 array; the multiplications wrap modulo 2**32."""
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x85EBCA6B)
    x = x ^ (x >> np.uint32(13))
    x = x * np.uint32(0xC2B2AE35)
    return x ^ (x >> np.uint32(16))


def hash_uniform(seed: int, index: np.ndarray) -> np.ndarray:
    """Uniforms in the open interval (0, 1) from a counter-based hash.

    The browser cockpit reimplements this function and `hash_normals`; the integer steps must
    stay bit-for-bit identical to it. With `index` a uint32 array and all arithmetic mod 2**32:
    `h = fmix32(fmix32(seed * 0x9E3779B1 + index) ^ 0x6A09E667)`, then `u = (h + 0.5) / 2**32`.
    """
    base = np.uint32((seed & _MASK32) * _SEED_MULTIPLIER & _MASK32)
    counter = np.asarray(index, dtype=np.uint32)
    hashed = _mix32(_mix32(counter + base) ^ np.uint32(_STREAM_XOR))
    return (hashed.astype(np.float64) + 0.5) / 4294967296.0


def hash_normals(seed: int, n_paths: int, n_assets: int) -> np.ndarray:
    """Standard normals (n_paths x n_assets) by Box-Muller, cosine branch only.

    The normal of path i, asset j has k = i * n_assets + j and uses the uniforms at indices 2k
    and 2k + 1: `z = sqrt(-2 ln u1) * cos(2 pi u2)`. Same JS-twin rule as `hash_uniform`.
    """
    count = n_paths * n_assets
    if 2 * count > _MAX_UNIFORM_INDEX:
        raise ValueError(f"{n_paths} paths x {n_assets} assets overflows the 32-bit counter")
    k = np.arange(count, dtype=np.uint32)
    u1 = hash_uniform(seed, k * np.uint32(2))
    u2 = hash_uniform(seed, k * np.uint32(2) + np.uint32(1))
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z.reshape(n_paths, n_assets)


@dataclass(frozen=True)
class Intervention:
    """do(node = value): the node is set on every path and its incoming edges are cut."""

    node: str
    value: float


@dataclass(frozen=True)
class Rollout:
    """Simulated asset outcomes, with the seed and interventions that produced them."""

    paths: np.ndarray
    assets: list[str]
    seed: int
    interventions: tuple[Intervention, ...]
    unreachable_interventions: list[str]
    diagonal_fallback: bool
    n_paths: int


def _cholesky_factor(cov: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cholesky factor of the covariance, or its clamped diagonal when not positive definite."""
    try:
        return np.linalg.cholesky(cov), False
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0))), True


def _check_inputs(
    fitted: FittedSEM,
    graph: CausalGraph,
    event_values: dict[str, float],
    interventions: tuple[Intervention, ...],
    n_paths: int,
) -> None:
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1; got {n_paths}")
    if fitted.graph_id != graph.graph_id:
        raise ValueError(f"fit is for graph_id {fitted.graph_id!r}, not {graph.graph_id!r}")
    node_ids = {node.id for node in graph.nodes}
    for intervention in interventions:
        if intervention.node not in node_ids:
            raise ValueError(f"intervention on {intervention.node!r}: not a node of the graph")
    forced = {i.node for i in interventions}
    for event_node in graph.event_node_ids:
        if event_node not in event_values and event_node not in forced:
            raise ValueError(f"event_values has no value for event node {event_node!r}")


def _unreachable(graph: CausalGraph, interventions: tuple[Intervention, ...]) -> list[str]:
    assets = {node.id for node in graph.nodes if node.kind is NodeKind.ASSET}
    return [i.node for i in interventions if not (descendants(graph, i.node) & assets)]


def _evaluate(
    fitted: FittedSEM,
    graph: CausalGraph,
    event_values: dict[str, float],
    forced: dict[str, float],
    residuals: np.ndarray,
) -> np.ndarray:
    """Topological evaluation over all paths; a forced node ignores its own equation."""
    kinds = {node.id: node.kind for node in graph.nodes}
    column = {asset: j for j, asset in enumerate(fitted.assets)}
    n_paths = residuals.shape[0]
    values: dict[str, np.ndarray] = {}
    for node in graph.topological_order():
        if node in forced:
            values[node] = np.full(n_paths, forced[node], dtype=float)
        elif kinds[node] is NodeKind.EVENT:
            values[node] = np.full(n_paths, event_values[node], dtype=float)
        else:
            value = np.full(n_paths, fitted.intercepts[node], dtype=float)
            for parent, coefficient in fitted.coefficients[node].items():
                value += coefficient * values[parent]
            if kinds[node] is NodeKind.ASSET:
                value += residuals[:, column[node]]
            values[node] = value
    return np.column_stack([values[asset] for asset in fitted.assets])


def _rollout(
    fitted: FittedSEM,
    graph: CausalGraph,
    event_values: dict[str, float],
    interventions: tuple[Intervention, ...],
    residuals: np.ndarray,
    seed: int,
    diagonal_fallback: bool,
) -> Rollout:
    forced = {i.node: i.value for i in interventions}
    return Rollout(
        paths=_evaluate(fitted, graph, event_values, forced, residuals),
        assets=list(fitted.assets),
        seed=seed,
        interventions=interventions,
        unreachable_interventions=_unreachable(graph, interventions),
        diagonal_fallback=diagonal_fallback,
        n_paths=residuals.shape[0],
    )


def _draw_residuals(fitted: FittedSEM, n_paths: int, seed: int) -> tuple[np.ndarray, bool]:
    factor, fallback = _cholesky_factor(fitted.residual_cov)
    z = hash_normals(seed, n_paths, len(fitted.assets))
    return z @ factor.T, fallback


def simulate(
    fitted: FittedSEM,
    graph: CausalGraph,
    event_values: dict[str, float],
    interventions: tuple[Intervention, ...] = (),
    n_paths: int = 10_000,
    seed: int = 0,
) -> Rollout:
    """Roll the fitted graph forward over `n_paths` paths; deterministic given `seed`."""
    _check_inputs(fitted, graph, event_values, interventions, n_paths)
    residuals, fallback = _draw_residuals(fitted, n_paths, seed)
    return _rollout(fitted, graph, event_values, interventions, residuals, seed, fallback)


def fan(rollout: Rollout, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> pd.DataFrame:
    """Per-asset quantiles of the simulated outcomes: index asset, one column per quantile."""
    table = np.quantile(rollout.paths, quantiles, axis=0).T
    return pd.DataFrame(table, index=rollout.assets, columns=list(quantiles))


def counterfactual_pair(
    fitted: FittedSEM,
    graph: CausalGraph,
    event_values: dict[str, float],
    intervention: Intervention,
    n_paths: int = 10_000,
    seed: int = 0,
) -> tuple[Rollout, Rollout]:
    """Factual and intervened rollouts on the same residuals; their difference is the effect."""
    _check_inputs(fitted, graph, event_values, (intervention,), n_paths)
    residuals, fallback = _draw_residuals(fitted, n_paths, seed)
    factual = _rollout(fitted, graph, event_values, (), residuals, seed, fallback)
    intervened = _rollout(fitted, graph, event_values, (intervention,), residuals, seed, fallback)
    return factual, intervened
