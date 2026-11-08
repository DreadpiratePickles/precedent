"""Placebo graph for the mechanism ladder: the template's degrees, rewired at random."""

from __future__ import annotations

import numpy as np

from precedent.mechanism.graph_schema import CausalGraph, Edge, NodeKind, is_acyclic

_SWAPS_PER_EDGE = 10


class PlaceboError(ValueError):
    """A graph has no degree-preserving rewiring that differs from it."""


def _new_edge(source: str, target: str, lag_band: str) -> Edge:
    """A bare edge; the final signs are drawn after all swaps (see the shuffler)."""
    return Edge.model_validate(
        {
            "source": source,
            "target": target,
            "sign": 1,
            "lag_band": lag_band,
            "source_doc_id": None,
            "evidence_span": None,
            "support": 1,
        }
    )


def _swapped(
    edges: list[Edge], i: int, j: int, kinds: dict[str, NodeKind], node_ids: list[str]
) -> list[Edge] | None:
    """`edges` with (a->b, c->d) at positions i, j replaced by (a->d, c->b), or None if invalid."""
    (a, b), (c, d) = (edges[i].source, edges[i].target), (edges[j].source, edges[j].target)
    present = {(e.source, e.target) for e in edges}
    if a == d or c == b or (a, d) in present or (c, b) in present:
        return None
    if NodeKind.EVENT in (kinds[d], kinds[b]):
        return None
    if any(
        kinds[s] is NodeKind.ASSET and kinds[t] is not NodeKind.ASSET for s, t in ((a, d), (c, b))
    ):
        return None
    proposal = list(edges)
    proposal[i] = _new_edge(a, d, edges[i].lag_band)
    proposal[j] = _new_edge(c, b, edges[j].lag_band)
    return proposal if is_acyclic(node_ids, proposal) else None


def shuffle_edges_preserving_degree(graph: CausalGraph, seed: int) -> CausalGraph:
    """The degree-preserving placebo of `graph`: same nodes, same in- and out-degree per node.

    Double-edge swaps: pick two edges (a->b, c->d) and propose (a->d, c->b), kept only when
    neither proposed edge duplicates an existing one or is a self-loop, no asset causes a
    non-asset, nothing points into an event node (event variables are exogenous), and the
    graph stays acyclic. Each new edge keeps the lag band of the edge it replaced. The placebo
    also randomises signs: every edge gets a sign drawn uniformly from {-1, +1}, and carries
    no evidence (`source_doc_id` and `evidence_span` are None, `support` is 1).

    If the random walk ends back on the original edge set, the first valid single swap (in a
    seeded order) is applied instead. Raises `PlaceboError` when no valid swap exists at all
    (a single edge, or a star-shaped graph); the original is never returned silently.
    Same `seed`, same placebo.
    """
    rng = np.random.default_rng(seed)
    node_ids = [node.id for node in graph.nodes]
    kinds = {node.id: node.kind for node in graph.nodes}
    original = [_new_edge(e.source, e.target, e.lag_band) for e in graph.edges]
    original_pairs = {(e.source, e.target) for e in original}
    edges = original
    if len(edges) >= 2:
        for _ in range(_SWAPS_PER_EDGE * len(edges)):
            i, j = (int(k) for k in rng.choice(len(edges), size=2, replace=False))
            edges = _swapped(edges, i, j, kinds, node_ids) or edges
        if {(e.source, e.target) for e in edges} == original_pairs:
            ordered = rng.permutation(len(original) * len(original))
            for k in ordered:
                i, j = divmod(int(k), len(original))
                if i != j and (proposal := _swapped(original, i, j, kinds, node_ids)):
                    edges = proposal
                    break
    if {(e.source, e.target) for e in edges} == original_pairs:
        raise PlaceboError(
            f"graph {graph.graph_id!r} has no degree-preserving rewiring that differs from it "
            f"({len(graph.edges)} edges)"
        )
    signs = rng.choice([-1, 1], size=len(edges))
    edges = [e.model_copy(update={"sign": int(s)}) for e, s in zip(edges, signs, strict=True)]
    return CausalGraph(
        graph_id=f"{graph.graph_id}:placebo-{seed}",
        graph_version=f"placebo-{seed}",
        event_key=graph.event_key,
        nodes=list(graph.nodes),
        edges=edges,
    )
