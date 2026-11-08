"""Causal graph schema: a closed node vocabulary, signed lagged edges, and a DAG check."""

from __future__ import annotations

import json
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from precedent.config import references_dir
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType

_EVENT_PREFIX = "event:"
_EVENT_FIELDS = ("severity_band", "direction_sign", "is_scheduled")
_TEMPLATE_PREFIX = "template-"


class NodeKind(StrEnum):
    EVENT = "event"
    INTERMEDIATE = "intermediate"
    ASSET = "asset"


class MechanismParams(BaseModel):
    """Mechanism parameters read from `references/mechanism/params.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    notes: dict[str, str]
    n_paths: int
    asset_panel: list[str]


def load_params(path: Path | None = None) -> MechanismParams:
    """Read and validate the mechanism parameters; the default path is the shipped reference."""
    target = path or references_dir() / "mechanism" / "params.json"
    return MechanismParams.model_validate_json(target.read_text(encoding="utf-8"))


def node_vocabulary(
    intermediates_path: Path, asset_panel: list[str], event_types: list[str]
) -> frozenset[str]:
    """Every node id a graph may use: event ids, listed intermediates (expanded) and assets."""
    ids: set[str] = set(asset_panel)
    for event_type in event_types:
        ids.add(f"{_EVENT_PREFIX}{event_type}")
        ids.update(f"{_EVENT_PREFIX}{event_type}:{name}" for name in _EVENT_FIELDS)
    entries = json.loads(intermediates_path.read_text(encoding="utf-8"))["intermediates"]
    for entry in entries:
        placeholder = entry.get("placeholder")
        if placeholder is None:
            ids.add(entry["id"])
        else:
            ids.update(entry["id"].replace(f"<{placeholder}>", v) for v in entry["values"])
    return frozenset(ids)


@cache
def _default_asset_panel() -> frozenset[str]:
    return frozenset(load_params().asset_panel)


@cache
def _default_vocabulary() -> frozenset[str]:
    return node_vocabulary(
        references_dir() / "mechanism" / "intermediates.json",
        sorted(_default_asset_panel()),
        [member.value for member in EventType],
    )


def derive_kind(node_id: str) -> NodeKind:
    """The kind an id implies: `event:` prefix, else asset panel member, else intermediate."""
    if node_id.startswith(_EVENT_PREFIX):
        return NodeKind.EVENT
    if node_id in _default_asset_panel():
        return NodeKind.ASSET
    return NodeKind.INTERMEDIATE


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: NodeKind


class Edge(BaseModel):
    """One signed, lagged cause-to-effect link, with where in a document it was read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str
    sign: Literal[-1, 1]
    lag_band: Literal["same_session", "days", "weeks"]
    source_doc_id: str | None
    evidence_span: str | None
    support: int = 1


def _kahn_order(node_ids: list[str], edges: list[Edge]) -> list[str]:
    """Topological order by Kahn's algorithm, ties broken by node order; partial if cyclic."""
    in_degree = dict.fromkeys(node_ids, 0)
    children: dict[str, list[str]] = {node: [] for node in node_ids}
    for edge in edges:
        in_degree[edge.target] += 1
        children[edge.source].append(edge.target)
    ready = [node for node in node_ids if in_degree[node] == 0]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in children[current]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    return order


def _event_type_of(node_id: str) -> str:
    return node_id.removeprefix(_EVENT_PREFIX).split(":")[0]


class CausalGraph(BaseModel):
    """A DAG from an event to the assets it may move, for one event key and graph version."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str
    graph_version: str
    event_key: EventKey
    nodes: list[Node]
    edges: list[Edge]

    @model_validator(mode="after")
    def _validate_graph(self) -> CausalGraph:
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node id")
        vocabulary = _default_vocabulary()
        unknown = [i for i in ids if i not in vocabulary]
        if unknown:
            raise ValueError(f"node ids outside the vocabulary: {unknown}")
        for node in self.nodes:
            if node.kind is not derive_kind(node.id):
                raise ValueError(
                    f"node {node.id!r} has kind {node.kind}, expected the derived kind"
                )
        known = set(ids)
        kinds = {node.id: node.kind for node in self.nodes}
        seen_pairs: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"edge {edge.source!r} -> {edge.target!r} has an endpoint not in nodes"
                )
            if (edge.source, edge.target) in seen_pairs:
                raise ValueError(f"duplicate edge {edge.source!r} -> {edge.target!r}")
            seen_pairs.add((edge.source, edge.target))
            if kinds[edge.source] is NodeKind.ASSET and kinds[edge.target] is not NodeKind.ASSET:
                raise ValueError(f"asset {edge.source!r} cannot cause non-asset {edge.target!r}")
        self._check_events()
        if len(_kahn_order(ids, self.edges)) < len(ids):
            raise ValueError("graph has a cycle")
        if not self.graph_version.startswith(_TEMPLATE_PREFIX):
            for edge in self.edges:
                if not edge.source_doc_id or not (edge.evidence_span or "").strip():
                    raise ValueError(
                        f"edge {edge.source!r} -> {edge.target!r} needs source_doc_id and an "
                        "evidence span in a graph that is not a template"
                    )
        return self

    def _check_events(self) -> None:
        events = self.event_node_ids
        if not events:
            raise ValueError("graph needs at least one event node")
        wrong = [i for i in events if _event_type_of(i) != self.event_key.event_type]
        if wrong:
            raise ValueError(
                f"event nodes {wrong} do not name event_type {self.event_key.event_type!r}"
            )

    @property
    def event_node_ids(self) -> list[str]:
        """Event-kind node ids, in node order."""
        return [node.id for node in self.nodes if node.kind is NodeKind.EVENT]

    @property
    def event_node_id(self) -> str:
        """The first event node."""
        return self.event_node_ids[0]

    def topological_order(self) -> list[str]:
        """Node ids with every cause before its effects."""
        return _kahn_order([node.id for node in self.nodes], self.edges)

    def parents(self, node_id: str) -> list[str]:
        """Sources of the edges that point at `node_id`, in edge order."""
        return [edge.source for edge in self.edges if edge.target == node_id]


def graph_json_schema() -> dict:
    """The JSON Schema of a causal graph; `references/mechanism/graph_schema_v1.json` copies it."""
    return CausalGraph.model_json_schema()


def descendants(graph: CausalGraph, node_id: str) -> set[str]:
    """Every node reachable from `node_id` along edges, excluding `node_id` itself."""
    children: dict[str, list[str]] = {}
    for edge in graph.edges:
        children.setdefault(edge.source, []).append(edge.target)
    reached: set[str] = set()
    frontier = [node_id]
    while frontier:
        for child in children.get(frontier.pop(), []):
            if child not in reached:
                reached.add(child)
                frontier.append(child)
    return reached
