"""Template graphs: hand-written textbook DAGs per event type, loaded and validated, no model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from precedent.mechanism.graph_schema import CausalGraph, Edge, Node, NodeKind, derive_kind
from precedent.spine.event_key import EventKey

TEMPLATE_VERSION = "template-v1"


class TemplateError(ValueError):
    """A template file that cannot become a valid graph; the message names the file."""


class TemplateEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str
    sign: Literal[-1, 1]
    lag_band: Literal["same_session", "days", "weeks"]


class TemplateFile(BaseModel):
    """The wrapper around one template's edges; `rationale` is documentation, not data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: str
    graph_version: Literal["template-v1"]
    rationale: str
    edges: list[TemplateEdge]


def _placeholder_key(event_type: str) -> EventKey:
    return EventKey(
        event_type=event_type,
        actor_kind="other",
        impacted_kinds=(),
        country_group="none",
        severity_band="mid",
        is_scheduled=False,
        direction_sign=0,
    )


def _build_graph(path: Path, vocabulary: frozenset[str]) -> CausalGraph:
    try:
        wrapper = TemplateFile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise TemplateError(f"template {path.name}: unreadable or malformed: {error}") from error
    if path.stem != wrapper.event_type:
        raise TemplateError(
            f"template {path.name}: file name must equal event_type {wrapper.event_type!r}"
        )
    event_id = f"event:{wrapper.event_type}:direction_sign"
    node_ids = [event_id]
    for edge in wrapper.edges:
        node_ids.extend((edge.source, edge.target))
    node_ids.extend(sorted(i for i in vocabulary if derive_kind(i) is NodeKind.ASSET))
    node_ids = list(dict.fromkeys(node_ids))
    outside = [i for i in node_ids if i not in vocabulary]
    if outside:
        raise TemplateError(f"template {path.name}: node ids outside the vocabulary: {outside}")
    try:
        return CausalGraph(
            graph_id=f"template:{wrapper.event_type}",
            graph_version=wrapper.graph_version,
            event_key=_placeholder_key(wrapper.event_type),
            nodes=[Node(id=i, kind=derive_kind(i)) for i in node_ids],
            edges=[
                Edge(
                    source=e.source,
                    target=e.target,
                    sign=e.sign,
                    lag_band=e.lag_band,
                    source_doc_id=None,
                    evidence_span=None,
                )
                for e in wrapper.edges
            ],
        )
    except ValidationError as error:
        raise TemplateError(f"template {path.name}: invalid graph: {error}") from error


def load_templates(templates_dir: Path, vocabulary: frozenset[str]) -> dict[str, CausalGraph]:
    """Load every `<event_type>.json` in `templates_dir`; the first bad file stops the load."""
    if not templates_dir.is_dir():
        raise TemplateError(f"templates directory {templates_dir} does not exist")
    paths = sorted(templates_dir.glob("*.json"))
    if not paths:
        raise TemplateError(f"templates directory {templates_dir} contains no *.json files")
    return {path.stem: _build_graph(path, vocabulary) for path in paths}


def template_for(key: EventKey, templates: dict[str, CausalGraph]) -> CausalGraph | None:
    """The template for `key.event_type` carrying `key`, or None when there is none."""
    graph = templates.get(key.event_type)
    if graph is None:
        return None
    return CausalGraph.model_validate(graph.model_copy(update={"event_key": key}).model_dump())
