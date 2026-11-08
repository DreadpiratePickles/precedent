"""Append-only store of causal graphs: one JSON Lines file per graph version."""

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from precedent.mechanism.graph_schema import CausalGraph
from precedent.spine.event_key import EventKey
from precedent.spine.jsonl_io import AppendReport, atomic_write_lines, merge_keyed, read_models

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class GraphConflictError(Exception):
    """A graph with the same (event key, graph_version) but different content."""


def _slug(graph_version: str) -> str:
    return _SLUG_UNSAFE.sub("_", graph_version)


class GraphStore:
    """Graphs keyed by (event key, graph_version), stored as `graphs_<slug>.jsonl` under root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, graph_version: str) -> Path:
        """The JSON Lines file that holds one graph version's graphs."""
        return self._root / f"graphs_{_slug(graph_version)}.jsonl"

    def append(self, graphs: Iterable[CausalGraph]) -> AppendReport:
        """Add new graphs; skip identical ones; refuse changed ones. All or nothing."""
        by_version: dict[str, list[CausalGraph]] = defaultdict(list)
        for graph in graphs:
            by_version[graph.graph_version].append(graph)
        self._refuse_slug_collisions(by_version)

        merged_by_version: dict[str, list[CausalGraph]] = {}
        added = skipped = 0
        for version, incoming in by_version.items():
            merged, version_added, version_skipped = merge_keyed(
                self.read(version),
                incoming,
                lambda graph: graph.event_key.as_string(),
                lambda graph, version=version: GraphConflictError(
                    f"graph for key {graph.event_key.as_string()!r} already stored for "
                    f"version {version!r} with different content"
                ),
            )
            skipped += version_skipped
            if version_added:
                merged_by_version[version] = merged
                added += version_added
        for version, merged in merged_by_version.items():
            atomic_write_lines(
                self.path_for(version), (graph.model_dump_json() for graph in merged)
            )
        return AppendReport(added=added, identical_skipped=skipped)

    def read(self, graph_version: str) -> list[CausalGraph]:
        """Every graph stored for one version, in append order."""
        graphs = read_models(self.path_for(graph_version), CausalGraph)
        for graph in graphs:
            if graph.graph_version != graph_version:
                raise ValueError(
                    f"{self.path_for(graph_version).name}: graph version "
                    f"{graph.graph_version!r} shares the file slug of {graph_version!r}"
                )
        return graphs

    def for_key(self, event_key: EventKey, graph_version: str) -> CausalGraph | None:
        """The graph for an event key at one version, or None."""
        for graph in self.read(graph_version):
            if graph.event_key == event_key:
                return graph
        return None

    def _refuse_slug_collisions(self, by_version: dict[str, list[CausalGraph]]) -> None:
        seen: dict[str, str] = {}
        for version in by_version:
            other = seen.setdefault(_slug(version), version)
            if other != version:
                raise ValueError(
                    f"graph versions {other!r} and {version!r} share the file slug "
                    f"{_slug(version)!r}"
                )
