"""The dated link store: links between companies, each valid only from its document's date.

Links come only from dated documents. `graph_asof` is the only way to read them, so a caller
cannot see a link before the document that states it was accepted, or after it expired.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from precedent.spine.jsonl_io import atomic_write_lines, read_models

GRAPH_COLUMNS = [
    "source_node",
    "receiver_node",
    "relation",
    "weight",
    "valid_from_utc",
    "valid_to_utc",
    "source_doc_id",
    "evidence_span",
    "extractor_version",
]
RULES_EXTRACTOR_PREFIX = "rules-"


class Relation(StrEnum):
    """The kind of link; the channel it feeds is decided in the factor stage."""

    SUPPLIER_OF = "supplier_of"
    CUSTOMER_OF = "customer_of"
    PEER_OF = "peer_of"
    INPUT_COMMODITY = "input_commodity"
    OUTPUT_COMMODITY = "output_commodity"
    COUNTRY_EXPOSURE = "country_exposure"


class Link(BaseModel):
    """One dated link. Timestamps are tz-aware; a model-extracted link must quote its evidence."""

    model_config = ConfigDict(extra="forbid")
    source_node: str = Field(min_length=1)
    receiver_node: str = Field(min_length=1)
    relation: Relation
    weight: float = Field(ge=0.0, le=1.0)
    valid_from_utc: AwareDatetime
    valid_to_utc: AwareDatetime
    source_doc_id: str
    evidence_span: str | None
    extractor_version: str

    @field_validator("valid_from_utc", "valid_to_utc")
    @classmethod
    def _normalise_to_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_window_and_evidence(self) -> Self:
        if self.source_node == self.receiver_node:
            raise ValueError("a link cannot join a node to itself")
        if self.valid_to_utc <= self.valid_from_utc:
            raise ValueError("valid_to_utc must be after valid_from_utc")
        needs_evidence = not self.extractor_version.startswith(RULES_EXTRACTOR_PREFIX)
        if needs_evidence and not (self.evidence_span and self.evidence_span.strip()):
            raise ValueError(
                f"extractor_version {self.extractor_version!r} is not rule-based, "
                "so evidence_span must be non-empty"
            )
        return self


@dataclass(frozen=True)
class AppendReport:
    """What an append did: links written, and links already stored byte-identically."""

    appended: int
    duplicates_skipped: int


class LinkStore:
    """Append-only JSON Lines store at `root / "links.jsonl"`."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "links.jsonl"

    def read_all(self) -> list[Link]:
        """Every stored link in file order; a store that does not exist yet reads as empty.

        Factor code must read links through `graph_asof` only, never through this method.
        """
        return read_models(self.path, Link)

    def append(self, links: Iterable[Link]) -> AppendReport:
        """Add links that are not already stored; the file is replaced atomically."""
        lines = (
            [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if self.path.exists()
            else []
        )
        seen = set(lines)
        appended = skipped = 0
        for link in links:
            line = link.model_dump_json()
            if line in seen:
                skipped += 1
                continue
            seen.add(line)
            lines.append(line)
            appended += 1
        if appended:
            atomic_write_lines(self.path, lines)
        return AppendReport(appended=appended, duplicates_skipped=skipped)


def graph_asof(
    store: LinkStore, cutoff_utc: pd.Timestamp, relation: Relation | None = None
) -> pd.DataFrame:
    """The links visible at `cutoff_utc`: `valid_from_utc <= cutoff_utc < valid_to_utc`.

    One row per (source_node, receiver_node, relation): the visible link with the latest
    `valid_from_utc` (the later stored one when two share it). An empty result keeps the columns.
    """
    if cutoff_utc.tzinfo is None:
        raise ValueError("cutoff_utc must be tz-aware (UTC)")
    rows = [
        {**link.model_dump(), "relation": link.relation.value}
        for link in store.read_all()
        if (relation is None or link.relation == relation)
        and link.valid_from_utc <= cutoff_utc < link.valid_to_utc
    ]
    if not rows:
        return pd.DataFrame(columns=GRAPH_COLUMNS)
    frame = pd.DataFrame(rows)[GRAPH_COLUMNS]
    frame = frame.sort_values("valid_from_utc", kind="stable")
    frame = frame.drop_duplicates(["source_node", "receiver_node", "relation"], keep="last")
    return frame.sort_values(["source_node", "receiver_node", "relation"]).reset_index(drop=True)
