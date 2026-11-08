"""A typer that replays a fixed mapping, for tests."""

from __future__ import annotations

from collections.abc import Mapping

from precedent.spine.typers.base import TypedText


class FakeTyper:
    def __init__(self, mapping: Mapping[str, TypedText], version: str = "fake-v1") -> None:
        self._mapping = dict(mapping)
        self.version = version

    def type_text(self, text: str) -> TypedText:
        found = self._mapping.get(text)
        if found is not None:
            return found
        return TypedText(
            status="untyped",
            key=None,
            reason="not in mapping",
            evidence_span=None,
            typer_version=self.version,
        )
