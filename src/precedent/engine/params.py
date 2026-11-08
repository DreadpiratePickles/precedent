"""Engine parameters: the `engine` section of `references/precedent_params.json`."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from precedent.config import references_dir

_SECTION = "engine"


class EngineParams(BaseModel):
    """Every value is proposed (see stages/20_precedent_libraries/CONTEXT.md)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int
    min_neff: float
    shrink_k: float
    story_window_sessions: int
    nearest_shown: int
    card_rows_per_type: int
    asset_horizons: tuple[int, ...]
    sequence_horizons: tuple[int, ...]
    wilson_z: float
    embedding_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    placebo_samples: int
    placebo_seed: int
    market: str
    asset_panel: tuple[str, ...]


def load_engine_params(path: Path | None = None) -> EngineParams:
    """Read only the `engine` section; unknown or missing fields raise `ValidationError`."""
    source = path if path is not None else references_dir() / "precedent_params.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    if _SECTION not in document:
        raise ValueError(f"no {_SECTION!r} section in {source}")
    return EngineParams(**document[_SECTION])
