"""Ripple parameters: read from `references/ripple_params.json` and validated at the boundary."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from precedent import config


class RippleParams(BaseModel):
    """The `ripple` section. Every value is a proposal until a result on data supports it."""

    model_config = ConfigDict(extra="forbid")
    alpha: float
    hops: int
    half_life_sessions: float
    max_age_sessions: int
    annual_link_expiry_months: int
    correlation_window_sessions: int
    correlation_top_k: int
    proposed: bool
    note: str


def load_ripple_params() -> dict[str, Any]:
    """The validated `ripple` section as a plain dict."""
    path = config.references_dir() / "ripple_params.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"ripple"}:
        raise ValueError(f"{path.name}: expected exactly one top-level key, 'ripple'")
    return RippleParams.model_validate(raw["ripple"]).model_dump()
