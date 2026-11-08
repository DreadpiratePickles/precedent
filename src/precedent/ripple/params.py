"""Ripple parameters: read from `references/ripple_params.json` and validated at the boundary."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from precedent import config


class RippleParams(BaseModel):
    """The `ripple` section. Every value is a proposal until a result on data supports it."""

    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(gt=0.0, le=1.0)
    hops: int = Field(ge=1)
    half_life_sessions: float = Field(gt=0.0)
    max_age_sessions: int = Field(ge=0)
    annual_link_expiry_months: int = Field(ge=1)
    correlation_window_sessions: int = Field(ge=2)
    correlation_top_k: int = Field(ge=1)
    placebo_margin_floor: float = Field(gt=0.0)
    placebo_margin_se_multiple: float = Field(ge=0.0)
    proposed: bool
    note: str


class RippleParamsFile(BaseModel):
    """The whole file: exactly one section, `ripple`."""

    model_config = ConfigDict(extra="forbid")
    ripple: RippleParams


def load_ripple_params() -> dict[str, Any]:
    """The validated `ripple` section as a plain dict."""
    path = config.references_dir() / "ripple_params.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RippleParamsFile.model_validate(raw).ripple.model_dump()
