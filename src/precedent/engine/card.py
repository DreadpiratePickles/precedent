"""The precedent card: a model built from summaries, rendered as text, JSON and PNG.

The card is assembled by code, never by a model. Text is the oracle; the PNG draws the same lines.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from precedent.config import ConfigError, references_dir
from precedent.libraries.common import OUTCOME_TYPES

__all__ = [
    "DEFAULT_FONT_PATH",
    "DISCLAIMER",
    "KEY_FLAGS_HINT",
    "SOURCE_UNITS",
    "CardModel",
    "CardRow",
    "NearestPrecedent",
    "SourceCoverage",
    "format_outcome_row",
    "render_json",
    "render_png",
    "render_text",
]

DISCLAIMER = "These are past outcomes of similar events, not a forecast."

KEY_FLAGS_HINT = (
    "could not type this text; ask by key instead: --event-type <type> --actor-kind <kind> "
    "--severity-band low|mid|high --scheduled|--unscheduled "
    "[--country-group US|DM|EM|none] [--direction-sign -1|0|1]"
)

SOURCE_UNITS = {"polymarket": "markets", "kalshi": "markets"}
"""Unit printed after a source's row count; any other source counts `events`."""

DEFAULT_FONT_PATH = references_dir() / "fonts" / "DejaVuSansMono.ttf"

_ALL = "all"
_KEY_WIDTH = 36
_FONT_SIZE_PX = 14
_PADDING_PX = 16
_LINE_GAP_PX = 4
_OUTCOME_HEADER = "horizon   rate|median   80% band        n   vs ordinary"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CardRow(_Frozen):
    outcome_type: str
    outcome_key: str
    horizon: int | None
    is_binary: bool
    n: int
    n_eff: float
    refused: bool
    k: int | None
    rate: float | None
    interval: tuple[float, float] | None
    median: float | None
    centre_shrunk: float | None
    band: tuple[float, float] | None
    vs_ordinary: Literal["inside", "outside"] | None

    @model_validator(mode="after")
    def _printed_values_are_present(self) -> CardRow:
        if self.refused:
            return self
        if self.is_binary:
            missing = [
                name for name in ("k", "rate", "interval") if getattr(self, name) is None
            ]
        else:
            missing = [name for name in ("median", "band") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"unrefused {'binary' if self.is_binary else 'continuous'} row "
                f"{self.outcome_key!r} lacks {', '.join(missing)}"
            )
        return self


class NearestPrecedent(_Frozen):
    date: dt.date
    one_line: str
    similarity: float
    source_ref: str


class SourceCoverage(_Frozen):
    name: str
    first_date: dt.date
    last_date: dt.date
    rows: int


class CardModel(_Frozen):
    schema_version: Literal["1.0"] = "1.0"
    headline: str
    typed_as: str
    typed_reason: str | None
    key: str | None
    asof_utc: dt.datetime
    horizon: int
    outcome_type: str
    n: int
    n_eff: float
    earliest: dt.date | None
    latest: dt.date | None
    refused: bool
    min_neff: float
    rows: list[CardRow]
    not_built: dict[str, str]
    nearest: list[NearestPrecedent]
    covers: list[SourceCoverage]
    hint: str | None
    disclaimer: Literal[DISCLAIMER] = DISCLAIMER

    @field_validator("asof_utc")
    @classmethod
    def _asof_is_timezone_aware(cls, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("asof_utc must be timezone-aware")
        return value

    @field_validator("outcome_type")
    @classmethod
    def _outcome_type_is_known(cls, value: str) -> str:
        if value != _ALL and value not in OUTCOME_TYPES:
            raise ValueError(f"outcome_type must be 'all' or one of {OUTCOME_TYPES}: {value!r}")
        return value


def _date_or_dash(value: dt.date | None) -> str:
    return "-" if value is None else str(value)


def format_outcome_row(row: CardRow, min_neff: float) -> str:
    """One outcome row, exactly as the card prints it (nothing is truncated)."""
    horizon = "res" if row.horizon is None else str(row.horizon)
    if row.refused:
        return (
            f"{row.outcome_key:<{_KEY_WIDTH}}{horizon:>7}   "
            f"refused: independent={math.floor(row.n_eff)}, need {min_neff:g}"
        )
    if row.is_binary:
        assert row.k is not None and row.rate is not None and row.interval is not None
        centre = f"{row.k}/{row.n} ({row.rate:.0%})"
        band = f"[{row.interval[0]:.0%}, {row.interval[1]:.0%}]"
    else:
        assert row.median is not None and row.band is not None
        centre = f"{row.median:+.1%}"
        band = f"[{row.band[0]:+.1%}, {row.band[1]:+.1%}]"
    verdict = "-" if row.vs_ordinary is None else row.vs_ordinary
    return (
        f"{row.outcome_key:<{_KEY_WIDTH}}{horizon:>7}   {centre:<14}{band:<13}{row.n:>4}   "
        f"{verdict}"
    )


def _outcome_lines(card: CardModel) -> list[str]:
    asked = OUTCOME_TYPES if card.outcome_type == _ALL else (card.outcome_type,)
    lines: list[str] = []
    for outcome_type in asked:
        if outcome_type in card.not_built:
            lines.append(f"{outcome_type:<{_KEY_WIDTH}}NOT BUILT: {card.not_built[outcome_type]}")
            continue
        lines.append(f"{outcome_type:<{_KEY_WIDTH}}{_OUTCOME_HEADER}")
        shown = [
            format_outcome_row(row, card.min_neff)
            for row in card.rows
            if row.outcome_type == outcome_type
        ]
        lines.extend(shown or [f"no completed outcomes at horizon {card.horizon}"])
    return lines


def _covers_line(card: CardModel) -> str:
    if not card.covers:
        return "library covers  nothing (no library loaded)"
    parts = [
        f"{c.name}: {c.first_date} to {c.last_date}, {c.rows} {SOURCE_UNITS.get(c.name, 'events')}"
        for c in card.covers
    ]
    return "library covers  " + "; ".join(parts)


def render_text(card: CardModel) -> str:
    """The card as plain text: lines joined with newlines, no trailing newline."""
    independent = math.floor(card.n_eff)
    lines = [
        f"PRECEDENT  ·  {card.headline}",
        f"typed as       {card.typed_as}",
        f"precedents     n={card.n}  independent={independent}  "
        f"earliest {_date_or_dash(card.earliest)}  latest {_date_or_dash(card.latest)}",
        "",
    ]
    if card.refused:
        lines.append(
            f"REFUSED        {independent} usable precedents, need {card.min_neff:g}. "
            "No band is drawn below that."
        )
    else:
        lines.extend(_outcome_lines(card))
        lines.append("")
    lines.append("nearest independent precedents (non-overlapping outcome windows)")
    if card.nearest:
        lines.extend(
            f"{n.date}  {n.one_line}  similarity {n.similarity:.2f}  {n.source_ref}"
            for n in card.nearest
        )
    else:
        lines.append("(none)")
    lines.append(_covers_line(card))
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def render_json(card: CardModel) -> str:
    """The card as indented JSON; round-trips through `CardModel.model_validate_json`."""
    return card.model_dump_json(indent=2)


def _load_font(font_path: Path | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default(size=_FONT_SIZE_PX)
    if not font_path.is_file():
        raise ConfigError(f"card font not bundled: {font_path}")
    return ImageFont.truetype(str(font_path), _FONT_SIZE_PX)


def render_png(card: CardModel, path: Path, font_path: Path | None = DEFAULT_FONT_PATH) -> Path:
    """Draw `render_text(card)` as black text on white, one text call per line.

    `font_path=None` uses Pillow's built-in font. A path that is not a file raises `ConfigError`
    before anything is drawn. No metadata is written, so the bytes are the same on every run.
    """
    font = _load_font(font_path)
    lines = render_text(card).split("\n")
    _, _, _, line_bottom = font.getbbox("Ag")
    line_height = math.ceil(line_bottom) + _LINE_GAP_PX
    text_width = max(math.ceil(font.getlength(line)) for line in lines)
    size = (text_width + 2 * _PADDING_PX, line_height * len(lines) + 2 * _PADDING_PX)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((_PADDING_PX, _PADDING_PX + index * line_height), line, fill="black", font=font)
    image.save(path, format="PNG")
    return path
