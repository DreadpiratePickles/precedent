"""GDELT 2.0 events as rule-typed EventRecords (parser `rules-gdelt-v1:<hash>`). No model is called.

An export row is typed from its CAMEO code by a committed table, sized from `GoldsteinScale` and
`NumSources`, and stamped with `DATEADDED` (never `SQLDATE`) so alignment is point-in-time.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from precedent.config import references_dir
from precedent.spine.event_key import build_event_key, load_country_groups
from precedent.spine.event_record import EventRecord
from precedent.spine.event_schema import ActorKind, EventType, NewsEvent
from precedent.spine.pit_alignment import align_records, unalign_before
from precedent.spine.session_calendar import (
    calendar_start_utc,
    default_cutoffs_for,
    load_nyse_calendar,
)

PARSER_VERSION_PREFIX = "rules-gdelt-v1"
_CAMEO_MAP_FILE = "cameo_to_event_type.json"
_PARAMS_FILE = "precedent_params.json"
SOURCE_NAME = "gdelt2"
EXPORT_COLUMN_COUNT = 61
UNTYPED = "untyped"

# 1-based positions in the 61-column GDELT 2.0 export, as in the Event Codebook V2.0.
GDELT_COLUMNS: dict[str, int] = {
    "GLOBALEVENTID": 1,
    "SQLDATE": 2,
    "Actor1Code": 6,
    "Actor1Name": 7,
    "Actor1CountryCode": 8,
    "Actor1Type1Code": 13,
    "Actor2Name": 17,
    "Actor2CountryCode": 18,
    "Actor2Type1Code": 23,
    "IsRootEvent": 26,
    "EventCode": 27,
    "EventBaseCode": 28,
    "EventRootCode": 29,
    "QuadClass": 30,
    "GoldsteinScale": 31,
    "NumMentions": 32,
    "NumSources": 33,
    "NumArticles": 34,
    "AvgTone": 35,
    "ActionGeo_CountryCode": 54,
    "DATEADDED": 60,
    "SOURCEURL": 61,
}

_ACTOR_KIND_BY_TYPE = {
    "GOV": ActorKind.government,
    "MIL": ActorKind.government,
    "IGO": ActorKind.international_org,
    "BUS": ActorKind.company,
}
_UNKNOWN_ACTOR = "unknown actor"
_NAME_MAX = 80
_ACTION_MAX = 120
_DATEADDED_FORMAT = "%Y%m%d%H%M%S"
_EXTRA_SESSION_DAYS = 15
_BANDS = ("low", "mid", "high")


class GdeltFormatError(ValueError):
    """A GDELT file, master-list line or row that does not have the documented shape."""


class GdeltParams(BaseModel):
    """Every threshold the GDELT typer uses; all values are proposals (decision B4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    min_sources: int = Field(default=3, ge=1)
    goldstein_cuts: tuple[float, float] = (10 / 3, 20 / 3)
    sources_bump_at: int = Field(default=10, ge=1)
    severity_by_band: dict[str, int] = {"low": 2, "mid": 5, "high": 8}
    latency_minutes: int = Field(default=30, ge=0)

    @field_validator("goldstein_cuts")
    @classmethod
    def _cuts_rise_from_positive(cls, cuts: tuple[float, float]) -> tuple[float, float]:
        if not 0 < cuts[0] < cuts[1]:
            raise ValueError(f"goldstein_cuts must satisfy 0 < low < high, got {cuts}")
        return cuts

    @field_validator("severity_by_band")
    @classmethod
    def _one_severity_per_band(cls, table: dict[str, int]) -> dict[str, int]:
        if set(table) != set(_BANDS):
            raise ValueError(f"severity_by_band needs exactly {list(_BANDS)}, got {sorted(table)}")
        if any(not 1 <= value <= 10 for value in table.values()):
            raise ValueError(f"severity_by_band values must be 1 to 10, got {table}")
        return table

    @classmethod
    def from_file(cls, path: Path) -> GdeltParams:
        """Read the `gdelt` section of the parameters file; other sections are not this model's."""
        try:
            section = json.loads(path.read_text(encoding="utf-8"))["gdelt"]
            return cls.model_validate(section)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read the gdelt section of {path}: {error!r}") from error


class CameoRule(BaseModel):
    """One mapping row: an event type (or `untyped`), optionally gated on Actor1's type."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: str
    requires_actor1_type: tuple[str, ...] = ()
    proposed: bool

    @field_validator("event_type")
    @classmethod
    def _known_event_type(cls, value: str) -> str:
        if value != UNTYPED and value not in {member.value for member in EventType}:
            raise ValueError(f"unknown event type {value!r}")
        return value


class CameoMap(BaseModel):
    """The committed CAMEO table: root rules 01-20, subcode rules, and code descriptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    roots: dict[str, CameoRule]
    subcodes: dict[str, CameoRule]
    descriptions: dict[str, str]

    @model_validator(mode="after")
    def _roots_complete_and_subcodes_belong(self) -> CameoMap:
        expected = {f"{number:02d}" for number in range(1, 21)}
        if set(self.roots) != expected:
            raise ValueError(f"roots must be exactly 01 to 20, got {sorted(self.roots)}")
        stray = [code for code in self.subcodes if code[:2] not in self.roots or len(code) < 3]
        if stray:
            raise ValueError(f"subcodes not under a known root: {stray}")
        return self


def gdelt_parser_version() -> str:
    """`rules-gdelt-v1:<12 hex>` over the CAMEO table's bytes and the params file's gdelt section.

    Editing either changes the version, so records made under different rules never share a
    vintage. The section is hashed as canonical JSON, so edits to other sections do not count.
    """
    mapping_file = references_dir() / _CAMEO_MAP_FILE
    params_file = references_dir() / _PARAMS_FILE
    try:
        mapping_bytes = mapping_file.read_bytes()
        section = json.loads(params_file.read_text(encoding="utf-8"))["gdelt"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot compute the GDELT parser version: {error!r}") from error
    canonical = json.dumps(section, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(mapping_bytes + canonical.encode("utf-8")).hexdigest()
    return f"{PARSER_VERSION_PREFIX}:{digest[:12]}"


def load_cameo_map(
    mapping_path: Path | None = None, eventcodes_path: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return {"roots", "subcodes", "descriptions"} read from references/, validated."""
    mapping_file = mapping_path or references_dir() / "cameo_to_event_type.json"
    codes_file = eventcodes_path or references_dir() / "cameo_eventcodes.json"
    try:
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        descriptions = json.loads(codes_file.read_text(encoding="utf-8"))["descriptions"]
        parsed = CameoMap(
            roots=mapping["roots"], subcodes=mapping["subcodes"], descriptions=descriptions
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"cannot load the CAMEO tables ({mapping_file.name}): {error}") from error
    return parsed.model_dump()


def read_export_csv(data: bytes) -> pd.DataFrame:
    """Read one export file (zip or plain TSV, no header) into the named columns, as strings."""
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise GdeltFormatError(f"expected one file in the zip, found {len(names)}")
            data = archive.read(names[0])
    positions = [position - 1 for position in GDELT_COLUMNS.values()]
    rows: list[list[str]] = []
    for line_number, line in enumerate(data.decode("utf-8", errors="replace").split("\n"), 1):
        line = line.removesuffix("\r")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != EXPORT_COLUMN_COUNT:
            raise GdeltFormatError(
                f"line {line_number} has {len(fields)} columns, expected {EXPORT_COLUMN_COUNT}"
            )
        rows.append([fields[position] for position in positions])
    return pd.DataFrame(rows, columns=list(GDELT_COLUMNS), dtype=str)


@dataclass(frozen=True)
class GdeltIngestReport:
    """What one conversion kept and set aside; the four row counts add up to `rows_read`."""

    rows_read: int
    kept: int
    dropped_untyped: int
    dropped_min_sources: int
    dropped_duplicate: int
    non_iso_country_codes: int


@dataclass(frozen=True)
class _Candidate:
    """A typed row before dedup: its record plus what the dedup rule needs."""

    record: EventRecord
    event_id: int
    actor1_country: str
    non_iso_codes: int


def _numbers(rows: pd.DataFrame, column: str, kind: type) -> list[Any]:
    values = []
    for event_id, text in zip(rows["GLOBALEVENTID"], rows[column], strict=True):
        try:
            values.append(kind(text))
        except ValueError as error:
            raise GdeltFormatError(
                f"{column} is not a number in event {event_id}: {text!r}"
            ) from error
    return values


def _first_seen(rows: pd.DataFrame) -> list[datetime]:
    stamps = []
    for event_id, text in zip(rows["GLOBALEVENTID"], rows["DATEADDED"], strict=True):
        try:
            stamps.append(datetime.strptime(text, _DATEADDED_FORMAT).replace(tzinfo=UTC))
        except ValueError as error:
            raise GdeltFormatError(
                f"DATEADDED is not YYYYMMDDHHMMSS in event {event_id}"
            ) from error
    return stamps


def _event_type_of(row: Mapping[str, str], cameo_map: dict) -> str:
    rule = cameo_map["subcodes"].get(row["EventCode"])
    if rule is not None:
        required = rule["requires_actor1_type"]
        if not required or row["Actor1Type1Code"] in required:
            return rule["event_type"]
    root_rule = cameo_map["roots"].get(row["EventRootCode"])
    return root_rule["event_type"] if root_rule is not None else UNTYPED


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _severity(goldstein: float, num_sources: int, params: GdeltParams) -> int:
    low_cut, high_cut = params.goldstein_cuts
    magnitude = abs(goldstein)
    band = 0 if magnitude < low_cut else 1 if magnitude < high_cut else 2
    if num_sources >= params.sources_bump_at:
        band = min(band + 1, len(_BANDS) - 1)
    return params.severity_by_band[_BANDS[band]]


def _iso3_codes(row: Mapping[str, str]) -> tuple[list[str], int]:
    """Actor1's then Actor2's ISO3 code, no repeats, plus how many non-empty codes were not ISO."""
    known = load_country_groups()
    linked: list[str] = []
    non_iso = 0
    for code in (row["Actor1CountryCode"], row["Actor2CountryCode"]):
        if not code:
            continue
        if code not in known:
            non_iso += 1
        elif code not in linked:
            linked.append(code)
    return linked, non_iso


def _record_for(
    row: Mapping[str, str],
    event_type: str,
    first_seen: datetime,
    goldstein: float,
    num_sources: int,
    cameo_map: dict,
    params: GdeltParams,
    ingested_utc: datetime,
    parser_version: str,
) -> tuple[EventRecord, int]:
    linked, non_iso = _iso3_codes(row)
    step = _sign(goldstein)
    actor_country = row["Actor1CountryCode"]
    impacted = (
        [
            {
                "name": row["Actor2Name"][:_NAME_MAX],
                "kind": "country",
                "ticker_hint": None,
                "direction_step": step,
            }
        ]
        if row["Actor2Name"]
        else []
    )
    event = NewsEvent.model_validate(
        {
            "schema_version": "1.0",
            "event_type": event_type,
            "actor": {
                "name": (row["Actor1Name"] or row["Actor1Code"] or _UNKNOWN_ACTOR)[:_NAME_MAX],
                "kind": _ACTOR_KIND_BY_TYPE.get(row["Actor1Type1Code"], ActorKind.other),
                "country_iso3": actor_country if actor_country in load_country_groups() else None,
            },
            "action": cameo_map["descriptions"].get(row["EventCode"], f"CAMEO {row['EventCode']}")[
                :_ACTION_MAX
            ],
            "impacted": impacted,
            "severity": _severity(goldstein, num_sources, params),
            "macro_direction_step": step,
            "confidence": "medium",
            "horizon": "weeks",
            "novelty": "new",
            "is_scheduled": False,
            "evidence_span": f"CAMEO {row['EventCode']}",
        }
    )
    named = {name: row[name] for name in GDELT_COLUMNS}
    body = json.dumps(named, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    record = EventRecord(
        article_id=f"gdelt:{row['GLOBALEVENTID']}",
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        source=SOURCE_NAME,
        published_utc=None,
        first_seen_utc=first_seen,
        ingested_utc=ingested_utc,
        available_utc=None,
        effective_session=None,
        parser_version=parser_version,
        event=event,
        resolved_impacts=[],
        linked_countries=linked,
    )
    return record, non_iso


def _default_cutoffs(first_seen: Sequence[datetime]) -> pd.Series:
    """Session cutoffs from the first row's date to 15 days past the last, so late rows align."""
    return default_cutoffs_for(
        [stamp.date() for stamp in first_seen], load_nyse_calendar(), _EXTRA_SESSION_DAYS
    )


def _keep_earliest(items: Sequence[tuple[EventRecord, str, int]]) -> list[EventRecord]:
    """One record per (event key, Actor1 country, session); earliest first_seen, then smaller id."""
    winners: dict[tuple[str, str, date], tuple[tuple[datetime, int], int]] = {}
    for position, (record, country, event_id) in enumerate(items):
        if record.effective_session is None:  # no session: kept as it is, never merged
            continue
        key = (build_event_key(record).as_string(), country, record.effective_session)
        rank = (record.first_seen_utc, event_id)
        if key not in winners or rank < winners[key][0]:
            winners[key] = (rank, position)
    winning_positions = {position for _, position in winners.values()}
    return [
        record
        for position, (record, _, _) in enumerate(items)
        if record.effective_session is None or position in winning_positions
    ]


def _event_id_of(record: EventRecord) -> int:
    return int(record.article_id.split(":", 1)[1])


def _actor1_country_key(record: EventRecord) -> str:
    """Actor1's ISO3 code when it is one, else "" (never Actor2's country)."""
    return record.event.actor.country_iso3 or ""


def dedupe_gdelt_records(records: list[EventRecord]) -> list[EventRecord]:
    """Apply the dedup rule across runs: stored records read back together.

    Same rule as inside one conversion: the Actor1 country key is Actor1's ISO3 code if it is
    ISO, else "" (the same `_actor1_country_key` serves both).
    """
    items = [(record, _actor1_country_key(record), _event_id_of(record)) for record in records]
    return _keep_earliest(items)


def events_from_gdelt_rows(
    rows: pd.DataFrame,
    cameo_map: dict,
    params: GdeltParams,
    *,
    ingested_utc: datetime,
    session_cutoffs_utc: pd.Series | None = None,
) -> tuple[list[EventRecord], GdeltIngestReport]:
    """Filter, type, align and deduplicate export rows; the report counts what was set aside.

    Order: too few sources, then unmapped CAMEO code, then one record per (event key, Actor1
    country, effective session). Dedup is per call; `dedupe_gdelt_records` does it across runs.
    """
    parser_version = gdelt_parser_version()
    goldstein = _numbers(rows, "GoldsteinScale", float)
    sources = _numbers(rows, "NumSources", int)
    first_seen = _first_seen(rows)
    below_min = untyped = 0
    candidates: list[_Candidate] = []
    for position, row in enumerate(rows.to_dict("records")):
        if sources[position] < params.min_sources:
            below_min += 1
            continue
        event_type = _event_type_of(row, cameo_map)
        if event_type == UNTYPED:
            untyped += 1
            continue
        record, non_iso = _record_for(
            row, event_type, first_seen[position], goldstein[position], sources[position],
            cameo_map, params, ingested_utc, parser_version,
        )  # fmt: skip
        candidates.append(
            _Candidate(record, int(row["GLOBALEVENTID"]), _actor1_country_key(record), non_iso)
        )
    cutoffs = session_cutoffs_utc
    use_default_cutoffs = cutoffs is None
    if cutoffs is None and candidates:
        cutoffs = _default_cutoffs([c.record.first_seen_utc for c in candidates])
    aligned = (
        align_records(
            [c.record for c in candidates], cutoffs, pd.Timedelta(minutes=params.latency_minutes)
        )
        if candidates and cutoffs is not None
        else []
    )
    if use_default_cutoffs:  # rows before the calendar's first year stay unaligned
        aligned = unalign_before(aligned, calendar_start_utc(load_nyse_calendar()))
    candidates = [
        _Candidate(record, c.event_id, c.actor1_country, c.non_iso_codes)
        for record, c in zip(aligned, candidates, strict=True)
    ]
    kept_records = _keep_earliest([(c.record, c.actor1_country, c.event_id) for c in candidates])
    kept_ids = {id(record) for record in kept_records}
    non_iso_total = sum(c.non_iso_codes for c in candidates if id(c.record) in kept_ids)
    report = GdeltIngestReport(
        rows_read=len(rows),
        kept=len(kept_records),
        dropped_untyped=untyped,
        dropped_min_sources=below_min,
        dropped_duplicate=len(candidates) - len(kept_records),
        non_iso_country_codes=non_iso_total,
    )
    return kept_records, report
