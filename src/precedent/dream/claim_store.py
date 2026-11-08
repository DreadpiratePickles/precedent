"""Append-only JSON Lines stores for claims, tensions, shifts and resolutions.

A store never rewrites or drops a line. A lifecycle change (expiry, closing, re-firing, a
decision) is a named method that appends an updated copy of the latest record for an id; reads
take the last record per id. `append` is idempotent on id: a record whose identity fields equal
the first one stored for that id is skipped, and one that differs raises `ConflictError`.
"""

from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel

from precedent.dream.claim_schema import Claim, Resolution, Shift, Tension
from precedent.spine.jsonl_io import AppendReport, atomic_write_lines, read_models

_Record = TypeVar("_Record", bound=BaseModel)

CLAIM_IDENTITY_FIELDS = (
    "subject",
    "attribute",
    "period",
    "value",
    "source_doc_id",
    "valid_from_utc",
    "evidence_span",
    "extractor_version",
)
TENSION_IDENTITY_FIELDS = ("kind", "group_key", "subject")  # members are lifecycle (D72)
SHIFT_IDENTITY_FIELDS = ("kind", "subject")
RESOLUTION_IDENTITY_FIELDS = ("tension_id", "hypothesis", "evidence_span", "proposer_version")


class ConflictError(Exception):
    """A record repeats a stored id with different identity fields, or a lifecycle change that
    the record's state does not allow."""


class _AppendOnlyStore(Generic[_Record]):
    """One JSON Lines file of one record type, keyed by an id field."""

    _filename: str
    _model: type[_Record]
    _id_field: str
    _identity_fields: tuple[str, ...]

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def path(self) -> Path:
        """The JSON Lines file this store owns."""
        return self._root / self._filename

    def read_all(self) -> list[_Record]:
        """Every line in file order, lifecycle records included."""
        return read_models(self.path, self._model)

    def latest(self) -> list[_Record]:
        """One record per id, the last one stored winning, in first-seen order."""
        by_id: dict[str, _Record] = {}
        for record in self.read_all():
            by_id[self._id_of(record)] = record
        return list(by_id.values())

    def append(self, records: Iterable[_Record]) -> AppendReport:
        """Add records with new ids; skip repeats of stored identity; refuse changed ones.

        All or nothing: a conflict anywhere in the batch writes no line.
        """
        stored = self.read_all()
        first_identity: dict[str, dict[str, Any]] = {}
        for record in stored:
            first_identity.setdefault(self._id_of(record), self._identity_of(record))
        new_records: list[_Record] = []
        skipped = 0
        for record in records:
            record_id = self._id_of(record)
            identity = self._identity_of(record)
            known = first_identity.get(record_id)
            if known is None:
                first_identity[record_id] = identity
                new_records.append(record)
            elif known == identity:
                skipped += 1
            else:
                raise ConflictError(
                    f"{self._filename}: id {record_id!r} already stored with different content"
                )
        if new_records:
            self._write(stored + new_records)
        return AppendReport(added=len(new_records), identical_skipped=skipped)

    def _append_update(
        self, record_id: str, change: Callable[[_Record], dict[str, Any]]
    ) -> _Record:
        """Append a validated copy of the latest record for `record_id` with `change` applied."""
        stored = self.read_all()
        latest = next(
            (r for r in reversed(stored) if self._id_of(r) == record_id),
            None,
        )
        if latest is None:
            raise KeyError(f"{self._filename}: no record with id {record_id!r}")
        updated = self._model.model_validate({**latest.model_dump(), **change(latest)})
        self._write([*stored, updated])
        return updated

    def _write(self, records: list[_Record]) -> None:
        atomic_write_lines(self.path, (record.model_dump_json() for record in records))

    def _id_of(self, record: _Record) -> str:
        return getattr(record, self._id_field)

    def _identity_of(self, record: _Record) -> dict[str, Any]:
        dumped = record.model_dump(mode="json", include=set(self._identity_fields))
        if "claim_ids" in dumped:
            dumped["claim_ids"] = sorted(dumped["claim_ids"])
        return dumped


class ClaimStore(_AppendOnlyStore[Claim]):
    """Claims in `claims.jsonl`."""

    _filename = "claims.jsonl"
    _model = Claim
    _id_field = "claim_id"
    _identity_fields = CLAIM_IDENTITY_FIELDS

    def supersede(
        self, old_claim_id: str, new_claim_id: str | None, at_utc: datetime, reason: str
    ) -> Claim:
        """Expire a claim at `at_utc`: append a copy with its end, successor and reason.

        `new_claim_id` is not checked against the store; a dated rule expiry has no successor.
        A claim that already ends later than `at_utc` has its validity shortened; one that
        already ended at or before `at_utc` raises `ConflictError`; `at_utc` before the claim's
        start raises `ValueError`.
        """

        def change(latest: Claim) -> dict[str, Any]:
            if at_utc < latest.valid_from_utc:
                raise ValueError(f"at_utc {at_utc} is before the claim's valid_from_utc")
            if latest.valid_to_utc is not None and latest.valid_to_utc <= at_utc:
                raise ConflictError(f"claim {old_claim_id!r} is already expired at {at_utc}")
            return {
                "valid_to_utc": at_utc,
                "superseded_by": new_claim_id,
                "expiry_reason": reason,
                "recorded_utc": at_utc,
            }

        return self._append_update(old_claim_id, change)


class TensionStore(_AppendOnlyStore[Tension]):
    """Tensions in `tensions.jsonl`."""

    _filename = "tensions.jsonl"
    _model = Tension
    _id_field = "tension_id"
    _identity_fields = TENSION_IDENTITY_FIELDS

    def close(self, tension_id: str, session: date, recorded_utc: datetime) -> Tension:
        """Mark a tension closed as of `session`. A closed tension cannot change again."""

        def change(latest: Tension) -> dict[str, Any]:
            self._refuse_if_closed(latest)
            return {"status": "closed", "closed_session": session, "recorded_utc": recorded_utc}

        return self._append_update(tension_id, change)

    def restate_members(
        self, tension_id: str, claim_ids: list[str], recorded_utc: datetime
    ) -> Tension:
        """Record that the claims behind a tension are now `claim_ids` (D72).

        The status and the opening session stay as they are. A closed tension cannot change.
        """

        def change(latest: Tension) -> dict[str, Any]:
            self._refuse_if_closed(latest)
            return {"claim_ids": sorted(claim_ids), "recorded_utc": recorded_utc}

        return self._append_update(tension_id, change)

    def reopen(
        self, tension_id: str, session: date, claim_ids: list[str], recorded_utc: datetime
    ) -> Tension:
        """Open a closed tension again when its group recurs: opened as of `session`.

        Only a closed tension reopens; the earlier opening and closing stay in the store.
        """

        def change(latest: Tension) -> dict[str, Any]:
            if latest.status != "closed":
                raise ConflictError(f"tension {latest.tension_id!r} is not closed")
            return {
                "status": "open",
                "closed_session": None,
                "opened_session": session,
                "claim_ids": sorted(claim_ids),
                "recorded_utc": recorded_utc,
            }

        return self._append_update(tension_id, change)

    def acknowledge(self, tension_id: str, recorded_utc: datetime) -> Tension:
        """Mark a tension as seen by the owner, without closing it."""

        def change(latest: Tension) -> dict[str, Any]:
            self._refuse_if_closed(latest)
            return {"status": "acknowledged", "recorded_utc": recorded_utc}

        return self._append_update(tension_id, change)

    @staticmethod
    def _refuse_if_closed(latest: Tension) -> None:
        if latest.status == "closed":
            raise ConflictError(f"tension {latest.tension_id!r} is already closed")


class ShiftStore(_AppendOnlyStore[Shift]):
    """Shifts in `shifts.jsonl`."""

    _filename = "shifts.jsonl"
    _model = Shift
    _id_field = "shift_id"
    _identity_fields = SHIFT_IDENTITY_FIELDS

    def refire(
        self, shift_id: str, statistic: float, detected_session: date, recorded_utc: datetime
    ) -> Shift:
        """Record that a later session fired the same shift with a new statistic."""
        return self._append_update(
            shift_id,
            lambda latest: {
                "statistic": statistic,
                "detected_session": detected_session,
                "recorded_utc": recorded_utc,
            },
        )


class ResolutionStore(_AppendOnlyStore[Resolution]):
    """Resolutions in `resolutions.jsonl`."""

    _filename = "resolutions.jsonl"
    _model = Resolution
    _id_field = "resolution_id"
    _identity_fields = RESOLUTION_IDENTITY_FIELDS

    def decide(
        self, resolution_id: str, status: Literal["accepted", "rejected"], decided_utc: datetime
    ) -> Resolution:
        """Record the owner's decision on a proposed resolution, recorded when it is made."""
        if status not in ("accepted", "rejected"):
            raise ValueError(f"a decision is 'accepted' or 'rejected', not {status!r}")
        return self._append_update(
            resolution_id,
            lambda latest: {
                "status": status,
                "decided_utc": decided_utc,
                "recorded_utc": decided_utc,
            },
        )


def supersede(
    store: ClaimStore,
    old_claim_id: str,
    new_claim_id: str | None,
    at_utc: datetime,
    reason: str,
) -> Claim:
    """Expire `old_claim_id` in `store`; see `ClaimStore.supersede`."""
    return store.supersede(old_claim_id, new_claim_id, at_utc, reason)


def tensions_asof(store: TensionStore, cutoff_utc: datetime) -> list[Tension]:
    """One record per tension id: the last one recorded by the cutoff (D13), first-seen order."""
    known: dict[str, Tension] = {}
    for tension in store.read_all():
        if tension.recorded_utc <= cutoff_utc:
            known[tension.tension_id] = tension
    return list(known.values())


_DATETIME_COLUMNS = ("valid_from_utc", "valid_to_utc", "recorded_utc")


def claims_asof(
    store: ClaimStore,
    cutoff_utc: datetime,
    subject: str | None = None,
    attribute: str | None = None,
) -> pd.DataFrame:
    """The claims a reader at `cutoff_utc` could see, as a frame with the full Claim columns.

    For each id the last record already recorded by the cutoff is used, so a cutoff before an
    expiry was recorded still sees the claim. A claim is visible when
    `valid_from_utc <= cutoff < valid_to_utc` (or it has no end). Enum columns hold their string
    values; the datetime columns are tz-aware UTC even when the frame is empty.
    """
    known = _known_by(store, cutoff_utc)
    rows = [
        claim.model_dump(mode="json")
        for claim in known.values()
        if claim.valid_from_utc <= cutoff_utc
        and (claim.valid_to_utc is None or claim.valid_to_utc > cutoff_utc)
        and (subject is None or claim.subject == subject)
        and (attribute is None or claim.attribute.value == attribute)
    ]
    return _claim_frame(rows)


def claims_history_asof(
    store: ClaimStore,
    cutoff_utc: datetime,
    lookback_start_utc: datetime,
    subject: str | None = None,
    attribute: str | None = None,
) -> pd.DataFrame:
    """Every claim that began in `[lookback_start_utc, cutoff_utc]`, expired or not (D71).

    A history detector (the direction CUSUM) needs the claims that have come and gone, which
    `claims_asof` hides once `valid_to_utc` has passed. The read is still as of the cutoff: for
    each id the last record already recorded by it is used, and a claim recorded later is absent.
    `valid_to_utc` is ignored. Same columns as `claims_asof`.
    """
    known = _known_by(store, cutoff_utc)
    rows = [
        claim.model_dump(mode="json")
        for claim in known.values()
        if lookback_start_utc <= claim.valid_from_utc <= cutoff_utc
        and (subject is None or claim.subject == subject)
        and (attribute is None or claim.attribute.value == attribute)
    ]
    return _claim_frame(rows)


def _known_by(store: ClaimStore, cutoff_utc: datetime) -> dict[str, Claim]:
    if cutoff_utc.tzinfo is None or cutoff_utc.utcoffset() is None:
        raise ValueError("cutoff_utc must be timezone-aware (a naive cutoff is rejected)")
    known: dict[str, Claim] = {}
    for claim in store.read_all():
        if claim.recorded_utc <= cutoff_utc:
            known[claim.claim_id] = claim
    return known


def _claim_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(Claim.model_fields))
    for column in _DATETIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")
    return frame
