"""The owner's decisions on Dream's proposed resolutions: `dream promote` and `dream review`.

`review` prints what is waiting, each with both claims' evidence side by side. `promote` accepts
or rejects one resolution. Accepting applies the hypothesis to the stores by named lifecycle
methods only (nothing is rewritten): `one_is_stale` supersedes the older claim, `extraction_error`
expires the named claim, both close the tension; `different_scope` closes it; `genuine_conflict`
acknowledges it and leaves it open. The decision is appended last, so a run that stopped half way
is finished by running it again; a step whose end state already holds is skipped. No model is
called and nothing here forecasts.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import UTC, date, datetime
from itertools import zip_longest
from pathlib import Path

import pandas as pd

from precedent.dream.claim_schema import Claim, Hypothesis, Resolution, Tension
from precedent.dream.claim_store import (
    ClaimStore,
    ConflictError,
    ResolutionStore,
    TensionStore,
)
from precedent.dream.locking import DreamLockedError, exclusive_lock
from precedent.dream.report import DEFAULT_OUT
from precedent.dream.report_render import DISCLAIMER
from precedent.spine.session_calendar import default_cutoffs_for, load_nyse_calendar

OK = 0
STORE_FAILURE = 1
USAGE_ERROR = 2
CLOSING_LOOKAHEAD_DAYS = 14  # far enough to cross a weekend and a holiday
COLUMN_WIDTH = 38  # one claim's column in `review`; two of them fit an 80-column terminal
SHORT_ID = 12


class PromotionError(Exception):
    """The resolution cannot be applied as it stands; nothing has been written."""


def _now() -> datetime:
    """The wall clock; the one place a decision's time comes from when `--at` is not given."""
    return datetime.now(UTC)


def _aware_utc(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 timestamp: {text!r}") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"{text!r} has no UTC offset (naive times are rejected)")
    return moment.astimezone(UTC)


def _parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace | int:
    try:
        return parser.parse_args(argv)
    except SystemExit as request:
        return request.code if isinstance(request.code, int) else USAGE_ERROR


# --- promote -------------------------------------------------------------------------------------


def _promote_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent dream promote",
        description="Accept or reject one proposed resolution of a tension.",
    )
    parser.add_argument("--resolution", required=True, metavar="ID")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--accept", action="store_true")
    choice.add_argument("--reject", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the dream's store root")
    parser.add_argument(
        "--at", type=_aware_utc, metavar="ISO8601", help="decision time with a UTC offset"
    )
    return parser


def run_promote(argv: list[str]) -> int:
    """Decide one resolution: 0 decided or already decided so, 1 a store problem or a different
    earlier decision, 2 a usage error or an unknown resolution id."""
    args = _parse(_promote_parser(), argv)
    if isinstance(args, int):
        return args
    try:
        with exclusive_lock(args.out, create=False):
            return _promote(args)
    except DreamLockedError as error:
        print(f"error: {error}", file=sys.stderr)
        return STORE_FAILURE


def _promote(args: argparse.Namespace) -> int:
    claims, tensions = ClaimStore(args.out / "claims"), TensionStore(args.out / "tensions")
    resolutions = ResolutionStore(args.out / "resolutions")
    resolution = next((r for r in resolutions.latest() if r.resolution_id == args.resolution), None)
    if resolution is None:
        print(f"error: no resolution {args.resolution!r} under {args.out}", file=sys.stderr)
        return USAGE_ERROR
    decision = "accepted" if args.accept else "rejected"
    if resolution.status != "proposed":
        if resolution.status == decision:
            print(f"resolution {resolution.resolution_id} is already {decision}; nothing to do")
            return OK
        print(
            f"error: resolution {resolution.resolution_id} is already {resolution.status}, "
            f"not {decision}; a decision is not reversed",
            file=sys.stderr,
        )
        return STORE_FAILURE
    at = args.at if args.at is not None else _now()
    if at < resolution.recorded_utc:
        print(
            f"error: --at {at.isoformat()} is before the proposal was recorded "
            f"({resolution.recorded_utc.isoformat()})",
            file=sys.stderr,
        )
        return USAGE_ERROR
    try:
        notes = (
            _accept(claims, tensions, resolution, at) if args.accept else ["nothing else changed"]
        )
        resolutions.decide(resolution.resolution_id, decision, at)
    except (PromotionError, ConflictError, KeyError, ValueError, OSError) as error:
        print(f"error: {error} (stores under {args.out})", file=sys.stderr)
        return STORE_FAILURE
    print(f"resolution {resolution.resolution_id} {decision} ({resolution.hypothesis.value})")
    for note in notes:
        print(f"  {note}")
    return OK


def _accept(
    claims: ClaimStore, tensions: TensionStore, resolution: Resolution, at: datetime
) -> list[str]:
    """Apply an accepted hypothesis; everything that can fail is checked before the first write."""
    tension = next((t for t in tensions.latest() if t.tension_id == resolution.tension_id), None)
    if tension is None:
        raise PromotionError(f"tension {resolution.tension_id!r} is not in the tension store")
    hypothesis = resolution.hypothesis
    target: tuple[Claim, str | None] | None = None
    if hypothesis in (Hypothesis.one_is_stale, Hypothesis.extraction_error):
        target = _claim_to_expire(claims, tension, resolution)
    close_in = _closing_session(at) if hypothesis is not Hypothesis.genuine_conflict else None

    notes: list[str] = []
    if target is not None:
        claim, successor = target
        stem = "resolution" if hypothesis is Hypothesis.one_is_stale else "extraction_error"
        if claim.valid_to_utc is not None and claim.valid_to_utc <= at:
            notes.append(f"claim {_short(claim.claim_id)} was already expired; left as it is")
        else:
            claims.supersede(claim.claim_id, successor, at, f"{stem}:{resolution.resolution_id}")
            notes.append(f"expired claim {_short(claim.claim_id)} as {stem}")
    if close_in is not None:
        if tension.status == "closed":
            notes.append("tension was already closed")
        else:
            tensions.close(tension.tension_id, close_in, at)
            notes.append(f"closed the tension as of session {close_in}")
    elif tension.status == "open":
        tensions.acknowledge(tension.tension_id, at)
        notes.append("acknowledged the tension; it stays open")
    else:
        notes.append(f"tension was already {tension.status}")
    return notes


def _claim_to_expire(
    claims: ClaimStore, tension: Tension, resolution: Resolution
) -> tuple[Claim, str | None]:
    """The claim to expire and its successor: the newest other claim of the tension, if any.

    A named `claim_id` must belong to the tension. Without one, `one_is_stale` takes the claim
    with the earliest `valid_from_utc` (ties: the smallest id); `extraction_error` needs the name.
    """
    in_tension = {c.claim_id: c for c in claims.latest() if c.claim_id in tension.claim_ids}
    named = resolution.claim_id
    if named is None and resolution.hypothesis is Hypothesis.extraction_error:
        raise PromotionError("an extraction_error resolution must name its claim_id")
    if named is not None and named not in tension.claim_ids:
        raise PromotionError(f"claim_id {named!r} is not one of the tension's claims")
    if named is not None and named not in in_tension:
        raise PromotionError(f"claim {named!r} is not in the claim store")
    if not in_tension:
        raise PromotionError("none of the tension's claims is in the claim store")
    claim = (
        in_tension[named]
        if named is not None
        else min(in_tension.values(), key=lambda c: (c.valid_from_utc, c.claim_id))
    )
    if resolution.hypothesis is Hypothesis.extraction_error:
        return claim, None
    others = [c for c in in_tension.values() if c.claim_id != claim.claim_id]
    newest = min(others, key=lambda c: (-c.valid_from_utc.timestamp(), c.claim_id), default=None)
    return claim, newest.claim_id if newest is not None else None


def _closing_session(at: datetime) -> date:
    """The first session whose cutoff is after `at`: the dream that will report the closing."""
    cutoffs = default_cutoffs_for([at.date()], load_nyse_calendar(), CLOSING_LOOKAHEAD_DAYS)
    later = cutoffs[cutoffs > pd.Timestamp(at)]
    if later.empty:
        raise PromotionError(f"no session on the NYSE calendar closes after {at.isoformat()}")
    return later.index[0].date()


def _short(identifier: str) -> str:
    return identifier[:SHORT_ID]


# --- review --------------------------------------------------------------------------------------


def _review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent dream review",
        description="Print the proposed resolutions awaiting the owner, evidence side by side.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the dream's store root")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="only resolutions proposed on or after this UTC date",
    )
    return parser


def run_review(argv: list[str]) -> int:
    """Print what is waiting for a decision: 0 always, 2 for a bad flag. Writes nothing."""
    args = _parse(_review_parser(), argv)
    if isinstance(args, int):
        return args
    try:
        with exclusive_lock(args.out, create=False):
            return _review(args)
    except DreamLockedError as error:
        print(f"error: {error}", file=sys.stderr)
        return STORE_FAILURE


def _review(args: argparse.Namespace) -> int:
    claims = {c.claim_id: c for c in ClaimStore(args.out / "claims").latest()}
    tensions = {t.tension_id: t for t in TensionStore(args.out / "tensions").latest()}
    waiting = sorted(
        (
            r
            for r in ResolutionStore(args.out / "resolutions").latest()
            if r.status == "proposed"
            and (args.since is None or r.proposed_utc.date() >= args.since)
        ),
        key=lambda r: (r.proposed_utc, r.resolution_id),
    )
    if not waiting:
        print("no proposed resolutions awaiting review")
    else:
        print(f"{len(waiting)} proposed resolution(s) awaiting review")
        for resolution in waiting:
            print()
            print("\n".join(_review_block(resolution, tensions.get(resolution.tension_id), claims)))
    print()
    print(DISCLAIMER)
    return OK


def _review_block(
    resolution: Resolution, tension: Tension | None, claims: dict[str, Claim]
) -> list[str]:
    about = f"{tension.kind.value} {tension.subject}" if tension else "tension not in the store"
    lines = [
        f"[{resolution.hypothesis.value}] {about}",
        f"  resolution {resolution.resolution_id}, proposed {resolution.proposed_utc.isoformat()} "
        f"by {resolution.proposer_version}",
    ]
    if resolution.claim_id is not None:
        lines.append(f"  names claim {_short(resolution.claim_id)}")
    if resolution.evidence_span:
        lines.append(f'  proposed explanation: "{_flat(resolution.evidence_span)}"')
    columns = [
        _claim_column(claims.get(claim_id), claim_id)
        for claim_id in _tension_claim_ids(tension, claims)
    ]
    for left, right in zip_longest(columns[0::2], columns[1::2], fillvalue=[]):
        rows = zip_longest(left, right, fillvalue="")
        lines.append("")
        lines.extend(f"  {a.ljust(COLUMN_WIDTH)} | {b}".rstrip() for a, b in rows)
    command = f"precedent dream promote --resolution {resolution.resolution_id}"
    lines += ["", f"  accept: {command} --accept", f"  reject: {command} --reject"]
    return lines


def _tension_claim_ids(tension: Tension | None, claims: dict[str, Claim]) -> list[str]:
    """The tension's claim ids, oldest claim first (a claim missing from the store sorts last)."""
    if tension is None:
        return []

    def order(claim_id: str) -> tuple[bool, float, str]:
        claim = claims.get(claim_id)
        return (claim is None, claim.valid_from_utc.timestamp() if claim else 0.0, claim_id)

    return sorted(tension.claim_ids, key=order)


def _claim_column(claim: Claim | None, claim_id: str) -> list[str]:
    if claim is None:
        return [f"claim {_short(claim_id)}: not in the claim store"]
    span = f'"{_flat(claim.evidence_span)}"' if claim.evidence_span else "(rule-built, no span)"
    text = [
        f"{claim.attribute.value} {claim.period.value}: {claim.value.value}",
        f"doc {claim.source_doc_id}",
        f"valid from {claim.valid_from_utc.date().isoformat()}",
        span,
    ]
    return [line for part in text for line in textwrap.wrap(part, COLUMN_WIDTH)]


def _flat(text: str | None) -> str:
    return " ".join((text or "").split())
