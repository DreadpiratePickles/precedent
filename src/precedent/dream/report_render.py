"""What the nightly dream says: pure functions from store records to the note and its twin.

Nothing here reads a store, a file or a clock, so the same records always give the same bytes.
`report.py` reads the stores and calls `build_sections`, `markdown_lines` and `jsonl_lines`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from precedent.dream.claim_schema import Claim, Resolution, Shift, Tension, TensionKind, link_id_for
from precedent.ripple.link_store import Link

DISCLAIMER = "Tensions and shifts are observations about documents and prices, not forecasts."
SPAN_LIMIT = 120
# (type in the .jsonl, key in the header counts, heading in the note)
SECTIONS = (
    ("opened_tension", "opened_tensions", "Opened tensions"),
    ("closed_tension", "closed_tensions", "Closed tensions"),
    ("shift", "silent_shifts", "Silent shifts"),
    ("expired_claim", "expired_claims", "Expired claims"),
    ("new_claim", "new_claims", "New claims"),
    ("proposed_resolution", "proposed_resolutions", "Proposed resolutions awaiting review"),
)
Item = dict[str, Any]  # one note line plus its ids; "text" is the line itself


@dataclass(frozen=True)
class Names:
    """Lookups for readable lines: claims by id, and link claim subjects by name."""

    claims: dict[str, Claim]
    links: dict[str, str]

    def subject(self, subject: str) -> str:
        return self.links.get(subject, subject if len(subject) <= 20 else subject[:12])


def build_sections(
    stored: list[Claim],
    tensions: list[Tension],
    shifts: list[Shift],
    resolutions: list[Resolution],
    links: list[Link],
    session: date,
    cutoff: datetime,
    previous: datetime | None,
) -> dict[str, list[Item]]:
    """The note's items by section type, from every record of each store in file order.

    Everything is read as of `cutoff` (a record recorded later is invisible), so a replay of an
    earlier session after later re-fires, closings or decisions gives the same items.
    `previous` is the previous session's cutoff.
    """
    stored = [c for c in stored if c.recorded_utc <= cutoff]
    latest_claim = {c.claim_id: c for c in stored}
    first_claim: dict[str, Claim] = {}
    for claim in stored:
        first_claim.setdefault(claim.claim_id, claim)
    tensions = list({t.tension_id: t for t in tensions if t.recorded_utc <= cutoff}.values())
    # A shift is listed when any record visible now was detected this session; that record shows.
    changed_here: dict[str, Shift] = {}
    for shift in shifts:
        if shift.recorded_utc <= cutoff and shift.detected_session == session:
            changed_here[shift.shift_id] = shift
    # Like every store (D13), by when a line was recorded: a proposal stored after the cutoff is
    # invisible, and a resolution decided after it is still proposed as of it.
    resolutions = list(
        {r.resolution_id: r for r in resolutions if r.recorded_utc <= cutoff}.values()
    )
    names = Names(
        latest_claim,
        {
            link_id_for(link.source_node, link.receiver_node, link.relation, link.source_doc_id): (
                f"{link.source_node} -> {link.receiver_node}"
            )
            for link in links
        },
    )

    def in_window(moment: datetime) -> bool:
        return moment <= cutoff and (previous is None or moment > previous)

    def claim_order(claim: Claim) -> tuple[str, ...]:
        return (claim.subject, claim.attribute.value, claim.source_doc_id, claim.claim_id)

    opened = sorted(
        (t for t in tensions if t.opened_session == session),
        key=lambda t: (t.kind.value, t.subject, t.tension_id),
    )
    closed = sorted(
        (t for t in tensions if t.closed_session == session),
        key=lambda t: (t.kind.value, t.subject, t.tension_id),
    )
    changed = sorted(
        changed_here.values(),
        key=lambda s: (s.kind.value, s.subject),
    )
    expired = sorted(
        (c for c in latest_claim.values() if c.expiry_reason and in_window(c.recorded_utc)),
        key=claim_order,
    )
    new = sorted((c for c in first_claim.values() if in_window(c.recorded_utc)), key=claim_order)
    proposed = sorted(
        (r for r in resolutions if r.status == "proposed"),
        key=lambda r: (r.tension_id, r.resolution_id),
    )
    return {
        "opened_tension": [_tension_item("opened_tension", t, names) for t in opened],
        "closed_tension": [_tension_item("closed_tension", t, names) for t in closed],
        "shift": [_shift_item(s, names) for s in changed],
        "expired_claim": [_claim_item("expired_claim", c, names) for c in expired],
        "new_claim": [_claim_item("new_claim", c, names) for c in new],
        "proposed_resolution": [_resolution_item(r, tensions) for r in proposed],
    }


def _cite(claim: Claim) -> str:
    span = " ".join((claim.evidence_span or "").split())
    if len(span) > SPAN_LIMIT:
        span = span[: SPAN_LIMIT - 3] + "..."
    return f'({claim.source_doc_id}, "{span}")' if span else f"({claim.source_doc_id})"


def _tension_item(kind: str, tension: Tension, names: Names) -> Item:
    cited = sorted(
        (names.claims[c] for c in tension.claim_ids if c in names.claims),
        key=lambda c: (c.value.value, c.source_doc_id),
    )
    details = tension.details
    if tension.kind is TensionKind.value_conflict:
        by_value: dict[str, list[str]] = {}
        for claim in cited:
            by_value.setdefault(claim.value.value, []).append(_cite(claim))
        sides = " vs ".join(f"{value} {' '.join(cites)}" for value, cites in by_value.items())
        body = f"{tension.subject} {details['attribute']} {details['period']}: {sides}"
    elif tension.kind is TensionKind.text_vs_price:
        move, z = float(details["abnormal_return"]), float(details["z"])
        body = (
            f"{tension.subject}: event_direction {details['claim_value']} {_cite(cited[0])}, "
            f"abnormal return {move:+.1%} (z {z:+.1f})"
        )
    elif tension.kind is TensionKind.asymmetric_link:
        pair = f"{details['source_node']} -> {details['receiver_node']}"
        body = f"{pair}: customer_of link with no supplier_of link back {_cite(cited[0])}"
    else:
        body = (
            f"{names.subject(tension.subject)}: link {details['age_sessions']} sessions old, "
            f"newer filing {details['newer_doc_id']} from the same source {_cite(cited[0])}"
        )
    if len(cited) < len(tension.claim_ids):
        body += f" (claims missing from the claim store: {len(tension.claim_ids) - len(cited)})"
    return {
        "type": kind,
        "text": f"- [{tension.kind.value}] {body}",
        "tension_id": tension.tension_id,
        "kind": tension.kind.value,
        "subject": tension.subject,
        "claim_ids": sorted(tension.claim_ids),
        "details": details,
        "opened_session": tension.opened_session.isoformat(),
    }


def _shift_item(shift: Shift, names: Names) -> Item:
    statistic, threshold = shift.statistic, shift.threshold
    detail = {
        "event_rate": f"event rate z {statistic:.1f} over {shift.window_sessions} sessions "
        f"against the baseline before them (threshold {threshold:.1f})",
        "direction_drift": f"CUSUM {statistic:+.1f} over {shift.window_sessions} sessions "
        f"(threshold {threshold:.1f})",
        "link_churn": f"link set distance {statistic:.2f} from {shift.window_sessions} sessions "
        f"ago (threshold {threshold:.2f})",
    }[shift.kind.value]
    return {
        "type": "shift",
        "text": f"- [{shift.kind.value}] {names.subject(shift.subject)}: {detail}",
        "shift_id": shift.shift_id,
        "kind": shift.kind.value,
        "subject": shift.subject,
        "statistic": statistic,
        "threshold": threshold,
        "window_sessions": shift.window_sessions,
    }


def _claim_item(kind: str, claim: Claim, names: Names) -> Item:
    tag = "expired" if kind == "expired_claim" else "claim"
    reason = f"; reason: {claim.expiry_reason}" if kind == "expired_claim" else ""
    line = (
        f"- [{tag}] {names.subject(claim.subject)} {claim.attribute.value} {claim.period.value}: "
        f"{claim.value.value} {_cite(claim)}{reason}"
    )
    item: Item = {
        "type": kind,
        "text": line,
        "claim_id": claim.claim_id,
        "subject": claim.subject,
        "attribute": claim.attribute.value,
        "period": claim.period.value,
        "value": claim.value.value,
        "source_doc_id": claim.source_doc_id,
    }
    if kind == "expired_claim":
        item["expiry_reason"] = claim.expiry_reason
    return item


def _resolution_item(resolution: Resolution, tensions: list[Tension]) -> Item:
    span = " ".join((resolution.evidence_span or "").split())
    cite = (
        f'({resolution.proposer_version}, "{span}")' if span else f"({resolution.proposer_version})"
    )
    tension = next((t for t in tensions if t.tension_id == resolution.tension_id), None)
    about = f"{tension.kind.value} {tension.subject}" if tension else resolution.tension_id[:12]
    return {
        "type": "proposed_resolution",
        "text": f"- [{resolution.hypothesis.value}] {about}: proposed explanation {cite}",
        "resolution_id": resolution.resolution_id,
        "tension_id": resolution.tension_id,
        "hypothesis": resolution.hypothesis.value,
    }


def markdown_lines(header: dict[str, Any], sections: dict[str, list[Item]]) -> list[str]:
    gap = "yes" if header["feed_gap"] else "no"
    prices = (
        f"; prices: none for {header['prices_none_for']}" if "prices_none_for" in header else ""
    )
    lines = [
        f"# Dream for {header['session']}   "
        f"(documents ingested: {header['documents_ingested']}; feed gap: {gap}{prices})",
        "",
    ]
    for kind, _, heading in SECTIONS:
        lines.append(f"## {heading} ({len(sections[kind])})")
        lines.extend(item["text"] for item in sections[kind])
        lines.append("")
    return [*lines, DISCLAIMER]


def jsonl_lines(header: dict[str, Any], sections: dict[str, list[Item]]) -> list[str]:
    rows = [header]
    for kind, _, _ in SECTIONS:
        rows.extend(sections[kind])
    rows.append({"type": "footer", "disclaimer": DISCLAIMER})
    return [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
