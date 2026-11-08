# Stage: 04_align

## Objective
Assign each event its `available_utc`, its `effective_session` and resolved instrument ids, using code only, so no row sees information that did not yet exist.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/03_parse/output/events.jsonl` | 4 | Authoritative for the parsed event and `first_seen_utc` | Yes | All records |
| `references/` session calendar, latency values, entity maps | 3 | Authoritative for cutoffs and matching | Yes | Design 4.1, 4.2 |
| Alignment convention (C1 default, C2, C3) | 3 | Authoritative; stated in the run manifest | Yes | Design 4.2 |

## Process
1. Deterministic: `available_utc = max(published_utc, first_seen_utc) + latency`. GDELT `SQLDATE` and EDGAR `filingDate` are never used.
2. Deterministic: `effective_session` is the first session whose decision cutoff is at or after `available_utc`; weekend and holiday news rolls forward.
3. Deterministic: match `impacted` names to instrument ids and record the match method; unmatched names are kept as unmatched.
4. Reject any record with a missing `first_seen_utc` or a naive timestamp.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/04_align/output/aligned_events.parquet` | Event record plus `available_utc`, `effective_session`, `resolved_impacts` | `05_factors` |
| `stages/04_align/output/unmatched.jsonl` | Name, article id, reason | Human review |

## Verify
- Leak tests: an event available after a cutoff never lands in that session; a truncated input gives the same earlier rows (design 4.5).
- Boundary cases: exactly at cutoff, weekend, holiday, daylight-saving change.
- `uv run pytest -q` exits 0.

## Approval
Review the convention and latency values before any model is trained on the output. No external action occurs in this stage.

## Failure Behavior
A record that cannot be aligned stops the stage with its identifier and cause; it is never given a default session. No retries. Output is written atomically.

Loop: single pass. Deterministic transform decided by tests (design 10.2).
