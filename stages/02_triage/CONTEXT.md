# Stage: 02_triage

## Objective
Deduplicate stored articles, pre-match entities, and apply a daily cap, using code only, to produce the list of articles worth parsing.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| Article store from `01_ingest` | 4 | Authoritative for item identity and `first_seen_utc` | Yes | All records in scope |
| `references/` entity and country maps | 3 | Authoritative for pre-match rules | Yes | Entity pre-match |
| Triage parameters (daily cap, duplicate window) | 3 | Authoritative; every threshold is named, not hard-coded in prose | Yes | Parameters file |

## Process
1. Deterministic: group near-duplicate articles by normalised text hash within the duplicate window; keep the earliest `first_seen_utc` as the group head.
2. Deterministic: pre-match entity names to candidate instruments with the reference maps.
3. Deterministic: rank survivors by a documented rule and cut at the daily cap; log every dropped article with its reason.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/02_triage/output/triaged.jsonl` | One record per kept article: `article_id`, group head, candidate entities | `03_parse` |
| `stages/02_triage/output/dropped.jsonl` | `article_id` and reason | Human review |

## Verify
- Unit tests on duplicate grouping, cap behaviour and entity pre-match, with fixed fixtures.
- Re-running on the same store yields identical output.
- Every input article appears in exactly one of the two outputs.
- `uv run pytest -q` exits 0.

## Approval
Review the dropped-article reasons and the daily cap value. No external action occurs in this stage.

## Failure Behavior
A malformed record stops the stage with its `article_id` and the preserved cause; it is never silently skipped. No retries (no external calls). Output is rewritten atomically or not at all.

Loop: single pass. Deterministic code decided by tests; no oracle-driven iteration is needed (design 10.2).
