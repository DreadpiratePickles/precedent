# Stage: 52_dream_report

## Objective
Write the nightly diff of what changed in tensions, shifts and resolutions since the previous session, as a readable note and a structured file.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/51_dream_detect/output/tensions.jsonl` | 4 | Authoritative for tensions; read-only | Yes | Records changed since the previous session |
| `stages/51_dream_detect/output/shifts.jsonl` | 4 | Authoritative for shifts; read-only | Yes | Records changed since the previous session |
| `stages/51_dream_detect/output/resolutions.jsonl` | 4 | Authoritative for resolutions; read-only | Yes | Records changed since the previous session |
| `stages/50_dream_claims/output/claims.jsonl` | 4 | Authoritative for the claims a tension cites; read-only | Yes | Claims named by a tension |
| `references/dream/detector_params.json` | 3 | Authoritative for `shift_rereport_growth` and the report thresholds | Yes | Reporting |
| `stages/52_dream_report/output/` | 4 | This stage's own earlier reports | Yes | Previous session's file |

## Process
1. Deterministic: compare the latest records with the previous session's report to find what is new, closed, acknowledged, re-fired or decided.
2. Deterministic: re-report a shift only when its statistic has grown by at least `shift_rereport_growth` of the last reported one.
3. Deterministic: write `dream_<session>.jsonl` (one line per change) and `dream_<session>.md` (the same changes for a reader, each with the evidence span and the source document id).

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/52_dream_report/output/dream_<session>.md` | Markdown, one section per kind of change | The owner |
| `stages/52_dream_report/output/dream_<session>.jsonl` | One change per line, with record ids | `53_dream_evaluate` |

Tensions and shifts are observations about documents and prices, not forecasts. The note describes what documents and prices show and names no expected move.

## Verify
- Every change in the `.md` appears in the `.jsonl` and the reverse.
- Running the stage twice for the same session gives the same bytes.
- Each cited claim id exists in the claim store.
- `uv run pytest -q` exits 0.

## Approval
The owner reads the note; no action follows from it automatically. Inputs are read-only. Nothing is sent, posted or published by this stage.

## Failure Behavior
A missing or invalid input stops the stage with the file and cause; a failed read never becomes an empty report. No retries. Output is written atomically.

Scheduled: not until the pipeline runs live (spec 10.2); see Task 9.
Loop: single pass. The report is a deterministic diff decided by tests, with no model and no oracle to iterate against.
