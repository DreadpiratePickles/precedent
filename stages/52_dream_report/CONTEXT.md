# Stage: 52_dream_report

## Objective
Run the nightly dream for one session: build claims, expire and detect as of the session cutoff, and write what changed in tensions, shifts, claims and resolutions as a readable note and a structured file.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `--events DIR` (default `<data_dir>/events`): the spine `EventStore` root; one `events_<version>.jsonl` file, chosen with `--parser-version` when several exist | 4 | Authoritative for events; read-only; GDELT records pass through `dedupe_gdelt_records` | Yes | Records available and ingested by the session cutoff |
| `--links DIR` (default `<data_dir>/ripple/links` when it holds `links.jsonl`): the Ripple `LinkStore` root | 4 | Authoritative for links; read-only | No | Links valid from by the cutoff |
| `--prices DIR` with `--market SYM`: Parquet prices written by `precedent prices import` | 4 | Authoritative for prices; read-only | No | Close-to-close returns up to the session; absent, or no row for the session, means the text-versus-price detector is skipped and counted (the note's title line then says `prices: none for <session>`) |
| `--out DIR` (default `stages/52_dream_report/output`): `claims/`, `tensions/`, `shifts/`, `resolutions/` | 4 | Authoritative for Dream's own records; append-only | Yes | Existing lines |
| `references/dream/detector_params.json` | 3 | Authoritative for expiry, windows, thresholds and `shift_rereport_growth` | Yes | All sections |
| `references/dream/claim_vocabulary.json` | 3 | Authoritative for the 8-K item map and conflicting value pairs | Yes | Claims, value conflicts |
| `references/nyse_holidays.json` | 3 | Authoritative for the session calendar and cutoffs | Yes | The session axis |

## Process
1. Deterministic: take the cutoff of `--date` from the committed calendar (a date that is not a session exits 2); keep the records available and ingested, and the links valid, by that cutoff.
2. Deterministic: append the rule claims of those records and links (a known id is skipped), then expire each visible claim with no end that began 15 months or more before the cutoff, by appending a superseding record with reason `rule:annual_15_months`.
3. Deterministic: run the tension and shift detectors as of the cutoff, with one abnormal-returns panel built per invocation; append new tensions; append a new shift or re-fire a known one when its statistic grew past `shift_rereport_growth`.
4. Deterministic: close every open or acknowledged tension, opened by this session or earlier, whose claims are no longer all visible at the cutoff.
5. Deterministic: read the stores back and write the sections: opened and closed tensions, silent shifts, expired claims, new claims and proposed resolutions awaiting review. No model is called and no wall-clock time is written.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<out>/dreams/dream_<session>.md` | Markdown, one section per kind of change, each item with its documents and evidence spans | The owner |
| `<out>/dreams/dream_<session>.jsonl` | A header line, one line per item in note order (with ids and its `text`), a footer line | `53_dream_evaluate` |
| `<out>/claims/`, `tensions/`, `shifts/`, `resolutions/` | Append-only JSON Lines stores; the latest record per id wins | This stage, `53_dream_evaluate` |

Tensions and shifts are observations about documents and prices, not forecasts. The note describes what documents and prices show and names no expected move.

## Verify
- Every item in the `.md` appears in the `.jsonl` and the reverse; every item line names its documents.
- Running the stage twice for the same session gives the same bytes and adds no store line.
- A record ingested after the cutoff, or a claim recorded after it, does not appear in that session's note, so a backfilled dream equals the live one.
- The inputs are byte-identical after a run.
- `uv run pytest -q` exits 0.

## Approval
The owner reads the note; no action follows from it automatically. Inputs are read-only. Nothing is sent, posted or published by this stage.

## Failure Behavior
Exit 2 for a usage error: `--date` missing or not a session, an events directory that is missing or holds no record, several parser versions without `--parser-version` (the versions found are named) or an unknown one, `--prices` without `--market` or a market with no file, a bad flag. Exit 1 for a store conflict (a stored id with different content) or an invalid input such as returns with no session in common with the axis, with the file and id on stderr. Nothing is written to `dreams/` on a non-zero exit; stores may already have appended, which is safe because the next run skips them. No retries. Output is written atomically.

Scheduled: not until the pipeline runs live (spec 10.2); see Task 9.
Loop: single pass. The report is a deterministic diff decided by tests, with no model and no oracle to iterate against.
