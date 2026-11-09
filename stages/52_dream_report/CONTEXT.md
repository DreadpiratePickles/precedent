# Stage: 52_dream_report

## Objective
Run the nightly dream for one session: build claims, expire and detect as of the session cutoff, and write what changed in tensions, shifts, claims and resolutions as a readable note and a structured file.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `--events DIR` (default `<data_dir>/events`): the spine `EventStore` root; one `events_<version>.jsonl` file, chosen with `--parser-version` when several exist | 4 | Authoritative for events; read-only; GDELT records pass through `dedupe_gdelt_records` | Yes | Records with `available_utc` by the session cutoff (`ingested_utc` is no gate, D69) |
| `--links DIR` (default `<data_dir>/ripple/links` when it holds `links.jsonl`): the Ripple `LinkStore` root | 4 | Authoritative for links; read-only | No | Links valid from by the cutoff |
| `--prices DIR` with `--market SYM`: Parquet prices written by `precedent prices import` | 4 | Authoritative for prices; read-only | No | Close-to-close returns up to the session; absent, or no row for the session, means the text-versus-price detector is skipped and counted (the JSONL header then carries `prices_none_for` and the note's `## Not run` section says so, D75) |
| `--out DIR` (default `stages/52_dream_report/output`): `claims/`, `tensions/`, `shifts/`, `resolutions/` | 4 | Authoritative for Dream's own records; append-only | Yes | Existing lines |
| `<out>/manifest.json` | 4 | Last session dreamed, parser version, vocabulary and parameter hashes (D70); read-only to the owner | Yes | Whole file |
| `<out>/.lock` | 4 | Exclusive-command sentinel (D74); removed with the lock | Yes | Held or absent |
| `references/dream/detector_params.json` | 3 | Authoritative for expiry, windows, thresholds and `shift_rereport_growth` | Yes | All sections |
| `references/dream/claim_vocabulary.json` | 3 | Authoritative for the 8-K item map and conflicting value pairs | Yes | Claims, value conflicts |
| `references/nyse_holidays.json` | 3 | Authoritative for the session calendar and cutoffs | Yes | The session axis |

## Process
1. Deterministic: take the cutoff of `--date` from the committed calendar (a date that is not a session exits 2); keep the records with `available_utc` by that cutoff (`ingested_utc` is no gate, D69) and the links valid from by it.
2. Deterministic: append the rule claims of those records and links (a known id is skipped), then expire each visible claim with no end that began 15 months or more before the cutoff, by appending a superseding record with reason `rule:annual_15_months`.
3. Deterministic: run the tension and shift detectors as of the cutoff, with one abnormal-returns panel built per invocation; append new tensions; append a new shift or re-fire a known one when its statistic grew past `shift_rereport_growth`, when the last stored report is more than its own `window_sessions` old (a new episode), or when a `direction_drift` reverses sign (D73).
4. Deterministic: close every open or acknowledged tension, opened by this session or earlier, that no longer holds — each kind by its own test on the visible claims and the link graph (D72). The close record names its cause: a `lapsed` close reopens on a plain recurrence; a `resolution` close (the owner's `promote --accept` decision) stands until a claim the closing record does not hold joins the group (D77).
5. Deterministic: a session is dreamed once, in order (D70). The manifest `<out>/manifest.json` records the last session dreamed and the parser version, vocabulary and parameter hashes; a session that is out of order or after a gap is refused unless `--allow-gap` is passed, and rebuilding an existing dream happens on a scratch copy of the stores and exits 1 if any store would change, so a past vintage is never rewritten. The whole command holds an exclusive lock on `<out>/.lock`; a second dream command exits 1 (D74).
6. Deterministic: read the stores back and write the sections: opened and closed tensions, silent shifts, expired claims, new claims, proposed resolutions awaiting review and what was not run. Every store is read as of the cutoff by `recorded_utc`, resolutions included, so a proposal stored or decided after the cutoff is not in that session's note (D61). A `## Not run` section names every detector that did not run and why (D75); within 60 days of the committed calendar's end the header warns that it must be extended (M7). No model is called and no wall-clock time is written.
7. The owner's step, outside the nightly run: `precedent dream review [--since DATE]` prints the proposed resolutions with both claims' evidence spans side by side; `precedent dream promote --resolution ID --accept|--reject [--at ISO8601]` decides one. Accepting `one_is_stale` expires the older claim (or the named `claim_id`) with `expiry_reason` `resolution:<id>` and closes the tension; `extraction_error` expires the named claim with `extraction_error:<id>` and closes it; `different_scope` closes it; `genuine_conflict` marks it `acknowledged` and leaves it open. Rejecting records the decision only. The decision is appended last; deciding the same way twice adds no line, a different second decision exits 1, an unknown id exits 2.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<out>/dreams/dream_<session>.md` | Markdown, one section per kind of change, each item with its documents and evidence spans | The owner |
| `<out>/dreams/dream_<session>.jsonl` | A header line, one line per item in note order (with ids and its `text`), a footer line | `53_dream_evaluate` |
| `<out>/claims/`, `tensions/`, `shifts/`, `resolutions/` | Append-only JSON Lines stores; the latest record per id wins; `promote` appends lifecycle lines to them | This stage, `53_dream_evaluate` |

Tensions and shifts are observations about documents and prices, not forecasts. The note describes what documents and prices show and names no expected move.

## Verify
- Every item in the `.md` appears in the `.jsonl` and the reverse; every item line names its documents.
- Running the stage twice for the same session gives the same bytes and adds no store line; a session out of order, or after a gap without `--allow-gap`, is refused, and a replay that would append to a store exits 1 (D70).
- A record whose `available_utc` is after the cutoff, or a claim recorded after it, does not appear in that session's note, so a backfilled dream equals the live one (D69).
- The inputs are byte-identical after a run.
- `uv run pytest -q` exits 0.

## Approval
The owner reads the note; no action follows from it automatically. A resolution changes a claim or a tension only when the owner runs `dream promote`; `dream review` is read-only. Inputs are read-only. Nothing is sent, posted or published by this stage.

## Failure Behavior
Exit 2 for a usage error: `--date` missing or not a session, an events directory that is missing or holds no record, several parser versions without `--parser-version` (the versions found are named) or an unknown one, `--prices` without `--market` or a market with no file, a bad flag. Exit 1 for a store conflict (a stored id with different content) or an invalid input such as returns with no session in common with the axis, with the file and id on stderr. Nothing is written to `dreams/` on a non-zero exit; stores may already have appended, which is safe because the next run skips them. No retries. Output is written atomically.

Scheduled: `scripts/dream_nightly.sh` and a proposed launchd job (`docs/DREAM_SCHEDULE.md`) exist; the owner installs the schedule by hand, and only once the spine ingest has run live (spec 10.2). Not installed.
Loop: single pass. The report is a deterministic diff decided by tests, with no model and no oracle to iterate against.
