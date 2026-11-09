# Stage: 51_dream_detect

## Objective
Detect tensions between claims, and between claims and prices, and silent shifts in event rates, direction and links, as of one session's cutoff.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `<out>/claims/claims.jsonl` | 4 | Authoritative for claims; read through `claims_asof` at the cutoff | Yes | Claims visible at the cutoff |
| `stages/01_ingest/output/` event store | 4 | Authoritative for events and their `available_utc` (`first_seen_utc` folded in by the spine); read-only | For shifts and text-vs-price | Events up to the cutoff |
| Ripple link store at `stages/30_*/output/`, once built | 4 | Authoritative for links; read-only | For link tensions and churn | Links up to the cutoff |
| `references/dream/detector_params.json` | 3 | Authoritative for thresholds and windows | Yes | All sections |
| `references/dream/claim_vocabulary.json` | 3 | Authoritative for conflicting value pairs (`event_direction` is excluded from value conflicts, D72) | Yes | `conflicting_value_pairs` |
| `references/dream/resolve_prompt_v*.txt` | 3 | Authoritative for the resolution prompt; versioned in `proposer_version` | For model proposals | Latest version |
| Development and held-out resolution labels | 3 | Authoritative for scoring; read-only to the worker | For scoring | Label splits |
| `<out>/tensions/`, `<out>/shifts/`, `<out>/resolutions/` (one `.jsonl` each) | 4 | This stage's own stores; `<out>` is the `dream` command's `--out` directory (default `stages/52_dream_report/output`), which holds all four Dream stores | Yes | Existing lines |

## Process
1. Deterministic: take the session cutoff and read only what `claims_asof`, the event store and the link store show at that cutoff; nothing recorded later is read. `direction_drift` additionally reads `claims_history_asof` — every claim that began in the lookback window, expired or not (D71), since a history detector cannot run on the claims still valid alone.
2. Deterministic: detect tensions (value conflict, text versus price, asymmetric link, stale reliance) and shifts (event rate, direction drift, link churn) with the parameters file.
3. Deterministic: append new tensions keyed by the group id (`kind` + group key, D72): a changed membership appends a `restate_members` update, a recurrence of a closed group reopens it, a shift that fires again is re-fired in place, and a tension that no longer holds is closed by appending an update, each kind by its own test. `event_direction` takes no part in value conflicts (D72).
4. Judgment (model call, injected proposer): for each open tension the model proposes one hypothesis (`one_is_stale`, `different_scope`, `genuine_conflict`, `extraction_error`) with an evidence span. A proposal is `proposed` until the owner accepts or rejects it.
5. No detector output is ever used as a prediction.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<out>/tensions/tensions.jsonl` | Tension records, append-only, latest record per id wins | `52_dream_report`, `53_dream_evaluate` |
| `<out>/shifts/shifts.jsonl` | Shift records, append-only, latest record per id wins | `52_dream_report`, `53_dream_evaluate` |
| `<out>/resolutions/resolutions.jsonl` | Resolution records, append-only, latest record per id wins | `52_dream_report` |

Tensions and shifts are observations about documents and prices, not forecasts.

## Verify
- Truncation invariance: running a detector at a cutoff gives the same output whether or not later data exists.
- Hand-computed fixtures for every detector; every record validates against its model.
- Re-running a session adds no line.
- No model is called in tests or CI; the proposer is injected.
- `uv run pytest -q` exits 0.

## Approval
A human approves detector parameters and each resolution prompt version, and accepts or rejects each proposed resolution. Inputs are read-only. Calling a paid model API is an owner gate.

## Failure Behavior
Retry only transient transport failures, at most 3 times with exponential backoff and jitter. A proposal that fails validation is recorded and not retried with a different prompt inside this run. A failed read never becomes "no tensions". A conflict on a stored id stops the stage with the id. Output is written atomically.

## Loop
Kind: ratchet
Oracle: the resolution scorer on the resolution development labels; green means the model's hypothesis agrees with the owner's label at least 0.7 of the time. Green is not acceptance.
Oracle protection: the labels and the scorer are read-only to the worker (file permissions or a separate trusted checkout); a change to either stops the loop and goes to a human.
Held-out check: the held-out label split, run once at acceptance; the worker never sees it.
Unit per iteration: one prompt change.
Revises against: the per-hypothesis agreement and the first error lines of the previous scorer run.
Mutable / protected: `references/dream/resolve_prompt_v*.txt` only / the labels, the scorer, the schema, existing tests (new tests may be added).
Isolation: a dedicated branch or worktree cut from the recorded baseline commit; nothing inside the loop publishes, sends, spends or deploys.
Worker / judge: standard-tier model works; no model judge (the scorer is the oracle).
Runner: a script, not the model, counts iterations, enforces the budget, detects no progress and repeated failure, and keeps or reverts the declared mutable files.
State: a results table (one row per iteration with the agreement) and git commits; read first, updated last.
Context: fresh per iteration with the results table and the last failure as handoff, because the state lives on disk.
Stop: oracle green; or 10 iterations; or 8 hours wall time; or zero spend by default (a paid call needs the owner's yes); or no improvement after 4 iterations; or the same first error line 3 times in a row; or blocked (a valid exit).
On stop: the best-so-far prompt version is kept, the run is logged in the results table and `docs/STATUS.md`, and the owner is told.
Single-pass baseline: recorded on the first run (one pass of the starting prompt scored on the development labels) and compared with the loop's best at the end.
Numbers proposed; confirmed in Task 8.
