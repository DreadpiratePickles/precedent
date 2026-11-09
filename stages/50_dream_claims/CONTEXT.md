# Stage: 50_dream_claims

## Objective
Build dated, evidenced claims from the event and link stores and expire them when a later claim, a restatement or a rule says they no longer hold.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/01_ingest/output/` event store (`events_<parser_version>.jsonl`) | 4 | Authoritative for events and their `available_utc` (`first_seen_utc` folded in by the spine); read-only | Yes | Records available by the session cutoff |
| Ripple link store at `stages/30_*/output/`, once built | 4 | Authoritative for links between instruments; read-only | When links are claimed | Link records |
| `references/dream/claim_vocabulary.json` | 3 | Authoritative for attributes, periods, values and the 8-K item map | Yes | All sections |
| `references/dream/claim_schema_v1.json` | 3 | Authoritative for the stored claim shape | Yes | All |
| `references/dream/detector_params.json` | 3 | Authoritative for expiry windows (`annual_claim_expiry_months`) | Yes | Expiry |
| `references/dream/claims_prompt_v*.txt` | 3 | Authoritative for the model claims prompt; versioned in `extractor_version` | For model claims | Latest version |
| Development and held-out claim labels | 3 | Authoritative for scoring; read-only to the worker | For scoring | Label splits |
| `<out>/claims/claims.jsonl` (the `ClaimStore` root is `<out>/claims`) | 4 | This stage's own store; `<out>` is the `dream` command's `--out` directory (default `stages/52_dream_report/output`), which holds all four Dream stores | Yes | Existing lines |

## Process
1. Deterministic: read events and links; skip any record without `available_utc`. Knowledge time is `available_utc`, not the ingest clock: `ingested_utc` is audit-only in the spine and no gate (D69); an event claim's `recorded_utc` is its `valid_from_utc`.
2. Deterministic: map 8-K items to claims with the item map; these claims carry a `rules-` extractor version and need no evidence span.
3. Judgment (model call, injected extractor): where a claims prompt applies, the model emits claims in the schema, each with an evidence span that occurs in the source text. The model never forecasts and never states an outcome.
4. Deterministic: validate each claim against the schema; reject unknown enums, naive timestamps and a missing evidence span.
5. Deterministic: append new claims keyed by `claim_id`; a repeat of stored content is skipped.
6. Deterministic: expire by appending an updated record (`supersede`): a later claim on the same subject, attribute and period, a restatement, or a dated rule expiry. Expiry is a write, never a delete; the original line stays.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<out>/claims/claims.jsonl` | Claim records (`claim_schema_v1.json`), append-only, latest record per `claim_id` wins | `51_dream_detect` |
| `<out>/claim_failures.jsonl` | Source id, error line, raw output hash | Human review |

Tensions and shifts are observations about documents and prices, not forecasts. Claims state what a document said and when it became known.

## Verify
- Every stored claim validates against the schema; a failure is a runtime bug, not a result.
- `claims_asof` shows a claim only when `valid_from_utc <= cutoff < valid_to_utc` and it was recorded by the cutoff.
- Re-running the stage on the same inputs adds no line.
- No model is called in tests or CI; the extractor is injected.
- `uv run pytest -q` exits 0.

## Approval
A human approves the claim vocabulary, the item map and each claims prompt version before a result depends on them. Inputs are read-only. Calling a paid model API and downloading model weights are owner gates.

## Failure Behavior
Retry only transient transport failures, at most 3 times with exponential backoff and jitter. A model output that fails validation goes to `claim_failures.jsonl` and is not retried with a different prompt inside this run. A failed read never becomes "no claims". A conflict on a stored id stops the stage with the id. Output is written atomically.

## Loop
Kind: ratchet
Oracle: the claims scorer on the claim development labels; green means claim precision at least 0.8 and recall reported alongside it. Green is not acceptance.
Oracle protection: the labels and the scorer are read-only to the worker (file permissions or a separate trusted checkout); a change to either stops the loop and goes to a human.
Held-out check: the held-out label split, run once at acceptance; the worker never sees it.
Unit per iteration: one prompt change.
Revises against: the per-attribute precision and recall and the first error lines of the previous scorer run.
Mutable / protected: `references/dream/claims_prompt_v*.txt` only / the labels, the scorer, the schema, existing tests (new tests may be added).
Isolation: a dedicated branch or worktree cut from the recorded baseline commit; nothing inside the loop publishes, sends, spends or deploys.
Worker / judge: standard-tier model works; no model judge (the scorer is the oracle).
Runner: a script, not the model, counts iterations, enforces the budget, detects no progress and repeated failure, and keeps or reverts the declared mutable files.
State: a results table (one row per iteration with precision and recall) and git commits; read first, updated last.
Context: fresh per iteration with the results table and the last failure as handoff, because the state lives on disk.
Stop: oracle green; or 10 iterations; or 8 hours wall time; or zero spend by default (a paid call needs the owner's yes); or no improvement after 4 iterations; or the same first error line 3 times in a row; or blocked (a valid exit).
On stop: the best-so-far prompt version is kept, the run is logged in the results table and `docs/STATUS.md`, and the owner is told.
Single-pass baseline: recorded on the first run (one pass of the starting prompt scored on the development labels) and compared with the loop's best at the end.
Numbers proposed; confirmed in Task 8.
