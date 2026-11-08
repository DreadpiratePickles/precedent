# Stage: 21_precedent_evaluate

## Objective
Score the baseline ladder (B0 pooled, B1 rules-typed exact key, B2 model-typed exact key, B3 key blended with canonical-text similarity) against the precedent estimate for each outcome type, in time order, so the precedent card is shown only where it beats a baseline.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `library/rows_*.parquet`, `library/placebo_asset.parquet` | 4 | Authoritative for outcomes | Yes | One file per outcome type |
| `library/events.parquet`, `library/sessions.parquet` | 4 | Authoritative for event time and key | Yes | `first_seen_utc`, key; `canonical_text` for B3 only |
| `library/MANIFEST.json` | 4 | Authoritative for file hashes | Yes | Verified before any file is read |
| `references/precedent_params.json` | 3 | Authoritative for `top_k`, `min_neff`, `shrink_k`, `embedding_weight` | Yes | `engine` section |
| Injected `TextEmbedder` (`engine.embedder`) | 3 | Test double or owner-approved model | B3 only | `embed`, `release_date` |

## Process
1. Deterministic: verify the manifest hashes, then load the library.
2. Deterministic: walk events in time order; for each, take only rows whose outcome was known before the event (an expanding window).
3. Deterministic: B0 is the unconditional placebo distribution; B1 is an exact match on a rules-based event type (design 8.4), B2 the same match on model-typed events; B3 blends the key with embedder similarity of `canonical_text` at `embedding_weight` and raises `ValueError` if the column or a query text is missing; the precedent estimate uses the top `top_k` key matches.
4. Deterministic: score each rung per outcome type with pinball loss per quantile and band coverage, with Wilson intervals at `wilson_z`.
5. Deterministic: write one table of scores per outcome type and a short report.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/21_precedent_evaluate/output/scores.parquet` | Rung (B0 to B3), outcome type, metric, interval, n | `23`, human review |
| `stages/21_precedent_evaluate/output/report.md` | Plain text, with the evidence label | Human review |

## Verify
- A score for an event never uses a row that resolved after the event's `first_seen_utc`.
- Truncating the library at a date gives the same earlier scores.
- The report states `SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL` or `NO LIVE RUN`.
- `uv run pytest -q` exits 0.

## Approval
A human reads the report before any claim of skill is made. No external action occurs in this stage.

## Failure Behavior
A manifest mismatch stops the stage before any file is read. An outcome type with fewer than `min_neff` effective matches is reported as insufficient, never scored. No retries. Output is written atomically.

Loop check: scoring B0 and B1 is a single pass over a fixed library, decided by tests, not a loop. Tuning the embedding similarity weight of rung B3 (Task 18, Step 6) is the one loop in this stage; its contract follows. Rungs B2 (exact key on model-typed events) and B3 (key blended with canonical-text similarity) are scored the same single-pass way, and need an owner gate before any model is downloaded or called.

## Loop
Every number below is `proposed`.
Kind: ratchet
Oracle: `score_precedent_ladder` (rungs B1 and B3) on the development events with the injected embedder; green means B3 pinball loss (asset) and Brier (binary) are each lower than at B1 by more than a noise tolerance of 0.5% relative, and `uv run ruff check . && uv run pytest -q` exits 0 (the regression check). Weight 0 equals B1 exactly, so a tie is not green. Green is not acceptance.
Oracle protection: `src/precedent/engine/evaluate.py`, the tests and the held-out year are read-only to the worker, enforced by a separate checkout; a change to any of them stops the loop and goes to a human.
Held-out check: the most recent year of events, scored once at acceptance; the worker never sees it. Contamination control: rung B2 (no embedding) is scored beside B3 there, because embedder candidates postdate January 2024 and may have seen those events.
Unit per iteration: one similarity-weight value in `[0, 1]`.
Revises against: the ladder table of the previous iteration (pinball loss per quantile, coverage, Brier, `n_refused`).
Mutable / protected: `references/precedent_params.json`, the `embedding_weight` key of its `engine` section only / everything else (new tests may be added).
Isolation: a branch cut from the recorded baseline commit; nothing inside the loop publishes, sends, spends, downloads or deploys.
Worker / judge: standard-tier model works; no model judge (the ladder scores are the oracle).
Runner: `scripts/run_precedent_weight_ratchet.py` (proposed; to be written at Step 6) counts iterations, enforces the budget, detects no progress and repeated failure, and keeps or reverts the mutable section.
State: `stages/21_precedent_evaluate/output/<run_id>/results.jsonl` (one row per iteration) and git commits; read first, updated last.
Context: fresh per iteration with the results file and the last ladder table as handoff, because the state lives on disk.
Stop: oracle green; or 10 iterations; or 4 hours wall time; or zero spend (a paid call or a download needs the owner's yes); or no improvement after 3 iterations; or the same first error line 3 times in a row; or blocked (a valid exit).
On stop: the best weight so far is kept, the run is logged in the results file and `docs/STATUS.md`, and the owner is told; if no weight beats B1, B1 and `RulesTyper` stay the defaults.
Single-pass baseline: B1 at weight 0, scored once on the development events and compared with the loop's best at the end.
