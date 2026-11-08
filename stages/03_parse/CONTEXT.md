# Stage: 03_parse

## Objective
Turn each triaged article into a validated `NewsEvent` by running a language model constrained to the event schema, so the model reads and structures text and never forecasts.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/02_triage/output/triaged.jsonl` | 4 | Authoritative for which articles to parse | Yes | All records |
| `references/` event schema and taxonomy | 3 | Authoritative for allowed fields and enums | Yes | Design 3.3 |
| Parser prompt and adapter configuration | 3 | Authoritative for parser behaviour; versioned in `parser_version` | Yes | Design 5.2 to 5.4 |
| Gold set (development and held-out splits) | 3 | Authoritative for scoring; read-only to the worker | For scoring | Design 5.5 |

## Process
1. Deterministic: build the prompt from the article text and schema; request output under a JSON grammar.
2. Judgment (model call, injected extractor): the model emits the event object. The schema has no outcome field.
3. Deterministic: validate the object against the schema; reject unknown enums; check the evidence span occurs in the text.
4. Deterministic: wrap the object in the record envelope with `parser_version`, keyed by (`article_id`, `parser_version`); append, never overwrite.
5. The rules typer (`precedent typers`, `PRECEDENT_TYPER=rules`) is deterministic and is single pass: it has no loop and needs none.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/03_parse/output/events.jsonl` | Event record (design 3.4), one vintage per `parser_version` | `04_align` |
| `stages/03_parse/output/parse_failures.jsonl` | `article_id`, error line, raw output hash | Human review |

## Verify
- Schema validation of every record; output that fails validation is a runtime bug, not a result.
- `score_parser` on the gold set reports every metric in design 5.5.
- A second run on the same machine and runtime returns identical output.
- No model is called in tests or CI; the extractor is injected.

## Approval
A human approves the parser version before any backtest pins it, and approves any use of a paid model API. Downloading model weights or adapters, calling a paid API and renting compute are owner gates.

## Failure Behavior
Retry only transient transport failures, at most 3 times with exponential backoff and jitter. A model output that fails validation is recorded in `parse_failures.jsonl` and not retried with a different prompt inside this run. An empty or failed parse never becomes "no event".

## Loop
Kind: ratchet
Oracle: `score_parser` on the gold development split; green means every parser threshold in design 5.5 is met (evidence span at least 95%, `event_type` agreement at least 80%, impacted-entity F1 at least 0.75, `severity` within 1 step at least 80%, direction sign at least 85%, same output on a second run 100%). Green is not acceptance.
Oracle protection: the gold set and the scorer are read-only to the worker (file permissions or a separate trusted checkout); a change to either stops the loop and goes to a human.
Held-out check: the gold held-out split, run once at acceptance; the worker never sees it.
Unit per iteration: one prompt change or one adapter change.
Revises against: the per-metric output and the first error lines of the previous `score_parser` run.
Mutable / protected: the parser prompt and adapter configuration, declared by Task 20 / the gold set, the scorer, the schema, existing tests (new tests may be added).
Isolation: a dedicated branch or worktree cut from the recorded baseline commit; nothing inside the loop publishes, sends, spends or deploys.
Worker / judge: standard-tier model works; no model judge (the scorer is the oracle).
Runner: a script, not the model, counts iterations, enforces the budget, detects no progress and repeated failure, and keeps or reverts the declared mutable files.
State: a results table (one row per iteration with the metrics) and git commits; read first, updated last.
Context: fresh per iteration with the results table and the last failure as handoff, because the state lives on disk.
Stop: oracle green; or 12 iterations; or 8 hours wall time; or zero spend by default (a paid call needs the owner's yes); or no improvement after 4 iterations; or the same first error line 3 times in a row; or blocked (a valid exit).
On stop: the best-so-far parser version is kept, the run is logged in the results table and `docs/STATUS.md`, and the owner is told.
Single-pass baseline: recorded on the first run (one pass of the starting prompt scored on the development split) and compared with the loop's best at the end.
