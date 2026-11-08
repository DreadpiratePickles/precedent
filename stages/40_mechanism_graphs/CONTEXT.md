# Stage: 40_mechanism_graphs

## Objective
Produce one validated causal graph per event key, from event variable to asset panel, first from templates and later from a model that reads documents; no model is called in this phase and nothing is published.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `references/mechanism/intermediates.json` | 3 | Authoritative for the intermediate node vocabulary | Yes | All entries |
| `references/mechanism/params.json` | 3 | Authoritative for the asset panel until `engine.asset_panel` exists in `references/precedent_params.json` | Yes | `asset_panel` |
| `references/mechanism/graph_schema_v1.json` | 3 | Authoritative for the graph shape; generated from `graph_json_schema()` | Yes | All |
| `references/mechanism/templates/<event_type>.json` | 3 | Authoritative for the hand-written template edges of one event type; `rationale` is documentation | Yes | `edges` |
| `references/mechanism/graph_prompt_v*.txt` | 3 | Authoritative for the graph proposer's instructions (Task 8) | Model phase only | Latest version |
| Event keys from the Precedent track or the spine | 4 | Authoritative for which graphs to build | Yes | `EventKey` |
| Gold set (40 development and 20 held-out passages) | 3 | Authoritative for scoring; read-only to the worker | Model phase only | Design Mechanism section |

## Process
1. Deterministic: build the node vocabulary from the intermediates file, the asset panel and the event types.
2. Deterministic: build template graphs for each event type (graph version starts with `template-`).
3. Judgment (model call, a later task, not in this phase): propose edges from a passage; each edge carries `source_doc_id` and an `evidence_span`.
4. Deterministic: validate every graph against `CausalGraph`; a node outside the vocabulary, a cycle, an asset that causes a non-asset, or a model edge without evidence is rejected, never repaired or added.
5. Deterministic: append valid graphs to the graph store, keyed by (event key, `graph_version`); never overwrite.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/40_mechanism_graphs/output/graphs_<graph_version>.jsonl` | `CausalGraph`, one JSON Lines file per `graph_version` | `41_mechanism_fit` |
| `stages/40_mechanism_graphs/output/dropped_edges.jsonl` | Edge, reason, document id (model phase) | Human review |

## Verify
- `uv run pytest -q tests/mechanism` exits 0: cycle, asset-causes-event, unknown-node, evidence-required, schema-file and store tests.
- The stored `graph_schema_v1.json` equals `graph_json_schema()`.
- Model phase: edge precision and recall on the development gold set, then the held-out split once.

## Approval
The owner approves the intermediates list, the asset panel and `n_paths`, and labels the gold set with the agent (Task 8 gate) before any model proposes a graph. Nothing is published or sent from this stage.

## Failure Behavior
An invalid graph stops the append with the graph id and the failing rule; the store is written atomically and all or nothing. A key already stored with different content raises `GraphConflictError`. No retries. A missing reference file stops the stage.

## Loop
Proposed; runs only after the owner gate in Task 8 (gold set labelled with the owner).

Kind: ratchet
Oracle: edge precision and recall of `parse_graph` on the 40-passage development gold set; green means both are at or above the thresholds the owner sets at the Task 8 gate. Green is not acceptance.
Oracle protection: the gold set and the scorer are read-only to the worker; a change stops the loop and goes to a human.
Held-out check: the 20 held-out passages, run once at acceptance.
Unit per iteration: one prompt change.
Revises against: the per-edge precision and recall table and the dropped-edge counts of the previous run.
Mutable / protected: `references/mechanism/graph_prompt_v*.txt` only / the gold set, the scorer, the schema, existing tests (new tests may be added).
Isolation: a branch cut from the recorded baseline commit; nothing inside the loop publishes, deploys or spends.
Worker / judge: a standard-tier model works; no model judge.
Runner: a script counts iterations, enforces the stops, and keeps or reverts the prompt.
State: the results table and git commits; read first, updated last.
Context: fresh per iteration, with the results table as the handoff.
Stop: oracle green; or 10 iterations; or 8 hours wall time; or any spend (the cost budget is zero by default; spending needs the owner's yes); or no improvement after 3 iterations; or the same first error line 3 times in a row; or blocked (a valid exit).
On stop: the best-so-far prompt is kept, the run is logged in `docs/STATUS.md`, and the owner is told.
Single-pass baseline: the first scored run of prompt v1.
