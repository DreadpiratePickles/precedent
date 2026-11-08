# Stage: 41_mechanism_fit

## Objective
Fit each edge's effect size on past events with a guarded estimator, so the graph supplies structure and the data supplies the numbers; no model is called in this phase and nothing is published.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/40_mechanism_graphs/output/graphs_<graph_version>.jsonl` | 4 | Authoritative for graph structure | Yes | All graphs |
| `stages/04_align/output/aligned_events.parquet` | 4 | Authoritative for events and `effective_session` | Yes | All records |
| Price history through the spine price store | 4 | Authoritative for returns | Yes | Asset panel |
| `references/mechanism/params.json` | 3 | Authoritative for the asset panel and `n_paths` | Yes | All |

## Process
1. Deterministic: align each past event of a graph's event key to its sessions and asset returns, using only data available at the event time.
2. Deterministic: estimate edge effects with the guarded fit (shrinkage and a minimum sample); an edge without enough history is left unfitted and reported, never given a default value.
3. Deterministic: write the fitted graph with its sample sizes.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/41_mechanism_fit/output/fits_<graph_version>.jsonl` | Fitted graph, one per event key | `42_mechanism_evaluate` |

## Verify
- Hand-computed fixture fits; a truncation-invariance test for every builder (design 4.5).
- `uv run pytest -q` exits 0.

## Approval
The owner reviews the guard settings before a fit is read as a result. No external action occurs in this stage.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. Output is written atomically.

Loop: single pass. A deterministic fit decided by tests, with no iteration toward a target.
