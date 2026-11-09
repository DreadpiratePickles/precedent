# Stage: 43_mechanism_cockpit

## Objective
Show a graph, its fitted effects and the evidence behind each edge to a person who can change the inputs, as history and structure and never as a forecast; no model is called in this phase and nothing is published.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/41_mechanism_fit/output/fits_<graph_version>.jsonl` | 4 | Authoritative for fitted graphs | Yes | All |
| `stages/42_mechanism_evaluate/output/comparison.parquet` | 4 | Authoritative for how the mechanism scored | Yes | All |
| `references/mechanism/graph_schema_v1.json` | 3 | Authoritative for the graph shape | Yes | All |

## Process
1. Deterministic: load graphs and fits with the core install only (no research extra).
2. Deterministic: render nodes, signed lagged edges, evidence spans and sample sizes; show the evidence label from `docs/STATUS.md`.
3. Deterministic: show, per asset row, the mechanism fan beside Precedent's p10 / median / p90 and both sample sizes (fit n, band n, and Precedent's `n_eff`); a suppressed row (no data, or a refused band) shows its reason and no fan. Name the underdetermined equations (node ids) and, when it happened, the fall-back of the residual covariance to its diagonal, in the text and on the page.
4. Deterministic: each slider runs over the fitted panel's 5th to 95th percentile of its node (embedded in the payload as `slider_ranges`), widened to include the query value; a node with no recorded range is pinned to its value. The residual draws use the Cholesky factor (or the clamped diagonal fallback) that Python computed once and embedded as `residual_factor`; the page never factorises.
5. Deterministic: apply the user's what-if inputs through the stored fits; nothing is refitted in the view.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| Cockpit view (local, not published) | Rendered page or terminal view | The owner |

## Verify
- The view renders from fixture graphs without network access.
- `uv run pytest -q` exits 0.

## Approval
The owner approves any publication of the view. Nothing is published, sent or deployed from this stage.

## Failure Behavior
A graph that fails validation is shown as an error with its id, never drawn partially. No retries.

Loop: single pass. A view of stored results; there is nothing to iterate toward.
