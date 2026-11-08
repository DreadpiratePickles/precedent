# Stage: 31_ripple_factors

## Objective
Propagate each session's abnormal-return shocks along the link graph as it stood at that session's cutoff, producing Ripple factors per (session, instrument), using code only.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `<data_dir>/ripple/links/links.jsonl`, read only through `graph_asof` | 4 | Authoritative for links and their validity windows | Yes | All links visible at each cutoff |
| `stages/04_align/output/aligned_events.parquet` | 4 | Authoritative for events and `effective_session` | Yes | Shock source events |
| `<data_dir>/ripple/abnormal_returns.parquet` (spine `abnormal_returns`, derived statistics) | 4 | Authoritative for realised shock sizes | Yes | Per (session, instrument) |
| `references/ripple_params.json` | 3 | Authoritative for alpha, hops, half-life and age cap | Yes | `ripple` section |

## Process
1. Deterministic: build each session's shock vector from aligned events and abnormal returns, using only what was known by that session's cutoff.
2. Deterministic: read the graph with `graph_asof` at the cutoff; build the adjacency `A[receiver, source]` and normalise each receiver's incoming weights to sum to 1.
3. Deterministic: compute `ripple = sum over k = 1..hops of alpha^k A^k shock`, per channel: `customer_of` feeds `ripple_customer`, `supplier_of` feeds `ripple_supplier`, `peer_of` feeds `ripple_peer`, `input_commodity` and `output_commodity` feed `ripple_commodity`. `country_exposure` has no factor column.
4. Deterministic: decay shocks by age in sessions (half-life, age cap) and write the panels.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<data_dir>/ripple/ripple_factors.parquet` | Index `(datetime, instrument)`, one column per channel; persistent store | `32_ripple_evaluate` |
| `<data_dir>/ripple/shocks.parquet` | Index `(datetime, instrument)`, propagated shock sizes; persistent store | `32_ripple_evaluate`, `33_ripple_report` |
| `<data_dir>/ripple/MANIFEST.json` | JSON: `evidence_label` (`SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL` or `NO LIVE RUN`) and input hashes; persistent | `33_ripple_report` |
| `stages/31_ripple_factors/output/run_manifest.json` | JSON: code version, parameters, input hashes for this run | The owner |

## Verify
- Every factor builder ships a truncation-invariance test (design 4.5), using `assert_truncation_invariant` from `tests/fakes.py`.
- Hand-computed propagation values on a small fixture graph.
- `uv run pytest -q` exits 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. Review factor definitions before they enter a model.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. Outputs are written atomically.

Loop: single pass. Deterministic code decided by tests (design 10.2).
