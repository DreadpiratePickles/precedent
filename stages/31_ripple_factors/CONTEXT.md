# Stage: 31_ripple_factors

## Objective
Propagate each session's abnormal-return shocks along the link graph as it stood at that session's cutoff, producing Ripple factors per (session, instrument), using code only.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| The graph's link store, read only through `graph_asof`: `<out_dir>/links/<graph>/links.jsonl` inside a `32_ripple_evaluate` run (the published copy of one graph is `<data_dir>/ripple/links/links.jsonl`) | 4 | Authoritative for links and their validity windows | Yes | All links visible at each cutoff |
| Event store: `<data_dir>/events/`, the file for the run's `--parser-version` (read by `scripts/run_ripple_m1.py` through `EventStore.to_frame`) | 4 | Authoritative for events and `effective_session` | Yes | Shock source events inside the session window |
| Abnormal returns computed in the run from the price files (`abnormal_return_frame`, derived statistics; `<data_dir>/ripple/abnormal_returns.parquet` is only their published copy) | 4 | Authoritative for realised shock sizes | Yes | Per (session, instrument) |
| `references/ripple_params.json` | 3 | Authoritative for alpha, hops, half-life and age cap | Yes | `ripple` section |

## Process
1. Deterministic: build each session's shock vector from the events and the abnormal returns, using only what was known by that session's cutoff.
2. Deterministic: read the graph with `graph_asof` at the cutoff; build the adjacency `A[receiver, source]` and normalise each receiver's incoming weights to sum to 1.
3. Deterministic: compute `ripple = sum over k = 1..hops of alpha^k A^k shock`, per channel: `customer_of` feeds `ripple_customer`, `supplier_of` feeds `ripple_supplier`, `peer_of` feeds `ripple_peer`, `input_commodity` and `output_commodity` feed `ripple_commodity`. `country_exposure` has no factor column.
4. Deterministic: decay shocks by age in sessions (half-life, age cap) and return the panel (it is held in memory for the ladder, not written).

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| The `ripple_` factor panel (in memory: `build_ripple_factor_panel` returns a frame indexed `(datetime, instrument)` with `ripple_customer`, `ripple_supplier`, `ripple_peer`, `ripple_commodity`, `ripple_hop2`, `ripple_unabsorbed`) | Not persisted: the ladder builds it per graph and passes it to the Qlib handler | `32_ripple_evaluate` |
| `<data_dir>/ripple/shocks.parquet` | Index `(datetime, instrument)`, the *source* shocks (abnormal returns of the shocked instruments, not propagated sizes); written by `32_ripple_evaluate --publish` | `33_ripple_report` |
| `<data_dir>/ripple/MANIFEST.json` | JSON: `evidence_label` (`SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL` or `NO LIVE RUN`), `published_graph`, `start`, `end` and file hashes; written by `32_ripple_evaluate --publish` | `33_ripple_report` |

## Verify
- Every factor builder ships a truncation-invariance test (design 4.5). The Ripple tests are `test_ripple_panel_passes_truncation_invariance` and its negative controls (`test_truncation_test_has_teeth_against_a_later_graph_lookup` and `..._a_lookup_one_to_three_sessions_ahead`) in `tests/ripple/test_ripple_factors.py`, the G2 snapshot tests in `tests/ripple/test_graph_builders.py` and `tests/ripple/test_control_graphs.py`, and the point-in-time G3 test in `tests/ripple/test_peer_links.py`; none uses `assert_truncation_invariant` from `tests/fakes.py`.
- The commodity channel cannot fire in this milestone: shocks come only from company 8-Ks, and commodity nodes have none, so a zero `ripple_commodity` column is not a finding. `text_shocks` exists but nothing consumes it yet.
- Hand-computed propagation values on a small fixture graph.
- `uv run pytest -q` exits 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. Review factor definitions before they enter a model.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. Outputs are written atomically.

Loop: single pass. Deterministic code decided by tests (design 10.2).
