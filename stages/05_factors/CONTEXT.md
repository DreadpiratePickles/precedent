# Stage: 05_factors

## Objective
Compute news, Precedent and Ripple factors per (session, instrument) from aligned events, using code only and never reading anything effective after the session.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/04_align/output/aligned_events.parquet` | 4 | Authoritative for events and `effective_session` | Yes | All records |
| `references/` factor parameters (half-life, window) | 3 | Authoritative for factor definitions | Yes | Design 6.2 |
| Precedent and Ripple outputs, once built | 4 | Authoritative for their own factors | When the rung needs them | Design 7.4 rungs M2 to M4 |

## Process
1. Deterministic: for each event compute `signed` and `attention` weights.
2. Deterministic: aggregate per session and instrument (count, maximum severity, signed sum, novelty share, decayed sums); age is counted in sessions since `effective_session`.
3. Deterministic: fill news columns with 0 where there is no news; leave price-feature gaps untouched.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/05_factors/output/news_factors.parquet` | Index `(datetime, instrument)`, one column per factor | `06_dataset` |

## Verify
- Every factor builder ships a truncation-invariance test (design 4.5), using `assert_truncation_invariant` from `tests/fakes.py`.
- Hand-computed fixture values for each factor.
- `uv run pytest -q` exits 0.

## Approval
Review factor definitions before they enter a model. No external action occurs in this stage.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. Output is written atomically.

Loop: single pass. Deterministic code decided by tests (design 10.2).
