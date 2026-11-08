# Stage: 22_precedent_factors

## Objective
Build `prec_` factors from the libraries as Qlib features, so the precedent estimate can enter the spine's ablation ladder. This stage needs the `research` extra.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `library/events.parquet`, `library/rows_*.parquet` | 4 | Authoritative for events and outcomes | Yes | Verified against `MANIFEST.json` first |
| `library/sessions.parquet` | 4 | Authoritative for the session index | Yes | `effective_session` |
| `references/precedent_params.json` | 3 | Authoritative for `top_k`, `shrink_k`, horizons | Yes | `engine` section |

## Process
1. Deterministic: for each instrument and session, find precedents known before that session's decision cutoff.
2. Deterministic: compute the shrunk mean and the effective match count per horizon, named `prec_<outcome>_<horizon>`.
3. Deterministic: join to the price loader first with `join="left"`; a missing value stays missing.
4. Deterministic: write the factor table.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/22_precedent_factors/output/prec_factors.parquet` | Index (session, instrument), one column per `prec_` factor | Spine `06_ablation` |

## Verify
- Truncation invariance: building on data cut at a session gives the same earlier rows (design 4.5).
- A test module starts with `pytest.importorskip` for each research package it imports.
- `uv run pytest -q` exits 0.

## Approval
A human reviews the ablation result before any factor is used. No external action occurs in this stage.

## Failure Behavior
A factor that depends on a row from the future stops the stage with the factor name and session. A missing library file stops the stage with its path. No retries. Output is written atomically.

Loop: single pass. A deterministic factor build decided by the truncation test.
