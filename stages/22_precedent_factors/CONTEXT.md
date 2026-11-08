# Stage: 22_precedent_factors

## Objective
Build the five `prec_` factors (`prec_car5_med`, `prec_car5_iqr`, `prec_neff`, `prec_hit_rate`, `prec_placebo_z`) from the libraries as Qlib features, so the precedent estimate can enter the Qlib dataset (`06_dataset`) and the spine's ablation ladder (`08_evaluate`). This stage needs the `research` extra.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `library/events.parquet`, `library/rows_*.parquet` | 4 | Authoritative for events and outcomes | Yes | Verified against `MANIFEST.json` first |
| `library/sessions.parquet` | 4 | Authoritative for the session index | Yes | `effective_session` |
| `library/placebo_asset.parquet` | 4 | Authoritative for the placebo draws | No | `outcome_key`, `horizon`, `value`, `response_complete_session` |
| `references/precedent_params.json` | 3 | Authoritative for `top_k`, `shrink_k`, horizons | Yes | `engine` section |

## Process
1. Deterministic: for each first-report event, find the asset precedents (horizon 5) whose response completed, by cutoff, at or before the event's `available_utc`. The event's instruments are the `outcome_key`s of those visible precedents, never of its own rows; an instrument with no visible precedent gets no cell.
2. Deterministic: per instrument, summarise the precedents (median, q0.75 - q0.25, effective size, hit rate). A summary refused for too small an effective size gives `prec_neff` and zeros.
3. Deterministic: `prec_placebo_z` = (median - mean placebo) / sd placebo, using only placebo rows whose `response_complete_session` cutoff is at or before the event's `available_utc`; 0.0 with no placebo, fewer than 2 values or sd 0.
4. Deterministic: a cell (session, instrument) takes the latest event within `max_age_sessions`; the join to the price loader is first with `join="left"`, so a missing value stays missing.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/22_precedent_factors/output/prec_panel.parquet` | Index (datetime, instrument), five float64 `prec_` columns | Qlib handler in `06_dataset`; the M2 rung scored by the ablation in `08_evaluate` |

## Verify
- Truncation invariance: building on data cut at a session gives the same earlier rows (design 4.5).
- A test module starts with `pytest.importorskip` for each research package it imports.
- `uv run pytest -q` exits 0.

## Approval
A human reviews the `08_evaluate` ablation result before any factor is used. No external action occurs in this stage.

## Failure Behavior
A factor that depends on a row from the future stops the stage with the factor name and session. A missing library file stops the stage with its path. No retries. Output is written atomically.

Loop: single pass. A deterministic factor build decided by the truncation test.
