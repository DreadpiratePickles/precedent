# Stage: 33_ripple_report

## Objective
Render the daily Ripple map: which names the dated graph says a shock reaches, shown as history with its evidence, not as a forecast.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `<data_dir>/ripple/links/` | 4 | Authoritative for links, read through `graph_asof` | Yes | Links visible at the map date |
| `<data_dir>/ripple/shocks.parquet` | 4 | Authoritative for propagated shock sizes | Yes | Map date |
| `<data_dir>/ripple/abnormal_returns.parquet` | 4 | Authoritative for realised abnormal returns (derived statistics) | Yes | Map date |
| `<data_dir>/ripple/MANIFEST.json` | 4 | Authoritative for what the data is, when and from where it was built | Yes | Evidence label |

## Process
1. Deterministic: read the manifest and choose the evidence banner from its `evidence_label` (`SYNTHETIC —`, `LIVE —` or `NO LIVE RUN —`).
2. Deterministic: read the graph with `graph_asof` at the map date's cutoff, and the shocks and abnormal returns for that date.
3. Deterministic: render the Markdown map, including `country_exposure` links, which have no factor column.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| A Markdown map whose first line is the evidence banner | Markdown | The owner |

## Verify
- The first line of the map is an evidence banner.
- The same inputs render byte-identical output.
- No vendor price appears in the map; only derived statistics.
- `uv run pytest -q` exits 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. The owner reads the map; nothing is published or sent.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. The map is written atomically.

Loop: single pass. The report is a single deterministic render.
