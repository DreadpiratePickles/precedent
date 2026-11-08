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
1. Deterministic: parse `--date` (YYYY-MM-DD), `--top` (>= 1, default 10) and `--out`; the date must be an NYSE session in the committed calendar, and its cutoff is the spine `calendar_cutoffs` value (16:00 New York, early close honoured). A usage error or a non-session date exits 2.
2. Deterministic: with no `MANIFEST.json`, render a map whose line 1 is `NO LIVE RUN — no Ripple data has been built under the configured data directory.` (no absolute path on line 1; the body names the path; exit 0). A manifest without `links/links.jsonl` stops with exit 1; an absent link store is never read as no links. Otherwise map `evidence_label` to the banner: `SYNTHETIC` to `SYNTHETIC —`, `LIVE-DATA` and `LIVE-MODEL` to `LIVE —`, `NO LIVE RUN` to `NO LIVE RUN —`; any other label stops with exit 1.
3. Deterministic: check each parquet file against the manifest (listed, present, SHA-256 and row count match); a mismatch stops with exit 1 naming the file.
4. Deterministic: read the graph with `graph_asof` at the cutoff, the session's rows of `shocks.parquet` and `abnormal_returns.parquet`; take the `--top` largest shocks by absolute size (ties by instrument name). No shock gives the line `No shocks on this session.`.
5. Deterministic: per shock, list the links whose source is the shocked instrument, by channel (customer, supplier, peer, commodity, then `country_exposure`). The propagated size is the hop-1 term `alpha * A_c[receiver, source] * shock`, with `A_c` the row-normalised adjacency of that channel (`alpha` from `references/ripple_params.json`); `country_exposure` shows `n/a (no factor channel)`. The move already made is the receiver's abnormal return on the session, `n/a` when missing, never zero.
6. Deterministic: write one table row per link, `| -> receiver | relation | size | move | doc: <source_doc_id> | evidence |`, where evidence is the quoted span or `rule: <extractor_version>`; write the map atomically to `--out` or to stdout.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| A Markdown map whose first line is the evidence banner | Markdown | The owner |

## Verify
- The first line of the map is an evidence banner, chosen by the manifest label.
- Every link row shows `doc:` and its evidence; a link not yet known at the cutoff is absent.
- The same inputs render byte-identical output.
- No vendor price appears in the map; only derived statistics.
- Exit codes: 0 rendered, 1 data error (unknown label, hash mismatch, missing file), 2 usage error or non-session date.
- `uv run pytest -q tests/ripple/test_report.py` and `uv run pytest -q` exit 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. The owner reads the map; nothing is published or sent.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. The map is written atomically.

Loop: single pass. The report is a single deterministic render.
