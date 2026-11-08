# Stage: 06_dataset

## Objective
Join price features to the news factors in a Qlib dataset handler and attach a label that starts strictly after the decision cutoff.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/05_factors/output/news_factors.parquet` | 4 | Authoritative for news columns | Yes | All columns |
| Price provider data (local, never committed) | 4 | Authoritative for prices | Yes | Price features and label |
| `references/` asset panel and convention C1 settings | 3 | Authoritative for label and deal price | Yes | Design 6.1, 6.3 |

## Process
1. Deterministic: initialise Qlib with `region="us"` and an SQLite experiment manager.
2. Deterministic: build a `NestedDataLoader` with the price loader first and `join="left"`.
3. Deterministic: label under convention C1, `Ref($open, -2)/Ref($open, -1) - 1`; `deal_price` is `open`; label horizon is 2 sessions.
4. Deterministic: fill news columns with 0; leave price gaps as NaN; allow only `DropnaLabel` among the learn processors.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/06_dataset/output/dataset_manifest.json` | JSON: loader order, label expression, row counts, input hashes | `07_train`, human review |
| Qlib dataset handler (code, built on demand) | `(datetime, instrument)` index | `07_train`, `08_evaluate` |

## Verify
- The row count with `join="left"` equals the price row count (the news loader must not drop rows).
- The label starts strictly after the cutoff; label expression and `deal_price` agree with the alignment convention.
- Tests of this stage are research-module tests and start with `pytest.importorskip("qlib")`.
- `uv run pytest -q` exits 0.

## Approval
Downloading a price dataset is an owner gate (name the source, size and cost first). No vendor prices are displayed or redistributed.

## Failure Behavior
If Qlib behaves differently from the design, stop and report; do not invent a substitute. A missing provider path is a typed error, never an empty dataset. No retries.

Loop: single pass. Deterministic assembly decided by tests (design 10.2).
