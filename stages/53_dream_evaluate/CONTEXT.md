# Stage: 53_dream_evaluate

## Objective
Sample tensions for the owner to check by hand and score each detector's precision from those checks.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `<out>/tensions/tensions.jsonl`: `<out>` is the `dream` command's `--out` directory (default `stages/52_dream_report/output`) | 4 | Authoritative for tensions to sample; read-only | Yes | Latest record per id |
| `<out>/shifts/shifts.jsonl` (same `<out>`) | 4 | Authoritative for shifts to sample; read-only | Yes | Latest record per id |
| `<out>/dreams/dream_<session>.jsonl` | 4 | Authoritative for what was reported; read-only | Yes | Sessions in the evaluation window |
| `references/dream/detector_params.json` | 3 | Authoritative for `review_sample_per_kind` and `detector_keep_precision` | Yes | Evaluation |
| `stages/53_dream_evaluate/output/` (owner's hand-check verdicts) | 4 | Authoritative for whether a record was correct; written by the owner | For scoring | Verdict file |

## Process
1. Deterministic: draw up to `review_sample_per_kind` records per detector kind with a fixed seed, so the same inputs give the same sample.
2. Deterministic: write the sample for the owner with the evidence span and source document for each record.
3. Deterministic: once the owner's verdicts exist, compute each detector's precision (confirmed over checked) and compare it with `detector_keep_precision`.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/53_dream_evaluate/output/review_sample.jsonl` | One sampled record id per line with its kind and evidence | The owner |
| `stages/53_dream_evaluate/output/detector_precision.json` | Kind, checked, confirmed, precision, keep or drop flag | `docs/STATUS.md` |

Tensions and shifts are observations about documents and prices, not forecasts. Precision here measures whether a record is a true observation of the documents, not whether it predicted a price.

## Verify
- The same inputs and seed give the same sample.
- Precision is computed by code from the verdict file; none is estimated or invented.
- A detector with no checked records reports "unknown", not zero or one.
- `uv run pytest -q` exits 0.

## Approval
The owner checks the sample and decides which detectors to keep. A detector below `detector_keep_precision` is flagged, not removed automatically. Inputs are read-only.

## Failure Behavior
A missing verdict file leaves precision unknown and stops scoring with that message. No retries. Output is written atomically.

Loop: single pass. The owner's hand-check is the judgment and there is no automatic oracle to iterate against.
