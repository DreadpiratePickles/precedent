# Stage: 30_ripple_links

## Objective
Turn dated 10-K and 8-K documents into dated company links in the append-only link store, each valid only from the day its document was accepted.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| 10-K and 8-K documents (stage 01 output; not yet fetched) | 4 | Authoritative for what a filing states and when it was accepted | Yes | Customer, supplier and competitor passages |
| 10-K Item 1 (Business) text per company, with each filing's acceptance time (stage 01 output; not yet fetched) | 4 | Authoritative for what a company says it does and when the filing was accepted | Only for the G3 text-similarity path | Item 1 up to Item 1A, Item 1B or Item 2 |
| `references/ripple_params.json` | 3 | Authoritative for link expiry (`annual_link_expiry_months`) | Yes | `ripple` section |

## Process
1. Deterministic: select the passages of each document that name customers, suppliers, competitors, input or output commodities, and country exposure.
2. Judgment, bounded: `parse_links` reads a passage and emits structure only (who, which relation, the quoted evidence). The model never forecasts and never supplies a weight.
3. Deterministic: validate every row as a `Link` (tz-aware UTC, `weight` in [0, 1], `valid_to_utc` after `valid_from_utc`); a link not made by a `rules-` extractor must quote a non-empty `evidence_span`.
4. Deterministic: set `valid_from_utc` to the document's acceptance time and `valid_to_utc` from the expiry parameter; append through `LinkStore.append`, which skips byte-identical duplicates.
5. Deterministic, G3 text-similarity path: `extract_business_section` (`src/precedent/ripple/filing_text.py`) takes Item 1 of each 10-K, or None when not found (counted, never an empty string); `embed_business_sections` embeds the texts in one batch through an injected `TextEmbedder` (the embedding model download is an owner gate; nothing has run); `peer_links_from_embeddings` (`src/precedent/ripple/peer_links.py`) emits each company's top-k most similar others as `peer_of` links (`rules-10k-similarity-v1`, no model-extracted evidence needed) starting at the later of the two filings' acceptance times and expiring after `annual_link_expiry_months`.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<data_dir>/ripple/links/links.jsonl` (the `LinkStore` root is `<data_dir>/ripple/links`) | JSON Lines of `Link` (`precedent.ripple.link_store`), append-only; a persistent store, not a per-run file | `31_ripple_factors` and `33_ripple_report` through `graph_asof` |
| `stages/30_ripple_links/output/run_manifest.json` | JSON: document ids read, links appended and skipped, prompt version | The owner |

## Verify
- `uv run pytest -q tests/ripple/test_link_store.py`: visibility is half-open at both ends, a restated link extends validity without duplicating, a model-extracted link needs evidence, naive timestamps are rejected.
- `uv run pytest -q tests/ripple/test_filing_text.py tests/ripple/test_peer_links.py`: Item 1 extraction (table of contents skipped, `Item 1A` and `Item 10` not taken as Item 1, None when absent), and G3 links start at the later filing, never self-link, respect top-k, round-trip through `LinkStore`, and reject bad input.
- A link is never visible before the document that states it was accepted (point-in-time test).
- `uv run pytest -q` exits 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. The owner approves the link-extraction prompt before any run on real filings, and the first document fetch is an owner gate.

## Failure Behavior
An invalid row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. The link store is written atomically.

## Loop
Status: contract only, filled by Task 8; nothing has run (NO LIVE RUN).

Kind: ratchet
Oracle: precision and recall of `parse_links` on a 100-passage development gold sample (not yet labelled; owner gate).
Oracle protection: gold files read-only to the worker, scorer in a separate module.
Held-out check: a second 100-passage sample run once at acceptance (R1: precision >= 0.8).
Unit per iteration: one prompt revision.
Revises against: the per-passage false positives and misses.
Mutable / protected: `references/link_prompt_v*.txt` only / gold samples, scorer, tests.
Isolation: a branch from the baseline commit.
Worker / judge: standard-tier model worker, deterministic scorer, no model judge.
Runner: a script that counts iterations.
State: results table under `stages/30_ripple_links/output/` (per-run artifact).
Context: fresh per iteration with the results table as handoff.
Stop: oracle at R1 on dev; budget 10 iterations, 4 hours wall, zero spend (local runtime); no progress after 3; same failure 3 times in a row; blocked.
On stop: best-so-far prompt kept, run logged in `docs/STATUS.md`.
Single-pass baseline: not yet run.
