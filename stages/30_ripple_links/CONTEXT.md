# Stage: 30_ripple_links

## Objective
Turn dated 10-K and 8-K documents into dated company links in the append-only link store, each valid only from the day its document was accepted.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| 10-K and 8-K documents (stage 01 output; not yet fetched) | 4 | Authoritative for what a filing states and when it was accepted | Yes | Customer, supplier and competitor passages |
| 10-K Item 1 (Business) text per company, with each filing's acceptance time (stage 01 output; not yet fetched) | 4 | Authoritative for what a company says it does and when the filing was accepted | Only for the G3 text-similarity path | Item 1 up to Item 1A, Item 1B or Item 2 |
| `references/link_prompt_v1.txt` | 3 | Authoritative for what the model is asked and the closed relation and kind lists (owner approves it before any run) | Only for the G4 model-extracted path | Whole file |
| `references/ripple_params.json` | 3 | Authoritative for link expiry (`annual_link_expiry_months`) | Yes | `ripple` section |

## Process
1. Deterministic: select the passages of each document that name customers, suppliers, competitors, input or output commodities, and country exposure.
2. Judgment, bounded: `parse_links` (`src/precedent/ripple/link_extraction.py`) sends one passage to an injected `EventExtractor` with `references/link_prompt_v1.txt` (or the `prompt_file` the caller names) and receives structure only: relation, counterparty name and kind, and a quoted `evidence_span`, at most 20 links. The model never forecasts and never supplies a weight.
3. Deterministic: validate every row as a `Link` (tz-aware UTC, `weight` in [0, 1], `valid_to_utc` after `valid_from_utc`); a link not made by a `rules-` extractor must quote a non-empty `evidence_span`.
4. Deterministic: set `valid_from_utc` to the document's acceptance time and `valid_to_utc` from the expiry parameter; append through `LinkStore.append`, which skips byte-identical duplicates.
5. Deterministic, G3 text-similarity path: `extract_business_section` (`src/precedent/ripple/filing_text.py`) takes Item 1 of each 10-K, or None when not found (counted, never an empty string); `embed_business_sections` embeds the texts in one batch through an injected `TextEmbedder` (the embedding model download is an owner gate; nothing has run); `peer_links_from_embeddings` (`src/precedent/ripple/peer_links.py`) emits each company's top-k most similar others as `peer_of` links (`rules-10k-similarity-v1`, no model-extracted evidence needed) starting at the later of the two filings' acceptance times and expiring after `annual_link_expiry_months`.
6. Deterministic, G4 model-extracted path: `parse_links` validates the output strictly (bad JSON or a schema violation gives no links and `invalid_output` in the report, never an exception), then keeps a link only if its whitespace-normalised quote occurs in the passage the model saw, on word edges (a quote cut mid-word fails) and `resolve(counterparty_name)` returns a node other than the filer, and the counterparty name itself occurs inside the quoted span (whitespace-normalised, case-insensitive, on word edges: a true sentence that does not name the counterparty is no evidence). Direction: the counterparty is `source_node` and the filer `receiver_node` (channels are named for where the shock comes from, so `customer_of` means the counterparty is a customer of the filer). Weight 1.0, `valid_from_utc` the filing's acceptance time, `valid_to_utc` from `annual_link_expiry_months`, `extractor_version` not starting with `rules-`. `LinkParseReport` counts `proposed`, `evidence_missing`, `name_not_in_evidence`, `unresolved`, `duplicates` and `kept` (they add up, and the code asserts it); a blank resolved node counts as unresolved; a name the passage does not give ("one customer accounted for 12%") yields no link. The extractor runtime, the gold sample and the universe run are owner gates (NO LIVE RUN).

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<data_dir>/ripple/links/links.jsonl` (the `LinkStore` root is `<data_dir>/ripple/links`) | JSON Lines of `Link` (`precedent.ripple.link_store`), append-only; a persistent store, not a per-run file | `31_ripple_factors` and `33_ripple_report` through `graph_asof` |
| `stages/30_ripple_links/output/run_manifest.json` | JSON: document ids read, links appended and skipped, prompt version | The owner |

## Verify
- `uv run pytest -q tests/ripple/test_link_store.py`: visibility is half-open at both ends, a restated link extends validity without duplicating, a model-extracted link needs evidence, naive timestamps are rejected.
- `uv run pytest -q tests/ripple/test_filing_text.py tests/ripple/test_peer_links.py`: Item 1 extraction (table of contents skipped, `Item 1A` and `Item 10` not taken as Item 1, None when absent), and G3 links start at the later filing, never self-link, respect top-k, round-trip through `LinkStore`, and reject bad input.
- `uv run pytest -q tests/ripple/test_link_extraction.py`: a link with a quote not in the passage is dropped and counted, an unnamed customer yields no link, an unresolved or self counterparty is counted and dropped, bad JSON or a schema violation (unknown relation, 21 links, kind mismatch, extra field) gives `invalid_output` and no links without raising, a rule-prefixed `extractor_version` or a naive time raises ValueError, the extractor is called once with the whole passage and the schema, and kept links round-trip through `LinkStore` and `graph_asof` (not visible one minute before acceptance). A fake extractor is used; no model is called.
- A link is never visible before the document that states it was accepted (point-in-time test).
- `uv run pytest -q` exits 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. The owner approves the link-extraction prompt before any run on real filings, and the first document fetch is an owner gate.

## Failure Behavior
An invalid row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. The link store is written atomically.

## Loop
Status: contract only, written in Task 8; nothing has run (NO LIVE RUN). The gold samples, the scorer and the runner script are not built: labelling the samples with the owner is an owner gate, and no model has been called. Every number below is `proposed`.

Kind: ratchet (one prompt, one metric; a model-extracted link is only as good as the prompt that produced it).
Oracle: `scripts/score_link_extraction.py` (not yet written) runs `parse_links` over the 100-passage development gold sample and reports precision and recall of the kept links against the gold links. A revision is kept only when dev precision improves and dev recall does not fall below the baseline prompt's recall (ratchet rule, proposed). Green means dev precision >= 0.8 (proposed); green is not acceptance.
Oracle protection: the gold files and the scorer live outside the mutable allowlist, in a read-only path (or a separate trusted checkout) that the worker cannot write; any change to them, to the tests or to `parse_links` stops the loop and goes to the owner.
Held-out check: a second 100-passage gold sample, labelled with the first and never shown to the worker, run once at acceptance. Success criterion R1: precision >= 0.8 on it.
Unit per iteration: one prompt revision, saved as `references/link_prompt_v<N>.txt`; one hypothesis about why proposals fail. The runner runs a revision by passing `prompt_file="link_prompt_v<N>.txt"` to `parse_links` (a bare name matching `link_prompt_v<digits>.txt`, so `parse_links` itself never changes).
Revises against: the per-passage false positives and misses from the last run (the scorer's results table), read first each iteration.
Mutable / protected: `references/link_prompt_v*.txt` (new versions only; the placeholders `{filer}` and `{passage}` and the closed lists stay) / gold samples, scorer, `src/precedent/ripple/link_extraction.py`, `tests/`, `references/ripple_params.json`.
Isolation: a branch (or worktree) cut from a recorded baseline commit; nothing inside the loop publishes, pushes, spends or fetches.
Worker / judge: standard-tier model worker proposing prompt revisions; the deterministic scorer judges; no model judge.
Runner: a script (not yet written) that counts iterations, enforces the budget, runs the scorer, keeps the best-so-far prompt, reverts only the declared mutable file, and stops on no-progress or repeated failure. The model does not decide when to stop.
State: results table and progress log under `stages/30_ripple_links/output/` (per-run artifact, not committed); the prompt versions and one commit per kept iteration.
Context: fresh per iteration, with the results table and the last run's per-passage errors as the handoff.
Stop: oracle green on dev; budget 10 iterations, 4 hours wall time, zero spend (local runtime); no kept revision (dev precision up with dev recall not below the baseline's) after 3 iterations; the same failing passage with the same first error 3 times in a row; blocked (needs a human).
On stop: the best-so-far prompt version is kept, the run is logged in `docs/STATUS.md`, and the owner is told; the held-out check then runs once, and R1 is judged on it only.
Single-pass baseline: not yet run; compare once the first prompt (`link_prompt_v1.txt`) is scored against the loop's best.
