# Precedent product amendment to the predate design

**Written:** 2026-11-07. **Amends:** `2026-11-06-predate-design.md` (the design) and the five plans under `docs/superpowers/plans/`. **Basis:** `docs/2026-11-07-product-thesis.md` (the thesis) and the owner's decisions of 2026-11-07, listed in section 1.
**Status:** draft for the owner's review. Everything in the design that this file does not amend stands. Facts marked *(checked 2026-11-07)* were read from the source named in section 12; everything else is design.

## 1. Decisions taken on 2026-11-07

| # | Decision | Owner's choice |
|---|---|---|
| A1 | What ships | All four tracks are built: Precedent (the product), Dream (the brief's content engine), Ripple (compute-only until a price licence exists), Mechanism (research and content). |
| A2 | Where | A new, self-contained git repository at `predate/predate-git/` (owner's name for the folder; decided after an earlier "inside predate" answer). The outer `predate` folder stays the research archive and ignores `predate-git/`. The Python package and console script are named `precedent`, because the product is the face of the repo; the folder and GitHub names are the owner's to set at publish time. Research tracks are subpackages. The design docs, thesis and plans are copied into `predate-git/docs/` so the new repo stands alone. |
| A3 | Precedent's first release | All three outcome libraries: asset response, market resolution, event sequence. |
| A4 | Typing free text | Deterministic rules first; two optional model seams (Anthropic API, local Ollama) behind one protocol. No model call in tests or CI. |
| A5 | Getting data | A prebuilt, versioned library snapshot on GitHub Releases plus a raw-ingest rebuild path. (Default taken by the agent; the owner did not answer this question.) |
| A6 | Order of work | Spine first, with its independent tasks in parallel; then the four tracks in parallel, each in its own worktree and session. |
| A7 | Human gates | Unchanged from the rulebook: no push to `main`, merge, publish, model or dataset download, spend or visibility change without the owner's yes. |

## 2. Naming and layout

```text
src/precedent/
  cli.py                 console script `precedent`; subcommands below
  config.py              the only module that names model ids, URLs and data-dir defaults
  spine/                 shared: schema, record, store, alignment, sources, prices, abnormal returns, typers
  engine/                the analog engine: keys, story clusters, retrieval, aggregate, card
  libraries/             the three outcome libraries (section 4)
  mcp_server.py          console script `precedent-mcp` (section 6)
  dream/  ripple/  mechanism/   research tracks, as their plans describe, under the new package name
references/              Layer 3: schema JSON, typing rules, CAMEO map, country groups, params
stages/                  ICM stage contracts (spine 01-09, Precedent 20-24, Dream 30s, Ripple 10s, Mechanism 40s as the plans number them)
data/sample/             a committed, small, labelled library slice for the offline demo (section 5)
docs/examples/           run-produced evidence files, banner on line 1 (section 9)
```

- Console scripts: `precedent` and `precedent-mcp`. The research tracks are subcommands (`precedent dream …`, `precedent ripple …`, `precedent mechanism …`); they never edit `cli.py` and are imported lazily, as the spine plan already says for `whatif`.
- `precedent ask` is the product command. A bare `precedent "<text>"` whose first argument is not a subcommand is sugar for `precedent ask "<text>"`.
- Every `predate.` import path in the plans becomes `precedent.`; `predate.precedent.*` becomes `precedent.engine.*`; environment variables `PREDATE_*` become `PRECEDENT_*`. This is a mechanical rename applied when the plans are updated.
- The repository has no GitHub remote yet. Creating it, naming it and pushing are the owner's steps (section 11); the agent never pushes.

## 3. The card

The card is the one output every surface renders: CLI text, JSON, PNG, MCP. It is produced by `engine/card.py` from a `PrecedentSummary` and never by a model.

```text
PRECEDENT  ·  <event in plain words, as typed>
typed as       <event_type> / <actor_kind> / <country_group> / severity <band> / <scheduled|unscheduled>
precedents     n=<n>  independent=<n_eff>  earliest <date>  latest <date>

<outcome type>                      horizon   rate|median   80% band        n   vs ordinary
<outcome key, e.g. trade_or_sanctions by G7>   20        14/23 (61%)   [43%, 77%]     23  outside
<outcome key, e.g. SPY abnormal return>        5         -0.8%         [-3.1%, +1.4%] 19  inside
...

nearest precedents
<date>  <one line>  similarity <s>  <source id>
...
library covers  <source>: <first date> to <last date>, <N> events; <source>: …
These are past outcomes of similar events, not a forecast.
```

The refusal replaces the outcome table and is the README's first screenshot:

```text
REFUSED        <n> usable precedents, need <min_neff>. No band is drawn below that.
nearest precedents
<date>  <one line>  …
library covers  …
These are past outcomes of similar events, not a forecast.
```

Rules: no number without its `n`; a binary outcome shows a rate with a Wilson 80% interval, a continuous outcome shows the weighted median with the weighted 10th to 90th percentile band; `vs ordinary` compares with the placebo distribution (design 8.3); the `library covers` line is mandatory because selection bias is real (thesis section 7); the disclaimer line is mandatory and tested. `--json` emits the same fields as one object with a `schema_version`. `--card <path>.png` renders the text card with Pillow in a monospace font bundled under `references/fonts/` (licence file beside it); no network, no model.

## 4. Three outcome libraries, one engine

The design's response library (P3) becomes one of three. All three produce rows of one shape, so `find_precedents`, `summarize_responses`, the time guard and the refusal are shared and tested once.

| Column | Meaning |
|---|---|
| `event_id` | the precedent event (first report of its story) |
| `outcome_type` | `asset`, `market_resolution`, `event_sequence` |
| `outcome_key` | asset ticker; or the market's typed `EventKey` string; or the follow-on `event_type` plus link kind |
| `horizon` | sessions, from the event's effective session; for market resolution, the sessions until resolution |
| `value` | float: CAR for assets; 1.0/0.0 for resolution and sequence |
| `is_binary` | selects rate-and-interval summary instead of quantiles |
| `effective_session`, `response_complete_session` | as the design; the guard reads the second |
| `source_ref` | GDELT `GlobalEventID`, market id, or filing accession; shown on the card |

**Asset response** (design P3, unchanged math): CAR of the asset panel at 1, 5 and 20 sessions from prices the user supplies (`precedent prices import <csv>`, columns `date, symbol, open, close`) or builds with the spine's provider. Nothing in the repo displays or redistributes vendor prices; the sample ships no prices. Shown as `NOT BUILT: no prices` on the card until the user imports some.

**Market resolution** (new): the universe is resolved binary markets from Polymarket and Kalshi. Each market's question is typed to an `EventKey` by the typer (section 7), with `source_ref` and the typer version stored. For a precedent event at `available_utc = t`, a market qualifies when it was open at `t` (created before, resolved after) and its typed key is not `UNTYPED`. `value` is 1.0 for a YES resolution; `response_complete_session` is the session on which the resolution time falls. The card groups by market key: "after the last 9 unscheduled central-bank holds in the US, `central_bank / cut / US` markets open at the time resolved YES 2 of 9". Untyped markets are counted and reported, never silently dropped.

**Event sequence** (new): for a precedent event with actor country `c` and key `K`, and each follow-on `event_type` `T` in the schema, `value` is 1.0 if a first-report event of type `T` linked to `c` (same actor country, or `c` as Actor2's country) has `effective_session` within `h` sessions after the precedent's, else 0.0; one row per `(T, h)` for `h` in {5, 20, 60}. The placebo for a binary outcome is the unconditional rate of `T` for country group `g` over random windows of length `h`. The source is GDELT 2.0 only (section 8).

Effective sample size, shrinkage, story clustering, the refusal threshold and the baseline ladder apply to all three; parameters stay in `references/precedent_params.json`.

## 5. Data: snapshot plus rebuild

- `precedent fetch [--version <tag>]` downloads `precedent-library-<tag>.tar.zst` and `MANIFEST.json` from this repository's GitHub Releases into `$PRECEDENT_DATA_DIR` (default `~/.cache/precedent/`), verifies the manifest's SHA-256 for every file before anything is read, and refuses on mismatch. The manifest records sources, date ranges, row counts, the builder's git commit and `precedent_version`.
- `precedent ingest gdelt --since <date> --until <date>`, `precedent ingest polymarket`, `precedent ingest kalshi`, `precedent ingest edgar --ciks <file>` rebuild the typed-event and market stores from raw sources; `precedent build` rebuilds the three libraries from the stores. Every ingest is resumable (one state file per source), bounded (`--max-files`, `--max-requests`), and writes counts of kept, dropped and untyped rows.
- `data/sample/` holds a real slice of GDELT 2.0 events (two calendar months, under 2 MB, cited as GDELT requires) and the resolved markets from the same window, with a manifest and the line-1 banner `LIVE —`. Tests never read it; tests use synthetic fixtures. The README demo runs offline against it and will mostly refuse, which is the point of the first screenshot.
- Publishing a snapshot is the owner's step: the agent builds it and writes the manifest; the owner uploads the release asset.
- Nothing under `data/` except `data/sample/` is committed. `stages/*/output/`, `vendor/` and `$PRECEDENT_DATA_DIR` are ignored.

## 6. MCP server

`precedent-mcp` runs over stdio with the official Python MCP SDK (section 12 for the package and version). Two tools, both read-only against the local library:

- `precedents(text: str | None, key: dict | None, horizon: int = 5, outcome_type: str = "all") -> dict`: the card's JSON.
- `card(text: str | None, key: dict | None, horizon: int = 5) -> str`: the text card.

The server imports `engine` only, never a provider; with `PRECEDENT_TYPER=rules` (the default) a text query that the rules cannot type returns the refusal with `typed_as: UNTYPED` and the key flags that would work. Tests call the tool functions directly and one test starts the server as a subprocess and lists tools over stdio.

## 7. Typers

`spine/typers/` is the only package allowed to import a model provider (the spine's global constraint, renamed).

```python
class Typer(Protocol):
    version: str
    def type_text(self, text: str) -> TypedText: ...   # TypedText: key: EventKey | None, status: Literal["typed","untyped","refused_invalid"], reason: str, evidence_span: str | None
```

- `RulesTyper`: keyword and pattern table in `references/typing_rules.json` (`event_type`, `actor_kind`, direction and scheduled cues; country names and ISO3 from `references/country_groups.json`). Deterministic, no network, the default and the CI path. Its precision on the market-question gold set is measured before the model typers are allowed to replace it (design 8.4's rule: the model must beat the rules).
- `AnthropicTyper` and `OllamaTyper`: send the text with the JSON Schema of the typing response, parse with the pydantic model, `extra="forbid"`, reject an `evidence_span` not found in the text, and return `refused_invalid` on any failure. Both honour `--dry-run` (no socket opens, returns `untyped`), `max_calls`, a timeout, and exit non-zero before any network call when the key or base URL is missing. Model ids and base URLs live in `config.py` and environment variables `PRECEDENT_TYPER`, `ANTHROPIC_API_KEY`, `PRECEDENT_OLLAMA_BASE_URL`, `PRECEDENT_OLLAMA_MODEL`.
- A gold set of 60 hand-labelled market questions and headlines (`references/gold/typing_cases.json`) scores every typer; the score and the typer version are printed by `precedent typers score` and committed under `docs/examples/` with a banner.

## 8. GDELT ingestion, resolved

The design left the CAMEO mapping open. Resolution:

- Source: GDELT 2.0 Events export files (15-minute, from 2015-02-19), listed by the master file list; `DATEADDED` is `first_seen_utc`; `Day` is never used for alignment (design 4.1). GDELT Cloud is not used (thesis section 2).
- Typing: `references/cameo_to_event_type.json` maps CAMEO root codes 01 to 20 to `event_type` or `untyped`. Starting table (every row is a proposal to be checked against the CAMEO manual in the task's Step 0): 14 protest → `election_or_political`; 15 force posture, 18 assault, 19 fight, 20 unconventional mass violence → `armed_conflict`; 16 reduce relations and 17 coerce, when Actor1 is a government and the code is a sanction or embargo subcode, → `trade_or_sanctions`; 13 threaten → `other`; 01 to 12 → `untyped` unless a subcode table says otherwise. Actor kind from `Actor1Type1Code` (GOV, MIL → `government`; IGO → `international_org`; BUS → `company`; else `other`). Severity band from `|GoldsteinScale|` thirds and `NumSources` (proposed; recorded in `precedent_params.json`). `is_scheduled` is false for all GDELT rows.
- Filter: keep rows with `NumSources >= 3` (proposed) and a mapped type. Counts of dropped rows are reported.
- Dedup: one record per (`EventKey`, `Actor1CountryCode`, session); the earliest `DATEADDED` wins; then story clustering at 3 sessions.
- Scheduled macro events (central-bank decisions, data releases) do not come from GDELT; they come from a committed calendar `references/scheduled_events.json` the owner approves, so `is_scheduled` is honest.

## 9. Evidence, verification, CI

- Every run-produced file under `docs/examples/` begins with `SYNTHETIC —`, `LIVE —` or `NO LIVE RUN —`, and a test pins it (method AIE-003). `docs/STATUS.md` keeps the design's four labels.
- Independent verifier after each track's "finished", report kept in the repo under `docs/verification/`, findings fixed in `fix: independent-review findings` commits (AIE-006).
- CI: GitHub Actions, `astral-sh/setup-uv` pinned by SHA, `uv sync --frozen`, `uv run ruff check .`, `uv run pytest -q`, a secret scan; no network to any data source or model in CI.
- Python 3.11 as the design says (the method's 3.12 default is overridden because Qlib 0.9.7 is the constraint); `uv`, hatchling, `src/` layout, ruff line length 100.

## 10. Execution shape

1. Spine Tasks 1 to 4 in this session, serially (they define the interfaces every other task imports).
2. Spine Tasks 5 to 16 in parallel subagents, grouped by file ownership: {5, 6} EDGAR and universe; {7, 8} prices and abnormal returns; {9, 10, 11} Qlib handler, ablation, leak checks; {16} GDELT; {typers} new; {12 to 15} live runs stay owner-gated.
3. When the spine suite is green and merged: four worktree sessions, one per track, each running subagent-driven development from its updated plan. Precedent's plan gains the card, the three libraries, the snapshot and the MCP server; Dream's gains the brief template; Ripple's and Mechanism's change only the package rename and the licence gate.
4. Each track ends with the independent verifier; the owner merges.

## 11. Pre-publish checklist (owner's gate)

Secret scan recorded; the copied research docs under `docs/` reviewed for what stays public; GitHub repository created and named by the owner (`precedent` suggested); PyPI name chosen (section 12); `LICENSE` Apache-2.0 with the owner's name; the securities-lawyer hour booked (thesis T5); the README's Status table lists every live path not exercised.

## 12. Facts checked on 2026-11-07

Filled from the fact-check report when it returns; each line names the URL read. Until then every API detail above is a proposal.

## 13. Open decisions for the owner

| # | Decision | Default until decided |
|---|---|---|
| B1 | PyPI distribution name | `precedent` if free, else `precedent-engine` |
| B2 | Rename the GitHub repository to `precedent` | at publish time |
| B3 | Which Polymarket and Kalshi categories to ingest first | politics, geopolitics, central banks, macro; no sports |
| B4 | The `NumSources >= 3` and severity-band thresholds for GDELT | as proposed, measured in the first live run |
| B5 | Whether the committed sample is real GDELT data or synthetic | real, two months, under 2 MB, cited |
