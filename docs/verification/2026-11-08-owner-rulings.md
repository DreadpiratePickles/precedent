# Owner rulings, 2026-11-08

On 2026-11-08 the owner delegated these decisions to the controller session, in chat: "i approve these you can go ahead and do these". The decisions cover amendment section 13 (B1 to B5) and the open CAMEO subcode 172 question. Each ruling records its cost if wrong.

| # | Ruling | Already true on `spine` | Cost if wrong |
|---|---|---|---|
| B1 | PyPI distribution name `precedent-engine`; the import name stays `precedent`. | yes: `pyproject.toml` `name = "precedent-engine"` | a later rename on PyPI; the import name does not change |
| B2 | The GitHub repository is named `precedent`. | yes: `DreadpiratePickles/precedent` (public since 2026-11-07) | none |
| B3 | Ingest these Polymarket and Kalshi categories first: politics, geopolitics, central banks, macro. No sports. | default in the plans | a thinner market library for other event types until categories are added |
| B4 | GDELT `NumSources >= 3` and the severity bands stay as proposed. They are re-measured in the first live run before any result is trusted. | default in the plans | too many or too few GDELT events; visible in the first live run's counts |
| B5 | The committed sample is real GDELT data: two months, under 2 MB, cited per the GDELT terms. Building it is a live download, so it waits for a live-run go-ahead. | not yet built | none until built |
| CAMEO 172 | `172` "Impose administrative sanctions" (1721 to 1724) is not a sanction subcode. It describes domestic political restriction (curfews, party bans, states of emergency), not trade or financial sanctions. It keeps root 17's `untyped`. Only `163` with a government Actor1 maps to `trade_or_sanctions`, as `stages/02_triage/CONTEXT.md` reads it. | yes: the triage contract and `references/cameo_to_event_type.json` | states of emergency are dropped rather than typed. If the product later wants an "emergency rule" event type, 1724 is the code to map |

Still open, and not decidable by the controller:
- the securities-lawyer hour (thesis T5);
- every live run and model call (separate go-ahead).

The facts the owner approved for checking (the MSCI developed-market list, the NYSE 2010-2014 holidays and the special closures, and the EDGAR `acceptanceDateTime` time zone) are recorded in the project once checked: `references/country_groups.json`, `references/nyse_holidays.json` and `stages/01_ingest/CONTEXT.md`.

## Owner rulings, 2026-11-08 night (Dream)

On 2026-11-08 the owner answered the supervisor's open-decision list in chat: "yes to everything". The supervising Claude Code session records each decision below exactly as it was put to the owner, with its cost if wrong.

| # | Ruling | Cost if wrong |
|---|---|---|
| D69 | Knowledge time is the spine's `available_utc`: a document is visible at session `t` when `available_utc <= cutoff(t)`. `ingested_utc` gates nothing. | A live run that needs exact reproduction of what the machine had on disk would need an extra ingest-time gate. |
| D72 | `event_direction` takes no part in `value_conflict`; `text_vs_price` and `direction_drift` own direction. | Direction disagreements between documents are not shown as value conflicts. |
| D1 | The closed claim vocabulary stands as listed in Dream Task 1 (12 attributes, 4 periods, 9 values). | A vocabulary change later bumps the parser version and needs a fresh `--out`. |
| D2 | The store is JSON Lines plus pandas, with DuckDB read-only when sizes demand it, and no graph server. | A migration if sizes outgrow it. |
| D3 | Hand-check budget: 25 tensions per kind at D-M1, then 100 passages and 60 tensions at Task 8. | Detector precision is measured on a small sample. |
| D4 | A detector is kept at precision of at least 0.7 on the hand-check. | Too many or too few detectors survive. |
| D5 | `asymmetric_link` is kept, scored separately, and dropped if it scores below the threshold. | One detector of little value is built and scored. |
| D6 | The nightly runs at 03:30 local on the Mac that holds `data/qlib_us/`, installed by hand after the spine ingest runs live. Installing the schedule is still a separate live step. | Another time or machine later. |
| T7 tie-break | When two records of one claim id have the same `recorded_utc`, store line order decides which record owns a visible span. The case is unreachable through the stores' append paths (identical appends are skipped, and D70 makes `recorded_utc` non-decreasing). | None while the case stays unreachable. |
| D24 | The owner approved reading SEC Form 8-K. See the result below. | A mapping change at the first live EDGAR run. |

Result of D24, read on 2026-11-08 from the SEC's Exchange Act Form 8-K Compliance and Disclosure Interpretations page (sec.gov, "Last Update: October 21, 2026"):
- All eight item numbers in `references/item_rules_8k.json` exist: 1.01, 1.03, 1.05, 2.02, 2.05, 2.06, 4.02 and 5.02.
- Three titles were read verbatim and match the reference file: 1.01 "Entry into a Material Definitive Agreement", 1.03 "Bankruptcy or Receivership" and 1.05 "Material Cybersecurity Incidents".
- The other five titles match the reference file only through a machine summary of the page, not verbatim. The form PDF could not be converted to text on this machine, and investor.gov refused the fetch. These five stay partly verified until the form text is read.

Still open, and not decidable by "yes": Mechanism M6, which asks what the placebo rung M-P should be. The supervisor's recommendation goes to the owner separately.
