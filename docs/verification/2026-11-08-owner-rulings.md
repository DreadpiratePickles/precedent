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
