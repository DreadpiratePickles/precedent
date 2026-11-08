# Status

Every result in this repository carries one label:

- `SYNTHETIC`: produced on synthetic or fixture data only.
- `LIVE-DATA`: produced on real source data, with no language model beyond the rules typer.
- `LIVE-MODEL`: produced with a real language model on real data.
- `NO LIVE RUN`: designed and tested, but not run against a live source or model.

| Date | Result | Label | Evidence | Notes |
|---|---|---|---|---|
| 2026-11-07 | CIK-to-ticker map is current-only (free source); historical constituents with changed or delisted tickers may stay unresolved | NO LIVE RUN | `src/precedent/spine/universe.py`, `tests/spine/test_universe.py` | `resolve_instrument` returns None for unmapped CIKs instead of guessing |
| 2026-11-07 | EDGAR 8-K events: field names, form 8-K item list and state codes are recalled or taken from the amendment spec, not read from a live submissions document | NO LIVE RUN | `src/precedent/spine/edgar_client.py`, `src/precedent/spine/edgar_8k_events.py`, `tests/spine/test_edgar_8k_events.py` | Needs the owner's declared User-Agent contact (`PRECEDENT_SEC_USER_AGENT`); item rules are proposals |
