# Precedent

Precedent takes a new event (a headline, a filing, a market question) and shows how events of the same kind turned out before, using news, prediction markets and SEC filings. It reports history, not a forecast: the output is a card with counts, the dates behind them, and a line saying what the library covers. When it cannot tell what kind of event a text describes, it refuses and says so. The language model never predicts anything; it only reads text and returns structure, and price history decides what that structure is worth.

## Status

| Part | State | Evidence label |
|---|---|---|
| Spine (ingest, typing, point-in-time alignment, prices, factors, ablation machinery) | Built and tested on synthetic data and fakes | `SYNTHETIC` / `NO LIVE RUN` |
| Product track (library build, `precedent ask`, the card, MCP server) | Not started; `precedent-mcp` is a stub that exits 2 | none |
| Ripple research track (dated link graph, factors, graph ladder, daily map) | Built and tested on synthetic data and fakes; live runs and model extraction are owner gates | `SYNTHETIC` / `NO LIVE RUN` |
| Research tracks (Dream, Mechanism) | Planned only | none |
| Live data (GDELT download, EDGAR fetch, Polymarket and Kalshi crawls) | Owner gate not exercised | `NO LIVE RUN` |
| Model calls (Anthropic, Ollama typers) | Owner gate not exercised | `NO LIVE RUN` |
| Real price provider or vendor prices | Owner gate not exercised | `NO LIVE RUN` |
| Committed real-data sample under `data/sample/` | Owner gate not exercised | `NO LIVE RUN` |
| Push, merge, release | Owner gate not exercised | n/a |

Nothing is live. `docs/STATUS.md` is the authoritative per-result record.

## Run the checks

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run pytest -q
uv run python scripts/ci_ask_lifecycle.py
uv run precedent --help
```

The first command downloads packages; the rest touch neither the network nor a model.

## Honest caveats

I have never run any of this against live data, a language model, or a real price provider. Every result so far comes from synthetic data and fakes, so it shows the code does what its tests say and nothing about the real world. Specifically:

- The NYSE holiday calendar for 2010 and 2011 is rule-derived and unverified against an NYSE source. The July 3 and December 24 early closes in 2012 and 2013 rest on a secondary source only. Everything else, including the special closures, is verified against NYSE or NYSE Euronext sources.
- The CAMEO-to-event-type table and the 8-K item rules are my proposals, not confirmed against their sources.
- The time zone of EDGAR's `acceptanceDateTime` is unconfirmed. The code reads its `Z` suffix as UTC; if that is wrong, every 8-K result is early by four to five hours. The check to run on the first live fetch is in `stages/01_ingest/CONTEXT.md`.
- Free price data is survivorship-biased, so any result on it is exploratory.
- Endpoint paths and field names for SEC, Polymarket, Kalshi and the model APIs are partly recalled, not read from a live response.

## Layout

This is an ICM workspace: plain files carry the instructions and the state.

- `CLAUDE.md`: layer 0, what this is, the one rule, global constraints.
- `CONTEXT.md`: layer 1, routes a task type to a stage.
- `stages/NN_name/CONTEXT.md`: layer 2, one contract per stage (`output/` is not committed).
- `references/`: layer 3, stable rules and tables that ship in the wheel.
- `src/precedent/`: the code (`spine/` is the shared base).
- `tests/`, `scripts/`: the checks.
- `docs/`: `DEV_MANUAL.md`, `STATUS.md`, specs and plans under `docs/superpowers/`.

The developer manual is `docs/DEV_MANUAL.md`.

## Licence

Apache-2.0. See `LICENSE`.
