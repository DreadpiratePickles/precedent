# CLAUDE.md

Layer 0 of this ICM workspace: where you are, the one rule, the global constraints, and where to go next.

## What this is

Precedent (distribution `precedent-engine`, import root `precedent`) finds past outcomes of events similar to a new one, from news, markets and filings, and reports them as history, not as a forecast. The shared spine (`src/precedent/spine/`, stages 01 to 09) ingests, parses, aligns and prices; the engine, libraries and research tracks build on it.

## The one rule

The language model never forecasts. It reads, and emits structure: what kind of event this is, who is involved, what it is linked to, what it resembles. Qlib owns time and numbers, and price history decides what that structure is worth. (Design section 1.2.)

## Global Constraints

- Python `>=3.11`, uv with `uv.lock` committed, hatchling, `src/` layout. Distribution `precedent-engine`, import root `precedent`. Console scripts `precedent` -> `precedent.cli:main` and `precedent-mcp` -> `precedent.mcp_server:main`. Ruff line length 100, rules E, F, I, UP, B, SIM. No `addopts` in `pyproject.toml`. Tests run with `uv run pytest -q`.
- Core dependencies only in `[project.dependencies]`; `pyqlib`, `lightgbm`, `scipy`, `pyyaml` and `networkx` only in the `research` extra. A module that imports any of them is a research module: its test module starts with `pytest.importorskip("<that package>")`, so a core install with only the `dev` extra still passes. CI installs every extra.
- No provider SDKs. Only `spine/typers/anthropic_typer.py`, `spine/typers/ollama_typer.py` and `spine/openai_compat_extractor.py` send text to a model, over `httpx`; `spine/event_parser.py` calls an injected extractor. Everything else is deterministic. No model is called in tests or CI.
- `src/precedent/config.py` is the only module that names a model id, a base URL or a data-dir default (a test greps for it). Environment variables: `PRECEDENT_SEC_USER_AGENT`, `PRECEDENT_DATA_DIR` (default `~/.cache/precedent`), `PRECEDENT_TYPER` (default `rules`), `PRECEDENT_OLLAMA_BASE_URL`, `PRECEDENT_OLLAMA_MODEL`, `PRECEDENT_MODEL_BASE_URL` (llama.cpp parser runtime), `ANTHROPIC_API_KEY`. No contact detail or secret is written into the repository.
- `first_seen_utc` is required on every record. Naive timestamps are rejected. Time is tz-aware UTC internally. GDELT `SQLDATE` and EDGAR `filingDate` are never used for alignment.
- Every factor builder ships with a truncation-invariance test (design 4.5).
- `qlib.init` always passes `region="us"` and an SQLite experiment manager. In `NestedDataLoader` the price loader is first and `join="left"`. Label under convention C1: `Ref($open, -2)/Ref($open, -1) - 1`; `deal_price` is `open`; `label_horizon` is 2.
- No article text, price data, raw source files or model weights are committed. Git ignores `stages/*/output/`, `data/*` except `data/sample/`, `vendor/`, `.env`, `dist/`, `.venv/`, `__pycache__/`. Nothing in the repository displays or redistributes vendor prices.
- Unit tests never touch the network: every client takes an injected `Transport` and `Clock`. Test file basenames are unique across `tests/` (pytest's default import mode). Tests write only under `tmp_path`; CI checks `git status --porcelain` is empty afterwards.
- Zero spend. Nothing pushes, merges, downloads a model or dataset, calls a paid API or rents compute without the owner's yes; such steps are marked **owner gate** and name the source, size and cost first. The remote is the owner's private GitHub repository; the agent pushes only when the owner asks, and never merges to `main`.
- Stage explicit files in every commit. Every result is recorded in `docs/STATUS.md` with one label: `SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL` or `NO LIVE RUN`. Every run-produced file under `docs/examples/` begins with `SYNTHETIC —`, `LIVE —` or `NO LIVE RUN —` (a test pins it).
- If Qlib, a data source or an API behaves differently from the spec, stop the task and report. Do not invent a substitute.
- The track ends with an independent verifier (Task 21), whose report is kept under `docs/verification/`; findings are fixed in `fix: independent-review findings` commits.

## Navigation

- `CONTEXT.md` routes a task type to a stage.
- `stages/NN_name/CONTEXT.md` is the stage contract: objective, inputs, process, outputs, verify, approval, failure behaviour, and a `## Loop` section when the stage loops.
- `references/` holds stable Layer 3 rules; `stages/*/output/` holds per-run artifacts and is not committed.
- `docs/superpowers/specs/` holds the design (`2026-11-06-predate-design.md`) and the product amendment; `docs/superpowers/plans/` holds the implementation plans.
- `docs/STATUS.md` records every result with its label; `docs/examples/README.md` states the evidence-banner rule.
- Run `uv run ruff check . && uv run pytest -q` before every commit.
