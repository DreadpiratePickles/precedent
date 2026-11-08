# references/

Layer 3 of the ICM workspace: stable constraints that stages internalise, never per-run inputs.

Expected contents, added by the tasks that need them: event schema JSON, typing rules, the CAMEO map, country groups, cost assumptions, the asset panel and parameter files. Each file is read by a stage only when that stage's `CONTEXT.md` lists it in its Inputs table.

This folder ships inside the wheel as `precedent/references/` (see `pyproject.toml`). Per-run artifacts belong in `stages/*/output/`, which Git ignores. Nothing here holds article text, price data, secrets or model weights.
