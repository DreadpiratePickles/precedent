# references/

Layer 3 of the ICM workspace: stable constraints that stages internalise, never per-run inputs.

Expected contents, added by the tasks that need them: event schema JSON, typing rules, the CAMEO map, country groups, cost assumptions, the asset panel and parameter files. Each file is read by a stage only when that stage's `CONTEXT.md` lists it in its Inputs table.

This folder ships inside the wheel as `precedent/references/` (see `pyproject.toml`). Per-run artifacts belong in `stages/*/output/`, which Git ignores. Nothing here holds article text, price data, secrets or model weights.

## GDELT 2.0 files

- `cameo_eventcodes.json`: CAMEO event code descriptions, a parsed copy of the GDELT lookup `CAMEO.eventcodes.txt` (12,157 bytes, read 2026-11-07; SHA-256 and size are inside the file). Read by `load_cameo_map`.
- `cameo_to_event_type.json`: root and subcode rules from CAMEO code to `event_type`; every row is a proposal until the owner approves it (see `stages/02_triage/CONTEXT.md`).
- `precedent_params.json`: tunable parameters, one section per user; the `gdelt` section holds the GDELT filter, severity and latency values, all proposed (decision B4).
- Citation terms, from the GDELT Project's terms page (`gdeltproject.org/about.html`, read 2026-11-07): the data may be used without fee for academic, commercial or governmental use, and may be redistributed, rehosted or mirrored in any form, but any use or redistribution "must include a citation to the GDELT Project" and a link to `https://www.gdeltproject.org/`.
- Citation wording used here, for any surface that shows GDELT-derived events or code descriptions: `Source: the GDELT Project, https://www.gdeltproject.org/`. No GDELT export row or article text is committed.

## Ablation rungs

- `rungs/*.yaml`: one file per rung of the ablation ladder, read by `load_rungs`; each has a `name` and the `panel_prefixes` of the factor panels it adds to the price features. `m2_precedent.yaml` adds `prec_` to the `news_` columns of `m1_news.yaml`; it is proposed and has never been run.
