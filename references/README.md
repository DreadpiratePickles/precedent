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

## Ripple

- `ripple_params.json`: the Ripple propagation, decay, link-expiry and correlation-peer parameters (design section 9.5) and the placebo noise floor (`placebo_margin_floor`, `placebo_margin_se_multiple`: a graph beats its placebo only when the margin clears the larger of the floor and that many standard errors of the per-fold difference), all proposed; read by `load_ripple_params` in `precedent.ripple.params`.
- `rungs/m3_ripple_*.yaml`: the Ripple ablation rungs (G1 sector, G2 correlation, G3 10-K text similarity, G4 model-extracted links, degree-preserving placebo), each `panel_prefixes: ["news_", "ripple_"]`; which graph builds the `ripple_` panel is decided by the caller, not by these files.
- `link_prompt_v1.txt`: the prompt `parse_links` sends for the G4 model-extracted links (`{filer}` and `{passage}` placeholders, closed relation and kind lists); the owner approves it before any run, and the Loop in `stages/30_ripple_links/CONTEXT.md` may add `link_prompt_v<N>.txt` versions.
