# Stage: 02_triage

## Objective
Deduplicate stored articles, pre-match entities, and apply a daily cap, using code only, to produce the list of articles worth parsing.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| Article store from `01_ingest` | 4 | Authoritative for item identity and `first_seen_utc` | Yes | All records in scope |
| `references/` entity and country maps | 3 | Authoritative for pre-match rules | Yes | Entity pre-match |
| Triage parameters (daily cap, duplicate window) | 3 | Authoritative; every threshold is named, not hard-coded in prose | Yes | Parameters file |

## Process
1. Deterministic: group near-duplicate articles by normalised text hash within the duplicate window; keep the earliest `first_seen_utc` as the group head.
2. Deterministic: pre-match entity names to candidate instruments with the reference maps.
3. Deterministic: rank survivors by a documented rule and cut at the daily cap; log every dropped article with its reason.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/02_triage/output/triaged.jsonl` | One record per kept article: `article_id`, group head, candidate entities | `03_parse` |
| `stages/02_triage/output/dropped.jsonl` | `article_id` and reason | Human review |

## Verify
- Unit tests on duplicate grouping, cap behaviour and entity pre-match, with fixed fixtures.
- Re-running on the same store yields identical output.
- Every input article appears in exactly one of the two outputs.
- `uv run pytest -q` exits 0.

## Approval
Review the dropped-article reasons and the daily cap value. No external action occurs in this stage.

## Failure Behavior
A malformed record stops the stage with its `article_id` and the preserved cause; it is never silently skipped. No retries (no external calls). Output is rewritten atomically or not at all.

Loop: single pass. Deterministic code decided by tests; no oracle-driven iteration is needed (design 10.2).

### Source: GDELT 2.0 (filter, dedup and CAMEO table check)
- Filter, in order: `NumSources < min_sources` (3) is dropped; a row whose CAMEO code maps to `untyped` is dropped; then dedup keeps one record per (`EventKey` string, Actor1 country, effective session), the earliest `DATEADDED` winning and ties going to the smaller `GLOBALEVENTID` (compared as a number). A record with no effective session is kept and never merged. Every drop is counted in `GdeltIngestReport`; `rows_read = kept + dropped_untyped + dropped_min_sources + dropped_duplicate`. All thresholds are proposals (decision B4) in the `gdelt` section of `references/precedent_params.json`.
- Severity: `|GoldsteinScale|` below the first cut (10/3) is band low, below the second (20/3) mid, else high; a row with `NumSources >= 10` is raised one band (high stays); bands map to severity 2, 5, 8. `step` (direction) is the sign of `GoldsteinScale`.
- CAMEO table check, 2026-11-07, against `CAMEO.eventcodes.txt` (12,157 bytes, SHA-256 `ccbfa4f5...c145d`, 310 codes, `references/cameo_eventcodes.json`). Every row of `references/cameo_to_event_type.json` is a proposal (`"proposed": true`). Where the amendment's section 8 wording and the code descriptions differ:
  - Root 14 `PROTEST` maps to `election_or_political`: the codes cover demonstrations, strikes, hunger strikes, blockades and riots (140 to 145), not elections. The type is coarser than the code; kept as the proposal, flagged for the owner.
  - Root 15 `EXHIBIT FORCE POSTURE` (150 to 154: alert status, mobilisation) maps to `armed_conflict` although no fighting is described; flagged.
  - Root 18 `ASSAULT` (180 to 186) includes kidnapping, torture, bombings and assassination by any actor, so `armed_conflict` is broad; roots 19 `FIGHT` and 20 `USE UNCONVENTIONAL MASS VIOLENCE` fit.
  - Root 13 `THREATEN` maps to `other`. It contains 1312 `Threaten to boycott, embargo, or sanction`; a threat is not an imposition, so it is not typed `trade_or_sanctions`.
  - Roots 16 and 17, sanction and embargo subcodes: by description only `163` (`Impose embargo, boycott, or sanctions`) is one; it has no four-digit children. It maps to `trade_or_sanctions` when `Actor1Type1Code` is `GOV`, else it falls back to root 16 (`untyped`). `172` `Impose administrative sanctions` (1721 to 1724: restrictions on political freedoms, bans on parties, curfew, state of emergency) describes domestic political restriction, not trade or financial sanctions, so it is not in the sanction subcode list; the amendment's phrase "sanction subcode" is read as `163` alone. `162` and `1621` to `1623` (reduce or stop aid) are not sanctions either.
  - Roots 01 to 12 stay `untyped`; no mismatch found.
