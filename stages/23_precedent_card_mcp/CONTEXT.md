# Stage: 23_precedent_card_mcp

## Objective
Render the precedent card for a typed event as text, JSON and PNG, and serve the same card over MCP, using code only: no model and no network.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `library/` bundle | 4 | Authoritative for precedents and outcomes | Yes | Verified against `MANIFEST.json` |
| An event key from `precedent.engine.keys` | 4 | Authoritative for the query | Yes | `typed_as_line` |
| `references/precedent_params.json` | 3 | Authoritative for `top_k`, `nearest_shown`, `card_rows_per_type` | Yes | `engine` section |
| A font file for the PNG | 3 | Layout only | No | Text and JSON work without it |

## Process
1. Deterministic: match the key, take the top `top_k` precedents known before the query time, and group outcomes by type.
2. Deterministic: show up to `card_rows_per_type` rows per type and the `nearest_shown` closest past events.
3. Deterministic: write the mandatory lines in this order: the typed-as line, then `library covers` (date range and counts), then the rows, then the disclaimer `These are past outcomes of similar events, not a forecast.`
4. Deterministic: render text and JSON; render the PNG from the same data.
5. Deterministic: expose the card as an MCP tool with the same output; the server makes no network call.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| Card text and JSON | Plain text; JSON with the same fields | Human, `precedent` CLI, MCP client |
| Card PNG | Image of the same card | Human |

## Verify
- A golden test pins the text card; a test fails if `library covers` or the disclaimer line is missing.
- The JSON, text and MCP outputs agree on every number.
- No test touches the network or a model; `uv run pytest -q` exits 0.

## Approval
Owner gate: downloading a font is the owner's step; name the source, size and cost first. Without it the PNG is skipped and the text card is still produced. No other external action occurs.

## Failure Behavior
A library that fails its manifest check stops the card before any file is read. An event with fewer than `min_neff` effective matches gets a card that says so; it never gets an estimate. No retries.

Loop: single pass. Rendering is deterministic and decided by golden tests.
