"""The `precedent-mcp` server: the precedent card as two read-only tools over stdio.

`precedents` returns the card as JSON, `card` returns the text card. Both take either free
`text` (typed with the configured typer) or a full event `key`, never both. Nothing is written.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from mcp.server.mcpserver import MCPServer

from precedent import config
from precedent.engine.ask import UNTYPED, run_card
from precedent.engine.card import KEY_FLAGS_HINT, CardModel, render_json, render_text
from precedent.engine.keys import key_from_dict
from precedent.engine.retrieval import PrecedentQuery
from precedent.spine.typers.select import typer_from_env

server = MCPServer("precedent")


def _build_card(text: str | None, key: dict | None, horizon: int, outcome_type: str) -> CardModel:
    if (text is None) == (key is None):
        raise ValueError("give exactly one of `text` and `key`")
    query: PrecedentQuery | str
    if key is not None:
        query = PrecedentQuery(key_from_dict(key), available_utc=pd.Timestamp.now(tz="UTC"))
    else:
        assert text is not None
        query = text
    return run_card(query, horizon, outcome_type, config.data_dir(), typer_from_env())


def precedents(
    text: str | None = None,
    key: dict | None = None,
    horizon: int = 5,
    outcome_type: str = "all",
) -> dict:
    """Past outcomes of events like this one, as the card's JSON (history, not a forecast).

    Give either `text` (a headline or description) or `key` (a full event key with the fields
    event_type, actor_kind, severity_band, is_scheduled and optionally country_group,
    impacted_kinds, direction_sign).
    """
    card = _build_card(text, key, horizon, outcome_type)
    result: dict[str, Any] = json.loads(render_json(card))
    return result


def card(text: str | None = None, key: dict | None = None, horizon: int = 5) -> str:
    """The precedent card as text. Give either `text` or a full event `key`."""
    model = _build_card(text, key, horizon, "all")
    rendered = render_text(model)
    if model.typed_as == UNTYPED:
        return rendered + "\n" + KEY_FLAGS_HINT
    return rendered


server.tool()(precedents)
server.tool()(card)


def main() -> None:
    """Serve over stdio until the client closes the pipe."""
    server.run()


def run(argv: list[str]) -> int:
    """Target of the `precedent mcp` command; `argv` is ignored."""
    main()
    return 0


if __name__ == "__main__":
    main()
