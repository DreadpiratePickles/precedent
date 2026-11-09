"""The `precedent-mcp` server: the precedent card as two read-only tools over stdio.

`precedents` returns the card as JSON, `card` returns the text card. Both take either free
`text` (typed with the configured typer) or a full event `key`, never both. Nothing is written.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from precedent import config
from precedent.config import ConfigError
from precedent.engine.ask import UNTYPED, refused_card_without_library, run_card
from precedent.engine.card import KEY_FLAGS_HINT, CardModel, render_json, render_text
from precedent.engine.keys import key_from_dict
from precedent.engine.retrieval import PrecedentQuery
from precedent.libraries.bundle import LibraryMissingError
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.spine.typers.select import typer_from_env

server = MCPServer("precedent")

_P = ParamSpec("_P")
_R = TypeVar("_R")
MISSING_LIBRARY_HINT = (
    "run `precedent fetch` to install a snapshot or `precedent build` to make one"
)


def _build_card(
    text: str | None, key: dict[str, Any] | None, horizon: int, outcome_type: str
) -> CardModel:
    if (text is None) == (key is None):
        raise ValueError("give exactly one of `text` and `key`")
    query: PrecedentQuery | str
    if key is not None:
        query = PrecedentQuery(key_from_dict(key), available_utc=pd.Timestamp.now(tz="UTC"))
    elif text is not None:
        query = text
    else:  # unreachable: exactly one of the two is given
        raise ValueError("give exactly one of `text` and `key`")
    typer = typer_from_env()
    try:
        return run_card(query, horizon, outcome_type, config.data_dir(), typer)
    except LibraryMissingError:
        if key is not None:
            raise
        # V-4: text the typer cannot type is refused whether or not a library is installed
        refusal = refused_card_without_library(query, horizon, outcome_type, typer)
        if refusal.typed_as != UNTYPED:
            raise
        return refusal


def precedents(
    text: str | None = None,
    key: dict[str, Any] | None = None,
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


def card(text: str | None = None, key: dict[str, Any] | None = None, horizon: int = 5) -> str:
    """The precedent card as text. Give either `text` or a full event `key`."""
    model = _build_card(text, key, horizon, "all")
    rendered = render_text(model)
    if model.typed_as == UNTYPED:
        return rendered + "\n" + KEY_FLAGS_HINT
    return rendered


def _tool_errors(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Re-raise the expected failures as `ToolError`, the only error text MCP sends to a client.

    The wrapped module function keeps raising its own exception; only the registered tool
    changes. Anything else is a crash and stays one.
    """

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except LibraryMissingError as error:
            raise ToolError(f"{error}; {MISSING_LIBRARY_HINT}") from error
        except ConfigError as error:
            raise ToolError(f"{error}\n{KEY_FLAGS_HINT}") from error
        except (ValueError, KeyError, LibraryIntegrityError) as error:
            raise ToolError(error.args[0] if isinstance(error, KeyError) else str(error)) from error

    return wrapper


server.tool(name="precedents")(_tool_errors(precedents))
server.tool(name="card")(_tool_errors(card))


def main() -> None:
    """Serve over stdio until the client closes the pipe."""
    server.run()


def run(argv: list[str]) -> int:
    """Target of the `precedent mcp` command; `argv` is ignored."""
    main()
    return 0


if __name__ == "__main__":
    main()
