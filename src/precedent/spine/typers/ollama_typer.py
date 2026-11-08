"""The Ollama typer: one local /api/chat call per text, schema-constrained, no SDK."""

from __future__ import annotations

from typing import Any

from precedent.config import require_env
from precedent.spine.http import Clock, HttpRequest, HttpxTransport, SystemClock, Transport
from precedent.spine.typers.typing_response import (
    ModelTyper,
    ReplyRefused,
    load_typing_prompt,
    prompt_digest,
    typing_response_json_schema,
)

DEFAULT_MAX_CALLS = 100


class OllamaTyper(ModelTyper):
    def __init__(
        self,
        base_url: str,
        model: str,
        transport: Transport,
        clock: Clock,
        max_calls: int = DEFAULT_MAX_CALLS,
        timeout_s: float = 60.0,
        dry_run: bool = False,
    ) -> None:
        super().__init__(transport, clock, max_calls, dry_run)
        self._chat_url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._timeout_s = timeout_s
        self._prompt = load_typing_prompt()
        self.version = f"ollama:{model}:prompt-{prompt_digest(self._prompt)}"

    @classmethod
    def from_env(
        cls,
        transport: Transport | None = None,
        clock: Clock | None = None,
        *,
        max_calls: int = DEFAULT_MAX_CALLS,
        timeout_s: float = 60.0,
        dry_run: bool = False,
    ) -> OllamaTyper:
        base_url = require_env("PRECEDENT_OLLAMA_BASE_URL")
        model = require_env("PRECEDENT_OLLAMA_MODEL")
        return cls(
            base_url,
            model,
            transport if transport is not None else HttpxTransport(),
            clock if clock is not None else SystemClock(),
            max_calls=max_calls,
            timeout_s=timeout_s,
            dry_run=dry_run,
        )

    def _build_request(self, text: str) -> HttpRequest:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": text},
            ],
            "format": typing_response_json_schema(),
            "stream": False,
            "options": {"temperature": 0},
        }
        return HttpRequest(
            method="POST",
            url=self._chat_url,
            headers={"content-type": "application/json"},
            json_body=body,
            timeout_s=self._timeout_s,
        )

    def _model_json(self, body: Any) -> str:
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ReplyRefused("schema: reply has no message content")
        return content


def from_env(max_calls: int | None = None, dry_run: bool = False) -> OllamaTyper:
    """The selector's entry point: `max_calls=None` means the default limit."""
    return OllamaTyper.from_env(
        max_calls=DEFAULT_MAX_CALLS if max_calls is None else max_calls, dry_run=dry_run
    )
