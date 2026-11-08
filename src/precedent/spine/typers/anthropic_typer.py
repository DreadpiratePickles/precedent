"""The Anthropic typer: one Messages API call per text, structured output, no SDK."""

from __future__ import annotations

from typing import Any

from precedent import config
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
MAX_TOKENS = 1024  # leaves room for adaptive thinking before the JSON answer
EFFORT = "low"


class AnthropicTyper(ModelTyper):
    def __init__(
        self,
        api_key: str,
        model: str,
        transport: Transport,
        clock: Clock,
        max_calls: int = DEFAULT_MAX_CALLS,
        timeout_s: float = 30.0,
        dry_run: bool = False,
    ) -> None:
        super().__init__(transport, clock, max_calls, dry_run)
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._prompt = load_typing_prompt()
        self.version = f"anthropic:{model}:prompt-{prompt_digest(self._prompt)}"

    @classmethod
    def from_env(
        cls,
        transport: Transport | None = None,
        clock: Clock | None = None,
        *,
        max_calls: int = DEFAULT_MAX_CALLS,
        timeout_s: float = 30.0,
        dry_run: bool = False,
    ) -> AnthropicTyper:
        api_key = require_env("ANTHROPIC_API_KEY")  # a missing key fails before any request
        return cls(
            api_key,
            config.ANTHROPIC_TYPER_MODEL,
            transport if transport is not None else HttpxTransport(),
            clock if clock is not None else SystemClock(),
            max_calls=max_calls,
            timeout_s=timeout_s,
            dry_run=dry_run,
        )

    def _build_request(self, text: str) -> HttpRequest:
        # No temperature, top_p, top_k, thinking or prefill: each is a 400 on this API version.
        body = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": self._prompt,
            "messages": [{"role": "user", "content": text}],
            "output_config": {
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": typing_response_json_schema()},
            },
        }
        return HttpRequest(
            method="POST",
            url=f"{config.ANTHROPIC_BASE_URL}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": config.ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json_body=body,
            timeout_s=self._timeout_s,
        )

    def _model_json(self, body: Any) -> str:
        if not isinstance(body, dict):
            raise ReplyRefused("schema: reply is not a JSON object")
        stop_reason = body.get("stop_reason")
        if stop_reason == "refusal":
            raise ReplyRefused("model refusal")
        if stop_reason == "max_tokens":
            raise ReplyRefused("truncated")
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise ReplyRefused("schema: reply has no content blocks")
        # A reply may open with thinking blocks, so read text blocks, never content[0].
        return "".join(
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )


def from_env(max_calls: int | None = None, dry_run: bool = False) -> AnthropicTyper:
    """The selector's entry point: `max_calls=None` means the default limit."""
    return AnthropicTyper.from_env(
        max_calls=DEFAULT_MAX_CALLS if max_calls is None else max_calls, dry_run=dry_run
    )
