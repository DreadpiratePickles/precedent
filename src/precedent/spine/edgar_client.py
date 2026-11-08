"""SEC EDGAR submissions client: a declared User-Agent, evenly spaced requests, bounded retries."""

from __future__ import annotations

import json
from typing import Any

from precedent.config import SEC_SUBMISSIONS_BASE_URL, ConfigError, require_env
from precedent.spine.http import (
    Clock,
    HttpError,
    HttpRequest,
    HttpxTransport,
    RateLimiter,
    RateLimitError,
    SystemClock,
    Transport,
    send_with_retries,
)

USER_AGENT_ENV = "PRECEDENT_SEC_USER_AGENT"
DEFAULT_MAX_REQUESTS_PER_SECOND = 5.0

__all__ = [
    "ConfigError",
    "EdgarClient",
    "EdgarError",
    "EdgarRateLimitError",
]


class EdgarError(RuntimeError):
    """A submissions request failed or returned something that is not a submissions document."""


class EdgarRateLimitError(EdgarError):
    """EDGAR kept answering 429 after the retry budget was spent."""


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        transport: Transport,
        clock: Clock,
        max_requests_per_second: float = DEFAULT_MAX_REQUESTS_PER_SECOND,
        max_retries: int = 3,
    ) -> None:
        if max_requests_per_second <= 0:
            raise ValueError(
                f"max_requests_per_second must be positive, got {max_requests_per_second}"
            )
        self._user_agent = user_agent
        self._transport = transport
        self._clock = clock
        self._max_retries = max_retries
        self._limiter = RateLimiter(1, 1 / max_requests_per_second, clock)

    @classmethod
    def from_env(
        cls, transport: Transport | None = None, clock: Clock | None = None
    ) -> EdgarClient:
        """Build a client from `PRECEDENT_SEC_USER_AGENT`; raises ConfigError when unset."""
        return cls(
            require_env(USER_AGENT_ENV),
            transport if transport is not None else HttpxTransport(),
            clock if clock is not None else SystemClock(),
        )

    def fetch_submissions(self, cik: int) -> dict[str, Any]:
        """The submissions document for one CIK (zero-padded to ten digits in the URL)."""
        name = f"CIK{cik:010d}"
        request = HttpRequest(
            "GET",
            f"{SEC_SUBMISSIONS_BASE_URL}/{name}.json",
            {"User-Agent": self._user_agent, "Accept": "application/json"},
        )
        self._limiter.acquire()
        try:
            response = send_with_retries(
                self._transport, self._clock, request, max_retries=self._max_retries
            )
        except RateLimitError as error:
            raise EdgarRateLimitError(
                f"EDGAR rate-limited the request for {name}: {error}"
            ) from error
        except HttpError as error:
            raise EdgarError(f"EDGAR request for {name} failed: {error}") from error
        try:
            document = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise EdgarError(f"EDGAR response for {name} is not valid JSON: {error}") from error
        if not isinstance(document, dict):
            raise EdgarError(
                f"EDGAR response for {name} is not a JSON object (got {type(document).__name__})"
            )
        return document
