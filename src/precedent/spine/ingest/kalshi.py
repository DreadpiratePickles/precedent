"""`precedent ingest kalshi [--max-requests N] [--category C ...]`."""

from __future__ import annotations

from precedent.spine.http import Clock, Transport
from precedent.spine.ingest.market_run import run_market_ingest
from precedent.spine.kalshi_client import KalshiClient
from precedent.spine.typers.base import Typer


def run(
    argv: list[str],
    transport: Transport | None = None,
    clock: Clock | None = None,
    typer: Typer | None = None,
) -> int:
    def pages_from(transport, clock, state_path, max_requests, report):
        return KalshiClient(transport, clock, state_path, max_requests).iter_pages(report)

    return run_market_ingest("kalshi", argv, pages_from, transport, clock, typer)
