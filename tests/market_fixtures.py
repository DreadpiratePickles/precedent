"""Shared market-ingest test support: the SYNTHETIC fixtures and a Kalshi fake transport."""

import json
from pathlib import Path

from fakes import FakeTransport, ok

from precedent.config import KALSHI_BASE_URL

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

POLYMARKET_FIXTURE = json.loads((_FIXTURE_DIR / "polymarket_markets.json").read_text())
KALSHI_FIXTURE = json.loads((_FIXTURE_DIR / "kalshi_markets.json").read_text())


def kalshi_transport(fixture: dict) -> FakeTransport:
    """Serve a Kalshi fixture by URL: cutoff, live and historical markets, events, series."""
    response = fixture["response"]
    mapping = {
        f"{KALSHI_BASE_URL}/historical/cutoff": ok(response["cutoff"]),
        f"{KALSHI_BASE_URL}/markets": ok(response["live"]),
        f"{KALSHI_BASE_URL}/historical/markets": ok(response["historical"]),
    }
    for ticker, body in response["events"].items():
        mapping[f"{KALSHI_BASE_URL}/events/{ticker}"] = ok(body)
    for ticker, body in response["series"].items():
        mapping[f"{KALSHI_BASE_URL}/series/{ticker}"] = ok(body)
    return FakeTransport(mapping)
