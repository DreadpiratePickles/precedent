"""The Dream note behind the brief example: what `precedent dream` writes for synthetic stores.

`tests/dream/fixtures/dream_2024-06-03.md` is this function's output with a banner line on top
(D67). `scripts/build_brief_example.py --refresh-fixture` rewrites it; a test re-renders it and
compares, so the example can never show a shape the command does not produce. Everything is
generated under `root`; no network, no model, no clock.
"""

from __future__ import annotations

from pathlib import Path

from dream_synthetic import (
    customer_link,
    gdelt_record,
    plant_capex_conflict,
    write_events,
    write_links,
    write_prices,
)

from precedent.dream import report

SESSION = "2024-06-03"
FIXTURE_BANNER = "SYNTHETIC — rendered by `precedent dream` from generated stores; not market data."


def render_example_dream(root: Path) -> str:
    """Run the real command on generated stores and return the note for 2024-06-03."""
    events, links, prices, out = (root / name for name in ("events", "links", "prices", "out"))
    write_events(
        events,
        [
            gdelt_record(1, "2023-09-01T15:00:00Z", instrument=None),  # history before the session
            gdelt_record(
                100,
                "2024-05-31T15:00:00Z",
                instrument="BETA",
                event_type="product",
                span="Beta unveiled a new product line",
            ),
            gdelt_record(
                101,
                "2024-06-03T14:00:00Z",
                span="Acme raised its full-year outlook",
                event_type="earnings",
            ),
            gdelt_record(
                103,
                "2024-06-03T16:00:00Z",
                instrument="BETA",
                event_type="mna",
                span="Beta agreed to acquire a rival",
            ),
        ],
    )
    write_links(links, [customer_link()])
    write_prices(prices, "2023-06-01", "2024-06-07", shocks={("ACME", SESSION): -0.08})
    plant_capex_conflict(out)
    argv = [
        "--date", SESSION, "--out", str(out), "--events", str(events), "--links", str(links),
        "--prices", str(prices), "--market", "MKT",
    ]  # fmt: skip
    if report.run(argv) != 0:
        raise RuntimeError("the example dream did not run")
    return (out / "dreams" / f"dream_{SESSION}.md").read_text(encoding="utf-8")


def fixture_text(root: Path) -> str:
    """The fixture file: the banner, a blank line, then the real note."""
    return f"{FIXTURE_BANNER}\n\n{render_example_dream(root)}"
