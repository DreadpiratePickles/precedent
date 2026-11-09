"""The named event sources of a library and the citation each one requires."""

from __future__ import annotations

from precedent.spine.edgar_8k_events import SOURCE as EDGAR_SOURCE
from precedent.spine.gdelt_events import SOURCE_NAME as GDELT_SOURCE

__all__ = ["CITATIONS", "EVENT_SOURCES"]

EVENT_SOURCES = (("gdelt2", GDELT_SOURCE), ("edgar_8k", EDGAR_SOURCE))
CITATIONS = {
    "gdelt2": (
        "The GDELT Project (gdeltproject.org), GDELT 2.0 Event Database; "
        "citation and link required on redistribution"
    ),
    "edgar_8k": "U.S. Securities and Exchange Commission, EDGAR submissions API",
    "polymarket": "Polymarket Gamma API, resolved markets (ids, typed keys and dates only)",
    "kalshi": "Kalshi trade API, settled markets (ids, typed keys and dates only)",
}
