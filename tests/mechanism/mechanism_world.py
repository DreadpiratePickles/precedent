"""A deterministic synthetic world shared by the Mechanism tests (a helper module, not collected).

`make_world` builds, the honest way, everything the Precedent engine reads: a session axis and its
cutoffs, a first-report events table of central-bank events with a random balanced direction sign,
daily returns with a planted event response, and the asset library from the REAL
`build_asset_library`. Tests 3b, 5b and 7b import it as `from mechanism_world import make_world`
(pytest puts this directory on `sys.path`, so it works alone and in the full suite).

Public API
----------
- `make_world(seed=0, n_sessions=900, first_event_position=300, spacing=12, severity_band="mid",
  is_scheduled=True, shocks=None) -> World`
- `World` (frozen): `events`, `rows`, `session_cutoffs_utc`, `sessions`, `returns`, `market`
  ("SPY"), `assets` (every return column, market included), `event_positions`, `planted`, `betas`;
  methods `event_id_at(position)`, `available_utc_at(position)`, `key(direction_sign=0)`,
  `planted_car(asset)`.
- `ASSETS`, `MARKET`, `HORIZONS`, `PLANTED_EFFECT`, `BETAS`: the planted constants.

The planted response. A central-bank event of direction d (+1 favourable, -1 unfavourable) adds
`PLANTED_EFFECT[asset] * d` to each asset's return on the first window session, `p + 2` (the
engine's window is `p + 2 .. p + 1 + h`, so the whole effect is inside every horizon). The signs
follow `references/mechanism/templates/central_bank.json`: a favourable event (d = +1) lowers
rates_expectation, which lifts TLT and risk_appetite (hence SPY and HYG) and weakens UUP.
The market (SPY) is moved too, so every other asset's cumulative abnormal return carries roughly `effect - beta_hat * market_effect`, where the market-model beta is pulled toward the
effect ratio by the event days in its own estimation window: that expectation is `planted_car`.

Events sit every `spacing` sessions from `first_event_position`; an event at position p is
available one hour before the cutoff of session p, so its window completes at session
`p + 1 + h`. With h = 20 and the default spacing of 12, the event just before a query is
unfinished at the query (the time guard drops it); with h = 5 it is finished.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from precedent.libraries.asset import build_asset_library
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS
from precedent.spine.event_key import EventKey, SeverityBand
from precedent.spine.pit_alignment import build_session_cutoffs

MARKET = "SPY"
ASSETS = ("SPY", "TLT", "HYG", "UUP")
HORIZONS = (1, 5, 20)
BETAS: dict[str, float] = {"SPY": 1.0, "TLT": 0.1, "HYG": 0.3, "UUP": -0.2}
PLANTED_EFFECT: dict[str, float] = {"SPY": 0.006, "TLT": 0.02, "HYG": 0.015, "UUP": -0.01}
_MARKET_SD = 0.003
_IDIO_SD = 0.002
_ONE_HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class World:
    """Everything the engine and the panel builder read, plus the planted constants."""

    events: pd.DataFrame
    rows: pd.DataFrame
    session_cutoffs_utc: pd.Series
    sessions: pd.DatetimeIndex
    returns: pd.DataFrame
    market: str
    assets: list[str]
    event_positions: list[int]
    planted: dict[str, float]
    betas: dict[str, float]
    severity_band: SeverityBand
    is_scheduled: bool
    spacing: int

    def event_id_at(self, position: int) -> str:
        """The `event_id` of the event effective at session `position`."""
        hit = self.events.loc[self.events["effective_position"] == position, "event_id"]
        if len(hit) != 1:
            raise KeyError(f"no event at position {position}")
        return str(hit.iloc[0])

    def available_utc_at(self, position: int) -> pd.Timestamp:
        """When an event effective at `position` becomes available: its cutoff minus one hour."""
        return self.session_cutoffs_utc.iloc[position] - _ONE_HOUR

    def key(self, direction_sign: int = 0) -> EventKey:
        """The central-bank event key every event in the world shares, with this direction."""
        return _key(self.severity_band, self.is_scheduled, direction_sign)

    def planted_car(self, asset: str) -> float:
        """The cumulative abnormal return per unit direction the library should carry, about.

        The market model is estimated over a window in which one session in `spacing` moves the
        asset and the market together, so the estimated beta is pulled toward the ratio of the
        two planted effects: beta_hat = (beta * var + f * e_a * e_m) / (var + f * e_m ** 2) with
        f = 1 / spacing, and the CAR is `e_a - beta_hat * e_m`. This is an expectation, accurate
        to the sampling noise of one panel, not an exact value. For the market itself it is the
        raw planted effect (the panel's market column is the market's own cumulative return).
        """
        e_market = self.planted[self.market]
        if asset == self.market:
            return e_market
        freq, var = 1.0 / self.spacing, _MARKET_SD**2
        e_asset = self.planted[asset]
        beta_hat = (self.betas[asset] * var + freq * e_asset * e_market) / (
            var + freq * e_market**2
        )
        return e_asset - beta_hat * e_market


def _key(severity_band: SeverityBand, is_scheduled: bool, direction_sign: int) -> EventKey:
    return EventKey(
        event_type="central_bank",
        actor_kind="central_bank",
        impacted_kinds=("index",),
        country_group="US",
        severity_band=severity_band,
        is_scheduled=is_scheduled,
        direction_sign=direction_sign,
    )


def _events_table(
    positions: list[int],
    directions: np.ndarray,
    sessions: pd.DatetimeIndex,
    cutoffs: pd.Series,
    severity_band: SeverityBand,
    is_scheduled: bool,
) -> pd.DataFrame:
    out = []
    for i, (position, direction) in enumerate(zip(positions, directions, strict=True)):
        event_id = f"e{i:03d}"
        out.append(
            {
                "event_id": event_id,
                "key": _key(severity_band, is_scheduled, int(direction)).as_string(),
                "event_type": "central_bank",
                "country_group": "US",
                "available_utc": cutoffs.iloc[position] - _ONE_HOUR,
                "effective_session": sessions[position].date(),
                "effective_position": position,
                "primary_actor": "US",
                "linked_countries": ["US"],
                "story_cluster_id": f"story-{i:03d}",
                "is_first_report": True,
                "one_line": f"US: central_bank {int(direction):+d}",
                "source": "synthetic",
                "source_ref": f"synthetic:{i}",
            }
        )
    frame = pd.DataFrame(out, columns=list(EVENTS_TABLE_COLUMNS))
    frame["effective_position"] = frame["effective_position"].astype("int64")
    frame["available_utc"] = frame["available_utc"].astype("datetime64[ns, UTC]")
    frame["is_first_report"] = frame["is_first_report"].astype(bool)
    return frame


def make_world(
    seed: int = 0,
    n_sessions: int = 900,
    first_event_position: int = 300,
    spacing: int = 12,
    severity_band: SeverityBand = "mid",
    is_scheduled: bool = True,
    shocks: Mapping[int, float] | None = None,
) -> World:
    """Build the world; the same arguments always give the same world.

    `shocks` maps an event's effective position to a return added to EVERY asset (the market too)
    on that event's first window session `p + 2`, on top of the planted response; use it to make
    one precedent's outcome stand out. Event positions run from `first_event_position` in steps of
    `spacing`, leaving room for the longest horizon; the direction signs are a seeded shuffle of
    a balanced +1 / -1 list, so both signs occur.
    """
    sessions = pd.bdate_range("2020-01-02", periods=n_sessions)
    cutoffs = build_session_cutoffs(sessions, dt.time(16, 0))
    last_start = n_sessions - 1 - max(HORIZONS)
    positions = list(range(first_event_position, last_start, spacing))
    if len(positions) < 2:
        raise ValueError("the world needs room for at least two events")
    rng = np.random.default_rng(seed)
    directions = rng.permutation(np.where(np.arange(len(positions)) % 2 == 0, 1, -1))
    market_noise = rng.normal(0.0, _MARKET_SD, size=n_sessions)
    data = {MARKET: market_noise}
    for asset in ASSETS:
        if asset != MARKET:
            data[asset] = BETAS[asset] * market_noise + rng.normal(0.0, _IDIO_SD, size=n_sessions)
    returns = pd.DataFrame(data, index=sessions)[list(ASSETS)]
    for position, direction in zip(positions, directions, strict=True):
        for asset in ASSETS:
            returns.iloc[position + 2, returns.columns.get_loc(asset)] += (
                PLANTED_EFFECT[asset] * direction
            )
    for position, shock in (shocks or {}).items():
        returns.iloc[position + 2] += shock
    events = _events_table(positions, directions, sessions, cutoffs, severity_band, is_scheduled)
    traded = [a for a in ASSETS if a != MARKET]
    rows, _ = build_asset_library(events, returns, MARKET, sessions, lambda _e: traded, HORIZONS)
    return World(
        events=events,
        rows=rows,
        session_cutoffs_utc=cutoffs,
        sessions=sessions,
        returns=returns,
        market=MARKET,
        assets=list(ASSETS),
        event_positions=positions,
        planted=dict(PLANTED_EFFECT),
        betas=dict(BETAS),
        severity_band=severity_band,
        is_scheduled=is_scheduled,
        spacing=spacing,
    )
