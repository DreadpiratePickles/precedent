"""Aggregate the responses of similar past events into a summary, or refuse.

Pure numpy. The summary is history, not a forecast: weighted quantiles, a shrunk centre, a hit
rate, or for a yes/no response a rate with a Wilson interval. When the effective sample size is
too small the summary is refused and keeps only the counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class PrecedentSummary:
    mode: Literal["continuous", "binary"]
    n: int
    n_eff: float
    refused: bool
    quantiles: dict[float, float] | None = None
    median: float | None = None
    centre_shrunk: float | None = None
    hit_rate: float | None = None
    k: int | None = None
    rate: float | None = None
    interval: tuple[float, float] | None = None
    placebo_rate: float | None = None
    inside_ordinary_range: bool | None = None


def _check_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got {w.ndim} dimensions")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return w


def effective_sample_size(
    weights: np.ndarray, window_start_positions: np.ndarray, horizon: int
) -> float:
    """The smaller of the Kish size and the count of non-overlapping windows.

    Kish is ``(sum w)^2 / sum w^2``. The window count is greedy in ascending position order: a
    start is accepted when it is at least ``horizon`` after the last accepted one. Empty input
    gives 0.0, and so do weights that are all zero.
    """
    w = _check_weights(weights)
    starts = np.asarray(window_start_positions)
    if starts.ndim != 1 or starts.shape[0] != w.shape[0]:
        raise ValueError(
            f"weights ({w.shape[0]}) and window_start_positions {starts.shape} differ in length"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if w.size == 0:
        return 0.0
    sum_sq = float(np.sum(w * w))
    if sum_sq == 0.0:
        return 0.0
    kish = float(np.sum(w)) ** 2 / sum_sq
    count = 0
    last_accepted: float | None = None
    for start in np.sort(starts):
        if last_accepted is None or start >= last_accepted + horizon:
            count += 1
            last_accepted = start
    return float(min(kish, count))


def wilson_interval(k: float, n: float, z: float = 1.2816) -> tuple[float, float]:
    """Wilson score interval for the proportion ``k / n``, clipped to [0, 1]."""
    if not n > 0:
        raise ValueError(f"n must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must lie in [0, n], got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half = z / denominator * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[float, float]:
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    midpoints = (np.cumsum(w) - w / 2.0) / np.sum(w)
    return {q: float(np.interp(q, midpoints, v)) for q in QUANTILE_LEVELS}


def summarize_responses(
    values: np.ndarray,
    weights: np.ndarray,
    window_start_positions: np.ndarray,
    horizon: int,
    placebo: np.ndarray | float | None,
    *,
    is_binary: bool = False,
    shrink_k: float = 10.0,
    min_neff: float = 8.0,
    z: float = 1.2816,
) -> PrecedentSummary:
    """Summarize weighted responses; refuse when the effective sample size is below ``min_neff``.

    Continuous ``placebo`` is an array of placebo responses; binary ``placebo`` is the
    unconditional rate. A refused summary keeps ``n``, ``n_eff`` and, for binary, ``k``. An
    all-zero weight vector (``n_eff == 0``) is refused even if ``min_neff`` is not positive.
    """
    v = np.asarray(values, dtype=float)
    w = _check_weights(weights)
    if v.ndim != 1 or v.shape[0] != w.shape[0]:
        raise ValueError(f"values {v.shape} and weights {w.shape} differ in length")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    mode: Literal["continuous", "binary"] = "binary" if is_binary else "continuous"
    if is_binary:
        if not np.all((v == 0.0) | (v == 1.0)):
            raise ValueError("binary values must be 0.0 or 1.0")
        if placebo is not None and np.ndim(placebo) != 0:
            raise ValueError("a binary summary needs a float placebo rate, not an array")
    elif placebo is not None and np.ndim(placebo) == 0:
        raise ValueError("a continuous summary needs an array of placebo values, not a float")

    n = int(v.shape[0])
    n_eff = effective_sample_size(w, window_start_positions, horizon)
    k = int(np.sum(v == 1.0)) if is_binary else None
    if n_eff < min_neff or n_eff <= 0.0:
        return PrecedentSummary(mode=mode, n=n, n_eff=n_eff, refused=True, k=k)

    if is_binary:
        rate = float(np.sum(w * v) / np.sum(w))
        interval = wilson_interval(min(rate * n_eff, n_eff), n_eff, z)
        placebo_rate = None if placebo is None else float(placebo)
        inside = None if placebo_rate is None else interval[0] <= placebo_rate <= interval[1]
        return PrecedentSummary(
            mode=mode,
            n=n,
            n_eff=n_eff,
            refused=False,
            k=k,
            rate=rate,
            interval=interval,
            placebo_rate=placebo_rate,
            inside_ordinary_range=inside,
        )

    quantiles = _weighted_quantiles(v, w)
    median = quantiles[0.5]
    inside = None
    if placebo is not None:
        placebo_values = np.asarray(placebo, dtype=float)
        if placebo_values.size > 0:
            low, high = np.percentile(placebo_values, [10, 90])
            inside = bool(low <= median <= high)
    return PrecedentSummary(
        mode=mode,
        n=n,
        n_eff=n_eff,
        refused=False,
        quantiles=quantiles,
        median=median,
        centre_shrunk=median * n_eff / (n_eff + shrink_k),
        hit_rate=float(np.sum(w[v > 0]) / np.sum(w)),
        inside_ordinary_range=inside,
    )
