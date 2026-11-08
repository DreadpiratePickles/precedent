"""The seam for text-embedding similarity: a protocol, and a blend of key match and text match.

`embedding_similarity` only produces scores. It is passed to `find_precedents` as its
`similarity`, so every precedent it surfaces still goes through the time guard there; a text
embedder cannot make an unfinished outcome visible. No embedder is chosen or downloaded here.

When an embedder is adopted (Task 18, Step 6), its `release_date` goes into `precedent_version`:
a model released after an event may have seen that event in training.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from precedent.engine.retrieval import PrecedentQuery, exact_key_similarity

__all__ = ["TextEmbedder", "embedding_similarity"]

_TEXT_COLUMN = "canonical_text"


class TextEmbedder(Protocol):
    """Anything that turns texts into vectors; `release_date` is when the model was released."""

    release_date: dt.date

    def embed(self, texts: list[str]) -> np.ndarray:
        """Shape `(len(texts), dim)`, floats; row `i` embeds `texts[i]`."""
        ...


def embedding_similarity(
    embedder: TextEmbedder, weight: float
) -> Callable[[PrecedentQuery, pd.DataFrame], pd.Series]:
    """A similarity function: ``(1 - weight) * exact_key + weight * cosine(text)``.

    `weight` lies in `[0, 1]`; 0 is the exact key match alone. The cosine of the canonical
    texts is clipped to `[0, 1]`, so an event with the wrong key but a near-identical text can
    still score and enter, which is what this rung tests. When the query has no
    `canonical_text`, or the events have no `canonical_text` column, the exact-key scores are
    returned unchanged and the embedder is not called. Otherwise the query text and every
    event text go to the embedder in one call, once per evaluation.
    """
    if not isinstance(weight, int | float) or math.isnan(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight!r}")

    def similarity(query: PrecedentQuery, events: pd.DataFrame) -> pd.Series:
        exact = exact_key_similarity(query, events)
        if query.canonical_text is None or _TEXT_COLUMN not in events.columns:
            return exact
        if events.empty:
            return exact
        texts = events[_TEXT_COLUMN]
        if texts.isna().any():
            raise ValueError(f"events have null {_TEXT_COLUMN} values")
        vectors = np.asarray(embedder.embed([query.canonical_text, *texts.tolist()]), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) + 1:
            raise ValueError(
                f"embedder returned shape {vectors.shape}, expected ({len(texts) + 1}, dim)"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("embedder returned non-finite values")
        cosine = _cosine_to_first(vectors)
        blended = (1.0 - weight) * exact.to_numpy(dtype=float) + weight * cosine
        return pd.Series(blended, index=events.index, dtype="float64", name=exact.name)

    return similarity


def _cosine_to_first(vectors: np.ndarray) -> np.ndarray:
    """Cosine of rows 1.. with row 0, clipped to [0, 1]; a zero vector has cosine 0."""
    norms = np.linalg.norm(vectors, axis=1)
    dots = vectors[1:] @ vectors[0]
    denominators = norms[1:] * norms[0]
    cosine = np.divide(dots, denominators, out=np.zeros_like(dots), where=denominators > 0.0)
    return np.clip(cosine, 0.0, 1.0)
