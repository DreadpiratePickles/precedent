"""Peer links from the text of 10-K business sections (graph G3, text similarity).

Companies whose Item 1 text embeds close together are peers. Each link is dated: it starts when
the later of the two filings was accepted and expires after a fixed number of months, so a
caller reads it only through `graph_asof`. The embedding model itself is injected through
`TextEmbedder`; nothing here downloads a model or fetches a filing.

Core dependencies only (numpy, pandas).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from precedent.ripple.link_store import Link, Relation

PEER_EXTRACTOR_VERSION = "rules-10k-similarity-v1"
_SIMILARITY_DECIMALS = 12  # equal vectors give equal weights, so ties break by name alone
_MIN_COMPANIES = 2


class TextEmbedder(Protocol):
    """Turns texts into vectors. Duplicates Precedent Task 18's TextEmbedder; consolidate when
    that lands."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """An array of shape (len(texts), d), one row per text, in order."""
        ...


def embed_business_sections(
    sections: Mapping[str, str], embedder: TextEmbedder
) -> dict[str, np.ndarray]:
    """One batched embedder call over the sections in sorted name order; name -> vector."""
    if not sections:
        raise ValueError("no sections to embed")
    names = sorted(sections)
    for name in names:
        if not sections[name].strip():
            raise ValueError(f"business section for {name!r} is empty")
    matrix = np.asarray(embedder.embed([sections[name] for name in names]), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(names):
        raise ValueError(
            f"embedder returned shape {matrix.shape}; expected {len(names)} rows of vectors"
        )
    return {name: matrix[position] for position, name in enumerate(names)}


def _validated(
    embeddings: Mapping[str, np.ndarray],
    accepted_utc: Mapping[str, pd.Timestamp],
    top_k: int,
    expiry_months: int,
) -> tuple[list[str], np.ndarray, dict[str, pd.Timestamp]]:
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if expiry_months < 1:
        raise ValueError(f"expiry_months must be at least 1, got {expiry_months}")
    only_embedded = sorted(set(embeddings) - set(accepted_utc))
    only_dated = sorted(set(accepted_utc) - set(embeddings))
    if only_embedded or only_dated:
        raise ValueError(
            "embeddings and accepted_utc must name the same companies; "
            f"no acceptance time for {only_embedded}, no embedding for {only_dated}"
        )
    names = sorted(embeddings)
    if len(names) < _MIN_COMPANIES:
        raise ValueError(f"at least two companies are needed, got {len(names)}")
    stamps: dict[str, pd.Timestamp] = {}
    vectors: list[np.ndarray] = []
    for name in names:
        vector = np.asarray(embeddings[name], dtype=float)
        if vector.ndim != 1:
            raise ValueError(f"embedding for {name!r} must be 1-D, got shape {vector.shape}")
        if vectors and vector.shape != vectors[0].shape:
            raise ValueError(
                f"embedding for {name!r} has length {vector.shape[0]}, "
                f"expected {vectors[0].shape[0]}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"embedding for {name!r} has a non-finite value")
        if not np.linalg.norm(vector) > 0.0:
            raise ValueError(f"embedding for {name!r} has zero norm")
        vectors.append(vector)
        stamp = pd.Timestamp(accepted_utc[name])
        if stamp.tzinfo is None:
            raise ValueError(f"acceptance time for {name!r} must be tz-aware")
        stamps[name] = stamp.tz_convert("UTC")
    return names, np.vstack(vectors), stamps


def peer_links_from_embeddings(
    embeddings: Mapping[str, np.ndarray],
    accepted_utc: Mapping[str, pd.Timestamp],
    top_k: int = 10,
    expiry_months: int = 15,
) -> list[Link]:
    """`peer_of` links from cosine similarity of business-section embeddings.

    For each company R (the receiver), its `top_k` most similar other companies S (the sources)
    with positive similarity become `Link(source=S, receiver=R)`; a shock at a peer reaches the
    receiver. The pair (A, B) can therefore appear in both directions. The weight is the cosine
    similarity (at most 1.0); the factor stage normalises each receiver's incoming weights.

    A link starts at the later of the two filings' acceptance times and expires `expiry_months`
    later. Its `source_doc_id` is `10k-similarity:<source>|<receiver>`: the inputs carry no
    accession number, so a document is identified by its company. Ties are broken by source name;
    output is sorted by (receiver, descending weight, source) and is independent of input order.
    The defaults match `top_k` and `annual_link_expiry_months` in `references/ripple_params.json`.
    Raises ValueError naming the offending company for a missing, non-finite, zero-norm,
    wrong-length or non-1-D embedding, a missing or naive timestamp, or fewer than two companies.
    """
    names, matrix, stamps = _validated(embeddings, accepted_utc, top_k, expiry_months)
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = np.round(unit @ unit.T, _SIMILARITY_DECIMALS)
    links: list[Link] = []
    for receiver_position, receiver in enumerate(names):
        candidates = [
            (-float(similarity[receiver_position, source_position]), source)
            for source_position, source in enumerate(names)
            if source_position != receiver_position
            and similarity[receiver_position, source_position] > 0.0
        ]
        for negative_weight, source in sorted(candidates)[:top_k]:
            valid_from = max(stamps[source], stamps[receiver])
            valid_to = valid_from + pd.DateOffset(months=expiry_months)
            links.append(
                Link(
                    source_node=source,
                    receiver_node=receiver,
                    relation=Relation.PEER_OF,
                    weight=min(1.0, -negative_weight),
                    valid_from_utc=valid_from.to_pydatetime(),
                    valid_to_utc=valid_to.to_pydatetime(),
                    source_doc_id=f"10k-similarity:{source}|{receiver}",
                    evidence_span=None,
                    extractor_version=PEER_EXTRACTOR_VERSION,
                )
            )
    return links
