"""Peer links from the text of 10-K business sections (graph G3, text similarity).

Companies whose Item 1 text embeds close together are peers. Each link is dated: it starts when
the later of the two filings was accepted, ends when a later filer displaces it from the
receiver's top-k or after a fixed number of months, so a caller reads it only through
`graph_asof` and the graph at any instant uses only filings accepted by then. The embedding
model itself is injected through `TextEmbedder`; nothing here downloads a model or fetches a filing.

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


def _top_sources(
    similarity: np.ndarray, receiver: int, pool: np.ndarray, top_k: int
) -> list[int]:
    """The receiver's `top_k` most similar sources in `pool` (positive similarity only).

    Ordered by descending similarity, ties broken by position (positions follow sorted names).
    """
    candidates = pool[(pool != receiver) & (similarity[receiver, pool] > 0.0)]
    order = np.lexsort((candidates, -similarity[receiver, candidates]))
    return [int(position) for position in candidates[order[:top_k]]]


def peer_links_from_embeddings(
    embeddings: Mapping[str, np.ndarray],
    accepted_utc: Mapping[str, pd.Timestamp],
    top_k: int = 10,
    expiry_months: int = 15,
) -> list[Link]:
    """`peer_of` links from cosine similarity of business-section embeddings, point in time.

    The links are built as events in acceptance order, so the graph visible at any instant uses
    only filings accepted by then. At each distinct acceptance instant the pool is every filing
    accepted at or before it; every receiver R in the pool whose candidates changed (R itself
    just filed, or a new filer has positive similarity to R) has its `top_k` most similar other
    companies S (positive cosine similarity) recomputed over the pool. `Link(source=S,
    receiver=R)` is valid from the instant it enters R's top-k, which is the later of the two
    filings' acceptance times (a later filer can only push a link out of the top-k, never in),
    until the earlier of: S is displaced from R's top-k by a later filer, and `expiry_months`
    after the link started. A link never restarts after it ends. A shock at a peer reaches the
    receiver, and the pair (A, B) can appear in both directions. Filings accepted at the same
    instant enter together and do not displace one another.

    The weight is the cosine similarity (at most 1.0); the factor stage normalises each
    receiver's incoming weights. Its `source_doc_id` is `10k-similarity:<source>|<receiver>`:
    the inputs carry no accession number, so a document is identified by its company. Ties are
    broken by source name; output is sorted by (receiver, valid_from, descending weight, source)
    and is independent of input order. The defaults match `correlation_top_k` and
    `annual_link_expiry_months` in `references/ripple_params.json` (the key shared with G2's
    correlation graph; G3 has no key of its own). Reading the links with `graph_asof(t)` equals
    building from only the filings accepted by t and reading at t.
    Raises ValueError naming the offending company for a missing, non-finite, zero-norm,
    wrong-length or non-1-D embedding, a missing or naive timestamp, or fewer than two companies.
    """
    names, matrix, stamps = _validated(embeddings, accepted_utc, top_k, expiry_months)
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = np.round(unit @ unit.T, _SIMILARITY_DECIMALS)
    stamp_of = [stamps[name] for name in names]
    in_pool = np.zeros(len(names), dtype=bool)
    # active[receiver][source] = the instant the link started
    active: dict[int, dict[int, pd.Timestamp]] = {}
    ended: list[tuple[int, int, pd.Timestamp, pd.Timestamp]] = []  # receiver, source, from, to

    def end(receiver: int, source: int, started: pd.Timestamp, instant: pd.Timestamp) -> None:
        expires = started + pd.DateOffset(months=expiry_months)
        ended.append((receiver, source, started, min(instant, expires)))

    for instant in sorted(set(stamp_of)):
        arrivals = np.array([i for i, stamp in enumerate(stamp_of) if stamp == instant])
        in_pool[arrivals] = True
        pool = np.flatnonzero(in_pool)
        for receiver in pool:
            receiver = int(receiver)
            if not (receiver in arrivals or (similarity[receiver, arrivals] > 0.0).any()):
                continue  # no new candidate for this receiver: its top-k cannot change
            top = _top_sources(similarity, receiver, pool, top_k)
            current = active.setdefault(receiver, {})
            for source in [s for s in current if s not in top]:
                end(receiver, source, current.pop(source), instant)
            for source in top:
                current.setdefault(source, instant)
    for receiver, current in active.items():
        for source, started in current.items():
            end(receiver, source, started, started + pd.DateOffset(months=expiry_months))
    links = [
        Link(
            source_node=names[source],
            receiver_node=names[receiver],
            relation=Relation.PEER_OF,
            weight=min(1.0, float(similarity[receiver, source])),
            valid_from_utc=started.to_pydatetime(),
            valid_to_utc=valid_to.to_pydatetime(),
            source_doc_id=f"10k-similarity:{names[source]}|{names[receiver]}",
            evidence_span=None,
            extractor_version=PEER_EXTRACTOR_VERSION,
        )
        for receiver, source, started, valid_to in ended
    ]
    links.sort(
        key=lambda link: (
            link.receiver_node,
            link.valid_from_utc,
            -link.weight,
            link.source_node,
        )
    )
    return links
