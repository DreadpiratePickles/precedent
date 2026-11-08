"""Research: build the receiver-by-source adjacency from a dated graph and propagate a shock.

Convention: `A[receiver, source] = weight`, then every row with a positive sum is divided by
that sum, so each receiver's incoming weights sum to 1 (a receiver with no links keeps an
all-zero row). A shock vector `s` holds one value per node, in the order of `nodes`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class AdjacencyReport:
    links_used: int
    links_dropped_unknown_node: int
    receivers_without_links: int


def build_adjacency(
    graph: pd.DataFrame, nodes: list[str]
) -> tuple[sparse.csr_matrix, AdjacencyReport]:
    """Row-normalised adjacency, rows = receivers, columns = sources, in the order of `nodes`.

    `graph` needs `source_node`, `receiver_node` and `weight`. A link naming a node outside
    `nodes` is dropped and counted. Links for the same (receiver, source) are summed before
    each row is divided by its sum. A non-finite or negative weight raises ValueError.
    """
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate node names in `nodes`")
    size = len(nodes)
    position = {name: index for index, name in enumerate(nodes)}
    weights = pd.to_numeric(graph["weight"], errors="raise").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("link weight must be finite and non-negative")
    sources = graph["source_node"].map(position)
    receivers = graph["receiver_node"].map(position)
    known = (sources.notna() & receivers.notna()).to_numpy()
    raw = sparse.coo_matrix(
        (
            weights[known],
            (receivers[known].to_numpy(dtype=np.int64), sources[known].to_numpy(dtype=np.int64)),
        ),
        shape=(size, size),
    ).tocsr()  # converting from COO sums duplicate (receiver, source) entries
    row_sums = np.asarray(raw.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sums, out=np.zeros(size), where=row_sums > 0)
    adjacency = sparse.diags(inverse).dot(raw).tocsr()
    adjacency.eliminate_zeros()
    report = AdjacencyReport(
        links_used=int(known.sum()),
        links_dropped_unknown_node=int((~known).sum()),
        receivers_without_links=int((row_sums <= 0).sum()),
    )
    return adjacency, report


def propagate_shock(
    adjacency: sparse.csr_matrix, shock: np.ndarray, alpha: float = 0.5, hops: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """The hop-1 and hop-2 terms of `sum_{k=1..hops} alpha^k A^k shock`, as two arrays.

    hop1 = alpha * (A @ shock)
    hop2 = alpha^2 * (A @ (A @ shock) - shock * return_mass), return_mass[i] = sum_j A[i,j] A[j,i]

    The subtracted term is exactly the two-hop paths i -> j -> i that return to their origin;
    paths i -> j -> k with k != i are kept even when k was itself shocked. With `hops == 1`
    the hop-2 term is all zeros. Both results are 1-d float64 arrays of the shock's length.
    """
    if hops not in (1, 2):
        raise ValueError(f"hops must be 1 or 2, got {hops}")
    shock = np.asarray(shock, dtype=np.float64)
    size = adjacency.shape[0]
    if shock.shape != (size,):
        raise ValueError(f"shock length {shock.shape} does not match {size} nodes")
    if not np.all(np.isfinite(shock)):
        raise ValueError("shock must be finite")
    first = np.asarray(adjacency @ shock, dtype=np.float64).ravel()
    hop1 = alpha * first
    if hops == 1:
        return hop1, np.zeros(size, dtype=np.float64)
    return_mass = np.asarray(adjacency.multiply(adjacency.T).sum(axis=1)).ravel()
    second = np.asarray(adjacency @ first, dtype=np.float64).ravel()
    hop2 = alpha**2 * (second - shock * return_mass)
    return hop1, hop2
