"""kNN-based feature extraction — port of ``.evaluateKNN``.

For each cell, find the k nearest neighbors in the PCA space of the
merged (real + artificial) matrix, then compute the features scDblFinder
feeds into its xgboost classifier:

    weighted                    distance-weighted fraction of artificial neighbors
    distanceToNearest           distance to nearest neighbor (any type)
    distanceToNearestDoublet    nearest artificial-doublet distance
    distanceToNearestReal       nearest real-cell distance
    nearestClass                type of nearest neighbor (0=real, 1=artificial)
    ratio.k{k}                  fraction of artificial neighbors at each k
    mostLikelyOrigin            (if origins given) modal origin of artificial neighbors
    difficulty, expected, observed  (when origins + expected given)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def default_k_grid(n_cells: int, k: Optional[list] = None) -> list:
    """R port of ``.defaultKnnKs`` — returns a small list of k values."""
    if k is not None:
        if np.isscalar(k):
            return [int(k)]
        return [int(x) for x in k]
    # scDblFinder default: three k values log-spaced
    return sorted(set([max(2, int(round(np.sqrt(n_cells) / 2))),
                       max(5, int(round(np.sqrt(n_cells)))),
                       max(10, int(round(1.5 * np.sqrt(n_cells))))]))


def evaluate_knn(
    pca: np.ndarray,
    ctype: np.ndarray,
    *,
    k: list[int] | int,
    origins: Optional[np.ndarray] = None,
    expected: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute per-cell kNN features.

    Parameters
    ----------
    pca : (n_cells, n_pcs) PCA embedding of the merged real+artificial matrix.
    ctype : 1-D array of ints — 0 for real, 1 for artificial-doublet.
    k : int or list of ints — neighborhood sizes. Max value determines how
        many neighbors are computed.
    origins : 1-D array of str/None — cluster-pair labels of each cell
        (None for real cells).
    expected : dict mapping origin-label → expected count of doublets of
        that origin (used to compute the ``difficulty`` feature).

    Returns
    -------
    pandas DataFrame with one row per cell and the features listed above.
    """
    pca = np.asarray(pca, dtype=np.float64)
    ctype = np.asarray(ctype, dtype=np.int32)
    if np.isscalar(k):
        k_list = [int(k)]
    else:
        k_list = sorted(int(x) for x in k)
    max_k = max(k_list)

    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", metric="euclidean")
    nn.fit(pca)
    dist, idx = nn.kneighbors(pca, return_distance=True)
    # Drop self (column 0)
    dist = dist[:, 1:]
    idx = idx[:, 1:]

    n_cells = pca.shape[0]
    # Replace distance-zero rows (identical cells) with the smallest positive nearest-neighbor distance
    zero_mask = dist[:, 0] == 0
    if zero_mask.any():
        positive = dist[~zero_mask, 0]
        if positive.size:
            fill = positive.min()
            dist[zero_mask, 0] = fill

    neigh_type = ctype[idx]

    md = dist[:, 0].max()
    # distanceToNearestDoublet / Real
    d_dbl = np.full(n_cells, 2 * md, dtype=np.float64)
    d_real = np.full(n_cells, 2 * md, dtype=np.float64)
    for i in range(n_cells):
        dbl_mask = neigh_type[i] == 1
        real_mask = neigh_type[i] == 0
        if dbl_mask.any():
            d_dbl[i] = dist[i, np.argmax(dbl_mask)]
        if real_mask.any():
            d_real[i] = dist[i, np.argmax(real_mask)]

    # Distance-weighted artificial fraction (uses all k_max neighbors)
    w = np.sqrt(max_k - np.arange(max_k)) / dist
    w = w / w.sum(axis=1, keepdims=True)
    weighted = (neigh_type * w).sum(axis=1)

    out = pd.DataFrame({
        "weighted": weighted,
        "distanceToNearest": dist[:, 0],
        "distanceToNearestDoublet": d_dbl,
        "distanceToNearestReal": d_real,
        "nearestClass": neigh_type[:, 0].astype(np.int32),
    })
    for ki in k_list:
        out[f"ratio.k{ki}"] = neigh_type[:, :ki].mean(axis=1)

    if origins is not None:
        origins = np.asarray(origins, dtype=object)
        most_likely, ambig = _most_likely_origins(idx, neigh_type, origins)
        out["mostLikelyOrigin"] = most_likely
        out["originAmbiguous"] = ambig
        if expected is not None:
            exp_counts = np.array([expected.get(mo, 0) if mo is not None else 0
                                   for mo in most_likely], dtype=np.float64)
            out["expected"] = exp_counts
            # observed = how many total have the same origin
            ob = pd.Series(most_likely).value_counts()
            out["observed"] = np.array([ob.get(mo, 0) if mo is not None else 0
                                        for mo in most_likely], dtype=np.float64)

    return out


def _most_likely_origins(idx, neigh_type, origins):
    """Port of ``.getMostLikelyOrigins`` — modal origin among artificial neighbors."""
    n = idx.shape[0]
    most = np.array([None] * n, dtype=object)
    ambig = np.zeros(n, dtype=bool)
    for i in range(n):
        art_mask = neigh_type[i] == 1
        if not art_mask.any():
            continue
        neigh_origins = origins[idx[i][art_mask]]
        neigh_origins = neigh_origins[neigh_origins != None]  # noqa: E711
        if neigh_origins.size == 0:
            continue
        uniq, counts = np.unique(neigh_origins, return_counts=True)
        top_count = counts.max()
        top = uniq[counts == top_count]
        most[i] = top[0]
        ambig[i] = top.size > 1
    return most, ambig
