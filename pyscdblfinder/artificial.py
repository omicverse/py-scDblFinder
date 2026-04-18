"""Artificial-doublet synthesis — port of ``getArtificialDoublets``.

Mirrors the R function's pair-selection logic (random vs cluster-aware)
and the ``createDoublets`` helper's size adjustments (halfSize, resamp,
adjustSize). Exposes a single ``get_artificial_doublets`` function that
returns a dict with ``counts`` (genes × n_doublets) and ``origins``
(labelled "clusterA+clusterB" per doublet, or None when no clusters).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp


def _col_sums(x) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.sum(axis=0)).ravel()
    return np.asarray(x).sum(axis=0)


def _trim_by_lsize(ls: np.ndarray, trim_q: tuple[float, float]) -> np.ndarray:
    """Keep columns whose library size is in the [q_low, q_high] range and > 0."""
    if ls.size == 0:
        return np.array([], dtype=int)
    qlo = float(np.quantile(ls, min(trim_q)))
    qhi = float(np.quantile(ls, max(trim_q)))
    return np.where((ls > 0) & (ls >= qlo) & (ls <= qhi))[0]


def _sample_pairs(n_cells: int, n_doublets: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n_doublets`` cell index pairs (with replacement), removing self-self."""
    if n_cells <= 1:
        raise ValueError("need at least 2 cells to sample doublet pairs")
    # Oversample 2x then drop self-matches so the final count is close to ``n_doublets``
    target = n_doublets * 2
    pairs = rng.integers(0, n_cells, size=(target, 2))
    keep = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[keep]
    if pairs.shape[0] > n_doublets:
        pairs = pairs[:n_doublets]
    while pairs.shape[0] < n_doublets:
        more = rng.integers(0, n_cells, size=(n_doublets - pairs.shape[0], 2))
        more = more[more[:, 0] != more[:, 1]]
        pairs = np.vstack([pairs, more])
    return pairs[:n_doublets]


def _create_doublets(
    counts: np.ndarray | sp.spmatrix,
    pairs: np.ndarray,
    *,
    clusters: Optional[np.ndarray] = None,
    resamp: float = 0.25,
    half_size: float = 0.25,
    adjust_size: float = 0.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Port of R ``createDoublets``.

    - ``adjust_size`` fraction is size-adjusted using cluster median library
      size ratios (requires ``clusters``).
    - ``half_size`` fraction has its library size halved.
    - ``resamp`` fraction is re-sampled from a Poisson around the summed
      counts (the other fraction is just rounded).
    Returns a dense ``(n_genes, n_doublets)`` float array.
    """
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts, dtype=np.float64)
    n_d = pairs.shape[0]

    if adjust_size > 0 and clusters is not None:
        n_adj = int(round(adjust_size * n_d))
        idx_adj = rng.choice(n_d, size=n_adj, replace=False)
    else:
        n_adj = 0
        idx_adj = np.array([], dtype=int)
    idx_nonadj = np.setdiff1d(np.arange(n_d), idx_adj, assume_unique=False)

    # Non-adjusted doublets: simple sum
    x1 = counts[:, pairs[idx_nonadj, 0]] + counts[:, pairs[idx_nonadj, 1]]

    if n_adj > 1 and clusters is not None:
        ls = counts.sum(axis=0)
        # cluster median library sizes
        uniq = np.unique(clusters)
        csz = {u: np.median(ls[clusters == u]) for u in uniq}
        pair_adj = pairs[idx_adj]
        ls1 = ls[pair_adj[:, 0]]
        ls2 = ls[pair_adj[:, 1]]
        ls_ratio = ls1 / (ls1 + ls2 + 1e-12)
        c1 = clusters[pair_adj[:, 0]]
        c2 = clusters[pair_adj[:, 1]]
        csz1 = np.array([csz[c] for c in c1])
        csz2 = np.array([csz[c] for c in c2])
        factor = (ls_ratio + csz1 / (csz1 + csz2 + 1e-12)) / 2
        factor = np.clip(factor, 0.2, 0.8)
        target_ls = ls1 + ls2
        x2 = counts[:, pair_adj[:, 0]] * factor + counts[:, pair_adj[:, 1]] * (1 - factor)
        # Renormalize to the target library size
        cur_ls = x2.sum(axis=0) + 1e-12
        x2 = x2 * (target_ls / cur_ls)[None, :]
        out = np.concatenate([x1, x2], axis=1)
    else:
        out = x1

    # Half-size a fraction
    if half_size > 0 and out.shape[1] > 0:
        n_h = int(np.ceil(half_size * out.shape[1]))
        h_idx = rng.choice(out.shape[1], size=n_h, replace=False)
        out[:, h_idx] = out[:, h_idx] / 2.0

    # Resample (Poisson) a fraction, else round
    if resamp > 0 and out.shape[1] > 0:
        n_r = int(np.ceil(resamp * out.shape[1]))
        r_idx = rng.choice(out.shape[1], size=n_r, replace=False)
        out[:, r_idx] = rng.poisson(np.clip(out[:, r_idx], 0, None)).astype(np.float64)
        # Round the rest
        rest = np.setdiff1d(np.arange(out.shape[1]), r_idx, assume_unique=False)
        out[:, rest] = np.round(out[:, rest])
    else:
        out = np.round(out)

    return out


def get_artificial_doublets(
    counts: np.ndarray | sp.spmatrix,
    n: int = 3000,
    *,
    clusters: Optional[np.ndarray] = None,
    prop_random: float = 0.1,
    resamp: float = 0.25,
    half_size: float = 0.25,
    adjust_size: float = 0.25,
    trim_q: tuple[float, float] = (0.05, 0.95),
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> dict:
    """Generate ``n`` artificial doublets from the ``counts`` matrix.

    Parameters mirror the R ``getArtificialDoublets`` one-to-one. The
    returned dict has two keys:

        "counts"  — dense float ``(n_genes, n_created)`` array (may differ
                    from ``n`` by a few due to self-self filtering).
        "origins" — 1-D array of cluster-pair labels (str) or ``None``.

    ``counts`` is expected in **genes × cells** orientation (Seurat style).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_genes, n_cells = counts.shape
    ls = _col_sums(counts)
    if clusters is None:
        keep = _trim_by_lsize(ls, trim_q)
        clusters_kept = None
    else:
        clusters = np.asarray(clusters)
        keep_masks = []
        for c in np.unique(clusters):
            idx = np.where(clusters == c)[0]
            ls_c = ls[idx]
            if idx.size < 10:
                keep_masks.append(idx[ls_c > 0])
                continue
            qlo, qhi = np.quantile(ls_c, sorted(trim_q))
            keep_masks.append(idx[(ls_c > 0) & (ls_c >= qlo) & (ls_c <= qhi)])
        keep = np.concatenate(keep_masks) if keep_masks else np.array([], dtype=int)
        keep = np.sort(keep)
        clusters_kept = clusters[keep]

    x = counts[:, keep].toarray() if sp.issparse(counts) else np.asarray(counts)[:, keep]
    n_usable = x.shape[1]

    if clusters_kept is None or prop_random >= 1.0:
        pairs = _sample_pairs(n_usable, n, rng)
        ad = _create_doublets(x, pairs, clusters=None,
                              resamp=resamp, half_size=half_size,
                              adjust_size=0.0, rng=rng)
        origins = np.array([None] * ad.shape[1], dtype=object)
        return {"counts": ad, "origins": origins}

    # Mixed: some random, rest cluster-paired
    n_random = int(np.ceil(n * prop_random))
    n_cluster = n - n_random

    out_counts = []
    out_origins = []

    if n_random > 0:
        pairs_r = _sample_pairs(n_usable, n_random, rng)
        ad_r = _create_doublets(x, pairs_r, clusters=clusters_kept,
                                resamp=resamp, half_size=half_size,
                                adjust_size=adjust_size, rng=rng)
        # Origins from the random pairs
        c1 = clusters_kept[pairs_r[:, 0]]
        c2 = clusters_kept[pairs_r[:, 1]]
        oc = np.array([f"{a}+{b}" if a != b else None for a, b in zip(c1, c2)], dtype=object)
        # _create_doublets may return more columns than ad_r has per-pair (due to adjust_size split);
        # pad origins to the actual length
        out_counts.append(ad_r)
        if ad_r.shape[1] == oc.size:
            out_origins.append(oc)
        else:
            out_origins.append(np.array([None] * ad_r.shape[1], dtype=object))

    if n_cluster > 0:
        # inter-cluster pairs, weighted "proportional" to cluster sizes
        cl_counts = np.bincount(np.unique(clusters_kept, return_inverse=True)[1])
        weights = cl_counts / cl_counts.sum()
        uniq = np.unique(clusters_kept)
        # Sample pairs of distinct clusters weighted by proportion^2
        cpairs = []
        for _ in range(n_cluster):
            a, b = rng.choice(uniq.size, size=2, replace=False, p=weights)
            cpairs.append((uniq[a], uniq[b]))
        pair_idx = []
        origins = []
        for ca, cb in cpairs:
            ia = np.random.default_rng(rng.integers(0, 2**31)).choice(
                np.where(clusters_kept == ca)[0]
            )
            ib = np.random.default_rng(rng.integers(0, 2**31)).choice(
                np.where(clusters_kept == cb)[0]
            )
            pair_idx.append([ia, ib])
            origins.append(f"{ca}+{cb}")
        pair_idx = np.asarray(pair_idx)
        ad_c = _create_doublets(x, pair_idx, clusters=clusters_kept,
                                resamp=resamp, half_size=half_size,
                                adjust_size=adjust_size, rng=rng)
        out_counts.append(ad_c)
        out_origins.append(np.asarray(origins, dtype=object) if ad_c.shape[1] == len(origins)
                           else np.array([None] * ad_c.shape[1], dtype=object))

    combined = np.concatenate(out_counts, axis=1) if out_counts else np.zeros((n_genes, 0))
    origins = np.concatenate(out_origins) if out_origins else np.array([], dtype=object)
    return {"counts": combined, "origins": origins}
