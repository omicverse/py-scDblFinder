"""PCA preprocessing for the merged real+artificial count matrix.

Corresponds to R's ``.defaultProcessing`` — log-normalize counts, select
top-variance genes if needed, then run PCA. Used both as the default
``processing='default'`` option and as a sanity layer before kNN.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp


def log_normalize(counts, scale_factor: float = 1e4) -> np.ndarray:
    """Per-cell log-normalization: ``log1p(counts / colSums * scale_factor)``.

    Expects counts in **genes × cells** orientation.
    """
    if sp.issparse(counts):
        lib = np.asarray(counts.sum(axis=0)).ravel()
        lib[lib == 0] = 1
        out = counts.multiply(1.0 / lib).multiply(scale_factor).log1p()
        return out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(axis=0)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib[np.newaxis, :] * scale_factor)


def default_processing(
    counts,
    *,
    dims: int = 20,
    do_norm: Optional[bool] = None,
    random_state: int = 0,
    backend: str = "auto",
) -> np.ndarray:
    """R ``.defaultProcessing``: ``scuttle::normalizeCounts`` (log2 of counts over centred
    library-size factors, pseudo-count 1; skipped above 50,000 cells, as in R) and a centred
    PCA of all features.

    backend
        ``'auto'`` (default) computes the leading principal components exactly (float64
        SVD); R uses irlba, an iterative solver for the same components. ``'torch'`` runs a
        low-rank SVD on the GPU via ``torch.pca_lowrank`` when available.
    """
    from .atac import log_norm_counts
    if do_norm is None:
        do_norm = counts.shape[1] <= 50000
    norm = log_norm_counts(counts) if do_norm else (
        counts.toarray() if sp.issparse(counts) else np.asarray(counts, dtype=np.float64))
    X = np.asarray(norm.T, dtype=np.float64)                 # cells x genes
    n_pcs = min(int(dims), min(X.shape) - 1)
    if backend == "torch":
        emb = _torch_pca(X.astype(np.float32), n_pcs, random_state)
        if emb is not None:
            return np.asarray(emb, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] <= X.shape[0]:
        # eigen-decomposition of the genes x genes covariance: exact and cheap for ~1,000 genes
        evals, evecs = np.linalg.eigh(X.T @ X)
        top = np.argsort(evals)[::-1][:n_pcs]
        return X @ evecs[:, top]
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_pcs] * S[:n_pcs]


def select_features(
    counts,
    clusters: Optional[np.ndarray] = None,
    n_features: int = 1352,
) -> np.ndarray:
    """R ``selFeatures`` with ``propMarkers = 0``.

    Without clusters: the ``n_features`` genes with the highest mean count (R
    ``order(rowMeans(counts), decreasing = TRUE)``; ties keep their row order).
    With clusters: per cluster, genes ordered by summed counts, interleaved
    across clusters and de-duplicated, as R does.
    """
    n_genes = counts.shape[0]
    n_features = min(int(n_features), n_genes)
    if n_genes <= n_features:
        return np.arange(n_genes)
    if clusters is None:
        sums = np.asarray(counts.sum(axis=1)).ravel()     # exact; float means break R's tie order
        return np.argsort(-sums, kind="stable")[:n_features]
    clusters = np.asarray(clusters)
    lv = np.unique(clusters)
    X = counts.tocsc() if sp.issparse(counts) else np.asarray(counts)
    sums = np.column_stack([np.asarray(X[:, clusters == c].sum(axis=1)).ravel() for c in lv])
    tops = np.argsort(-sums, axis=0, kind="stable")[:n_features]          # n_features x n_clusters
    _, first = np.unique(tops.ravel(), return_index=True)                   # row-major = R's t(apply(...))
    order = tops.ravel()[np.sort(first)]
    return order[:n_features]
