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
    do_norm: bool = True,
    random_state: int = 0,
) -> np.ndarray:
    """Return a ``(n_cells, dims)`` PCA embedding of the merged matrix.

    Mirrors R's ``.defaultProcessing`` (scater's ``runPCA`` on log-normalized
    counts with centering; we rely on scikit-learn's PCA for the SVD step).
    """
    from sklearn.decomposition import PCA

    if do_norm:
        norm = log_normalize(counts)
    else:
        norm = counts.toarray() if sp.issparse(counts) else np.asarray(counts, dtype=np.float64)

    # PCA on cells x genes (transpose)
    X = norm.T
    n_pcs = min(int(dims), min(X.shape) - 1)
    pca = PCA(n_components=n_pcs, random_state=random_state)
    return pca.fit_transform(X)


def select_features(
    counts,
    clusters: Optional[np.ndarray] = None,
    n_features: int = 1352,
) -> np.ndarray:
    """Simple variance-based feature selection (genes × cells).

    The R version has a marker-aware variant; we ship only the purely
    variance-driven path here, which matches the ``propMarkers=0`` branch
    used by default for the scRNA-seq path.
    """
    if sp.issparse(counts):
        mean = np.asarray(counts.mean(axis=1)).ravel()
        mean2 = np.asarray(counts.multiply(counts).mean(axis=1)).ravel()
        var = mean2 - mean ** 2
    else:
        counts = np.asarray(counts, dtype=np.float64)
        mean = counts.mean(axis=1)
        var = counts.var(axis=1, ddof=0)
    n_features = min(int(n_features), counts.shape[0])
    return np.argsort(-var)[:n_features]
