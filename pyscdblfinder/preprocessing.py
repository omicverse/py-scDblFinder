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
    backend: str = "auto",
) -> np.ndarray:
    """Return a ``(n_cells, dims)`` PCA embedding of the merged matrix.

    Mirrors R's ``.defaultProcessing`` (scater's ``runPCA`` on log-normalized
    counts with centering).

    backend
        ``'auto'`` (default) uses scikit-learn's **randomized** truncated SVD —
        many times faster than the full solver on large cell numbers with only
        ~20 PCs, and numerically equivalent for a leading low-rank embedding.
        ``'torch'`` runs the SVD on GPU via ``torch.pca_lowrank`` (falls back to
        CPU torch if CUDA is unavailable); ``'sklearn-full'`` forces the exact
        full solver.
    """
    if do_norm:
        norm = log_normalize(counts)
    else:
        norm = counts.toarray() if sp.issparse(counts) else np.asarray(counts, dtype=np.float64)

    # PCA on cells x genes (transpose)
    X = np.asarray(norm.T, dtype=np.float32)
    n_pcs = min(int(dims), min(X.shape) - 1)

    if backend == "torch":
        emb = _torch_pca(X, n_pcs, random_state)
        if emb is not None:
            return emb
        backend = "auto"

    from sklearn.decomposition import PCA
    solver = "full" if backend == "sklearn-full" else "randomized"
    pca = PCA(n_components=n_pcs, svd_solver=solver, random_state=random_state)
    return pca.fit_transform(X)


def _torch_pca(X, n_pcs, random_state):
    """GPU/torch PCA via ``torch.pca_lowrank``. Returns None if torch missing."""
    try:
        import torch
    except Exception:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(random_state))
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    Xt = Xt - Xt.mean(dim=0, keepdim=True)                     # center like PCA
    q = min(n_pcs + 7, min(Xt.shape))
    U, S, V = torch.pca_lowrank(Xt, q=q, center=False, niter=7)
    emb = (Xt @ V[:, :n_pcs])
    return emb.cpu().numpy().astype(np.float64)


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
