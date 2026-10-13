"""scATAC-seq processing for scDblFinder — port of ``R/atac_processing.R``.

With default parameters scDblFinder does poorly on scATAC because reads are
spread over hundreds of thousands of peaks or tiles. The R package therefore
*aggregates* correlated features into a small number of meta-features before
creating artificial doublets (``aggregateFeatures=TRUE``), and offers ATAC
specific processing of the merged matrix (``processing="normFeatures"`` or
``"atac"``). The scDblFinder ATAC vignette recommends::

    scDblFinder(sce, aggregateFeatures=TRUE, nfeatures=25,
                processing="normFeatures")

which in Python is::

    sc_dbl_finder(counts, aggregate_features=True, n_features=25,
                  processing="normFeatures")

Faithfulness notes (R behaviour that is kept on purpose):

* ``aggregateFeatures`` keeps the ``use.subset`` cells with the largest
  library sizes. R computes ``min(mean(use.subset, ncol(x)), 2*use.subset)``,
  and ``mean(a, b)`` treats ``b`` as the trim argument, so this is always
  ``use.subset`` cells; the port does the same.
* Within ``scDblFinder``, a single numeric ``aggregateFeatures`` value is
  ignored by R (it is replaced by ``seq_len(dims)[-1]``); a vector of
  dimensions is honoured. The port does the same.
* Feature clustering uses k-means with random initial rows, like R's
  ``kmeans``. R uses the Hartigan-Wong algorithm and its own RNG, so
  the clusters themselves are not reproducible across languages; results
  are compared at the level of doublet scores and calls.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# normalisations
# ---------------------------------------------------------------------------
def tfidf(x, sf: float = 10000.0):
    """R ``TFIDF``: ``log1p(sf * idf * tf)`` on a features x cells matrix
    (Stuart & Butler et al. 2019). Returns a sparse CSR matrix with NaN
    (all-zero features or cells) set to 0.
    """
    x = sp.csr_matrix(x, dtype=np.float64)
    cs = np.asarray(x.sum(axis=0)).ravel()
    rs = np.asarray(x.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_cs = 1.0 / cs
        idf = x.shape[1] / rs
    out = sp.diags(idf) @ x @ sp.diags(inv_cs)
    out = out.tocsr()
    out.data = np.log1p(sf * out.data)
    out.data[~np.isfinite(out.data)] = 0.0
    out.eliminate_zeros()
    return out


def log_norm_counts(x, pseudo_count: float = 1.0) -> np.ndarray:
    """scuttle ``logNormCounts`` with library-size factors: size factors are
    the column sums divided by their mean, and the result is
    ``log2(x / sf + pseudo_count)`` (features x cells, dense).
    """
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    lib = x.sum(axis=0)
    sf = lib / lib.mean()
    if np.any(sf <= 0):
        raise ValueError("size factors should be positive (empty cells)")
    return np.log2(x / sf[np.newaxis, :] + pseudo_count)


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------
def _uncentered_pca_scores(x, rank: int, random_state: int = 0) -> np.ndarray:
    """Row scores ``U * d`` of the rank-``rank`` truncated SVD of ``x``,
    without centring — the ``$x`` of ``BiocSingular::runPCA(x, center=FALSE,
    BSPARAM=IrlbaParam())``. Rows of ``x`` are the observations.
    """
    from scipy.sparse.linalg import svds
    rank = int(min(rank, min(x.shape) - 1))
    rng = np.random.default_rng(random_state)
    v0 = rng.standard_normal(min(x.shape))
    xs = sp.csr_matrix(x, dtype=np.float64) if sp.issparse(x) else \
        np.asarray(x, dtype=np.float64)
    u, s, vt = svds(xs, k=rank, v0=v0, which="LM")
    order = np.argsort(-s)
    u, s = u[:, order], s[order]
    # sign convention: largest-|loading| element of each component positive
    vt = vt[order]
    flip = np.sign(vt[np.arange(rank), np.abs(vt).argmax(axis=1)])
    flip[flip == 0] = 1
    return u * (s * flip)[np.newaxis, :]


def atac_processing(e, dims: int = 20, random_state: int = 0) -> np.ndarray:
    """``processing="atac"``: TF-IDF followed by an uncentred truncated SVD,
    returning a cells x ``dims`` embedding of the merged (real + artificial)
    features x cells matrix ``e``.

    R's ``.atacProcessing`` passes the features x cells TF-IDF matrix straight
    to ``runPCA``, which treats rows as observations, so its embedding has
    one row per *feature* and ``scDblFinder(..., processing="atac")`` fails in
    R (see the parity notes in the README). The port embeds the cells, which
    is the evident intent; the vignette's recommended setting is
    ``processing="normFeatures"``.
    """
    return _uncentered_pca_scores(tfidf(e).T.tocsr(), rank=dims,
                                  random_state=random_state)[:, :dims]


# ---------------------------------------------------------------------------
# feature aggregation
# ---------------------------------------------------------------------------
def _kmeans(X: np.ndarray, k: int, num_init: int, use_mbk: bool,
            random_state: int) -> np.ndarray:
    if use_mbk:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, n_init=num_init,
                             random_state=random_state)
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=num_init, max_iter=100, init="random",
                    random_state=random_state)
    return km.fit_predict(X) + 1      # R cluster ids are 1-based


def _cluster_features_step(x, k: int, dims_use: Sequence[int],
                           use_mbk: Optional[bool], num_init: int,
                           random_state: int) -> np.ndarray:
    dims_use = np.asarray(list(dims_use), dtype=int)
    pca = _uncentered_pca_scores(x, rank=int(dims_use.max()),
                                 random_state=random_state)
    pca = pca[:, dims_use - 1]
    if use_mbk is None:
        # R: mini-batch only if > 30,000 features *and* the optional mbkmeans
        # package is installed; it is not by default, so R uses kmeans
        use_mbk = False
    return _kmeans(pca, k, num_init, use_mbk, random_state)


def _sum_rows_by_group(m, groups: np.ndarray) -> np.ndarray:
    """Sum the rows of ``m`` within each group; groups in sorted order."""
    levels, inv = np.unique(groups, return_inverse=True)
    agg = sp.csr_matrix((np.ones(groups.size), (inv, np.arange(groups.size))),
                        shape=(levels.size, groups.size))
    out = agg @ (m if sp.issparse(m) else np.asarray(m))
    return out.toarray() if sp.issparse(out) else np.asarray(out)


def aggregate_features(
    x,
    dims_use: Sequence[int] = range(2, 13),
    k: int = 1000,
    num_init: int = 3,
    use_mbk: Optional[bool] = None,
    use_subset: int = 20000,
    min_count: int = 1,
    norm_fn: Callable = tfidf,
    two_pass: bool = False,
    random_state: int = 0,
    return_groups: bool = False,
):
    """Port of R ``aggregateFeatures``: aggregate similar features (rows).

    Parameters
    ----------
    x
        Features x cells count matrix (sparse or dense).
    dims_use
        1-based SVD components used to cluster the features (R default 2..12).
    k
        Approximate number of meta-features.
    num_init
        k-means restarts.
    use_mbk
        Use mini-batch k-means. ``None`` (default) uses full k-means, which is
        what R does unless the optional ``mbkmeans`` package is installed (R
        then switches to mini-batch above 30,000 features). Pass ``True`` for
        very large feature sets.
    use_subset
        Maximum number of cells used to cluster the features (the ones with
        the largest library sizes, see the module notes).
    min_count
        Minimum feature count (in the cell subset) for a feature to be kept.
    norm_fn
        Normalisation applied before the SVD (TF-IDF by default).
    two_pass
        Repeat the clustering on cell-aggregated profiles (R ``twoPass``).

    Returns
    -------
    The ``k_eff`` x cells aggregated counts (dense ``float64``), and with
    ``return_groups=True`` also the per-kept-feature group labels and the
    indices of the kept features.
    """
    xo = sp.csc_matrix(x) if sp.issparse(x) else np.asarray(x)
    xs = xo
    if xo.shape[1] > use_subset:
        cs = np.asarray(xo.sum(axis=0)).ravel()
        order = np.argsort(-cs, kind="stable")
        xs = xo[:, np.sort(order[: int(min(use_subset, 2 * use_subset))])]
    rs = np.asarray(xs.sum(axis=1)).ravel()
    kept = np.flatnonzero(rs >= min_count)
    xo = xo[kept]
    xs = xs[kept]
    k = int(min(k, xs.shape[0]))
    xn = norm_fn(xs)
    fc = _cluster_features_step(xn, k=k, dims_use=dims_use, use_mbk=use_mbk,
                                num_init=num_init, random_state=random_state)
    if two_pass and use_subset < xo.shape[1]:
        y = log_norm_counts(_sum_rows_by_group(xo, fc)).T
        y = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
        y = np.nan_to_num(y)
        from sklearn.cluster import KMeans
        ncl = int(min(1000, np.ceil(use_subset / 2)))
        cellclust = KMeans(n_clusters=ncl, n_init=num_init, max_iter=100,
                           init="random", random_state=random_state).fit_predict(y)
        xc = _sum_rows_by_group(xo.T, cellclust).T
        xn = norm_fn(xc)
        fc = _cluster_features_step(xn, k=k, dims_use=range(1, max(dims_use) + 1),
                                    use_mbk=use_mbk, num_init=num_init,
                                    random_state=random_state)
    agg = _sum_rows_by_group(xo, fc)
    if return_groups:
        return agg, fc, kept
    return agg
