"""cxds (coexpression-based doublet scoring) — port of R ``cxds2``.

Implements the binomial-test co-expression score: for each gene pair,
compute the expected vs observed frequency of cells where exactly one of
the pair is expressed, and sum −log(p_binom) contributions per cell. The
result is rescaled to [0, 1].
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom


def cxds_score(
    x,
    which_dbls: Optional[np.ndarray] = None,
    n_top: int = 500,
    bin_thresh: Optional[int] = None,
) -> np.ndarray:
    """Port of R ``cxds2``.

    Parameters
    ----------
    x : (n_genes, n_cells) array or sparse matrix of counts.
    which_dbls : indices of columns to exclude when computing expected
        co-expression (artificial doublets should be excluded; this matches
        the R ``whichDbls`` argument).
    n_top : number of top-variance genes to use.
    bin_thresh : binarization threshold; auto-chosen if ``None``.

    Returns
    -------
    score : 1-D array of length ``n_cells``, rescaled to [0, 1].
    """
    if sp.issparse(x):
        dense = x.toarray().astype(np.float64)
    else:
        dense = np.asarray(x, dtype=np.float64)
    dense = np.nan_to_num(dense, nan=0.0)

    n_genes, n_cells = dense.shape

    # Auto bin threshold
    if bin_thresh is None:
        p_nz = (dense > 0).sum() / dense.size
        if p_nz > 0.5:
            # Filter to the sparsest n_top genes
            p_per_gene = (dense > 0).sum(axis=1) / n_cells
            sel = np.argsort(p_per_gene)[:n_top]
            dense = dense[sel]
            bin_thresh = max(1.0, float(np.quantile(dense[dense > 0], 0.5 * p_per_gene[sel].mean())))
        else:
            bin_thresh = 1.0

    B = (dense >= bin_thresh).astype(np.float64)
    ps = B.mean(axis=1)  # per-gene prob of being expressed

    if B.shape[0] > n_top:
        hvg = np.argsort(-(ps * (1 - ps)))[:n_top]
        B = B[hvg]
        ps = ps[hvg]

    Bp = B.copy()
    if which_dbls is not None and len(which_dbls) > 0:
        mask = np.ones(Bp.shape[1], dtype=bool)
        mask[np.asarray(which_dbls, dtype=int)] = False
        Bp = Bp[:, mask]
    n_eff = Bp.shape[1]
    if n_eff < 2:
        return np.zeros(n_cells, dtype=np.float64)

    # Expected co-expression probabilities: p_i * (1 - p_j) + p_j * (1 - p_i)
    prb = np.outer(ps, 1 - ps)
    prb = prb + prb.T

    # Observed exclusive-expression counts per gene pair
    notBp = 1.0 - Bp
    obs = Bp @ notBp.T
    obs = obs + obs.T

    # p(>= obs | Binomial(n_eff, prb))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = binom.logsf(np.asarray(obs) - 1, n_eff, prb)
    S = np.where(np.isfinite(S), S, 0.0)
    if np.all(S == 0):
        return np.zeros(n_cells, dtype=np.float64)
    # Per-cell score = -sum over gene-pairs of B_i * S_ij * B_j
    s = -(B * (S @ B)).sum(axis=0)
    s = s - s.min()
    m = s.max()
    if m > 0:
        s = s / m
    return s
