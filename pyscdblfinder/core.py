"""Top-level scDblFinder pipeline — port of R ``scDblFinder`` (single-sample).

Follows the R function's single-sample path (``scDblFinder.R`` lines ~330-520)
end-to-end: feature selection → optional clustering → artificial-doublet
synthesis → merged-matrix feature extraction → kNN → xgboost scoring →
thresholding. Multi-sample (``samples=``), ATAC ``aggregateFeatures``,
``knownDoublets``, and ``clustCor`` paths are not yet ported — follow-up
work flagged in the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .artificial import get_artificial_doublets
from .classifier import scDbl_score
from .cxds import cxds_score
from .knn_features import default_k_grid, evaluate_knn
from .preprocessing import default_processing, select_features
from .thresholding import doublet_thresholding


@dataclass
class ScDblFinderResult:
    """Output of :func:`sc_dbl_finder`. Mirrors R's ``returnType='table'``.

    Attributes
    ----------
    table : per-cell DataFrame with everything scDblFinder computes —
        ``type`` (real/doublet), ``src`` (real/artificial), ``score``,
        ``class``, kNN features, cxds_score, library size, etc.
    score_threshold : the threshold used to produce ``class``.
    n_real : number of real cells (always the *first* rows of ``table``).
    """

    table: pd.DataFrame
    score_threshold: float
    n_real: int

    def real_cells(self) -> pd.DataFrame:
        """Rows of ``table`` corresponding to real cells only (not artificial)."""
        return self.table.iloc[: self.n_real]


def sc_dbl_finder(
    counts,
    *,
    clusters: Optional[np.ndarray] = None,
    artificial_doublets: Optional[int] = None,
    dbr: Optional[float] = None,
    dbr_sd: Optional[float] = None,
    dbr_per1k: float = 0.008,
    n_features: int = 1352,
    dims: int = 20,
    k: Optional[int | list[int]] = None,
    prop_random: float = 0.0,
    include_pcs: int = 19,
    metric: str = "logloss",
    nrounds: float | int = 0.25,
    max_depth: int = 4,
    iter: int = 3,
    unident_th: Optional[float] = None,
    threshold: bool = True,
    random_state: int = 0,
    knn_fn=None,
    knn_backend: str = "auto",
    pca_backend: str = "auto",
    verbose: bool = False,
) -> ScDblFinderResult:
    """Python port of R ``scDblFinder`` (single-sample path).

    Parameters largely match the R signature. ``counts`` must be a
    **genes × cells** matrix (dense ndarray or scipy sparse).
    """
    counts = counts if sp.issparse(counts) else np.asarray(counts, dtype=np.float64)
    n_genes, n_real = counts.shape

    if unident_th is None:
        unident_th = 0.2 if clusters is None else 0.0

    # 1) Feature selection (purely variance-based; see R's selFeatures)
    if n_features < n_genes:
        sel = select_features(counts, clusters, n_features=n_features)
        counts_sel = counts[sel] if sp.issparse(counts) else counts[sel, :]
    else:
        counts_sel = counts

    # 2) Artificial doublet count
    if artificial_doublets is None:
        n_ad = int(min(25000, max(1500, np.ceil(n_real * 0.8),
                                  10 * (1 if clusters is None else len(np.unique(clusters))) ** 2)))
    else:
        n_ad = int(artificial_doublets)
    if n_ad <= 2:
        n_ad = int(min(round(n_ad * n_real), 25000))

    if verbose:
        print(f"[scDblFinder] creating ~{n_ad} artificial doublets "
              f"({'random' if clusters is None else 'cluster-aware'})")

    ad = get_artificial_doublets(
        counts_sel, n=n_ad, clusters=clusters, prop_random=prop_random,
        seed=random_state,
    )
    ad_counts = ad["counts"]  # (n_genes_sel, n_ad_actual)
    n_ad_actual = ad_counts.shape[1]
    origins = ad["origins"]

    # 3) Merge real + artificial
    if sp.issparse(counts_sel):
        merged = sp.hstack([counts_sel, sp.csr_matrix(ad_counts)]).tocsc()
    else:
        merged = np.concatenate([np.asarray(counts_sel), ad_counts], axis=1)

    ctype = np.concatenate([np.zeros(n_real, dtype=int),
                            np.ones(n_ad_actual, dtype=int)])
    all_origins = np.array([None] * n_real + list(origins), dtype=object)

    # 4) cxds_score, library size, n_features, n_above_2
    if verbose:
        print("[scDblFinder] computing cxds + size features")
    lsizes = np.asarray(merged.sum(axis=0)).ravel() if sp.issparse(merged) else merged.sum(axis=0)
    n_feat = np.asarray((merged > 0).sum(axis=0)).ravel() if sp.issparse(merged) else (merged > 0).sum(axis=0)
    n_above2 = np.asarray((merged > 2).sum(axis=0)).ravel() if sp.issparse(merged) else (merged > 2).sum(axis=0)
    artificial_idx = np.where(ctype == 1)[0]
    cxds = cxds_score(merged, which_dbls=artificial_idx)

    # 5) PCA
    if verbose:
        print(f"[scDblFinder] PCA (dims={dims})")
    pca = default_processing(merged, dims=dims, random_state=random_state,
                             backend=pca_backend)

    # 6) kNN features
    if verbose:
        print("[scDblFinder] kNN features")
    k_list = default_k_grid(n_real, k)
    d = evaluate_knn(pca, ctype, k=k_list, origins=all_origins,
                     knn_fn=knn_fn, knn_backend=knn_backend)
    d["type"] = np.where(ctype == 0, "real", "doublet")
    d["src"] = np.where(ctype == 0, "real", "artificial")
    d["cxds_score"] = cxds
    d["lsizes"] = lsizes
    d["nfeatures"] = n_feat
    d["nAbove2"] = n_above2

    # 7) xgboost classification loop
    if verbose:
        print("[scDblFinder] xgboost scoring")
    # Include the top principal components as additional predictors
    n_pc_use = int(include_pcs) if np.isscalar(include_pcs) else len(include_pcs)
    add_vals = pca[:, : min(n_pc_use, pca.shape[1] - 1)]
    d = scDbl_score(
        d, add_vals=add_vals,
        nrounds=nrounds, max_depth=max_depth, iter=iter,
        dbr=dbr, dbr_per1k=dbr_per1k, unident_th=float(unident_th),
        metric=metric, random_state=random_state, verbose=verbose,
    )

    # 8) Threshold
    if threshold:
        th = doublet_thresholding(
            d, dbr=dbr, dbr_sd=dbr_sd, dbr_per1k=dbr_per1k,
            return_type="threshold",
        )
        d["class"] = np.where(d["score"].values >= th, "doublet", "singlet")
    else:
        th = float("nan")
        d["class"] = "singlet"

    return ScDblFinderResult(table=d, score_threshold=float(th), n_real=n_real)
