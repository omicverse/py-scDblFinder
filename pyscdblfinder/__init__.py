"""
pyscdblfinder: Pure-Python port of R scDblFinder (Germain et al., 2022).

scDblFinder detects doublets in single-cell RNA-seq data via a two-stage
approach: synthesize artificial doublets by combining pairs of real cells,
train a gradient-boosted classifier on the real-vs-artificial task, and
predict per-cell doublet probability. This Python port mirrors the
single-sample path of the R package.

Quick-start
-----------
>>> from pyscdblfinder import ScDblFinder
>>> sdf = ScDblFinder(adata)
>>> sdf.run(dbr=0.07)
>>> adata.obs["scDblFinder_score"], adata.obs["scDblFinder_class"]

Low-level functional API
------------------------
>>> from pyscdblfinder import sc_dbl_finder
>>> result = sc_dbl_finder(counts_genes_by_cells, clusters=None, dbr=0.07)
>>> result.table.head()   # per-cell scores + features

scATAC-seq
----------
>>> res = sc_dbl_finder(peaks_by_cells, aggregate_features=True,
...                     n_features=25, processing="normFeatures")
>>> am = amulet("atac_fragments.tsv.gz")          # AMULET on fragments
>>> combined = fisher_combine(1 - score, am["p.value"])
"""
from __future__ import annotations

from .artificial import get_artificial_doublets
from .atac import aggregate_features, atac_processing, log_norm_counts, tfidf
from .clamulet import clamulet
from .fragments import (amulet, amulet_from_counts, fisher_combine,
                        get_fragment_overlaps)
from .classifier import scDbl_score
from .core import ScDblFinderResult, sc_dbl_finder
from .cxds import cxds_score
from .knn_features import default_k_grid, evaluate_knn
from .preprocessing import default_processing, log_normalize, select_features
from .scdblfinder import ScDblFinder
from .thresholding import doublet_thresholding

__version__ = "0.3.0"

__all__ = [
    # class API
    "ScDblFinder",
    # main functional entry
    "sc_dbl_finder",
    "ScDblFinderResult",
    # building blocks
    "get_artificial_doublets",
    "cxds_score",
    "evaluate_knn",
    "default_k_grid",
    "scDbl_score",
    "doublet_thresholding",
    "default_processing",
    "log_normalize",
    "select_features",
    # scATAC
    "amulet",
    "amulet_from_counts",
    "get_fragment_overlaps",
    "clamulet",
    "aggregate_features",
    "atac_processing",
    "tfidf",
    "log_norm_counts",
    "fisher_combine",
]
