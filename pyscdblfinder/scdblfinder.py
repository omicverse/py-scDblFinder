"""AnnData-native ``ScDblFinder`` class — high-level wrapper, same lifecycle
as ``milor_py.Milo`` / ``monocle2_py.Monocle`` / ``pydoubletfinder.DoubletFinder``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from . import core as _core


class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def _log(msg: str, level: str = "info") -> None:
    c = {"info": Colors.BLUE, "ok": Colors.GREEN,
         "warn": Colors.WARNING, "err": Colors.FAIL}[level]
    print(f"{c}{msg}{Colors.ENDC}")


@dataclass
class ScDblFinder:
    """AnnData-native wrapper around :func:`sc_dbl_finder`.

    Usage
    -----
    >>> sdf = ScDblFinder(adata)
    >>> sdf.run(dbr=0.07)
    >>> adata.obs[['scDblFinder_score', 'scDblFinder_class']]
    """

    adata: AnnData
    layer: Optional[str] = None   # which layer holds raw counts (None = .X)
    random_state: int = 0

    # populated by .run()
    result: Optional[_core.ScDblFinderResult] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _counts_gene_by_cell(self):
        """Return the counts matrix in genes × cells orientation (Seurat-style)."""
        mat = self.adata.layers[self.layer] if self.layer is not None else self.adata.X
        if sp.issparse(mat):
            return mat.T.tocsc()
        return np.asarray(mat).T

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        clusters: Optional[str | np.ndarray] = None,
        artificial_doublets: Optional[int] = None,
        dbr: Optional[float] = None,
        dbr_sd: Optional[float] = None,
        dbr_per1k: float = 0.008,
        n_features: int = 1352,
        dims: int = 20,
        k: Optional[int | list[int]] = None,
        prop_random: float = 0.0,
        include_pcs: int = 19,
        nrounds: float | int = 0.25,
        max_depth: int = 4,
        iter: int = 3,
        threshold: bool = True,
        knn_fn=None,
        knn_backend: str = "auto",
        pca_backend: str = "auto",
        verbose: bool = False,
    ) -> AnnData:
        """Run scDblFinder and write results to ``adata.obs``.

        Output columns:

            * ``scDblFinder_score``  — per-cell doublet score (0..1)
            * ``scDblFinder_class``  — "doublet" / "singlet"

        """
        counts = self._counts_gene_by_cell()
        if isinstance(clusters, str):
            clusters_arr = self.adata.obs[clusters].to_numpy()
        else:
            clusters_arr = clusters
        if verbose:
            _log(f"[ScDblFinder] {counts.shape[1]} cells × {counts.shape[0]} genes")

        self.result = _core.sc_dbl_finder(
            counts, clusters=clusters_arr,
            artificial_doublets=artificial_doublets,
            dbr=dbr, dbr_sd=dbr_sd, dbr_per1k=dbr_per1k,
            n_features=n_features, dims=dims, k=k,
            prop_random=prop_random, include_pcs=include_pcs,
            nrounds=nrounds, max_depth=max_depth, iter=iter,
            threshold=threshold,
            knn_fn=knn_fn, knn_backend=knn_backend, pca_backend=pca_backend,
            random_state=self.random_state,
            verbose=verbose,
        )
        real = self.result.real_cells()
        self.adata.obs["scDblFinder_score"] = real["score"].values.astype(float)
        self.adata.obs["scDblFinder_class"] = real["class"].values
        self.adata.uns.setdefault("scdblfinder", {}).update({
            "threshold": self.result.score_threshold,
            "n_artificial": int(self.result.table.shape[0] - self.result.n_real),
        })
        _log(
            f"[ScDblFinder] wrote scDblFinder_score + scDblFinder_class — "
            f"threshold={self.result.score_threshold:.3f}",
            level="ok",
        )
        return self.adata
