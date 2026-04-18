"""Minimal end-to-end example for pyscdblfinder.

Run as a script or drop into a Jupyter cell.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from pyscdblfinder import ScDblFinder


def _make_synthetic_adata(rng: np.random.Generator) -> ad.AnnData:
    """500 singlets + 40 artificial doublets, 2 cell types."""
    n_genes = 400
    mu_A = rng.gamma(2.0, 1.0, size=n_genes)
    mu_B = rng.gamma(2.0, 1.0, size=n_genes)
    mu_A[:80] *= 4
    mu_B[200:280] *= 4
    real = np.concatenate([
        rng.poisson(mu_A[:, None], size=(n_genes, 250)),
        rng.poisson(mu_B[:, None], size=(n_genes, 250)),
    ], axis=1)
    # 40 planted doublets
    ia = rng.integers(0, 250, size=40); ib = rng.integers(250, 500, size=40)
    dbl = real[:, ia] + real[:, ib]
    X = np.concatenate([real, dbl], axis=1).T   # cells x genes
    truth = ["singlet"] * 500 + ["doublet"] * 40
    obs = pd.DataFrame({"truth": truth},
                       index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)


def main():
    adata = _make_synthetic_adata(np.random.default_rng(0))
    sdf = ScDblFinder(adata, random_state=0)
    sdf.run(dbr=40 / adata.n_obs, dims=15, n_features=300,
            artificial_doublets=600, iter=2, nrounds=0.25, verbose=True)
    print(adata.obs.groupby("truth")["scDblFinder_class"].value_counts())


if __name__ == "__main__":
    main()
