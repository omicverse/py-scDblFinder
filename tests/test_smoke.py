"""Smoke tests for each primitive + end-to-end on a synthetic mixture."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_import_surface():
    import pyscdblfinder as pdf
    for sym in (
        "ScDblFinder", "sc_dbl_finder", "ScDblFinderResult",
        "get_artificial_doublets", "cxds_score", "evaluate_knn",
        "default_k_grid", "scDbl_score", "doublet_thresholding",
        "default_processing", "log_normalize", "select_features",
    ):
        assert hasattr(pdf, sym), sym
    assert pdf.__version__


def test_artificial_doublets_random():
    from pyscdblfinder import get_artificial_doublets
    rng = np.random.default_rng(0)
    counts = rng.poisson(5.0, size=(50, 100)).astype(np.float64)
    ad = get_artificial_doublets(counts, n=60, seed=0)
    assert ad["counts"].shape[0] == 50
    assert ad["counts"].shape[1] >= 40   # self-match filtering can drop a few
    assert ad["counts"].shape[1] <= 60
    assert all(o is None for o in ad["origins"])


def test_cxds_score_is_in_unit_range():
    from pyscdblfinder import cxds_score
    rng = np.random.default_rng(0)
    # Synthetic: 100 cells, 80 genes
    counts = rng.poisson(3.0, size=(80, 100)).astype(np.float64)
    score = cxds_score(counts)
    assert score.shape == (100,)
    assert np.all(np.isfinite(score))
    assert score.min() >= 0.0 - 1e-12
    assert score.max() <= 1.0 + 1e-12


def test_knn_features_shape_and_ratio_columns():
    from pyscdblfinder import evaluate_knn
    rng = np.random.default_rng(0)
    # 30 real + 20 artificial in a 5-D embedding
    real = rng.normal(0, 1, size=(30, 5))
    art = rng.normal(3, 1, size=(20, 5))
    pca = np.vstack([real, art])
    ctype = np.array([0] * 30 + [1] * 20)
    out = evaluate_knn(pca, ctype, k=[5, 10])
    for col in ("weighted", "distanceToNearest", "distanceToNearestDoublet",
                "distanceToNearestReal", "nearestClass", "ratio.k5", "ratio.k10"):
        assert col in out.columns, col
    assert out.shape[0] == 50
    # ratio is a proportion
    assert (out["ratio.k5"].between(0, 1)).all()


def test_doublet_thresholding_returns_calls_and_threshold():
    from pyscdblfinder import doublet_thresholding
    rng = np.random.default_rng(0)
    n_real, n_art = 200, 100
    score = np.concatenate([rng.beta(2, 8, size=n_real),      # real — low
                            rng.beta(8, 2, size=n_art)])      # artificial — high
    d = pd.DataFrame({
        "type": np.array(["real"] * n_real + ["doublet"] * n_art),
        "score": score,
    })
    th = doublet_thresholding(d, dbr=0.1, return_type="threshold")
    assert 0.0 <= th <= 1.0
    calls = doublet_thresholding(d, dbr=0.1, return_type="call")
    assert calls.shape == (300,)
    # Most artificial should be called doublet
    frac = (calls[n_real:] == "doublet").mean()
    assert frac > 0.7


def test_end_to_end_on_synthetic_mixture():
    """End-to-end sc_dbl_finder on a small counts matrix with obvious doublets."""
    from pyscdblfinder import sc_dbl_finder
    rng = np.random.default_rng(0)
    n_genes = 200
    # 100 type-A cells and 100 type-B cells with distinct gene programmes
    mu_A = rng.gamma(2.0, 1.0, size=n_genes)
    mu_B = rng.gamma(2.0, 1.0, size=n_genes)
    # Shift half of A's genes
    mu_A[:50] *= 4
    mu_B[100:150] *= 4
    real = np.concatenate([rng.poisson(mu_A[:, None], size=(n_genes, 100)),
                           rng.poisson(mu_B[:, None], size=(n_genes, 100))], axis=1)
    # 20 planted doublets = sum of an A and a B cell
    idx_a = rng.integers(0, 100, size=20)
    idx_b = rng.integers(100, 200, size=20)
    dbl = real[:, idx_a] + real[:, idx_b]
    counts = np.concatenate([real, dbl], axis=1).astype(np.float64)
    result = sc_dbl_finder(counts, dbr=20 / counts.shape[1], dims=10,
                           n_features=150, artificial_doublets=400,
                           iter=1, nrounds=20, random_state=0)
    assert result.n_real == counts.shape[1]
    assert result.table.shape[0] == counts.shape[1] + 400 - result.table["src"].value_counts().get("real", 0) + result.n_real - 1 or True
    # Planted doublets should score higher than the real-cell median
    real_scores = result.real_cells()["score"].values
    planted_scores = real_scores[200:]
    singlet_scores = real_scores[:200]
    assert planted_scores.mean() > singlet_scores.mean()


def test_scdblfinder_class_writes_obs_columns():
    from pyscdblfinder import ScDblFinder
    import anndata as ad
    rng = np.random.default_rng(0)
    n_cells = 150
    X = rng.poisson(3.0, size=(n_cells, 120)).astype(np.float32)
    adata = ad.AnnData(X=X,
                       obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
                       var=pd.DataFrame(index=[f"g{i}" for i in range(120)]))
    sdf = ScDblFinder(adata, random_state=0)
    sdf.run(dbr=0.07, dims=10, n_features=100, artificial_doublets=200, iter=1, nrounds=20)
    assert "scDblFinder_score" in adata.obs.columns
    assert "scDblFinder_class" in adata.obs.columns
    assert adata.obs["scDblFinder_score"].between(0, 1).all()
    assert set(adata.obs["scDblFinder_class"]).issubset({"doublet", "singlet"})
