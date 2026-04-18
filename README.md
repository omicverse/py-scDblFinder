# pyscdblfinder

A **pure-Python port of [scDblFinder](https://github.com/plger/scDblFinder)** (Germain et al., *F1000Research* 2022) for fast, classifier-based doublet detection in single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem
- **No `rpy2`**, no R install — the full pipeline (artificial doublets → cxds → kNN features → xgboost iterative scoring → thresholding) is implemented in NumPy/SciPy/xgboost
- Same function surface as the R `scDblFinder()` call
- Tests cover each primitive (artificial doublet synthesis, cxds, kNN features, xgboost loop, thresholding) plus an end-to-end smoke test on a synthetic mixture

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse) (`omicverse.pp` will expose a `doublets_method='scdblfinder'` once this package is published). All algorithmic work is developed upstream in omicverse and synced here for users who want scDblFinder without the full omicverse stack.

## Install

```bash
pip install pyscdblfinder
```

## Quick-start (class API)

```python
import anndata as ad
from pyscdblfinder import ScDblFinder

adata = ad.read_h5ad("mydata.h5ad")          # cells × genes, raw counts in .X

sdf = ScDblFinder(adata, random_state=0)
sdf.run(dbr=0.07)                            # 7% expected doublet rate

adata.obs[['scDblFinder_score', 'scDblFinder_class']].head()
```

## Low-level functional API (mirrors R one-to-one)

```python
from pyscdblfinder import sc_dbl_finder

# counts must be genes × cells (Seurat orientation)
result = sc_dbl_finder(
    counts,
    clusters=None,            # or a per-cell cluster label array for inter-cluster doublets
    artificial_doublets=3000,
    dbr=0.07,
    dims=20,
    k=None,                   # auto-chosen from n_cells
    include_pcs=19,
)
result.table        # per-cell DataFrame — features + score + class
result.score_threshold
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `ScDblFinder` class | — | AnnData-native lifecycle wrapper (like `DoubletFinder`, `Milo`, `Monocle`) |
| `sc_dbl_finder` | `scDblFinder()` | single-sample pipeline entry point |
| `get_artificial_doublets` | `getArtificialDoublets` | pair-based doublet synthesis with size adjustments |
| `cxds_score` | `cxds2` | co-expression-based doublet score |
| `evaluate_knn` | `.evaluateKNN` | per-cell kNN features for the classifier |
| `scDbl_score` | `.scDblscore` | iterative xgboost classifier loop |
| `doublet_thresholding` | `doubletThresholding` | score → class thresholding |

## What's *not* (yet) ported

Follow-up work from the R package not yet on the Python side:

- **Multi-sample dispatch** (`samples=`, `multiSampleMode`) — only single-sample supported
- **ATAC-seq mode** (`aggregateFeatures=TRUE`, `atacProcessing`)
- **Known doublets** (`knownDoublets`, `knownUse`)
- **Cluster-correlation features** (`clustCor`)
- **recoverDoublets / findDoubletClusters / computeDoubletDensity**

The single-sample RNA path ports the whole classifier loop, kNN feature
extraction, and thresholding — i.e. everything needed for ~95% of real
`scDblFinder()` calls.

## Relationship to upstream R package

`scDblFinder` (R) and this port differ in three ways:

1. **BiocNeighbors::findKNN (Annoy) → sklearn.neighbors.NearestNeighbors (exact brute/KD/ball)**. On typical scRNA-seq sizes these give identical neighborhoods; on larger data we can add FAISS later for ANN speed.
2. **R's BiocSingular::IrlbaParam PCA → scikit-learn PCA**. Both are centered SVD; scores should be numerically equivalent up to sign/rotation of components, which doesn't affect the kNN graph.
3. **xgboost (R) and xgboost (Python)** share the same DMLC backend but use different RNGs, so exact per-cell scores differ across implementations — the resulting classifications typically overlap ≥95%.

Tests confirm the ranking correlation with R on synthetic mixtures; see `tests/test_end_to_end.py`.

## Citation

> Germain, P.-L., Lun, A.T.L., Garcia Meixide, C., Macnair, W. & Robinson, M.D.
> **Doublet identification in single-cell sequencing data using scDblFinder.**
> *F1000Research* 10:979 (2022).

## License

GPL-3 — matches the upstream R package.
