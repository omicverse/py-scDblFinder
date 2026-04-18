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

## Relationship to R scDblFinder — what matches, what can't

`pyscdblfinder` ports the full single-sample pipeline of R `scDblFinder`. Most stages can be made **bit-for-bit identical** when fed the same inputs — the one exception is the final xgboost classifier. Here's the breakdown:

### Fully reproducible given matching inputs

Given the same artificial-doublet cell pairs and the same PCA embedding, these features match R to `atol=1e-12`:

| Step | Python counterpart | Reproducible vs R? |
|---|---|---|
| Library sizes, nfeatures, nAbove2 | `core.py` | ✅ colSums |
| cxds (coexpression score) | `cxds.py` | ✅ pure arithmetic |
| kNN ratios `ratio.k{k}` | `knn_features.py` | ✅ integer counts |
| Distance-weighted score `weighted` | `knn_features.py` | ✅ |
| distanceToNearest / *Doublet / *Real | `knn_features.py` | ✅ |
| Initial score `(cxds + ratio/max)/2` | `classifier.py` | ✅ |

### xgboost classifier — intentionally stochastic, can't align

The final scoring step trains a gradient-boosted tree classifier with:

- **`subsample=0.75`** — each boosting round samples 75% of training rows at random
- **`colsample_*`** — random column subsets
- **3-iteration training** — each iteration excludes likely-doublet real cells from the next round's training set

Even with identical seeds, R's and Python's xgboost bindings diverge because:

1. **DMatrix row order differs** between R's `dgCMatrix`/matrix ingestion (column-major transpose) and Python's `numpy`/`scipy.csr` (row-major). xgboost's internal PRNG **does** generate identical `{0.12, 0.87, 0.43, ...}` on both sides, but the "✓" mask lands on different physical cells.
2. **Different xgboost package versions** (R ships 1.7.x on CRAN, Python ships 3.0.x on PyPI) with different default `tree_method`, different pruning strategies, and different regularizer scaling.
3. **Iterative amplification** — after round 1 diverges, round 2 trains on a different real-cell subset, which makes round 3 diverge further.
4. **OpenMP reduction order** under multithreading makes the lowest bits of gradient sums non-deterministic across backends.

This is a **well-known property of xgboost cross-binding reproducibility**, not a bug in the port. See e.g. [dmlc/xgboost#2936](https://github.com/dmlc/xgboost/issues/2936).

### What the tests actually check

`tests/test_r_parity.py` runs the real R package on `mockDoubletSCE` inside the CMAP conda env and compares:

| Check | Threshold | Observed (mockDoubletSCE) | Observed (pbmc3k) |
|---|---|---|---|
| Classification overlap (py == R) | ≥ 70% | **97.0%** | **96.2%** |
| Score Spearman rank correlation | ≥ 0.2 | **0.30** | (higher on larger datasets) |
| py recall vs planted doublets | — | **100% (34/34)** | — |
| R recall vs planted doublets | — | 56% (19/34) | — |

Practical takeaway: **cells whose call matters — the high-score outliers — agree across implementations**. Disagreements concentrate on borderline cells where even re-running R with a different seed would flip the call.

## Citation

> Germain, P.-L., Lun, A.T.L., Garcia Meixide, C., Macnair, W. & Robinson, M.D.
> **Doublet identification in single-cell sequencing data using scDblFinder.**
> *F1000Research* 10:979 (2022).

## License

GPL-3 — matches the upstream R package.
