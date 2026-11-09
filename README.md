# pyscdblfinder

A **pure-Python port of [scDblFinder](https://github.com/plger/scDblFinder)** (Germain et al., *F1000Research* 2022) for fast, classifier-based doublet detection in single-cell RNA-seq **and scATAC-seq** data, including scDblFinder's re-implementation of **AMULET** (Thibodeau et al., *Genome Biology* 2021).

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

## scATAC-seq (new in 0.3.0)

Three approaches, as in the scDblFinder ATAC vignette:

```python
import pyscdblfinder as psd

# 1) scDblFinder with feature aggregation (peaks/tiles -> 25 meta-features)
sdf = psd.ScDblFinder(adata_atac, random_state=0)   # cells x peaks, raw counts
sdf.run(aggregate_features=True, n_features=25, processing="normFeatures")

# 2) AMULET on the fragment file (loci covered by > 2 fragments)
am = psd.amulet("atac_fragments.tsv.gz")            # p.value / q.value per barcode

# 3) combine the two (Fisher's method), recommended by the vignette as the
#    most robust overall
bc = adata_atac.obs_names.intersection(am.index)
combined_p = psd.fisher_combine(
    1 - adata_atac.obs.loc[bc, "scDblFinder_score"], am.loc[bc, "p.value"])
```

`psd.clamulet(fragments)` (classifier on fragment coverages) is ported too;
the R authors report it is slower and outperformed by the combination above.

AMULET recommends excluding repeats as well as mitochondrial and sex
chromosomes: pass a BED file or `(chrom, start, end)` tuples to
`regions_to_exclude`. A tabix-indexed fragment file (`.tbi` next to it) is
processed chromosome by chromosome, as in R.

### R behaviour kept on purpose

The port follows R scDblFinder exactly, including a few edge cases worth
knowing about:

- The default `regions_to_exclude` covers the sex and mitochondrial
  chromosomes only over `[1, 1e8]`, so the part of human chrX beyond 100 Mb is
  not excluded.
- With a tabix-indexed file, R reads each chromosome only over
  `[1, 1e8]`: fragments beyond 100 Mb (human chr1-chr12) are ignored. The port
  does the same by default (`tabix_window=1e8`); pass `tabix_window=None` or
  `full_in_memory=True` to use whole chromosomes.
- `processing="atac"` fails in R (its embedding has one row per feature);
  the port embeds the cells (TF-IDF + uncentred SVD), which is the evident
  intent. The vignette uses `processing="normFeatures"`.
- R's `aggregateFeatures` uses `mbkmeans` only if that optional package is
  installed; by default it runs `kmeans`, and so does the port
  (`use_mbk=True` switches to mini-batch k-means).

- In R, `amuletFromCounts` filters features with Holm-adjusted p-values
  (`p.adjust`'s default) before the BH q-values; the port does the same.

Parity against R scDblFinder 1.16.0 is documented in
[`PARITY_ATAC.md`](PARITY_ATAC.md): `amulet` and `amuletFromCounts` match R
to ~1e-15, the stochastic paths agree with R as well as R agrees with
itself across seeds, and the port is 9-600x faster (AMULET on a 68 M
fragment file: 5-8 min, where R did not finish in 4.5 h). Parity is with R
scDblFinder's `amulet()`, not with the original AMULET scripts.

On clusters with old glibc (e.g. CentOS 7), xgboost 3.x wheels may not
install; xgboost 2.1.x works (`xgboost>=1.7` is the only constraint).

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
| `sc_dbl_finder(aggregate_features=..., processing=...)` | `scDblFinder(aggregateFeatures=, processing=)` | scATAC mode |
| `aggregate_features` | `aggregateFeatures` | aggregate peaks/tiles into meta-features |
| `tfidf` | `TFIDF` | TF-IDF normalisation |
| `atac_processing` | `.atacProcessing` | TF-IDF + uncentred SVD (cells) |
| `log_norm_counts` | `scuttle::logNormCounts` | library-size log-normalisation |
| `get_fragment_overlaps` | `getFragmentOverlaps` | per-barcode loci covered by > 2 fragments |
| `amulet` | `amulet` | AMULET p/q-values from a fragment file |
| `amulet_from_counts` | `amuletFromCounts` | count-matrix approximation of AMULET |
| `clamulet` | `clamulet` | classifier on fragment coverages |
| `fisher_combine` | vignette's `aggregation::fisher` | combine scDblFinder and AMULET |

## What's *not* (yet) ported

Follow-up work from the R package not yet on the Python side:

- **Multi-sample dispatch** (`samples=`, `multiSampleMode`) — only single-sample supported
- **Known doublets** (`knownDoublets`, `knownUse`)
- **Cluster-correlation features** (`clustCor`)
- **recoverDoublets / findDoubletClusters / computeDoubletDensity**

The single-sample RNA path ports the whole classifier loop, kNN feature
extraction, and thresholding — i.e. everything needed for ~95% of real
`scDblFinder()` calls.

## Relationship to R scDblFinder

Since 0.3.1 the RNA path follows R scDblFinder 1.16.0 step by step: feature selection
(`rowMeans` order), scuttle log-normalisation and PCA, artificial-doublet pairing, the
`.defaultKnnKs` grid, the training-exclusion rules, `.xgbtrain` (exact trees, cross-validated
number of rounds) and `doubletThresholding` (Brent minimisation, as `stats::optimize`).

Steps without random numbers are checked against stored R output in
`tests/test_rna_reference.py` (no R needed): selected features, kNN grid, library-size features
and thresholds (relative error < 1e-9). Artificial doublets and xgboost use random numbers in
both languages, so the scores are compared with R's own seed-to-seed agreement.

On the 16 real datasets of the Xi & Li (2021) benchmark (168,239 cells, two seeds per
implementation; scripts in `report/scripts`):

| | R vs R (seed 1 vs 2) | Python vs R |
|---|---|---|
| score Spearman, median | 0.874 | 0.874 |
| doublet-call Jaccard, median | 0.770 | 0.768 |

| | scDblFinder 1.16.0 | pyscdblfinder 0.3.1 |
|---|---|---|
| mean AUPRC | 0.562 | 0.560 |
| mean AUROC | 0.796 | 0.794 |

On one thread, pyscdblfinder ran 1.1–2.5× faster than R (median 1.4×). ATAC-mode parity is
documented in [`PARITY_ATAC.md`](PARITY_ATAC.md).

## Citation

> Germain, P.-L., Lun, A.T.L., Garcia Meixide, C., Macnair, W. & Robinson, M.D.
> **Doublet identification in single-cell sequencing data using scDblFinder.**
> *F1000Research* 10:979 (2022).

## License

GPL-3 — matches the upstream R package.
