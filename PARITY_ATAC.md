# ATAC-mode parity with R scDblFinder (pre-registered)

The gates below were written **before** any Python-vs-R comparison was run.
Results are appended at the end, pass or fail.

## Reference

- R scDblFinder **1.16.0** (Bioconductor 3.18, bioconda), R 4.3.3, R
  xgboost **1.7.6.1**. (With R xgboost 3.x, scDblFinder 1.16's `.xgbtrain`
  fails inside `xgb.cv`, which now requires an `xgb.DMatrix`; 1.7 is the
  series 1.16 was written for.) The
  ATAC functions (`getFragmentOverlaps`, `amulet`, `amuletFromCounts`,
  `aggregateFeatures`, `TFIDF`, `clamulet`, `.removeHighOverlapSites`,
  `.defaultKnnKs`) were compared with the current development source
  (1.27.10) and are unchanged apart from formatting. The one relevant
  difference is `.xgbtrain`: 1.16 uses xgboost's default learning rate (0.3),
  later versions pass `learning_rate=1`. The port follows 1.16.
- The driver is `tests/r_reference_driver_atac.R`. It runs outside the
  package; the Python package never calls R.

## Fixture

Public data: GEO **GSE162690** (Germain et al. 2021 benchmark; cell-line
scATAC with demuxlet genotype calls), the *LowLoading* sample.

- 3,000 cells among the 11,173 that passed ArchR QC (the cells of the
  released peak matrix): all 336 demuxlet doublets (DBL) among them plus
  2,664 singlets (SNG) drawn at random, i.e. 11.2% doublets. ArchR QC had
  already removed many doublets, so a random draw would have left ~100;
- plus 1,500 ambient barcodes (demuxlet AMB) in the fragment file, so that
  the `minFrags` barcode filter is exercised;
- fragments of these 4,500 barcodes on chr1, chr19-chr22 and chrX only. chr1 is
  longer than 100 Mb and chrX is a default exclusion chromosome longer than
  100 Mb, which exercises the two R edge cases documented in
  `pyscdblfinder/fragments.py`;
- the ArchR peak matrix released with the GEO series, restricted to the
  3,000 cells, for the count-based paths.

Doublet ground truth: demuxlet `DBL` vs `SNG`.

## Gates

**G1 · amulet, deterministic.** Three configurations:
(a) whole file, not indexed;
(b) bgzipped + tabix-indexed (per-chromosome mode);
(c) explicit `barcodes` (the 3,000 cells) plus an extra repeat-like exclusion
region.

For each: the same barcode set; `nFrags`, `uniqFrags`, `nAbove2` and
`total.nAbove2` exactly equal; `p.value` and `q.value` max relative
difference <= 1e-10.

**G2 · amuletFromCounts, deterministic.** q-values max absolute
difference <= 1e-10 (NaN in the same places), for (a) the peak count
matrix passed as a plain matrix and (b) the SingleCellExperiment with peak
ranges, `exclude=NULL, maxWidth=1000`. Amended before G2 was compared:
R's default call on an SCE with row ranges errors in scDblFinder 1.16
(`seqnames(granges(x)) %in% exclude` on an Rle), and the ArchR peaks are
501 bp, so the default `maxWidth=500` would drop them all.

**G3 · scDblFinder ATAC mode, stochastic.** The vignette setting
`aggregateFeatures=TRUE, nfeatures=25, processing="normFeatures"`, run
with 3 seeds in R and 3 in Python. Feature clustering (irlba, k-means) and
xgboost are random in both languages and cannot be aligned across
languages, so the yardstick is R's own seed-to-seed agreement:

- G3a: median Spearman(Python, R) over the 9 pairs >= 0.9 x the median
  Spearman(R, R) over the 3 R pairs;
- G3b: Jaccard of called doublets at the same rate (the top `n` cells, with
  `n` the mean number of R doublet calls): median Python-R >= 0.9 x
  median R-R;
- G3c: mean AUPRC against demuxlet >= mean R AUPRC - 0.05.

**G4 · clamulet, stochastic.** As G3, with clamulet on the fragment
fixture restricted to the 3,000 cells: G4a Spearman, G4b Jaccard, G4c
AUPRC, same rules.

**G5 · RNA mode not degraded.** The existing tests
(`tests/test_smoke.py`, `tests/test_r_parity.py`) pass unchanged. The ATAC
changes do not alter the RNA code path, which keeps `processing="default"`
and no aggregation.

## Results

Run on 2026-10-13: R scDblFinder 1.16.0 / R xgboost 1.7.6.1 / R 4.3.3, and
pyscdblfinder 0.3.0 / xgboost 3.4.2 / scikit-learn 1.9.1 / numpy 2.4.6.
Comparison script: `examples/parity_atac.py`. **All gates pass.**

| Gate | Check | Result |
|---|---|---|
| G1a amulet, whole file | 1,403 barcodes; 4 count columns | identical; p, q max rel. diff 6.5e-15 — **pass** |
| G1b amulet, tabix | 935 barcodes (per-chromosome R semantics) | identical; max rel. diff 2.0e-15 — **pass** |
| G1c amulet, barcodes + extra exclusion | 3,000 barcodes | identical; max rel. diff 4.3e-15 — **pass** |
| G2a amuletFromCounts, matrix | 3,000 cells | max abs. diff 4.1e-15 — **pass** |
| G2b amuletFromCounts, SCE `maxWidth=1000` | 3,000 cells | max abs. diff 4.1e-15 — **pass** |
| G3a scDblFinder ATAC, Spearman | Py-R 0.724 vs R-R 0.729 (gate >= 0.656) | **pass** |
| G3b scDblFinder ATAC, Jaccard of top 138 | Py-R 0.415 vs R-R 0.401 (gate >= 0.361) | **pass** |
| G3c scDblFinder ATAC, AUPRC | Py 0.259 vs R 0.271 (gate >= 0.221) | **pass** |
| G4a clamulet, Spearman | Py-R 0.688 vs R-R 0.715 (gate >= 0.643) | **pass** |
| G4b clamulet, Jaccard of top 336 | Py-R 0.254 vs R-R 0.263 (gate >= 0.237) | **pass** |
| G4c clamulet, AUPRC | Py 0.100 vs R 0.101 (gate >= 0.051) | **pass** |
| G5 RNA mode | scores identical to 0.2.1 (Spearman to R 0.5467, call overlap 0.972 on mockDoubletSCE, both versions) | **pass** |

Timing on the fixture (2.1 M fragments, 3,000 cells; shared CPU node):

| | R | Python | speed-up |
|---|---|---|---|
| amulet, whole file | 177 s | 2.7 s | 66x |
| amulet, tabix | 1,186 s | 1.9 s | ~600x |
| scDblFinder ATAC (mean of 3 seeds) | 140 s | 15 s | 9x |
| clamulet (mean of 3 seeds) | 895 s | 11.5 s | 78x |

Agent D (Epione) measured R `amulet` on the full GSE162690 high-loading
file (68 M fragments) not finishing in 4.5 h, against 5-8 min for this port.

### Notes from the run

- Accuracy on this fixture is low for every method in both languages (AUROC
  0.33-0.57). ArchR QC had already removed the easy doublets, and the
  fragment fixture keeps only six chromosomes. clamulet scores are
  anti-correlated with the demuxlet calls (AUROC 0.34 in R and Python alike),
  consistent with the vignette's advice not to use it.
- Python AMULET + Fisher combination with scDblFinder (seed 1): AUPRC 0.227
  vs 0.112 for AMULET alone.
- Findings in R scDblFinder 1.16 made while building the reference:
  `processing="atac"` errors ("arguments imply differing number of rows");
  `amuletFromCounts()` on an SCE with row ranges errors with the default
  `exclude`; `amuletFromCounts` filters features with Holm-adjusted p-values
  (R's `p.adjust` default) and only the final q-values use BH; with R
  xgboost 3.x, `clamulet` (and any `.xgbtrain` cross-validation) fails.
- The existing RNA test `test_py_scores_rank_correlate_with_R` fails
  (rho 0.183 < 0.2) against this R reference with both 0.2.1 and 0.3.0; its
  threshold depends on the R/xgboost versions of the reference env.
- Cluster note: on an oversubscribed node, OpenMP spin-waiting made xgboost
  ~100x slower with 2 threads than with 1; `OMP_WAIT_POLICY=PASSIVE` fixed
  it. Separately, `nthreads=0` now resolves to the CPUs the process may use
  rather than every core on the machine.
- Parity is with R scDblFinder's `amulet()`, not with the original AMULET
  scripts (UcarLab), which differ in places (e.g. their
  `FragmentFileOverlapCounter.py` does not flush the last chromosome at end
  of file).
