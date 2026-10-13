# ATAC-mode parity with R scDblFinder (pre-registered)

The gates below were written **before** any Python-vs-R comparison was run.
Results are appended at the end, pass or fail.

## Reference

- R scDblFinder **1.16.0** (Bioconductor 3.18, bioconda), R 4.3.3. The
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

- 3,000 cells drawn at random from the demuxlet singlets and doublets
  (2,464 SNG, 536 DBL), plus 1,500 ambient barcodes (demuxlet AMB) so that
  the `minFrags` barcode filter is exercised;
- fragments on chr1, chr19-chr22 and chrX only (2.32 M fragments). chr1 is
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

**G2 · amuletFromCounts, deterministic.** On the peak matrix (with peak
ranges): q-values max absolute difference <= 1e-10.

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

(appended after the runs)
