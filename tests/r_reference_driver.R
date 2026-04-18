#!/usr/bin/env Rscript
# Drive R scDblFinder on a synthetic doublet-rich matrix and emit TSV +
# JSON outputs the Python parity test consumes.
#
# Usage: Rscript r_reference_driver.R <outdir>
#
# Because scDblFinder uses xgboost with an internal RNG that differs
# between R and Python, this test checks rank correlation + classification
# overlap on the same input rather than bit-for-bit equality.

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scDblFinder)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "r_ref_out"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(123)

# Use the built-in synthetic doublet dataset (mockDoubletSCE) so both
# sides consume the exact same counts.
sce <- mockDoubletSCE()
counts_mat <- as.matrix(counts(sce))
truth <- sce$type
stopifnot(all(truth %in% c("singlet","doublet")))

write.table(counts_mat,
            file = file.path(outdir, "counts.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(data.frame(cell = colnames(sce), truth = truth),
            file = file.path(outdir, "truth.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Run scDblFinder
sce <- scDblFinder(sce, verbose = FALSE)
score <- sce$scDblFinder.score
cls   <- as.character(sce$scDblFinder.class)

write.table(data.frame(cell = colnames(sce), score = score, class = cls),
            file = file.path(outdir, "r_result.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

meta <- list(n_cells = ncol(sce), n_genes = nrow(sce),
             r_pkg_version = as.character(packageVersion("scDblFinder")))
write(toJSON(meta, auto_unbox = TRUE),
      file = file.path(outdir, "meta.json"))
cat(sprintf("[r_reference_driver] wrote %d rows to %s\n", ncol(sce), outdir))
