#!/usr/bin/env Rscript
# Run R scDblFinder on an external counts matrix (genes × cells TSV).
# Emits score + class TSVs for the Python comparison notebook.
#
# Usage: Rscript r_driver_pbmc3k.R <counts_tsv> <outdir> [dbr]

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scDblFinder)
})

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[[1]]
outdir      <- args[[2]]
dbr         <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.075

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
set.seed(42)

cat(sprintf("[R] reading %s\n", counts_path))
counts <- as.matrix(read.table(counts_path, sep = "\t", header = TRUE,
                               row.names = 1, check.names = FALSE))

sce <- SingleCellExperiment(list(counts = counts))
cat(sprintf("[R] running scDblFinder (dbr=%.3f) on %d cells × %d genes\n",
            dbr, ncol(sce), nrow(sce)))
sce <- scDblFinder(sce, dbr = dbr, verbose = FALSE)

res <- data.frame(
  cell = colnames(sce),
  score = sce$scDblFinder.score,
  class = as.character(sce$scDblFinder.class)
)
write.table(res, file = file.path(outdir, "r_result.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[R] wrote %d rows → %s (%d doublets called)\n",
            nrow(res), file.path(outdir, "r_result.tsv"),
            sum(res$class == "doublet")))
