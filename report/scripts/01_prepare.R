# Convert the Xi & Li (2021) real doublet benchmark (Zenodo 4562782, real_datasets.zip) into tasks
# both packages read: <out>/<name>/{counts.mtx.gz (genes x cells), genes.tsv, cells.tsv, labels.tsv}.
# Usage: Rscript 01_prepare.R <dir with *.rds> <out_dir>
suppressMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; out <- args[2]
for (f in Sys.glob(file.path(src, "*.rds"))) {
  name <- sub("\\.rds$", "", basename(f))
  x <- readRDS(f); m <- as(x[[1]], "CsparseMatrix"); lab <- as.character(x[[2]])
  if (is.null(colnames(m))) colnames(m) <- sprintf("cell%d", seq_len(ncol(m)))
  if (is.null(rownames(m))) rownames(m) <- sprintf("gene%d", seq_len(nrow(m)))
  d <- file.path(out, name); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  writeMM(m, file.path(d, "counts.mtx")); system2("gzip", c("-f", file.path(d, "counts.mtx")))
  writeLines(rownames(m), file.path(d, "genes.tsv")); writeLines(colnames(m), file.path(d, "cells.tsv"))
  writeLines(lab, file.path(d, "labels.tsv"))
  cat(sprintf("%-20s %6d genes %6d cells %5d doublets\n", name, nrow(m), ncol(m), sum(lab == "doublet")))
}
