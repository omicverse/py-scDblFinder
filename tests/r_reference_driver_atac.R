#!/usr/bin/env Rscript
# R reference for the ATAC-mode parity checks (see PARITY_ATAC.md).
#
# Usage:
#   Rscript r_reference_driver_atac.R <fixture_dir> <outdir> [what] [seeds]
#
# <fixture_dir> must contain
#   frag_subset.tsv.gz          fragments (not indexed)
#   tabix/frag_subset.tsv.gz    the same, bgzipped + tabix-indexed
#   cells.tsv                   barcode <TAB> demuxlet call (SNG/DBL)
#   peaks.mtx, peaks.bed, peaks_cells.txt   peak counts (written by `export`)
# what: comma-separated subset of export,amulet,fromcounts,scdbl,clamulet
# seeds: comma-separated seeds for the stochastic methods (default 1,2,3)

suppressPackageStartupMessages({
  library(scDblFinder)
  library(GenomicRanges)
  library(SummarizedExperiment)
  library(SingleCellExperiment)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
fx <- args[[1]]
out <- args[[2]]
what <- if (length(args) >= 3) strsplit(args[[3]], ",")[[1]] else
  c("amulet", "fromcounts", "scdbl", "clamulet")
seeds <- if (length(args) >= 4) as.integer(strsplit(args[[4]], ",")[[1]]) else 1:3
dir.create(out, showWarnings = FALSE, recursive = TRUE)

cells <- read.delim(file.path(fx, "cells.tsv"), header = FALSE,
                    col.names = c("barcode", "call"))

wtab <- function(d, f) {
  d <- cbind(barcode = row.names(d), as.data.frame(d))
  write.table(d, file.path(out, f), sep = "\t", quote = FALSE,
              row.names = FALSE)
}

if ("export" %in% what) {
  # peak matrix restricted to the fixture cells (written for Python)
  se <- readRDS(Sys.getenv("PEAKMATRIX_RDS"))
  colnames(se) <- sub("^.*#", "", colnames(se))
  se <- se[, cells$barcode]
  writeMM(assay(se), file.path(fx, "peaks.mtx"))
  rr <- rowRanges(se)
  write.table(data.frame(chrom = as.character(seqnames(rr)),
                         start = start(rr) - 1L, end = end(rr)),
              file.path(fx, "peaks.bed"), sep = "\t", quote = FALSE,
              row.names = FALSE, col.names = FALSE)
  writeLines(colnames(se), file.path(fx, "peaks_cells.txt"))
}

if ("amulet" %in% what) {
  t0 <- Sys.time()
  a <- amulet(file.path(fx, "frag_subset.tsv.gz"), verbose = FALSE)
  wtab(a, "amulet_full.tsv")
  t1 <- Sys.time()
  b <- amulet(file.path(fx, "tabix", "frag_subset.tsv.gz"), verbose = FALSE)
  wtab(b, "amulet_tabix.tsv")
  t2 <- Sys.time()
  ex <- suppressWarnings(c(
    GRanges(c("M", "chrM", "MT", "X", "Y", "chrX", "chrY"),
            IRanges(1L, width = 10^8)),
    GRanges(c("chr19", "chr1"), IRanges(c(1000000L, 50000000L),
                                        c(3000000L, 50500000L)))))
  cc <- amulet(file.path(fx, "frag_subset.tsv.gz"), barcodes = cells$barcode,
               regionsToExclude = ex, verbose = FALSE)
  wtab(cc, "amulet_barcodes.tsv")
  write(toJSON(list(full_s = as.numeric(t1 - t0, units = "secs"),
                    tabix_s = as.numeric(t2 - t1, units = "secs")),
               auto_unbox = TRUE), file.path(out, "amulet_time.json"))
}

if (any(c("fromcounts", "scdbl") %in% what)) {
  m <- readMM(file.path(fx, "peaks.mtx"))
  m <- as(m, "CsparseMatrix")
  bed <- read.delim(file.path(fx, "peaks.bed"), header = FALSE)
  colnames(m) <- readLines(file.path(fx, "peaks_cells.txt"))
  rownames(m) <- paste0("p", seq_len(nrow(m)))
  sce <- SingleCellExperiment(list(counts = m),
    rowRanges = GRanges(bed$V1, IRanges(bed$V2 + 1L, bed$V3)))
}

if ("fromcounts" %in% what) {
  # The default call on an SCE with row ranges fails in scDblFinder 1.16
  # (`seqnames(...) %in% exclude` on an Rle); record the error and compare
  # (a) a plain count matrix (no feature filtering) and (b) the SCE with
  # exclude=NULL and maxWidth=1000 (ArchR peaks are 501 bp, so the default
  # maxWidth=500 would drop them all).
  err <- tryCatch({amuletFromCounts(sce); "ok"},
                  error = function(e) conditionMessage(e))
  writeLines(err, file.path(out, "amulet_from_counts_default.txt"))
  qa <- amuletFromCounts(as.matrix(counts(sce)))
  qb <- amuletFromCounts(sce, maxWidth = 1000L, exclude = NULL)$amuletFromCounts.q
  write.table(data.frame(barcode = colnames(sce), q_matrix = qa, q_w1000 = qb),
              file.path(out, "amulet_from_counts.tsv"), sep = "\t",
              quote = FALSE, row.names = FALSE)
}

if ("scdbl" %in% what) {
  for (s in seeds) {
    set.seed(s)
    t0 <- Sys.time()
    x <- scDblFinder(sce, aggregateFeatures = TRUE, nfeatures = 25,
                     processing = "normFeatures", verbose = FALSE)
    el <- as.numeric(Sys.time() - t0, units = "secs")
    write.table(data.frame(barcode = colnames(x), score = x$scDblFinder.score,
                           class = as.character(x$scDblFinder.class)),
                file.path(out, sprintf("scdbl_seed%d.tsv", s)), sep = "\t",
                quote = FALSE, row.names = FALSE)
    write(toJSON(list(seconds = el), auto_unbox = TRUE),
          file.path(out, sprintf("scdbl_seed%d.json", s)))
  }
  # processing="atac" as documented: record whether R accepts it
  set.seed(1)
  r <- tryCatch({
    scDblFinder(sce, aggregateFeatures = TRUE, nfeatures = 25,
                processing = "atac", verbose = FALSE)
    "ok"
  }, error = function(e) conditionMessage(e))
  writeLines(r, file.path(out, "scdbl_processing_atac.txt"))
}

if ("clamulet" %in% what) {
  for (s in seeds) {
    set.seed(s)
    t0 <- Sys.time()
    d <- clamulet(file.path(fx, "frag_subset.tsv.gz"),
                  barcodes = cells$barcode, verbose = FALSE)
    el <- as.numeric(Sys.time() - t0, units = "secs")
    wtab(d, sprintf("clamulet_seed%d.tsv", s))
    write(toJSON(list(seconds = el), auto_unbox = TRUE),
          file.path(out, sprintf("clamulet_seed%d.json", s)))
  }
}

meta <- list(scDblFinder = as.character(packageVersion("scDblFinder")),
             xgboost = as.character(packageVersion("xgboost")),
             R = R.version.string)
write(toJSON(meta, auto_unbox = TRUE), file.path(out, "meta.json"))
cat("done\n")
