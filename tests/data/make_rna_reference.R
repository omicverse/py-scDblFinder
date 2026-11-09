# Reference values for tests/test_rna_reference.py (RNA mode).
# Run from the repository root:  Rscript tests/data/make_rna_reference.R <J293t-dm task dir>
# Reference: R 4.3.3, scDblFinder 1.16.0, xgboost 1.7.6.1 (bioconda).
# Fixture: the J293t-dm sample of the Xi & Li (2021) doublet benchmark (Zenodo 4562782; 500 cells,
# demuxlet doublet labels), restricted to 3,000 genes with non-zero counts drawn at random (seed 42).
suppressMessages({library(Matrix); library(SingleCellExperiment); library(scDblFinder)})
td <- commandArgs(TRUE)[1]
d <- "tests/data"
w <- function(x, f) write.table(x, file.path(d, f), sep = "\t", quote = FALSE, row.names = FALSE)
g17 <- function(x) sprintf("%.17g", x)
m <- as(readMM(gzfile(file.path(td, "counts.mtx.gz"))), "CsparseMatrix")
rownames(m) <- readLines(file.path(td, "genes.tsv")); colnames(m) <- readLines(file.path(td, "cells.tsv"))
set.seed(42)
m <- m[sort(sample(which(rowSums(m) > 0), 3000)), ]
writeMM(m, file.path(d, "rna_counts.mtx")); system2("gzip", c("-f", file.path(d, "rna_counts.mtx")))
w(data.frame(label = readLines(file.path(td, "labels.tsv"))), "rna_labels.tsv")
sce <- SingleCellExperiment(list(counts = m))

# deterministic: feature selection and kNN sizes
sel <- scDblFinder:::selFeatures(sce, NULL, nfeatures = 1352)
w(data.frame(index = match(sel, rownames(m)) - 1L), "rna_selfeatures.tsv")
ns <- c(50, 500, 3000, 26426, 100000)
w(data.frame(n = rep(ns, sapply(ns, function(n) length(scDblFinder:::.defaultKnnKs(NULL, n)))),
             k = unlist(lapply(ns, function(n) scDblFinder:::.defaultKnnKs(NULL, n)))), "rna_knn_ks.tsv")

# stochastic: three seeds
for (s in 1:3) {
  set.seed(s)
  tab <- scDblFinder(sce, returnType = "table", verbose = FALSE)
  r <- tab$src == "real"
  w(data.frame(lsizes = tab$lsizes[r], nfeatures = tab$nfeatures[r], nAbove2 = tab$nAbove2[r],
               score = g17(tab$score[r]), class = as.character(tab$class[r])), sprintf("rna_seed%d.tsv", s))
  if (s == 1) {
    # deterministic given a score table: doubletThresholding
    dd <- as.data.frame(tab[, c("type", "src", "score", "include.in.training")])
    dd$type <- as.character(dd$type); dd$src <- as.character(dd$src)
    w(data.frame(type = dd$type, src = dd$src, score = g17(dd$score), include = dd$include.in.training), "rna_thr_input.tsv")
    gd <- 0.01 * sum(dd$src == "real") / 1000
    th <- c(doubletThresholding(dd, stringency = 0.5),
            doubletThresholding(dd, dbr.sd = 0.3 * gd + 0.025, stringency = 0.5),
            doubletThresholding(dd, dbr = 0.05, stringency = 0.5))
    call07 <- as.character(doubletThresholding(dd, dbr.sd = 0.3 * gd + 0.025, stringency = 0.7, returnType = "call"))
    w(data.frame(case = c("default", "dbr_sd", "dbr_0.05"), threshold = g17(th)), "rna_thresholds.tsv")
    w(data.frame(call = call07), "rna_calls_07.tsv")
  }
}
