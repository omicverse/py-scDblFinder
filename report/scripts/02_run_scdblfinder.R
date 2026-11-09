# Reference run: Bioconductor scDblFinder 1.16.0 (default single-sample RNA mode) on one task and seed.
# Usage: Rscript 02_run_scdblfinder.R <task_dir> <out_dir> <seed>
suppressMessages({library(Matrix); library(SingleCellExperiment); library(scDblFinder); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
td <- args[1]; od <- args[2]; seed <- as.integer(args[3])
dir.create(od, recursive = TRUE, showWarnings = FALSE)
m <- as(readMM(gzfile(file.path(td, "counts.mtx.gz"))), "CsparseMatrix")
rownames(m) <- readLines(file.path(td, "genes.tsv")); colnames(m) <- readLines(file.path(td, "cells.tsv"))
sce <- SingleCellExperiment(list(counts = m))
set.seed(seed)
t0 <- proc.time()[["elapsed"]]
sce <- scDblFinder(sce)
elapsed <- proc.time()[["elapsed"]] - t0
write.table(data.frame(cell = colnames(sce), score = sce$scDblFinder.score, class = as.character(sce$scDblFinder.class)),
            gzfile(file.path(od, "scores.tsv.gz")), sep = "\t", quote = FALSE, row.names = FALSE)
write_json(list(engine = "scDblFinder", version = as.character(packageVersion("scDblFinder")),
                xgboost = as.character(packageVersion("xgboost")), R = R.version.string, seed = seed,
                seconds = elapsed), file.path(od, "timing.json"), auto_unbox = TRUE, digits = NA)
cat(basename(td), seed, sprintf("%.1fs", elapsed), "\n")
