# Benchmark: pyscdblfinder vs R scDblFinder 1.16.0

Real-data comparison on the 16 datasets of the Xi & Li (2021) doublet benchmark
(Zenodo 10.5281/zenodo.4062232, `real_datasets.zip`), two seeds per implementation.

```
Rscript scripts/01_prepare.R <real_datasets_dir> <tasks_dir>
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 8 1,2   # one core per run
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures
```

R: scDblFinder 1.16.0, R 4.3.3, xgboost 1.7.6.1. Python: pyscdblfinder 0.3.2, xgboost 3.0.5.
`figures/comparison_per_dataset.tsv` holds the per-dataset numbers.
