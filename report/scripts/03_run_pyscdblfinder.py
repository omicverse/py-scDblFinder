"""pyscdblfinder run on one task and seed, mirroring 02_run_scdblfinder.R.

Usage: python 03_run_pyscdblfinder.py <task_dir> <out_dir> <seed>
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pyscdblfinder


def main():
    td, od, seed = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    od.mkdir(parents=True, exist_ok=True)
    m = sp.csc_matrix(scipy.io.mmread(td / "counts.mtx.gz"))          # genes x cells, as in R
    cells = (td / "cells.tsv").read_text().split("\n")[: m.shape[1]]
    t0 = time.perf_counter()
    res = pyscdblfinder.sc_dbl_finder(m, random_state=seed, progress=False)
    elapsed = time.perf_counter() - t0
    tab = res.table.iloc[: res.n_real]                                 # real cells come first
    pd.DataFrame({"cell": cells, "score": tab["score"].to_numpy(), "class": tab["class"].to_numpy()}).to_csv(
        od / "scores.tsv.gz", sep="\t", index=False, float_format="%.17g")
    import xgboost
    (od / "timing.json").write_text(json.dumps({
        "engine": "pyscdblfinder", "version": getattr(pyscdblfinder, "__version__", "?"),
        "xgboost": xgboost.__version__, "python": sys.version.split()[0], "seed": seed, "seconds": elapsed}))
    print(td.name, seed, f"{elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
