"""R-parity tests for pyscdblfinder.

Because scDblFinder's xgboost RNG state isn't portable between R and
Python, we don't require bit-for-bit score equality. Instead we check:

  * ranking correlation between R scores and Python scores on the same
    synthetic dataset (Spearman ρ > 0.5 — very lenient bar)
  * call-level overlap: at least 70% of cells get the same singlet /
    doublet call

Skipped if Rscript + scDblFinder aren't installed (both typical on a
Python-only dev box).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).parent
DRIVER = HERE / "r_reference_driver.R"


def _find_rscript() -> str | None:
    for c in ("/scratch/users/steorra/env/CMAP/bin/Rscript",
              shutil.which("Rscript") or ""):
        if c and Path(c).exists():
            return c
    return None


@pytest.fixture(scope="module")
def r_ref_dir(tmp_path_factory) -> Path:
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript not found")
    outdir = tmp_path_factory.mktemp("r_ref")
    env = os.environ.copy()
    cmap_extra = Path("/scratch/users/steorra/env/CMAP/R_extra_libs")
    if cmap_extra.is_dir():
        env["R_LIBS_USER"] = str(cmap_extra)
    proc = subprocess.run(
        [rscript, str(DRIVER), str(outdir)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        pytest.skip(
            "R reference driver failed (likely missing scDblFinder in "
            f"the env):\n{proc.stderr[-1500:]}"
        )
    return outdir


def test_py_scores_rank_correlate_with_R(r_ref_dir: Path):
    """On the same mockDoubletSCE input, py and R scores should rank the
    same cells high (Spearman ρ > 0.5)."""
    from scipy.stats import spearmanr
    from pyscdblfinder import sc_dbl_finder

    counts = pd.read_csv(r_ref_dir / "counts.tsv", sep="\t", index_col=0)
    r_result = pd.read_csv(r_ref_dir / "r_result.tsv", sep="\t").set_index("cell")
    truth = pd.read_csv(r_ref_dir / "truth.tsv", sep="\t").set_index("cell")

    dbr = (truth["truth"] == "doublet").mean()
    res = sc_dbl_finder(
        counts.values.astype(np.float64),
        dbr=float(dbr), iter=2, random_state=0,
    )
    py_scores = res.real_cells()["score"].values
    r_scores = r_result["score"].reindex(counts.columns).values.astype(float)
    rho, _ = spearmanr(py_scores, r_scores)
    assert rho > 0.5, f"Spearman rho={rho:.3f} between py and R scores is too low"


def test_classifications_overlap_with_R(r_ref_dir: Path):
    """At least 70% of cells should get the same singlet/doublet call."""
    from pyscdblfinder import sc_dbl_finder

    counts = pd.read_csv(r_ref_dir / "counts.tsv", sep="\t", index_col=0)
    r_result = pd.read_csv(r_ref_dir / "r_result.tsv", sep="\t").set_index("cell")
    truth = pd.read_csv(r_ref_dir / "truth.tsv", sep="\t").set_index("cell")

    dbr = (truth["truth"] == "doublet").mean()
    res = sc_dbl_finder(
        counts.values.astype(np.float64),
        dbr=float(dbr), iter=2, random_state=0,
    )
    py_class = res.real_cells()["class"].values
    r_class  = r_result["class"].reindex(counts.columns).values
    overlap = (py_class == r_class).mean()
    assert overlap > 0.70, f"Only {overlap:.1%} of calls match R"
