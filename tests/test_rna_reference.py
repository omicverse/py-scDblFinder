"""RNA mode against stored R scDblFinder 1.16.0 reference values (no R needed).

References come from tests/data/make_rna_reference.R on 500 cells x 3,000 genes of the J293t-dm
sample of the Xi & Li (2021) doublet benchmark. Steps that do not use random numbers must match R;
the xgboost scores are random in both languages, so they are compared with R's own seed-to-seed
agreement.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr

import pyscdblfinder
from pyscdblfinder.clamulet import r_default_knn_ks
from pyscdblfinder.preprocessing import select_features
from pyscdblfinder.thresholding import doublet_thresholding

DATA = pathlib.Path(__file__).parent / "data"


def _read(name):
    return pd.read_csv(DATA / name, sep="\t", float_precision="round_trip")


@pytest.fixture(scope="module")
def counts():
    return sp.csc_matrix(scipy.io.mmread(DATA / "rna_counts.mtx.gz"))


def test_select_features(counts):
    np.testing.assert_array_equal(select_features(counts, None, n_features=1352), _read("rna_selfeatures.tsv")["index"])


def test_default_knn_ks():
    ref = _read("rna_knn_ks.tsv")
    for n, grp in ref.groupby("n"):
        assert r_default_knn_ks(int(n)) == grp["k"].astype(int).tolist()


def test_doublet_thresholding():
    d = _read("rna_thr_input.tsv").rename(columns={"include": "include.in.training"})
    d["include.in.training"] = d["include.in.training"].astype(str).str.upper() == "TRUE"
    th = _read("rna_thresholds.tsv").set_index("case")["threshold"]
    n_real = int((d["src"] == "real").sum())
    gdbr = 0.01 * n_real / 1000
    np.testing.assert_allclose(doublet_thresholding(d, stringency=0.5), th["default"], rtol=1e-9)
    np.testing.assert_allclose(doublet_thresholding(d, dbr_sd=0.3 * gdbr + 0.025, stringency=0.5), th["dbr_sd"], rtol=1e-9)
    np.testing.assert_allclose(doublet_thresholding(d, dbr=0.05, stringency=0.5), th["dbr_0.05"], rtol=1e-9)
    calls = doublet_thresholding(d, dbr_sd=0.3 * gdbr + 0.025, stringency=0.7, return_type="call")
    np.testing.assert_array_equal(calls, _read("rna_calls_07.tsv")["call"])


@pytest.fixture(scope="module")
def py_runs(counts):
    return [pyscdblfinder.sc_dbl_finder(counts, random_state=s, progress=False) for s in (1, 2, 3)]


def test_real_cell_features(py_runs):
    ref = _read("rna_seed1.tsv")
    tab = py_runs[0].table.iloc[: py_runs[0].n_real]
    for col in ("lsizes", "nfeatures", "nAbove2"):
        np.testing.assert_array_equal(tab[col].to_numpy(float), ref[col].to_numpy(float))


def test_scores_agree_as_well_as_r_seeds(py_runs):
    r = [_read(f"rna_seed{s}.tsv")["score"].to_numpy(float) for s in (1, 2, 3)]
    p = [run.table["score"].to_numpy(float)[: run.n_real] for run in py_runs]
    rr = np.median([spearmanr(r[i], r[j])[0] for i in range(3) for j in range(i + 1, 3)])
    pr = np.median([spearmanr(a, b)[0] for a in p for b in r])
    assert pr >= 0.9 * rr, (pr, rr)


def test_accuracy_matches_r(py_runs):
    from sklearn.metrics import average_precision_score
    y = (_read("rna_labels.tsv")["label"] == "doublet").to_numpy()
    r = np.mean([average_precision_score(y, _read(f"rna_seed{s}.tsv")["score"]) for s in (1, 2, 3)])
    p = np.mean([average_precision_score(y, run.table["score"].to_numpy(float)[: run.n_real]) for run in py_runs])
    assert p >= r - 0.05, (p, r)
