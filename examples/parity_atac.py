"""ATAC-mode parity run against R scDblFinder (see PARITY_ATAC.md).

Usage::

    python examples/parity_atac.py <fixture_dir> <r_out_dir> <py_out_dir> [step ...]

Steps: ``amulet``, ``fromcounts``, ``scdbl``, ``clamulet``, ``compare``.
The R side is produced by ``tests/r_reference_driver_atac.R``.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pyscdblfinder as psd

SEEDS = (1, 2, 3)


def _cells(fx: Path) -> pd.DataFrame:
    return pd.read_csv(fx / "cells.tsv", sep="\t", header=None,
                       names=["barcode", "call"])


def _peaks(fx: Path):
    m = sp.csc_matrix(scipy.io.mmread(fx / "peaks.mtx"))
    bed = pd.read_csv(fx / "peaks.bed", sep="\t", header=None,
                      names=["chrom", "start", "end"])
    cells = (fx / "peaks_cells.txt").read_text().split()
    return m, bed, cells


def run_amulet(fx: Path, out: Path):
    t = {}
    t0 = time.time()
    psd.amulet(str(fx / "frag_subset.tsv.gz")).to_csv(out / "amulet_full.tsv", sep="\t")
    t["full_s"] = time.time() - t0
    t0 = time.time()
    psd.amulet(str(fx / "tabix" / "frag_subset.tsv.gz")).to_csv(
        out / "amulet_tabix.tsv", sep="\t")
    t["tabix_s"] = time.time() - t0
    ex = list(zip(["M", "chrM", "MT", "X", "Y", "chrX", "chrY"], [0] * 7, [10 ** 8] * 7))
    ex += [("chr19", 999999, 3000000), ("chr1", 49999999, 50500000)]
    psd.amulet(str(fx / "frag_subset.tsv.gz"), barcodes=list(_cells(fx)["barcode"]),
               regions_to_exclude=ex).to_csv(out / "amulet_barcodes.tsv", sep="\t")
    (out / "amulet_time.json").write_text(json.dumps(t))


def run_fromcounts(fx: Path, out: Path):
    m, bed, cells = _peaks(fx)
    w = (bed["end"] - bed["start"]).to_numpy()
    qa = psd.amulet_from_counts(m)                          # plain matrix
    qb = psd.amulet_from_counts(m, widths=w, max_width=1000)  # exclude=NULL
    pd.DataFrame({"barcode": cells, "q_matrix": qa, "q_w1000": qb}).to_csv(
        out / "amulet_from_counts.tsv", sep="\t", index=False)


def run_scdbl(fx: Path, out: Path):
    m, bed, cells = _peaks(fx)
    for s in SEEDS:
        t0 = time.time()
        res = psd.sc_dbl_finder(m, aggregate_features=True, n_features=25,
                                processing="normFeatures", random_state=s,
                                progress=False)
        el = time.time() - t0
        real = res.real_cells()
        pd.DataFrame({"barcode": cells, "score": real["score"].to_numpy(),
                      "class": real["class"].to_numpy()}).to_csv(
            out / f"scdbl_seed{s}.tsv", sep="\t", index=False)
        (out / f"scdbl_seed{s}.json").write_text(json.dumps({"seconds": el}))


def run_clamulet(fx: Path, out: Path):
    bcs = list(_cells(fx)["barcode"])
    for s in SEEDS:
        t0 = time.time()
        d = psd.clamulet(str(fx / "frag_subset.tsv.gz"), barcodes=bcs, random_state=s)
        el = time.time() - t0
        d.to_csv(out / f"clamulet_seed{s}.tsv", sep="\t")
        (out / f"clamulet_seed{s}.json").write_text(json.dumps({"seconds": el}))


# ---------------------------------------------------------------------------
def _auprc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / den)) if a.size else 0.0


def compare(fx: Path, rdir: Path, pdir: Path) -> dict:
    from scipy.stats import spearmanr
    res: dict = {}
    cells = _cells(fx).set_index("barcode")
    # G1 amulet
    g1 = {}
    for f in ("amulet_full.tsv", "amulet_tabix.tsv", "amulet_barcodes.tsv"):
        r = pd.read_csv(rdir / f, sep="\t", index_col=0, float_precision="round_trip")
        p = pd.read_csv(pdir / f, sep="\t", index_col=0, float_precision="round_trip")
        same = set(r.index) == set(p.index)
        p = p.reindex(r.index)
        ints = {c: bool((r[c].to_numpy() == p[c].to_numpy()).all())
                for c in ("nFrags", "uniqFrags", "nAbove2", "total.nAbove2")
                if c in r.columns}
        g1[f] = {"n_barcodes_R": int(len(r)), "same_barcodes": bool(same),
                 "integer_columns_equal": ints,
                 "p_max_rel": _rel(p["p.value"], r["p.value"]),
                 "q_max_rel": _rel(p["q.value"], r["q.value"])}
        g1[f]["pass"] = bool(same and all(ints.values())
                             and g1[f]["p_max_rel"] <= 1e-10
                             and g1[f]["q_max_rel"] <= 1e-10)
    res["G1_amulet"] = g1
    # G2 amulet_from_counts
    if (rdir / "amulet_from_counts.tsv").exists():
        R = pd.read_csv(rdir / "amulet_from_counts.tsv", sep="\t", index_col=0,
                        float_precision="round_trip")
        P = pd.read_csv(pdir / "amulet_from_counts.tsv", sep="\t", index_col=0,
                        float_precision="round_trip").reindex(R.index)
        g2 = {}
        for c in ("q_matrix", "q_w1000"):
            r, p = R[c].to_numpy(float), P[c].to_numpy(float)
            both_nan = np.isnan(r) & np.isnan(p)
            d = float(np.max(np.abs(p[~both_nan] - r[~both_nan]))) if (~both_nan).any() else 0.0
            g2[c] = {"max_abs": d, "n_nan_R": int(np.isnan(r).sum()),
                     "n_nan_py": int(np.isnan(p).sum()),
                     "pass": bool(d <= 1e-10 and (np.isnan(r) == np.isnan(p)).all())}
        res["G2_amulet_from_counts"] = g2

    def stochastic(prefix, score_col="score"):
        R = {s: pd.read_csv(rdir / f"{prefix}_seed{s}.tsv", sep="\t", index_col=0)
             for s in SEEDS if (rdir / f"{prefix}_seed{s}.tsv").exists()}
        P = {s: pd.read_csv(pdir / f"{prefix}_seed{s}.tsv", sep="\t", index_col=0)
             for s in SEEDS if (pdir / f"{prefix}_seed{s}.tsv").exists()}
        idx = cells.index
        y = (cells["call"] == "DBL").astype(int).to_numpy()
        rs = {s: d[score_col].reindex(idx).to_numpy(float) for s, d in R.items()}
        ps = {s: d[score_col].reindex(idx).to_numpy(float) for s, d in P.items()}
        if "class" in next(iter(R.values())).columns:
            n_call = int(np.mean([(d["class"] == "doublet").sum() for d in R.values()]))
        else:                         # clamulet has no calls: use the truth rate
            n_call = int(y.sum())

        def top(v):
            return set(np.argsort(-v, kind="stable")[:n_call])

        def jac(a, b):
            A, B = top(a), top(b)
            return len(A & B) / len(A | B)
        rr_sp = [spearmanr(rs[a], rs[b])[0] for a, b in combinations(rs, 2)]
        pr_sp = [spearmanr(ps[a], rs[b])[0] for a, b in product(ps, rs)]
        rr_j = [jac(rs[a], rs[b]) for a, b in combinations(rs, 2)]
        pr_j = [jac(ps[a], rs[b]) for a, b in product(ps, rs)]
        auprc_r = [_auprc(y, v) for v in rs.values()]
        auprc_p = [_auprc(y, v) for v in ps.values()]
        out = {
            "n_called": n_call,
            "spearman_RR_median": float(np.median(rr_sp)),
            "spearman_PR_median": float(np.median(pr_sp)),
            "jaccard_RR_median": float(np.median(rr_j)),
            "jaccard_PR_median": float(np.median(pr_j)),
            "auprc_R": auprc_r, "auprc_py": auprc_p,
            "auroc_R": [_auroc(y, v) for v in rs.values()],
            "auroc_py": [_auroc(y, v) for v in ps.values()],
            "seconds_R": [json.loads((rdir / f"{prefix}_seed{s}.json").read_text())["seconds"] for s in R],
            "seconds_py": [json.loads((pdir / f"{prefix}_seed{s}.json").read_text())["seconds"] for s in P],
        }
        out["a_pass"] = out["spearman_PR_median"] >= 0.9 * out["spearman_RR_median"]
        out["b_pass"] = out["jaccard_PR_median"] >= 0.9 * out["jaccard_RR_median"]
        out["c_pass"] = float(np.mean(auprc_p)) >= float(np.mean(auprc_r)) - 0.05
        return out
    if (rdir / "scdbl_seed1.tsv").exists() and (pdir / "scdbl_seed1.tsv").exists():
        res["G3_scdblfinder_atac"] = stochastic("scdbl")
    if (rdir / "clamulet_seed1.tsv").exists() and (pdir / "clamulet_seed1.tsv").exists():
        res["G4_clamulet"] = stochastic("clamulet")
    # informational: AMULET and Fisher combination accuracy on the 3,000 cells
    try:
        y = (cells["call"] == "DBL").astype(int)
        am = pd.read_csv(pdir / "amulet_barcodes.tsv", sep="\t", index_col=0).reindex(cells.index)
        info = {"amulet_auprc": _auprc(y, -am["p.value"].to_numpy()),
                "amulet_auroc": _auroc(y, -am["p.value"].to_numpy())}
        if (pdir / "scdbl_seed1.tsv").exists():
            sc = pd.read_csv(pdir / "scdbl_seed1.tsv", sep="\t", index_col=0)["score"].reindex(cells.index)
            comb = psd.fisher_combine(1 - sc.to_numpy(), am["p.value"].to_numpy())
            info["fisher_auprc"] = _auprc(y, -comb)
            info["fisher_auroc"] = _auroc(y, -comb)
        res["info_accuracy_py"] = info
    except FileNotFoundError:
        pass
    return res


def main(argv):
    fx, rdir, pdir = map(Path, argv[1:4])
    pdir.mkdir(parents=True, exist_ok=True)
    steps = argv[4:] or ["amulet", "fromcounts", "scdbl", "clamulet", "compare"]
    for st in steps:
        if st == "compare":
            r = compare(fx, rdir, pdir)
            (pdir / "parity_results.json").write_text(json.dumps(r, indent=1, default=float))
            print(json.dumps(r, indent=1, default=float))
        else:
            globals()[f"run_{st}"](fx, pdir)
            print(f"[parity_atac] {st} done", flush=True)


if __name__ == "__main__":
    main(sys.argv)
