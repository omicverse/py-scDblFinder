"""Doublet-score thresholding, ported from R scDblFinder 1.16 (``doubletThresholding``,
``.optimThreshold``, ``.estimateHeterotypicDbRate``, ``getExpectedDoublets``, ``.gdbr``).

``method='optim'`` minimises, over the threshold, the squared relative deviation of the number of
real cells called doublets from the expected number (zero inside ``dbr ± dbr.sd``), plus the
false-negative rate on artificial doublets and the false-positive rate on real cells, each weighted
by the stringency. As in R, the minimum on [0, 1] is found with Brent's method (``stats::optimize``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

GDBR_PER1K = 0.01          # scDblFinder 1.16 .gdbr: 1% per thousand cells
_OPTIMIZE_TOL = np.finfo(float).eps ** 0.25   # stats::optimize default tol


def gdbr(d: pd.DataFrame, dbr: Optional[float] = None) -> float:
    """R ``.gdbr`` (single sample): the given rate, else 1% per thousand real cells."""
    if dbr is not None:
        return float(dbr)
    n_real = int((d["src"].values == "real").sum()) if "src" in d else int((d["type"].values == "real").sum())
    return GDBR_PER1K * n_real / 1000.0


def get_expected_doublets(clusters, dbr: Optional[float] = None, only_heterotypic: bool = True) -> pd.Series:
    """R ``getExpectedDoublets``: expected number of doublets per (heterotypic) cluster pair."""
    cl = pd.Series(np.asarray(clusters)).astype(str)
    lv = sorted(cl.unique(), key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
    n = len(cl)
    if dbr is None:
        dbr = GDBR_PER1K * n / 1000.0
    if len(lv) == 1:
        return pd.Series([n * dbr])
    cs = cl.value_counts().reindex(lv).to_numpy(float) / n
    ex = np.outer(cs, cs) * dbr * n
    i, j = np.meshgrid(np.arange(len(lv)), np.arange(len(lv)), indexing="ij")
    if only_heterotypic:
        keep = i < j
        vals = 2 * ex[keep]
        return pd.Series(vals, index=[f"{lv[a]}+{lv[b]}" for a, b in zip(i[keep], j[keep])])
    keep = i <= j
    vals = np.where(i == j, ex, ex + ex.T)[keep]
    return pd.Series(vals, index=[f"{lv[a]}+{lv[b]}" for a, b in zip(i[keep], j[keep])])


def _prop_homotypic(clusters) -> float:
    p = pd.Series(np.asarray(clusters)).value_counts(normalize=True).to_numpy(float)
    return float((p ** 2).sum() / p.sum() ** 2)


def estimate_heterotypic_dbr(d: pd.DataFrame, dbr: Optional[float] = None) -> float:
    """R ``.estimateHeterotypicDbRate`` (single sample)."""
    dbr = gdbr(d, dbr)
    src = d["src"].values if "src" in d else d["type"].values
    if "cluster" not in d or d["cluster"].isna().all():
        art = src == "artificial"
        th = art.sum() / len(d)
        prop_homo = (art & (d["score"].values < th)).sum() / art.sum()
        return dbr * (1 - prop_homo)
    real = src == "real"
    return float(get_expected_doublets(d.loc[real, "cluster"].values, dbr=dbr).sum() / real.sum())


def _brent_fmin(f, ax: float, bx: float, tol: float) -> float:
    """R ``stats::optimize`` (Brent_fmin in optimize.c)."""
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(np.finfo(float).eps)
    a, b = ax, bx
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def optim_threshold(d: pd.DataFrame, dbr: Optional[float] = None, dbr_sd: Optional[float] = None,
                    stringency: float = 0.5) -> float:
    """R ``.optimThreshold``."""
    if not 0 < stringency < 1:
        raise ValueError("`stringency` should be >0 and <1.")
    if dbr is None:
        dbr = gdbr(d, estimate_heterotypic_dbr(d))
    dbr = np.atleast_1d(float(dbr))
    if dbr_sd is not None:
        dbr = np.array([max(0.0, dbr[0] - dbr_sd), min(1.0, dbr[0] + dbr_sd)])
    src = d["src"].values if "src" in d else d["type"].values
    is_real = d["type"].values == "real"
    score = d["score"].to_numpy(float)
    expected = dbr * (src == "real").sum() + 1
    incl = d["include.in.training"].to_numpy(bool) if "include.in.training" in d else np.ones(len(d), bool)
    clusters = d["cluster"].values if "cluster" in d else np.ones(len(d))
    e_fn = 0.0
    if len(pd.unique(clusters)) > 1:
        names = d.index.astype(str)
        e_fn = names.str.startswith("rDbl.").sum() * _prop_homotypic(clusters[src == "real"])
    n_dbl = (~is_real).sum()
    ti, si = is_real[incl], score[incl]

    def totfn(x):
        called = 1 + (score >= x)[is_real].sum()                        # .prop.dev
        if len(expected) > 1 and expected.min() < called < expected.max():
            dev = 0.0
        else:
            dev = float(np.min(np.abs(called - expected) / expected))
        fnr = max(0.0, ((~is_real) & (score < x)).sum() - e_fn) / n_dbl     # .FNR
        fpr = (ti & (si >= x)).sum() / ti.sum() if ti.size else 0.0         # .FPR
        return dev ** 2 + 2 * (1 - stringency) * fnr + fpr * 2 * stringency

    return float(_brent_fmin(totfn, 0.0, 1.0, _OPTIMIZE_TOL))


def doublet_thresholding(
    d: pd.DataFrame,
    *,
    dbr: Optional[float] = None,
    dbr_sd: Optional[float] = None,
    stringency: float = 0.5,
    p: float = 0.1,
    method: str = "auto",
    return_type: str = "threshold",
    dbr_per1k: Optional[float] = None,
) -> np.ndarray | float:
    """R ``doubletThresholding`` (single sample).

    ``return_type='threshold'`` returns the threshold; ``'call'`` returns
    ``"doublet"`` where ``score > threshold`` (as R), else ``"singlet"``.
    ``dbr_per1k`` is accepted for backward compatibility: with ``dbr=None`` it sets
    ``dbr = dbr_per1k * n_real / 1000``.
    """
    if dbr is None and dbr_per1k is not None:
        n_real = int((d["src"].values == "real").sum()) if "src" in d else int((d["type"].values == "real").sum())
        dbr = dbr_per1k * n_real / 1000.0
    if "src" not in d:
        d = d.assign(src=d["type"].values)
    if dbr_sd is None:
        dbr_sd = 0.4 * gdbr(d, dbr)
    het = estimate_heterotypic_dbr(d, dbr)
    if method == "auto":
        method = "optim" if {"type", "score"} <= set(d.columns) else "dbr"
    if method == "optim":
        th = optim_threshold(d, dbr=gdbr(d, het), dbr_sd=dbr_sd, stringency=stringency)
        if return_type == "threshold":
            return th
        return np.where(d["score"].to_numpy(float) > th, "doublet", "singlet")
    real = d[d["src"].values == "real"]
    if method == "dbr":
        th = float(np.quantile(real["score"].to_numpy(float), 1 - het))
        if return_type == "threshold":
            return th
        return np.where(real["score"].to_numpy(float) > th, "doublet", "singlet")
    if method == "griffiths":
        from scipy.stats import norm
        s = real["score"].to_numpy(float)
        med = np.median(s)
        dev = s - med
        mad = np.median(dev[dev > 0]) * 1.4826
        if return_type == "threshold":
            return float(norm.isf(p, loc=med, scale=mad))
        return np.where(norm.sf(s, loc=med, scale=mad) < p, "doublet", "singlet")
    raise ValueError(f"Unknown method '{method}'")
