"""clamulet — port of scDblFinder's classification-powered AMULET (``R/atac.R``).

clamulet generates artificial doublets by summing the fragment *coverages* of
two cells, so the AMULET statistic (loci covered by more than two fragments)
can be computed for artificial doublets too and used, with kNN features on
aggregated window counts, to train a classifier. The scDblFinder vignette
notes that it is slow, memory hungry and outperformed by the simple p-value
combination of scDblFinder and AMULET (see :func:`fisher_combine`); it is
ported for completeness.

Kept R behaviour:

* ``min_count`` of :func:`clamulet` is not passed on by R to the table
  builder, which always uses 0.001; the port does the same.
* artificial-doublet loci are dropped when they overlap *any* merged
  real-cell locus (R passes all reduced ranges as the exclusion set);
* if no real cell scores above 0.6 in an iteration, R's ``m[-integer(0),]``
  selects zero rows, training fails and the previous score is kept; the port
  keeps the previous score in that case as well.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .atac import aggregate_features
from .fragments import (_as_fragment_table, _as_regions, _filter_fragments,
                        _high_overlap_mask, _iter_chromosomes, _loci_ge3,
                        _r_round)
from .knn_features import _compute_knn


def r_default_knn_ks(n: int, k=None) -> list[int]:
    """R ``.defaultKnnKs``."""
    if k is not None:
        ks = [int(k)] if np.isscalar(k) else [int(v) for v in k]
        return [v for v in ks if v <= np.ceil(n / 2)]
    kmax = max(int(np.ceil(np.sqrt(n / 2))), 25)
    out = []
    for v in (3, 10, 15, 20, 25, 50, kmax):
        if v <= kmax and v not in out:
            out.append(v)
    return out


def xgbtrain_r(X: np.ndarray, y: np.ndarray, nrounds=None, max_depth: int = 6,
               nfold: int = 5, eta: float = 0.3, tree_method: str = "exact",
               subsample: float = 0.75, nthreads: int = 1,
               metric: str = "logloss", random_state: int = 0):
    """R ``.xgbtrain``: cross-validated round selection (``nrounds`` NULL or
    <= 1), then a final fit with early stopping on the training set.

    ``eta`` defaults to xgboost's 0.3, as in scDblFinder <= 1.16 (later
    versions pass ``learning_rate=1``)."""
    import xgboost as xgb
    params = {"objective": "binary:logistic", "eval_metric": metric,
              "max_depth": int(max_depth), "learning_rate": float(eta),
              "subsample": float(subsample), "tree_method": tree_method,
              "nthread": int(nthreads), "seed": int(random_state),
              "verbosity": 0}
    dtrain = xgb.DMatrix(X, label=y.astype(np.float32))
    if nrounds is None:
        nrounds = 0
    if nrounds <= 1:
        cv = xgb.cv(params, dtrain, num_boost_round=200, nfold=nfold,
                    early_stopping_rounds=2, seed=int(random_state),
                    verbose_eval=False)
        mcol = next(c for c in cv.columns if c.startswith("test") and c.endswith("-mean"))
        best = int(np.argmin(cv[mcol].to_numpy())) + 1
        if nrounds == 0:
            nrounds = best
        else:
            scol = mcol.replace("-mean", "-std")
            ac = cv[mcol].iloc[best - 1] + nrounds * cv[scol].iloc[best - 1]
            nrounds = int(np.min(np.flatnonzero(cv[mcol].to_numpy() <= ac)) + 1)
    bst = xgb.train(params, dtrain, num_boost_round=int(nrounds),
                    evals=[(dtrain, "train")], early_stopping_rounds=2,
                    verbose_eval=False)
    return bst


def _predict(bst, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb
    it = getattr(bst, "best_iteration", None)
    rng = (0, it + 1) if it is not None else (0, 0)
    return bst.predict(xgb.DMatrix(X), iteration_range=rng)


def evaluate_knn_r(X: np.ndarray, is_doublet: np.ndarray, k) -> pd.DataFrame:
    """The ``weighted`` and ``ratio.k*`` columns of R ``.evaluateKNN``.

    R weights the j-th of K neighbours by ``sqrt(K - j) / distance``
    (j = 1..K, so the K-th weight is 0) and replaces zero distances by the
    smallest positive nearest-neighbour distance.
    """
    ks = sorted(int(v) for v in k)
    K = max(ks)
    dist, idx = _compute_knn(X, K + 1)
    dist, idx = np.asarray(dist, float)[:, 1:], idx[:, 1:]
    if np.any(dist == 0):
        pos = dist[:, 0][dist[:, 0] > 0]
        dist[dist == 0] = pos.min()
    t = is_doublet[idx].astype(float)
    dw = np.sqrt(K - np.arange(1, K + 1))[np.newaxis, :] / dist
    dw = dw / dw.sum(axis=1, keepdims=True)
    out = {"weighted": (t * dw).sum(axis=1)}
    for ki in k:
        out[f"ratio.k{int(ki)}"] = t[:, : int(ki)].sum(axis=1) / int(ki)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# coverage helpers
# ---------------------------------------------------------------------------
def _segments(code: np.ndarray, s: np.ndarray, e: np.ndarray, w: np.ndarray):
    """Piecewise-constant per-barcode coverage: ``(code, start, end, cov)``
    for every segment with cov > 0."""
    pos = np.concatenate([s, e])
    who = np.concatenate([code, code])
    delta = np.concatenate([w, -w])
    o = np.lexsort((pos, who))
    pos, who, delta = pos[o], who[o], delta[o]
    new = np.ones(pos.size, dtype=bool)
    new[1:] = (pos[1:] != pos[:-1]) | (who[1:] != who[:-1])
    g = np.flatnonzero(new)
    pos, who = pos[g], who[g]
    cov = np.cumsum(np.add.reduceat(delta, g))
    keep = cov[:-1] > 0
    keep &= who[1:] == who[:-1]
    i = np.flatnonzero(keep)
    return who[i], pos[i], pos[i + 1], cov[i]


def _runs(starts: np.ndarray, ends: np.ndarray, lower: int):
    """Runs where the number of intervals covering a base is >= ``lower``."""
    if starts.size == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z
    code = np.zeros(starts.size, dtype=np.int64)
    _, rs, re = _loci_ge3(code, starts, ends, np.ones(starts.size, np.int64),
                          lower=lower)
    return rs, re


def _reduce(starts: np.ndarray, ends: np.ndarray, min_gapwidth: int):
    """IRanges ``reduce``: merge ranges separated by fewer than
    ``min_gapwidth`` positions (half-open: gap = next_start - end)."""
    if starts.size == 0:
        return starts, ends
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    new = np.ones(s.size, dtype=bool)
    new[1:] = (s[1:] - np.maximum.accumulate(e)[:-1]) >= min_gapwidth
    gid = np.cumsum(new) - 1
    rs = s[new]
    re = np.maximum.reduceat(e, np.flatnonzero(new))
    return rs, re


def _window_max(seg_code, seg_s, seg_e, seg_cov, win_s, win_e, n_cells):
    """Max coverage of each cell within each window -> windows x cells."""
    if win_s.size == 0 or seg_code.size == 0:
        return sp.csr_matrix((win_s.size, n_cells))
    lo = np.searchsorted(win_e, seg_s, side="right")
    hi = np.searchsorted(win_s, seg_e, side="left")
    n = np.maximum(hi - lo, 0)
    rep = np.repeat(np.arange(seg_code.size), n)
    widx = np.repeat(lo, n) + (np.arange(rep.size) - np.repeat(np.cumsum(n) - n, n))
    cells = seg_code[rep]
    vals = seg_cov[rep]
    key = widx.astype(np.int64) * n_cells + cells
    o = np.lexsort((-vals, key))
    key, vals = key[o], vals[o]
    first = np.ones(key.size, dtype=bool)
    first[1:] = key[1:] != key[:-1]
    key, vals = key[first], vals[first]
    return sp.csr_matrix((vals.astype(np.float64), (key // n_cells, key % n_cells)),
                         shape=(win_s.size, n_cells))


def _overlaps_any_sorted(s, e, rs, re):
    """Half-open overlap of each [s, e) with any of the (sorted, disjoint)
    ranges [rs, re)."""
    if rs.size == 0 or s.size == 0:
        return np.zeros(s.size, dtype=bool)
    cm = np.maximum.accumulate(re)
    k = np.searchsorted(rs, e, side="left")
    hit = np.zeros(s.size, dtype=bool)
    has = k > 0
    hit[has] = cm[k[has] - 1] > s[has]
    return hit


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def clamulet(
    x,
    artificial_doublets: Optional[int] = None,
    iter: int = 2,
    k=None,
    min_count: float = 0.001,
    max_n: int = 500,
    nfeatures: int = 25,
    max_depth: int = 5,
    threshold: float = 0.75,
    return_all: bool = False,
    barcodes: Optional[Sequence[str]] = None,
    regions_to_exclude="default",
    min_frags: float = 500,
    unique_frags: bool = True,
    max_frag_size: int = 1000,
    random_state: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Port of R ``clamulet()``.

    Parameters follow R (``artificialDoublets``, ``iter``, ``k``,
    ``minCount``, ``maxN``, ``nfeatures``, ``max_depth``, ``threshold``,
    ``returnAll``) plus the :func:`get_fragment_overlaps` arguments.
    ``threshold`` and ``min_count`` are accepted but unused, as in R.

    Returns a DataFrame indexed by barcode with ``total``, ``nAbove2``,
    ``total.nAbove2``, the kNN features, ``include.in.training`` and
    ``score`` (higher = more likely doublet).
    """
    rng = np.random.default_rng(random_state)
    reg = _as_regions(regions_to_exclude)

    # 1) fragments, filtered as getFragmentOverlaps(fullInMemory=TRUE)
    if isinstance(x, pd.DataFrame):
        chunks = [g for _, g in _as_fragment_table(x).groupby("chrom", sort=False)]
    else:
        chunks = [g for _, g in _iter_chromosomes(x)]
    chunks = [_filter_fragments(g, reg, max_frag_size) for g in chunks]
    chunks = [g for g in chunks if len(g)]
    allbc = pd.concat([g["barcode"] for g in chunks])
    uniq = allbc.value_counts()
    n_tot = int(uniq.sum())
    mf = _r_round(min_frags * n_tot) if 0 < min_frags < 1 else min_frags
    if barcodes is None:
        cells = np.array(sorted(uniq.index[uniq >= mf]))
    else:
        cells = np.array(sorted(set(map(str, barcodes)) & set(uniq.index)))
    ncell = cells.size
    if verbose:
        print(f"[clamulet] {ncell} barcodes")

    per_chrom = []
    for g in chunks:
        g = g[g["barcode"].isin(cells)]
        if not len(g):
            continue
        code = pd.Categorical(g["barcode"], categories=cells).codes.astype(np.int64)
        w = (np.ones(len(g), np.int64) if unique_frags
             else g["count"].to_numpy(np.int64))
        per_chrom.append((g["chrom"].iloc[0], code,
                          g["start"].to_numpy(np.int64), g["end"].to_numpy(np.int64), w))

    # 2) windows from (sampled) covered regions
    mc = max(int(_r_round(0.001 * ncell)), 2)      # R always uses 0.001
    reg_cells, reg_chrom, reg_s, reg_e = [], [], [], []
    segs = []
    for ci, (chrom, code, s, e, w) in enumerate(per_chrom):
        sc, ss, se, scov = _segments(code, s, e, w)
        segs.append((sc, ss, se, scov))
        rc, rs, re = _loci_ge3(code, s, e, w, lower=1)
        reg_cells.append(rc); reg_chrom.append(np.full(rc.size, ci))
        reg_s.append(rs); reg_e.append(re)
    rc = np.concatenate(reg_cells); rch = np.concatenate(reg_chrom)
    rs = np.concatenate(reg_s); re = np.concatenate(reg_e)
    keep = np.ones(rc.size, dtype=bool)
    counts_per_cell = np.bincount(rc, minlength=ncell)
    for c in np.flatnonzero(counts_per_cell > max_n):
        idx = np.flatnonzero(rc == c)
        drop = np.setdiff1d(idx, rng.choice(idx, size=max_n, replace=False))
        keep[drop] = False
    rc, rch, rs, re = rc[keep], rch[keep], rs[keep], re[keep]
    windows = []
    for ci in range(len(per_chrom)):
        m = rch == ci
        ws, we = _runs(rs[m], re[m], lower=mc)
        ws, we = _reduce(ws, we, min_gapwidth=100)
        windows.append((ws, we))

    # 3) window counts (max coverage per cell)
    mats = [_window_max(*segs[ci], ws, we, ncell)
            for ci, (ws, we) in enumerate(windows)]
    counts = sp.vstack(mats).tocsr()
    if verbose:
        print(f"[clamulet] {counts.shape[0]} windows")
    agg = aggregate_features(counts, dims_use=range(1, min(ncell - 1, 10) + 1),
                             k=min(counts.shape[0], nfeatures),
                             random_state=random_state)

    # 4) artificial doublets
    n_ad = ncell if artificial_doublets is None else int(artificial_doublets)
    v = rng.integers(0, ncell, size=2 * int(np.floor(n_ad * 2.2)))
    comb = v.reshape(2, -1).T                       # matrix(..., ncol=2)
    comb = comb[comb[:, 0] != comb[:, 1]]
    _, first = np.unique(comb, axis=0, return_index=True)
    comb = comb[np.sort(first)][:n_ad]
    acounts = agg[:, comb[:, 0]] + agg[:, comb[:, 1]]
    n_art = comb.shape[0]

    total = np.concatenate([agg.sum(axis=0), acounts.sum(axis=0)])
    nab = np.zeros(ncell + n_art, dtype=np.int64)
    tnab = np.zeros(ncell + n_art, dtype=np.int64)

    # 5) loci for real cells, high-overlap filter, exclusion set
    lb, ls, le, lg = [], [], [], []
    for ci, (chrom, code, s, e, w) in enumerate(per_chrom):
        b, a, z = _loci_ge3(code, s, e, w, lower=3)
        lb.append(b); ls.append(a); le.append(z); lg.append(np.full(b.size, ci))
    lb = np.concatenate(lb); ls = np.concatenate(ls)
    le = np.concatenate(le); lg = np.concatenate(lg)
    tnab[:ncell] = np.bincount(lb, minlength=ncell)
    rm, site = _high_overlap_mask(ls, le, lg)
    nab[:ncell] = np.bincount(lb[~rm], minlength=ncell)
    excl = {}
    for ci in range(len(per_chrom)):
        m = lg == ci
        if m.any():
            es, ee = _reduce(ls[m], le[m], min_gapwidth=0)
            # reduce(min.gapwidth=0) keeps abutting ranges separate
            excl[ci] = (es, ee)

    # 6) loci for artificial doublets (summed coverages)
    for ci, (chrom, code, s, e, w) in enumerate(per_chrom):
        order = np.argsort(code, kind="stable")
        code_s, s_s, e_s, w_s = code[order], s[order], e[order], w[order]
        bounds = np.searchsorted(code_s, np.arange(ncell + 1))
        parts_c, parts_s, parts_e, parts_w = [], [], [], []
        for j, (a, b) in enumerate(comb):
            for c in (a, b):
                lo, hi = bounds[c], bounds[c + 1]
                if hi > lo:
                    parts_c.append(np.full(hi - lo, j, dtype=np.int64))
                    parts_s.append(s_s[lo:hi]); parts_e.append(e_s[lo:hi])
                    parts_w.append(w_s[lo:hi])
        if not parts_c:
            continue
        ab, as_, ae = _loci_ge3(np.concatenate(parts_c), np.concatenate(parts_s),
                                np.concatenate(parts_e), np.concatenate(parts_w), 3)
        tnab[ncell:] += np.bincount(ab, minlength=n_art)
        if ci in excl:
            hit = _overlaps_any_sorted(as_, ae, *excl[ci])
            ab = ab[~hit]
        nab[ncell:] += np.bincount(ab, minlength=n_art)

    labels = np.array(["real"] * ncell + ["doublet"] * n_art)
    names = list(cells) + [f"artificial {i + 1}" for i in range(n_art)]
    d = pd.DataFrame({"total": total, "nAbove2": nab, "total.nAbove2": tnab},
                     index=names)

    # 7) normalised, per-cell standardised meta-feature matrix and kNN
    X = 10000.0 * np.concatenate([agg, acounts], axis=1) / total[np.newaxis, :]
    X = X.T
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)
    ks = r_default_knn_ks(ncell, k)
    kd = evaluate_knn_r(X, (labels == "doublet").astype(int), ks)
    kd.index = d.index
    d = pd.concat([d, kd], axis=1)
    M = d.to_numpy(dtype=np.float64)
    y = (labels == "doublet").astype(int)

    # 8) iterative training
    d["include.in.training"] = True
    d["type"] = labels
    d["score"] = _predict(xgbtrain_r(M, y, max_depth=3, random_state=random_state), M)
    w = np.empty(0, dtype=int)
    for it in range(int(iter)):
        w = np.flatnonzero((labels == "real") & (d["score"].to_numpy() > 0.6))
        if verbose:
            print(f"[clamulet] iter={it}, {w.size} cells excluded from training.")
        if w.size == 0:
            continue      # R: m[-integer(0),] is empty -> error -> keep score
        keep = np.ones(M.shape[0], dtype=bool)
        keep[w] = False
        try:
            fit = xgbtrain_r(M[keep], y[keep], max_depth=max_depth, nthreads=1,
                             random_state=random_state)
            d["score"] = _predict(fit, M)
        except Exception:
            pass
    d.iloc[w, d.columns.get_loc("include.in.training")] = False
    if not return_all:
        d = d[d["type"] == "real"].drop(columns="type")
    return d
