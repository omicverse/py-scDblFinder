"""xgboost training loop + iterative doublet scoring.

Mirrors R's ``.scDblscore`` with ``scoreType='xgb'``: train a gradient-
boosted binary classifier (real vs artificial doublet), predict scores
for all cells, iteratively remove likely-doublet real cells from the
training set, and retrain. Returns final per-cell score.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _progress_iter(iterable, total=None, desc="", enabled=True):
    """Wrap an iterable in a tqdm progress bar when available and ``enabled``."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


DEFAULT_EXCLUDE_COLS = {
    "mostLikelyOrigin", "originAmbiguous", "distanceToNearestDoublet",
    "type", "src", "distanceToNearest", "class", "nearestClass",
    "cluster", "sample", "expected", "include.in.training", "observed",
}


def _default_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DEFAULT_EXCLUDE_COLS]


def _xgb_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    nrounds: float | int = 0.25,
    max_depth: int = 4,
    eta: float = 0.3,
    metric: str = "logloss",
    subsample: float = 0.75,
    nfold: int = 5,
    nthreads: int = 0,
    tree_method: str = "exact",
    device: str = "cpu",
    random_state: int = 0,
):
    """R ``.xgbtrain`` (scDblFinder 1.16).

    With ``nrounds <= 1`` the number of rounds comes from a 5-fold (stratified, as R's
    ``xgb.cv``) cross-validation of up to 200 rounds with early stopping after 2: for
    ``nrounds == 0`` the best round, otherwise the first round whose mean test error is within
    ``nrounds`` standard deviations of the best. ``subsample`` is used in the cross-validation
    only, as in R. ``tree_method='exact'`` as in R; ``'hist'`` is faster on very large data.
    """
    import xgboost as xgb

    params = {
        "objective": "binary:logistic",
        "eval_metric": metric,
        "max_depth": int(max_depth),
        "learning_rate": float(eta),
        "tree_method": tree_method,
        "nthread": _resolve_nthreads(nthreads),
        "verbosity": 0,
        "seed": int(random_state),
    }
    resolved_device = _resolve_xgb_device(device)
    if resolved_device == "cuda":
        params["device"] = "cuda"
    dtrain = xgb.DMatrix(X, label=y.astype(np.float32))

    def _fit(p):
        n = nrounds
        if n is None:
            n = 0
        if float(n) <= 1.0:
            cv = xgb.cv({**p, "subsample": float(subsample)}, dtrain, num_boost_round=200, nfold=nfold,
                        stratified=True, early_stopping_rounds=2, seed=int(random_state), verbose_eval=False)
            mean_col = next(c for c in cv.columns if c.startswith("test") and c.endswith("-mean"))
            std_col = mean_col.replace("-mean", "-std")
            mean = cv[mean_col].to_numpy(float)
            best = int(np.argmin(mean))                       # 0-based best iteration
            if float(n) == 0:
                n = best + 1
            else:
                ac = mean[best] + float(n) * cv[std_col].to_numpy(float)[best]
                n = int(np.flatnonzero(mean <= ac)[0]) + 1
        return xgb.train(p, dtrain, num_boost_round=int(n))

    try:
        return _fit(params)
    except xgb.core.XGBoostError:
        # e.g. a non-GPU xgboost build with device='cuda' — retry on CPU.
        params.pop("device", None)
        return _fit(params)


def _resolve_nthreads(nthreads: int) -> int:
    """``0`` means "all CPUs this process may use". xgboost's own ``0`` counts
    every core of the machine, which oversubscribes badly inside a container
    or a cluster job limited to a few CPUs."""
    if int(nthreads) > 0:
        return int(nthreads)
    try:
        import os
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        import os
        return max(1, os.cpu_count() or 1)


def _resolve_xgb_device(device: str) -> str:
    """'auto' → 'cuda' only when torch reports a GPU (xgboost still validated at
    train time with a CPU fallback); otherwise 'cpu'."""
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def scDbl_score(
    d: pd.DataFrame,
    *,
    add_vals: Optional[np.ndarray] = None,
    features: Optional[list[str]] = None,
    nrounds: float | int = 0.25,
    max_depth: int = 4,
    iter: int = 3,
    dbr: Optional[float] = None,
    dbr_sd: Optional[float] = None,
    dbr_per1k: Optional[float] = None,
    unident_th: float = 0.2,
    filter_unidentifiable: bool = True,
    metric: str = "logloss",
    eta: float = 0.3,
    tree_method: str = "exact",
    nthreads: int = 1,
    random_state: int = 0,
    verbose: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """R ``.scDblscore`` (``scoreType='xgb'``, single sample).

    ``d`` has ``type`` ("real"/"doublet"), ``src`` ("real"/"artificial") and the features of
    ``evaluate_knn``, ``cxds_score``, ``lsizes``, ``nfeatures`` and ``nAbove2``. The score starts
    as the mean of ``cxds_score`` and the scaled largest-k neighbour ratio; each iteration then
    trains xgboost without the real cells called doublets (stringency 0.7), the artificial
    doublets scoring below ``unident_th`` and cells not ``include.in.training``, and rescores all
    cells. Adds ``score`` and updates ``include.in.training``.
    """
    from .thresholding import doublet_thresholding, gdbr as _gdbr

    d = d.copy()
    if "include.in.training" not in d:
        d["include.in.training"] = True
    if dbr is None and dbr_per1k is not None:
        dbr = dbr_per1k * int((d["src"].values == "real").sum()) / 1000.0
    if dbr_sd is None:
        dbr_sd = 0.3 * _gdbr(d, dbr) + 0.025
    feat_cols = _default_features(d) if features is None else [c for c in features if c in d.columns]
    X = d[feat_cols].astype(float).values
    if add_vals is not None:
        X = np.concatenate([X, np.asarray(add_vals, dtype=float)], axis=1)
    y = (d["type"].values == "doublet").astype(np.int32)

    ratio_cols = [c for c in d.columns if c.startswith("ratio.k")]
    rat = d[ratio_cols[-1]].astype(float).values
    d["score"] = (d["cxds_score"].astype(float).values + rat / rat.max()) / 2.0

    is_real = d["type"].values == "real"
    exclude = np.zeros(len(d), dtype=bool)
    for it in _progress_iter(range(int(iter)), total=int(iter), desc="scDblFinder: training", enabled=progress):
        called = doublet_thresholding(d, dbr=dbr, dbr_sd=dbr_sd, stringency=0.7, return_type="call") == "doublet"
        exclude = ((is_real & called)
                   | (~is_real & (d["score"].values < unident_th) & filter_unidentifiable)
                   | ~d["include.in.training"].to_numpy(bool))
        if verbose:
            print(f"[scDblscore] iter={it}, {int(exclude.sum())} cells excluded from training.")
        try:
            bst = _xgb_train(X[~exclude], y[~exclude], nrounds=nrounds, max_depth=max_depth, eta=eta,
                             metric=metric, tree_method=tree_method, nthreads=nthreads,
                             random_state=random_state + it)
            d["score"] = bst.predict(_xgb_dmatrix(X))
        except Exception as exc:  # pragma: no cover - R keeps the previous score on error
            if verbose:
                print(f"[scDblscore] xgboost failed: {exc}; keeping previous score")
    d.loc[exclude, "include.in.training"] = False
    return d


def _xgb_dmatrix(X):
    import xgboost as xgb
    return xgb.DMatrix(X)
