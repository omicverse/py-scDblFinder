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
    tree_method: str = "hist",
    device: str = "cpu",
    random_state: int = 0,
):
    """Train the binary xgboost classifier.

    If ``nrounds <= 1``, use 5-fold CV to pick a round count, then subtract
    ``nrounds * sd(CV error)`` from the best round (matches R behavior).
    Otherwise use ``nrounds`` directly.

    Performance
    -----------
    ``tree_method='hist'`` (xgboost's own modern default) is 5-10x faster than
    the old ``'exact'`` at 10^5+ cells with near-identical accuracy, and is the
    single biggest speedup for large-atlas doublet detection. ``device='auto'``
    trains on the GPU when a CUDA-capable xgboost build is present (falling back
    to CPU otherwise); ``nthreads=0`` uses all cores.
    """
    import xgboost as xgb

    params = {
        "objective": "binary:logistic",
        "eval_metric": metric,
        "max_depth": int(max_depth),
        "learning_rate": float(eta),
        "subsample": float(subsample),
        "tree_method": tree_method,
        "nthread": int(nthreads),
        "verbosity": 0,
        "seed": int(random_state),
    }
    resolved_device = _resolve_xgb_device(device)
    if resolved_device == "cuda":
        params["device"] = "cuda"
    dtrain = xgb.DMatrix(X, label=y.astype(np.float32))

    def _fit(p):
        if nrounds is None or float(nrounds) <= 1.0:
            cv = xgb.cv(
                p, dtrain, num_boost_round=100,
                nfold=min(nfold, max(3, X.shape[0] // 10)),
                early_stopping_rounds=10, seed=int(random_state), verbose_eval=False,
            )
            err_col = next(c for c in cv.columns if c.endswith("-mean") and "test" in c)
            sd_col = err_col.replace("-mean", "-std")
            best = int(cv[err_col].idxmin())
            best -= int(round(float(nrounds) * cv[sd_col].iloc[best]))
            n = max(5, best)
        else:
            n = int(nrounds)
        return xgb.train(p, dtrain, num_boost_round=n)

    try:
        return _fit(params)
    except xgb.core.XGBoostError:
        # e.g. a non-GPU xgboost build with device='cuda' — retry on CPU.
        params.pop("device", None)
        return _fit(params)


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
    dbr_per1k: float = 0.008,
    unident_th: float = 0.1,
    metric: str = "logloss",
    random_state: int = 0,
    verbose: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Port of R ``.scDblscore`` (``scoreType='xgb'``).

    ``d`` must have columns ``type`` ("real"/"doublet"), ``src`` ("real"/"artificial"),
    and the numeric features from ``evaluate_knn`` / ``cxds_score``.
    Adds a ``score`` column (predicted doublet probability) to ``d``.
    """
    import xgboost as xgb

    d = d.copy()
    if features is None:
        feat_cols = _default_features(d)
    else:
        feat_cols = [c for c in features if c in d.columns]
    X = d[feat_cols].astype(float).values
    if add_vals is not None:
        X = np.concatenate([X, np.asarray(add_vals, dtype=float)], axis=1)

    y = (d["type"].values == "doublet").astype(np.int32)

    # Initial score — average of cxds_score and normalized ratio (R line:
    # d$score <- (d$cxds_score + d[[ratio]]/max(d[[ratio]]))/2)
    ratio_cols = [c for c in d.columns if c.startswith("ratio.k")]
    ratio_col = ratio_cols[-1] if ratio_cols else None
    if ratio_col is not None and "cxds_score" in d.columns:
        rat = d[ratio_col].astype(float).values
        rat = rat / (rat.max() if rat.max() > 0 else 1.0)
        d["score"] = (d["cxds_score"].astype(float).values + rat) / 2.0
    elif ratio_col is not None:
        d["score"] = d[ratio_col].astype(float).values
    else:
        d["score"] = 0.5

    n_real = int((d["type"].values == "real").sum())
    n_dbl = int((d["type"].values == "doublet").sum())
    if dbr is None:
        # Expected doublet rate from dbr.per1k: fraction ≈ dbr_per1k * n_real / 1000
        dbr = dbr_per1k * n_real / 1000.0
    # Deviation budget
    for it in _progress_iter(range(int(iter)), total=int(iter),
                             desc="scDblFinder: training", enabled=progress):
        # Exclude cells that look like doublets (top-dbr-ish fraction) from training
        from .thresholding import doublet_thresholding
        exclude_real = np.where(
            (d["type"].values == "real") &
            (doublet_thresholding(d, dbr=dbr, stringency=0.7, return_type="call") == "doublet")
        )[0]
        # Cap the excluded fraction so we don't starve the training set
        if exclude_real.size > n_real // 3:
            sort_idx = np.argsort(-d["score"].values)
            exclude_real = [i for i in sort_idx if d["type"].values[i] == "real"][:int(0.2 * n_real)]
            exclude_real = np.asarray(exclude_real, dtype=int)

        exclude_dbl = np.where(
            (d["type"].values == "doublet") & (d["score"].values < unident_th)
        )[0]
        if exclude_dbl.size > n_dbl // 4:
            sort_idx = np.argsort(d["score"].values)
            exclude_dbl = [i for i in sort_idx if d["type"].values[i] == "doublet"][:int(0.1 * n_dbl)]
            exclude_dbl = np.asarray(exclude_dbl, dtype=int)

        include = np.ones(len(d), dtype=bool)
        include[exclude_real] = False
        include[exclude_dbl] = False

        if verbose:
            print(f"[scDblscore] iter={it}  excluding {(~include).sum()} cells from training")

        try:
            bst = _xgb_train(
                X[include], y[include],
                nrounds=nrounds, max_depth=max_depth, metric=metric,
                random_state=random_state,
            )
            dmat_all = xgb.DMatrix(X)
            d["score"] = bst.predict(dmat_all)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[scDblscore] xgboost failed: {exc}; keeping previous score")

    return d
