"""Doublet-score thresholding — simplified port of R ``doubletThresholding``.

scDblFinder's ``method='optim'`` picks the threshold that minimizes a
weighted misclassification rate plus the squared deviation from the
expected doublet rate (``dbr``). This port implements the same objective
with a 1-D grid search; the R package uses ``stats::optimize`` but the
objective is flat enough at typical resolutions that a grid search is
equivalent and simpler to audit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _optim_threshold(
    d: pd.DataFrame,
    *,
    dbr: Optional[float],
    dbr_sd: float,
    stringency: float,
    grid: np.ndarray,
) -> float:
    """Return the threshold on ``d['score']`` that minimizes the combined loss."""
    y = (d["type"].values == "doublet").astype(int)
    real_mask = d["type"].values == "real"
    scores = d["score"].values.astype(float)
    best_th = float(np.median(grid))
    best_loss = np.inf
    for th in grid:
        # False negatives: artificial doublets called as singlet
        call = scores >= th
        # Misclassification: weighted FP+FN
        fn = ((call == 0) & (y == 1)).sum()
        fp = ((call == 1) & (y == 0)).sum()
        # Deviation from expected doublet rate on real cells
        if dbr is not None:
            observed_rate = call[real_mask].mean()
            # Zone of no penalty within [dbr - dbr_sd, dbr + dbr_sd]
            dev = max(0.0, abs(observed_rate - dbr) - dbr_sd)
        else:
            dev = 0.0
        n_art = max(1, int(y.sum()))
        n_real = max(1, int(real_mask.sum()))
        loss = (stringency * fp / n_real) + ((1 - stringency) * fn / n_art) + dev
        if loss < best_loss:
            best_loss = loss
            best_th = th
    return float(best_th)


def doublet_thresholding(
    d: pd.DataFrame,
    *,
    dbr: Optional[float] = None,
    dbr_sd: Optional[float] = None,
    dbr_per1k: float = 0.008,
    stringency: float = 0.5,
    method: str = "optim",
    return_type: str = "call",
) -> np.ndarray | float:
    """Threshold scores into doublet/singlet calls.

    Matches the R signature closely. ``return_type="call"`` returns a
    string array ("doublet"/"singlet"), ``"threshold"`` returns a float.
    """
    if "type" not in d.columns or "score" not in d.columns:
        raise ValueError("d must have columns 'type' and 'score'")

    if dbr is None:
        n_real = int((d["type"].values == "real").sum())
        dbr = dbr_per1k * n_real / 1000.0
    if dbr_sd is None:
        dbr_sd = 0.4 * dbr
    score_min = float(np.min(d["score"]))
    score_max = float(np.max(d["score"]))
    grid = np.linspace(score_min, score_max, 200)
    th = _optim_threshold(d, dbr=dbr, dbr_sd=dbr_sd, stringency=stringency, grid=grid)

    if return_type == "threshold":
        return th
    return np.where(d["score"].values >= th, "doublet", "singlet")
