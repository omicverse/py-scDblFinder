"""Figures, tables and the numbers quoted in the pyscdblfinder technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir>
Compares scDblFinder 1.16.0 and pyscdblfinder on the Xi & Li (2021) real doublet benchmark, two seeds each.
Because both use random artificial doublets and xgboost, agreement is measured against R's own
seed-to-seed agreement.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY, C_TRUE = "#eb6834", "#2a78d6", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "scDblFinder", "py": "pyscdblfinder"}
SEEDS = [1, 2]


def load(engine, task, seed):
    return pd.read_csv(RUNS / ENG[engine] / task / f"seed{seed}" / "scores.tsv.gz", sep="\t")


def secs(engine, task, seed):
    return json.load(open(RUNS / ENG[engine] / task / f"seed{seed}" / "timing.json"))["seconds"]


def jaccard(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


tasks = sorted(p.name for p in TASKS.iterdir() if (p / "labels.tsv").exists())
rows = []
for t in tasks:
    y = (pd.read_csv(TASKS / t / "labels.tsv", header=None)[0] == "doublet").to_numpy()
    s = {(e, k): load(e, t, k) for e in ENG for k in SEEDS}
    for d in s.values():
        assert len(d) == len(y)
    sc = {key: d["score"].to_numpy() for key, d in s.items()}
    call = {key: (d["class"] == "doublet").to_numpy() for key, d in s.items()}
    rr = [("R", 1), ("R", 2)]
    pp = [("py", 1), ("py", 2)]
    pr = list(itertools.product(pp, rr))
    row = dict(dataset=t, cells=len(y), doublet_rate=y.mean())
    row["spearman_RR"] = spearmanr(sc[rr[0]], sc[rr[1]])[0]
    row["spearman_pp"] = spearmanr(sc[pp[0]], sc[pp[1]])[0]
    row["spearman_pR"] = np.mean([spearmanr(sc[a], sc[b])[0] for a, b in pr])
    row["jaccard_RR"] = jaccard(call[rr[0]], call[rr[1]])
    row["jaccard_pR"] = np.mean([jaccard(call[a], call[b]) for a, b in pr])
    for e in ENG:
        keys = [(e, k) for k in SEEDS]
        auprc = [average_precision_score(y, sc[k]) for k in keys]
        auroc = [roc_auc_score(y, sc[k]) for k in keys]
        row[f"auprc_{e}"], row[f"auprc_{e}_min"], row[f"auprc_{e}_max"] = np.mean(auprc), min(auprc), max(auprc)
        row[f"auroc_{e}"] = np.mean(auroc)
        row[f"called_{e}"] = np.mean([call[k].mean() for k in keys])
        row[f"sec_{e}"] = np.mean([secs(e, t, k) for k in SEEDS])
    rows.append(row)
tab = pd.DataFrame(rows).sort_values("cells").reset_index(drop=True)
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 4.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.38)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


top = gs[0].subgridspec(1, 4, width_ratios=[0.55, 1, 1, 1], wspace=0.38)
logo = top[0].subgridspec(2, 1, height_ratios=[1.3, 0.6], hspace=0.05)
for k, f in enumerate(["pyscdblfinder_logo.png", "omicverse_logo.png"]):
    axl = fig.add_subplot(logo[k])
    axl.imshow(plt.imread(ASSETS / f))
    axl.set_axis_off()

axa = fig.add_subplot(top[1])
identity(axa, 0, 1)
axa.errorbar(tab.auprc_R, tab.auprc_py,
             xerr=[tab.auprc_R - tab.auprc_R_min, tab.auprc_R_max - tab.auprc_R],
             yerr=[tab.auprc_py - tab.auprc_py_min, tab.auprc_py_max - tab.auprc_py],
             fmt="o", ms=3, color=C_PY, ecolor="#9ec5f4", elinewidth=0.6, mec="white", mew=0.3)
axa.set_title("AUPRC")
axa.set_xlabel("scDblFinder (R)")
axa.set_ylabel("pyscdblfinder")
panel(axa, "a")

for ax, xcol, ycol, title, lab in [
    (fig.add_subplot(top[2]), "spearman_RR", "spearman_pR", "Score Spearman", "b"),
    (fig.add_subplot(top[3]), "jaccard_RR", "jaccard_pR", "Doublet-call Jaccard", "c"),
]:
    identity(ax, 0.5, 1)
    ax.scatter(tab[xcol], tab[ycol], s=10, color=C_PY, edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("R seed 1 vs R seed 2")
    ax.set_ylabel("Python vs R")
    panel(ax, lab)

bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.22)
axd = fig.add_subplot(bot[0])
axd.scatter(tab.cells, tab.sec_R, s=10, color=C_R, edgecolor="white", linewidth=0.3, label="scDblFinder 1.16.0 (R)")
axd.scatter(tab.cells, tab.sec_py, s=10, color=C_PY, edgecolor="white", linewidth=0.3, label="pyscdblfinder")
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xlabel("cells")
axd.set_ylabel("wall time, 1 thread (s)")
axd.set_title("Run time")
axd.grid(color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
panel(axd, "d", dx=-0.3)

axe = fig.add_subplot(bot[1])
x = np.arange(len(tab))
w = 0.28
axe.bar(x - w, tab.doublet_rate * 100, w * 0.9, color="#c7c6c0", label="annotated")
axe.bar(x, tab.called_R * 100, w * 0.9, color=C_R, label="called, scDblFinder (R)")
axe.bar(x + w, tab.called_py * 100, w * 0.9, color=C_PY, label="called, pyscdblfinder")
axe.set_xticks(x, [f"{d} ({c / 1000:.1f}k)" for d, c in zip(tab.dataset, tab.cells)], rotation=50, ha="right",
               rotation_mode="anchor", fontsize=5)
axe.set_xlim(-0.6, len(tab) - 0.4)
axe.set_ylabel("doublets (% of cells)")
axe.set_title("Doublets per dataset")
axe.grid(axis="y", color=GRID, lw=0.5)
axe.set_axisbelow(True)
axe.legend(fontsize=5.5, loc="upper left", ncol=3, handlelength=1, columnspacing=1)
axe.set_ylim(0, 44)
panel(axe, "e", dx=-0.1)
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
ratio_sp = tab.spearman_pR / tab.spearman_RR
ratio_j = tab.jaccard_pR / tab.jaccard_RR
speed = tab.sec_R / tab.sec_py
big = tab.iloc[tab.cells.idxmax()]
jlow = tab.iloc[ratio_j.idxmin()]
called_diff = (tab.called_py - tab.called_R).abs() * 100
macros = {
    "nDatasets": len(tab), "nCellsMin": f"{int(tab.cells.min()):,}", "nCellsMax": f"{int(tab.cells.max()):,}",
    "nCellsTotal": f"{int(tab.cells.sum()):,}",
    "rateMin": f"{tab.doublet_rate.min() * 100:.0f}", "rateMax": f"{tab.doublet_rate.max() * 100:.0f}",
    "auprcR": f"{tab.auprc_R.mean():.3f}", "auprcPy": f"{tab.auprc_py.mean():.3f}",
    "aurocR": f"{tab.auroc_R.mean():.3f}", "aurocPy": f"{tab.auroc_py.mean():.3f}",
    "auprcDiffMax": f"{(tab.auprc_py - tab.auprc_R).abs().max():.3f}",
    "auprcSeedRangeMaxR": f"{(tab.auprc_R_max - tab.auprc_R_min).max():.3f}",
    "spRRmed": f"{tab.spearman_RR.median():.3f}", "spPRmed": f"{tab.spearman_pR.median():.3f}",
    "spRatioMed": f"{ratio_sp.median():.2f}", "spRatioMin": f"{ratio_sp.min():.2f}",
    "jRRmed": f"{tab.jaccard_RR.median():.3f}", "jPRmed": f"{tab.jaccard_pR.median():.3f}",
    "jRatioMed": f"{ratio_j.median():.2f}", "jRatioMin": f"{ratio_j.min():.2f}",
    "speedMin": f"{speed.min():.1f}", "speedMax": f"{speed.max():.1f}", "speedMed": f"{speed.median():.1f}",
    "tBigR": f"{big.sec_R:.0f}", "tBigPy": f"{big.sec_py:.0f}", "bigName": big.dataset,
    "bigCells": f"{int(big.cells):,}",
    "calledDiffMax": f"{called_diff.max():.1f}", "calledDiffMed": f"{called_diff.median():.1f}",
    "calledMinPct": f"{min(tab.called_R.min(), tab.called_py.min()) * 100:.0f}",
    "calledMaxPct": f"{max(tab.called_R.max(), tab.called_py.max()) * 100:.0f}",
    "jLowName": jlow.dataset, "jLowCells": f"{int(jlow.cells):,}", "jLowPR": f"{jlow.jaccard_pR:.2f}",
    "jLowRR": f"{jlow.jaccard_RR:.2f}", "jLowCalled": f"{jlow.called_R * jlow.cells:.0f}",
    "spLowName": tab.iloc[ratio_sp.idxmin()].dataset,
    "aurocDiffMax": f"{(tab.auroc_py - tab.auroc_R).abs().max():.3f}",
}
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
print(tab[["dataset", "cells", "doublet_rate", "auprc_R", "auprc_py", "spearman_RR", "spearman_pR", "jaccard_RR",
           "jaccard_pR", "called_R", "called_py", "sec_R", "sec_py"]].round(3).to_string())
