"""UMAP of one benchmark dataset coloured by annotated doublets, R and Python scores, and calls.

Usage: python 06_umap.py <runs_dir> <tasks_dir> <out_dir> [dataset]
Uses seed 1 of each package (scores from 02/03); the embedding is a standard scanpy pipeline.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
NAME = sys.argv[4] if len(sys.argv) > 4 else "pbmc-2ctrl-dm"
INK, INK2 = "#0b0b0b", "#52514e"
SIZE = {"R only": 2.2, "Python only": 2.2}
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "pdf.fonttype": 42, "savefig.dpi": 300,
    "legend.frameon": False,
})

td = TASKS / NAME
counts = sp.csr_matrix(scipy.io.mmread(td / "counts.mtx.gz").T)          # cells x genes
ad = sc.AnnData(counts)
ad.obs["annotated"] = pd.read_csv(td / "labels.tsv", header=None)[0].to_numpy()
for eng, lab in [("scDblFinder", "R"), ("pyscdblfinder", "py")]:
    s = pd.read_csv(RUNS / eng / NAME / "seed1" / "scores.tsv.gz", sep="\t")
    ad.obs[f"score_{lab}"] = s["score"].to_numpy()
    ad.obs[f"call_{lab}"] = (s["class"] == "doublet").to_numpy()

sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
sc.pp.pca(ad, n_comps=30, random_state=0)
sc.pp.neighbors(ad, n_neighbors=15, random_state=0)
sc.tl.umap(ad, random_state=0)
xy = ad.obsm["X_umap"]

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), gridspec_kw=dict(wspace=0.06))


def base(ax, title, letter):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    ax.text(-0.04, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=INK)
    ax.set_aspect("equal")


def cats(ax, labels, order, colors, title, letter):
    for name, col in zip(order, colors):
        m = labels == name
        ax.scatter(xy[m, 0], xy[m, 1], s=SIZE.get(name, 0.7), c=col, lw=0, rasterized=True,
                   label=f"{name} ({m.sum():,})")
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=5.5, handletextpad=0.1,
                    columnspacing=0.8)
    for h in leg.legend_handles:
        h.set_sizes([10])
    base(ax, title, letter)


ann = ad.obs["annotated"].to_numpy()
cats(axes[0], ann, ["singlet", "doublet"], ["#d9d8d3", "#2a2a2a"], "Annotated (demuxlet)", "a")

for ax, lab, title, letter in [(axes[1], "R", "Score, scDblFinder (R)", "b"),
                               (axes[2], "py", f"Score, pyscdblfinder", "c")]:
    v = ad.obs[f"score_{lab}"].to_numpy()
    o = np.argsort(v)
    im = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=0.6, cmap="magma_r", vmin=0, vmax=1, lw=0, rasterized=True)
    base(ax, title, letter)
cax = axes[1].inset_axes([0.55, -0.16, 0.9, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
rho = spearmanr(ad.obs.score_R, ad.obs.score_py)[0]
cax.set_title(f"doublet score (Spearman, Python vs R: {rho:.2f})", fontsize=5.5, color=INK2, pad=2)

r, p = ad.obs.call_R.to_numpy(), ad.obs.call_py.to_numpy()
call = np.select([r & p, r & ~p, ~r & p], ["both", "R only", "Python only"], "neither")
cats(axes[3], call, ["neither", "both", "R only", "Python only"], ["#d9d8d3", "#2a2a2a", "#eb6834", "#2a78d6"],
     "Doublet calls", "d")

fig.savefig(OUT / "fig2.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig2.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
both, ronly, ponly = (call == "both").sum(), (call == "R only").sum(), (call == "Python only").sum()
macros = {"umapName": NAME, "umapCells": f"{ad.n_obs:,}", "umapAnnot": f"{(ann == 'doublet').sum():,}",
          "umapSpearman": f"{rho:.2f}", "umapBoth": f"{both:,}", "umapROnly": f"{ronly:,}",
          "umapPyOnly": f"{ponly:,}"}
# append to numbers.tex written by 05_figures.py (run 05 first)
tex = OUT / "numbers.tex"
keep = [ln for ln in tex.read_text().splitlines() if not ln.startswith("\\newcommand{\\umap")]
tex.write_text("\n".join(keep + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]) + "\n")
print(tex.read_text())
