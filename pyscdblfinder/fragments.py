"""Fragment-file doublet statistics — port of scDblFinder's
``getFragmentOverlaps`` / ``amulet`` / ``amuletFromCounts``.

AMULET (Thibodeau et al., *Genome Biology* 2021) flags droplets with
unexpectedly many genomic loci covered by more than two fragments: a diploid
nucleus contributes at most two. scDblFinder re-implements it in R
(``R/getFragmentOverlaps.R``, ``R/atac.R``); this module follows that R code,
including its edge cases, so that counts and p-values match it exactly:

* fragments are read as BED (0-based start, exclusive end) and their width is
  ``end - start``; fragments wider than ``max_frag_size`` are dropped before
  region exclusion;
* the default excluded regions are ``M, chrM, MT, X, Y, chrX, chrY`` over
  ``[1, 1e8]`` (1-based), exactly as in R — so on a human genome the part of
  chrX beyond 100 Mb is *not* excluded, as in R;
* a locus is a maximal run of positions covered by >= 3 fragments of the same
  barcode; abutting runs at different depths are one locus;
* ``remove_high_overlap_sites`` merges *overlapping* (not merely abutting)
  loci across barcodes, as ``reduce(min.gapwidth=0)``, and drops merged sites
  whose number of contributing loci has a Poisson upper-tail p < 0.01;
* a tabix-indexed file (``.tbi`` next to it) is processed per chromosome, as R
  does, which changes two details: the per-chromosome ``minFrags`` becomes
  ``round(1e-5 * n_fragments_on_chromosome)`` and the high-overlap filter uses
  the per-chromosome mean. ``full_in_memory=True`` (or a non-indexed file)
  uses the whole-file semantics.
* in that per-chromosome mode R reads each chromosome with
  ``which=GRanges(chrom, IRanges(1, 1e8))``, so fragments starting beyond
  100 Mb (e.g. on human chr1-chr12) are **not read at all**. The port
  mirrors this for parity (``tabix_window=1e8``); pass ``tabix_window=None``
  or ``full_in_memory=True`` to use the whole chromosome.

The file is streamed in chunks; only per-barcode counts and the (few) loci
covered by >= 3 fragments are kept in memory.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson

DEFAULT_EXCLUDE_CHROMS = ("M", "chrM", "MT", "X", "Y", "chrX", "chrY")


# ---------------------------------------------------------------------------
# small R-equivalent helpers
# ---------------------------------------------------------------------------
def p_adjust_bh(p) -> np.ndarray:
    """R ``p.adjust(p, method="BH")``."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if n == 0:
        return p.copy()
    out = np.full(n, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    i = np.arange(m, 0, -1, dtype=np.float64)
    o = np.argsort(-pv, kind="stable")
    ro = np.empty(m, dtype=np.int64)
    ro[o] = np.arange(m)
    q = np.minimum(1.0, np.minimum.accumulate(m / i * pv[o]))
    out[ok] = q[ro]
    return out


def p_adjust_holm(p) -> np.ndarray:
    """R ``p.adjust(p)`` with its default ``method="holm"``."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.size, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    o = np.argsort(pv, kind="stable")
    ro = np.empty(m, dtype=np.int64)
    ro[o] = np.arange(m)
    i = np.arange(1, m + 1, dtype=np.float64)
    q = np.minimum(1.0, np.maximum.accumulate((m - i + 1) * pv[o]))
    out[ok] = q[ro]
    return out


def ppois_upper(q, lam) -> np.ndarray:
    """R ``ppois(q, lam, lower.tail=FALSE)`` = P(X > q)."""
    return poisson.sf(np.asarray(q, dtype=np.float64), lam)


def _r_round(x: float) -> float:
    """R ``round`` (IEC 60559: half to even), same as numpy."""
    return float(np.round(x))


# ---------------------------------------------------------------------------
# regions to exclude
# ---------------------------------------------------------------------------
def _default_regions_to_exclude() -> pd.DataFrame:
    # GRanges(c("M",...), IRanges(1L, width=10^8)) -> 1-based [1, 1e8]
    # -> 0-based half-open [0, 1e8)
    return pd.DataFrame({"chrom": list(DEFAULT_EXCLUDE_CHROMS),
                         "start": 0, "end": 10 ** 8})


def _as_regions(regions) -> Optional[pd.DataFrame]:
    """Normalise ``regions_to_exclude`` to a 0-based half-open DataFrame.

    Accepts ``None``, ``"default"``, a BED path, a DataFrame with
    ``chrom/start/end`` (0-based half-open, BED convention) or a list of
    ``(chrom, start, end)`` tuples in the same convention.
    """
    if regions is None:
        return None
    if isinstance(regions, str) and regions == "default":
        return _default_regions_to_exclude()
    if isinstance(regions, (str, os.PathLike)):
        df = pd.read_csv(regions, sep="\t", header=None, comment="#",
                         usecols=[0, 1, 2], names=["chrom", "start", "end"],
                         dtype={"chrom": str})
        df = df[~df["chrom"].str.startswith(("track", "browser"))]
        return df.astype({"start": np.int64, "end": np.int64})
    if isinstance(regions, pd.DataFrame):
        df = regions.rename(columns=str.lower)[["chrom", "start", "end"]].copy()
    else:
        df = pd.DataFrame(list(regions), columns=["chrom", "start", "end"])
    df["chrom"] = df["chrom"].astype(str)
    return df.astype({"start": np.int64, "end": np.int64})


def _overlaps_any(starts: np.ndarray, ends: np.ndarray,
                  reg: Optional[pd.DataFrame]) -> np.ndarray:
    """Whether each half-open interval overlaps any region (same chromosome).

    Intervals are 0-based half-open; R's ``overlapsAny`` on the 1-based
    ranges ``[s+1, e]`` is equivalent to ``s < re and rs < e``.
    """
    hit = np.zeros(starts.size, dtype=bool)
    if reg is None or len(reg) == 0 or starts.size == 0:
        return hit
    rs = reg["start"].to_numpy(np.int64)
    re = reg["end"].to_numpy(np.int64)
    o = np.argsort(rs, kind="stable")
    rs, re = rs[o], re[o]
    # running max of region ends lets a single searchsorted decide overlap
    cummax_end = np.maximum.accumulate(re)
    # regions with rs < e are candidates; the furthest-reaching one decides
    k = np.searchsorted(rs, ends, side="left")  # number of regions with rs < e
    has = k > 0
    hit[has] = cummax_end[k[has] - 1] > starts[has]
    return hit


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _open_text(path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "rt")


def _iter_chromosomes(path, chunksize: int = 5_000_000
                      ) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(chrom, fragments)`` for a fragments file sorted by chromosome.

    If the file is not sorted (a chromosome reappears later), the pieces are
    concatenated in memory before being yielded.
    """
    cols = ["chrom", "start", "end", "barcode", "count"]
    pending: dict[str, list[pd.DataFrame]] = {}
    order: list[str] = []
    with _open_text(path) as fh:            # 4-column files have no count
        first = next((ln for ln in fh if ln.strip() and not ln.startswith("#")), "")
    ncol = min(len(first.rstrip("\n").split("\t")), 5)
    if ncol < 4:
        raise ValueError(f"{path}: expected >= 4 tab-separated columns")
    with _open_text(path) as fh:
        reader = pd.read_csv(
            fh, sep="\t", header=None, comment="#", usecols=list(range(ncol)),
            names=cols[:ncol], chunksize=chunksize,
            dtype={"chrom": str, "start": np.int64, "end": np.int64,
                   "barcode": str, "count": np.int64},
        )
        for chunk in reader:
            if "count" not in chunk.columns:
                chunk["count"] = 1
            for chrom, sub in chunk.groupby("chrom", sort=False):
                if chrom not in pending:
                    pending[chrom] = []
                    order.append(chrom)
                pending[chrom].append(sub)
    for chrom in order:
        yield chrom, pd.concat(pending.pop(chrom), ignore_index=True)


def _as_fragment_table(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        df = x.rename(columns=str.lower).copy()
        if "name" in df.columns and "barcode" not in df.columns:
            df = df.rename(columns={"name": "barcode"})
        if "seqnames" in df.columns and "chrom" not in df.columns:
            df = df.rename(columns={"seqnames": "chrom"})
        if "score" in df.columns and "count" not in df.columns:
            df = df.rename(columns={"score": "count"})
        if "count" not in df.columns:
            df["count"] = 1
        return df[["chrom", "start", "end", "barcode", "count"]]
    raise TypeError("`x` must be a fragments-file path or a DataFrame with "
                    "chrom/start/end/barcode[/count] (0-based, BED-like).")


def _is_tabix_indexed(path) -> bool:
    return isinstance(path, (str, os.PathLike)) and \
        os.path.exists(str(path) + ".tbi")


# ---------------------------------------------------------------------------
# per-chromosome core
# ---------------------------------------------------------------------------
@dataclass
class _ChromLoci:
    chrom: str
    barcode: np.ndarray    # int codes into the chromosome's barcode list
    start: np.ndarray      # 0-based, half-open
    end: np.ndarray


def _loci_ge3(bc: np.ndarray, starts: np.ndarray, ends: np.ndarray,
              weight: np.ndarray, lower: int = 3) -> tuple[np.ndarray, ...]:
    """Maximal runs with per-barcode coverage >= ``lower``.

    Coverage is computed per barcode code with a sweep over start/end events
    (fragments are half-open, i.e. 1-based ``[start+1, end]``).
    Returns ``(barcode_code, run_start, run_end)``.
    """
    if bc.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e
    pos = np.concatenate([starts, ends])
    who = np.concatenate([bc, bc])
    delta = np.concatenate([weight, -weight])
    o = np.lexsort((pos, who))
    pos, who, delta = pos[o], who[o], delta[o]
    # merge events at the same (barcode, position)
    new = np.ones(pos.size, dtype=bool)
    new[1:] = (pos[1:] != pos[:-1]) | (who[1:] != who[:-1])
    grp = np.flatnonzero(new)
    pos, who = pos[grp], who[grp]
    delta = np.add.reduceat(delta, grp)
    cov = np.cumsum(delta)          # per-barcode sums are 0, so no reset needed
    above = cov >= lower
    prev = np.zeros_like(above)
    prev[1:] = above[:-1]
    prev[1:] &= who[1:] == who[:-1]
    starts_idx = np.flatnonzero(above & ~prev)
    # run end: next event index after the run where coverage drops below lower
    nxt = np.ones_like(above)
    nxt[:-1] = above[1:] & (who[1:] == who[:-1])
    ends_idx = np.flatnonzero(above & ~nxt) + 1
    return who[starts_idx], pos[starts_idx], pos[ends_idx]


def _high_overlap_mask(starts: np.ndarray, ends: np.ndarray,
                       group: Optional[np.ndarray] = None,
                       pthres: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``.removeHighOverlapSites``.

    ``reduce(gr, min.gapwidth=0)`` merges loci that overlap (abutting loci are
    not merged), per chromosome given by ``group``. A merged site is dropped
    when ``ppois(n_loci, mean(n_loci), lower.tail=FALSE) < pthres``, with the
    mean taken over all merged sites passed in.

    Returns ``(remove_mask, site_id)``.
    """
    n = starts.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    if group is None:
        group = np.zeros(n, dtype=np.int64)
    o = np.lexsort((starts, group))
    big = np.int64(1) << np.int64(40)          # > any coordinate
    s = starts[o] + group[o].astype(np.int64) * big
    e = ends[o] + group[o].astype(np.int64) * big
    # a locus starts a new site unless it overlaps the furthest-reaching
    # earlier locus of the same group (the offset makes groups disjoint)
    new = np.ones(n, dtype=bool)
    new[1:] = s[1:] >= np.maximum.accumulate(e)[:-1]
    site = np.cumsum(new) - 1
    hol = np.bincount(site)
    p = ppois_upper(hol, hol.mean())
    drop_site = p < pthres
    remove = np.empty(n, dtype=bool)
    remove[o] = drop_site[site]
    site_id = np.empty(n, dtype=np.int64)
    site_id[o] = site
    return remove, site_id


def _filter_fragments(df: pd.DataFrame, reg: Optional[pd.DataFrame],
                      max_frag_size: int) -> pd.DataFrame:
    df = df[(df["end"].to_numpy() - df["start"].to_numpy()) <= max_frag_size]
    if reg is not None and len(df):
        chrom = df["chrom"].iloc[0]
        r = reg[reg["chrom"] == chrom]
        if len(r):
            hit = _overlaps_any(df["start"].to_numpy(np.int64),
                                df["end"].to_numpy(np.int64), r)
            df = df[~hit]
    return df


def _chrom_stats(df: pd.DataFrame, barcodes: Optional[set], min_frags: float,
                 unique_frags: bool, remove_high_overlap: bool,
                 per_chrom: bool):
    """Process one chromosome; returns (stats DataFrame, loci or None)."""
    n = len(df)
    if per_chrom and 0 < min_frags < 1:
        min_frags = _r_round(min_frags * n)
    counts = df["barcode"].value_counts(sort=False)   # unique fragments
    if barcodes is None:
        keep = counts[counts >= min_frags]
    else:
        keep = counts[counts.index.isin(barcodes)]
    df = df[df["barcode"].isin(keep.index)]
    if len(df) == 0:
        return None, None
    names = np.array(sorted(keep.index))
    code = pd.Categorical(df["barcode"], categories=names).codes.astype(np.int64)
    if unique_frags:
        w = np.ones(len(df), dtype=np.int64)
        nfrags = np.bincount(code, minlength=names.size)
    else:
        w = df["count"].to_numpy(np.int64)
        nfrags = np.bincount(code, weights=w, minlength=names.size).astype(np.int64)
    lb, ls, le = _loci_ge3(code, df["start"].to_numpy(np.int64),
                           df["end"].to_numpy(np.int64), w)
    stats = pd.DataFrame({
        "nFrags": nfrags,
        "uniqFrags": keep.reindex(names).to_numpy(np.int64),
        "nAbove2": 0,
        "total.nAbove2": np.bincount(lb, minlength=names.size),
    }, index=names)
    if per_chrom:
        if remove_high_overlap and lb.size:
            rm, _ = _high_overlap_mask(ls, le)
            stats["nAbove2"] = np.bincount(lb[~rm], minlength=names.size)
        else:
            stats["nAbove2"] = stats["total.nAbove2"]
        return stats, None
    return stats, _ChromLoci(df["chrom"].iloc[0], names[lb], ls, le)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def get_fragment_overlaps(
    x,
    barcodes: Optional[Sequence[str]] = None,
    regions_to_exclude="default",
    min_frags: float = 500,
    unique_frags: bool = True,
    max_frag_size: int = 1000,
    remove_high_overlap_sites: bool = True,
    full_in_memory: bool = False,
    tabix_window: Optional[int] = 10 ** 8,
    verbose: bool = False,
) -> pd.DataFrame:
    """Port of R ``getFragmentOverlaps(..., ret="stats")``.

    Parameters
    ----------
    x
        Path to a fragments file (``chrom, start, end, barcode, count``;
        gzipped or plain, optionally tabix-indexed), or a DataFrame with those
        columns in BED convention.
    barcodes
        Barcodes to keep (a list, or a path to a one-per-line text file). If
        ``None``, barcodes with at least ``min_frags`` unique fragments
        (after size filtering and exclusion) are kept.
    regions_to_exclude
        ``"default"`` (mitochondrial and sex chromosomes over [1, 1e8], as R),
        ``None``, a BED path, a DataFrame or ``(chrom, start, end)`` tuples
        (0-based half-open). As in AMULET, adding repeats is recommended.
    min_frags, unique_frags, max_frag_size, remove_high_overlap_sites,
    full_in_memory
        As in R. A tabix-indexed path is processed per chromosome unless
        ``full_in_memory=True``; see the module docstring for the
        resulting differences, which R has as well.
    tabix_window
        Per-chromosome mode only: like R, read only fragments overlapping
        ``[1, tabix_window]`` (1-based) of each chromosome. ``None`` reads
        whole chromosomes (recommended for analysis; differs from R).

    Returns
    -------
    DataFrame indexed by barcode with ``nFrags``, ``uniqFrags``, ``nAbove2``
    (loci covered by >= 3 fragments, after the high-overlap filter) and
    ``total.nAbove2`` (before it).
    """
    if barcodes is not None:
        if isinstance(barcodes, (str, os.PathLike)) and os.path.exists(barcodes):
            with open(barcodes) as fh:
                barcodes = [ln.strip() for ln in fh if ln.strip()]
        barcodes = set(map(str, barcodes))
    reg = _as_regions(regions_to_exclude)

    if isinstance(x, (str, os.PathLike)):
        if not os.path.exists(x):
            raise FileNotFoundError(x)
        per_chrom = (not full_in_memory) and _is_tabix_indexed(x)
        chrom_iter: Iterable = _iter_chromosomes(x)
    else:
        per_chrom = False
        df_all = _as_fragment_table(x)
        chrom_iter = df_all.groupby("chrom", sort=False)

    if per_chrom:
        parts = []
        for chrom, df in chrom_iter:
            if verbose:
                print(f"[getFragmentOverlaps] {chrom}", flush=True)
            if tabix_window is not None:
                df = df[df["start"].to_numpy() < tabix_window]
            df = _filter_fragments(df, reg, max_frag_size)
            if len(df) == 0:
                continue
            st, _ = _chrom_stats(df, barcodes, 0.00001, unique_frags,
                                 remove_high_overlap_sites, per_chrom=True)
            if st is not None:
                parts.append(st)
        cols = ["nFrags", "uniqFrags", "nAbove2", "total.nAbove2"]
        if not parts:
            return pd.DataFrame(columns=cols[:3])
        tot = pd.concat(parts).groupby(level=0, sort=True)[cols].sum()
        if barcodes is None:
            keep = tot.index[tot["nFrags"] >= min_frags]
            return tot.loc[keep].astype(np.int64)
        # R returns every requested barcode, with zeros where absent
        return tot.reindex(sorted(barcodes)).fillna(0).astype(np.int64)

    # whole-file semantics: per-barcode stats need the global min_frags, the
    # high-overlap filter needs all loci; loci per barcode are independent of
    # other barcodes, so they are computed chromosome by chromosome.
    frames, loci = [], []
    n_total = 0
    kept_bc_counts: list[pd.Series] = []
    filtered = []
    for chrom, df in chrom_iter:
        df = _filter_fragments(df, reg, max_frag_size)
        if len(df):
            filtered.append(df)
            n_total += len(df)
            kept_bc_counts.append(df["barcode"].value_counts(sort=False))
    if not filtered:
        return pd.DataFrame(columns=["nFrags", "uniqFrags", "nAbove2"])
    uniq = pd.concat(kept_bc_counts).groupby(level=0).sum()
    mf = min_frags
    if 0 < mf < 1:
        mf = _r_round(mf * n_total)
    if barcodes is None:
        keep = set(uniq.index[uniq >= mf])
    else:
        keep = set(uniq.index) & barcodes
    for df in filtered:
        st, lc = _chrom_stats(df, keep, 0, unique_frags,
                              remove_high_overlap_sites, per_chrom=False)
        if st is not None:
            frames.append(st)
            loci.append(lc)
    cols = ["nFrags", "uniqFrags", "nAbove2", "total.nAbove2"]
    tot = pd.concat(frames).groupby(level=0, sort=True)[cols].sum()
    tot["uniqFrags"] = uniq.reindex(tot.index).to_numpy(np.int64)
    lb = np.concatenate([lc.barcode for lc in loci])
    ls = np.concatenate([lc.start for lc in loci])
    le = np.concatenate([lc.end for lc in loci])
    lg = np.concatenate([np.full(lc.start.size, i) for i, lc in enumerate(loci)])
    if remove_high_overlap_sites and ls.size:
        rm, _ = _high_overlap_mask(ls, le, lg)
        lb = lb[~rm]
    vc = pd.Series(lb).value_counts()
    tot["nAbove2"] = vc.reindex(tot.index).fillna(0).to_numpy(np.int64)
    return tot.astype(np.int64)


def amulet(x, **kwargs) -> pd.DataFrame:
    """Port of R ``amulet()``: fragment-overlap doublet statistics.

    Runs :func:`get_fragment_overlaps` (all keyword arguments are passed on)
    and adds ``p.value = ppois(nAbove2, mean(nAbove2), lower.tail=FALSE)``
    and its Benjamini-Hochberg ``q.value``. Low values indicate doublets.

    Examples
    --------
    >>> res = amulet("atac_fragments.tsv.gz", regions_to_exclude="default")
    >>> doublets = res.index[res["q.value"] < 0.05]
    """
    d = get_fragment_overlaps(x, **kwargs)
    nab = d["nAbove2"].to_numpy(np.float64)
    d["p.value"] = ppois_upper(nab, nab.mean()) if nab.size else []
    d["q.value"] = p_adjust_bh(d["p.value"].to_numpy())
    return d


def amulet_from_counts(counts, widths: Optional[np.ndarray] = None,
                       chroms: Optional[np.ndarray] = None,
                       max_width: int = 500,
                       exclude: Sequence[str] = ("chrM", "M", "Mt")) -> np.ndarray:
    """Port of R ``amuletFromCounts``: AMULET q-values from a count matrix.

    ``counts`` is features x cells. If ``widths``/``chroms`` are given (the
    feature ranges), features on ``exclude`` chromosomes or wider than
    ``max_width`` are dropped first, as R does for a SingleCellExperiment
    with row ranges. R documents this approximation as considerably worse
    than :func:`amulet` on fragment files.

    Returns the per-cell q-values.
    """
    import scipy.sparse as sp
    y = counts
    if chroms is not None:
        keep = ~np.isin(np.asarray(chroms).astype(str), list(exclude))
        y = y[np.flatnonzero(keep)]
        if widths is not None:
            widths = np.asarray(widths)[keep]
    if widths is not None:
        y = y[np.flatnonzero(np.asarray(widths) <= max_width)]
    y = sp.csr_matrix(y) if sp.issparse(y) else np.asarray(y)
    above = (y > 2)
    rs = np.asarray(above.sum(axis=1)).ravel().astype(np.float64)
    # R: p.adjust() with its default method, Holm (the final q-values use BH)
    rp = p_adjust_holm(ppois_upper(rs, rs.mean())) if rs.size else rs
    above = above[np.flatnonzero(rp > 0.01)]
    nab = np.asarray(above.sum(axis=0)).ravel().astype(np.float64)
    return p_adjust_bh(ppois_upper(nab, nab.mean()))


def fisher_combine(*pvalues, floor: float = 0.001) -> np.ndarray:
    """Fisher's method across per-cell p-values, as in the scDblFinder ATAC
    vignette (``aggregation::fisher`` with p-values floored at 0.001).

    The vignette recommends combining ``1 - scDblFinder.score`` with the
    AMULET ``p.value``; it found this more robust than either method alone.
    """
    from scipy.stats import chi2
    P = np.column_stack([np.asarray(p, dtype=np.float64) for p in pvalues])
    P = np.maximum(P, floor)
    stat = -2.0 * np.log(P).sum(axis=1)
    return chi2.sf(stat, df=2 * P.shape[1])
