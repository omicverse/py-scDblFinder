"""ATAC-mode tests: amulet / getFragmentOverlaps, aggregateFeatures, ATAC
processing, clamulet. Run without R; expected values for the bundled
``tests/data/example_fragments.tsv.gz`` (the example file shipped with R
scDblFinder) were produced by R scDblFinder 1.16.0."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

HERE = Path(__file__).parent
EXAMPLE = HERE / "data" / "example_fragments.tsv.gz"


# ---------------------------------------------------------------------------
# helpers: brute-force reference implementations (independent of the port)
# ---------------------------------------------------------------------------
def _brute_loci(frags: pd.DataFrame) -> dict:
    """{barcode: [(chrom, start, end), ...]} of maximal runs with coverage >= 3,
    from an explicit per-base coverage array."""
    out = {}
    for (bc, chrom), g in frags.groupby(["barcode", "chrom"]):
        lo, hi = g["start"].min(), g["end"].max()
        cov = np.zeros(hi - lo + 1, dtype=int)
        for s, e in zip(g["start"], g["end"]):
            cov[s - lo:e - lo] += 1
        above = cov >= 3
        runs, i = [], 0
        while i < above.size:
            if above[i]:
                j = i
                while j < above.size and above[j]:
                    j += 1
                runs.append((chrom, lo + i, lo + j))
                i = j
            else:
                i += 1
        out.setdefault(bc, []).extend(runs)
    return out


def _random_fragments(seed=0, n_cells=30, n_frag=400, span=20000):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        k = n_frag + (n_frag if c < 5 else 0)          # 5 "doublets"
        st = rng.integers(0, span, size=k)
        ln = rng.integers(50, 400, size=k)
        ch = rng.choice(["chr1", "chr2"], size=k)
        for a, b, h in zip(st, st + ln, ch):
            rows.append((h, int(a), int(b), f"cell{c:02d}", 1))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "barcode", "count"])
    return df.sort_values(["chrom", "start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def test_import_surface_atac():
    import pyscdblfinder as psd
    for sym in ("amulet", "amulet_from_counts", "get_fragment_overlaps",
                "clamulet", "aggregate_features", "atac_processing", "tfidf",
                "log_norm_counts", "fisher_combine"):
        assert hasattr(psd, sym), sym


def test_p_adjust_bh_matches_scipy():
    from scipy.stats import false_discovery_control
    from pyscdblfinder.fragments import p_adjust_bh
    p = np.random.default_rng(1).uniform(size=200) ** 3
    p[:10] = p[10]          # ties
    assert np.allclose(p_adjust_bh(p), false_discovery_control(p), rtol=0, atol=1e-15)


def test_p_adjust_holm_matches_R():
    from pyscdblfinder.fragments import p_adjust_holm
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    # R: p.adjust(c(0.01,0.04,0.03,0.005,0.5)) -> 0.04 0.09 0.09 0.025 0.5
    assert np.allclose(p_adjust_holm(p), [0.04, 0.09, 0.09, 0.025, 0.5])


def test_loci_match_brute_force():
    from pyscdblfinder.fragments import _loci_ge3
    df = _random_fragments(seed=3, n_cells=8, n_frag=300, span=5000)
    ref = _brute_loci(df)
    for chrom, g in df.groupby("chrom"):
        names = np.array(sorted(g["barcode"].unique()))
        code = pd.Categorical(g["barcode"], categories=names).codes.astype(np.int64)
        b, s, e = _loci_ge3(code, g["start"].to_numpy(np.int64),
                            g["end"].to_numpy(np.int64), np.ones(len(g), np.int64))
        got = sorted((names[i], chrom, int(a), int(z)) for i, a, z in zip(b, s, e))
        exp = sorted((bc, c, a, z) for bc, runs in ref.items() for c, a, z in runs
                     if c == chrom)
        assert got == exp


def test_amulet_example_matches_R():
    """R scDblFinder 1.16.0: amulet(system.file('extdata',
    'example_fragments.tsv.gz', package='scDblFinder'))."""
    from pyscdblfinder import amulet
    d = amulet(str(EXAMPLE))
    assert list(d.index) == [f"barcode{i}" for i in range(1, 6)]
    assert d["nFrags"].tolist() == [878, 2401, 2325, 1882, 1355]
    assert d["nAbove2"].tolist() == [1, 0, 1, 0, 6]
    assert d["total.nAbove2"].tolist() == [1, 0, 1, 0, 6]
    r_p = [0.4750690532138959199, 0.7981034820053446444, 0.4750690532138959199,
           0.7981034820053446444, 0.0013357612615199526]
    r_q = [0.7917817553564931998, 0.7981034820053446444, 0.7917817553564931998,
           0.7981034820053446444, 0.0066788063075997632]
    assert np.allclose(d["p.value"], r_p, rtol=1e-12, atol=0)
    assert np.allclose(d["q.value"], r_q, rtol=1e-12, atol=0)


def test_amulet_example_options_match_R():
    """amulet(f, minFrags=1000, uniqueFrags=FALSE, maxFragSize=300) in R."""
    from pyscdblfinder import amulet
    d = amulet(str(EXAMPLE), min_frags=1000, unique_frags=False, max_frag_size=300)
    assert list(d.index) == ["barcode2", "barcode3", "barcode4", "barcode5"]
    assert d["nFrags"].tolist() == [1947, 1861, 1594, 1039]
    assert d["nAbove2"].tolist() == [0, 0, 0, 6]
    assert np.allclose(d["p.value"], [0.77686983985157020971] * 3
                       + [0.00092599191352464247], rtol=1e-12, atol=0)
    assert np.allclose(d["q.value"], [0.7768698398515702097] * 3
                       + [0.0037039676540985699], rtol=1e-12, atol=0)


def test_amulet_detects_planted_doublets_and_exclusion():
    from pyscdblfinder import amulet, get_fragment_overlaps
    # sparse coverage: singlets rarely reach 3 fragments, doublets (2x) do
    df = _random_fragments(seed=0, n_cells=30, n_frag=400, span=200000)
    d = amulet(df, min_frags=0, regions_to_exclude=None,
               remove_high_overlap_sites=False)
    top5 = set(d.sort_values("p.value").index[:5])
    assert top5 == {f"cell{c:02d}" for c in range(5)}
    # excluding chr2 entirely removes its fragments and loci
    d2 = get_fragment_overlaps(df, min_frags=0, regions_to_exclude=[("chr2", 0, 10**8)],
                               remove_high_overlap_sites=False)
    only1 = get_fragment_overlaps(df[df.chrom == "chr1"], min_frags=0,
                                  regions_to_exclude=None,
                                  remove_high_overlap_sites=False)
    pd.testing.assert_frame_equal(d2, only1)


def test_default_exclusion_window_is_R_like():
    """R's default excludes chrX only over [1, 1e8]."""
    from pyscdblfinder import get_fragment_overlaps
    rows = [("chrX", 5_000_000 + i, 5_000_200 + i, "a", 1) for i in range(600)]
    rows += [("chrX", 120_000_000 + i, 120_000_200 + i, "a", 1) for i in range(600)]
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "barcode", "count"])
    d = get_fragment_overlaps(df, min_frags=0, remove_high_overlap_sites=False)
    assert d.loc["a", "nFrags"] == 600


def test_tabix_mode_reads_R_window(tmp_path):
    """Per-chromosome (tabix) mode reads only [1, 1e8] of each chromosome,
    as R does; tabix_window=None reads everything."""
    from pyscdblfinder import get_fragment_overlaps
    rows = [("chr1", 1000 + 10 * i, 1200 + 10 * i, "a", 1) for i in range(600)]
    rows += [("chr1", 150_000_000 + 10 * i, 150_000_200 + 10 * i, "a", 1) for i in range(600)]
    f = tmp_path / "frags.tsv"
    pd.DataFrame(rows).to_csv(f, sep="\t", header=False, index=False)
    (tmp_path / "frags.tsv.tbi").write_bytes(b"")      # presence marks "indexed"
    d = get_fragment_overlaps(str(f), min_frags=0, regions_to_exclude=None)
    assert d.loc["a", "nFrags"] == 600
    d = get_fragment_overlaps(str(f), min_frags=0, regions_to_exclude=None,
                              tabix_window=None)
    assert d.loc["a", "nFrags"] == 1200
    d = get_fragment_overlaps(str(f), min_frags=0, regions_to_exclude=None,
                              full_in_memory=True)
    assert d.loc["a", "nFrags"] == 1200


def test_high_overlap_merge_ignores_abutting():
    from pyscdblfinder.fragments import _high_overlap_mask
    s = np.array([0, 10, 20, 25], dtype=np.int64)
    e = np.array([10, 20, 30, 40], dtype=np.int64)
    _, site = _high_overlap_mask(s, e)
    # [0,10) and [10,20) abut -> separate; [20,30) and [25,40) overlap -> same
    assert site[0] != site[1] and site[1] != site[2] and site[2] == site[3]


def test_tfidf_matches_formula():
    from pyscdblfinder import tfidf
    rng = np.random.default_rng(0)
    x = rng.poisson(0.5, size=(40, 30)).astype(float)
    x[:, 0] += 1
    got = tfidf(x).toarray()
    tf = x / x.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        idf = x.shape[1] / x.sum(axis=1)
        exp = np.log1p(1e4 * idf[:, None] * tf)
    exp[~np.isfinite(exp)] = 0
    assert np.allclose(got, exp)


def test_log_norm_counts_matches_scuttle_formula():
    from pyscdblfinder import log_norm_counts
    x = np.array([[1., 0, 4], [3, 2, 0], [0, 6, 2]])
    sf = x.sum(0) / x.sum(0).mean()
    assert np.allclose(log_norm_counts(x), np.log2(x / sf + 1))


def test_aggregate_features_preserves_counts():
    from pyscdblfinder import aggregate_features
    rng = np.random.default_rng(0)
    x = sp.random(2000, 300, density=0.02, random_state=1, format="csr",
                  data_rvs=lambda n: rng.integers(1, 4, size=n)).astype(float)
    agg, groups, kept = aggregate_features(x, k=25, return_groups=True)
    assert agg.shape == (25, 300)
    assert np.allclose(agg.sum(axis=0), np.asarray(x[kept].sum(axis=0)).ravel())
    assert set(np.unique(groups)) == set(range(1, 26))


def _synthetic_atac(seed=0, n_cells=600, n_types=4, n_peaks=3000, dbl=0.1):
    rng = np.random.default_rng(seed)
    prof = rng.gamma(0.3, 1.0, size=(n_types, n_peaks))
    prof /= prof.sum(axis=1, keepdims=True)
    t = rng.integers(0, n_types, size=n_cells)
    lam = prof[t] * rng.integers(800, 1600, size=n_cells)[:, None]
    nd = int(dbl * n_cells)
    is_d = np.zeros(n_cells, bool)
    is_d[:nd] = True
    t2 = (t[:nd] + 1 + rng.integers(0, n_types - 1, size=nd)) % n_types
    lam[:nd] += prof[t2] * rng.integers(800, 1600, size=nd)[:, None]
    x = rng.poisson(lam).T            # peaks x cells
    return sp.csc_matrix(x.astype(float)), is_d


@pytest.mark.parametrize("processing", ["normFeatures", "atac"])
def test_scdblfinder_atac_mode_finds_heterotypic_doublets(processing):
    from sklearn.metrics import roc_auc_score
    from pyscdblfinder import sc_dbl_finder
    x, is_d = _synthetic_atac()
    res = sc_dbl_finder(x, aggregate_features=True, n_features=25,
                        processing=processing, random_state=0, progress=False)
    score = res.real_cells()["score"].to_numpy()
    assert roc_auc_score(is_d, score) > 0.8


def test_scdblfinder_class_api_atac():
    import anndata as ad
    from pyscdblfinder import ScDblFinder
    x, _ = _synthetic_atac(n_cells=300)
    adata = ad.AnnData(sp.csr_matrix(x.T))
    ScDblFinder(adata, random_state=0).run(aggregate_features=True, n_features=25,
                                           processing="normFeatures", progress=False)
    assert {"scDblFinder_score", "scDblFinder_class"} <= set(adata.obs.columns)


def test_amulet_from_counts_runs():
    from pyscdblfinder import amulet_from_counts
    x, is_d = _synthetic_atac(n_cells=300)
    q = amulet_from_counts(x)
    assert q.shape == (300,) and np.all((q >= 0) & (q <= 1))


def test_fisher_combine():
    from scipy.stats import combine_pvalues
    from pyscdblfinder import fisher_combine
    a = np.array([0.5, 0.01, 0.0001])
    b = np.array([0.2, 0.3, 0.04])
    exp = [combine_pvalues([max(u, 1e-3), max(v, 1e-3)], method="fisher")[1]
           for u, v in zip(a, b)]
    assert np.allclose(fisher_combine(a, b), exp)


def test_clamulet_smoke():
    from pyscdblfinder import clamulet
    df = _random_fragments(seed=1, n_cells=40, n_frag=500, span=2_000_000)
    d = clamulet(df, min_frags=0, regions_to_exclude=None, k=[3, 5],
                 nfeatures=5, random_state=0)
    assert len(d) == 40
    assert {"total", "nAbove2", "total.nAbove2", "weighted", "score"} <= set(d.columns)
    assert np.all((d["score"] >= 0) & (d["score"] <= 1))
