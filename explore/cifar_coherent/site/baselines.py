"""Site data for the "Sets from" control and the "Baselines and nulls" tab (build_site.py --baselines r20s0c).

Inputs (read-only): data/<tag>/ (explore/cifar_baselines/SCHEMA.md), data/r20s0b/ (the method's labels, probe, points),
explore/cifar_baselines/out/bench/summary.json (NOTES_B.md) and explore/cifar_baselines/out/transfer/ (TRANSFER.md).
Writes viewer/site/d/src/ (explorer, lazily fetched per block) and viewer/site/d/bn/*.json (the tab).

Numbers that the inputs do not store and that the page needs are computed here with the study's own functions
(explore/cifar_baselines/stats.py, imported, unchanged): the predicted-class source's agreement rows, the agreement of
the stored null draws with the method, the source-by-source matrices of the representative blocks and the
snapshot-by-snapshot set agreement. Each uses N_PERM_SITE size-preserving permutations and is cached in
~/.cache/cm-site-bn/ (key: input sizes and mtimes plus CACHE_VERSION), so a rebuild takes seconds.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                    # explore/cifar_coherent
BASE = ROOT.parent / "cifar_baselines"
sys.path.insert(0, str(BASE))
import stats  # noqa: E402  (explore/cifar_baselines/stats.py)

CACHE = Path.home() / ".cache" / "cm-site-bn"
CACHE_VERSION = "bn-v3"
N_PERM_SITE = 200
LAB_SRC = ["L1", "L2", "L3", "G1", "G1n", "G2", "C2", "C3", "N3"]
REP_DRAWS = ["N2a", "N2b", "N3", "N4", "N5"]


def log(*a):
    print(*a, flush=True)


def fin(x, nd=4):
    """float -> rounded float or None (JSON null) for non-finite."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def rnd(o, nd=4):
    if isinstance(o, dict):
        return {str(k): rnd(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v, nd) for v in o]
    if isinstance(o, (np.ndarray,)):
        return rnd(o.tolist(), nd)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return fin(o, nd)
    return o


def pack_nibbles(lab: np.ndarray) -> np.ndarray:
    v = (lab.astype(np.int16) + 1).astype(np.uint8)
    if v.shape[-1] % 2:
        v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,), np.uint8)], -1)
    return (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8)


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rnd(obj), separators=(",", ":"), allow_nan=False))


def stamp(paths) -> str:
    h = hashlib.sha256(CACHE_VERSION.encode())
    for p in sorted(map(str, paths)):
        st = Path(p).stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def cached(name, deps, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}-{stamp(deps)}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t0 = time.time()
    out = fn()
    f.write_bytes(pickle.dumps(out))
    log(f"  computed {name} in {time.time() - t0:.0f} s (cached in {f})")
    return out


# --------------------------------------------------------------------------------------------- agreement
def agree_fields_row(fields, a, b, ka, kb, seed):
    """A SCHEMA agreement row (fields in meta_c order) of source labels b against the method's labels a, with the N7
    nulls from N_PERM_SITE permutations of b (stats.fast_ami_null). Class fields are left NaN.
    Returns (row, joint AMI draws, matched-Jaccard draws) for max-over-blocks bands."""
    row = dict.fromkeys(fields, np.nan)
    ag = stats.agreement(a, b, ka=ka, kb=kb)
    for k in ("ami_all", "ami_joint", "joint_coverage", "ari_all", "mj_max", "mj_legacy", "mj_weighted"):
        row[k] = ag[k]
    row["unassigned"] = float((np.asarray(b) < 0).mean())
    alln = stats.fast_ami_null(a, b, n_perm=N_PERM_SITE, seed=seed, ka=ka, kb=kb, return_draws=True)
    row["ami_null_q95"] = alln["ami"]["q95"]
    row["mj_null_mean"], row["mj_null_q95"] = alln["mj_max"]["mean"], alln["mj_max"]["q95"]
    m = (a >= 0) & (b >= 0)
    jd = np.full(N_PERM_SITE, np.nan)
    if m.sum() > 2 and len(np.unique(a[m])) > 1 and len(np.unique(b[m])) > 1:
        jn = stats.fast_ami_null(a[m], b[m], n_perm=N_PERM_SITE, seed=seed + 1, return_draws=True)
        row["ami_joint_null_q95"] = jn["ami"]["q95"]
        jd = jn["ami_draws"]
    return np.array([row[k] for k in fields], np.float32), jd, alln["mj_draws"]


def quick_agree(a, b, ka, kb, seed, with_mj=False):
    """AMI on jointly assigned points, coverage, matched Jaccard and the joint-AMI permutation 95th percentile;
    with_mj: also the permutation mean and 95th percentile of matched Jaccard (labels of b permuted, sizes kept)."""
    m = (a >= 0) & (b >= 0)
    cov = float(m.mean())
    aj = stats.ami(a, b, "joint") if m.sum() > 2 else np.nan
    mj = stats.matched_jaccard(a, b, ka=ka, kb=kb)
    q = np.nan
    if m.sum() > 2 and len(np.unique(a[m])) > 1 and len(np.unique(b[m])) > 1:
        q = stats.fast_ami_null(a[m], b[m], n_perm=N_PERM_SITE, seed=seed)["ami"]["q95"]
    if not with_mj:
        return [aj, cov, mj, q]
    nm = stats.fast_ami_null(a, b, n_perm=N_PERM_SITE, seed=seed + 7, ka=ka, kb=kb)["mj_max"]   # b permuted, sizes kept
    return [aj, cov, mj, q, nm["mean"], nm["q95"]]


AG_NULL = ("ami_joint_null_q95", "ami_joint_null_mean", "mj_null_mean", "mj_null_q95", "n_perm")


def ag_keep(x):
    """agreement dict -> the fields the page shows, with their own permutation nulls (schema 1.1)."""
    return {k: x.get(k) for k in ("ami_joint", "joint_coverage", "mj_max", "error") + AG_NULL} if isinstance(x, dict) else None


def ov_val(x):
    """overlap in schema 1.0 (number) or 1.1 ({mean_cos2, chance}) -> (value, chance)."""
    return (x.get("mean_cos2"), x.get("chance")) if isinstance(x, dict) else (x, None)


def abl_entry(fac, v):
    ent = {k: v[k] for k in ("lambda_P", "vals", "spectrum_of", "laziness", "note", "error", "k", "diag_median", "off_edge_mass",
                             "components", "stationary", "vals_includes_leading_vector", "vals_convention") if k in v}
    if v.get("agreement_with_M0"):
        ent["agree"] = {r: ag_keep(x) for r, x in v["agreement_with_M0"].items()}
    for k in ("overlap_with_M0", "right_overlap", "left_overlap"):
        if isinstance(v.get(k), dict):
            ent[k] = {r: ov_val(x)[0] for r, x in v[k].items()}
            ent[k + "_chance"] = {r: ov_val(x)[1] for r, x in v[k].items()}
    if isinstance(v.get("S1"), list) and v["S1"]:          # level rebuilt on 5 subsets, against its own full-set labels
        rs = v["S1"][0]["per_r"].keys()
        ent["S1"] = {r: {"ami_joint": [d["per_r"][r].get("ami_joint") for d in v["S1"]],
                         "coverage": [d["per_r"][r].get("joint_coverage") for d in v["S1"]],
                         "null_q95": [d["per_r"][r].get("ami_joint_null_q95") for d in v["S1"]]} for r in rs}
    n3 = v.get("n3")
    if isinstance(n3, dict) and "excess_over_q95" in n3:
        ent["n3"] = {"excess": n3["excess_over_q95"], "q95": [b.get("q95") for b in n3.get("lambda_P_band", [])], "draws": n3.get("draws")}
    elif isinstance(n3, dict):
        ent["n3"] = {"undefined": True}
    return ent


# --------------------------------------------------------------------------------------------- builder
class BN:
    def __init__(self, b, tag: str):
        self.b, self.tag = b, tag
        self.src = ROOT / "data" / tag
        self.base = b.src                                  # data/r20s0b
        self.out = b.d / "src"
        self.bn = b.d / "bn"
        self.meta_c = json.loads((self.src / "meta_c.json").read_text())
        assert self.meta_c["kernel_rule"] == 0
        self.fields = self.meta_c["agree_fields"]
        self.max_fields = self.meta_c["max_fields"]
        self.R = self.meta_c["seba_r"]
        assert self.R == b.seba_r, (self.R, b.seba_r)
        self.N = b.n
        self.steps, self.sites = b.steps, b.sites
        self.labels = np.load(self.base / "probe.npz")["labels"].astype(np.int64)
        self.facts = {}

    # ---------------------------------------------------------------- the method's labels (mode 0)
    def m0(self, fam, anchor_name):
        with np.load(self.base / f"{fam}_{anchor_name}.npz") as z:
            return z["seba_label"][0].astype(np.int8)           # (P, 5, N), kernel rule 0

    def pred(self, step):
        with np.load(self.base / f"points_{step}.npz") as z:
            return z["pred"].astype(np.int8)

    def files(self):
        out = []
        for t, s in enumerate(self.steps):
            out.append(("layer", t, str(s), self.src / f"sources_layer_{s}.npz", self.b.meta["layer_pairs"]))
        for si, s in enumerate(self.sites):
            out.append(("time", si, s, self.src / f"sources_time_{s}.npz", self.b.meta["time_pairs"]))
        return [f for f in out if f[3].exists()]

    def run(self):
        t0 = time.time()
        self.prefetch()
        self.explorer()
        self.reps_draws()
        self.tab()
        tot = sum(p.stat().st_size for d in (self.out, self.bn) for p in d.rglob("*") if p.is_file())
        self.facts["bn.site_mb"] = f"{tot / 1e6:.1f}"
        log(f"baselines: {tot / 1e6:.1f} MB in d/src and d/bn ({time.time() - t0:.0f} s)")
        return self.facts

    # ---------------------------------------------------------------- parallel warm-up of the caches (fork, 3 workers)
    def prefetch(self):
        global _G
        _G = self
        jobs = [("y2", f) for f in self.files()]
        jobs += [("rep", rid) for rid in self.meta_c["reps"]]
        import multiprocessing as mp
        with mp.get_context("fork").Pool(3) as pool:
            for _ in pool.imap_unordered(_job, jobs):
                pass

    # ---------------------------------------------------------------- predicted-class rows (computed)
    def y2_rows(self, fam, anchor, anchor_name, pairs, path):
        def comp():
            M0 = self.m0(fam, anchor_name)
            P = len(pairs)
            rows = np.full((P, len(self.R), len(self.fields)), np.nan, np.float32)
            jd = np.full((P, len(self.R), N_PERM_SITE), np.nan)
            md = np.full((P, len(self.R), N_PERM_SITE), np.nan)
            preds = {}
            for p, (a, b) in enumerate(pairs):
                step = self.steps[anchor] if fam == "layer" else self.steps[b]
                if step not in preds:
                    preds[step] = self.pred(step).astype(np.int64)
                y = preds[step]
                for ri, r in enumerate(self.R):
                    rows[p, ri], jd[p, ri], md[p, ri] = agree_fields_row(self.fields, M0[p, ri].astype(np.int64), y, r, 10, seed=1000 * p + ri)
            mx = np.zeros((len(self.R), len(self.max_fields), 2), np.float32) * np.nan
            for ri in range(len(self.R)):
                for fi, (name, draws) in enumerate([("ami_joint", jd), ("mj_max", md)]):
                    j = self.max_fields.index(name)
                    mxd = np.nanmax(draws[:, ri, :], axis=0)
                    mx[ri, j] = [np.nanmean(mxd), np.nanquantile(mxd, .95)]
            return rows, mx
        return cached(f"y2-{fam}-{anchor}", [path, self.base / f"{fam}_{anchor_name}.npz"] + [self.base / f"points_{s}.npz" for s in self.steps], comp)

    # ---------------------------------------------------------------- explorer chunks
    def explorer(self):
        idx = {"tag": self.tag, "kernel_rule": 0, "n_perm": self.meta_c["n_perm"], "n_perm_site": N_PERM_SITE,
               "fields": self.fields, "max_fields": self.max_fields, "seba_r": self.R, "files": {}, "reps": {}}
        total = 0
        for fam, anchor, name, path, pairs in self.files():
            t0 = time.time()
            z = np.load(path)
            lab = [s for s in LAB_SRC if s in z.files]
            extra = {}
            if fam == "layer" and name == str(self.steps[-1]) and (self.src / f"I3_seed1_layer_{name}.npz").exists():
                with np.load(self.src / f"I3_seed1_layer_{name}.npz") as zz:
                    extra["I3"] = zz["seba_label"].astype(np.int8)
                lab.append("I3")
            agree = [k[6:] for k in z.files if k.startswith("agree_")]
            y2, y2max = self.y2_rows(fam, anchor, name, pairs, path)
            agree.append("Y2")
            maxnull = {k[8:]: z[k] for k in z.files if k.startswith("maxnull_")}
            maxnull["Y2"] = y2max
            P = len(pairs)
            L = {s: (z[s] if s in z.files else extra[s]) for s in lab}
            A = {s: (z["agree_" + s] if s != "Y2" else y2) for s in agree}
            for s in lab:
                assert L[s].shape == (P, len(self.R), self.N), (path, s, L[s].shape)
            d = self.out / fam / str(anchor)
            d.mkdir(parents=True, exist_ok=True)
            for p in range(P):
                blob = b"".join([pack_nibbles(L[s][p]).tobytes() for s in lab] +
                                [np.ascontiguousarray(A[s][p], "<f4").tobytes() for s in agree])
                gz = gzip.compress(blob, compresslevel=6, mtime=0)
                (d / f"{p}.bin.gz").write_bytes(gz)
                total += len(gz)
            idx["files"][f"{fam}/{anchor}"] = {"name": name, "P": P, "lab": lab, "agree": agree,
                                              "maxnull": {k: v.tolist() for k, v in maxnull.items()}}
            log(f"  sources {fam} {name}: {len(lab)} label sources, {len(agree)} agreement rows ({time.time() - t0:.1f} s)")
        with np.load(self.src / "I1_pixels.npz") as z:
            i1 = z["M0_input"].astype(np.int8)
        (self.out / "i1.bin").write_bytes(pack_nibbles(i1).tobytes())
        self.idx = idx
        self.facts["bn.src_chunks_mb"] = f"{total / 1e6:.1f}"
        self.facts["bn.n_perm"] = f"{self.meta_c['n_perm']:,}"
        self.facts["bn.n_perm_site"] = str(N_PERM_SITE)
        self.facts["bn.n_grid_blocks"] = f"{sum(v['P'] for v in idx['files'].values()):,}"

    # ---------------------------------------------------------------- representative blocks: stored null draws
    def rep_block(self, rid):
        rp = self.meta_c["reps"][rid]
        fam = rp["family"]
        if fam == "layer":
            anchor, anchor_name = self.steps.index(int(rp["key"])), str(rp["key"])
            pairs = self.b.meta["layer_pairs"]
        else:
            anchor, anchor_name = self.sites.index(rp["key"]), rp["key"]
            pairs = self.b.meta["time_pairs"]
        a, b = rp["a"], rp["b"]          # site indices (layer) or checkpoint indices (time), as in meta_c.json
        p = pairs.index([a, b])
        return dict(fam=fam, anchor=anchor, anchor_name=anchor_name, a=a, b=b, pair=p)

    def reps_draws(self):
        self.reps = {}
        for rid in self.meta_c["reps"]:
            rb = self.rep_block(rid)
            self.reps[rid] = rb
            f = self.src / f"rep_{rid}_nulls_draws.npz"
            if not f.exists():
                continue
            with np.load(f) as z:
                draws = [k for k in REP_DRAWS if k in z.files]
                D = {k: z[k].astype(np.int8) for k in draws}
            def comp(D=D, rb=rb):
                M0 = self.m0(rb["fam"], rb["anchor_name"])[rb["pair"]]
                return {k: np.stack([agree_fields_row(self.fields, M0[ri].astype(np.int64), D[k][ri].astype(np.int64), r, r, seed=77 + ri)[0]
                                     for ri, r in enumerate(self.R)]) for k in D}
            ag = cached(f"repdraws-{rid}", [f, self.base / f"{rb['fam']}_{rb['anchor_name']}.npz"], comp)
            (self.out / "rep").mkdir(parents=True, exist_ok=True)
            (self.out / "rep" / f"{rid}.bin").write_bytes(b"".join(pack_nibbles(D[k]).tobytes() for k in draws))
            self.idx["reps"][rid] = {**{k: rb[k] for k in ("fam", "anchor", "a", "b", "pair")}, "draws": draws,
                                     "agree": {k: v.tolist() for k, v in ag.items()}}
        dump(self.out / "index.json", self.idx)

    # ---------------------------------------------------------------- the tab
    def tab(self):
        reps = json.loads((self.src / "reps.json").read_text())
        out = {"reps": {}, "order": list(self.meta_c["reps"]), "seba_r": self.R, "n": self.N, "n_perm_site": N_PERM_SITE,
               "haar": json.loads((self.src / "n6_haar.json").read_text())}
        grid_rows = {}
        for rid in out["order"]:
            rb = self.reps[rid]
            z = np.load(self.src / ("sources_layer_%s.npz" % rb["anchor_name"] if rb["fam"] == "layer" else "sources_time_%s.npz" % rb["anchor_name"]))
            grid_rows[rid] = {k[6:]: z[k][rb["pair"]] for k in z.files if k.startswith("agree_")}
            out["reps"][rid] = self.tab_rep(rid, reps[rid], grid_rows[rid])
        del reps
        se = json.loads((self.src / "stability_extra.json").read_text())
        out["s1_cells"] = {s: [self.s1_summary(x) for x in v] for s, v in se["s1_cells_64000"].items()}
        out["n10k"] = {rid: {k: {"lambda_M": v.get("lambda_M"), "per_r": {r: {kk: pr.get(kk) for kk in ("ami_joint", "joint_coverage", "mj_max", "overlap_on_probe", "overlap_chance") + AG_NULL}
                                                                            for r, pr in v["per_r"].items()}} for k, v in d.items() if isinstance(v, dict) and "per_r" in v}
                       for rid, d in se["n10k"].items()}
        dump(self.bn / "reps.json", out)
        self.tab_matrix()
        self.tab_dist()
        self.tab_bench()
        self.tab_transfer()
        f = self.facts
        f["bn.n_reps"] = str(len(out["order"]))
        # joint second-order surrogate: stored change of cross-snapshot second moments after the unit diagonal is restored
        mm = {rid: out["reps"][rid]["nulls"].get("N2b", {}).get("mismatch") for rid in out["order"]}
        mm = {k: v for k, v in mm.items() if v}
        if mm:
            f["bn.n2b_mm_max"] = f"{100 * max(v['second_moment_max'] for v in mm.values()):.1f}%"
            f["bn.n2b_mm_b4b6"] = f"{100 * mm['b4-b6']['second_moment_max']:.1f}%" if "b4-b6" in mm else "n/a"
        # worked example: layer block b4..b6 at the final step, cosine PCA + SEBA, r = 4 (stored grid row)
        F, ri = self.fields, self.R.index(4)
        row = grid_rows["b4-b6"]["L1"][ri]
        mx = self.idx["files"][f"layer/{len(self.steps) - 1}"]["maxnull"]["L1"][ri]
        g = lambda k: float(row[F.index(k)])
        f.update({"bn.ex.ami": f"{g('ami_joint'):.2f}", "bn.ex.cov": f"{100 * g('joint_coverage'):.0f}%", "bn.ex.q95": f"{g('ami_joint_null_q95'):.3f}",
                  "bn.ex.max95": f"{mx[self.max_fields.index('ami_joint')][1]:.3f}", "bn.ex.mj": f"{g('mj_max'):.2f}",
                  "bn.ex.mj_null": f"{g('mj_null_mean'):.2f}", "bn.ex.mj_max95": f"{mx[self.max_fields.index('mj_max')][1]:.2f}",
                  "bn.ex.nblocks": str(self.idx["files"][f"layer/{len(self.steps) - 1}"]["P"])})
        # run-to-run variation of the planted benchmarks: content/bn_facts.json (copied from NOTES_B.md, cited there)
        fp = HERE / "content" / "bn_facts.json"
        if fp.exists():
            for k, v in json.loads(fp.read_text())["facts"].items():
                f[f"bn.{k}"] = v
        f["bn.haar_draws"] = f"{out['haar'][str(self.R[0])]['draws']:,}"

    @staticmethod
    def s1_summary(x):
        return {r: {k: pr.get(k) for k in ("ami_joint", "joint_coverage", "mj_max", "overlap", "overlap_chance") + AG_NULL} for r, pr in x["per_r"].items()}

    def tab_rep(self, rid, R, grid):
        F = self.fields
        g = lambda src, ri, k: float(grid[src][ri][F.index(k)]) if src in grid else None
        rb = self.reps[rid]
        nl = R["nulls"]
        o = {"fam": rb["fam"], "anchor": rb["anchor"], "a": rb["a"], "b": rb["b"], "cells": nl["cells"]}
        o["real"] = {k: nl["real"][k] for k in ("lambda_P", "lambda_P_bottom", "gaps", "participation", "spectrum_of")}
        o["real"]["seba"] = nl["real"]["seba"]
        o["bands"] = {k: v["lambda_P"] for k, v in nl["bands"].items()}
        o["nulls"] = {}
        for k in REP_DRAWS:
            if k not in nl:
                continue
            draws = nl[k]
            per_r = {}
            for ri, r in enumerate(self.R):
                rs = str(r)
                col = lambda f: [d["seba"][rs].get(f) for d in draws if rs in d["seba"]]
                per_r[rs] = {"unassigned": stats.band(col("unassigned")), "min_size": stats.band([min(d["seba"][rs]["sizes"]) for d in draws]),
                             "clipped": stats.band(col("clipped_negative_mass")), "class_ami": stats.band(col("class_ami")),
                             "ami_joint_M0": stats.band([d["agreement_with_M0"][rs]["ami_joint"] for d in draws]),
                             "cov_M0": stats.band([d["agreement_with_M0"][rs]["joint_coverage"] for d in draws])}
            o["nulls"][k] = {"n": len(draws), "per_r": per_r,
                             "bottom": stats.band([d["lambda_P_bottom"][0] for d in draws])}
            if k == "N2b":
                mm = [d["moment_mismatch"] for d in draws]
                o["nulls"][k]["mismatch"] = {"diag_min": min(x["diag_before_min"] for x in mm), "diag_max": max(x["diag_before_max"] for x in mm),
                                             "second_moment_max": max(abs(x["second_moment_rel_change_max"]) for x in mm),
                                             "mean_norm2_max": max(max(abs(v) for v in x["mean_norm2_rel_change"]) for x in mm)}
            if k == "N5":
                kept = [c["original_edges_kept"] for d in draws for c in d["rewire"]]
                o["nulls"][k]["kept"] = [min(kept), max(kept)]
        # N7 values for the method's class agreement on this block (stored grid row of M0)
        o["m0_class"] = {str(r): {"class_ami": g("M0", ri, "class_ami"), "class_ami_null_q95": g("M0", ri, "class_ami_null_q95"),
                                  "class_ami_joint": g("M0", ri, "class_ami_joint"), "class_ami_joint_null_q95": g("M0", ri, "class_ami_joint_null_q95")}
                         for ri, r in enumerate(self.R)}
        # a permutation reference for statistics stored without their own null (S1, S2, N6 rotation, ablations):
        # range over this block's stored grid rows of the N7 95th percentile of the joint AMI
        q = [[float(grid[s][ri][F.index("ami_joint_null_q95")]) for s in grid if np.isfinite(grid[s][ri][F.index("ami_joint_null_q95")])] for ri in range(len(self.R))]
        qm = [[float(grid[s][ri][F.index("mj_null_mean")]) for s in grid if np.isfinite(grid[s][ri][F.index("mj_null_mean")])] for ri in range(len(self.R))]
        o["ref_null"] = {str(r): {"ami_joint_q95": [min(q[ri]), max(q[ri])], "mj_mean": [min(qm[ri]), max(qm[ri])]} for ri, r in enumerate(self.R)}
        # stability
        S = R["stability"]
        o["S1"] = [self.s1_summary(x) for x in S["S1"]]
        o["S2"] = {"n_fit": S["S2"]["n_fit"], "n_test": S["S2"]["n_test"], "flagged": S["S2"]["flagged_small_lambda"],
                   "src": {s: {r: {k: v.get(k) for k in ("test_retention", "test_null", "sizes", "ami_joint", "joint_coverage", "mj_max", "unassigned") + AG_NULL}
                               for r, v in S["S2"][s].items()} for s in ("M0", "L1", "L3") if s in S["S2"]}}
        o["S3"] = S["S3"]
        o["LOCO"] = S["LOCO"]
        o["N6_rotation"] = {r: [{k: d.get(k) for k in ("ami_joint", "joint_coverage", "mj_max") + AG_NULL} for d in v] for r, v in S["N6_rotation"].items()}
        # linear content and CKA
        L = R["linear"]
        o["linear"] = {k: L[k] for k in ("linear_content", "cka", "cka_debiased", "cka_perm_null", "cka_perm_null_biased", "cka_debiased_perm_null",
                                         "cka_perm_null_pair", "subspace_overlap_cells", "pca_dirichlet_under_cells", "method_dirichlet_under_cells",
                                         "dirichlet_convention") if k in L}
        o["linear"]["pca_C_vs_HCH"] = [{"mean_cos2": x["overlap"]["mean_cos2"], "min_cos": x["overlap"]["min_cos"], "top_C_constant_cosine": x.get("top_C_constant_cosine")}
                                       for x in L.get("pca_C_vs_HCH", [])]
        o["linear"]["set_agree_cells"] = self.cell_agree(rid, rb)
        # ablations: per factor, per level (schema 1.1: agreement nulls, overlap {mean_cos2, chance}, S1 and n3 per
        # Markov level, explicit spectrum conventions)
        A = R["ablate"]["ablations"]
        abl = {}
        for fac, levels in A.items():
            if not isinstance(levels, dict):
                continue
            rows = {}
            for lev, v in levels.items():
                if not isinstance(v, dict):
                    rows[lev] = {"matrix": v}
                    continue
                if lev == "matched_effective_neighbours":
                    rows[lev] = {"target": v.get("target"), "levels": {kk: {"params": vv.get("params"), **abl_entry(fac, {**(vv.get("summary") or {}), **{k: x for k, x in vv.items() if k != "summary"}})}
                                                                       for kk, vv in v.items() if isinstance(vv, dict)}}
                    continue
                if fac == "G":
                    rows[lev] = {r: {k: x.get(k) for k in ("c", "iterations", "unassigned", "sizes", "ami_joint", "joint_coverage", "mj_max", "error", "class_ami") + AG_NULL}
                                 for r, x in v.items() if isinstance(x, dict)}
                    continue
                rows[lev] = abl_entry(fac, v)
            abl[fac] = rows
        o["ablate"] = abl
        return o

    # snapshot-by-snapshot agreement of the single-snapshot sets within a representative block (computed)
    def cell_agree(self, rid, rb):
        n = rb["b"] - rb["a"] + 1
        if n < 2:
            return None
        pairs = self.b.meta["layer_pairs"] if rb["fam"] == "layer" else self.b.meta["time_pairs"]
        def comp():
            M0 = self.m0(rb["fam"], rb["anchor_name"])
            single = [M0[pairs.index([c, c])] for c in range(rb["a"], rb["b"] + 1)]
            res = {}
            for r in (3, 6):
                ri = self.R.index(r)
                aj = np.full((n, n), np.nan); q = np.full((n, n), np.nan); cov = np.full((n, n), np.nan)
                for i in range(n):
                    for j in range(n):
                        if j < i:
                            aj[i, j], cov[i, j], q[i, j] = aj[j, i], cov[j, i], q[j, i]
                            continue
                        x = quick_agree(single[i][ri].astype(np.int64), single[j][ri].astype(np.int64), r, r, seed=31 * i + j)
                        aj[i, j], cov[i, j], q[i, j] = x[0], x[1], x[3]
                res[str(r)] = {"ami_joint": aj.tolist(), "coverage": cov.tolist(), "null_q95": q.tolist()}
            return res
        return cached(f"cellagree-{rid}", [self.base / f"{rb['fam']}_{rb['anchor_name']}.npz"], comp)

    # ---------------------------------------------------------------- (b) source-by-source matrices
    def labels_for(self, rb):
        """Every source's hard labels on a representative block: {id: (5, N) int array, k (sets) or None for k = r}."""
        fam, anchor, name, p = rb["fam"], rb["anchor"], rb["anchor_name"], rb["pair"]
        pairs = self.b.meta["layer_pairs"] if fam == "layer" else self.b.meta["time_pairs"]
        M0 = self.m0(fam, name)
        L = {"M0": M0[p]}
        z = np.load(self.src / (f"sources_layer_{name}.npz" if fam == "layer" else f"sources_time_{name}.npz"))
        for s in ("L1", "L2", "L3", "G1", "G1n", "G2"):
            L[s] = z[s][p]
        a, b = rb["a"], rb["b"]
        for nm, c in (("C1first", a), ("C1mid", (a + b) // 2), ("C1last", b)):
            L[nm] = M0[pairs.index([c, c])]
        L["C2"], L["C3"] = z["C2"][p], z["C3"][p]
        with np.load(self.src / "I1_pixels.npz") as zz:
            L["I1"] = zz["M0_input"]
        if fam == "layer":
            L["I2"] = self.m0("layer", str(self.steps[0]))[p]
            f3 = self.src / f"I3_seed1_layer_{name}.npz"
            if f3.exists():
                with np.load(f3) as zz:
                    L["I3"] = zz["seba_label"][p]
        L["Y1"] = np.broadcast_to(self.labels, (len(self.R), self.N))
        L["Y2"] = np.broadcast_to(self.pred(self.steps[anchor] if fam == "layer" else self.steps[b]), (len(self.R), self.N))
        L["N3"] = z["N3"][p]
        f = self.src / f"rep_{rb['rid']}_nulls_draws.npz"
        if f.exists():
            with np.load(f) as zz:
                for k in ("N2a", "N2b", "N4", "N5"):
                    if k in zz.files:
                        L[k] = zz[k]
        return {k: np.asarray(v).astype(np.int64) for k, v in L.items()}

    def tab_matrix(self):
        out = {"order": [], "blocks": {}}
        for rid, rb in self.reps.items():
            out["blocks"][rid] = self.matrix_block(rid, {**rb, "rid": rid})
            out["order"].append(rid)
        dump(self.bn / "matrix.json", out)

    def matrix_block(self, rid, rb):
        if True:
            deps = [self.base / f"{rb['fam']}_{rb['anchor_name']}.npz", self.base / f"layer_{self.steps[0]}.npz", self.src / "I1_pixels.npz",
                    self.src / (f"sources_layer_{rb['anchor_name']}.npz" if rb["fam"] == "layer" else f"sources_time_{rb['anchor_name']}.npz")]
            deps += [p for p in (self.src / f"rep_{rid}_nulls_draws.npz", self.src / f"I3_seed1_layer_{rb['anchor_name']}.npz") if p.exists()]
            def comp(rb=rb):
                L = self.labels_for(rb)
                ids = list(L)
                k = lambda s, r: 10 if s in ("Y1", "Y2") else r
                M = {}
                for ri, r in enumerate(self.R):
                    n = len(ids)
                    T = np.full((6, n, n), np.nan)
                    for i in range(n):
                        for j in range(i, n):
                            x = quick_agree(L[ids[i]][ri], L[ids[j]][ri], k(ids[i], r), k(ids[j], r), seed=97 * i + j + ri, with_mj=True)
                            T[:, i, j] = T[:, j, i] = x
                    M[str(r)] = {"ami_joint": T[0].tolist(), "coverage": T[1].tolist(), "mj_max": T[2].tolist(), "null_q95": T[3].tolist(),
                                  "mj_null_mean": T[4].tolist(), "mj_null_q95": T[5].tolist()}
                return {"ids": ids, "r": M}
            return cached(f"matrix2-{rid}", deps, comp)

    # ---------------------------------------------------------------- (b) distributions over every explorer block
    def tab_dist(self):
        F = self.fields
        out = {"files": {}, "fields": ["ami_joint", "joint_coverage", "mj_max", "ami_joint_null_q95", "mj_null_mean"]}
        for key, v in self.idx["files"].items():
            fam, anchor = key.split("/")
            z = np.load(self.src / (f"sources_layer_{v['name']}.npz" if fam == "layer" else f"sources_time_{v['name']}.npz"))
            ent = {"name": v["name"], "P": v["P"], "src": {}}
            for s in v["agree"]:
                if s == "M0":
                    continue
                A = z["agree_" + s] if s != "Y2" else self.y2_rows(fam, int(anchor), v["name"],
                                                                        self.b.meta["layer_pairs" if fam == "layer" else "time_pairs"],
                                                                        self.src / (f"sources_layer_{v['name']}.npz" if fam == "layer" else f"sources_time_{v['name']}.npz"))[0]
                ent["src"][s] = {"vals": {str(r): [[fin(A[p, ri, F.index(f)], 3) for p in range(v["P"])] for f in out["fields"]] for ri, r in enumerate(self.R)},
                                 "max": {str(r): {"ami_joint": v["maxnull"][s][ri][self.max_fields.index("ami_joint")],
                                                  "mj_max": v["maxnull"][s][ri][self.max_fields.index("mj_max")]} for ri, r in enumerate(self.R)}}
            # the method's class agreement (Y1 is the classes): stored as the M0 row's class fields
            A = z["agree_M0"]
            ent["src"]["Y1"] = {"vals": {str(r): [[fin(A[p, ri, F.index(f)], 3) for p in range(v["P"])] for f in
                                                  ("class_ami_joint", "unassigned", "class_mj_max", "class_ami_joint_null_q95", "class_mj_null_mean")] for ri, r in enumerate(self.R)},
                                "max": {str(r): {"ami_joint": v["maxnull"]["M0"][ri][self.max_fields.index("class_ami_joint")],
                                                 "mj_max": v["maxnull"]["M0"][ri][self.max_fields.index("class_mj_max")]} for ri, r in enumerate(self.R)},
                                "coverage_is": "1 - unassigned"}
            dump(self.bn / "dist" / f"{fam}_{anchor}.json", {"fields": out["fields"], **ent})
            out["files"][key] = {"name": v["name"], "P": v["P"]}
        dump(self.bn / "dist_index.json", out)       # one file per checkpoint (layer) or site (time), fetched on demand

    # ---------------------------------------------------------------- (g) planted benchmarks (NOTES_B.md chart spec)
    def tab_bench(self):
        path = BASE / "out" / "bench" / "summary.json"
        if not path.exists():
            log("  no bench summary.json: planted benchmarks left out")
            return
        want = {}
        for c in BENCH_CHARTS:
            for s in c["series"]:
                for m in c["methods"]:
                    want.setdefault(c["bench"], set()).add((m, s["Tstat"] if (m.startswith("T_") and s.get("Tstat")) else s["stat"]))
            for s in c.get("ref", []):
                for m in (c["methods"] if s.get("perMethod") else [s.get("method", "oracle")]):
                    want.setdefault(c["bench"], set()).add((m, s["stat"]))
        def comp():
            rows = []
            for o in stream_rows(path):
                w = want.get(o["bench"])
                if w and (o["method"], o["stat"]) in w:
                    rows.append({k: o[k] for k in ("bench", "level", "N", "block", "r", "method", "stat", "n", "median", "q10", "q90")})
            import re
            mo = re.search(r'"n_units":\s*(\d+)', path.read_text()[:4000])
            return {"rows": rows, "head": {"n_units": int(mo.group(1)) if mo else None}}
        res = cached("bench2", [path], comp)
        dump(self.bn / "bench.json", {"charts": BENCH_CHARTS, "rows": res["rows"], "n_units": res["head"].get("n_units")})
        self.facts["bn.bench_units"] = f"{res['head'].get('n_units') or 0:,}"
        log(f"  bench: {len(res['rows'])} summary rows kept")

    # ---------------------------------------------------------------- (h) transfer pilot
    def tab_transfer(self):
        T = BASE / "out" / "transfer"
        out = {}
        syn = json.loads((T / "synthetic.json").read_text()) if (T / "synthetic.json").exists() else None
        cur = json.loads((T / "curve_p8.json").read_text()) if (T / "curve_p8.json").exists() else None
        if cur:
            agg = {}
            for u in cur["units"]:
                key = f"{u['estimator']}_m{u['m']}"
                agg.setdefault(key, {}).setdefault(u["n_fit"], []).append(u["test_corr"])
            out["curve"] = {"sigma": cur["sigma"], "truth": cur["units"][0]["truth"],
                            "series": {k: {str(n): np.nanmean(np.array([x + [np.nan] * (6 - len(x)) for x in v], float), 0).tolist() for n, v in sorted(d.items())}
                                       for k, d in agg.items()},
                            "n_seeds": {k: max(len(v) for v in d.values()) for k, d in agg.items()}}
        if syn:
            ctl = {}
            for k, u in syn.get("p8", {}).items():
                ctl[k] = {"test_corr": u.get("test_corr"), "control_q95": ((u.get("control") or {}).get("test_corr") or {}).get("q95"),
                          "n_perm": (u.get("control") or {}).get("n_perm")}
            out["syn_p8"] = ctl
            out["syn_truth"] = syn.get("truth")
            ch = syn.get("chain") or {}
            out["chain"] = {"truth": ch.get("truth"), "n_fit": ch.get("n_fit"), "m": ch.get("m"),
                            "rows": {k: [{"mb": (r.get("dims") or {}).get("m_b"), "unavailable": bool(r.get("unavailable")),
                                          "partial": (r.get("partial") or {}).get("topk_rel"), "regulariser": (r.get("regulariser") or {}).get("topk_rel"),
                                          "heldout": (r.get("heldout_residual_crossmoment") or {}).get("topk_rel"), "factor": r.get("factor")}
                                         for r in v.get("rows", [])] for k, v in ch.items() if isinstance(v, dict) and "rows" in v}}
        real = T / "real" / "transfer_pilot.json"
        if real.exists():
            R = json.loads(real.read_text())
            head = {k: R[k] for k in ("schema", "status", "n_images", "split_sizes", "n_splits", "n_perm", "n_perm_fit_only", "estimators", "m_values", "k", "stability_min", "sets_status") if k in R}
            # held-out stability = mean over split pairs (i, j) of the mean cos^2 of two q-dim spans on the n_ij images
            # outside fit_i and fit_j (transfer_pilot.heldout_stability); chance q/(n_ij - 1), splits reproduced from
            # transfer_pilot.predeclared_splits (seed read from that file, not imported)
            import re as _re
            seed = int(_re.search(r"^SPLIT_SEED\s*=\s*(\d+)", (BASE / "transfer_pilot.py").read_text(), _re.M).group(1))
            n_img, sz = R["n_images"], R["split_sizes"]
            fits = [np.sort(np.random.default_rng([seed, n_img, s_]).permutation(n_img)[:sz[0]]) for s_ in range(R["n_splits"])]
            n_shared = [n_img - len(np.union1d(fits[i], fits[j])) for i in range(len(fits)) for j in range(i + 1, len(fits))]
            head["stab_n_shared"] = [int(min(n_shared)), int(max(n_shared))]
            stab_chance = lambda q: float(np.mean([q / (n - 1) for n in n_shared])) if q else None
            pairs = []
            for p in R["pairs"]:
                res = {}
                for e, v in p["results"].items():
                    tc = np.array([[np.nan if x is None else x for x in (row or [])] + [np.nan] * (6 - len(row or [])) for row in v["test_corr"]], float)
                    cq = np.array([[np.nan if x is None else x for x in (row or [])] + [np.nan] * (6 - len(row or [])) for row in v["control_q95"]], float)
                    res[e] = {"test_med": np.nanmedian(tc, 0).tolist(), "test_min": np.nanmin(tc, 0).tolist(), "test_max": np.nanmax(tc, 0).tolist(),
                              "ctl_med": np.nanmedian(cq, 0).tolist(), "ctl_max": np.nanmax(cq, 0).tolist(),
                              "n_support": v.get("n_support_splits"), "leading": v.get("leading_supported"), "rule": v.get("support_rule"),
                              "vamp2_med": float(np.nanmedian(v["vamp2_test"])), "ctl_vamp2_med": float(np.nanmedian(v["control_vamp2_q95"])),
                              "agreement": v.get("agreement"), "stability": v.get("stability"),
                              "stab_chance": {side: stab_chance((v.get("stability") or {}).get(side, {}).get("q")) for side in ("src", "dst")},
                              "has_sets": v.get("sets") is not None, "sets_sides": (v.get("sets") or {}).get("sides")}
                pairs.append({"id": p["id"], "kind": p["kind"], "src": p["src"], "dst": p["dst"], "reference": p["reference"], "results": res})
            comp = []
            for c in R.get("composition", []):
                ests = {}
                for split in c["per_split"]:
                    for e, v in split.items():
                        for row in v.get("rows", []):
                            d = ests.setdefault(e, {}).setdefault(str(row["factor"]), {"mb": [], "partial": [], "regulariser": [], "total": [], "heldout": [], "unavailable": 0})
                            if row.get("unavailable"):
                                d["unavailable"] += 1
                                continue
                            d["mb"].append(row["dims"]["m_b"])
                            for k in ("partial", "regulariser", "total"):
                                d[k].append(row[k]["topk_rel"])
                            d["heldout"].append(row["heldout_residual_crossmoment"]["topk_rel"])
                summ = {e: {f: {k: (float(np.median(v)) if isinstance(v, list) and v else v if not isinstance(v, list) else None) for k, v in d.items()} for f, d in fs.items()} for e, fs in ests.items()}
                comp.append({"id": c["id"], "views": c["views"], "n_splits": len(c["per_split"]), "est": summ})
            out["real"] = {"head": head, "pairs": pairs, "composition": comp}
            del R
        r10 = T / "real" / "transfer_pilot_10k.json"
        if r10.exists():
            R = json.loads(r10.read_text())
            out["n10k"] = {"head": {k: R[k] for k in ("n_images", "split_sizes", "curve", "n_perm", "checkpoint") if k in R},
                           "pairs": [{"id": p["id"], "kind": p["kind"], "curve": {n: {e: {"test_corr": v.get("test_corr"), "control_q95": v.get("control_q95"), "n_funcs": v.get("n_funcs")}
                                                                                         for e, v in d.items()} for n, d in p["curve"].items()}} for p in R["pairs"]]}
        dump(self.bn / "transfer.json", out)
        if "real" in out:
            h = out["real"]["head"]
            self.facts.update({"bn.tr_splits": str(h["n_splits"]), "bn.tr_perm": str(h["n_perm"]), "bn.tr_split_sizes": " / ".join(f"{x:,}" for x in h["split_sizes"]),
                               "bn.tr_pairs": str(len(out["real"]["pairs"])), "bn.tr_triples": str(len(out["real"]["composition"]))})
        if "n10k" in out:
            h = out["n10k"]["head"]
            self.facts.update({"bn.tr10k_split": " / ".join(f"{x:,}" for x in h["split_sizes"]), "bn.tr10k_perm": str(h["n_perm"]),
                               "bn.tr10k_curve": ", ".join(f"{x:,}" for x in h["curve"])})
        if "curve" in out:
            self.facts["bn.tr_sigma"] = str(out["curve"]["sigma"])


_G = None


def _job(j):
    kind, x = j
    if kind == "y2":
        fam, anchor, name, path, pairs = x
        _G.y2_rows(fam, anchor, name, pairs, path)
    else:
        rb = {**_G.rep_block(x), "rid": x}
        _G.matrix_block(x, rb)
        _G.cell_agree(x, rb)
    return j[0]


def stream_rows(path: Path):
    """Iterate over summary.json["rows"] one object at a time (the file is about 40 MB)."""
    txt = path.read_text()
    i = txt.index("[", txt.index('"rows"')) + 1
    dec = json.JSONDecoder()
    n = len(txt)
    while i < n:
        while txt[i] in " ,\n\r\t":
            i += 1
        if txt[i] == "]":
            break
        o, i = dec.raw_decode(txt, i)
        yield o


# NOTES_B.md "Chart spec for the site builder", one entry per chart. series: one statistic per panel; methods: the
# lines; ref: grey reference lines (oracle) or chance rows; N: 2000 unless "byN"; block "all" unless stated.
def _c(bench, chart, x, methods, series, ref=(), caption="", **kw):
    return dict(bench=bench, chart=chart, x=x, methods=methods, series=series, ref=list(ref), caption=caption, **kw)


BENCH_CHARTS = [
    _c("P0", 1, "cross-cell correlation ρ", ["M0", "L1", "L3", "G2", "C1first"], [{"stat": "ret_excess_mean", "r": 2, "label": "retention above the size null"}],
       zero=True, caption="Retention of each method's sets above the size null, isotropic Gaussian cells, by cross-cell correlation ρ."),
    _c("P0", 2, "cross-cell correlation ρ", ["M0", "L1", "L3", "G2", "C1first"], [{"stat": "stab_ami_0", "r": 2, "label": "AMI, all images vs 80% subsample"}],
       zero=True, caption="AMI between sets on all images and on an 80% subsample."),
    _c("P0d", 1, "ambient dimension d", ["M0", "L1", "L3"], [{"stat": "ret_excess_mean", "r": 2, "label": "retention above the size null"}],
       ref=[{"stat": "cka_cells_01", "label": "oracle: linear CKA between two independent cells", "panel": True}], zero=True, xlog=True,
       caption="Retention above the size null against ambient dimension (ρ = 0); panel: CKA between two independent cells."),
    _c("P1", 1, "separation a", ["M0", "L1", "L2", "L3", "G1", "G2", "C1first", "C3"], [{"stat": "ami", "r": 4, "label": "AMI with the planted groups"}],
       ref=[{"stat": "pooled_kappa", "label": "oracle κ, pooled"}, {"stat": "min_cell_kappa", "label": "oracle κ, worst cell"}], zero=True, byN=True,
       caption="AMI with the planted groups against separation a; grey: oracle κ."),
    _c("P2", 1, "factor strength a", ["M0", "L1", "C1first", "T_linear", "T_nystrom"],
       [{"stat": "contrast_captured", "r": 4, "label": "share of the shared-factor span captured", "Tstat": "src_span_u_captured"}],
       ref=[{"stat": "mean_cell_r2", "label": "oracle R², after row normalisation"}, {"stat": "raw_mean_cell_r2", "label": "oracle R², before row normalisation"},
            {"stat": "contrast_chance", "method": "M0", "r": 4, "label": "chance (random span of the same dimension)", "chance": True}],
       caption="Share of the shared-factor span captured against factor strength a; grey: oracle R² after and before row normalisation."),
    _c("P3", 1, "nuisance strength b/a", ["M0", "G2", "V_concat_graph", "L1", "L3", "C1first", "C3"], [{"stat": "ami", "r": 2, "label": "AMI with the persistent group g"}],
       ref=[{"stat": "min_cell_kappa", "label": "oracle κ, worst cell"}], zero=True, caption="AMI with the persistent group g against nuisance strength b/a (L = 6)."),
    _c("P3", 2, "nuisance strength b/a", ["M0", "G2", "V_concat_graph", "L1", "L3", "C1first", "C3"], [{"stat": "ami_nuisance_max", "r": 2, "label": "largest AMI with a cell-specific nuisance"}],
       ref=[{"stat": "nuisance_cell_kappa_mean", "label": "oracle κ of the nuisance, mean over cells"}], zero=True, caption="Largest AMI with any cell-specific nuisance h_l."),
    _c("P3L", 1, "number of cells L", ["M0", "L1", "L3", "C1first", "C3"], [{"stat": "ami", "r": 2, "label": "AMI with g"}],
       ref=[{"stat": "pooled_kappa", "label": "oracle κ, pooled"}], zero=True, caption="AMI with g against the number of cells L at b/a = 2."),
    _c("P4", 1, "number of noise cells", ["M0", "L1", "V_product", "V_two_end", "C1mid"], [{"stat": "ami", "r": 4, "label": "AMI with g"}],
       ref=[{"stat": "pooled_kappa", "label": "oracle κ, pooled"}, {"stat": "min_cell_kappa", "label": "oracle κ, worst cell"}], zero=True,
       caption="AMI with g against the number of noise cells; grey: oracle, pooled and worst cell."),
    _c("P4", 2, "cell index", ["M0"], [{"stat": "ret_excess_cell", "r": 4, "label": "retention above the size null, per cell"}], zero=True, percell=6,
       caption="Retention of the method's sets above the size null in each cell, one line per number of noise cells."),
    _c("P5", 1, "noise s", ["M0", "G2", "L1", "L2", "L3", "C1first", "V_product"], [{"stat": "ami", "r": 2, "label": "AMI with ring membership"}],
       ref=[{"stat": "min_cell_kappa", "label": "oracle κ, worst cell"}, {"stat": "pooled_kappa", "label": "oracle κ, pooled"}], zero=True,
       caption="AMI with ring membership against noise s (linked great circles)."),
    _c("P6", 1, "density modulation η", ["M0", "L1", "G2", "C1first"],
       [{"stat": "density_corr_max", "r": 2, "label": "largest correlation of a direction with the density"},
        {"stat": "contrast_captured", "r": 3, "label": "share of the first-harmonic span captured"}],
       ref=[{"stat": "cell0_eta_hat", "label": "oracle: apparent η in the preprocessed view", "panel": True}],
       caption="Largest correlation of a method direction with the sampling density; share of the first-harmonic span captured."),
    _c("P7", 1, "contamination", ["M0", "L1", "L3", "G2"], [{"stat": "best_jaccard_group3", "r": 4, "label": "best Jaccard with the 5% group"}, {"stat": "ami", "r": 4, "label": "AMI with the planted groups"}],
       ref=[{"stat": "min_cell_kappa", "label": "oracle κ, worst cell"}], categorical=["none", "0.5%-persistent", "2%-persistent", "0.5%-cell", "2%-cell"], zero=True,
       caption="Best Jaccard with the 5% group, and AMI, by contamination condition."),
    _c("P8", 1, "phase noise σ", ["M0", "C1first", "C1last", "L1", "V_product", "C3"],
       [{"stat": "span_theta2_captured", "r": 3, "label": "share of span{cos 2θ, sin 2θ} captured"}, {"stat": "span_theta1_captured", "r": 3, "label": "share of span{cos θ, sin θ} captured"}],
       ref=[{"stat": "contrast_chance", "method": "M0", "r": 3, "label": "chance (random span of the same dimension)", "chance": True}],
       categorical=["0.0", "0.25", "0.5", "1.0", "independent"], caption="Share of span{cos 2θ, sin 2θ} captured; companion: span{cos θ, sin θ}."),
    _c("P8", 2, "phase noise σ", ["T_linear", "T_nystrom", "T_diffusion", "T_soft_ulam"], [{"stat": "test_corr0", "r": None, "label": "held-out correlation, first function pair"}],
       ref=[{"stat": "theory_sv0", "method": "T_nystrom", "r": None, "label": "exp(−σ²/2)"}, {"stat": "perm_q95_corr0", "r": None, "label": "permutation 95% point", "perMethod": True}],
       categorical=["0.0", "0.25", "0.5", "1.0", "independent"], zero=True,
       caption="Held-out correlation of the first transfer function pair; grey line: exp(−σ²/2); dotted, in each estimator's colour: that estimator's own permutation 95% point."),
    _c("P9", 1, "signal a", ["M0", "L1", "L3", "G2", "C3"], [{"stat": "ami", "r": 4, "label": "AMI with the joint (u, v) partition"}, {"stat": "contrast_captured", "r": 4, "label": "share of span(u, v, w) captured"}],
       ref=[{"stat": "cell0_own_kappa", "label": "oracle κ, cell 0 (own contrast)"}, {"stat": "cell1_own_kappa", "label": "oracle κ, cell 1"}, {"stat": "cell2_own_kappa", "label": "oracle κ, cell 2"}],
       zero=True, caption="AMI with the joint (u, v) partition and share of span(u, v, w) captured against signal a."),
    _c("P10", 1, "depth signal β_max", ["M0", "L1"], [{"stat": "ami", "r": 4, "label": "AMI with the leaves (r = 4)"}, {"stat": "ami_top", "r": 2, "label": "AMI with the top split (r = 2)"}],
       ref=[{"stat": "leaf_last_cell_kappa", "label": "oracle κ of the leaves, last cell"}, {"stat": "leaf_first_cell_kappa", "label": "oracle κ of the leaves, first cell"}],
       blocks=["early", "late", "all"], zero=True, caption="AMI with the leaves (r = 4) and with the top split (r = 2), per block."),
    _c("P11", 1, "number of disassortative cells", ["M0", "V_meanM2", "V_product", "V_two_end", "G2op", "C2"], [{"stat": "ami", "r": 2, "label": "AMI with the communities"}],
       ref=[{"stat": "min_cell_kappa", "label": "oracle κ, worst cell"}], zero=True,
       caption="AMI with the communities against the number of disassortative cells (operator-level fixture: graphs are generated directly, so only the sources that read operators are scored)."),
    _c("P12", 1, "training steps", ["M0", "L1", "L3", "C1first", "C1last"], [{"stat": "ami", "r": 8, "label": "AMI with y = (A + B) mod 8"}, {"stat": "ami_A", "r": 8, "label": "AMI with A"}],
       ref=[{"stat": "pooled_kappa", "label": "oracle κ, pooled"}], categorical=["0", "250", "1000", "4000"], zero=True,
       caption="AMI with y = (A + B) mod 8 and with A against training steps. The oracle's kNN vote can partly read the (A, B) combination itself, so its value at step 0 is far above chance (NOTES_B.md)."),
]
# P4 chart 2 reads ret_excess_cell0..5
for _ch in BENCH_CHARTS:
    if _ch.get("percell"):
        _ch["series"] = [{"stat": f"ret_excess_cell{j}", "r": 4, "label": f"cell {j}"} for j in range(_ch["percell"])]


def build(b, tag: str):
    return BN(b, tag).run()
