"""Turn the spectral-alignment measurement (SPEC.md) into the compact assets of the site's Alignment tab.

    /path/to/uv run --with numpy --with scipy --with dtw-python \
        --with torch --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        python explore/spectral_align/analyze.py explore/spectral_align/data [--out DIR] [--perms 1000] [--workers 3]

Input: a data directory in the measurement's layout (cells.json, pairs_{0,1}.npz, cka.npy, eig_{0,1}.npy,
span_{0,1}.npy, U_<net>_<step>.npy, probe.npz, iso_accuracy.json, accuracies.json; fixture.json marks synthetic data).
Output (default spectral_align/analysis/<data dir name>/), all read by site/alignment.py and js/alignment.js:

  index.json         networks, cell groups, shipped segments and their offsets, distances, quantization scales,
                     W4 Spearman matrices, the cross-network comparison list, stamp, fixture flag
  facts.json         flat {name: formatted string} table for {{fact:al.NAME}} in content/alignment.html
  summary_r<r>.json  per distance and comparison: DTW cost, path length, architectural gap, stage-end hits, inversion
                     fractions, permutation null quantiles, constant-matrix reference, seed-floor costs, unresolved share
  s/<dist>_r<r>.bin.gz  uint16 distances of every shipped segment (65535 = not finite), scale in index.json
  s/<dist>_r<r>.json    DTW paths and unconstrained argmins of every cross-network comparison
  diag_r<r>.bin      per cell: lambda_0..lambda_81 then span_1..span_80, uint16 / 65534 (65535 = NaN)
  U/<net>_<step>_<site>.bin  64 f32 column scales, then int16 (2000, 64) eigenvectors (rule 0)

Shipped segments (the only parts of the all-pairs tensors that reach the browser):
  self   one network, one checkpoint: site x site (W1, W3)
  traj   one network, one site: checkpoint x checkpoint over the common checkpoints (W2)
  cross  two networks: site x site at every common checkpoint (iso-step, incl. checkpoint 0), and ResNet-20 final
         against each deeper network at its iso-accuracy checkpoint, plus the two deeper seeds at their iso-accuracy
         checkpoints (their seed floor) (C1-C4)

Definitions used here and not fixed by SPEC.md are stated where they are computed (search "Definition").
DTW, the architectural reference and the inversion fraction come from align.py. The block-permutation null is
recomputed with a batched dynamic programme (dtw_cost_batch) that reproduces align.block_permutation_null draw for
draw (same generator, same permutations; tested in test_analyze.py) because dtw-python costs ~5 ms per call.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FORMAT = "al-v3"                       # bump when the output layout or a definition changes
STAB_THRESHOLD = 0.5                   # SPEC: a cut with min span_k < 0.5 (or NaN) in either cell is unresolved
N_DIAG_EIG, N_SPAN = 82, 80
Q16 = 65534                            # uint16 quantization; 65535 = not finite
FACT_DISTS = ["proj_k4", "proj_k8", "proj_k9", "proj_k16", "proj_k32", "proj_k64", "heat_t1", "heat_t16", "heat_t256",
              "eigd_k8", "eigd_k64", "cka"]


def log(*a):
    print(*a, flush=True)


def grids():
    try:
        from align import K_GRID, T_GRID
    except ImportError:                # align.py needs torch; the grids are fixed by SPEC.md
        K_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64)
        T_GRID = (1, 4, 16, 64, 256)
    return tuple(K_GRID), tuple(T_GRID)


def distances(K_GRID, T_GRID):
    """[(id, family, parameter, tensor, column)] in display order."""
    out = [(f"proj_k{k}", "proj", k, "proj", i) for i, k in enumerate(K_GRID)]
    out += [(f"heat_t{t}", "heat", t, "heat", i) for i, t in enumerate(T_GRID)]
    out += [(f"eigd_k{k}", "eigd", k, "eigd", i) for i, k in enumerate(K_GRID)]
    out += [("cka", "cka", None, "cka", 0)]
    return out


def short(net):
    """resnet56-relu-seed1 -> r56s1"""
    d = "".join(ch for ch in net.split("-")[0] if ch.isdigit())
    return f"r{d}s{net.rsplit('seed', 1)[-1]}"


# ------------------------------------------------------------------------------------------------ small pieces
def horizon(span, threshold=STAB_THRESHOLD) -> int:
    """Largest k whose leading-k span is reproducible (span_k >= threshold); 0 if none. As cifar_more/cm3.horizon."""
    ok = np.flatnonzero(np.nan_to_num(np.asarray(span, float), nan=-1) >= threshold)
    return int(ok[-1]) + 1 if len(ok) else 0


def resolved(span, ks) -> np.ndarray:
    """(ncell, len(ks)) bool: span_k >= 0.5 for the cut k | k+1 (NaN counts as unresolved)."""
    s = np.asarray(span, float)[:, np.asarray(ks) - 1]
    return np.nan_to_num(s, nan=-1) >= STAB_THRESHOLD


def q16(v, vmax):
    """Values in [0, vmax] -> uint16 (0..65534); non-finite -> 65535."""
    v = np.asarray(v, np.float64)
    out = np.full(v.shape, 65535, np.uint16)
    ok = np.isfinite(v)
    out[ok] = np.clip(np.round(v[ok] / vmax * Q16), 0, Q16).astype(np.uint16)
    return out


def dq16(q, vmax):
    v = q.astype(np.float64) * vmax / Q16
    v[q == 65535] = np.nan
    return v


def quantize_vectors(u):
    """(n, k) float -> (scales f32 (k,), int16 (n, k)); per-column scale max|u|."""
    u = np.asarray(u, np.float64)
    sc = np.abs(u).max(0)
    sc[sc == 0] = 1
    q = np.clip(np.round(u / sc * 32767), -32767, 32767).astype(np.int16)
    return sc.astype(np.float32), q


def projector_dk(ua, ub, ks):
    g2 = (ua.T @ ub) ** 2
    cum = g2.cumsum(0).cumsum(1)
    ix = np.asarray(ks) - 1
    return 1 - cum[ix, ix] / (ix + 1)


def ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spearman_matrix(cols):
    """cols (n, m) -> (m, m) Spearman rank correlation (average ranks for ties)."""
    r = np.stack([ranks(cols[:, j]) for j in range(cols.shape[1])], 1)
    r -= r.mean(0)
    nrm = np.linalg.norm(r, axis=0)
    nrm[nrm == 0] = np.nan
    return (r.T @ r) / np.outer(nrm, nrm)


def dtw_cost_batch(d):
    """Normalized symmetric2 DTW optimum for a batch of cost matrices d (R, LA, LB), pinned at both corners.

    g[0,0] = d[0,0]; g[i,j] = min(g[i-1,j] + d, g[i-1,j-1] + 2 d, g[i,j-1] + d); result g[-1,-1] / (LA + LB - 1),
    which is align.dtw_alignment's 'cost' (equality tested in test_analyze.py)."""
    d = np.moveaxis(np.asarray(d, np.float64), 0, -1)        # (LA, LB, R): contiguous over the batch
    LA, LB = d.shape[:2]
    g = np.empty_like(d)
    g[0, 0] = d[0, 0]
    for j in range(1, LB):
        g[0, j] = g[0, j - 1] + d[0, j]
    for i in range(1, LA):
        g[i, 0] = g[i - 1, 0] + d[i, 0]
        a = np.minimum(g[i - 1, 1:] + d[i, 1:], g[i - 1, :-1] + 2 * d[i, 1:])
        for j in range(1, LB):
            g[i, j] = np.minimum(a[j - 1], g[i, j - 1] + d[i, j])
    return g[-1, -1] / (LA + LB - 1)


def permutation_null(cost, repetitions=1000, seed=0):
    """align.block_permutation_null, batched: the same generator calls give the same permutations of B's residual
    blocks (stem, pool, logits fixed), and dtw_cost_batch gives the same optimum."""
    d = np.asarray(cost, np.float64)
    rng = np.random.default_rng(seed)
    perms = np.empty((repetitions, d.shape[1]), np.int64)
    for rep in range(repetitions):
        p = np.arange(d.shape[1])
        p[1:-2] = rng.permutation(p[1:-2])
        perms[rep] = p
    return dtw_cost_batch(d[:, perms].transpose(1, 0, 2))


def first_stable(values_ok):
    """Index of the earliest position from which every later entry is True; None if the last is False."""
    ok = list(values_ok)
    if not ok or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return i


# ------------------------------------------------------------------------------------------------ the study layout
class Layout:
    """Networks, cell groups (network, step) and the shipped segments, from cells.json and iso_accuracy.json."""

    def __init__(self, cells, iso, accuracies):
        self.cells = cells
        nets = []
        for c in cells:
            if c["net"] not in nets:
                nets.append(c["net"])
        self.net_names = nets
        self.nets = []
        for name in nets:
            cs = [c for c in cells if c["net"] == name]
            sites = sorted({(c["site_index"], c["site_name"], c["stage"]) for c in cs})
            steps = sorted({c["step"] for c in cs})
            acc = {c["step"]: c.get("val_acc") for c in cs}
            info = iso.get(name) if isinstance(iso, dict) else None
            self.nets.append(dict(name=name, id=short(name), depth=int(cs[0]["depth"]), seed=int(cs[0]["seed"]),
                                  sites=[s[1] for s in sites], stages=[int(s[2]) for s in sites], steps=steps,
                                  val_acc=[acc[s] for s in steps], final_acc=accuracies.get(name, acc[steps[-1]]),
                                  iso=info))
        self.main_steps = sorted(set.intersection(*[set(n["steps"]) for n in self.nets]))
        self.final = self.main_steps[-1]
        self.lookup = {(c["net"], c["step"], c["site_index"]): c["index"] for c in cells}
        self.groups, self.gid = [], {}
        for ni, n in enumerate(self.nets):
            for s in n["steps"]:
                self.gid[(ni, s)] = len(self.groups)
                self.groups.append(dict(net=ni, step=s, main=s in self.main_steps,
                                        iso=bool(n["iso"] and n["iso"].get("available") and n["iso"].get("step") == s),
                                        cells=[self.lookup[(n["name"], s, si)] for si in range(len(n["sites"]))]))
        self.segs = []          # dict(kind, rows=[cells], cols=[cells], ...)
        self.self_seg = {}
        for g, grp in enumerate(self.groups):
            self.self_seg[g] = self._add("self", grp["cells"], grp["cells"], group=g)
        depths = sorted({n["depth"] for n in self.nets})
        self.base_depth = depths[0]
        self.cross = []         # comparisons between two networks (C1-C4)
        self._xseg = {}
        for a in range(len(self.nets)):
            for b in range(a + 1, len(self.nets)):
                for s in self.main_steps:
                    ga, gb = self.gid[(a, s)], self.gid[(b, s)]
                    seg = self._add("cross", self.groups[ga]["cells"], self.groups[gb]["cells"], ga=ga, gb=gb)
                    self._xseg[(ga, gb)] = seg
                    self.cross.append(dict(seg=seg, a=a, b=b, kind="step", step=s, ga=ga, gb=gb))
        # iso-accuracy: base-depth final against each deeper network at its accepted checkpoint
        iso_g = {ni: self.gid[(ni, n["iso"]["step"])] for ni, n in enumerate(self.nets)
                 if n["iso"] and n["iso"].get("available") and (ni, n["iso"].get("step")) in self.gid}
        for a, na in enumerate(self.nets):
            if na["depth"] != self.base_depth:
                continue
            for b, gb in iso_g.items():
                ga = self.gid[(a, self.final)]
                seg = self._add("cross", self.groups[ga]["cells"], self.groups[gb]["cells"], ga=ga, gb=gb)
                self.cross.append(dict(seg=seg, a=a, b=b, kind="iso", step=None, ga=ga, gb=gb))
        for a in iso_g:     # deeper seeds against each other at their accepted checkpoints: the seed floor for iso
            for b in iso_g:
                if a < b and self.nets[a]["depth"] == self.nets[b]["depth"]:
                    ga, gb = iso_g[a], iso_g[b]
                    seg = self._add("cross", self.groups[ga]["cells"], self.groups[gb]["cells"], ga=ga, gb=gb)
                    self._xseg[("iso", a, b)] = seg
                    self.cross.append(dict(seg=seg, a=a, b=b, kind="iso", step=None, ga=ga, gb=gb))
        for ni, n in enumerate(self.nets):
            for si in range(len(n["sites"])):
                cs = [self.lookup[(n["name"], s, si)] for s in self.main_steps]
                self._add("traj", cs, cs, net=ni, site=si)
        self.seg_of_cross = {}
        for x in self.cross:
            # networks of different depth with the same seed share their initial draws for the stem and the first
            # blocks (same generator, same construction order): those pairings are shown, labelled, and kept out of
            # every cross-depth summary, which uses different-seed pairings only
            na, nb = self.nets[x["a"]], self.nets[x["b"]]
            x["same_seed"] = na["depth"] != nb["depth"] and na["seed"] == nb["seed"]
            x["floor_a"], x["floor_b"] = self._floor(x, x["a"]), self._floor(x, x["b"])
        off = 0
        for s in self.segs:
            s["offset"] = off
            off += len(s["rows"]) * len(s["cols"])
        self.n_entries = off
        self.rowidx = np.concatenate([np.repeat(s["rows"], len(s["cols"])) for s in self.segs]).astype(np.int64)
        self.colidx = np.concatenate([np.tile(s["cols"], len(s["rows"])) for s in self.segs]).astype(np.int64)

    def _add(self, kind, rows, cols, **kw):
        self.segs.append(dict(kind=kind, rows=list(rows), cols=list(cols), **kw))
        return len(self.segs) - 1

    def sibling(self, ni):
        n = self.nets[ni]
        sib = [j for j, m in enumerate(self.nets) if j != ni and m["depth"] == n["depth"]]
        return sib[0] if sib else None

    def _floor(self, x, ni):
        """Definition (seed floor of one side of a comparison): the same-depth cross-seed segment at the same
        checkpoint (iso-step); for a network at its iso-accuracy checkpoint, its sibling at the sibling's own
        iso-accuracy checkpoint; for the base-depth network in an iso comparison, its sibling at the final step."""
        sib = self.sibling(ni)
        if sib is None:
            return -1
        lo, hi = min(ni, sib), max(ni, sib)
        if x["kind"] == "step":
            return self._xseg.get((self.gid[(lo, x["step"])], self.gid[(hi, x["step"])]), -1)
        if self.nets[ni]["depth"] == self.base_depth:
            return self._xseg.get((self.gid[(lo, self.final)], self.gid[(hi, self.final)]), -1)
        return self._xseg.get(("iso", lo, hi), -1)

    def seg_matrix(self, values, seg):
        s = self.segs[seg]
        o, nr, nc = s["offset"], len(s["rows"]), len(s["cols"])
        return values[o:o + nr * nc].reshape(nr, nc)

    def masks(self):
        """W4 pair sets (Definition):
        within  every unordered pair of distinct cells of one network that share a common checkpoint or a site
                (the pairs W1 and W2 display), all networks;
        cross   every (site of A, site of B) pair of two networks with different seeds at the final checkpoint
                (same-depth cross-seed pairs included; same-seed pairs of different depth, which share initial draws,
                excluded)."""
        within, cross = [], []
        for s in self.segs:
            nr, nc, o = len(s["rows"]), len(s["cols"]), s["offset"]
            if s["kind"] == "self" and self.groups[s["group"]]["main"] or s["kind"] == "traj":
                iu = np.triu_indices(nr, 1)
                within.append(o + iu[0] * nc + iu[1])
        for x in self.cross:
            if x["kind"] == "step" and x["step"] == self.final and self.nets[x["a"]]["seed"] != self.nets[x["b"]]["seed"]:
                s = self.segs[x["seg"]]
                cross.append(s["offset"] + np.arange(len(s["rows"]) * len(s["cols"])))
        return {"within": np.concatenate(within), "cross": np.concatenate(cross)}


# ------------------------------------------------------------------------------------------------ per-distance work
_W = {}


def _dtw_slice(job):
    """DTW, argmins, reference gaps and the permutation null of every cross comparison, for one distance."""
    did, r, fam, k_idx = job
    from align import dtw_alignment, architectural_reference
    L, values, res, perms = _W["layout"], _W["values"][(did, r)], _W["resolved"], _W["perms"]
    recs = []
    for xi, x in enumerate(L.cross):
        D = L.seg_matrix(values, x["seg"]).astype(np.float64)
        da, db = L.nets[x["a"]]["depth"], L.nets[x["b"]]["depth"]
        if not np.isfinite(D).all():
            recs.append(None)
            continue
        fit = dtw_alignment(D)
        ref = architectural_reference(da, db, fit["path"])
        null = permutation_null(D, perms, seed=0) if perms else np.array([np.nan])
        rec = dict(cost=fit["cost"], plen=fit["path_length"], path=fit["path"], ra=fit["row_argmins"],
                   ca=fit["column_argmins"], inv_r=fit["row_inversion_fraction"], inv_c=fit["column_inversion_fraction"],
                   gap=ref["mean_relative_depth_gap"], ends=[bool(e) for e in ref["passes_stage_ends"]],
                   null=[float(np.quantile(null, q)) for q in (.05, .5, .95)] if perms else [None] * 3,
                   pnull=float(np.mean(null <= fit["cost"])) if perms else None, const=float(D.mean()))
        if fam == "proj":
            ga, gb = L.groups[x["ga"]]["cells"], L.groups[x["gb"]]["cells"]
            rec["unres"] = float(np.mean([not (res[r][ga[i], k_idx] and res[r][gb[j], k_idx]) for i, j in fit["path"]]))
        recs.append(rec)
    return did, r, recs


def run(data: Path, out: Path, perms=1000, workers=3):
    t0 = time.time()
    K_GRID, T_GRID = grids()
    DISTS = distances(K_GRID, T_GRID)
    cells = json.loads((data / "cells.json").read_text())
    iso = json.loads((data / "iso_accuracy.json").read_text()) if (data / "iso_accuracy.json").exists() else {}
    accs = json.loads((data / "accuracies.json").read_text()) if (data / "accuracies.json").exists() else {}
    fixture = (data / "fixture.json").exists()
    L = Layout(cells, iso, accs)
    nc = len(cells)
    assert [c["index"] for c in cells] == list(range(nc)), "cells.json indices must be 0..n-1 in order"
    log(f"{nc} cells, {len(L.nets)} networks, {len(L.segs)} segments, {L.n_entries:,} shipped entries per distance, "
        f"{len(L.cross)} cross-network comparisons")
    out.mkdir(parents=True, exist_ok=True)
    (out / "s").mkdir(exist_ok=True)
    rules = [r for r in (0, 1) if (data / f"pairs_{r}.npz").exists()]

    # ---- per-cell diagnostics
    eig = {r: np.load(data / f"eig_{r}.npy", mmap_mode="r") for r in rules}
    span = {r: np.load(data / f"span_{r}.npy").astype(np.float64) for r in rules}
    N = int(eig[rules[0]].shape[1])
    # Snapshots whose operator could not be built under a rule (NaN eigenvalues; data/rule<r>_unavailable.json lists
    # them): unavailable for that rule. Their distances stay NaN: drawn as missing, never as zeros.
    unavail = {}
    for r in rules:
        bad = set(np.flatnonzero(~np.isfinite(np.asarray(eig[r][:, 1], np.float64))).tolist())
        f = data / f"rule{r}_unavailable.json"
        if f.exists():
            bad |= {int(c["index"]) for c in json.loads(f.read_text())}
        unavail[r] = sorted(bad)
        if bad:
            log(f"  rule {r}: {len(bad)} snapshots unavailable")
    res_k = {r: resolved(span[r], K_GRID) for r in rules}
    hz = {r: np.array([horizon(s) for s in span[r]]) for r in rules}
    for r in rules:
        hz[r][unavail[r]] = -1
    for r in rules:
        e = np.asarray(eig[r][:, :N_DIAG_EIG], np.float64)
        blob = np.concatenate([q16(e, 1.0), q16(span[r][:, :N_SPAN], 1.0)], 1)
        (out / f"diag_r{r}.bin").write_bytes(np.ascontiguousarray(blob, "<u2").tobytes())

    # ---- gather the shipped entries of every distance (full precision), one tensor in memory at a time
    values = {}
    for r in rules:
        with np.load(data / f"pairs_{r}.npz") as z:
            for tname in ("proj", "eigd", "heat"):
                T = z[tname]
                assert T.shape[:2] == (nc, nc), (tname, T.shape)
                V = np.asarray(T[L.rowidx, L.colidx], np.float32)            # (E, P)
                del T
                for did, fam, p, tn, col in DISTS:
                    if tn == tname:
                        values[(did, r)] = V[:, col].copy()
                log(f"  rule {r} {tname}: gathered ({time.time() - t0:.0f} s)")
    if (data / "cka.npy").exists():
        C = np.load(data / "cka.npy", mmap_mode="r")
        v = np.asarray(C[L.rowidx, L.colidx], np.float32)
        for r in rules:
            values[("cka", r)] = v
    dists = [d for d in DISTS if (d[0], rules[0]) in values]

    # ---- quantized slices
    scale, shipped = {}, 0
    for did, fam, p, tn, col in dists:
        for r in (rules if fam != "cka" else rules[:1]):
            v = values[(did, r)]
            vmax = 1.0 if fam == "proj" else float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
            vmax = vmax if vmax > 0 else 1.0
            name = did if fam == "cka" else f"{did}_r{r}"
            scale[name] = vmax
            gz = gzip.compress(np.ascontiguousarray(q16(v, vmax), "<u2").tobytes(), 6, mtime=0)
            (out / "s" / f"{name}.bin.gz").write_bytes(gz)
            shipped += len(gz)
    log(f"slices written ({shipped / 1e6:.1f} MB gzipped, {time.time() - t0:.0f} s)")

    # ---- DTW and nulls
    _W.update(layout=L, values=values, resolved=res_k, perms=perms)
    jobs = [(did, r, fam, K_GRID.index(p) if fam == "proj" else None) for did, fam, p, tn, col in dists
            for r in (rules if fam != "cka" else rules[:1])]
    dtw = {}
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for i, (did, r, recs) in enumerate(pool.imap_unordered(_dtw_slice, jobs)):
                dtw[(did, r)] = recs
                if i % 10 == 0:
                    log(f"  dtw {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)")
    else:
        for job in jobs:
            did, r, recs = _dtw_slice(job)
            dtw[(did, r)] = recs
    for r in rules:
        dtw[("cka", r)] = dtw[("cka", rules[0])]
    log(f"dtw done ({time.time() - t0:.0f} s)")

    # ---- per-slice path files and per-rule summaries
    FIELDS = ["cost", "plen", "gap", "ends", "inv_r", "inv_c", "n05", "n50", "n95", "pnull", "const", "unres",
              "floor_a", "floor_b"]
    summaries = {}
    xi_of_seg = {x["seg"]: xi for xi, x in enumerate(L.cross)}
    for r in rules:
        summ = {"fields": FIELDS, "dists": {}}
        for did, fam, p, tn, col in dists:
            recs = dtw[(did, r)]
            name = did if fam == "cka" else f"{did}_r{r}"
            if fam != "cka" or r == rules[0]:
                paths = [None if x is None else dict(p=[v for ij in x["path"] for v in ij], ra=x["ra"], ca=x["ca"])
                         for x in recs]
                (out / "s" / f"{name}.json").write_text(json.dumps(paths, separators=(",", ":")))
            rows = []
            for xi, x in enumerate(recs):
                if x is None:
                    rows.append([None] * len(FIELDS))
                    continue
                cx = L.cross[xi]
                fcost = lambda seg: (recs[xi_of_seg[seg]] or {}).get("cost") if seg >= 0 else None
                fa, fb = fcost(cx["floor_a"]), fcost(cx["floor_b"])
                rows.append([x["cost"], x["plen"], x["gap"], sum(x["ends"]), x["inv_r"], x["inv_c"], *x["null"], x["pnull"],
                             x["const"], x.get("unres"), fa, fb])
            summ["dists"][did] = [[None if v is None else v if isinstance(v, int) else round(float(v), 5) for v in row]
                                  for row in rows]
        summaries[r] = summ
        (out / f"summary_r{r}.json").write_text(json.dumps(summ, separators=(",", ":")))

    # ---- W4: Spearman across distances
    masks = L.masks()
    w4 = {}
    for r in rules:
        for pname, m in masks.items():
            cols = np.stack([values[(d[0], r)][m] for d in dists], 1).astype(np.float64)
            ok = np.isfinite(cols).all(1)
            S = spearman_matrix(cols[ok])
            w4[f"{pname}_r{r}"] = dict(n=int(ok.sum()), dropped=int((~ok).sum()), z=[[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in S])

    # ---- eigenvectors (rule 0), int16
    (out / "U").mkdir(exist_ok=True)
    vec_cells, vec_bytes, worst_q, worst_store = [], 0, 0.0, 0.0
    rng = np.random.default_rng(0)
    for path in sorted(data.glob("U_*.npy")):
        net, step = path.stem[2:].rsplit("_", 1)
        step = int(step)
        U = np.load(path, mmap_mode="r")
        ni = L.net_names.index(net)
        assert U.shape[0] == len(L.nets[ni]["sites"]), (path.name, U.shape)
        for si in range(U.shape[0]):
            u = np.asarray(U[si], np.float64)
            sc, q = quantize_vectors(u)
            blob = sc.astype("<f4").tobytes() + q.astype("<i2").tobytes()
            (out / "U" / f"{short(net)}_{step}_{si}.bin").write_bytes(blob)
            vec_bytes += len(blob)
            vec_cells.append(dict(net=ni, step=step, site=si, cell=L.lookup[(net, step, si)]))
        del U
    # reconstruction check on random pairs of shipped cells: int16 vs float, and float vs the stored tensor
    if vec_cells and 0 in rules:
        with np.load(data / "pairs_0.npz") as z:
            P0 = z["proj"]
            for _ in range(40):
                a, b = rng.choice(len(vec_cells), 2, replace=False)
                A, B = vec_cells[a], vec_cells[b]
                ua = np.load(data / f"U_{L.net_names[A['net']]}_{A['step']}.npy", mmap_mode="r")[A["site"]].astype(np.float64)
                ub = np.load(data / f"U_{L.net_names[B['net']]}_{B['step']}.npy", mmap_mode="r")[B["site"]].astype(np.float64)
                sa, qa = quantize_vectors(ua)
                sb, qb = quantize_vectors(ub)
                full = projector_dk(ua, ub, K_GRID)
                quant = projector_dk(qa * (sa / 32767), qb * (sb / 32767), K_GRID)
                worst_q = max(worst_q, float(np.abs(full - quant).max()))
                worst_store = max(worst_store, float(np.abs(full - P0[A["cell"], B["cell"]]).max()))
            del P0
    log(f"eigenvectors: {len(vec_cells)} cells, {vec_bytes / 1e6:.1f} MB; int16 d_k error {worst_q:.2e}; "
        f"vs stored proj {worst_store:.2e}")

    # ---- probe
    probe = {}
    if (data / "probe.npz").exists():
        with np.load(data / "probe.npz") as p:
            probe = dict(idx=p["idx"].astype(int).tolist(), labels=p["labels"].astype(int).tolist())

    # ---- architectural reference of every comparison (align.architectural_reference)
    from align import architectural_reference
    ref_maps = []
    for x in L.cross:
        ref = architectural_reference(L.nets[x["a"]]["depth"], L.nets[x["b"]]["depth"])
        ref_maps.append(dict(ref_map=[int(v) for v in ref["map"]], stage_ends=ref["stage_ends"]))

    # ---- index
    index = dict(
        format=FORMAT, fixture=fixture, n=N, k_grid=list(K_GRID), t_grid=list(T_GRID), rules=rules,
        rule_names={0: "union 15-NN graph", 1: "perplexity 100"}, stab_threshold=STAB_THRESHOLD,
        n_diag_eig=N_DIAG_EIG, n_span=N_SPAN, main_steps=L.main_steps, final=L.final, base_depth=L.base_depth,
        nets=L.nets, groups=L.groups,
        segs=[dict(kind=s["kind"], o=s["offset"], nr=len(s["rows"]), nc=len(s["cols"]),
                   **{k: s[k] for k in ("group", "ga", "gb", "net", "site") if k in s}) for s in L.segs],
        cross=[dict({k: x[k] for k in ("seg", "a", "b", "kind", "step", "ga", "gb", "floor_a", "floor_b", "same_seed")}, **ref_maps[i])
               for i, x in enumerate(L.cross)],
        dists=[dict(id=d[0], fam=d[1], p=d[2]) for d in dists], scale=scale, n_entries=L.n_entries,
        w4=w4, w4_dists=[d[0] for d in dists], vec_cells=vec_cells, probe=probe, n_perm=perms,
        horizon={str(r): hz[r].tolist() for r in rules},
        unavailable={str(r): unavail[r] for r in rules},
        missing_cross={str(r): [xi for xi, x in enumerate(dtw[("proj_k8" if ("proj_k8", r) in dtw else dists[0][0], r)]) if x is None]
                       for r in rules},
        checks=dict(int16_dk_max_error=worst_q, stored_dk_max_error=worst_store),
        sizes=dict(slices_gz=shipped, vectors=vec_bytes),
    )
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False, default=_nan_none))

    facts = make_facts(L, values, dtw, summaries, w4, dists, rules, res_k, hz, K_GRID, index, fixture)
    subset_fail = json.loads((data / "subset_failures.json").read_text()) if (data / "subset_failures.json").exists() else []
    facts["al.subset_failures"] = str(len(subset_fail))
    facts["al.subset_failures_note"] = ("No subset rebuild failed, so every snapshot has its span_k." if not subset_fail else
                                        f"For {len(subset_fail)} snapshot{'s' if len(subset_fail) > 1 else ''} the subset rebuild failed, "
                                        "so span_k is missing and their cuts count as unresolved.")
    for r in rules:
        bad = unavail[r]
        facts[f"al.missing.rule{r}.snapshots"] = str(len(bad))
        groups = {}
        for c in bad:
            groups.setdefault((short(cells[c]["net"]), cells[c]["step"]), []).append(cells[c]["site_name"])
        facts[f"al.missing.rule{r}.list"] = "; ".join(
            f"{n} step {st:,}: {len(v)} site{'s' if len(v) > 1 else ''} ({', '.join(v) if len(v) <= 4 else v[0] + ' to ' + v[-1]})"
            for (n, st), v in groups.items()) or "none"
        facts[f"al.missing.rule{r}.comparisons"] = str(len(index["missing_cross"][str(r)]))
        facts[f"al.missing.rule{r}.comparisons_of"] = str(len(L.cross))
    (out / "facts.json").write_text(json.dumps(facts, indent=1))
    (out / "stamp.json").write_text(json.dumps(dict(stamp=input_stamp(data), format=FORMAT, perms=perms)))
    log(f"analysis written to {out} ({time.time() - t0:.0f} s); {len(facts)} facts")
    return out


def _nan_none(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def input_stamp(data: Path) -> str:
    h = hashlib.sha256(FORMAT.encode())
    for p in sorted(data.iterdir()):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ facts
def f3(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"


def f2(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"


def make_facts(L, values, dtw, summaries, w4, dists, rules, res_k, hz, K_GRID, index, fixture):
    """Flat table of formatted strings; every name starts with 'al.'. Values only, no interpretation."""
    F = {}
    did_ok = {d[0] for d in dists}
    fdists = [d for d in FACT_DISTS if d in did_ok]
    F["al.data_kind"] = "synthetic fixture" if fixture else "measured"
    F["al.n_cells"] = f"{len(L.cells):,}"
    F["al.n_nets"] = str(len(L.nets))
    F["al.n_images"] = f"{index['n']:,}"
    F["al.n_perm"] = f"{index['n_perm']:,}"
    F["al.k_grid"] = ", ".join(map(str, K_GRID))
    F["al.t_grid"] = ", ".join(map(str, index["t_grid"]))
    F["al.steps"] = ", ".join(f"{s:,}" for s in L.main_steps)
    F["al.final_step"] = f"{L.final:,}"
    F["al.vec_cells"] = str(len(index["vec_cells"]))
    F["al.int16_dk_max_error"] = f"{index['checks']['int16_dk_max_error']:.1e}"
    F["al.stored_dk_max_error"] = f"{index['checks']['stored_dk_max_error']:.1e}"
    for k in (1, 8, 16, 64):
        F[f"al.chance.k{k}"] = f"{1 - k / (index['n'] - 1):.4f}"
    for n in L.nets:
        F[f"al.acc.{n['id']}.final"] = f"{100 * n['final_acc']:.2f}%" if n["final_acc"] is not None else "n/a"
        F[f"al.sites.{n['id']}"] = str(len(n["sites"]))
        if n["iso"]:
            i = n["iso"]
            F[f"al.iso.{n['id']}.status"] = "available" if i.get("available") else "unavailable"
            F[f"al.iso.{n['id']}.target"] = f"{100 * i['target']:.2f}%" if i.get("target") is not None else "n/a"
            F[f"al.iso.{n['id']}.step"] = f"{i['step']:,}" if i.get("step") is not None else "n/a"
            F[f"al.iso.{n['id']}.accuracy"] = f"{100 * i['accuracy']:.2f}%" if i.get("accuracy") is not None else "n/a"
            F[f"al.iso.{n['id']}.overshoot"] = f"{100 * i['overshoot']:+.2f} points" if i.get("overshoot") is not None else "n/a"
            F[f"al.iso.{n['id']}.reason"] = (
                "accepted (overshoot within 0.5 points)" if i.get("available") else
                "no saved checkpoint reached the target" if i.get("step") is None else
                f"the earliest checkpoint at or above the target overshoots it by {100 * i['overshoot']:.2f} points (more than 0.5)")

    base = [ni for ni, n in enumerate(L.nets) if n["depth"] == L.base_depth]
    final_g = {ni: L.gid[(ni, L.final)] for ni in range(len(L.nets))}
    kidx = {k: i for i, k in enumerate(K_GRID)}
    for r in rules:
        rn = f"rule{r}"
        # W1: largest adjacent-layer distance at the final checkpoint
        for ni in base:
            n = L.nets[ni]
            for did in fdists:
                D = L.seg_matrix(values[(did, r)], L.self_seg[final_g[ni]])
                adj = np.array([D[i, i + 1] for i in range(len(D) - 1)])
                o = np.argsort(-np.nan_to_num(adj, nan=-1))
                key = f"al.w1.{n['id']}.{rn}.{did}"
                F[f"{key}.largest_jump"] = f"{n['sites'][o[0]]}–{n['sites'][o[0] + 1]}"
                F[f"{key}.largest_jump_value"] = f3(adj[o[0]])
                F[f"{key}.second_jump"] = f"{n['sites'][o[1]]}–{n['sites'][o[1] + 1]}"
                F[f"{key}.second_jump_value"] = f3(adj[o[1]])
                F[f"{key}.adjacent_median"] = f3(float(np.nanmedian(adj)))
        # W2: Definition (settled): earliest common checkpoint from which on the distance to the same site at the
        # final checkpoint stays at or below that site's seed floor at the final checkpoint.
        for ni in base:
            n, sib = L.nets[ni], L.sibling(ni)
            if sib is None:
                continue
            fl_seg = L._xseg[(final_g[min(ni, sib)], final_g[max(ni, sib)])]
            for did in fdists:
                fl = np.diag(L.seg_matrix(values[(did, r)], fl_seg))
                parts = []
                for si in range(len(n["sites"])):
                    tseg = next(j for j, s in enumerate(L.segs) if s["kind"] == "traj" and s["net"] == ni and s["site"] == si)
                    row = L.seg_matrix(values[(did, r)], tseg)[:, -1]
                    if did.startswith("proj_"):
                        # the trajectory's own cuts: checkpoint t and the final checkpoint must both be resolved
                        kj = kidx[int(did.split("_k")[1])]
                        tc = L.segs[tseg]["rows"]
                        unres_t = [t for t in range(len(tc) - 1) if not (res_k[r][tc[t], kj] and res_k[r][tc[-1], kj])]
                    else:
                        unres_t = []
                    if not (np.isfinite(row).all() and np.isfinite(fl[si])):
                        parts.append(f"{n['sites'][si]} n/a (missing)")
                        continue
                    if did.startswith("proj_"):
                        kj = kidx[int(did.split("_k")[1])]
                        fs = L.segs[fl_seg]
                        if not (res_k[r][fs["rows"][si], kj] and res_k[r][fs["cols"][si], kj]):
                            parts.append(f"{n['sites'][si]} n/a (floor cut unresolved)")
                            continue
                    i = first_stable(row <= fl[si])
                    late = [t for t in unres_t if i is None or t >= i]
                    if late:
                        parts.append(f"{n['sites'][si]} n/a (cut unresolved at step {L.main_steps[late[-1]]:,})")
                        continue
                    parts.append(f"{n['sites'][si]} {L.main_steps[i]:,}" if i is not None else f"{n['sites'][si]} never")
                F[f"al.w2.{n['id']}.{rn}.{did}.settled"] = "; ".join(parts)
        # W3: stability horizons at the final checkpoint; first k where most adjacent cuts are unresolved
        for ni, n in enumerate(L.nets):
            cs = L.groups[final_g[ni]]["cells"]
            h = hz[r][cs]
            h = h[h >= 0]
            F[f"al.w3.{n['id']}.{rn}.horizon_median"] = f"{np.median(h):g}" if len(h) else "n/a"
            F[f"al.w3.{n['id']}.{rn}.horizon_range"] = f"{h.min()}–{h.max()}" if len(h) else "n/a"
            frac = [np.mean([not (res_k[r][cs[i], j] and res_k[r][cs[i + 1], j]) for i in range(len(cs) - 1)]) for j in range(len(K_GRID))]
            kmaj = next((K_GRID[j] for j, f in enumerate(frac) if f > .5), None)
            F[f"al.w3.{n['id']}.{rn}.k_most_unresolved"] = str(kmaj) if kmaj else "none in the grid"
            for k in (8, 16, 32, 64):
                F[f"al.w3.{n['id']}.{rn}.unresolved_share_k{k}"] = f"{100 * frac[kidx[k]]:.0f}%"
        # W4: Spearman between selected distances
        names = [d[0] for d in dists]
        for pname in ("within", "cross"):
            S = np.array([[np.nan if v is None else v for v in row] for row in w4[f"{pname}_r{r}"]["z"]])
            F[f"al.w4.{pname}.{rn}.n_pairs"] = f"{w4[f'{pname}_r{r}']['n']:,}"
            pi = [i for i, d in enumerate(names) if d.startswith("proj_")]
            sub = S[np.ix_(pi, pi)]
            F[f"al.w4.{pname}.{rn}.proj_min"] = f2(float(np.nanmin(sub)))
            for a_, b_ in [("proj_k1", "proj_k64"), ("heat_t16", "cka"), ("heat_t1", "heat_t256")] + [
                    (f"proj_k{k}", o) for k in (8, 9) for o in ("proj_k64", "proj_k16", "heat_t16", "cka", f"eigd_k{k}")]:
                if a_ in names and b_ in names:
                    F[f"al.w4.{pname}.{rn}.{a_}.{b_}"] = f2(S[names.index(a_), names.index(b_)])

    # C1-C4 per depth pair: values of the (up to) four seed pairings, separated by " · "
    FI = {f: i for i, f in enumerate(summaries[rules[0]]["fields"])}
    by_dp = {}
    for xi, x in enumerate(L.cross):
        da, db = L.nets[x["a"]]["depth"], L.nets[x["b"]]["depth"]
        ck = "final" if x["kind"] == "step" and x["step"] == L.final else "init" if x["kind"] == "step" and x["step"] == L.main_steps[0] else "iso" if x["kind"] == "iso" else None
        if ck is None:
            continue
        by_dp.setdefault((f"{da}-{db}", ck, "sameseed" if x["same_seed"] else ""), []).append(xi)
    for r in rules:
        rn = f"rule{r}"
        for (dp, ck, sk), xs in sorted(by_dp.items()):
            for did in fdists:
                rows = [summaries[r]["dists"][did][xi] for xi in xs]
                key = f"al.c2.{dp}.{ck}.{sk + '.' if sk else ''}{rn}.{did}"
                jn = lambda f, fmt=f3: " · ".join(fmt(row[FI[f]]) for row in rows)
                F[f"{key}.gap"] = jn("gap")
                F[f"{key}.cost"] = jn("cost")
                F[f"{key}.null05"] = jn("n05")
                F[f"{key}.pnull"] = jn("pnull")
                F[f"{key}.ends"] = " · ".join(f"{row[FI['ends']]}/3" if row[FI['ends']] is not None else "n/a" for row in rows)
                F[f"{key}.inv_r"] = jn("inv_r", f2)
                F[f"{key}.floor_a"] = jn("floor_a")
                F[f"{key}.floor_b"] = jn("floor_b")
                F[f"{key}.pairings"] = " · ".join(f"{L.nets[L.cross[xi]['a']]['id']}–{L.nets[L.cross[xi]['b']]['id']}" for xi in xs)
        # C1: DTW match of every site of A (final checkpoint), every pairing
        for xi0, x in enumerate(L.cross):
            if not (x["kind"] == "step" and x["step"] == L.final):
                continue
            na, nb = L.nets[x["a"]], L.nets[x["b"]]
            for did in fdists:
                rec = dtw[(did, r)][xi0]
                if rec is None:
                    continue
                m = {}
                for i, j in rec["path"]:
                    m.setdefault(i, []).append(j)
                parts = []
                for i in range(len(na["sites"])):
                    js = m[i]
                    tgt = nb["sites"][js[0]] if len(js) == 1 else f"{nb['sites'][js[0]]}–{nb['sites'][js[-1]]}"
                    parts.append(f"{na['sites'][i]}→{tgt}")
                F[f"al.c1.{na['id']}-{nb['id']}.{rn}.{did}.map"] = ", ".join(parts)
        # C3: largest ratio of the distance along the DTW path to the seed floor at the same depth
        # Definition: floor at a path point (i, j) = mean of A's floor at site i and B's floor at site j.
        for (dp, ck, sk), xs in sorted(by_dp.items()):
            if ck != "final":
                continue
            for xi in xs:
                x = L.cross[xi]
                na, nb = L.nets[x["a"]], L.nets[x["b"]]
                if x["floor_a"] < 0 or x["floor_b"] < 0:
                    continue
                for did in fdists:
                    rec = dtw[(did, r)][xi]
                    if rec is None:
                        continue
                    v = values[(did, r)]
                    D = L.seg_matrix(v, x["seg"])
                    fa, fb = np.diag(L.seg_matrix(v, x["floor_a"])), np.diag(L.seg_matrix(v, x["floor_b"]))
                    ratio = [(D[i, j] / max(1e-12, (fa[i] + fb[j]) / 2), i, j) for i, j in rec["path"]]
                    ratio = [t for t in ratio if np.isfinite(t[0])]
                    n_all = len(ratio)
                    if did.startswith("proj_"):
                        # only path points whose cut is resolved in the pair and in all four snapshots of the floor
                        kj = kidx[int(did.split("_k")[1])]
                        ok = lambda c: bool(res_k[r][c, kj])
                        sa, sb = L.segs[x["floor_a"]], L.segs[x["floor_b"]]
                        ga, gb = L.groups[x["ga"]]["cells"], L.groups[x["gb"]]["cells"]
                        ratio = [t for t in ratio if ok(ga[t[1]]) and ok(gb[t[2]]) and ok(sa["rows"][t[1]]) and ok(sa["cols"][t[1]])
                                 and ok(sb["rows"][t[2]]) and ok(sb["cols"][t[2]])]
                    key = f"al.c3.{na['id']}-{nb['id']}.{rn}.{did}"
                    F[f"{key}.n_points"] = f"{len(ratio)} of {n_all}"
                    if not ratio:
                        F[f"{key}.max_ratio"] = F[f"{key}.median_ratio"] = "n/a (no resolved path point)"
                        F[f"{key}.max_at"] = "n/a"
                        continue
                    best = max(ratio, key=lambda t: t[0])
                    F[f"{key}.max_ratio"] = f"{best[0]:.1f}"
                    F[f"{key}.max_at"] = f"{na['sites'][best[1]]} / {nb['sites'][best[2]]}"
                    F[f"{key}.median_ratio"] = f"{np.median([t[0] for t in ratio]):.1f}"
        # C4: Definition (appears): earliest common checkpoint from which on the DTW cost is below the 5th
        # percentile of the permutation null in every seed pairing of the depth pair.
        pairs_by_dp = {}
        for xi, x in enumerate(L.cross):
            if x["kind"] == "step":
                da, db = L.nets[x["a"]]["depth"], L.nets[x["b"]]["depth"]
                pairs_by_dp.setdefault(f"{da}-{db}" + (".sameseed" if x["same_seed"] else ""), {}).setdefault(x["step"], []).append(xi)
        for dp, bystep in pairs_by_dp.items():
            for did in fdists:
                ok = []
                for s in L.main_steps:
                    rows = [summaries[r]["dists"][did][xi] for xi in bystep.get(s, [])]
                    ok.append(bool(rows) and all(row[FI["cost"]] is not None and row[FI["n05"]] is not None
                                                 and row[FI["cost"]] < row[FI["n05"]] for row in rows))
                i = first_stable(ok)
                F[f"al.c4.{dp}.{rn}.{did}.below_null_from"] = f"{L.main_steps[i]:,}" if i is not None else "not at the final checkpoint"
                F[f"al.c4.{dp}.{rn}.{did}.pairings"] = " · ".join(f"{L.nets[L.cross[xi]['a']]['id']}–{L.nets[L.cross[xi]['b']]['id']}" for xi in bystep[L.final])
    return F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="measurement directory (spectral_align/data or data_fixture)")
    ap.add_argument("--out", default=None, help="default: spectral_align/analysis/<data dir name>")
    ap.add_argument("--perms", type=int, default=1000, help="block-permutation null draws (SPEC: 1,000)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    data = Path(a.data).resolve()
    out = Path(a.out) if a.out else HERE / "analysis" / data.name
    return run(data, out, a.perms, a.workers)


if __name__ == "__main__":
    main()
