"""Build the Mamba coherent-sets page: one self-contained HTML file (SPEC section 9; Pythia SPEC 11 inherited).

    export TMPDIR=/var/tmp/cm-mb
    uv run --with numpy --with matplotlib --with pillow python -I page/build_page.py _fixture    # MOCKUP page
    uv run --with numpy --with matplotlib --with pillow python -I page/build_page.py m130        # real data
    ... page/build_page.py <tag> --facts        # print every {{fact:NAME}} value and exit

COPY of explore/lm_coherent/page/build_page.py (the Pythia page builder, never edited), adapted. What changes:
  - schema "mamba-toy-1": checkpoints are labels (init, trained) instead of training steps; no training blocks;
  - three views (point, state, stack) at both checkpoints; reading blocks for each view, depth blocks (point);
  - the explorer gets per-view layouts (umap grid_state, grid_stack) and a two-panel init/trained strip;
  - comparisons point_state / point_stack / state_stack (F6'), init vs trained (Q4), cross_arch.json (F12);
  - the verdict is SPEC section 7 (C1a, C1b, C2, P1-P3, H1-H7), computed here from the data;
  - F0 draws 24 Mamba blocks, each with its recurrent state (conv buffer + SSM state).

Reads explore/mamba_coherent/data/<tag>/ and page/{content,js,css}/; copies the ResNet site's palette, fonts and KaTeX
from explore/cifar_coherent/site/ (read only). Static figures come from make_figures.py (run automatically when they
are missing or older than the data; --figures forces it). cross_arch.json is optional: without it F12 and H7 say
"not yet computed".

Output: page/mamba_coherent.html, or page/mamba_coherent_MOCKUP.html for a synthetic fixture. Everything is inlined:
data are two gzip+base64 blobs (a JSON blob and a binary pack of little-endian typed arrays, layout in the JSON
blob's "index"), decoded in the browser with DecompressionStream, so the file works from file:// with no fetches.
The build prints the size and fails above 10 MB (target: under 8 MB).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # explore/mamba_coherent/page
ROOT = HERE.parent                                # explore/mamba_coherent
RESNET = ROOT.parent / "cifar_coherent" / "site"  # copied from, never edited
MAX_MB, TARGET_MB = 10.0, 8.0
JS_ORDER = ["colors.js", "charts.js", "data.js", "figs.js", "explorer.js", "zoom.js", "main.js"]
SHOW_R = [2, 4]
VCODE = {"point": "p", "state": "s", "stack": "k"}     # view -> one-letter key used in the browser ("b": bag)
T1_LAYER = 16                                          # SPEC 6: T1 is the point reading block at L16


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------- shared helpers
def layout_u8(xy: np.ndarray) -> np.ndarray:
    """(..., N, 2) layouts -> uint8 in a per-panel square box (0.5-99.5 percentile, aspect kept). Display only.
    make_figures.py uses the same function so the hero matches the explorer."""
    xy = np.asarray(xy, dtype=np.float64)
    lo = np.percentile(xy, 0.5, axis=-2, keepdims=True)
    hi = np.percentile(xy, 99.5, axis=-2, keepdims=True)
    span = (hi - lo).max(-1, keepdims=True)
    mid = (hi + lo) / 2
    q = np.clip((xy - mid) / (span + 1e-12) * 0.96 + 0.5, 0, 1)
    return np.round(q * 255).astype(np.uint8)


def excess(s, s0):
    s, s0 = np.asarray(s, dtype=np.float64), np.asarray(s0, dtype=np.float64)
    return (s - s0) / (1 - s0)


def int8_floor(vecs: np.ndarray, qmax: float) -> np.ndarray:
    """int16 eigenvectors -> int8 with a non-zero floor (an entry that is not zero never rounds to 0, so nodal signs
    stay exact), as in the ResNet build."""
    q8 = np.round(vecs.astype(np.float64) * 127.0 / qmax)
    q8 = np.where((q8 == 0) & (vecs != 0), np.sign(vecs), q8)
    return np.clip(q8, -127, 127).astype(np.int8)


def nibbles(lab2: np.ndarray, lab4: np.ndarray) -> np.ndarray:
    """two label arrays in -1..14 -> one uint8 per item: low nibble r=2 label + 1, high nibble r=4 label + 1."""
    return ((lab2.astype(np.int16) + 1) | ((lab4.astype(np.int16) + 1) << 4)).astype(np.uint8)


class Pack:
    """Concatenated little-endian arrays, each 8-byte aligned; index name -> [offset, dtype, shape]."""
    CODES = {"f4": "<f4", "i1": "i1", "u1": "u1", "i2": "<i2", "u2": "<u2", "i4": "<i4"}

    def __init__(self):
        self.buf = bytearray()
        self.index = {}

    def add(self, name, arr, code):
        assert name not in self.index, name
        a = np.ascontiguousarray(np.asarray(arr), dtype=self.CODES[code])
        while len(self.buf) % 8:
            self.buf.append(0)
        self.index[name] = [len(self.buf), code, list(a.shape)]
        self.buf += a.tobytes()


def b64gz(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode("ascii")


def clean(o):
    """JSON-safe: numpy -> python, non-finite -> None."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def g3(v, d=3):
    """3 significant figures as a plain string; 'n/a' for missing."""
    if v is None or not np.isfinite(v):
        return "n/a"
    if v == 0:
        return "0"
    return f"{v:.{d}g}" if abs(v) >= 1e-3 else f"{v:.2e}"


def f2(v, d=2):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ----------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, tag: str):
        self.tag = tag
        self.src = ROOT / "data" / tag
        if not (self.src / "meta.json").exists():
            sys.exit(f"no {self.src}/meta.json")
        self.meta = m = json.loads((self.src / "meta.json").read_text())
        assert m.get("schema") == "mamba-toy-1", f"expected schema mamba-toy-1, got {m.get('schema')}"
        self.probe = json.loads((self.src / "probe.json").read_text())
        self.fixture = bool(m.get("fixture")) or bool(self.probe.get("fixture"))
        self.ckpts, self.layers, self.positions, self.sources = m["checkpoints"], m["layers"], m["positions"], m["sources"]
        self.views = m.get("views", ["point", "state", "stack"])
        assert self.views == ["point", "state", "stack"], self.views
        self.final = "trained"
        assert self.final in self.ckpts and "init" in self.ckpts, self.ckpts
        self.n = m["n"]
        self.seba_r = m["seba_r"]
        self.ri = [self.seba_r.index(r) for r in SHOW_R]
        self.cells = {c: np.load(self.src / f"cells_{c}.npz") for c in self.ckpts}
        self.single = {c: np.load(self.src / f"single_{c}.npz") for c in self.ckpts}
        self.blocks = np.load(self.src / f"blocks_{self.final}.npz")
        self.umap = np.load(self.src / "umap.npz")
        self.compare = json.loads((self.src / "compare.json").read_text())["rows"]
        self.trees = json.loads((self.src / "trees.json").read_text())
        self.validation = json.loads((self.src / "validation.json").read_text())
        ca = self.src / "cross_arch.json"
        self.cross = json.loads(ca.read_text()) if ca.exists() else None
        items = m.get("items") or list(range(self.n))
        by_id = {it["id"]: it for it in self.probe["items"]}
        self.items = [by_id[i] if i in by_id else self.probe["items"][i] for i in items]
        self.src_idx = np.array([self.sources.index(it["source"]) for it in self.items], np.uint8)
        self.follow = list(m["follow_items"])
        self.li = {l: i for i, l in enumerate(self.layers)}
        self.ti = {t: i for i, t in enumerate(self.positions)}
        self.vi = {v: i for i, v in enumerate(self.views)}

    def mtime(self):
        return max(p.stat().st_mtime for p in self.src.iterdir())

    def tree(self, tid):
        return next(t for t in self.trees if t["id"] == tid)

    def rows(self, kind, **kw):
        return [r for r in self.compare if r["kind"] == kind and all(r.get(k) == v for k, v in kw.items())]


# ----------------------------------------------------------------------------------------------- payload
def build_payload(d: Data):
    P = Pack()
    n, L, T = d.n, len(d.layers), len(d.positions)
    r2, r4 = d.ri
    # --- probe: sources and tokens (as indices into a string table)
    P.add("src", d.src_idx, "u1")
    tok_ids = np.array([it["tokens"] for it in d.items], np.int64)
    strs = {}
    for it in d.items:
        for t_, s_ in zip(it["tokens"], it["token_str"]):
            strs.setdefault(t_, s_)
    vocab = sorted(strs)
    pos_of = {t_: i for i, t_ in enumerate(vocab)}
    P.add("tok", np.vectorize(pos_of.get)(tok_ids), "u2" if len(vocab) < 65536 else "i4")
    tokstr = [strs[t_] for t_ in vocab]
    # --- layouts: grid [bag + layers, positions] (point), grid_state / grid_stack [layers, positions],
    #     strip [layers, checkpoints] (point, t = 128)
    P.add("lay_grid", layout_u8(d.umap["grid"]), "u1")
    P.add("lay_grid_s", layout_u8(d.umap["grid_state"]), "u1")
    P.add("lay_grid_k", layout_u8(d.umap["grid_stack"]), "u1")
    P.add("lay_strip", layout_u8(d.umap["strip"]), "u1")

    # --- single cells: every view at both checkpoints, and the bag
    recs = []          # (view code, checkpoint, layer index or -1, position index, npz, prefix, index tuple)
    for c in d.ckpts:
        for v in d.views:
            for li in range(L):
                for ti in range(T):
                    recs.append((VCODE[v], c, li, ti, d.single[c], "", (d.vi[v], li, ti)))
    for c in d.ckpts:
        for ti in range(T):
            recs.append(("b", c, -1, ti, d.single[c], "bag_", (ti,)))
    K = len(recs)
    psi = np.zeros((K, 2, n), np.int8)
    lab = np.zeros((K, n), np.uint8)
    vals = np.zeros((K, 9), np.float32)
    pn = np.zeros((K, 8), np.float32)
    retp = np.full((K, 2, 4, 6), np.nan, np.float32)    # axis: 0 along t, 1 along depth
    rets = np.full((K, 2, 6, 6), np.nan, np.float32)
    top = np.full((K, 6, 24), -1, np.int16)
    sj = np.full((K, 2, 5), np.nan, np.float32)
    keys = []
    for k, (view, c, li, ti, z, pre, ix) in enumerate(recs):
        keys.append(f"{view}:{c}:{li}:{ti}")
        g = lambda name: z[pre + name][ix]   # noqa: E731
        vecs = g("vecs")
        qmax = 32767.0 if vecs.dtype == np.int16 else 127.0
        psi[k] = int8_floor(vecs[:2], qmax)
        sl = g("seba_label")
        lab[k] = nibbles(sl[r2], sl[r4])
        vals[k] = g("vals")
        pn[k] = d.cells[c][pre + "pn"][ix]
        rp, rs = g("retention_psi"), g("retention_seba")
        retp[k, 0] = rp[:4]
        rets[k, 0, :2], rets[k, 0, 2:] = rs[r2, :2], rs[r4, :4]
        if view != "b":
            rpd, rsd = g("retention_psi_depth"), g("retention_seba_depth")
            retp[k, 1] = rpd[:4]
            rets[k, 1, :2], rets[k, 1, 2:] = rsd[r2, :2], rsd[r4, :4]
        tm = g("top_members")
        top[k, :2], top[k, 2:] = tm[r2, :2], tm[r4, :4]
        jac, nul, ceil = g("src_jacc"), g("src_jacc_null"), g("src_jacc_ceiling")
        for a in range(2):
            sj[k, a] = [jac[a], nul[a, 0], nul[a, 1], nul[a, 2], ceil[a]]
    for name, arr, code in [("c_psi", psi, "i1"), ("c_lab", lab, "u1"), ("c_vals", vals, "f4"), ("c_pn", pn, "f4"),
                            ("c_retp", retp, "f4"), ("c_rets", rets, "f4"), ("c_top", top, "i2"), ("c_sj", sj, "f4")]:
        P.add(name, arr, code)

    # --- blocks (trained): reading [view, layer], depth [position] (point view)
    brecs = [("reading", VCODE[v], li, f"reading_{v}_", (li,)) for v in d.views for li in range(L)]
    brecs += [("depth", "p", ti, "depth_point_", (ti,)) for ti in range(T)]
    KB = len(brecs)
    b_psi = np.zeros((KB, 2, n), np.int8)
    b_lab = np.zeros((KB, n), np.uint8)
    b_vals = np.zeros((KB, 9), np.float32)
    b_pn = np.zeros((KB, 8), np.float32)
    b_retp = np.full((KB, 4, 6), np.nan, np.float32)
    b_rets = np.full((KB, 6, 6), np.nan, np.float32)
    b_top = np.full((KB, 6, 24), -1, np.int16)
    b_sj = np.full((KB, 2, 5), np.nan, np.float32)
    b_stat = np.full((KB, 8), np.nan, np.float32)      # T, T0, Tlo, Thi, W, W0, Wlo, Whi
    bkeys = []
    z = d.blocks
    for k, (kind, v, idx, pre, ix) in enumerate(brecs):
        bkeys.append(f"reading:{v}:{idx}" if kind == "reading" else f"depth:{idx}")
        g = lambda name: z[pre + name][ix]   # noqa: E731
        vecs = g("vecs")
        qmax = 32767.0 if vecs.dtype == np.int16 else 127.0
        b_psi[k] = int8_floor(vecs[:2], qmax)
        sl = g("seba_label")
        b_lab[k] = nibbles(sl[r2], sl[r4])
        b_vals[k] = g("vals")
        b_pn[k] = g("pn")
        rp, rs = g("retention_psi"), g("retention_seba")
        X = rp.shape[-1]
        b_retp[k, :, :X] = rp[:4]
        b_rets[k, :2, :X], b_rets[k, 2:, :X] = rs[r2, :2], rs[r4, :4]
        tm = g("top_members")
        b_top[k, :2], b_top[k, 2:] = tm[r2, :2], tm[r4, :4]
        jac, nul, ceil = g("src_jacc"), g("src_jacc_null"), g("src_jacc_ceiling")
        for a in range(2):
            b_sj[k, a] = [jac[a], nul[a, 0], nul[a, 1], nul[a, 2], ceil[a]]
        Tb, Wb = g("T_band"), g("W_band")
        b_stat[k] = [g("T"), g("T0"), Tb[0], Tb[1], g("W"), g("W0"), Wb[0], Wb[1]]
    for name, arr, code in [("b_psi", b_psi, "i1"), ("b_lab", b_lab, "u1"), ("b_vals", b_vals, "f4"),
                            ("b_pn", b_pn, "f4"), ("b_retp", b_retp, "f4"), ("b_rets", b_rets, "f4"),
                            ("b_top", b_top, "i2"), ("b_sj", b_sj, "f4"), ("b_stat", b_stat, "f4")]:
        P.add(name, arr, code)

    # --- per-cell statistics: [checkpoint, view, layer, position, ...]; bag [checkpoint, position, ...]
    def stack(key, bag=False):
        return np.stack([d.cells[c][("bag_" if bag else "") + key] for c in d.ckpts])
    for key in ["T", "T0", "T_band", "W", "W0", "W_band", "X", "X_band", "min_dist", "rogue1", "rogue5", "row_err"]:
        P.add("s_" + key, stack(key), "f4")
        P.add("sb_" + key, stack(key, True), "f4")
    P.add("s_pn1", stack("pn")[..., 0], "f4")
    P.add("sb_pn1", stack("pn", True)[..., 0], "f4")
    P.add("s_minset", stack("min_set_r4"), "i2")
    P.add("sb_minset", stack("min_set_r4", True), "i2")
    P.add("s_iters", stack("sinkhorn_iters"), "i2")
    P.add("s_own", stack("own_src"), "u1")          # point view
    P.add("s_follow", stack("follow_mass"), "u1")   # point view
    P.add("s_norm", d.cells[d.final]["norm_mean"], "f4")

    # --- JSON part
    trees = []
    for tr in d.trees:
        t2 = dict(tr)
        if tr["id"] == "P1":
            leaf = np.asarray(tr["planted_leaf"])
            t2["nodes"] = [dict(nd, planted=np.bincount(leaf[nd["members"]], minlength=8).tolist())
                           for nd in tr["nodes"]]
        trees.append(t2)
    probe_info = {k: d.probe.get(k) for k in ("filters", "seeds", "rejections", "note")}
    J = {
        "tag": d.tag, "fixture": d.fixture, "n": n, "checkpoints": d.ckpts, "final": d.final, "views": d.views,
        "layers": d.layers, "positions": d.positions, "n_layers": d.meta.get("n_layers"),
        "sources": d.sources, "seba_r_shown": SHOW_R, "follow": d.follow, "k": d.meta.get("k"),
        "model": d.meta.get("model"), "n_perm": d.meta.get("n_perm"), "threshold": d.meta.get("threshold"),
        "threshold_rule": d.meta.get("threshold_rule"), "tree_meta": d.meta.get("trees"),
        "layout": d.meta.get("layout"), "t1_layer": T1_LAYER,
        "cell_keys": keys, "block_keys": bkeys, "tokstr": tokstr, "items": [it.get("id") for it in d.items],
        "compare": d.compare, "trees": trees, "validation": d.validation, "probe": probe_info,
        "cross": d.cross, "palette": json.loads((RESNET / "palette.json").read_text()),
    }
    return P, clean(J)


# ----------------------------------------------------------------------------------------------- facts
def ex_view(d: Data, c, key, view):
    """excess share [layer, position] of one view at checkpoint c."""
    z = d.cells[c]
    v = d.vi[view]
    return excess(z[key][v], z[key + "0"][v])


def make_facts(d: Data, J) -> dict:
    f = {}
    L, T, S = d.layers, d.positions, d.sources
    li, ti = d.li, d.ti
    c = d.cells[d.final]
    f["model"] = (d.meta.get("model") or "?").split("/")[-1]
    f["model_full"] = d.meta.get("model", "?")
    f["revision"] = (d.meta.get("revision") or "n/a")[:12]
    f["n_items"] = f"{d.n:,}"
    f["n_per_source"] = f"{d.n // len(S):,}"
    f["n_sources"] = str(len(S))
    f["layers"] = ", ".join(f"L{x}" for x in L)
    f["positions"] = ", ".join(str(x) for x in T)
    f["k"] = str(d.meta.get("k"))
    f["n_perm"] = str(d.meta.get("n_perm"))
    f["n_cells_view"] = str(len(d.ckpts) * len(L) * len(T))
    f["n_cells_all"] = str(len(d.ckpts) * len(L) * len(T) * len(d.views) + len(d.ckpts) * len(T))
    f["tag"] = d.tag
    f["fingerprint"] = (d.meta.get("construction_fingerprint") or "")[:12]
    f["transformers"] = str(d.meta.get("transformers_version") or "n/a")
    f["torch"] = str(d.meta.get("torch_version") or "n/a")
    mc = d.meta.get("model_config") or {}
    f["n_layers"] = str(mc.get("num_hidden_layers", d.meta.get("n_layers", "?")))
    f["d_model"] = str(mc.get("hidden_size", "?"))
    f["d_inner"] = str(mc.get("intermediate_size", "?"))
    f["d_state"] = str(mc.get("state_size", "?"))
    f["conv_kernel"] = str(mc.get("conv_kernel", "?"))
    E, Sd, Kc = mc.get("intermediate_size"), mc.get("state_size"), mc.get("conv_kernel")
    if E and Sd and Kc:
        f["state.conv_numbers"] = f"{E * (Kc - 1):,}"
        f["state.ssm_numbers"] = f"{E * Sd:,}"
        f["state.layer_numbers"] = f"{E * (Kc - 1) + E * Sd:,}"
        f["state.whole_numbers"] = f"{(E * (Kc - 1) + E * Sd) * int(f['n_layers']):,}"
    else:
        for k in ("conv_numbers", "ssm_numbers", "layer_numbers", "whole_numbers"):
            f["state." + k] = "n/a"
    f["init_seed"] = str(d.meta.get("init_seed", "n/a"))
    # T1 block (point reading block at L16, trained), r = 4
    b = d.blocks
    l1 = li[T1_LAYER]
    w = b["reading_point_vals"][l1].astype(np.float64)
    lab4 = b["reading_point_seba_label"][l1, d.ri[1]]
    for k in (1, 2, 3, 4):
        f[f"t1.lambda{k}"] = f"{w[k]:.3f}"
    f["t1.gap"] = f"{w[3] - w[4]:.3f}"
    sizes = [int((lab4 == j).sum()) for j in range(4)]
    f["t1.set_sizes"] = ", ".join(str(x) for x in sizes)
    f["t1.n_unassigned"] = str(int((lab4 < 0).sum()))
    sjv, sjn, sjc = (b[f"reading_point_src_jacc{s}"][l1] for s in ("", "_null", "_ceiling"))
    f["t1.jacc_r4"], f["t1.jacc_r4_null"], f["t1.jacc_r4_ceiling"] = f"{sjv[1]:.2f}", f"{sjn[1, 2]:.2f}", f"{sjc[1]:.2f}"
    for j in range(4):
        mk = np.bincount(d.src_idx[lab4 == j], minlength=len(S))
        f[f"t1.set{j + 1}.makeup"] = ", ".join(f"{S[k]} {mk[k]}" for k in np.argsort(-mk) if mk[k] >= 10) or "none above 10"
    mku = np.bincount(d.src_idx[lab4 < 0], minlength=len(S))
    f["t1.unassigned.makeup"] = ", ".join(f"{S[k]} {mku[k]}" for k in np.argsort(-mku) if mku[k] >= 10) or "none above 10"

    # excess shares (trained unless named)
    Tp, Wp = ex_view(d, d.final, "T", "point"), ex_view(d, d.final, "W", "point")
    Ws, Wk = ex_view(d, d.final, "W", "state"), ex_view(d, d.final, "W", "stack")
    Ts, Tk = ex_view(d, d.final, "T", "state"), ex_view(d, d.final, "T", "stack")
    Wp0 = ex_view(d, "init", "W", "point")
    Wbag = excess(c["bag_W"], c["bag_W0"])

    # ------------------------------------------------------------- verdicts (SPEC 7)
    V = {}

    def verdict(key, ok, value, kind="hyp"):
        if kind == "hyp":
            V[key] = ("met" if ok else "not met") if ok is not None else "not determinable"
        else:
            V[key] = ("pass" if ok else "fail") if ok is not None else "not run"
        f[f"v.{key}.status"] = V[key]
        f[f"v.{key}.value"] = value
    val = d.validation
    c1a = val.get("C1a") or {}
    if c1a.get("pass") is None:
        verdict("C1a", None, str(c1a.get("status") or "not in the data"), "ctl")
    else:
        verdict("C1a", bool(c1a["pass"]), f"float64 max relative error {g3(c1a.get('max_rel_err'))} (tolerance "
                f"{sci(c1a.get('tolerance'))}); float32 {g3(c1a.get('float32_max_rel_err'))} (hazard)", "ctl")
    c1b = val.get("C1b") or {}
    runs = {k: v for k, v in c1b.items() if isinstance(v, dict) and v.get("pass") is not None}
    if not runs:
        st = {str(v.get("status")) for v in c1b.values() if isinstance(v, dict)}
        verdict("C1b", None, "; ".join(sorted(st)) or "not in the data", "ctl")
    else:
        verdict("C1b", all(v["pass"] for v in runs.values()),
                "; ".join(f"{k}: max relative error {g3(v.get('max_rel_err'))} over {v.get('steps', '?')} resumed steps"
                          for k, v in runs.items()) + f" (tolerance {sci(next(iter(runs.values())).get('tolerance'))})", "ctl")
    c2 = val.get("C2") or {}
    verdict("C2", all(x.get("identical") for x in c2.values()) if c2 else None,
            "; ".join(f"{k}: max |difference| {g3(x.get('max_abs_diff'))}" for k, x in c2.items()) or "n/a", "ctl")
    p1 = d.tree("P1")
    thr1 = p1.get("threshold")
    sl = [x["upsilon"] for x in p1.get("structureless", [])]
    pu = p1.get("planted_upsilon") or []
    p1ok = bool(p1.get("match", {}).get("exact")) and thr1 is not None and bool(pu) and min(pu) >= thr1 > max(sl or [0])
    verdict("P1", p1ok, f"{'exact' if p1.get('match', {}).get('exact') else 'not exact'}; planted Υ ≥ {g3(min(pu or [np.nan]))}, "
            f"structureless ≤ {g3(max(sl or [np.nan]))}; threshold {g3(thr1)}", "ctl")
    p2 = val.get("P2") or {}
    verdict("P2", bool(p2.get("pass")) if p2 else None,
            (f"sweep / SEBA Jaccard: transient cell {p2['transient_cell']['sweep_jaccard']:.2f} / "
             f"{p2['transient_cell']['seba2_jaccard']:.2f}, block {p2['block']['sweep_jaccard']:.2f} / "
             f"{p2['block']['seba2_jaccard']:.2f}") if p2.get("block") else "n/a", "ctl")
    p3 = val.get("P3") or {}
    gi = p3.get("gram_identity") or {}
    pc = p3.get("per_checkpoint") or {}
    devs = [x[k] for ck in pc.values() for blk in ck.values() if isinstance(blk, dict) for x in [blk] for k in x if k.startswith("dev_")]
    verdict("P3", bool(p3.get("pass")) if p3 else None,
            f"block means: largest deviation {max(devs):.1e}; Gram identity (state, stack at t = {gi.get('t', '?')}): "
            f"{float(gi.get('max_dev', np.nan)):.1e} (tolerance {sci(gi.get('tolerance'))})" if devs else "n/a", "ctl")
    # H1: T~ higher at L4 than at L24 at every t (trained, point)
    a, z_ = li[4], li[24]
    verdict("H1", bool(all(Tp[a, j] > Tp[z_, j] for j in range(len(T)))),
            "T̃ at L4 minus L24: " + ", ".join(f"{Tp[a, j] - Tp[z_, j]:+.3f}" for j in range(len(T))))
    # H2: W~(128) - W~(4) >= 0.2 at L12..L24, point
    dW = Wp[li[12]:, -1] - Wp[li[12]:, 0]
    verdict("H2", bool(np.all(dW >= 0.2)), "W̃(128) − W̃(4) at L12..L24: " + ", ".join(f"{x:.2f}" for x in dW))
    # H3: X(maths, poetry) < 0.5 at (L16, 16), point
    im, ip = S.index("maths"), S.index("poetry")
    x_mp = float(c["X"][0, li[16], ti[16], im, ip])
    verdict("H3", x_mp < 0.5, f"X(maths, poetry) at (L16, 16) = {x_mp:.3g}")
    # H4: whole-state cell (stack L24, t = 128) W~ >= max point W~ over layers - 0.05
    wk, wpm = float(Wk[li[24], -1]), float(np.nanmax(Wp[:, -1]))
    lmax = L[int(np.nanargmax(Wp[:, -1]))]
    verdict("H4", wk >= wpm - 0.05, f"W̃ at t = 128: whole state {wk:.2f}; largest point value {wpm:.2f} (L{lmax})")
    # H5: W~ at (L16, 128), point: init < half of trained
    w0, wf = float(Wp0[li[16], -1]), float(Wp[li[16], -1])
    verdict("H5", w0 < 0.5 * wf, f"W̃ at (L16, 128): init {w0:.2f}, trained {wf:.2f}")
    # H6: T1's first cut, >= 80 % of code and of poetry on opposite sides
    t1 = d.tree("T1")
    root = t1["nodes"][0]
    kids = [t1["nodes"][k] for k in root.get("children") or []]
    ic, ipo = S.index("code"), S.index("poetry")
    if len(kids) == 2 and "makeup" in kids[0]:
        fr = lambda nd, s: nd["makeup"][s] / max(1, kids[0]["makeup"][s] + kids[1]["makeup"][s])   # noqa: E731
        opp = max(min(fr(kids[0], ic), fr(kids[1], ipo)), min(fr(kids[1], ic), fr(kids[0], ipo)))
        verdict("H6", opp >= 0.8, f"code / poetry on child 0: {fr(kids[0], ic):.0%} / {fr(kids[0], ipo):.0%}")
    else:
        verdict("H6", None, "T1's root was not split")
    # H7: cross_arch point vs point at t = 128, a3 >= 0.3 at Pythia L6..L12 vs Mamba L12..L24
    if d.cross:
        rr = {r["pythia_layer"]: r for r in d.cross["rows"] if r["pair"] == "point_point" and r["t"] == 128}
        a3s = [(pl, rr[pl]["layer"], rr[pl]["a3"]) for pl in (6, 8, 10, 12) if pl in rr]
        verdict("H7", all(x[2] >= 0.3 for x in a3s) if len(a3s) == 4 else None,
                "a₃ at t = 128: " + ", ".join(f"P{p}–M{m} {a3:.2f}" for p, m, a3 in a3s))
    else:
        verdict("H7", None, "cross_arch.json not yet computed (cross_arch.py runs after the fetch)")
    hyp = [k for k in V if k.startswith("H")]
    ctl = [k for k in V if not k.startswith("H")]
    f["v.n_met"] = str(sum(V[k] == "met" for k in hyp))
    f["v.n_notmet"] = str(sum(V[k] == "not met" for k in hyp))
    f["v.n_nd"] = str(sum(V[k] == "not determinable" for k in hyp))
    f["v.n_hyp"] = str(len(hyp))
    f["v.n_pass"] = str(sum(V[k] == "pass" for k in ctl))
    f["v.n_ctl"] = str(len(ctl))
    f["v.n_notrun"] = str(sum(V[k] == "not run" for k in ctl))
    f["v.met_list"] = ", ".join(k for k in hyp if V[k] == "met") or "none"
    f["v.notmet_list"] = ", ".join(k for k in hyp if V[k] == "not met") or "none"
    f["v.nd_list"] = ", ".join(k for k in hyp if V[k] == "not determinable") or "none"
    f["v.ctl_fail_list"] = ", ".join(k for k in ctl if V[k] == "fail") or "none"
    f["v.ctl_notrun_list"] = ", ".join(k for k in ctl if V[k] == "not run") or "none"

    # ------------------------------------------------------------- trees, validation, diagnostics
    thr = d.meta.get("threshold")
    f["threshold"] = g3(thr) if thr is not None else "none"
    f["threshold_rule"] = str(d.meta.get("threshold_rule"))
    for tr in d.trees:
        tid = tr["id"]
        splits = [nd for nd in tr["nodes"] if nd.get("children")]
        f[f"{tid}.root.upsilon"] = g3(tr["nodes"][0].get("upsilon"))
        f[f"{tid}.n_splits"] = str(len(splits))
        f[f"{tid}.n_weak"] = str(sum(1 for nd in splits if (nd.get("upsilon") or 0) < thr)) if thr is not None else "n/a"
    f["p1.exact"] = "exact" if p1.get("match", {}).get("exact") else "not exact"
    f["p1.min_planted_upsilon"] = g3(min(pu or [np.nan]))
    f["p1.max_structureless_upsilon"] = g3(max(sl or [np.nan]))
    f["sinkhorn_failures"] = str(val.get("sinkhorn_failures"))
    f["t4.root_gram_dev"] = g3((val.get("T4_gram_route") or {}).get("root_gram_dev"))
    allc = [d.cells[k] for k in d.ckpts]
    f["min_dist.min"] = g3(min(np.nanmin(x["min_dist"]) for x in allc))
    f["pn1.min"] = g3(min(np.nanmin(x["pn"][..., 0]) for x in allc))
    f["row_err.max"] = g3(max(np.nanmax(x["row_err"]) for x in allc))
    f["minset.min"] = str(min(int(np.min(x["min_set_r4"])) for x in allc))
    f["build_date"] = dt.date.today().isoformat()
    real_facts(d, f, dict(Tp=Tp, Wp=Wp, Ws=Ws, Wk=Wk, Ts=Ts, Tk=Tk, Wp0=Wp0, Wbag=Wbag))
    return f


def rng(a, d=2):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    lo, hi = f"{a.min():.{d}f}", f"{a.max():.{d}f}"
    return lo if lo == hi else f"{lo} to {hi}"


def sci(v):
    """a tolerance such as 1e-12, printed as written in the SPEC."""
    return "n/a" if v is None else f"{float(v):.0e}".replace("e-0", "e-")


def rng3(a):
    """range at 3 significant figures (for quantities spanning orders of magnitude, such as the clustering ratio)."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    lo, hi = g3(a.min()), g3(a.max())
    return lo if lo == hi else f"{lo} to {hi}"


def pct(x):
    return f"{100 * x:.1f}".rstrip("0").rstrip(".") + " %"


def matched_jaccard(a, b, r=4):
    """Mean Jaccard of r sets after the best one-to-one matching (brute force over r! pairings; r = 4 here)."""
    import itertools
    J = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            A, B = a == i, b == j
            u = (A | B).sum()
            J[i, j] = (A & B).sum() / u if u else 0.0
    return max(np.mean([J[i, p[i]] for i in range(r)]) for p in itertools.permutations(range(r)))


def seba_soft(v, rot, perm):
    """SEBA vectors from the stored rotation (as make_figures.seba_vectors / the ResNet site's recomputation)."""
    z = v @ rot
    mu = 0.99 / np.sqrt(len(v))
    sv = np.sign(z) * np.maximum(np.abs(z) - mu, 0)
    sv /= np.linalg.norm(sv, axis=0)
    sg = np.sign(sv.sum(0))
    sg[sg == 0] = 1
    sv *= sg
    sv /= sv.max(0)
    return np.maximum(sv[:, perm], 0)


def real_facts(d: Data, f: dict, E: dict):
    """Facts cited by the Measurements prose (computed from the data; the same names exist for fixture builds)."""
    L, T, S = d.layers, d.positions, d.sources
    li, ti = d.li, d.ti
    c, c0 = d.cells[d.final], d.cells["init"]
    Tp, Wp, Ws, Wk, Wp0, Wbag = E["Tp"], E["Wp"], E["Ws"], E["Wk"], E["Wp0"], E["Wbag"]
    deep = [li[x] for x in (12, 16, 20, 24)]
    # Q1
    f["Tex.L4.range"], f["Tex.L24.range"] = rng(Tp[li[4]]), rng(Tp[li[24]])
    f["Tex.L4.init.range"] = rng(ex_view(d, "init", "T", "point")[li[4]])
    f["Wex.init.range"] = rng(Wp0)
    for t in (4, 32, 128):
        f[f"Wex.L16.t{t}"] = f"{Wp[li[16], ti[t]]:.2f}"
    dW = Wp[li[12]:, -1] - Wp[li[12]:, 0]
    f["H2.diffs"] = ", ".join(f"{x:.2f}" for x in dW)
    f["H2.min"] = f"{dW.min():.2f}"
    f["Wex.max.point"] = f"{np.nanmax(Wp):.2f}"
    f["Wex.bag.t4"], f["Wex.bag.t128"] = f"{Wbag[0]:.2f}", f"{Wbag[-1]:.2f}"
    peak = [T[int(np.nanargmax(Wp[i]))] for i in deep]
    f["Wex.peak_t.L12_24"] = ", ".join(str(p) for p in peak)
    # sets against sources
    b = d.blocks
    for v in d.views:
        sj = b[f"reading_{v}_src_jacc"][:, 1]
        f[f"rd.sj.{v}.range"] = rng(sj)
        f[f"rd.ratio.{v}.range"] = rng(sj / b[f"reading_{v}_src_jacc_ceiling"][:, 1])
    f["rd.null"] = f"{max(np.max(b[f'reading_{v}_src_jacc_null'][:, 1, 2]) for v in d.views):.2f}"
    f["rd.ceiling"] = rng(np.concatenate([b[f"reading_{v}_src_jacc_ceiling"][:, 1] for v in d.views]))
    sg = d.single[d.final]
    cp, sp, npnt = sg["src_jacc_ceiling"][0, :, :, 1], sg["src_jacc"][0, :, :, 1], sg["src_jacc_null"][0, :, :, 1, 2]
    f["cell.sj.point.range"] = rng(sp)
    f["cell.ratio.point.range"] = rng(sp / cp)
    f["cell.above_null.point"] = f"{int((sp > npnt).sum())} of {sp.size}"
    for v in ("state", "stack"):
        f[f"cell.sj.{v}.t128.range"] = rng(sg["src_jacc"][d.vi[v], :, -1, 1])
    f["cell.sj.stack.L24.t128"] = f"{sg['src_jacc'][d.vi['stack'], li[24], -1, 1]:.2f}"
    f["depth.sj.range"] = rng(b["depth_point_src_jacc"][:, 1])
    f["cell.sj.bag.t128"] = f"{sg['bag_src_jacc'][-1, 1]:.2f}"
    # Q2: maths vs poetry (point view, trained)
    im, ip, iw, i_f, ic = (S.index(x) for x in ("maths", "poetry", "wiki", "forum", "code"))
    X = c["X"][0]
    f["X.mp.L16.t16"] = f"{X[li[16], ti[16], im, ip]:.2f}"
    f["X.mp.L16.t16.band"] = f"{c['X_band'][0, li[16], ti[16], im, ip, 0]:.2f} to {c['X_band'][0, li[16], ti[16], im, ip, 1]:.2f}"
    f["X.wf.L16.t16"] = f"{X[li[16], ti[16], iw, i_f]:.2f}"
    f["X.cp.L16.t16"] = f"{X[li[16], ti[16], ic, ip]:.2f}"
    pairs = sorted((X[li[16], ti[16], a, b2], a, b2) for a in range(len(S)) for b2 in range(a + 1, len(S)))
    rk = 1 + [(a, b2) for _, a, b2 in pairs].index(tuple(sorted((im, ip))))
    f["X.mp.L16.t16.rank"] = f"{rk} of {len(pairs)}"
    f["X.mp.L4.range"] = rng(X[li[4], :, im, ip])
    f["X.mp.L12_24.t32_128"] = rng(X[li[12]:, ti[32]:, im, ip])
    f["X.mp.bag.t128"] = f"{c['bag_X'][-1, im, ip]:.3f}"
    f["X.mp.L16.init"] = f"{c0['X'][0, li[16], ti[16], im, ip]:.2f}"
    first_below = []
    for i in deep:
        below = [T[j] for j in range(len(T)) if X[i, j, im, ip] < 0.5]
        first_below.append(below[0] if below else None)
    f["X.mp.first_t_below_half.L12_24"] = ", ".join("never" if x is None else str(x) for x in first_below)
    fol = d.follow
    fa = next((k for k, i in enumerate(fol) if S[d.src_idx[i]] == "maths"), 0)
    fb = next((k for k, i in enumerate(fol) if S[d.src_idx[i]] == "poetry"), 1)
    ia, ib = fol[fa], fol[fb]
    same = 0
    for lj in range(len(L)):
        for tj in range(len(T)):
            lab = sg["seba_label"][0, lj, tj, d.ri[1]]
            same += int(lab[ia] >= 0 and lab[ia] == lab[ib])
    f["follow.same_cells"] = f"{same} of {len(L) * len(T)}"
    fm = c["follow_mass"]
    f["follow.maths.own.L16"] = rng(fm[fa, li[16], :, im] / 255)
    f["follow.poetry.own.L16"] = rng(fm[fb, li[16], :, ip] / 255)
    f["follow.maths.item"], f["follow.poetry.item"] = str(ia), str(ib)
    # Q3: residual vs layer state vs whole state (trained)
    names = {"point_state": "ps", "point_stack": "pk", "state_stack": "sk"}
    for kind, s_ in names.items():
        rows = d.rows(kind, checkpoint=d.final)
        f[f"{s_}.a3.range"] = rng([r["a3"] for r in rows])
        f[f"{s_}.a3.t128.range"] = rng([r["a3"] for r in rows if r["t"] == 128])
        f[f"{s_}.jacc.range"] = rng([r["jacc"] for r in rows])
        f[f"{s_}.null_hi.max"] = f"{max(r['jacc_null'][2] for r in rows):.2f}"
    pk = {(r["layer"], r["t"]): r for r in d.rows("point_stack", checkpoint=d.final)}
    f["pk.a3.L24.t128"] = f"{pk[(24, 128)]['a3']:.2f}" if (24, 128) in pk else "n/a"
    f["pk.jacc.L24.t128"] = f"{pk[(24, 128)]['jacc']:.2f}" if (24, 128) in pk else "n/a"
    f["pk.a3.L24.range"] = rng([pk[(24, t)]["a3"] for t in T if (24, t) in pk])
    f["a3.null"] = f"{3 / (d.n - 1):.3f}"
    for v, W_ in (("point", Wp), ("state", Ws), ("stack", Wk)):
        f[f"Wex.{v}.t128.range"] = rng(W_[:, -1])
    f["Wex.stack.L24.t128"] = f"{Wk[li[24], -1]:.2f}"
    f["Wex.point.L24.t128"] = f"{Wp[li[24], -1]:.2f}"
    f["Wex.point.t128.max"] = f"{np.nanmax(Wp[:, -1]):.2f}"
    parts = d.rows("state_parts", checkpoint=d.final)
    for p in ("c", "z"):
        f[f"parts.{p}.a3.range"] = rng([r["a3"] for r in parts if r["part"] == p])
    # Q4: init vs trained
    f["Wex.L16.t128.init"], f["Wex.L16.t128.trained"] = f"{Wp0[li[16], -1]:.2f}", f"{Wp[li[16], -1]:.2f}"
    for v in d.views:
        W0v = ex_view(d, "init", "W", v)
        f[f"Wex.{v}.init.range"] = rng(W0v)
        for t in (16, 128):
            r = next((r for r in d.rows("to_final", checkpoint="init", view=v) if r["layer"] == 16 and r["t"] == t), None)
            f[f"tf.{v}.t{t}"] = f"{r['jacc']:.2f} (a₃ {r['a3']:.2f}; null up to {r['jacc_null'][2]:.2f})" if r else "n/a"
    # Q5 order control
    rs = {(r["view"], r["layer"]): r for r in d.rows("real_vs_S")}
    for v in d.views:
        jj = [rs[(v, l)]["jacc"] for l in L if (v, l) in rs]
        f[f"S.jacc.{v}.range"] = rng(jj) if jj else "n/a"
        f[f"S.jacc.{v}.by_layer"] = ", ".join(f"L{l} {rs[(v, l)]['jacc']:.2f}" for l in L if (v, l) in rs)
    f["S.null.range"] = rng([r["jacc_null"][0] for r in rs.values()])
    # Q6 cross-architecture
    if d.cross:
        cr = d.cross["rows"]
        for pair, s_ in (("point_point", "pp"), ("slab_state", "pss"), ("slab_stack", "psk")):
            rr = [r for r in cr if r["pair"] == pair]
            f[f"ca.{s_}.a3.range"] = rng([r["a3"] for r in rr])
            f[f"ca.{s_}.a3.t128.range"] = rng([r["a3"] for r in rr if r["t"] == 128])
            f[f"ca.{s_}.jacc.range"] = rng([r["jacc"] for r in rr])
            f[f"ca.{s_}.null_hi.max"] = f"{max(r['jacc_null'][2] for r in rr):.2f}"
        ct = d.cross.get("t1_root_crosstab") or {}
        cnt = ct.get("counts") or [[0, 0], [0, 0]]
        f["ca.ct.counts"] = f"{cnt[0][0]}, {cnt[0][1]} / {cnt[1][0]}, {cnt[1][1]}"
        f["ca.ct.jacc"] = f"{ct.get('jacc', np.nan):.2f}"
        f["ca.ct.null_hi"] = f"{(ct.get('jacc_null') or [np.nan] * 3)[2]:.2f}"
        f["ca.pythia.dir"] = str((d.cross.get("pythia") or {}).get("dir"))
        f["ca.status"] = "from cross_arch.json"
    else:
        for s_ in ("pp", "pss", "psk"):
            for k in ("a3.range", "a3.t128.range", "jacc.range", "null_hi.max"):
                f[f"ca.{s_}.{k}"] = "n/a"
        f["ca.ct.counts"] = f["ca.ct.jacc"] = f["ca.ct.null_hi"] = f["ca.pythia.dir"] = "n/a"
        f["ca.status"] = "cross_arch.json not yet computed"
    # zoom
    trees = {t["id"]: t for t in d.trees}
    tids = [t for t in ("T1", "T2", "T3", "T4") if t in trees]
    for tid in tids:
        u = [nd["upsilon"] for nd in trees[tid]["nodes"] if nd.get("children")]
        f[f"{tid}.ups.range"] = rng3(u) if u else "no splits"
    allu = [nd["upsilon"] for tid in tids for nd in trees[tid]["nodes"] if nd.get("children")]
    f["ups.real.range"] = rng3(allu) if allu else "n/a"
    f["ups.real.n"] = str(len(allu))
    thr = d.meta.get("threshold")
    f["ups.real.n_clear"] = str(sum(u >= thr for u in allu)) if thr is not None else "n/a"
    p1 = trees["P1"]
    f["ups.structureless.range"] = rng3([x["upsilon"] for x in p1.get("structureless", [])])
    f["ups.planted.min"] = f"{min(p1.get('planted_upsilon') or [np.nan]):.2f}"
    ncomp = [nd.get("n_components") for tid in tids for nd in trees[tid]["nodes"] if nd.get("children")]
    f["zoom.max_components"] = str(max([x for x in ncomp if x is not None] or [0]))
    s2 = [nd["seba2_vs_cut"] for tid in tids for nd in trees[tid]["nodes"] if nd.get("children")]
    f["zoom.seba2.range"] = rng(s2) if s2 else "n/a"
    r1 = trees["T1"]["nodes"]
    if r1[0].get("children"):
        k0, k1 = (r1[k] for k in r1[0]["children"])
        tot = np.maximum(1, np.array(k0["makeup"]) + np.array(k1["makeup"]))
        f["T1.root.child0"] = ", ".join(f"{pct(k0['makeup'][k] / tot[k])} of {S[k]}" for k in np.argsort(-np.array(k0["makeup"]) / tot))
        f["T1.root.ret"] = rng(r1[0]["retention_cut"])
    else:
        f["T1.root.child0"] = f["T1.root.ret"] = "n/a"
    # hazards
    r1c = c["rogue1"][0]
    f["rogue1.point.max"] = f"{np.nanmax(r1c):.2f}"
    f["rogue1.point.range"] = rng(r1c)
    f["rogue1.init.max"] = f"{np.nanmax(c0['rogue1'][0]):.2f}"
    f["rogue5.point.max"] = f"{max(np.nanmax(d.cells[k]['rogue5'][0]) for k in d.ckpts):.3f}"
    nm = c["norm_mean"]
    f["norm.pos0.range"] = rng(nm[:, 0], 1)
    f["norm.rest.median.range"] = rng(np.median(nm[:, 1:], 1), 1)
    c1a = d.validation.get("C1a") or {}
    f32 = c1a.get("float32_by_layer") or {}
    f["c1a.f32.by_layer"] = ", ".join(f"{k} {float(x):.1e}" for k, x in f32.items()) or "n/a (not run on this data)"
    f["c1a.f64"] = g3(c1a.get("max_rel_err")) if c1a.get("max_rel_err") is not None else "n/a"
    pn1 = c["pn"][..., 0]
    f["pn1.trained.by_view"] = "; ".join(f"{v} {np.nanmin(pn1[d.vi[v]]):.0f}" for v in d.views)
    empty = 0
    for k in d.ckpts:
        z = d.cells[k]
        empty += int((z["min_set_r4"] == 0).sum()) + int((z["bag_min_set_r4"] == 0).sum())
    f["n_empty_sets"] = str(empty)
    zn = d.validation.get("zero_norm") or {}
    f["zero_norm"] = "none" if not any(zn.values()) else json.dumps(zn)
    f["removed_items"] = str(len(d.validation.get("removed_items") or []))
    readme = ROOT / "README.md"
    txt = readme.read_text() if readme.exists() else ""
    mo = re.search(r"Final ledger: \*\*\$([\d.]+)\*\*", txt)
    f["run.cost"] = f"USD {float(mo.group(1)):.2f}" if mo else "n/a (not yet run)"


# ----------------------------------------------------------------------------------------------- banner (F0)
def arch_svg(d: Data) -> str:
    """F0: the token row (EOT + 128 tokens, sampled positions marked) under the stack of Mamba blocks, recorded layers
    tagged; beside each block its recurrent state (conv buffer + SSM state); the column of state boxes is the whole
    state Sigma_t. Generated from meta.model_config, layers and positions."""
    mc = d.meta.get("model_config") or {}
    nl = int(mc.get("num_hidden_layers") or d.meta.get("n_layers") or max(d.layers))
    npos = max(d.positions)
    W, x0, y0, rh = 700, 50, 20, 4.4
    sx = 610                                  # left edge of the state column
    dx = (sx - 18 - x0) / (npos + 1)
    H = y0 + nl * rh + 34
    xs = lambda p: x0 + (p + 0.5) * dx          # noqa: E731
    ys = lambda l: y0 + (nl - l) * rh           # noqa: E731  (layer l = 1 at the bottom)
    out = [f'<svg class="arch-svg" viewBox="0 0 {W} {H:.0f}" role="img" aria-label="Architecture: {nl} Mamba blocks over '
           f'EOT plus {npos} snippet tokens, each block with its recurrent state; recorded layers and sampled positions marked">']
    for l in range(1, nl + 1):
        rec = l in d.layers
        out.append(f'<rect x="{x0:.1f}" y="{ys(l):.1f}" width="{(npos + 1) * dx:.1f}" height="{rh - 0.8:.1f}" rx="0.8" '
                   f'class="{"arch-rec" if rec else "arch-blk"}"/>')
        # state box: conv buffer (narrow) + SSM state (wide)
        out.append(f'<rect x="{sx:.1f}" y="{ys(l):.1f}" width="9" height="{rh - 0.8:.1f}" class="arch-conv"/>')
        out.append(f'<rect x="{sx + 11:.1f}" y="{ys(l):.1f}" width="36" height="{rh - 0.8:.1f}" class="arch-ssm"/>')
        out.append(f'<line x1="{(npos + 1) * dx + x0 + 2:.1f}" x2="{sx - 2:.1f}" y1="{ys(l) + rh / 2 - 0.4:.1f}" '
                   f'y2="{ys(l) + rh / 2 - 0.4:.1f}" class="arch-link"/>')
        if rec:
            out.append(f'<text x="{x0 - 6}" y="{ys(l) + rh - 0.6:.1f}" text-anchor="end" class="arch-t">L{l}</text>')
    out.append(f'<rect x="{sx - 3:.1f}" y="{ys(nl) - 2:.1f}" width="56" height="{nl * rh + 3:.1f}" rx="3" class="arch-sigma"/>')
    out.append(f'<text x="{sx + 25:.1f}" y="{ys(nl) - 6:.1f}" text-anchor="middle" class="arch-m">whole state Σ<tspan dy="2" font-size="8">t</tspan></text>')
    out.append(f'<text x="{sx + 4:.1f}" y="{y0 + nl * rh + 14:.1f}" text-anchor="middle" class="arch-m">c</text>')
    out.append(f'<text x="{sx + 29:.1f}" y="{y0 + nl * rh + 14:.1f}" text-anchor="middle" class="arch-m">z</text>')
    out.append(f'<text x="{x0 - 6}" y="{ys(nl) - 7:.1f}" text-anchor="end" class="arch-m">layer</text>')
    ty = y0 + nl * rh + 4
    for p in range(npos + 1):
        cls = "arch-eot" if p == 0 else ("arch-samp" if p in d.positions else "arch-tok")
        out.append(f'<rect x="{xs(p) - dx * 0.38:.2f}" y="{ty:.1f}" width="{dx * 0.76:.2f}" height="7" class="{cls}"/>')
    for p in d.positions:
        for l in d.layers:
            out.append(f'<circle cx="{xs(p):.1f}" cy="{ys(l) + (rh - 0.8) / 2:.1f}" r="2.1" class="arch-cell"/>')
        out.append(f'<text x="{xs(p):.1f}" y="{ty + 18:.1f}" text-anchor="middle" class="arch-t">{p}</text>')
    out.append(f'<text x="{xs(0):.1f}" y="{ty + 18:.1f}" text-anchor="middle" class="arch-m">EOT</text>')
    out.append(f'<text x="{x0 - 6}" y="{ty + 7}" text-anchor="end" class="arch-m">t</text>')
    out.append("</svg>")
    bits = [f"<b>{html.escape((d.meta.get('model') or '').split('/')[-1])}</b>"]
    if mc.get("hidden_size"):
        bits.append(f"{nl} Mamba blocks, width {mc['hidden_size']}, inner width {mc.get('intermediate_size')}, "
                    f"SSM state {mc.get('state_size')}, conv kernel {mc.get('conv_kernel')}")
    bits.append(f"{len(d.ckpts)} checkpoints: {', '.join(d.ckpts)}")
    txt = (f'<div class="arch-txt">{" · ".join(bits)}<br><span>A <b>cell</b> is (checkpoint, layer, position, view) '
           f'after <b>t tokens read</b>. Dots: the {len(d.layers) * len(d.positions)} residual cells per checkpoint. '
           f'Right: each block\'s recurrent state, conv buffer <b>c</b> and SSM state <b>z</b>; the column is the whole '
           f'state Σ<sub>t</sub>, everything the model carries forward.</span></div>')
    return f'<div class="arch" role="note">{"".join(out)}{txt}</div>'


# ----------------------------------------------------------------------------------------------- assembly
def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def inline_css() -> str:
    pal = json.loads((RESNET / "palette.json").read_text())
    tok = [":root {"]
    tok += [f"  --c-class-{i}: {c};" for i, c in enumerate(pal["classes"])]
    tok += [f"  --c-set-{i}: {c};" for i, c in enumerate(pal["sets"])]
    tok += [f"  --c-{k}: {pal['diverging'][k]};" for k in ("neg", "mid", "pos")]
    tok += [f"  --c-seq-{i}: {c};" for i, c in enumerate(pal["sequential"])]
    tok += [f"  --c-ord-{i}: {c};" for i, c in enumerate(pal["ordinal"])]
    tok += [f"  --{k}: {pal[k]};" for k in ("unassigned", "accent", "bg", "surface", "fg", "mute", "line")]
    tok.append("}")
    site = (HERE / "css" / "site.css").read_text()
    site = re.sub(r"url\(\.\./vendor/fonts/([^)]+)\)",
                  lambda m: f"url({data_uri(RESNET / 'vendor' / 'fonts' / m.group(1), 'font/woff2')})", site)
    kat = (RESNET / "vendor" / "katex" / "katex.min.css").read_text()
    kat = re.sub(r',url\(fonts/[^)]+\.(?:woff|ttf)\) format\("(?:woff|truetype)"\)', "", kat)
    kat = re.sub(r"url\(fonts/([^)]+\.woff2)\)",
                 lambda m: f"url({data_uri(RESNET / 'vendor' / 'katex' / 'fonts' / m.group(1), 'font/woff2')})", kat)
    assert "url(fonts/" not in kat and "../vendor" not in site
    page = (HERE / "css" / "page.css").read_text()
    return "\n".join([kat, "\n".join(tok), site, page])


def bundle_js() -> str:
    """Concatenate page/js modules into one inline module: single-line imports are dropped and `export` keywords
    stripped (the files share one scope, so top-level names must be unique; checked here)."""
    parts, names = [], {}
    decl = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function\*?|const|let|class)\s+([A-Za-z_$][\w$]*)", re.M)
    for fn in JS_ORDER:
        src = (HERE / "js" / fn).read_text()
        src = re.sub(r"^import\s+[^;]+;\s*$", "", src, flags=re.M)
        src = re.sub(r"^export\s+(?=(?:async\s+)?(?:function|const|let|class))", "", src, flags=re.M)
        assert not re.search(r"^\s*(import|export)\b", src, re.M), f"{fn}: unbundlable import/export left"
        for m in decl.finditer(src):
            if m.group(1) in names:
                sys.exit(f"bundle: top-level name '{m.group(1)}' in {fn} also in {names[m.group(1)]}")
            names[m.group(1)] = fn
        parts.append(f"// ---- {fn}\n{src}")
    return "\n".join(parts).replace("</script", "<\\/script")


def sub_facts(text: str, facts: dict, where: str) -> str:
    def sub(mo):
        key = mo.group(1).strip()
        if key not in facts:
            sys.exit(f"{where}: unknown fact '{key}'. Known facts:\n  " + "\n  ".join(sorted(facts)))
        return html.escape(str(facts[key]))
    return re.sub(r"\{\{\s*fact:([^}]+)\}\}", sub, text)


def sub_figs(text: str, figs: dict, where: str) -> str:
    def sub(mo):
        key = mo.group(1).strip()
        if key not in figs:
            sys.exit(f"{where}: unknown figure '{key}' (have {sorted(figs)})")
        return figs[key]
    return re.sub(r"\{\{\s*img:([^}]+)\}\}", sub, text)


def ensure_figures(d: Data, force: bool) -> Path:
    figdir = HERE / "build" / d.tag / "fig"
    need = ["hero.png", "step1.png", "step2.png", "step3.png", "step4.png", "step5.png", "cube.png"]
    stale = force or any(not (figdir / x).exists() or (figdir / x).stat().st_mtime < d.mtime() for x in need)
    if stale:
        sys.path.insert(0, str(HERE))
        import make_figures   # noqa: E402
        make_figures.main([d.tag])
    return figdir


def assemble(d: Data, facts: dict, P: Pack, J: dict, figdir: Path) -> str:
    J["index"] = P.index
    blob_json = b64gz(json.dumps(J, separators=(",", ":"), allow_nan=False).encode())
    blob_bin = b64gz(bytes(P.buf))
    figs = {p.stem: data_uri(p, "image/png") for p in sorted(figdir.glob("*.png"))}
    hdr = json.loads((HERE / "content" / "header.json").read_text())
    title = hdr["title"] + (" (MOCKUP: synthetic data)" if d.fixture else "")
    frag = {}
    for name in ["overview", "explorer", "zoom", "findings", "measurements", "method"]:
        p = HERE / "content" / f"{name}.html"
        t = p.read_text()
        drop = "real" if d.fixture else "mockup"      # <!-- mockup --> blocks only on MOCKUP pages, <!-- real --> only on real ones
        t = re.sub(rf"<!-- {drop} -->.*?<!-- /{drop} -->", "", t, flags=re.S)
        t = sub_figs(sub_facts(t, facts, p.name), figs, p.name)
        frag[name] = re.sub(r"<img(?![^>]*\bloading=)", '<img loading="lazy" decoding="async"', t)
    mock = ""
    if d.fixture:
        mock = ('<div class="mockup-note" role="note"><b>MOCKUP.</b> Built from the synthetic fixture '
                f'<code>{html.escape(d.tag)}</code>: planted structure, fake tokens and fake text. Every number and '
                'picture below shows that the page reads the data correctly; none says anything about Mamba or '
                'Pythia.</div>')
    tpl = (HERE / "content" / "page.html").read_text()
    rep = {
        "{{TITLE}}": html.escape(title), "{{SHORT_TITLE}}": html.escape(hdr["short_title"] + (" (MOCKUP)" if d.fixture else "")),
        "{{SUBTITLE}}": html.escape(hdr.get("subtitle", "")), "{{DATE}}": html.escape(hdr.get("date") or facts["build_date"]),
        "{{BANNER_TEXT}}": html.escape(hdr.get("banner", "")), "{{MOCKUP_NOTE}}": mock, "{{ARCH}}": arch_svg(d),
        "{{CSS}}": inline_css(), "{{KATEX_JS}}": (RESNET / "vendor" / "katex" / "katex.min.js").read_text(),
        "{{AUTORENDER_JS}}": (RESNET / "vendor" / "katex" / "auto-render.min.js").read_text(),
        "{{JS}}": bundle_js(), "{{BLOB_JSON}}": blob_json, "{{BLOB_BIN}}": blob_bin,
        "{{TAG}}": html.escape(d.tag), "{{BUILD_DATE}}": facts["build_date"],
        "{{FIXTURE_NOTE}}": " (SYNTHETIC FIXTURE: MOCKUP)" if d.fixture else "",
    }
    for k in ["overview", "explorer", "zoom", "findings", "measurements", "method"]:
        rep["{{CONTENT_" + k.upper() + "}}"] = frag[k]
    out = tpl
    big = {"{{CSS}}", "{{KATEX_JS}}", "{{AUTORENDER_JS}}", "{{JS}}", "{{BLOB_JSON}}", "{{BLOB_BIN}}"}
    # content first (it may contain $...$ but no template placeholders), then the small fields, then the big blobs
    for k in [k for k in rep if k.startswith("{{CONTENT_")] + [k for k in rep if not k.startswith("{{CONTENT_") and k not in big]:
        out = out.replace(k, rep[k])
    left = [x for x in re.findall(r"\{\{[A-Z_]+\}\}", out) if x not in big]
    assert not left, left
    for k in big:
        v = rep[k] if k not in ("{{KATEX_JS}}", "{{AUTORENDER_JS}}") else rep[k].replace("</script", "<\\/script")
        assert out.count(k) == 1, k
        out = out.replace(k, v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--facts", action="store_true", help="print the {{fact:NAME}} values and exit")
    ap.add_argument("--figures", action="store_true", help="re-run make_figures.py first")
    ap.add_argument("--out", default=None, help="output path (default: page/mamba_coherent[_MOCKUP].html)")
    a = ap.parse_args(argv)
    d = Data(a.tag)
    P, J = build_payload(d)
    facts = make_facts(d, J)
    if a.facts:
        for k, v in sorted(facts.items()):
            print(f"{k:32s} {v}")
        return
    figdir = ensure_figures(d, a.figures)
    page = assemble(d, facts, P, J, figdir)
    out = Path(a.out) if a.out else HERE / ("mamba_coherent_MOCKUP.html" if d.fixture else "mamba_coherent.html")
    if d.fixture and "MOCKUP" not in out.name:
        sys.exit("a page built from a fixture must carry MOCKUP in its file name")
    out.write_text(page)
    mb = out.stat().st_size / 1e6
    log(f"payload: binary {len(P.buf) / 1e6:.2f} MB raw, {len(P.index)} arrays; json {len(json.dumps(J)) / 1e6:.2f} MB raw")
    log(f"wrote {out} ({mb:.2f} MB; target < {TARGET_MB:.0f} MB, cap {MAX_MB:.0f} MB)")
    if mb > MAX_MB:
        sys.exit(f"page is {mb:.2f} MB, above the {MAX_MB:.0f} MB cap")
    if mb > TARGET_MB:
        log("warning: above the 8 MB target")


if __name__ == "__main__":
    main()
