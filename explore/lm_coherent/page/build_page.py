"""Build the LM toy page: one self-contained HTML file (SPEC section 11).

    export TMPDIR=/var/tmp/cm-lm
    uv run --with numpy --with matplotlib --with pillow python -I page/build_page.py _fixture    # MOCKUP page
    uv run --with numpy --with matplotlib --with pillow python -I page/build_page.py p160t       # real data
    ... page/build_page.py <tag> --facts        # print every {{fact:NAME}} value and exit

Reads explore/lm_coherent/data/<tag>/ (schema "lm-toy-1", SPEC 10.1) and page/{content,js,css}/; copies the ResNet
site's palette, fonts and KaTeX from explore/cifar_coherent/site/ (read only). Static figures come from
make_figures.py (run automatically when they are missing or older than the data; --figures forces it).

Output: page/lm_coherent.html, or page/lm_coherent_MOCKUP.html for a synthetic fixture. Everything is inlined:
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

HERE = Path(__file__).resolve().parent            # explore/lm_coherent/page
ROOT = HERE.parent                                # explore/lm_coherent
RESNET = ROOT.parent / "cifar_coherent" / "site"  # copied from, never edited
MAX_MB, TARGET_MB = 10.0, 8.0
JS_ORDER = ["colors.js", "charts.js", "data.js", "figs.js", "explorer.js", "zoom.js", "main.js"]
SHOW_R = [2, 4]
# Model facts for the banner. meta.json does not carry the model config yet (asked of the pipeline); these are the
# published values for the one model the spec names (SPEC section 2).
MODEL_CONFIG = {"EleutherAI/pythia-160m": {"n_layer": 12, "d_model": 768, "rotary_pct": 0.25, "parallel": True,
                                           "steps_total": 143000, "tokens_per_step": "about 2M"}}


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
        assert m.get("schema") == "lm-toy-1", f"expected schema lm-toy-1, got {m.get('schema')}"
        self.probe = json.loads((self.src / "probe.json").read_text())
        self.fixture = bool(m.get("fixture")) or bool(self.probe.get("fixture"))
        self.steps, self.layers, self.positions, self.sources = m["steps"], m["layers"], m["positions"], m["sources"]
        self.final = self.steps[-1]
        self.n = m["n"]
        self.seba_r = m["seba_r"]
        self.ri = [self.seba_r.index(r) for r in SHOW_R]
        self.cells = {s: np.load(self.src / f"cells_{s}.npz") for s in self.steps}
        self.single = {s: np.load(self.src / f"single_{s}.npz") for s in self.steps}
        self.blocks = np.load(self.src / f"blocks_{self.final}.npz")
        self.train = np.load(self.src / "train.npz")
        self.umap = np.load(self.src / "umap.npz")
        self.compare = json.loads((self.src / "compare.json").read_text())["rows"]
        self.trees = json.loads((self.src / "trees.json").read_text())
        self.validation = json.loads((self.src / "validation.json").read_text())
        items = m.get("items") or list(range(self.n))
        by_id = {it["id"]: it for it in self.probe["items"]}
        self.items = [by_id[i] if i in by_id else self.probe["items"][i] for i in items]
        self.src_idx = np.array([self.sources.index(it["source"]) for it in self.items], np.uint8)
        self.follow = list(m["follow_items"])

    def mtime(self):
        return max(p.stat().st_mtime for p in self.src.iterdir())

    # T1's block (reading block at L8, final step) is the page's running example
    def tree(self, tid):
        return next(t for t in self.trees if t["id"] == tid)


# ----------------------------------------------------------------------------------------------- payload
def build_payload(d: Data):
    P = Pack()
    n, L, T, S = d.n, len(d.layers), len(d.positions), len(d.steps)
    fi = d.steps.index(d.final)
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
    # --- layouts: grid [bag + layers, positions, N, 2] at the final step; strip [layers, steps, N, 2] at t = 128
    P.add("lay_grid", layout_u8(d.umap["grid"]), "u1")
    P.add("lay_strip", layout_u8(d.umap["strip"]), "u1")

    # --- single cells: point (all steps), slab (final step), bag (final step)
    recs = []          # (key, npz, prefix, index tuple, view, step, layer index or -1, position index)
    for si, s in enumerate(d.steps):
        for li in range(L):
            for ti in range(T):
                recs.append(("p", s, li, ti, d.single[s], "", (0, li, ti)))
    for li in range(L):
        for ti in range(T):
            recs.append(("s", d.final, li, ti, d.single[d.final], "", (1, li, ti)))
    for ti in range(T):
        recs.append(("b", d.final, -1, ti, d.single[d.final], "bag_", (ti,)))
    K = len(recs)
    psi = np.zeros((K, 2, n), np.int8)
    psiscale = np.zeros((K, 2), np.float32)
    lab = np.zeros((K, n), np.uint8)
    vals = np.zeros((K, 9), np.float32)
    pn = np.zeros((K, 8), np.float32)
    retp = np.full((K, 3, 4, 6), np.nan, np.float32)
    rets = np.full((K, 3, 6, 6), np.nan, np.float32)
    top = np.full((K, 6, 24), -1, np.int16)
    sj = np.full((K, 2, 5), np.nan, np.float32)
    keys = []
    for k, (view, s, li, ti, z, pre, ix) in enumerate(recs):
        keys.append(f"{view}:{s}:{li}:{ti}")
        g = lambda name: z[pre + name][ix]   # noqa: E731
        vecs, sc = g("vecs"), g("scale")
        qmax = 32767.0 if vecs.dtype == np.int16 else 127.0
        psi[k] = int8_floor(vecs[:2], qmax)
        psiscale[k] = sc[:2]
        sl = g("seba_label")
        lab[k] = nibbles(sl[r2], sl[r4])
        vals[k] = g("vals")
        pn[k] = d.cells[s][pre + "pn"][ix]
        rp, rs = g("retention_psi"), g("retention_seba")
        retp[k, 0] = rp[:4]
        rets[k, 0, :2], rets[k, 0, 2:] = rs[r2, :2], rs[r4, :4]
        if view != "b":
            rpd, rsd = g("retention_psi_depth"), g("retention_seba_depth")
            retp[k, 1] = rpd[:4]
            rets[k, 1, :2], rets[k, 1, 2:] = rsd[r2, :2], rsd[r4, :4]
        if view == "p" and ti == T - 1:
            si = d.steps.index(s)
            retp[k, 2, :, :S] = d.train["single_retention_psi"][li, si, :4]
            srs = d.train["single_retention_seba"][li, si]
            rets[k, 2, :2, :S], rets[k, 2, 2:, :S] = srs[r2, :2], srs[r4, :4]
        tm = g("top_members")
        top[k, :2], top[k, 2:] = tm[r2, :2], tm[r4, :4]
        jac, nul, ceil = g("src_jacc"), g("src_jacc_null"), g("src_jacc_ceiling")
        for a in range(2):
            sj[k, a] = [jac[a], nul[a, 0], nul[a, 1], nul[a, 2], ceil[a]]
    for name, arr, code in [("c_psi", psi, "i1"), ("c_psiscale", psiscale, "f4"), ("c_lab", lab, "u1"),
                            ("c_vals", vals, "f4"), ("c_pn", pn, "f4"), ("c_retp", retp, "f4"),
                            ("c_rets", rets, "f4"), ("c_top", top, "i2"), ("c_sj", sj, "f4")]:
        P.add(name, arr, code)

    # --- blocks: reading at the final step [layer], depth [position], training [layer, run]
    brecs = [("reading", li, d.blocks, "reading_", (li,)) for li in range(L)]
    brecs += [("depth", ti, d.blocks, "depth_", (ti,)) for ti in range(T)]
    brecs += [("train", (li, ru), d.train, "train_", (li, ru)) for li in range(L) for ru in range(2)]
    KB = len(brecs)
    b_psi = np.zeros((KB, 2, n), np.int8)
    b_psiscale = np.zeros((KB, 2), np.float32)
    b_lab = np.zeros((KB, n), np.uint8)
    b_vals = np.zeros((KB, 9), np.float32)
    b_pn = np.zeros((KB, 8), np.float32)
    b_retp = np.full((KB, 4, 6), np.nan, np.float32)
    b_rets = np.full((KB, 6, 6), np.nan, np.float32)
    b_top = np.full((KB, 6, 24), -1, np.int16)
    b_sj = np.full((KB, 2, 5), np.nan, np.float32)
    b_stat = np.full((KB, 8), np.nan, np.float32)      # T, T0, Tlo, Thi, W, W0, Wlo, Whi
    bkeys = []
    for k, (kind, idx, z, pre, ix) in enumerate(brecs):
        bkeys.append(f"{kind}:{idx[0]}:{idx[1]}" if isinstance(idx, tuple) else f"{kind}:{idx}")
        g = lambda name: z[pre + name][ix]   # noqa: E731
        vecs, sc = g("vecs"), g("scale")
        qmax = 32767.0 if vecs.dtype == np.int16 else 127.0
        b_psi[k] = int8_floor(vecs[:2], qmax)
        b_psiscale[k] = sc[:2]
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
    for name, arr, code in [("b_psi", b_psi, "i1"), ("b_psiscale", b_psiscale, "f4"), ("b_lab", b_lab, "u1"),
                            ("b_vals", b_vals, "f4"), ("b_pn", b_pn, "f4"), ("b_retp", b_retp, "f4"),
                            ("b_rets", b_rets, "f4"), ("b_top", b_top, "i2"), ("b_sj", b_sj, "f4"),
                            ("b_stat", b_stat, "f4")]:
        P.add(name, arr, code)

    # --- per-cell statistics, all steps: point/slab [step, view, layer, position, ...], bag [step, position, ...]
    def stack(key, bag=False):
        return np.stack([d.cells[s][("bag_" if bag else "") + key] for s in d.steps])
    for key in ["T", "T0", "T_band", "W", "W0", "W_band", "X", "X_band", "min_dist", "rogue1", "rogue5", "row_err"]:
        P.add("s_" + key, stack(key), "f4")
        P.add("sb_" + key, stack(key, True), "f4")
    P.add("s_pn1", stack("pn")[..., 0], "f4")
    P.add("sb_pn1", stack("pn", True)[..., 0], "f4")
    P.add("s_minset", stack("min_set_r4"), "i2")
    P.add("sb_minset", stack("min_set_r4", True), "i2")
    P.add("s_iters", stack("sinkhorn_iters"), "i2")
    P.add("s_own", stack("own_src"), "u1")
    P.add("s_follow", stack("follow_mass"), "u1")
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
        "tag": d.tag, "fixture": d.fixture, "n": n, "steps": d.steps, "layers": d.layers, "positions": d.positions,
        "sources": d.sources, "seba_r_shown": SHOW_R, "follow": d.follow, "k": d.meta.get("k"),
        "model": d.meta.get("model"), "n_perm": d.meta.get("n_perm"), "threshold": d.meta.get("threshold"),
        "threshold_rule": d.meta.get("threshold_rule"), "tree_meta": d.meta.get("trees"),
        "train_runs": d.meta.get("train_runs"), "layout": d.meta.get("layout"),
        "cell_keys": keys, "block_keys": bkeys, "tokstr": tokstr, "items": [it.get("id") for it in d.items],
        "compare": d.compare, "trees": trees, "validation": d.validation, "probe": probe_info,
        "palette": json.loads((RESNET / "palette.json").read_text()),
        "reading_by_step": reading_by_step(d),
    }
    return P, clean(J)


def reading_by_step(d: Data):
    """Reading blocks at every checkpoint (only the final ones are shipped whole): r = 4 matched Jaccard against
    sources [step][layer] with its null 97.5 % and ceiling, and the block's excess W."""
    out = {"sj": [], "null_hi": [], "ceiling": [], "W": []}
    for s in d.steps:
        z = np.load(d.src / f"blocks_{s}.npz")
        out["sj"].append(z["reading_src_jacc"][:, 1].tolist())
        out["null_hi"].append(z["reading_src_jacc_null"][:, 1, 2].tolist())
        out["ceiling"].append(z["reading_src_jacc_ceiling"][:, 1].tolist())
        out["W"].append(excess(z["reading_W"], z["reading_W0"]).tolist())
    return out


# ----------------------------------------------------------------------------------------------- facts
def make_facts(d: Data, J) -> dict:
    f = {}
    L, T = d.layers, d.positions
    fi = d.steps.index(d.final)
    li8 = L.index(8)
    c = d.cells[d.final]
    f["model"] = d.meta.get("model", "?")
    f["n_items"] = f"{d.n:,}"
    f["n_per_source"] = f"{d.n // len(d.sources):,}"
    f["n_sources"] = str(len(d.sources))
    f["n_steps"] = str(len(d.steps))
    f["steps"] = ", ".join(f"{s:,}" for s in d.steps)
    f["final_step"] = f"{d.final:,}"
    f["layers"] = ", ".join(f"L{x}" for x in L)
    f["positions"] = ", ".join(str(x) for x in T)
    f["k"] = str(d.meta.get("k"))
    f["n_perm"] = str(d.meta.get("n_perm"))
    f["n_cells"] = str(len(d.steps) * len(L) * len(T))
    f["tag"] = d.tag
    f["fingerprint"] = (d.meta.get("construction_fingerprint") or "")[:12]
    f["transformers"] = str(d.meta.get("transformers_version") or "n/a")
    # T1 block (reading, L8, final step), r = 4
    b = d.blocks
    w = b["reading_vals"][li8].astype(np.float64)
    lab4 = b["reading_seba_label"][li8, d.ri[1]]
    for k in (1, 2, 3, 4):
        f[f"t1.lambda{k}"] = f"{w[k]:.3f}"
    f["t1.gap"] = f"{w[3] - w[4]:.3f}"
    sizes = [int((lab4 == j).sum()) for j in range(4)]
    f["t1.set_sizes"] = ", ".join(str(x) for x in sizes)
    f["t1.n_unassigned"] = str(int((lab4 < 0).sum()))
    rs = b["reading_retention_seba"][li8, d.ri[1], :4]
    f["t1.ret_min"], f["t1.ret_max"] = f"{np.nanmin(rs):.3f}", f"{np.nanmax(rs):.3f}"
    sjv, sjn, sjc = b["reading_src_jacc"][li8], b["reading_src_jacc_null"][li8], b["reading_src_jacc_ceiling"][li8]
    f["t1.jacc_r4"], f["t1.jacc_r4_null"], f["t1.jacc_r4_ceiling"] = f"{sjv[1]:.2f}", f"{sjn[1, 2]:.2f}", f"{sjc[1]:.2f}"
    # excess shares
    Tt = excess(c["T"][0], c["T0"][0])        # [layer, position]
    Wt = excess(c["W"][0], c["W0"][0])
    Wt_bag = excess(c["bag_W"], c["bag_W0"])
    f["Wex.L8.t128"] = f2(Wt[li8, -1])
    f["Wex.bag.t128"] = f2(Wt_bag[-1])
    f["Tex.L2.max"] = f2(np.nanmax(Tt[0]))
    f["Tex.L12.max"] = f2(np.nanmax(Tt[-1]))
    # verdicts (SPEC 8)
    V = {}
    def verdict(key, ok, value):
        V[key] = ("met" if ok else "not met") if ok is not None else "not determinable"
        f[f"v.{key}.status"] = V[key]
        f[f"v.{key}.value"] = value
    e1 = [Tt[0, j] > Tt[-1, j] for j in range(len(T))]
    verdict("E1", all(e1), f"T̃ at L2 minus L12: " + ", ".join(f"{Tt[0, j] - Tt[-1, j]:+.3f}" for j in range(len(T))))
    li6 = L.index(6)
    dW = Wt[li6:, -1] - Wt[li6:, 0]
    verdict("H1", bool(np.all(dW >= 0.2)), "W̃(128) − W̃(4) at L6..L12: " + ", ".join(f"{x:.2f}" for x in dW))
    im, ip = d.sources.index("maths"), d.sources.index("poetry")
    x_mp = float(c["X"][0, li8, T.index(16), im, ip])
    verdict("H2", x_mp < 0.5, f"X(maths, poetry) at (L8, 16) = {x_mp:.3g}")
    ps = {(r["layer"], r["t"]): r for r in d.compare if r["kind"] == "point_slab" and r["step"] == d.final}
    a2, a12 = ps.get((2, 128), {}).get("a3"), ps.get((12, 128), {}).get("a3")
    verdict("H3", (a12 > a2) if a2 is not None and a12 is not None else None, f"a₃ at t = 128: L2 {g3(a2)}, L12 {g3(a12)}")
    c0 = d.cells[d.steps[0]]
    w0 = float(excess(c0["W"][0, li8, -1], c0["W0"][0, li8, -1]))
    wf = float(Wt[li8, -1])
    verdict("H4", w0 < 0.5 * wf, f"W̃ at (L8, 128): step 0 {w0:.2f}, step {d.final:,} {wf:.2f}")
    t1 = d.tree("T1")
    root = t1["nodes"][0]
    kids = [t1["nodes"][k] for k in root["children"]]
    ic, ipo = d.sources.index("code"), d.sources.index("poetry")
    if len(kids) == 2 and "makeup" in kids[0]:
        fr = lambda nd, s: nd["makeup"][s] / max(1, kids[0]["makeup"][s] + kids[1]["makeup"][s])   # noqa: E731
        opp = max(min(fr(kids[0], ic), fr(kids[1], ipo)), min(fr(kids[1], ic), fr(kids[0], ipo)))
        verdict("H5", opp >= 0.8, f"code / poetry on child 0: {fr(kids[0], ic):.0%} / {fr(kids[0], ipo):.0%}")
    else:
        verdict("H5", None, "n/a")
    f["v.n_met"] = str(sum(v == "met" for v in V.values()))
    f["v.n_total"] = str(len(V))
    # trees
    thr = d.meta.get("threshold")
    f["threshold"] = g3(thr) if thr is not None else "none"
    f["threshold_rule"] = str(d.meta.get("threshold_rule"))
    for tr in d.trees:
        tid = tr["id"]
        splits = [nd for nd in tr["nodes"] if nd.get("children")]
        f[f"{tid}.root.upsilon"] = g3(tr["nodes"][0].get("upsilon"))
        f[f"{tid}.n_splits"] = str(len(splits))
        if thr is not None:
            f[f"{tid}.n_weak"] = str(sum(1 for nd in splits if (nd.get("upsilon") or 0) < thr))
    p1 = d.tree("P1")
    f["p1.exact"] = "exact" if p1.get("match", {}).get("exact") else "not exact"
    sl = [x["upsilon"] for x in p1.get("structureless", [])]
    thr1 = p1.get("threshold")
    f["v.P1.status"] = "pass" if p1.get("match", {}).get("exact") and thr1 is not None and min(p1.get("planted_upsilon") or [0]) >= thr1 > max(sl or [0]) else "fail"
    f["p1.min_planted_upsilon"] = g3(min(p1.get("planted_upsilon") or [np.nan]))
    f["p1.max_structureless_upsilon"] = g3(max(s["upsilon"] for s in p1.get("structureless", [{"upsilon": np.nan}])))
    # validation and diagnostics
    v = d.validation
    f["sinkhorn_failures"] = str(v.get("sinkhorn_failures"))
    c2 = v.get("C2", {})
    f["c2.status"] = "identical at every checkpoint" if c2 and all(x.get("identical") for x in c2.values()) else "not identical"
    c1 = v.get("C1", {})
    if c1.get("status"):
        f["c1.status"] = c1["status"]
    elif c1:
        f["c1.status"] = (f"{c1.get('precision', '')} max relative error {g3(c1.get('max_rel_err'))} (gate {g3(c1.get('tolerance'))})"
                          + (f"; float32 up to {float(c1['float32_max_rel_err']):.1e}, a reported hazard" if c1.get("float32_max_rel_err") is not None else ""))
    else:
        f["c1.status"] = "n/a"
    f["v.C1.status"] = "not run" if (not c1 or "not run" in str(c1.get("status", ""))) else ("pass" if c1.get("pass", True) else "fail")
    f["v.C2.status"] = "pass" if f["c2.status"].startswith("identical") else "fail"
    allc = [d.cells[s] for s in d.steps]
    f["min_dist.min"] = g3(min(np.nanmin(x["min_dist"]) for x in allc))
    f["pn1.min"] = g3(min(np.nanmin(x["pn"][0, ..., 0]) for x in allc))
    f["rogue1.max"] = g3(max(np.nanmax(x["rogue1"][0]) for x in allc))
    f["row_err.max"] = g3(max(np.nanmax(x["row_err"]) for x in allc))
    f["minset.min"] = str(min(int(np.min(x["min_set_r4"][0])) for x in allc))
    nm = c["norm_mean"]
    f["norm.pos0.median"] = g3(float(np.median(nm[:, 0])))
    f["norm.rest.median"] = g3(float(np.median(nm[:, 1:])))
    f["build_date"] = dt.date.today().isoformat()
    real_facts(d, f, Tt, Wt, Wt_bag)
    return f


def rng(a, d=2):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    lo, hi = f"{a.min():.{d}f}", f"{a.max():.{d}f}"
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


def real_facts(d: Data, f: dict, Tt, Wt, Wt_bag):
    """Facts cited by the result prose (computed from the data; the same names exist for fixture builds)."""
    L, T, S = d.layers, d.positions, d.sources
    li = {l: i for i, l in enumerate(L)}
    ti = {t: i for i, t in enumerate(T)}
    c = d.cells[d.final]
    mid = [li[x] for x in (4, 6, 8, 10, 12)]
    # E1 / Q1
    f["Tex.L2.range"], f["Tex.L12.range"] = rng(Tt[0]), rng(Tt[-1])
    c0 = d.cells[d.steps[0]]
    f["Tex.L2.step0.range"] = rng(excess(c0["T"][0, 0], c0["T0"][0, 0]))
    f["Tex.L8.t4"], f["Tex.L8.t128"] = f"{Tt[li[8], 0]:.2f}", f"{Tt[li[8], -1]:.2f}"
    for t in (4, 32, 128):
        f[f"Wex.L8.t{t}"] = f"{Wt[li[8], ti[t]]:.2f}"
    f["Wex.t128.L4_12"] = rng(Wt[mid, -1])
    f["Wex.max.point"] = f"{np.nanmax(Wt):.2f}"
    f["Wex.bag.t4"], f["Wex.bag.t128"] = f"{Wt_bag[0]:.2f}", f"{Wt_bag[-1]:.2f}"
    dW = Wt[li[6]:, -1] - Wt[li[6]:, 0]
    f["H1.diffs"] = ", ".join(f"{x:.3f}" for x in dW)
    f["H1.max"] = f"{dW.max():.2f}"
    peak = [T[int(np.nanargmax(Wt[i]))] for i in mid]
    f["Wex.peak_t"] = ", ".join(str(p) for p in sorted(set(peak)))
    # slab
    Ws = excess(c["W"][1], c["W0"][1])
    f["Wex.slab.t128.L2_10"] = rng(Ws[:li[10] + 1, -1])
    f["Wex.slab.t128.L12"] = f"{Ws[-1, -1]:.2f}"
    f["Wex.point.t128.L2_10"] = rng(Wt[:li[10] + 1, -1])
    # Q2: maths vs poetry
    im, ip, iw, i_f, ic = (S.index(x) for x in ("maths", "poetry", "wiki", "forum", "code"))
    X = c["X"][0]
    f["X.mp.L8.t4"], f["X.mp.L8.t16"], f["X.mp.L8.t32"] = (f"{X[li[8], ti[t], im, ip]:.2f}" for t in (4, 16, 32))
    f["X.mp.L8.t16.band"] = f"{c['X_band'][0, li[8], ti[16], im, ip, 0]:.2f} to {c['X_band'][0, li[8], ti[16], im, ip, 1]:.2f}"
    f["X.wf.L8.t16"] = f"{X[li[8], ti[16], iw, i_f]:.2f}"
    f["X.cp.L8.t16"] = f"{X[li[8], ti[16], ic, ip]:.2f}"
    f["X.cf.L8.t16"] = f"{X[li[8], ti[16], ic, i_f]:.2f}"
    pairs = sorted((X[li[8], ti[16], a, b2], a, b2) for a in range(len(S)) for b2 in range(a + 1, len(S)))
    rk = 1 + [(a, b2) for _, a, b2 in pairs].index(tuple(sorted((im, ip))))
    f["X.mp.L8.t16.rank"] = ["most", "second most", "third most", "fourth most", "fifth most"][rk - 1] if rk <= 5 else f"number {rk} most"
    f["X.pairs.n"] = str(len(pairs))
    f["Wex.step0.range"] = rng(excess(c0["W"][0], c0["W0"][0]))
    f["X.mp.L2.range"] = rng(X[0, :, im, ip])
    f["X.mp.L6_12.t32_128"] = rng(X[li[6]:, ti[32]:, im, ip])
    f["X.mp.bag.t128"] = f"{c['bag_X'][-1, im, ip]:.3f}"
    f["X.wf.bag.t128"] = f"{c['bag_X'][-1, iw, i_f]:.2f}"
    f["X.mp.L8.step0"] = f"{c0['X'][0, li[8], ti[16], im, ip]:.2f}"
    first_below = []
    for i in mid:
        below = [T[j] for j in range(len(T)) if X[i, j, im, ip] < 0.5]
        first_below.append(below[0] if below else None)
    f["X.mp.first_t_below_half.L6_12"] = ", ".join("never" if x is None else str(x) for x in first_below[1:])
    # reading blocks vs single cells vs sources (validation)
    b = d.blocks
    sjr = b["reading_src_jacc"][:, 1]
    f["rd.sj.L2"], f["rd.sj.L4_12"] = f"{sjr[0]:.2f}", rng(sjr[1:])
    f["rd.null"] = f"{np.max(b['reading_src_jacc_null'][:, 1, 2]):.2f}"
    f["rd.ceiling"] = rng(b["reading_src_jacc_ceiling"][:, 1])
    f["rd.lambda1.range"] = rng(b["reading_vals"][:, 1], 3)
    sg = d.single[d.final]
    f["cell.sj.point.range"] = rng(sg["src_jacc"][0, :, :, 1])
    f["cell.sj.point.max"] = f"{np.nanmax(sg['src_jacc'][0, :, :, 1]):.2f}"
    f["cell.sj.slab.t64_128.L2_10"] = rng(sg["src_jacc"][1, :li[10] + 1, ti[64]:, 1])
    f["cell.sj.bag.r2.t128"] = f"{sg['bag_src_jacc'][-1, 0]:.2f}"
    f["depth.sj.range"] = rng(b["depth_src_jacc"][:, 1])
    rbs = reading_by_step(d)
    f["rd.sj.L8.by_step"] = ", ".join(f"{x[li[8]]:.2f}" for x in rbs["sj"])
    f["rd.sj.step0.max"] = f"{max(rbs['sj'][0]):.2f}"
    f["rd.sj.step512.range"] = rng(rbs["sj"][1])
    t1lab = b["reading_seba_label"][li[8], d.ri[1]]
    for j in range(4):
        mk = np.bincount(d.src_idx[t1lab == j], minlength=len(S))
        order = np.argsort(-mk)
        f[f"t1.set{j + 1}.makeup"] = ", ".join(f"{S[k]} {mk[k]}" for k in order if mk[k] >= 10)
    mku = np.bincount(d.src_idx[t1lab < 0], minlength=len(S))
    f["t1.unassigned.makeup"] = ", ".join(f"{S[k]} {mku[k]}" for k in np.argsort(-mku) if mku[k] >= 10) or "none"
    # Q3 point vs slab
    ps = [r for r in d.compare if r["kind"] == "point_slab"]
    fin = [r for r in ps if r["step"] == d.final]
    f["a3.t128.final"] = rng([r["a3"] for r in fin if r["t"] == 128])
    f["a3.final.max"] = f"{max(r['a3'] for r in fin):.2f}"
    f["a3.all.max"] = f"{max(r['a3'] for r in ps):.2f}"
    f["ps.jacc.final.range"] = rng([r["jacc"] for r in fin])
    # Q4 training
    for s in d.steps:
        cs = d.cells[s]
        f[f"Wex.L8.t128.step{s}"] = f"{excess(cs['W'][0, li[8], -1], cs['W0'][0, li[8], -1]):.2f}"
    tf = [r for r in d.compare if r["kind"] == "to_final" and r["layer"] == 8]
    f["tf.t16"] = ", ".join(f"{r['jacc']:.2f}" for r in sorted(tf, key=lambda r: r["step"]) if r["t"] == 16 and r["step"] != d.final)
    f["tf.t128.range"] = rng([r["jacc"] for r in tf if r["t"] == 128 and r["step"] != d.final])
    f["tf.t128.null_hi.range"] = rng([r["jacc_null"][2] for r in tf if r["t"] == 128 and r["step"] != d.final])
    # Q5 order control
    rs = {r["layer"]: r for r in d.compare if r["kind"] == "real_vs_S"}
    f["S.jacc.L2"], f["S.jacc.L4"] = f"{rs[2]['jacc']:.2f}", f"{rs[4]['jacc']:.2f}"
    f["S.jacc.L6_12"] = rng([rs[l]["jacc"] for l in (6, 8, 10, 12)])
    f["S.null.range"] = rng([rs[l]["jacc_null"][0] for l in L])
    f["S.W.L12"] = f"{rs[12]['W_a']:.3f} for the original and {rs[12]['W_b']:.3f} for the reordered probe"
    f["S.W.higher_layers"] = ", ".join(f"L{l}" for l in L if rs[l]["W_b"] > rs[l]["W_a"])
    f["S.W.lower_layers"] = ", ".join(f"L{l}" for l in L if rs[l]["W_b"] < rs[l]["W_a"])
    f["S.W.all"] = "; ".join(f"L{l} {rs[l]['W_a']:.3f} to {rs[l]['W_b']:.3f}" for l in L)
    f["S.W.min_b"] = f"{min(rs[l]['W_b'] for l in L):.2f}"
    # zoom
    trees = {t["id"]: t for t in d.trees}
    for tid in ("T1", "T2", "T3", "T4"):
        u = [nd["upsilon"] for nd in trees[tid]["nodes"] if nd.get("children")]
        f[f"{tid}.ups.range"] = rng(u)
    allu = [nd["upsilon"] for tid in ("T1", "T2", "T3", "T4") for nd in trees[tid]["nodes"] if nd.get("children")]
    f["ups.real.range"] = rng(allu)
    f["ups.real.n"] = str(len(allu))
    p1 = trees["P1"]
    sl = [x["upsilon"] for x in p1.get("structureless", [])]
    f["ups.structureless.range"] = rng(sl)
    f["ups.planted.min"] = f"{min(p1.get('planted_upsilon') or [np.nan]):.2f}"
    ncomp = [nd.get("n_components") for tid in ("T1", "T2", "T3", "T4") for nd in trees[tid]["nodes"] if nd.get("children")]
    f["zoom.max_components"] = str(max([x for x in ncomp if x is not None] or [0]))
    s2 = [nd["seba2_vs_cut"] for tid in ("T1", "T2", "T3", "T4") for nd in trees[tid]["nodes"] if nd.get("children")]
    f["zoom.seba2.range"] = rng(s2)
    r1 = trees["T1"]["nodes"]
    k0, k1 = (r1[k] for k in r1[0]["children"])
    tot = np.array(k0["makeup"]) + np.array(k1["makeup"])
    f["T1.root.child0"] = ", ".join(f"{pct(k0['makeup'][k] / tot[k])} of {S[k]}" for k in np.argsort(-np.array(k0["makeup"]) / tot))
    f["T1.root.seba2"] = f"{r1[0]['seba2_vs_cut']:.2f}"
    f["T1.root.ret"] = rng(r1[0]["retention_cut"])
    # hazards
    r1c = c["rogue1"][0]
    f["rogue1.final.L4_10.t4_32"] = rng(r1c[li[4]:li[10] + 1, :ti[32] + 1])
    f["rogue1.final.t64_128.max"] = f"{np.nanmax(r1c[:, ti[64]:]):.2f}"
    f["rogue1.final.L2.max"] = f"{np.nanmax(r1c[0]):.2f}"
    f["rogue1.final.L12.max"] = f"{np.nanmax(r1c[-1]):.2f}"
    f["rogue1.step16000.max"] = f"{np.nanmax(d.cells[16000]['rogue1'][0]):.2f}" if 16000 in d.cells else "n/a"
    early = [s for s in d.steps if s <= 2000]
    f["rogue1.early.max"] = f"{max(np.nanmax(d.cells[s]['rogue1'][0]) for s in early):.3f}"
    f["rogue5.max"] = f"{max(np.nanmax(d.cells[s]['rogue5'][0]) for s in d.steps):.3f}"
    nm = c["norm_mean"]
    f["norm.pos0.range"] = rng(nm[:, 0], 0)
    f["norm.rest.median.range"] = rng(np.median(nm[:, 1:], 1), 0)
    v = d.validation.get("C1", {})
    f32 = v.get("float32_by_layer") or {}
    f["c1.f32.by_layer"] = ", ".join(f"{k} {float(x):.1e}" for k, x in f32.items()) or "n/a"
    f["c1.f32.max"] = f"{float(v.get('float32_max_rel_err')):.1e}" if v.get("float32_max_rel_err") is not None else "n/a"
    f["c1.f64"] = g3(v.get("max_rel_err")) if v.get("max_rel_err") is not None else "n/a"
    p3 = (d.validation.get("P3") or {}).get("per_step") or {}
    devs = [max(x.get(k, 0) for k in x if k.startswith("dev_")) for x in p3.values()]
    f["p3.maxdev"] = f"{max(devs):.1e}" if devs else "n/a"
    p2 = d.validation.get("P2") or {}
    f["p2.summary"] = (f"sweep and SEBA Jaccard {p2['transient_cell']['sweep_jaccard']:.2f} / {p2['transient_cell']['seba2_jaccard']:.2f} "
                       f"(transient cell), {p2['block']['sweep_jaccard']:.2f} / {p2['block']['seba2_jaccard']:.2f} (block)") if p2.get("block") else "n/a"
    # size-aware comparison with sources: score / size ceiling, and how many comparisons beat their own null
    cp, sp, npnt = sg["src_jacc_ceiling"][0, :, :, 1], sg["src_jacc"][0, :, :, 1], sg["src_jacc_null"][0, :, :, 1, 2]
    f["cell.ceiling.point.range"] = rng(cp)
    f["cell.ratio.point.range"] = rng(sp / cp)
    f["rd.ratio.L4_12"] = rng(sjr[1:] / b["reading_src_jacc_ceiling"][1:, 1])
    f["cell.above_null.point"] = f"{int((sp > npnt).sum())} of {sp.size}"
    k = np.unravel_index(np.nanargmax(sp), sp.shape)
    f["cell.best.point"] = f"{sp[k]:.2f} (null {npnt[k]:.2f})"
    dn = b["depth_src_jacc"][:, 1] > b["depth_src_jacc_null"][:, 1, 2]
    tz = d.train
    tn = tz["train_src_jacc"][..., 1] > tz["train_src_jacc_null"][..., 1, 2]
    f["depth.above_null"] = f"{int(dn.sum())} of {dn.size}"
    f["train.above_null"] = f"{int(tn.sum())} of {tn.size}"
    f["train.sj.max"] = f"{np.nanmax(tz['train_src_jacc'][..., 1]):.2f}"
    # retention of T1's soft SEBA vectors: lazy walk, its non-lazy part, and the shuffled-vector baseline
    q = 32767.0 if b["reading_vecs"].dtype == np.int16 else 127.0
    psi = (b["reading_vecs"][li[8]].astype(np.float64) * b["reading_scale"][li[8]][:, None] / q).T
    v = np.concatenate([np.full((d.n, 1), 1 / np.sqrt(d.n)), psi[:, :3]], 1)
    sv = seba_soft(v, b["reading_seba_rot"][li[8], d.ri[1], :4, :4].astype(np.float64), b["reading_seba_perm"][li[8], d.ri[1], :4].astype(int))
    base = 0.5 + 0.5 * ((sv.sum(0) ** 2 / (sv ** 2).sum(0)) - 1) / (d.n - 1)
    rs4 = b["reading_retention_seba"][li[8], d.ri[1], :4]
    f["t1.ret_base"] = rng(base)
    f["t1.ret_nonlazy"] = rng(2 * rs4 - 1)
    t1l = b["reading_seba_label"][li[8], d.ri[1]]
    f["t1_vs_bag.t128"] = f"{matched_jaccard(t1l, sg['bag_seba_label'][-1, d.ri[1]]):.2f}"
    f["t1_vs_slab.L8.t128"] = f"{matched_jaccard(t1l, sg['seba_label'][1, li[8], -1, d.ri[1]]):.2f}"
    f["t1_vs_point.L8.t128"] = f"{matched_jaccard(t1l, sg['seba_label'][0, li[8], -1, d.ri[1]]):.2f}"
    # localised leading modes of single cells at early positions (final step, L2..L10)
    sgf = d.single[d.final]
    early_t = [ti[t] for t in (4, 8, 16)]
    pn1 = c["pn"][0, :li[10] + 1][:, early_t, 0]
    lam1 = sgf["vals"][0, :li[10] + 1][:, early_t, 1]
    f["loc.pn1.range"] = rng(pn1, 0)
    f["loc.lam1.min"] = f"{lam1.min():.3f}"
    ms = c["min_set_r4"][0, :li[10] + 1][:, early_t]
    f["loc.minset"] = ", ".join(rng(ms[:, j], 0) for j in range(3))
    nlfrac = []
    for j, tt in zip(early_t, (4, 8, 16)):
        lab = sgf["seba_label"][0, li[8], j, d.ri[1]]
        sizes = [(lab == k).sum() for k in range(4)]
        km = int(np.argmin(sizes))
        nl = np.array(["\n" in it["token_str"][tt - 1] for it in d.items])
        nlfrac.append(f"{100 * nl[lab == km].mean():.0f} %" if sizes[km] else "n/a")
    f["loc.newline_frac"] = ", ".join(nlfrac)
    f["pn1.t32_128.min"] = f"{np.nanmin(c['pn'][0, :, ti[32]:, 0]):.0f}"
    # the default follow pair (first maths and first poetry follow item)
    fol = d.follow
    fa = next(k for k, i in enumerate(fol) if S[d.src_idx[i]] == "maths")
    fb = next(k for k, i in enumerate(fol) if S[d.src_idx[i]] == "poetry")
    ia, ib = fol[fa], fol[fb]
    same = 0
    for lj in range(len(L)):
        for tj in range(len(T)):
            lab = sgf["seba_label"][0, lj, tj, d.ri[1]]
            same += int(lab[ia] >= 0 and lab[ia] == lab[ib])
    f["follow.same_cells"] = f"{same} of {len(L) * len(T)}"
    fm = c["follow_mass"]
    f["follow.maths.own.L8"] = rng(fm[fa, li[8], :, im] / 255)
    f["follow.poetry.own.L8"] = rng(fm[fb, li[8], :, ip] / 255)
    f["follow.maths.own.L12.t128"] = f"{fm[fa, li[12], -1, im] / 255:.2f}"
    f["follow.poetry.own.L12.t128"] = f"{fm[fb, li[12], -1, ip] / 255:.2f}"
    f["follow.maths.item"], f["follow.poetry.item"] = str(ia), str(ib)
    sm = c["min_set_r4"][0, :li[10] + 1, ti[4]]
    f["loc.minset.t4"] = rng(sm, 0)
    f["T1.root.m_min"] = str(trees["T1"]["nodes"][0].get("m_min"))
    # run cost and wall-clock, read from the README (not part of the data)
    readme = ROOT / "README.md"
    txt = readme.read_text() if readme.exists() else ""
    mo = re.search(r"Final ledger: \*\*\$([\d.]+)\*\*", txt)
    f["run.cost"] = f"USD {float(mo.group(1)):.2f}" if mo else "n/a"   # no dollar sign: KaTeX would read it as maths
    mo = re.search(r"([\d.]+) minutes of pilot and ([\d.]+) minutes of full run", txt)
    f["run.wall"] = f"{mo.group(1)} minutes of pilot and {mo.group(2)} minutes of full run" if mo else "n/a"
    empty = []
    for s in d.steps:
        z = d.cells[s]
        empty += [1 for _ in np.argwhere(z["min_set_r4"] == 0)] + [1 for _ in np.argwhere(z["bag_min_set_r4"] == 0)]
    f["n_empty_sets"] = str(len(empty))
    rej = d.probe.get("rejections") or {}
    def rcount(src, reason):
        x = rej.get(src, {})
        x = x.get("by_reason", x) if isinstance(x, dict) else {}
        return x.get(reason, "n/a")
    f["rej.code.dup4"], f["rej.forum.dup4"] = str(rcount("code", "dup_multiset_t4")), str(rcount("forum", "dup_multiset_t4"))


# ----------------------------------------------------------------------------------------------- banner (F0)
def arch_svg(d: Data) -> str:
    """F0: the token row (EOT + 128 tokens, sampled positions marked) under the 12-block stack, recorded layers
    tagged; their intersections are the cells. Generated from meta (layers, positions, model)."""
    mc = d.meta.get("model_config")
    if mc:   # read from the HF config by the pipeline (U1)
        cfg = {"n_layer": mc.get("num_hidden_layers", max(d.layers)), "d_model": mc.get("hidden_size"), "parallel": mc.get("use_parallel_residual")}
    else:
        cfg = MODEL_CONFIG.get(d.meta.get("model"), {"n_layer": max(d.layers), "d_model": None, "parallel": None})
    nl, npos = cfg["n_layer"], max(d.positions)
    W, x0, y0, rh = 640, 50, 12, 7.0
    dx = (W - x0 - 8) / (npos + 1)
    H = y0 + nl * rh + 34
    xs = lambda p: x0 + (p + 0.5) * dx          # noqa: E731
    ys = lambda l: y0 + (nl - l) * rh           # noqa: E731  (layer l = 1 at the bottom)
    out = [f'<svg class="arch-svg" viewBox="0 0 {W} {H:.0f}" role="img" aria-label="Architecture: {nl} blocks over '
           f'EOT plus {npos} snippet tokens; recorded layers and sampled positions marked">']
    for l in range(1, nl + 1):
        rec = l in d.layers
        out.append(f'<rect x="{x0:.1f}" y="{ys(l):.1f}" width="{(npos + 1) * dx:.1f}" height="{rh - 1:.1f}" rx="1" '
                   f'class="{"arch-rec" if rec else "arch-blk"}"/>')
        if rec:
            out.append(f'<text x="{x0 - 6}" y="{ys(l) + rh - 1.2:.1f}" text-anchor="end" class="arch-t">L{l}</text>')
    out.append(f'<text x="{x0 - 6}" y="{ys(nl) - 1:.1f}" text-anchor="end" class="arch-m">layer</text>')
    ty = y0 + nl * rh + 4
    for p in range(npos + 1):
        cls = "arch-eot" if p == 0 else ("arch-samp" if p in d.positions else "arch-tok")
        out.append(f'<rect x="{xs(p) - dx * 0.38:.2f}" y="{ty:.1f}" width="{dx * 0.76:.2f}" height="7" class="{cls}"/>')
    for p in d.positions:
        for l in d.layers:
            out.append(f'<circle cx="{xs(p):.1f}" cy="{ys(l) + (rh - 1) / 2:.1f}" r="2.1" class="arch-cell"/>')
        out.append(f'<text x="{xs(p):.1f}" y="{ty + 18:.1f}" text-anchor="middle" class="arch-t">{p}</text>')
    out.append(f'<text x="{xs(0):.1f}" y="{ty + 18:.1f}" text-anchor="middle" class="arch-m">EOT</text>')
    out.append(f'<text x="{x0 - 6}" y="{ty + 7}" text-anchor="end" class="arch-m">t</text>')
    out.append("</svg>")
    bits = [f"<b>{html.escape(d.meta.get('model', '').split('/')[-1])}</b>"]
    if cfg.get("d_model"):
        bits.append(f"{nl} transformer layers{' (attention and MLP in parallel)' if cfg.get('parallel') else ''}, width {cfg['d_model']}")
    bits.append(f"{len(d.steps)} checkpoints: steps {', '.join(f'{s:,}' for s in d.steps)}")
    txt = (f'<div class="arch-txt">{" · ".join(bits)}<br><span>A <b>cell</b> is (checkpoint, layer, position): the '
           f'residual stream after block l at position t, where <b>t = tokens read</b>. Dots: the '
           f'{len(d.layers) * len(d.positions)} cells per checkpoint.</span></div>')
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
                'picture below shows that the page reads the data correctly; none says anything about a language '
                'model.</div>')
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
    ap.add_argument("--out", default=None, help="output path (default: page/lm_coherent[_MOCKUP].html)")
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
    out = Path(a.out) if a.out else HERE / ("lm_coherent_MOCKUP.html" if d.fixture else "lm_coherent.html")
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
