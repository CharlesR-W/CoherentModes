"""Build the token-context coherent-sets page for mamba-130m (SPEC section 9): one self-contained HTML file.

    cd explore/mamba_contexts
    uv run -q --with numpy python -I page/build_page.py data/m130c           # real data -> page/mamba_contexts.html
    uv run -q --with numpy python -I page/build_page.py data/_fixture        # MOCKUP -> page/mamba_contexts_MOCKUP.html
    uv run -q --with numpy python -I page/build_page.py data/_fixture_flat   # MOCKUP -> page/mamba_contexts_flat_MOCKUP.html
    ... page/build_page.py <data dir> --facts        # print every {{fact:NAME}} value and exit
    options: --diag PATH (v1 diagnosis, default ../mamba_coherent/diagnosis/diag_v1.json)
             --bench PATH (full-size calibration check, default runs/bench.json; used when the data directory's
             calibration.json does not carry it)   --out PATH

Engine: `notebook.py` (a copy of the experiment-notebooks skill's engine, unedited). Tabs: Overview, Findings,
Explorer (F7), Measurements (F1-F6, F8), Method.

Prose lives in `page/prose/*.html` and is the only place to edit text that is not a figure caption:
  - `{{fact:NAME}}` is replaced by a number computed here from the data (`--facts` lists every name); an unknown name
    stops the build. Never type a number into prose.
  - `<!-- mockup --> ... <!-- /mockup -->` blocks appear only on fixture (MOCKUP) builds, `<!-- real --> ...
    <!-- /real -->` only on real builds. The real Findings go in the `real` block of `prose/findings.html`.
  - A file is split into sections at `<!-- section: NAME -->` markers; the builder places each section (overview has
    `answer`, `v2`, `found`, `map`; method has several). Unplaced sections stop the build.
Captions are definitional and are built here (function `captions`), with their numbers from the data.

Everything is embedded except Plotly and MathJax, which load from CDNs (the skill's engine). Big arrays are
base64-packed little-endian typed arrays decoded in the browser (helper `A`).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # explore/mamba_contexts/page
ROOT = HERE.parent                              # explore/mamba_contexts
sys.path.insert(0, str(HERE))
from notebook import Notebook, Radio, Select, add_figure, clean, render_figure, table_html  # noqa: E402

DIAG_DEFAULT = ROOT.parent / "mamba_coherent" / "diagnosis" / "diag_v1.json"
BENCH_DEFAULT = ROOT / "runs" / "bench.json"
GROUPS = ["T", "Nx", "U", "A", "W", "Cc", "Cn"]               # SPEC 5 groupings
SHARES = ["T", "Nx", "A", "kappa", "kappaT", "U", "W", "Cc", "Cn"]   # F2 quantities, in SPEC 9 order
LAG_G = ["T", "Cc", "U", "W", "Ttau"]
GLABEL = {"T": "token at t (T)", "Nx": "token at t+1 (Nx)", "U": "snippet (U)", "A": "model's top-1 prediction (A)",
          "W": "source (W)", "Cc": "class at t (C_c)", "Cn": "class at t+1 (C_n)", "kappa": "predictive coherence κ̃",
          "kappaT": "κ̃ beyond the token, κ̃|T", "Ttau": "token at t+τ (T_τ)"}
GSHORT = {"T": "T", "Nx": "Nx", "U": "U", "A": "A", "W": "W", "Cc": "C_c", "Cn": "C_n", "kappa": "κ̃",
          "kappaT": "κ̃|T", "Ttau": "T_τ"}
GKIND = {"T": "label-free (input)", "Nx": "label-free (training target)", "U": "label-free (input)",
         "A": "network output", "W": "validation", "Cc": "validation", "Cn": "validation",
         "kappa": "network output", "kappaT": "network output", "Ttau": "label-free (input)"}
# hue = what kind of thing groups the items; dash = variant (t vs t+1, raw vs conditional)
GSTYLE = {"T": ["#2a78d6", "solid"], "Nx": ["#2a78d6", "dash"], "Ttau": ["#2a78d6", "dot"],
          "A": ["#1baf7a", "dot"], "kappa": ["#008300", "solid"], "kappaT": ["#008300", "dash"],
          "U": ["#eb6834", "solid"], "W": ["#eb6834", "dash"],
          "Cc": ["#6250d6", "solid"], "Cn": ["#6250d6", "dash"]}
CLASS_PAL = ["#2a78d6", "#16a39a", "#eda100", "#d6336c", "#7a8b12", "#eb6834", "#008300", "#e87ba4", "#6250d6",
             "#33508f", "#8c6d46", "#9a9a9a"]
SOURCE_PAL = ["#d6336c", "#2a78d6", "#1baf7a", "#eda100", "#6250d6"]
SET_PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#33508f",
           "#7a8b12", "#16a39a", "#d6336c", "#8c6d46", "#5598e7", "#b07aa1", "#9c755f"]
OBJ_LABEL = {"Bpoint": "B-point", "Bearly": "B-point-early (L1–8)", "Bmid": "B-point-mid (L9–16)",
             "Blate": "B-point-late (L17–24)", "Bstate": "B-state", "whole": "whole state",
             "Blagpoint": "B-lag-point (L16 along reading)", "Blagwhole": "B-lag-whole (whole state along reading)",
             "bag": "bag (reference)", "recent": "recent (reference)", "pred": "pred (reference)",
             "Bk8": "B-point, 8 neighbours", "Bk30": "B-point, 30 neighbours"}
F4_OBJS = ["Bpoint", "Bearly", "Bmid", "Blate", "Bstate", "whole", "Blagpoint", "Blagwhole", "bag", "recent", "pred"]
F3_OBJS = ["Bpoint", "Bstate", "whole", "bag", "recent", "pred", "Bearly", "Bmid", "Blate", "Blagpoint", "Blagwhole",
           "Bk8", "Bk30"]


def olabel(o: str) -> str:
    if o in OBJ_LABEL:
        return OBJ_LABEL[o]
    m = re.match(r"(point|state)_L(\d+)(?:_lag(\d+))?$", o)
    if m:
        return f"{m.group(1)} L{m.group(2)}" + (f", τ = {m.group(3)}" if m.group(3) else "")
    m = re.match(r"whole_lag(\d+)$", o)
    return f"whole, τ = {m.group(1)}" if m else o


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------- formatting
def fin(v):
    if v is None or isinstance(v, (bool, str)):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def f2(v, d=2):
    return f"{float(v):.{d}f}" if fin(v) else "n/a"


def f3(v):
    return f2(v, 3)


def g3(v):
    if not fin(v):
        return "n/a"
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.3g}"


def sci(v):
    return f"{float(v):.1e}" if fin(v) else "n/a"


def rng(vals, fmt=f2):
    a = [float(x) for x in vals if fin(x)]
    if not a:
        return "n/a"
    lo, hi = fmt(min(a)), fmt(max(a))
    return lo if lo == hi else f"{lo} to {hi}"


def pct(x):
    return f"{100 * x:.0f}%" if fin(x) else "n/a"


def excess(s, s0):
    return (s - s0) / (1 - s0)


def tokdisp(s: str) -> str:
    """a token as text with its whitespace visible (facts and captions)."""
    s = s.replace("\n", "↵").replace("\t", "⇥")
    return "'" + s.replace(" ", "␣") + "'"


def enc(a, code):
    """little-endian typed array -> {b: base64, t: code, s: shape} (decoded in the browser by A())."""
    dtp = {"u1": "u1", "i1": "i1", "u2": "<u2", "i2": "<i2", "i4": "<i4", "f4": "<f4"}[code]
    a = np.ascontiguousarray(np.asarray(a), dtype=dtp)
    return {"b": base64.b64encode(a.tobytes()).decode("ascii"), "t": code, "s": list(a.shape)}


# ----------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, ddir: Path, diag: Path, bench: Path):
        self.dir = ddir
        j = lambda n: json.loads((ddir / n).read_text())   # noqa: E731
        if not (ddir / "meta.json").exists():
            sys.exit(f"no {ddir}/meta.json")
        self.meta = m = j("meta.json")
        assert m.get("schema") == "mamba-contexts-1", f"expected schema mamba-contexts-1, got {m.get('schema')}"
        self.fixture = m.get("fixture") or None                  # 'planted' | 'flat' | None (real)
        self.ck = m["checkpoints"]
        assert "trained" in self.ck, self.ck
        self.n = int(m["n"])
        self.ranks = m["ranks"]
        self.layers = m["layers"]
        self.lags = m["lags"]
        self.classes, self.sources = m["classes"], m["sources"]
        self.cells = {c: j(f"cells_{c}.json") for c in self.ck}
        self._spec = {c: np.load(ddir / f"spectra_{c}.npz") for c in self.ck}
        self._cache = {}
        self.sets = {c: j(f"sets_{c}.json") for c in self.ck}
        with np.load(ddir / "items.npz") as z:
            self.items = {k: z[k] for k in z.files}
        self.vocab = {int(k): v for k, v in j("vocab.json")["tokens"].items()}
        self.lagret = j("lags.json")
        self.compare = j("compare.json")["rows"]
        self.ups = j("upsilon.json")
        self.trees = {t["id"]: t for t in j("trees.json")}
        with np.load(ddir / "layouts.npz") as z:
            self.layouts = {k: z[k] for k in z.files}
        self.targets = j("targets.json")
        self.validation = j("validation.json")
        self.calib = j("calibration.json")
        self.diag = json.loads(diag.read_text())
        self.diag_path = diag
        cf = self.calib.get("calib_full") or self.calib.get("full_size")
        self.calib_full_src = "calibration.json"
        if cf is None and bench.exists():
            cf = json.loads(bench.read_text()).get("calib_full")
            self.calib_full_src = str(bench.relative_to(ROOT)) if bench.is_relative_to(ROOT) else str(bench)
        self.calib_full = cf
        self.objects = m["objects"]
        self.layout_objects = [o for o in m.get("layout_objects", []) if o in self.layouts]

    def sp(self, ck, o, key):
        k = (ck, o, key)
        if k not in self._cache:
            name = f"{o}__{key}"
            z = self._spec[ck]
            self._cache[k] = z[name] if name in z.files else None
        return self._cache[k]

    def tok(self, i):
        return self.vocab.get(int(i), f"<{int(i)}>")

    def cell(self, ck, name):
        return self.cells.get(ck, {}).get(name)

    def ex(self, ck, name, g):
        """excess share (or kappa excess) of a cell or block; None if absent."""
        c = self.cell(ck, name)
        if not c:
            return None
        if g in ("kappa", "kappaT"):
            k = c.get("kappa")
            return None if not k else k["kappa_ex" if g == "kappa" else "kappa_ex_T"]
        return c[g]["ex"] if g in c else None

    def ex_band(self, ck, name, g):
        c = self.cell(ck, name)
        if not c:
            return None
        if g == "kappaT":
            k = c.get("kappa")
            if not k:
                return None
            kt = k["kappaT"]
            return [excess(b, kt) for b in k["kappaT_band"]]
        if g == "kappa" or g not in c:
            return None
        s0 = c[g]["S0"]
        return [excess(b, s0) for b in c[g]["band"]]

    def ups_rows(self, ck, o):
        return self.ups["by_checkpoint"].get(ck, {}).get(o)


# ----------------------------------------------------------------------------------------------- token table
class Toks:
    """compact index for every token id the page shows."""

    def __init__(self, d: Data):
        self.d = d
        self.ids = {}
        self.strs = []

    def __call__(self, i):
        i = int(i)
        if i < 0:
            return 65535
        if i not in self.ids:
            self.ids[i] = len(self.strs)
            self.strs.append(self.d.tok(i))
        return self.ids[i]

    def arr(self, a):
        a = np.asarray(a)
        out = np.full(a.shape, 65535, np.uint16)
        flat = a.ravel()
        o = out.ravel()
        for k, v in enumerate(flat):
            o[k] = self(v)
        assert len(self.strs) < 65535, "token table too large for uint16"
        return out


# ----------------------------------------------------------------------------------------------- payloads
def payload_explorer(d: Data, tk: Toks):
    I = d.items
    n = d.n
    items = {
        "T": enc(tk.arr(I["T"]), "u2"), "Nx": enc(tk.arr(I["Nx"]), "u2"), "A": enc(tk.arr(I["A"]), "u2"),
        "Cc": enc(I["Cc"], "u1"), "Cn": enc(I["Cn"], "u1"), "W": enc(I["W"], "u1"),
        "U": enc(I["U"], "i4"), "t": enc(I["t"], "u1"),
        "ctx": enc(tk.arr(I["ctx"]), "u2"), "top5": enc(tk.arr(I["top5"]), "u2"),
        "top5p": enc(np.round(np.asarray(I["top5_p"]) * 1000), "u2"),
    }
    lay = {}
    for o in d.layout_objects:
        xy = d.layouts[o].astype(np.float64)
        lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
        span = float((hi - lo).max()) or 1.0
        mid = (hi + lo) / 2
        q = np.clip((xy - mid) / span * 0.96 + 0.5, 0, 1)
        rec = {"xy": enc(np.round(q * 65535), "u2")}
        lab = d.sp("trained", o, "seba_label")
        rec["lab"] = {str(r): enc(lab[d.ranks.index(r)], "i1") for r in (8, 16) if lab is not None}
        un = d.sp("trained", o, "unassigned")
        rec["unassigned"] = {str(r): int(un[d.ranks.index(r)]) for r in (8, 16)} if un is not None else {}
        v, s = d.sp("trained", o, "vecs"), d.sp("trained", o, "scale")
        if v is not None:
            psi = v[:4].astype(np.float64)
            mx = np.abs(psi).max(1, keepdims=True)
            mx[mx == 0] = 1
            rec["psi"] = enc(np.round(127 * psi / mx), "i1")
        ends = d.sp("trained", o, "ends")
        rec["ends"] = ends.tolist() if ends is not None else None
        vals = d.sp("trained", o, "vals")
        rec["lam"] = vals[1:17].tolist() if vals is not None else None
        eta = {}
        for g in GROUPS + ["Ttau"]:
            e = d.sp("trained", o, f"eta2_{g}")
            if e is not None:
                eta[g] = {"v": e.tolist(), "c": float(d.sp("trained", o, f"eta2_{g}_chance")),
                          "hi": d.sp("trained", o, f"eta2_{g}_band")[1].tolist()}
        rec["eta"] = eta
        sets = {}
        for r in ("8", "16"):
            rows = d.sets["trained"].get(o, {}).get(r, [])
            out = []
            for s_ in rows:
                if not s_.get("size"):
                    out.append({"set": s_["set"], "size": 0})
                    continue
                out.append({"set": s_["set"], "size": s_["size"], "snip": s_.get("snippets"),
                            "tt": [[tk(a), c] for a, c in s_.get("tok_t", [])],
                            "tn": [[tk(a), c] for a, c in s_.get("tok_next", [])],
                            "share": s_.get("top_token_share"), "n90": s_.get("tokens_for_90"),
                            "ct": s_.get("class_t"), "cn": s_.get("class_next"), "src": s_.get("source"),
                            "ex": s_.get("examples", [])})
            sets[r] = out
        rec["sets"] = sets
        lay[o] = rec
    return {"n": n, "objs": d.layout_objects, "labels": {o: olabel(o) for o in d.layout_objects},
            "items": items, "lay": lay, "classes": d.classes, "sources": d.sources,
            "before": d.meta.get("context", [24, 4])[0], "after": d.meta.get("context", [24, 4])[1]}


def payload_f1(d: Data):
    D = d.diag["d1"]
    ts = sorted(D["depth"], key=int)
    rs = sorted(D["depth"][ts[0]], key=int)
    gk = {"W": "source", "Cc": "cur_class", "Cn": "next_class", "T": "cur_id"}

    def summ(vals):
        a = [v for v in vals if v is not None]
        return [float(np.mean(a)), float(min(a)), float(max(a))] if a else [None] * 3

    depth = {g: [summ([D["depth"][t][r][k]["ami"] for t in ts if k in D["depth"][t][r]]) for r in rs]
             for g, k in gk.items()}
    R = D["reading"]
    blocks = list(R)
    reading = {"W": [summ([R[b][r]["source"]["ami"] for b in blocks]) for r in rs]}
    for g, k in (("Cc", "cur_class_by_t"), ("Cn", "next_class_by_t")):
        reading[g] = [summ([v["ami"] for b in blocks for v in R[b][r][k].values()]) for r in rs]
    reading["T"] = [summ([v for b in blocks for v in R[b][r]["cur_id_ami_by_t"].values()]) for r in rs]
    sets = []
    for t in sorted(D["depth_sets_r8"], key=int):
        for s_ in D["depth_sets_r8"][t]:
            if s_["set"] < 0 or not s_.get("size"):
                continue
            mk = s_["makeup"]
            sets.append({"t": int(t), "set": s_["set"] + 1, "size": s_["size"], "cls": s_["majority_cur_class"],
                         "cs": s_["majority_cur_share"], "src": d.sources[int(np.argmax(mk))] if mk else "?",
                         "ss": max(mk) / max(1, sum(mk)),
                         "top": " ".join(tokdisp(a) for a, _ in s_.get("top_cur", [])[:5])})
    return {"r": [int(r) for r in rs], "depth": depth, "reading": reading, "sets": sets, "classes": d.classes,
            "n_reading": len(blocks), "n_t": len(ts)}


def payload_f2(d: Data):
    out = {"layers": d.layers, "q": SHARES, "views": {}, "objs": {}}
    for view in ("point", "state"):
        out["views"][view] = {}
        for ck in d.ck:
            out["views"][view][ck] = {q: [d.ex(ck, f"{view}_L{l}", q) for l in d.layers] for q in SHARES}
    objs = ["Bpoint", "Bstate", "whole", "bag", "recent", "pred"]
    out["objn"] = [olabel(o) for o in objs]
    for ck in d.ck:
        out["objs"][ck] = {q: [d.ex(ck, o, q) for o in objs] for q in SHARES}
    hw = [abs(b[1] - b[0]) / 2 for view in ("point", "state") for l in d.layers for q in SHARES
          for b in [d.ex_band("trained", f"{view}_L{l}", q)] if b]
    out["band_hw"] = max(hw) if hw else None
    return out


def calib_summary(d: Data):
    """per k: the structureless maximum and the planted minimum (at the planted k, s in {1, 2}) on the check seeds."""
    rec = d.calib.get("records", [])
    check = set(d.calib.get("seeds", {}).get("check", []))
    sl, pl = {}, {}
    for r_ in rec:
        if r_.get("seed") not in check:
            continue
        if not r_.get("s"):
            for k, v in r_["k"].items():
                if fin(v.get("ups")):
                    sl[int(k)] = max(sl.get(int(k), 0), v["ups"])
        elif r_["s"] >= 1 and r_.get("k_planted") is not None:
            v = r_["k"].get(str(r_["k_planted"]), {})
            if fin(v.get("ups")):
                k = int(r_["k_planted"])
                pl[k] = min(pl.get(k, math.inf), v["ups"])
    gaps = [r_["gap"] for r_ in rec if r_.get("seed") in check and not r_.get("s")]
    return {"sl": {str(k): v for k, v in sorted(sl.items())}, "pl": {str(k): v for k, v in sorted(pl.items())},
            "gaps": gaps}


def payload_f3(d: Data):
    objs = {}
    for ck in d.ck:
        for o in F3_OBJS:
            rows = d.ups_rows(ck, o)
            mu = d.sp(ck, o, "mu")
            if rows is None or mu is None:
                continue
            key = o if ck == "trained" else f"{o}@{ck}"
            objs[key] = {"label": olabel(o) + ("" if ck == "trained" else f" ({ck})"), "mu": mu.tolist(),
                         "gap": d.sp(ck, o, "gap").tolist(),
                         "k": [r_["k"] for r_ in rows], "ups": [r_["ups"] for r_ in rows],
                         "seba": [r_["ups_seba"] for r_ in rows], "km": [r_["ups_kmeans"] for r_ in rows],
                         "cap": [r_["cap"] for r_ in rows], "out": [r_["outcome"] for r_ in rows]}
    theta = d.ups.get("theta")
    for tid in ("T1", "T4"):
        R = d.diag["d2_recompute"].get(tid)
        if not R:
            continue
        ks = sorted(int(k) for k in R["by_k"] if k.isdigit())
        mu = R["mu"]
        rows = [R["by_k"][str(k)] for k in ks]
        out = []
        for k, r_ in zip(ks, rows):
            if theta is None:
                out.append("n/a")
            elif r_["upsilon"] >= theta:
                out.append("certified")
            elif r_["ceiling"] < theta:
                out.append("excluded")
            else:
                out.append("undetermined")
        objs[f"v1{tid}"] = {"label": f"v1 root {tid} (diagnosis)", "mu": mu,
                            "gap": [mu[i] / mu[i - 1] for i in range(1, len(mu))], "k": ks,
                            "ups": [r_["upsilon"] for r_ in rows], "seba": [r_["upsilon"] for r_ in rows], "km": None,
                            "cap": [r_["ceiling"] for r_ in rows], "out": out, "v1": True,
                            "src5": R["by_k"].get("source_k5", {}).get("upsilon")}
    return {"objs": objs, "order": list(objs), "theta": theta, "calib": calib_summary(d), "R": d.ranks}


def payload_f4(d: Data):
    sets = {"blocks": F4_OBJS, "point": [f"point_L{l}" for l in d.meta["single_layers"]],
            "state": [f"state_L{l}" for l in d.meta["single_layers"]]}
    vals = {}
    for o in {x for v in sets.values() for x in v}:
        rec = {}
        for g in GROUPS:
            a = d.sp("trained", o, f"ami_{g}")
            e = d.sp("trained", o, f"eta2_{g}_weighted")
            rec[g] = {"a8": float(a[0]) if a is not None else None, "a16": float(a[1]) if a is not None else None,
                      "eta": float(e) if e is not None else None,
                      "c": float(d.sp("trained", o, f"eta2_{g}_weighted_chance")) if e is not None else None,
                      "hi": float(d.sp("trained", o, f"eta2_{g}_weighted_band")[1]) if e is not None else None}
        vals[o] = rec
    return {"sets": sets, "vals": vals, "labels": {o: olabel(o) for o in vals}, "groups": GROUPS}


def payload_f5(d: Data):
    L = d.lagret
    views = list(L["views"])
    ret = {v: {"mean": L["views"][v]["mean"], "lo": np.min(L["views"][v]["per_vector"], 0).tolist(),
               "hi": np.max(L["views"][v]["per_vector"], 0).tolist()} for v in views}
    sh = {}
    for v in views:
        sh[v] = {}
        for g in LAG_G:
            row = []
            for tau in L["lags"]:
                name = v if tau == 0 else f"{v}_lag{tau}"
                gg = "T" if (g == "Ttau" and tau == 0) else g
                row.append(d.ex("trained", name, gg))
            sh[v][g] = row
    return {"lags": L["lags"], "views": views, "labels": {v: olabel(v) for v in views}, "ret": ret, "sh": sh,
            "chance": L.get("chance"), "source": L.get("source")}


def payload_f6(d: Data, tk: Toks):
    out = {}
    for tid, tr in d.trees.items():
        nodes = []
        for nd in tr["nodes"]:
            nodes.append({"id": nd["id"], "parent": nd["parent"], "depth": nd["depth"], "n": nd["n"],
                          "children": nd.get("children") or [], "ups": nd.get("upsilon"),
                          "mu": (nd.get("mu") or [])[:8], "gap": (nd.get("gap") or [])[:8], "phi": nd.get("phi"),
                          "tt": [[tk(a), c] for a, c in nd.get("tok_t", [])],
                          "tn": [[tk(a), c] for a, c in nd.get("tok_next", [])],
                          "ct": nd.get("class_t"), "cn": nd.get("class_next"), "src": nd.get("source"),
                          "snip": nd.get("snippets"), "seba2": nd.get("seba2_vs_cut"),
                          "comp": nd.get("n_components")})
        out[tid] = {"block": tr.get("block"), "threshold": tr.get("threshold"), "nodes": nodes}
    return out


def payload_f8(d: Data):
    objs = ["Bpoint", "whole"] + [f"{v}_L{l}" for v in ("point", "state") for l in d.meta["single_layers"]]
    objs = [o for o in objs if all(d.cell(ck, o) for ck in d.ck) and all(d.sp(ck, o, "ami_T") is not None for ck in d.ck)]
    sh = {o: {ck: [d.ex(ck, o, q) for q in SHARES] for ck in d.ck} for o in objs}
    ami = {o: {ck: [(lambda a: float(a[0]) if a is not None else None)(d.sp(ck, o, f"ami_{g}")) for g in GROUPS]
               for ck in d.ck} for o in objs}
    return {"objs": objs, "labels": {o: olabel(o) for o in objs}, "sh": sh, "ami": ami, "ck": d.ck}


# ----------------------------------------------------------------------------------------------- facts
def make_facts(d: Data) -> dict:
    f = {}
    m = d.meta
    I = d.items
    f["model"] = (m.get("model") or "?").split("/")[-1]
    f["model_full"] = m.get("model") or "?"
    f["revision"] = (m.get("revision") or "n/a")[:12]
    f["fingerprint"] = (m.get("construction_fingerprint") or "")[:12]
    f["data_dir"] = d.dir.name
    f["fixture"] = d.fixture or "none (real data)"
    f["n_items"] = f"{d.n:,}"
    f["n_snippets"] = f"{len(np.unique(I['snippet'])):,}"
    f["per_snippet"] = f2(d.n / max(1, len(np.unique(I["snippet"]))), 0)
    f["t_min"], f["t_max"] = str(int(I["t"].min())), str(int(I["t"].max()))
    f["n_tokens_t"] = f"{len(np.unique(I['T'])):,}"
    f["n_classes"] = str(len(d.classes))
    f["n_sources"] = str(len(d.sources))
    f["sources"] = ", ".join(d.sources)
    f["classes"] = ", ".join(c.replace("_", " ") for c in d.classes)
    f["k"] = str(m.get("k"))
    f["k_sens"] = " and ".join(str(x) for x in m.get("k_sens", []))
    f["n_perm"] = str(m.get("n_perm"))
    f["lags"] = ", ".join(str(x) for x in d.lags)
    f["ranks"] = ", ".join(str(x) for x in d.ranks)
    f["n_layers"] = str((m.get("model_config") or {}).get("num_hidden_layers", len(d.layers)))
    f["single_layers"] = ", ".join(f"L{x}" for x in m["single_layers"])
    f["lag_layers"] = ", ".join(f"L{x}" for x in m["lag_layers"])
    f["context_before"], f["context_after"] = (str(x) for x in m.get("context", [24, 4]))
    f["dedup_redraws"] = str((m.get("dedup") or {}).get("redraws", "n/a"))
    f["theta"] = g3(d.ups.get("theta"))
    f["theta_rule"] = str(d.ups.get("rule"))
    f["build_date"] = dt.date.today().isoformat()
    f["n_eta"] = str(m.get("n_eta", 16))
    f["a_null"] = f3(8 / (d.n - 1))

    # every cell and block: excess shares, kappa
    for ck in d.ck:
        for name, c in d.cells[ck].items():
            if re.search(r"_k\d+$", name) or "T" not in c:
                continue
            for g in GROUPS + ["Ttau"]:
                if g in c:
                    f[f"share.{ck}.{name}.{g}"] = f2(c[g]["ex"])
                    f[f"share.{ck}.{name}.{g}.band"] = rng(d.ex_band(ck, name, g), f3)
            k = c.get("kappa")
            if k:
                f[f"share.{ck}.{name}.kappa"] = f2(k["kappa_ex"])
                f[f"share.{ck}.{name}.kappaT"] = f2(k["kappa_ex_T"])
                f[f"share.{ck}.{name}.kappaT.band"] = rng(d.ex_band(ck, name, "kappaT"), f3)
                f[f"kappa.{ck}.{name}"] = f3(k["kappa"])
                f[f"kappa.{ck}.{name}.kappaT"] = f3(k["kappaT"])
                f[f"kappa.{ck}.{name}.band"] = rng(k["kappaT_band"], f3)
                f[f"kappa.{ck}.{name}.unique_share"] = pct(k.get("unique_share"))
    # every spectral object: AMI, eta^2, PN, unassigned
    for ck in d.ck:
        for o in d.objects.get(ck, []):
            for g in GROUPS + ["Ttau"]:
                a = d.sp(ck, o, f"ami_{g}")
                if a is not None:
                    f[f"ami8.{ck}.{o}.{g}"], f[f"ami16.{ck}.{o}.{g}"] = f2(a[0]), f2(a[1])
                e = d.sp(ck, o, f"eta2_{g}_weighted")
                if e is not None:
                    f[f"eta.{ck}.{o}.{g}"] = f2(e)
                    f[f"eta.{ck}.{o}.{g}.chance"] = f3(d.sp(ck, o, f"eta2_{g}_weighted_chance"))
            pn = d.sp(ck, o, "pn")
            if pn is not None:
                f[f"pn1.{ck}.{o}"] = f2(pn[0], 0)
            un = d.sp(ck, o, "unassigned")
            if un is not None:
                f[f"unassigned8.{ck}.{o}"] = str(int(un[d.ranks.index(8)]))
    # Upsilon outcomes
    for ck, objs in d.ups["by_checkpoint"].items():
        for o, rows in objs.items():
            for r_ in rows:
                f[f"ups.{ck}.{o}.k{r_['k']}"] = g3(r_["ups"])
                f[f"cap.{ck}.{o}.k{r_['k']}"] = g3(r_["cap"])
                f[f"outcome.{ck}.{o}.k{r_['k']}"] = r_["outcome"]
            for oc in ("certified", "excluded", "undetermined"):
                ks = [str(r_["k"]) for r_ in rows if r_["outcome"] == oc]
                f[f"{oc}.{ck}.{o}"] = ("k = " + ", ".join(ks)) if ks else "none"
            f[f"ups.{ck}.{o}.max"] = g3(max((r_["ups"] for r_ in rows if fin(r_["ups"])), default=None))
            f[f"cap.{ck}.{o}.range"] = rng([r_["cap"] for r_ in rows], g3)
    # set descriptors: class purity, token concentration
    for ck in d.ck:
        for o, by_r in d.sets[ck].items():
            for r, rows in by_r.items():
                rows = [s_ for s_ in rows if s_.get("size")]
                if not rows:
                    continue
                cs = [max(s_["class_t"]) / s_["size"] for s_ in rows]
                ss = [max(s_["source"]) / s_["size"] for s_ in rows]
                f[f"sets{r}.{ck}.{o}.median_class_share"] = f2(np.median(cs))
                f[f"sets{r}.{ck}.{o}.n_class70"] = f"{sum(x >= 0.7 for x in cs)} of {len(rows)}"
                f[f"sets{r}.{ck}.{o}.n_source70"] = f"{sum(x >= 0.7 for x in ss)} of {len(rows)}"
                f[f"sets{r}.{ck}.{o}.median_top_token_share"] = f2(np.median([s_["top_token_share"] for s_ in rows]))
                f[f"sets{r}.{ck}.{o}.median_tokens_for_90"] = f2(np.median([s_["tokens_for_90"] for s_ in rows]), 0)
    # lag retention
    L = d.lagret
    f["ret.chance"] = f"{L.get('chance', float('nan')):.1e}"
    for v, rec in L["views"].items():
        mean = rec["mean"]
        for tau, x in zip(L["lags"], mean):
            f[f"ret.{v}.tau{tau}"] = f2(x)
        half = next((tau for tau, x in zip(L["lags"], mean) if x < 0.5 * mean[0]), None)
        f[f"ret.{v}.half_tau"] = f"τ = {half}" if half is not None else f"beyond τ = {L['lags'][-1]}"
    for v, rec in L["views"].items():
        keeps = []
        for j, row in enumerate(rec["per_vector"]):
            f[f"retv.{v}.v{j + 1}.tau0"], f[f"retv.{v}.v{j + 1}.tau16"] = f2(row[0]), f2(row[-1])
            kp = row[-1] / row[0] if row[0] else float("nan")
            f[f"retv.{v}.v{j + 1}.keep16"] = pct(kp)
            keeps.append((kp, j + 1))
        kmin, jmin = min(keeps)
        f[f"retv.{v}.min_keep16"] = f"vector {jmin} keeps {pct(kmin)} of its lag-0 value"
        f[f"retv.{v}.keep16.range_others"] = rng([k for k, j in keeps if j != jmin], pct)
    # comparisons
    for r_ in d.compare:
        a = r_["a"] + ("" if r_.get("checkpoint_a", "trained") == "trained" else "@" + r_["checkpoint_a"])
        b = r_["b"] + ("" if r_.get("checkpoint_b", "trained") == "trained" else "@" + r_["checkpoint_b"])
        for s_ in ("a8", "a16", "ami8", "ami16"):
            f[f"cmp.{a}.{b}.{s_}"] = f2(r_.get(s_))
    # trees
    for tid, tr in d.trees.items():
        nodes = tr["nodes"]
        thr = tr.get("threshold")
        splits = [nd for nd in nodes if nd.get("children")]
        f[f"tree.{tid}.n_nodes"] = str(len(nodes))
        f[f"tree.{tid}.n_splits"] = str(len(splits))
        f[f"tree.{tid}.threshold"] = g3(thr)
        f[f"tree.{tid}.n_clear"] = (f"{sum(1 for nd in splits if fin(nd.get('upsilon')) and nd['upsilon'] >= thr)} "
                                    f"of {len(splits)}") if thr is not None else "n/a"
        f[f"tree.{tid}.root.ups"] = g3(nodes[0].get("upsilon"))
        for ci, cid in enumerate(nodes[0].get("children") or []):
            nd = nodes[cid]
            ct = nd.get("class_t") or [0]
            j = int(np.argmax(ct))
            f[f"tree.{tid}.child{ci}.n"] = str(nd["n"])
            f[f"tree.{tid}.child{ci}.class"] = f"{d.classes[j].replace('_', ' ')} {pct(ct[j] / max(1, nd['n']))}"
            sr = nd.get("source") or [0]
            k = int(np.argmax(sr))
            f[f"tree.{tid}.child{ci}.source"] = f"{d.sources[k]} {pct(sr[k] / max(1, nd['n']))}"
            f[f"tree.{tid}.child{ci}.tokens"] = ", ".join(tokdisp(d.tok(a)) for a, _ in (nd.get("tok_t") or [])[:4])
    # shares along depth: max, min, range per view (trained and init)
    for ck in d.ck:
        for view in ("point", "state"):
            for g in SHARES:
                vals = [d.ex(ck, f"{view}_L{l}", g) for l in d.layers]
                ok = [(v, l) for v, l in zip(vals, d.layers) if fin(v)]
                if not ok:
                    continue
                vmax, lmax = max(ok)
                vmin, lmin = min(ok)
                f[f"byl.{ck}.{view}.{g}.max"] = f"{f2(vmax)} at L{lmax}"
                f[f"byl.{ck}.{view}.{g}.min"] = f"{f2(vmin)} at L{lmin}"
                f[f"byl.{ck}.{view}.{g}.range"] = rng([v for v, _ in ok])
    # per-set descriptors (r = 8) of the layout objects, 1-based set numbers as on the page
    for o in d.layout_objects:
        for s_ in d.sets["trained"].get(o, {}).get("8", []):
            if not s_.get("size"):
                continue
            n_ = s_["size"]
            key = f"set8.{o}.s{s_['set'] + 1}"
            ct, sr = np.asarray(s_["class_t"]), np.asarray(s_["source"])
            f[key + ".size"] = f"{n_:,}"
            f[key + ".snippets"] = f"{s_['snippets']:,}"
            f[key + ".class"] = ", ".join(f"{d.classes[j].replace('_', ' ')} {pct(ct[j] / n_)}" for j in np.argsort(-ct, kind="stable")[:2])
            f[key + ".source"] = ", ".join(f"{d.sources[j]} {pct(sr[j] / n_)}" for j in np.argsort(-sr, kind="stable")[:2])
            f[key + ".tokens"] = ", ".join(tokdisp(d.tok(a)) for a, _ in s_.get("tok_t", [])[:4])
            f[key + ".next"] = ", ".join(tokdisp(d.tok(a)) for a, _ in s_.get("tok_next", [])[:4])
            f[key + ".top_share"] = pct(s_.get("top_token_share"))
            f[key + ".n90"] = f"{s_.get('tokens_for_90'):,}"
    for tid, tr in d.trees.items():
        nodes = tr["nodes"]
        for ci, cid in enumerate(nodes[0].get("children") or []):
            nd = nodes[cid]
            ct, sr = np.asarray(nd.get("class_t") or [0]), np.asarray(nd.get("source") or [0])
            f[f"tree.{tid}.child{ci}.classes3"] = ", ".join(d.classes[j].replace("_", " ") for j in np.argsort(-ct, kind="stable")[:4])
            f[f"tree.{tid}.child{ci}.sources"] = ", ".join(f"{d.sources[j]} {sr[j]:,}" for j in np.argsort(-sr, kind="stable") if sr[j] >= 0.05 * nd["n"])
        clear = [nd for nd in nodes if nd.get("children") and fin(nd.get("upsilon")) and tr.get("threshold") is not None
                 and nd["upsilon"] >= tr["threshold"]]
        f[f"tree.{tid}.clear"] = "; ".join(
            f"node {nd['id']} (n = {nd['n']:,}, Υ = {g3(nd['upsilon'])}; children by source: "
            + " | ".join(", ".join(f"{d.sources[j]} {nodes[c]['source'][j]:,}" for j in np.argsort(-np.asarray(nodes[c]['source']))[:2])
                         for c in nd["children"]) + ")" for nd in clear) or "none"
    # targets, controls, verdict
    verdict_facts(d, f)
    # calibration
    c = d.calib
    f["calib.planted_min"] = g3(c.get("planted_min"))
    f["calib.structureless_max"] = g3(c.get("structureless_max"))
    f["calib.P1"] = "passes" if c.get("P1_prime") else "fails"
    f["calib.rule"] = str(c.get("rule"))
    f["calib.detection"] = "; ".join(f"s = {k}: {pct(v)}" for k, v in (c.get("detection_by_s") or {}).items())
    f["calib.n_planted"], f["calib.n_structureless"] = str(c.get("n_planted")), str(c.get("n_structureless"))
    cf = d.calib_full or {}
    f["calib.full.planted"] = g3(cf.get("planted_ups"))
    f["calib.full.structureless"] = g3(cf.get("structureless_max"))
    f["calib.full.status"] = ("calibrated at full size" if cf.get("calibrated") else
                              "not calibrated at full size" if cf else "full-size check not found")
    # validation
    for ck in d.ck:
        re_, md, pn = (list(filter(fin, x)) for x in gate_values(d, ck))
        sfx = "" if ck == "trained" else "." + ck
        f["val.row_err.max" + sfx] = g3(max(re_, default=None))
        f["val.min_dist.min" + sfx] = sci(min(md, default=None))
        f["val.pn.min" + sfx] = f2(min(pn, default=None), 0)
    R = d.validation.get("references") or d.meta.get("references") or {}
    for k, v in R.items():
        f[f"ref.{k}.n_dup"] = str(v.get("n_dup_pairs"))
        f[f"ref.{k}.min_dist"] = sci(v.get("min_dist"))
        f[f"ref.{k}.tie_sensitive"] = "tie-sensitive" if v.get("tie_sensitive") else "not tie-sensitive"
    f["val.seba_agree"] = f3(d.validation.get("minimum_seba_label_agreement"))
    f["val.records"] = str(d.validation.get("spectral_records"))
    diag_facts(d, f)
    return f


def target_rows(d: Data):
    """(id, target, label, value text, status) for SPEC 7, statuses recomputed from targets.json values and checked."""
    T = d.targets
    rows = []

    def hyp(key, text, value, ok):
        stored = (T.get(key) or {}).get("met")
        if ok is not None and stored is not None and bool(stored) != bool(ok):
            log(f"warning: {key}: targets.json says met={stored}, recomputed {ok}; using the recomputed value")
        st = "not determinable" if ok is None else ("met" if ok else "not met")
        rows.append((key, text, "hypothesis", value, st))

    t = T.get("K1") or {}
    hyp("K1", "B-point, r = 8 sets: AMI with C_c exceeds AMI with W (this target has no null)",
        f"AMI(C_c) = {f2(t.get('ami_Cc'))}, AMI(W) = {f2(t.get('ami_W'))}",
        (t["ami_Cc"] > t["ami_W"]) if fin(t.get("ami_Cc")) and fin(t.get("ami_W")) else None)
    t = T.get("K1b") or {}
    hyp("K1b", "B-point: κ above the upper end of the within-token permutation band (κ̃|T > 0 beyond the band)",
        f"κ = {f3(t.get('kappa'))}, band upper end {f3(t.get('band_hi'))}, κ̃|T = {f3(t.get('kappa_ex_T'))}",
        (t["kappa"] > t["band_hi"]) if fin(t.get("kappa")) and fin(t.get("band_hi")) else None)
    t = T.get("K2") or {}
    ok = None
    if all(fin(t.get(k)) for k in ("ami_W", "ami_Cc", "U_whole", "U_Bpoint")):
        ok = t["ami_W"] > t["ami_Cc"] and t["U_whole"] >= t["U_Bpoint"] + 0.2
    hyp("K2", "whole cell, r = 8 sets: AMI with W exceeds AMI with C_c; and Ũ(whole) ≥ Ũ(B-point) + 0.2",
        f"AMI(W) = {f2(t.get('ami_W'))}, AMI(C_c) = {f2(t.get('ami_Cc'))}; Ũ: whole {f2(t.get('U_whole'))}, "
        f"B-point {f2(t.get('U_Bpoint'))}", ok)
    t = T.get("K3") or {}
    ok = None
    if all(fin(t.get(k)) for k in ("T_L4", "T_L24", "kappa_L4", "kappa_L24")):
        ok = t["T_L4"] > t["T_L24"] and t["kappa_L24"] - t["kappa_L4"] >= 0.1
    hyp("K3", "point cells: T̃ higher at L4 than at L24, and κ̃ higher at L24 than at L4 by at least 0.1",
        f"T̃: L4 {f2(t.get('T_L4'))}, L24 {f2(t.get('T_L24'))}; κ̃: L4 {f2(t.get('kappa_L4'))}, "
        f"L24 {f2(t.get('kappa_L24'))}", ok)
    t = T.get("K4") or {}
    ok = None
    if all(fin(t.get(k)) for k in ("point_L16_ret0", "point_L16_ret2", "whole_ret0", "whole_ret16")):
        ok = t["point_L16_ret2"] <= 0.5 * t["point_L16_ret0"] and t["whole_ret16"] >= 0.5 * t["whole_ret0"]
    hyp("K4", "B-point's r = 8 set vectors lose half their lag-0 retention by τ = 2 (point L16); the whole "
              "cell's keep half at τ = 16",
        f"point L16: {f2(t.get('point_L16_ret0'))} → {f2(t.get('point_L16_ret2'))} at τ = 2; whole: "
        f"{f2(t.get('whole_ret0'))} → {f2(t.get('whole_ret16'))} at τ = 16", ok)
    t = T.get("K5") or {}
    cert = t.get("certified")
    hyp("K5", "some k ∈ R is certified in B-point (trained)",
        "certified: " + (", ".join(f"k = {k}" for k in cert) if cert else "none"), bool(cert) if cert is not None else None)
    t = T.get("K6") or {}
    ok = None
    if all(fin(t.get(k)) for k in ("kappa_L24_init", "kappa_L24_trained", "W_whole_init", "W_whole_trained")):
        ok = t["kappa_L24_init"] < 0.5 * t["kappa_L24_trained"] and t["W_whole_init"] < 0.5 * t["W_whole_trained"]
    hyp("K6", "at init: κ̃ at point L24 and W̃ of the whole cell are each below half their trained values",
        f"κ̃(L24): init {f2(t.get('kappa_L24_init'))}, trained {f2(t.get('kappa_L24_trained'))}; W̃(whole): "
        f"init {f2(t.get('W_whole_init'))}, trained {f2(t.get('W_whole_trained'))}", ok)
    return rows


REFS = ("bag", "recent", "pred")


def gate_values(d: Data, ck: str):
    """the pilot-gate quantities, read from the cells and spectra (SPEC 7): Sinkhorn row errors of every snapshot,
    smallest off-diagonal distance of every view snapshot (references exempt, SPEC 11), PN(psi_1) of every block."""
    cells = d.cells.get(ck, {})
    re_ = [c.get("row_err") for c in cells.values()]
    md = [c.get("min_dist") for n, c in cells.items() if n.split("_k")[0] not in REFS and not c.get("block")]
    pn = [(lambda a: a[0] if a is not None else None)(d.sp(ck, o, "pn")) for o in d.meta.get("blocks", [])
          if o in d.objects.get(ck, [])]
    return re_, md, pn


def control_rows(d: Data):
    rows = []
    V = (d.validation.get("controls") or {}).get("trained") or {}

    def ctl(key, text, label, rec):
        if rec is None:
            rows.append((key, text, label, "not in the data", "not run"))
            return
        if "pass" in rec:
            val = "; ".join(f"{k.replace('_', ' ')} {g3(v) if isinstance(v, (int, float)) else v}" for k, v in rec.items()
                            if k not in ("pass", "deviations", "definition", "precision") and not isinstance(v, (dict, list)))
            rows.append((key, text, label, val, "pass" if rec["pass"] else "fail"))
        else:
            rows.append((key, text, label, str(rec.get("status", "?")), "not run"))

    ctl("C1a", "v1's decode check, rerun on the trained checkpoint", "identity", V.get("C1a"))
    ctl("C1b", "v1's resumed-decode check, rerun on the trained checkpoint", "identity", V.get("C1b"))
    ctl("C3", "every lag-0 vector equals v1's stored value where v1 stored one (relative 1e-6)", "identity", V.get("C3"))
    p3 = V.get("P3")
    ctl("P3", "Gram identity for state and whole on a 500-item subsample (1e-12)", "[thm]",
        None if p3 is None else {"pass": p3.get("pass"), "max deviation": p3.get("max_dev"),
                                 "tolerance": p3.get("tolerance")})
    c = d.calib
    rows.append(("P1'", "fixture check seeds: planted blocks (s ≥ 1) certified at their k; structureless "
                        "certified at no k ∈ R", "planted",
                 f"planted minimum Υ̂ {g3(c.get('planted_min'))}, structureless maximum "
                 f"{g3(c.get('structureless_max'))}, θ = {g3(c.get('theta'))} ({c.get('rule')})",
                 "pass" if c.get("P1_prime") else "fail"))
    re_, md, pn = gate_values(d, "trained")
    zn = [bool(x) for x in ((d.validation.get("controls") or {}).get(ck, {}).get("zero_norm") for ck in d.ck) if x is not None]

    def gate(key, text, vals, ok_fn, fmt):
        vals = [v for v in vals if fin(v)]
        if not vals:
            rows.append((key, text, "gate", "not in the data", "not run"))
        else:
            rows.append((key, text, "gate", fmt(vals), "pass" if ok_fn(vals) else "fail"))
    gate("gate: Sinkhorn", "row error below 2e-12 in every snapshot", re_, lambda v: max(v) < 2e-12,
         lambda v: f"largest {g3(max(v))}")
    gate("gate: duplicates", "smallest off-diagonal cosine distance above 1e-9 in every view snapshot", md,
         lambda v: min(v) > 1e-9, lambda v: f"smallest {sci(min(v))}")
    rows.append(("gate: zero norm", "no zero-norm item in any view or constituent", "gate",
                 "none found" if not any(zn) else "zero-norm items present", "pass" if not any(zn) else "fail"))
    gate("gate: PN", "min PN(ψ₁) ≥ 10 over blocks", pn, lambda v: min(v) >= 10, lambda v: f"smallest {f2(min(v), 0)}")
    R = d.validation.get("references") or d.meta.get("references")
    if R:
        bad = [k for k, v in R.items() if v.get("tie_sensitive")]
        rows.append(("references", "reference snapshots are exempt from the duplicate check (SPEC 11); any with a "
                     "duplicate pair must not be tie-sensitive", "amendment",
                     "; ".join(f"{k}: {v.get('n_dup_pairs')} pair(s) below 1e-9 (smallest {sci(v.get('min_dist'))})"
                               for k, v in R.items()), "fail" if bad else "pass"))
    return rows


def verdict_facts(d: Data, f: dict):
    T = d.targets
    for k, rec in T.items():
        for kk, v in rec.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                f[f"tgt.{k}.{kk}"] = f2(v) if abs(v) >= 0.01 or v == 0 else f3(v)
    hy = target_rows(d)
    ct = control_rows(d)
    for key, _, _, val, st in hy + ct:
        f[f"v.{key}.status"] = st
        f[f"v.{key}.value"] = val
    f["v.n_hyp"] = str(len(hy))
    for st in ("met", "not met", "not determinable"):
        lst = [r[0] for r in hy if r[4] == st]
        f[f"v.n_{st.replace(' ', '_')}"] = str(len(lst))
        f[f"v.{st.replace(' ', '_')}_list"] = ", ".join(lst) or "none"
    f["v.n_ctl"] = str(len(ct))
    for st in ("pass", "fail", "not run"):
        lst = [r[0] for r in ct if r[4] == st]
        f[f"v.n_ctl_{st.replace(' ', '_')}"] = str(len(lst))
        f[f"v.ctl_{st.replace(' ', '_')}_list"] = ", ".join(lst) or "none"


def diag_facts(d: Data, f: dict):
    D = d.diag
    P = payload_f1(d)
    rs = P["r"]
    dep, rd = P["depth"], P["reading"]
    f["v1.r_list"] = "/".join(str(r) for r in rs)
    for g, name in (("Cc", "cc"), ("W", "src"), ("Cn", "cn"), ("T", "tok")):
        f[f"v1.depth.{name}.by_r"] = "/".join(f2(x[0]) for x in dep[g])
        f[f"v1.depth.{name}.range"] = rng([x[0] for x in dep[g]])
    i8 = rs.index(8) if 8 in rs else -1
    f["v1.depth.cc.r8"], f["v1.depth.src.r8"] = f2(dep["Cc"][i8][0]), f2(dep["W"][i8][0])
    f["v1.reading.src.range"] = rng([x for row in rd["W"] for x in row[1:]])
    f["v1.reading.cc.max"] = f2(max(x[2] for x in rd["Cc"]))
    f["v1.n_reading"] = str(P["n_reading"])
    f["v1.n_t"] = str(P["n_t"])
    sets = P["sets"]
    f["v1.sets.n"] = str(len(sets))
    f["v1.sets.median_class_share"] = f2(np.median([s_["cs"] for s_ in sets]))
    f["v1.sets.n_class70"] = str(sum(s_["cs"] >= 0.7 for s_ in sets))
    f["v1.sets.n_source70"] = str(sum(s_["ss"] >= 0.7 for s_ in sets))
    # within-source / within-class AMIs
    ts = list(D["d1"]["depth"])
    rr = list(D["d1"]["depth"][ts[0]])          # mean over the positions t, per rank r (as DIAG_NOTES reports)
    wcs = [np.mean([D["d1"]["depth"][t][r]["cur_class_within_source_ami"] for t in ts]) for r in rr]
    wsc = [np.mean([D["d1"]["depth"][t][r]["source_within_cur_class_ami"] for t in ts]) for r in rr]
    f["v1.cc_within_source.range"] = rng(wcs)
    f["v1.src_within_class.range"] = rng(wsc)
    ref = D["d1"].get("reference") or {}
    f["v1.ami_source_class.range"] = rng([v.get("ami_source_cur_class") for v in ref.values()])
    S = D["d2_trees"]["summary"]
    f["v1.tree.internal"] = str(S["internal_nodes"])
    f["v1.tree.balanced2pct"] = str(S["within_2pct_balanced"])
    f["v1.tree.ups_ge3"] = str(S["upsilon_ge_3"])
    f["v1.tree.ceiling_ge3"] = str(S["spectral_ceiling_ge_3"])
    f["v1.tree.root_ups"] = ", ".join(f"{k} {f2(v)}" for k, v in S["root_upsilon"].items())
    for tid in ("T1", "T4"):
        R = D["d2_recompute"][tid]
        ks = sorted(int(k) for k in R["by_k"] if k.isdigit())
        f[f"v1.{tid}.ups.range"] = rng([R["by_k"][str(k)]["upsilon"] for k in ks])
        f[f"v1.{tid}.cap.range"] = rng([R["by_k"][str(k)]["ceiling"] for k in ks])
        f[f"v1.{tid}.k.range"] = f"{ks[0]} to {ks[-1]}"
        f[f"v1.{tid}.src5"] = f2(R["by_k"].get("source_k5", {}).get("upsilon"))
        ge3 = [str(k) for k in ks if R["by_k"][str(k)]["ceiling"] >= 3]
        f[f"v1.{tid}.cap_ge3"] = ("k = " + ", ".join(ge3)) if ge3 else "no k"
    allu = [D["d2_recompute"][t]["by_k"][k]["upsilon"] for t in ("T1", "T4") for k in D["d2_recompute"][t]["by_k"] if k.isdigit()]
    f["v1.ups_k.range"] = rng(allu)
    X = D["d3"]
    S_ = X["sources"]
    ic, i_f = S_.index("code"), S_.index("forum")
    for cellname in ("stack_L24_t128", "point_L16_t128"):
        if cellname in X:
            f[f"v1.X.{cellname}.code_forum"] = f2(X[cellname]["X"][ic][i_f])
            f[f"v1.X.{cellname}.null_hi"] = f2(X[cellname]["null_band_hi"][ic][i_f])
            other = [X[cellname]["X"][a][b] for a in range(len(S_)) for b in range(a + 1, len(S_)) if {a, b} != {ic, i_f}]
            f[f"v1.X.{cellname}.others_max"] = f2(max(other))


# ----------------------------------------------------------------------------------------------- prose
def load_prose(name: str, facts: dict, real: bool) -> dict:
    p = HERE / "prose" / f"{name}.html"
    t = p.read_text()
    drop = "mockup" if real else "real"
    t = re.sub(rf"<!-- {drop} -->.*?<!-- /{drop} -->", "", t, flags=re.S)
    t = re.sub(r"<!--(?!\s*section:).*?-->", "", t, flags=re.S)       # other comments (incl. kept markers)

    def sub(mo):
        key = mo.group(1).strip()
        if key not in facts:
            near = [k for k in facts if k.split(".")[0] == key.split(".")[0]][:20]
            sys.exit(f"{p.name}: unknown fact '{key}'. Some facts with the same prefix:\n  " + "\n  ".join(near))
        v = html.escape(str(facts[key]))
        return f'<span class="tex2jax_ignore mathjax_ignore">{v}</span>' if "$" in v else v
    t = re.sub(r"\{\{\s*fact:([^}]+)\}\}", sub, t)
    parts = re.split(r"<!-- section: ([\w-]+) -->", t)
    out = {"_head": parts[0]}
    for k in range(1, len(parts), 2):
        out[parts[k]] = parts[k + 1]
    return out


def take(sections: dict, key: str, where: str) -> str:
    if key not in sections:
        sys.exit(f"prose/{where}.html has no section '{key}'")
    return sections.pop(key)


def md_inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def readme_deviations() -> str:
    """the README's DEVIATIONS list, rendered (kept in sync by building from the file)."""
    txt = (ROOT / "README.md").read_text()
    mo = re.search(r"^## DEVIATIONS[^\n]*\n(.*?)(?=^## )", txt, flags=re.S | re.M)
    if not mo:
        return "<p>(README.md has no DEVIATIONS section.)</p>"
    items = re.split(r"\n(?=\d+\. )", mo.group(1).strip())
    lis = []
    for it in items:
        it = re.sub(r"^\d+\.\s*", "", it.strip())
        it = re.sub(r"\s*\n\s*", " ", it)
        lis.append(f"<li>{md_inline(it)}</li>")
    return "<ol class=\"devs tex2jax_ignore mathjax_ignore\">" + "".join(lis) + "</ol>"   # dollar amounts are not TeX


# ----------------------------------------------------------------------------------------------- banner
def arch_banner(d: Data) -> str:
    mc = d.meta.get("model_config") or {}
    nl = int(mc.get("num_hidden_layers") or len(d.layers))
    W, x0, bw, gap = 990, 160, 30, 4.5
    y_res, y_blk, y_st = 46, 30, 86
    parts = [f'<svg viewBox="0 0 {W} 150" style="width:100%;min-width:640px;max-width:{W}px;font-family:system-ui;font-size:11px" '
             f'role="img" aria-label="mamba-130m: {nl} blocks; point view after each block, state view per block, '
             f'whole state over all blocks">']
    parts.append(f'<text x="4" y="{y_res - 4}" font-weight="650">token context (i, t)</text>')
    parts.append(f'<text x="4" y="{y_res + 10}" fill="#6b665e">[EOT] + tokens 1..t</text>')
    x_end = x0 + nl * (bw + gap)
    parts.append(f'<line x1="{x0 - 12}" y1="{y_res}" x2="{x_end + 10}" y2="{y_res}" stroke="#8d877c" stroke-width="1.4"/>')
    for l in range(1, nl + 1):
        x = x0 + (l - 1) * (bw + gap)
        parts.append(f'<rect x="{x}" y="{y_blk}" width="{bw}" height="32" rx="4" fill="#dfe8f3" stroke="#8d877c"/>')
        parts.append(f'<text x="{x + bw / 2}" y="{y_blk + 20}" text-anchor="middle" font-size="9.5">{l}</text>')
        parts.append(f'<circle cx="{x + bw + gap / 2}" cy="{y_res}" r="2.6" fill="#2a78d6"/>')
        parts.append(f'<rect x="{x + 3}" y="{y_st}" width="8" height="16" fill="#f6e7d8" stroke="#b9875a"/>')
        parts.append(f'<rect x="{x + 13}" y="{y_st}" width="{bw - 16}" height="16" fill="#f1d9bf" stroke="#b9875a"/>')
        parts.append(f'<line x1="{x + bw / 2}" y1="{y_blk + 32}" x2="{x + bw / 2}" y2="{y_st}" stroke="#b9875a" stroke-dasharray="2,2"/>')
    parts.append(f'<rect x="{x0 - 4}" y="{y_st - 5}" width="{x_end - x0 + 4}" height="27" rx="5" fill="none" stroke="#c2410c" stroke-width="1.2"/>')
    parts.append(f'<text x="{x0 - 10}" y="{y_st + 12}" text-anchor="end" fill="#8a4a2a">state c, z</text>')
    parts.append(f'<text x="{x_end + 14}" y="{y_res + 4}">p<tspan dy="3" font-size="8">t</tspan></text>')
    parts.append(f'<text x="{x0}" y="{y_st + 40}" fill="#8a4a2a">whole state Σ<tspan dy="3" font-size="8">t</tspan>'
                 f'<tspan dy="-3">: all {2 * nl} constituents (c and z of every layer), Gram-averaged</tspan></text>')
    parts.append(f'<text x="{x0}" y="{y_res - 22}" fill="#2a78d6">● point view: residual h</text>')
    parts.append(f'<text x="{x0 + 150}" y="{y_res - 22}" fill="#6b665e">grey boxes: Mamba blocks 1..{nl}; blue dots: the residual after each block</text>')
    parts.append("</svg>")
    bits = [f"<b>{html.escape((d.meta.get('model') or '?').split('/')[-1])}</b>", f"{nl} Mamba blocks"]
    for k, lab in (("hidden_size", "width"), ("intermediate_size", "inner width"), ("state_size", "SSM state"),
                   ("conv_kernel", "conv kernel")):
        if mc.get(k):
            bits.append(f"{lab} {mc[k]}")
    bits.append(f"checkpoints: {', '.join(d.ck)}")
    bits.append(f"N = {d.n:,} token contexts from {len(np.unique(d.items['snippet'])):,} snippets")
    txt = (" · ".join(bits) + "<br><span style='color:#6b665e'>An <b>item</b> is a token context: a snippet "
           "prefix ending at a sampled token t. A <b>snapshot</b> is one view at one layer (and lag), at one checkpoint: "
           "<b>point</b> = the residual after block l, <b>state</b> = layer l's recurrent state (conv buffer c and SSM "
           "state z), <b>whole</b> = all of them. The model's next-token distribution p<sub>t</sub> enters only as a "
           "reference and for validation.</span>")
    mock = ""
    if d.fixture:
        what = ("planted structure: k = 8 Zipf-sized classes of items shared across 24 snapshots, tokens nested in "
                "classes, snippets of 5 sources, lags that decay" if d.fixture == "planted"
                else "a structureless twin: every view is noise, with the same tokens and groupings")
        mock = (f'<div class="mockup-note"><b>MOCKUP.</b> Built from the synthetic fixture <code>{html.escape(d.dir.name)}'
                f'</code> ({what}). Tokens and text are fake. The numbers show that the page reads the data correctly; '
                'none of them is a finding about Mamba.</div>')
    head = " · ".join(bits)
    defs = txt[len(head):]
    return mock + (f'<figure class="arch"><figcaption>{head}</figcaption><details class="archd"><summary>model '
                   f'diagram ({nl} blocks, the views read at each) and definitions of item and snapshot</summary><div class="archscroll">{"".join(parts)}</div><figcaption>'
                   f'{defs.removeprefix("<br>")}</figcaption></details></figure>')


# ----------------------------------------------------------------------------------------------- JS
JS_HELPERS = r"""
const DEC = new Map();
function A(o) { if (!o) return null; if (DEC.has(o)) return DEC.get(o);
  const raw = atob(o.b), u = new Uint8Array(raw.length); for (let i = 0; i < raw.length; i++) u[i] = raw.charCodeAt(i);
  const C = {u1: Uint8Array, i1: Int8Array, u2: Uint16Array, i2: Int16Array, i4: Int32Array, f4: Float32Array}[o.t];
  const a = new C(u.buffer); DEC.set(o, a); return a; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
function visTok(s) { return s.replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/ /g, '␣'); }
function runTok(s) { return s.replace(/\n/g, '↵').replace(/\t/g, '⇥'); }
const CLASS_PAL = __CLASS_PAL__, SOURCE_PAL = __SOURCE_PAL__, SET_PAL = __SET_PAL__, GSTYLE = __GSTYLE__,
      GLABEL = __GLABEL__, GSHORT = __GSHORT__, UNASSIGNED = '#d4d4d4';
const TOKS = DATA.toks;
function f2(v) { return v == null ? 'n/a' : Number(v).toFixed(2); }
function f3(v) { return v == null ? 'n/a' : Number(v).toFixed(3); }
function g3(v) { if (v == null) return 'n/a'; const a = Math.abs(v); return a >= 1e4 || (a > 0 && a < 1e-3) ? v.toExponential(2) : Number(v.toPrecision(3)).toString(); }
// panel layout: n panels side by side when the figure is wide, stacked when narrow
function PANELS(el, n, h, ratios) {
  const wide = el.clientWidth >= 720, gap = wide ? 0.07 : 0.16, lay = {}, dom = [];
  ratios = ratios || Array(n).fill(1); const tot = ratios.reduce((a, b) => a + b, 0);
  let pos = 0;
  for (let i = 0; i < n; i++) {
    const xs = i ? 'xaxis' + (i + 1) : 'xaxis', ys = i ? 'yaxis' + (i + 1) : 'yaxis', sfx = i ? String(i + 1) : '';
    const w = (1 - gap * (n - 1)) * (wide ? ratios[i] / tot : 1 / n);
    const d = [pos, pos + w]; pos += w + gap; dom.push(d);
    if (wide) { lay[xs] = {domain: d, anchor: 'y' + sfx}; lay[ys] = {domain: [0, 1], anchor: 'x' + sfx}; }
    else { lay[xs] = {domain: [0, 1], anchor: 'y' + sfx}; lay[ys] = {domain: [1 - d[1], 1 - d[0]], anchor: 'x' + sfx}; }
  }
  const title = (i, text) => wide
    ? {text: '<b>' + text + '</b>', x: dom[i][0], xref: 'paper', y: 1.0, yref: 'paper', yanchor: 'bottom', xanchor: 'left', showarrow: false, font: {size: 12.5}}
    : {text: '<b>' + text + '</b>', x: 0, xref: 'paper', y: 1 - dom[i][0], yref: 'paper', yanchor: 'bottom', xanchor: 'left', showarrow: false, font: {size: 12.5}};
  return {wide, lay, height: wide ? h : Math.round(h * n * 0.78), title};
}
const DASH = {solid: '', dash: '6,3', dot: '2,2.5', dashdot: '6,2,2,2'};
const GLYPH = {circle: '●', x: '✕', 'diamond-open': '◇', diamond: '◆', star: '★', square: '■', 'triangle-up-open': '△', 'triangle-down-open': '▽', 'circle-open': '○'};
// an HTML legend above the plot (Plotly's own legend collides with panel titles when it wraps)
function htmlLegend(el, items) {
  let box = el.previousElementSibling;
  if (!box || !box.classList.contains('hleg')) { box = document.createElement('div'); box.className = 'hleg'; el.parentNode.insertBefore(box, el); }
  box.innerHTML = items.map(it => {
    if (it.head) return `<b>${esc(it.head)}</b>`;
    const sw = it.glyph ? `<span class="gl" style="color:${it.color};opacity:${it.opacity || 1}">${GLYPH[it.glyph] || it.glyph}</span>`
      : `<svg width="28" height="10"><line x1="1" y1="5" x2="27" y2="5" stroke="${it.color}" stroke-width="${it.width || 2.4}" stroke-dasharray="${DASH[it.dash || 'solid']}" opacity="${it.opacity || 1}"/></svg>`;
    return `<span>${sw}${esc(it.name)}</span>`; }).join('');
}
const gItem = g => ({name: GLABEL[g], color: GSTYLE[g][0], dash: GSTYLE[g][1]});
function legendOnly(name, color, dash, mode) { return {x: [null], y: [null], name, mode: mode || 'lines', line: {color, dash: dash || 'solid', width: 2}, marker: {color}, showlegend: true, hoverinfo: 'skip'}; }
function bar(counts, pal, names) {
  const tot = counts.reduce((a, b) => a + b, 0) || 1;
  return '<div class="mkbar">' + counts.map((c, i) => c ? `<span style="width:${100 * c / tot}%;background:${pal[i % pal.length]}" title="${esc(names[i])}: ${c} (${(100 * c / tot).toFixed(0)}%)"></span>` : '').join('') + '</div>';
}
function key(names, pal) { return '<div class="key">' + names.map((n, i) => `<span><i style="background:${pal[i % pal.length]}"></i>${esc(n.replace(/_/g, ' '))}</span>`).join('') + '</div>'; }
function chips(list) { return (list || []).map(([t, c]) => `<span class="chip"><code class="tk">${esc(visTok(TOKS[t]))}</code><span class="cnt">${c}</span></span>`).join(' '); }
function row(lab, val) { return `<div class="row"><span class="lab">${lab}</span><span class="rv">${val}</span></div>`; }
"""

JS_EXPLORER = r"""
const EX = {set: null, item: null};
let EXI = null;
function exItems() {
  if (EXI) return EXI;
  const I = DATA.ex.items, o = {};
  for (const k of Object.keys(I)) o[k] = A(I[k]);
  o.n = DATA.ex.n; o.w = DATA.ex.before + DATA.ex.after;
  o.hover = new Array(o.n);
  for (let i = 0; i < o.n; i++) {
    let s = '';
    for (let j = Math.max(0, DATA.ex.before - 10); j < o.w; j++) { const t = o.ctx[i * o.w + j]; if (t === 65535) continue;
      const str = esc(runTok(TOKS[t])); s += j === DATA.ex.before - 1 ? '<b>⟦' + str + '⟧</b>' : (j >= DATA.ex.before ? '<span style="color:#999">' + str + '</span>' : str); }
    o.hover[i] = s + '<br>t = ' + o.t[i] + ' · class ' + DATA.ex.classes[o.Cc[i]] + ' · ' + DATA.ex.sources[o.W[i]];
  }
  EXI = o; return o;
}
function ctxHTML(i, before) {
  const I = exItems(), B = DATA.ex.before; let s = '';
  for (let j = Math.max(0, B - (before || B)); j < I.w; j++) { const t = I.ctx[i * I.w + j]; if (t === 65535) continue;
    const str = esc(runTok(TOKS[t]));
    s += j === B - 1 ? '<mark>' + str.replace(/^ /, '\u00a0') + '</mark>' : (j >= B ? '<span class="aft">' + str + '</span>' : str); }
  return '<span class="ctx">' + s + '</span>';
}
function exCategory(c, o, I) {
  // -> {cat: Int array (-1 = unassigned), names, pal} or {cont: Float array}
  if (c.col === 'sets') { const lab = A(o.lab[String(c.r)]); return {cat: lab, names: Array.from({length: c.r}, (_, j) => 'set ' + (j + 1)), pal: SET_PAL}; }
  if (c.col === 'cls') return {cat: I.Cc, names: DATA.ex.classes, pal: CLASS_PAL};
  if (c.col === 'clsn') return {cat: I.Cn, names: DATA.ex.classes, pal: CLASS_PAL};
  if (c.col === 'src') return {cat: I.W, names: DATA.ex.sources, pal: SOURCE_PAL};
  if (c.col === 'snip') return {snip: true};
  const k = Number(c.col.slice(3)); const psi = A(o.psi), n = I.n; const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = psi[(k - 1) * n + i];
  return {cont: v, k};
}
function drawLayout(el, d, c) {
  if (EX.obj !== c.obj || EX.r !== c.r) { if (EX.obj !== undefined) EX.set = null; EX.obj = c.obj; EX.r = c.r; }
  const E = DATA.ex, I = exItems(), o = E.lay[c.obj], n = I.n, xy = A(o.xy);
  const X = new Float32Array(n), Y = new Float32Array(n);
  for (let i = 0; i < n; i++) { X[i] = xy[2 * i] / 65535; Y[i] = xy[2 * i + 1] / 65535; }
  const C = exCategory(c, o, I), traces = [];
  const ms = n > 4000 ? 3.2 : 4.5;
  const pick = idx => ({x: idx.map(i => X[i]), y: idx.map(i => Y[i]), customdata: idx, text: idx.map(i => I.hover[i])});
  if (C.cat) {
    const groups = {}; for (let i = 0; i < n; i++) { const g = C.cat[i]; (groups[g] = groups[g] || []).push(i); }
    const keys = Object.keys(groups).map(Number).sort((a, b) => a - b);
    for (const g of keys) {
      const dim = c.col === 'sets' && EX.set != null && g !== EX.set;
      traces.push(Object.assign(pick(groups[g]), {type: 'scattergl', mode: 'markers', name: g < 0 ? 'unassigned' : C.names[g],
        marker: {size: ms, color: g < 0 ? UNASSIGNED : C.pal[g % C.pal.length], opacity: dim ? 0.12 : (g < 0 ? 0.55 : 0.85)},
        hovertemplate: '%{text}<extra>' + esc(g < 0 ? 'unassigned' : C.names[g]) + '</extra>'}));
    }
  } else if (C.snip) {
    const idx = Array.from({length: n}, (_, i) => i);
    const pal = SET_PAL.concat(CLASS_PAL);
    traces.push(Object.assign(pick(idx), {type: 'scattergl', mode: 'markers', name: 'snippet (colours repeat)',
      marker: {size: ms, color: idx.map(i => pal[(I.U[i] * 7919) % pal.length]), opacity: 0.85}, hovertemplate: '%{text}<extra>snippet</extra>'}));
  } else {
    const idx = Array.from({length: n}, (_, i) => i).sort((a, b) => Math.abs(C.cont[a]) - Math.abs(C.cont[b]));
    traces.push(Object.assign(pick(idx), {type: 'scattergl', mode: 'markers', name: 'ψ' + C.k,
      marker: {size: ms, color: idx.map(i => C.cont[i]), cmin: -127, cmax: 127, colorscale: [[0, '#2166ac'], [0.5, '#e8e7e2'], [1, '#b2182b']],
        colorbar: {title: {text: 'ψ' + C.k + ' / max|ψ' + C.k + '|', side: 'right'}, tickvals: [-127, 0, 127], ticktext: ['−1', '0', '1'], thickness: 10, len: 0.5}},
      hovertemplate: '%{text}<extra>ψ' + C.k + '</extra>'}));
  }
  if (EX.item != null) traces.push({type: 'scattergl', mode: 'markers', x: [X[EX.item]], y: [Y[EX.item]], name: 'selected item',
    marker: {size: 14, color: 'rgba(0,0,0,0)', line: {color: '#16161a', width: 2}}, hoverinfo: 'skip'});
  const h = Math.max(380, Math.min(680, el.clientWidth * 0.78));
  Plotly.react(el, traces, LAYOUT({height: h, margin: {l: 6, r: 6, t: 6, b: 6}, hovermode: 'closest',
    xaxis: {visible: false, range: [-0.02, 1.02], constrain: 'domain'}, yaxis: {visible: false, range: [-0.02, 1.02], scaleanchor: 'x'},
    legend: el.clientWidth >= 720 ? {orientation: 'v', x: 1.0, y: 1, xanchor: 'left', font: {size: 11}} : {orientation: 'h', y: -0.02, font: {size: 10.5}},
    uirevision: c.obj, hoverlabel: {align: 'left', font: {size: 12}}}), CONFIG);
  if (!el._bound) { el._bound = true;
    el.on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (!p || p.customdata == null) return; EX.item = p.customdata; drawTab('explorer'); }); }
}
function drawItem(el, d, c) {
  const I = exItems();
  if (EX.item == null) { el.innerHTML = '<p class="hint">Click a point in the layout to see its context here.</p>'; return; }
  const i = EX.item, o = DATA.ex.lay[c.obj], lab = A(o.lab[String(c.r)])[i];
  let preds = ''; for (let j = 0; j < 5; j++) preds += `<code class="tk">${esc(visTok(TOKS[I.top5[5 * i + j]]))}</code><span class="cnt">${(I.top5p[5 * i + j] / 10).toFixed(1)}%</span> `;
  el.innerHTML = `<div class="itemcard tex2jax_ignore mathjax_ignore"><div>${ctxHTML(i)}</div>
    <div class="meta">item ${i} · snippet ${I.U[i]} · t = ${I.t[i]} · token at t <code class="tk">${esc(visTok(TOKS[I.T[i]]))}</code>, at t+1 <code class="tk">${esc(visTok(TOKS[I.Nx[i]]))}</code>
    · in ${esc(DATA.ex.labels[c.obj])} at r = ${c.r}: ${lab < 0 ? 'unassigned' : 'set ' + (lab + 1)}</div>
    <div class="meta"><span class="val-tag">validation</span> class at t: ${esc(DATA.ex.classes[I.Cc[i]])} · source: ${esc(DATA.ex.sources[I.W[i]])}
    · trained model's top-5 next tokens: ${preds}</div></div>`;
}
function drawSets(el, d, c) {
  const o = DATA.ex.lay[c.obj], sets = (o.sets || {})[String(c.r)] || [], un = (o.unassigned || {})[String(c.r)];
  const total = DATA.ex.n;
  let h = c.obj.startsWith('Blag') ? `<div class="mockup-note"><b>Lag block.</b> This block averages walks over the same items read 0..16 tokens later; its sets are described with the tokens, classes and contexts at lag 0 (the item's own token t), not with what was read later.</div>` : '';
  h += `<p class="hint">${esc(DATA.ex.labels[c.obj])}, hard SEBA sets at r = ${c.r}: ${sets.filter(s => s.size).length} sets, ${un == null ? '?' : un} of ${total} items unassigned. Click a set's header to isolate it in the layout (colour by sets). Class and source make-up are validation only.</p>`;
  h += key(DATA.ex.classes, CLASS_PAL) + key(DATA.ex.sources, SOURCE_PAL);
  h += '<div class="setgrid">';
  for (const s of sets) {
    const col = SET_PAL[s.set % SET_PAL.length], on = EX.set === s.set;
    if (!s.size) { h += `<div class="setcard"><div class="sethead"><i style="background:${col}"></i><b>set ${s.set + 1}</b> · empty</div></div>`; continue; }
    const ex = (s.ex || []).map(i => '<li>' + ctxHTML(i, 16) + '</li>');
    h += `<div class="setcard${on ? ' on' : ''}"><div class="sethead" data-set="${s.set}"><i style="background:${col}"></i><b>set ${s.set + 1}</b> · ${s.size} items (${(100 * s.size / total).toFixed(1)}%) · ${s.snip} snippets</div>
      ${row('token at t', chips(s.tt))}
      ${row('concentration', 'commonest token ' + (100 * s.share).toFixed(0) + '% of the set; ' + s.n90 + ' distinct tokens cover 90%')}
      ${row('token at t+1', chips(s.tn))}
      ${row('class at t', bar(s.ct, CLASS_PAL, DATA.ex.classes))}
      ${row('class at t+1', bar(s.cn, CLASS_PAL, DATA.ex.classes))}
      ${row('source', bar(s.src, SOURCE_PAL, DATA.ex.sources))}
      <ol class="exs">${ex.slice(0, 4).join('')}</ol>${ex.length > 4 ? '<details><summary>' + (ex.length - 4) + ' more examples</summary><ol class="exs" start="5">' + ex.slice(4).join('') + '</ol></details>' : ''}</div>`;
  }
  h += '</div>';
  el.innerHTML = '<div class="tex2jax_ignore mathjax_ignore">' + h + '</div>';
  el.querySelectorAll('.sethead[data-set]').forEach(x => x.onclick = () => { const j = Number(x.dataset.set); EX.set = EX.set === j ? null : j; drawTab('explorer'); });
}
function drawEnds(el, d, c) {
  const o = DATA.ex.lay[c.obj], k = Number(c.k);
  if (!o.ends) { el.innerHTML = '<p class="hint">No eigenfunction ends stored for this object.</p>'; return; }
  const e = o.ends[k - 1], lam = o.lam ? o.lam[k - 1] : null;
  const col = (lst, title) => `<div><h4>${title}</h4><ol class="exs">${lst.map(i => '<li>' + ctxHTML(i, 16) + '</li>').join('')}</ol></div>`;
  let eta = '';
  for (const g of Object.keys(o.eta || {})) { const r = o.eta[g]; const v = r.v[k - 1], hi = r.hi[k - 1];
    eta += `<span class="eta${v > hi ? ' above' : ''}"><b>${esc(GSHORT[g])}</b> ${f2(v)} <small>(chance ${f3(r.c)}, band top ${f3(hi)})</small></span>`; }
  el.innerHTML = `<div class="tex2jax_ignore mathjax_ignore"><p class="hint">${esc(DATA.ex.labels[c.obj])}, ψ<sub>${k}</sub> (λ<sub>${k}</sub> = ${lam == null ? 'n/a' : lam.toFixed(4)}). η² of ψ<sub>${k}</sub> against each grouping (between-group share of its variance; bold border = above the permutation band; W, C_c, C_n are validation): </p><div class="etas">${eta}</div>
    <div class="cols2">${col(e[1], 'most negative ψ<sub>' + k + '</sub> (12 items)')}${col(e[0], 'most positive ψ<sub>' + k + '</sub> (12 items)')}</div></div>`;
}
"""

JS_F1 = r"""(el, d, c) => {
  const P = PANELS(el, 3, 400, [1, 1, 1.15]), tr = [], ann = [];
  const G = ['Cc', 'W', 'Cn', 'T'];
  [['depth', 'x', 'y'], ['reading', 'x2', 'y2']].forEach(([blk, xa, ya], pi) => {
    for (const g of G) { const s = d[blk][g]; if (!s) continue;
      tr.push({x: d.r, y: s.map(v => v[0]), xaxis: xa, yaxis: ya, mode: 'lines+markers', name: GLABEL[g], legendgroup: g, showlegend: pi === 0,
        line: {color: GSTYLE[g][0], dash: GSTYLE[g][1], width: 2}, marker: {size: 6},
        error_y: {type: 'data', symmetric: false, array: s.map(v => v[2] - v[0]), arrayminus: s.map(v => v[0] - v[1]), thickness: 1, width: 3, color: GSTYLE[g][0]},
        hovertemplate: GLABEL[g] + '<br>r = %{x}: mean AMI %{y:.2f}<extra>' + blk + '</extra>'}); }
  });
  const byCls = {}; d.sets.forEach(s => (byCls[s.cls] = byCls[s.cls] || []).push(s));
  for (const cl of Object.keys(byCls)) { const L = byCls[cl], ci = d.classes.indexOf(cl);
    tr.push({x: L.map(s => s.ss), y: L.map(s => s.cs), xaxis: 'x3', yaxis: 'y3', mode: 'markers', name: cl.replace(/_/g, ' '), legendgroup: 'cls' + cl, showlegend: false,
      marker: {size: L.map(s => 5 + Math.sqrt(s.size) / 1.6), color: CLASS_PAL[ci % CLASS_PAL.length], opacity: 0.8, line: {color: 'white', width: 0.5}},
      text: L.map(s => `t = ${s.t}, set ${s.set}, ${s.size} items<br>majority class ${s.cls} (${(100 * s.cs).toFixed(0)}%)<br>majority source ${s.src} (${(100 * s.ss).toFixed(0)}%)<br>${esc(s.top)}`),
      hovertemplate: '%{text}<extra></extra>'}); }
  tr.push({x: [0, 1], y: [0, 1], xaxis: 'x3', yaxis: 'y3', mode: 'lines', line: {color: '#bbb', dash: 'dot', width: 1}, showlegend: false, hoverinfo: 'skip'});
  htmlLegend(el, [{head: '(a, b)'}, ...G.map(gItem), {head: '(c) majority class:'}, ...Object.keys(byCls).map(cl => ({name: cl.replace(/_/g, ' '), color: CLASS_PAL[d.classes.indexOf(cl) % CLASS_PAL.length], glyph: 'circle'}))]);
  const ax = t => merge(AXIS, {title: {text: t}});
  Plotly.react(el, tr, LAYOUT(merge(P.lay, {height: P.height, margin: {t: 30, l: 56, r: 12, b: 48}, showlegend: false,
    xaxis: merge(P.lay.xaxis, ax('SEBA rank r')), yaxis: merge(P.lay.yaxis, merge(ax('AMI (sets vs grouping)'), {range: [0, 1]})),
    xaxis2: merge(P.lay.xaxis2, ax('SEBA rank r')), yaxis2: merge(P.lay.yaxis2, merge(ax('AMI'), {range: [0, 1]})),
    xaxis3: merge(P.lay.xaxis3, merge(ax('majority-source share of the set'), {range: [0, 1.02]})), yaxis3: merge(P.lay.yaxis3, merge(ax('majority class-at-t share'), {range: [0, 1.02]})),
    annotations: [P.title(0, '(a) v1 depth blocks (one t, L4..L24)'), P.title(1, '(b) v1 reading blocks (one layer, t = 4..128)'), P.title(2, '(c) v1 depth-block sets, r = 8')]})), CONFIG);
}"""

JS_F1_OV = r"""(el, d, c) => {
  const tr = [], G = ['Cc', 'W'];
  const xs = ['depth blocks<br>(unscored in v1)', 'reading blocks<br>(v1 headline)'];
  for (const g of G) {
    const v = [d.depth[g], d.reading[g]].map(s => s[s.length - 1]);
    tr.push({type: 'bar', x: xs, y: v.map(x => x[0]), name: GLABEL[g], marker: {color: GSTYLE[g][0], opacity: g === 'W' ? 0.55 : 1},
      error_y: {type: 'data', symmetric: false, array: v.map(x => x[2] - x[0]), arrayminus: v.map(x => x[0] - x[1]), color: '#555', thickness: 1},
      hovertemplate: GLABEL[g] + ': mean AMI %{y:.2f}<extra>r = ' + d.r[d.r.length - 1] + '</extra>'});
  }
  Plotly.react(el, tr, LAYOUT({height: 300, barmode: 'group', yaxis: merge(AXIS, {title: {text: 'AMI of v1 sets (r = ' + d.r[d.r.length - 1] + ')'}, range: [0, 1]}),
    xaxis: AXIS, margin: {t: 30, l: 60, r: 10, b: 50}}), CONFIG);
}"""

JS_F2 = r"""(el, d, c) => {
  const P = PANELS(el, 3, 430, [1, 1, 0.7]), tr = [];
  let lo = 0, hi = 0;
  const upd = a => a.forEach(v => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
  const show = c.f2q === 'all' ? d.q : d.q.filter(q => ({input: ['T', 'Nx', 'U'], output: ['A', 'kappa', 'kappaT'], validation: ['W', 'Cc', 'Cn']})[c.f2q].includes(q));
  [['point', 'x', 'y'], ['state', 'x2', 'y2']].forEach(([v, xa, ya], pi) => {
    for (const ck of Object.keys(d.views[v]).sort((a, b) => a === 'trained' ? 1 : -1)) {
      const init = ck !== 'trained';
      for (const q of show) { const y = d.views[v][ck][q]; if (!y || y.every(x => x == null)) continue; upd(y);
        tr.push({x: d.layers, y, xaxis: xa, yaxis: ya, mode: 'lines', name: GLABEL[q], legendgroup: q, showlegend: pi === 0 && !init,
          line: {color: GSTYLE[q][0], dash: GSTYLE[q][1], width: init ? 1.2 : 2.2}, opacity: init ? 0.35 : 1,
          hovertemplate: GLABEL[q] + '<br>' + v + ' L%{x}: %{y:.3f}<extra>' + ck + '</extra>'}); }
    }
  });
  for (const ck of Object.keys(d.objs)) { const init = ck !== 'trained';
    for (const q of show) { const y = d.objs[ck][q]; if (!y || y.every(x => x == null)) continue; upd(y);
      tr.push({x: d.objn, y, xaxis: 'x3', yaxis: 'y3', mode: 'markers', name: GLABEL[q], legendgroup: q, showlegend: false,
        marker: {color: GSTYLE[q][0], size: init ? 6 : 9, symbol: GSTYLE[q][1] === 'solid' ? 'circle' : (GSTYLE[q][1] === 'dash' ? 'diamond' : 'square'), opacity: init ? 0.35 : 1},
        hovertemplate: GLABEL[q] + '<br>%{x}: %{y:.3f}<extra>' + ck + '</extra>'}); } }
  const yr = [Math.min(-0.05, lo - 0.03), Math.max(0.2, hi + 0.05)];
  htmlLegend(el, [...show.map(gItem), {name: 'init (same style, faint)', color: '#888', opacity: 0.4, width: 1.2}]);
  const ax = t => merge(AXIS, {title: {text: t}});
  Plotly.react(el, tr, LAYOUT(merge(P.lay, {height: P.height, margin: {t: 30, l: 56, r: 12, b: 70}, showlegend: false,
    xaxis: merge(P.lay.xaxis, ax('layer l')), yaxis: merge(P.lay.yaxis, merge(ax('excess share'), {range: yr})),
    xaxis2: merge(P.lay.xaxis2, ax('layer l')), yaxis2: merge(P.lay.yaxis2, {range: yr, ...AXIS}),
    xaxis3: merge(P.lay.xaxis3, merge(AXIS, {tickangle: -35})), yaxis3: merge(P.lay.yaxis3, {range: yr, ...AXIS}),
    shapes: ['y', 'y2', 'y3'].map(y => ({type: 'line', xref: 'paper', x0: 0, x1: 1, yref: y, y0: 0, y1: 0, line: {color: '#999', width: 1, dash: 'dot'}})),
    annotations: [P.title(0, 'point: residual after block l'), P.title(1, "state: layer l's c and z"), P.title(2, 'blocks, whole state, references')]})), CONFIG);
}"""

JS_F3 = r"""(el, d, c) => {
  const P = PANELS(el, 2, 420, [1, 1.1]), tr = [], o = d.objs[c.f3obj], th = d.theta;
  for (const g of d.calib.gaps) tr.push({x: g.map((_, i) => i + 2), y: g, mode: 'lines', line: {color: '#cfcfcf', width: 1, dash: 'dot'}, showlegend: false, hoverinfo: 'skip'});
  tr.push(legendOnly('fixture, structureless (check seeds)', '#bdbdbd', 'dot'));
  for (const k of d.order) { if (k === c.f3obj) continue; const q = d.objs[k];
    tr.push({x: q.gap.map((_, i) => i + 2), y: q.gap, mode: 'lines', line: {color: '#9fb6cf', width: 1}, opacity: 0.6, showlegend: false, name: q.label, hovertemplate: q.label + '<br>k = %{x}: %{y:.3f}<extra></extra>'}); }
  tr.push({x: o.gap.map((_, i) => i + 2), y: o.gap, mode: 'lines+markers', name: o.label + ': μ_k/μ_{k−1}', line: {color: '#16161a', width: 2.2}, marker: {size: 4},
    hovertemplate: 'k = %{x}: μ_k/μ_{k−1} = %{y:.3f}<extra></extra>'});
  const ok = {certified: '#1b7a3e', excluded: '#b04a4a', undetermined: '#c08a3e', 'n/a': '#999'};
  const sym = {certified: 'circle', excluded: 'x', undetermined: 'diamond-open', 'n/a': 'circle-open'};
  const slk = Object.keys(d.calib.sl).map(Number);
  tr.push({x: slk, y: slk.map(k => d.calib.sl[k]), xaxis: 'x2', yaxis: 'y2', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(180,180,180,0.18)', line: {color: '#bbb', width: 1}, name: 'fixture: structureless max Υ̂ (check seeds)', hovertemplate: 'k = %{x}: %{y:.2f}<extra>structureless</extra>'});
  const plk = Object.keys(d.calib.pl).map(Number);
  tr.push({x: plk, y: plk.map(k => d.calib.pl[k]), xaxis: 'x2', yaxis: 'y2', mode: 'markers', marker: {symbol: 'star', size: 10, color: '#9a9a9a'}, name: 'fixture: planted min Υ̂ at its k (s ≥ 1)', hovertemplate: 'k = %{x}: %{y:.3g}<extra>planted</extra>'});
  tr.push({x: o.k, y: o.cap, xaxis: 'x2', yaxis: 'y2', mode: 'lines+markers', line: {color: '#16161a', dash: 'dash', width: 1.5}, marker: {size: 4}, name: 'cap 2μ_k/μ_{k−1}', hovertemplate: 'k = %{x}: cap %{y:.3g}<extra></extra>'});
  if (o.km) {
    tr.push({x: o.k, y: o.seba, xaxis: 'x2', yaxis: 'y2', mode: 'markers', marker: {size: 6, symbol: 'triangle-up-open', color: '#2a78d6'}, name: 'Υ̂ (SEBA rounding)', hovertemplate: 'k = %{x}: %{y:.3g}<extra>SEBA</extra>'});
    tr.push({x: o.k, y: o.km, xaxis: 'x2', yaxis: 'y2', mode: 'markers', marker: {size: 6, symbol: 'triangle-down-open', color: '#eb6834'}, name: 'Υ̂ (k-means rounding)', hovertemplate: 'k = %{x}: %{y:.3g}<extra>k-means</extra>'});
  }
  tr.push({x: o.k, y: o.ups, xaxis: 'x2', yaxis: 'y2', mode: 'lines+markers', line: {color: '#16161a', width: 2}, name: 'Υ̂(k), kept', text: o.out,
    marker: {size: 11, color: o.out.map(x => ok[x]), symbol: o.out.map(x => sym[x]), line: {width: 2, color: o.out.map(x => ok[x])}},
    hovertemplate: 'k = %{x}: Υ̂ = %{y:.3g} (%{text})<extra></extra>'});
  for (const oc of ['certified', 'undetermined', 'excluded']) tr.push(Object.assign(legendOnly(oc, ok[oc], 'solid', 'markers'), {xaxis: 'x2', yaxis: 'y2', marker: {color: ok[oc], symbol: sym[oc], size: 10}}));
  const all = o.ups.concat(o.cap, Object.values(d.calib.pl)).filter(v => v != null && v > 0);
  htmlLegend(el, [{head: 'left:'}, {name: o.label, color: '#16161a'}, {name: 'other objects', color: '#9fb6cf', width: 1.2}, {name: 'fixture, structureless', color: '#bdbdbd', dash: 'dot', width: 1.2},
    {head: 'right:'}, {name: 'Υ̂(k), kept', color: '#16161a'}, {name: 'cap 2μₖ/μₖ₋₁', color: '#16161a', dash: 'dash', width: 1.5}, {name: 'θ', color: '#1b7a3e', dash: 'dashdot', width: 1.2},
    ...(o.km ? [{name: 'SEBA rounding', color: '#2a78d6', glyph: 'triangle-up-open'}, {name: 'k-means rounding', color: '#eb6834', glyph: 'triangle-down-open'}] : []),
    {name: 'certified', color: ok.certified, glyph: 'circle'}, {name: 'undetermined', color: ok.undetermined, glyph: 'diamond-open'}, {name: 'excluded', color: ok.excluded, glyph: 'x'},
    {name: 'fixture structureless max', color: '#bbb', glyph: 'square'}, {name: 'fixture planted min at its k', color: '#9a9a9a', glyph: 'star'}]);
  const ymax = Math.max(th * 3, Math.min(1e5, Math.max(...all) * 1.5)), ymin = Math.min(0.5, Math.min(...all) * 0.8);
  const ax = t => merge(AXIS, {title: {text: t}});
  Plotly.react(el, tr, LAYOUT(merge(P.lay, {height: P.height, margin: {t: 30, l: 56, r: 12, b: 50}, showlegend: false,
    xaxis: merge(P.lay.xaxis, merge(ax('k'), {range: [1.5, 33]})), yaxis: merge(P.lay.yaxis, merge(ax('gap ratio μ_k / μ_{k−1}'), {type: 'log'})),
    xaxis2: merge(P.lay.xaxis2, merge(ax('k (log)'), {type: 'log', tickvals: d.R, ticktext: d.R.map(String)})),
    yaxis2: merge(P.lay.yaxis2, merge(ax('Υ̂(k), cap (log)'), {type: 'log', range: [Math.log10(ymin), Math.log10(ymax)]})),
    shapes: th == null ? [] : [{type: 'line', xref: 'x2 domain', x0: 0, x1: 1, yref: 'y2', y0: th, y1: th, line: {color: '#1b7a3e', width: 1.2, dash: 'dashdot'}}],
    annotations: [P.title(0, 'gap ratios (all objects; selected in black)'), P.title(1, o.label + ': multi-way ratio and cap'),
      ...(th == null ? [] : [{xref: 'x2 domain', x: 1, y: Math.log10(th), yref: 'y2', text: 'θ = ' + th, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: {color: '#1b7a3e', size: 11}}])]})), CONFIG);
}"""

JS_F4 = r"""(el, d, c) => {
  const objs = d.sets[c.f4set], tr = [], stat = c.f4stat;
  for (const g of d.groups) {
    const y = objs.map(o => { const r = d.vals[o][g]; return r ? (stat === 'eta' ? r.eta : r[stat]) : null; });
    if (y.every(v => v == null)) continue;
    const extra = objs.map(o => { const r = d.vals[o][g]; return stat === 'eta' && r && r.eta != null ? `chance ${f3(r.c)}, band top ${f3(r.hi)}` : ''; });
    tr.push({type: 'bar', x: objs.map(o => d.labels[o]), y, name: GLABEL[g], text: extra,
      marker: {color: GSTYLE[g][0], opacity: {solid: 1, dash: 0.45, dot: 0.7}[GSTYLE[g][1]], line: {color: GSTYLE[g][0], width: 1}},
      hovertemplate: GLABEL[g] + '<br>%{x}: %{y:.3f} %{text}<extra></extra>'});
  }
  const ylab = stat === 'eta' ? 'η², λ-weighted over ψ₁..ψ₁₆' : 'AMI (hard SEBA sets, r = ' + stat.slice(1) + ')';
  htmlLegend(el, d.groups.map(g => ({name: GLABEL[g], color: GSTYLE[g][0], glyph: 'square', opacity: {solid: 1, dash: 0.45, dot: 0.7}[GSTYLE[g][1]]})));
  Plotly.react(el, tr, LAYOUT({height: el.clientWidth >= 720 ? 430 : 520, barmode: 'group', bargap: 0.22, showlegend: false,
    yaxis: merge(AXIS, {title: {text: ylab}, range: [0, 1.02]}), xaxis: merge(AXIS, el.clientWidth >= 720 ? {tickangle: -30} : {tickangle: -75, tickfont: {size: 9.5}}),
    margin: {t: 16, l: 60, r: 10, b: 110}}), CONFIG);
}"""

JS_F5 = r"""(el, d, c) => {
  const P = PANELS(el, 2, 400), tr = [], xs = d.lags.map(String);
  const col = v => v.startsWith('whole') ? '#16161a' : (v.startsWith('point') ? ['#86b6ef', '#2a78d6', '#104281'] : ['#f2b07f', '#eb6834', '#9c3d0e'])[['L8', 'L16', 'L24'].indexOf(v.split('_')[1])] || '#888';
  for (const v of d.views) { const r = d.ret[v], sel = v === c.f5view;
    if (sel) { tr.push({x: xs, y: r.hi, mode: 'lines', line: {width: 0}, showlegend: false, hoverinfo: 'skip'});
      tr.push({x: xs, y: r.lo, mode: 'lines', line: {width: 0}, fill: 'tonexty', fillcolor: 'rgba(120,120,120,0.18)', showlegend: false, hoverinfo: 'skip'}); }
    tr.push({x: xs, y: r.mean, mode: 'lines+markers', name: d.labels[v], line: {color: col(v), width: sel ? 3 : 1.6, dash: v.startsWith('state') ? 'dash' : 'solid'}, marker: {size: sel ? 7 : 4},
      hovertemplate: d.labels[v] + '<br>τ = %{x}: mean retention %{y:.3f}<extra></extra>'}); }
  for (const g of ['T', 'Cc', 'U', 'W', 'Ttau']) { const y = d.sh[c.f5view][g];
    tr.push({x: xs, y, xaxis: 'x2', yaxis: 'y2', mode: 'lines+markers', name: GLABEL[g], line: {color: GSTYLE[g][0], dash: GSTYLE[g][1], width: 2}, marker: {size: 5},
      hovertemplate: GLABEL[g] + '<br>τ = %{x}: %{y:.3f}<extra>' + d.labels[c.f5view] + '</extra>'}); }
  htmlLegend(el, [{head: 'left:'}, ...d.views.map(v => ({name: d.labels[v], color: col(v), dash: v.startsWith('state') ? 'dash' : 'solid', width: v === c.f5view ? 3 : 1.6})),
    {name: 'min to max over the 8 vectors (selected view)', color: 'rgba(120,120,120,0.35)', glyph: 'square'}, {name: 'chance', color: '#999', dash: 'dot', width: 1},
    {head: 'right:'}, ...['T', 'Cc', 'U', 'W', 'Ttau'].map(gItem)]);
  const ax = t => merge(AXIS, {title: {text: t}});
  Plotly.react(el, tr, LAYOUT(merge(P.lay, {height: P.height, margin: {t: 30, l: 56, r: 12, b: 50}, showlegend: false,
    xaxis: merge(P.lay.xaxis, merge(ax('lag τ (tokens read after t)'), {type: 'category'})), yaxis: merge(P.lay.yaxis, merge(ax('retention f·P_c f / f·f'), {range: [-0.1, 1.05]})),
    xaxis2: merge(P.lay.xaxis2, merge(ax('lag τ'), {type: 'category'})), yaxis2: merge(P.lay.yaxis2, merge(ax('excess share'), {range: [-0.05, 1.02]})),
    shapes: [{type: 'line', xref: 'x domain', x0: 0, x1: 1, yref: 'y', y0: d.chance, y1: d.chance, line: {color: '#999', dash: 'dot', width: 1}}],
    annotations: [P.title(0, 'retention of lag-0 r = 8 set vectors'), P.title(1, 'shares along reading: ' + d.labels[c.f5view])]})), CONFIG);
}"""

JS_F6 = r"""(el, d, c) => {
  const T = d[c.tree], th = T.threshold, N = T.nodes;
  const lab = N.map(nd => { const top = (nd.tt || []).slice(0, 3).map(([t]) => esc(visTok(TOKS[t]))).join(' ');
    return `n = ${nd.n}` + (nd.children.length ? `<br>Υ = ${g3(nd.ups)}` : '') + `<br>${top}`; });
  const colr = N.map(nd => nd.children.length ? Math.log10(Math.max(0.1, nd.ups)) : null);
  const tr = [{type: 'icicle', ids: N.map(nd => String(nd.id)), labels: N.map(nd => 'node ' + nd.id), parents: N.map(nd => nd.parent < 0 ? '' : String(nd.parent)), values: N.map(nd => nd.n), branchvalues: 'total',
    text: lab, textinfo: 'text', hovertemplate: 'node %{id}<br>%{text}<extra></extra>', tiling: {orientation: 'v'}, maxdepth: 4, sort: false,
    marker: {colors: N.map(nd => nd.children.length ? (nd.ups >= th ? '#1b7a3e' : '#9ec5f4') : '#eeeeea'), line: {color: 'white', width: 1.5}},
    pathbar: {visible: false}, insidetextfont: {size: 11}}];
  Plotly.react(el, tr, LAYOUT({height: 380, margin: {t: 10, l: 4, r: 4, b: 4}}), CONFIG);
  const show = id => { const nd = N.find(x => String(x.id) === String(id)); if (!nd) return; const box = document.getElementById('f6-node');
    box.innerHTML = `<div class="itemcard tex2jax_ignore mathjax_ignore"><b>${esc(c.tree)} node ${nd.id}</b> · depth ${nd.depth} · n = ${nd.n} · ${nd.snip} snippets` +
      (nd.children.length ? ` · v1 two-way Υ = ${g3(nd.ups)} (${nd.ups >= th ? 'at or above' : 'below'} ${th}) · seba(r = 2) vs cut ${f2(nd.seba2)}` : ' · leaf') +
      row('μ₁..μ₈', (nd.mu || []).map(f3).join(', ')) + row('gap ratios', (nd.gap || []).map(f2).join(', ')) +
      row('token at t', chips(nd.tt)) + row('token at t+1', chips(nd.tn)) +
      row('class at t', bar(nd.ct || [], CLASS_PAL, DATA.ex.classes)) + row('class at t+1', bar(nd.cn || [], CLASS_PAL, DATA.ex.classes)) +
      row('source', bar(nd.src || [], SOURCE_PAL, DATA.ex.sources)) + key(DATA.ex.classes, CLASS_PAL) + key(DATA.ex.sources, SOURCE_PAL) + '</div>'; };
  if (!el._bound) { el._bound = true; el.on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p) { el._sel = p.id; show(p.id); } }); }
  if (el._tree !== c.tree) { el._tree = c.tree; el._sel = N[0].id; }
  show(el._sel);
}"""

JS_F8 = r"""(el, d, c) => {
  const P = PANELS(el, 2, 400, [1.25, 1]), tr = [], o = c.f8obj, ck = d.ck;
  const Q = ['T', 'Nx', 'A', 'kappa', 'kappaT', 'U', 'W', 'Cc', 'Cn'], G = ['T', 'Nx', 'U', 'A', 'W', 'Cc', 'Cn'];
  for (const k of ck) { const init = k !== 'trained';
    tr.push({type: 'bar', x: Q.map(q => GSHORT[q]), y: d.sh[o][k], name: k, marker: {color: Q.map(q => GSTYLE[q][0]), opacity: init ? 0.35 : 1, line: {color: '#333', width: init ? 0 : 0.5}},
      hovertemplate: '%{x}: %{y:.3f}<extra>' + k + '</extra>', showlegend: false});
    tr.push({type: 'bar', x: G.map(g => GSHORT[g]), y: d.ami[o][k], xaxis: 'x2', yaxis: 'y2', name: k, marker: {color: G.map(g => GSTYLE[g][0]), opacity: init ? 0.35 : 1, line: {color: '#333', width: init ? 0 : 0.5}},
      hovertemplate: '%{x}: AMI %{y:.3f}<extra>' + k + '</extra>', showlegend: false}); }
  tr.push(Object.assign(legendOnly('init (faint, left of each pair)', 'rgba(80,80,80,0.35)', 'solid', 'markers'), {marker: {symbol: 'square', size: 12, color: 'rgba(80,80,80,0.35)'}}));
  tr.push(Object.assign(legendOnly('trained (solid, right)', '#555', 'solid', 'markers'), {marker: {symbol: 'square', size: 12, color: '#555'}}));
  htmlLegend(el, [{name: 'init (faint, left bar of each pair)', color: '#555', glyph: 'square', opacity: 0.35}, {name: 'trained (solid, right bar)', color: '#555', glyph: 'square'}, {head: 'hue as in F2'}]);
  const ax = t => merge(AXIS, {title: {text: t}});
  Plotly.react(el, tr, LAYOUT(merge(P.lay, {height: P.height, barmode: 'group', margin: {t: 30, l: 56, r: 12, b: 50}, showlegend: false,
    xaxis: merge(P.lay.xaxis, AXIS), yaxis: merge(P.lay.yaxis, merge(ax('excess share'), {range: [-0.05, 1.02]})),
    xaxis2: merge(P.lay.xaxis2, AXIS), yaxis2: merge(P.lay.yaxis2, merge(ax('AMI, r = 8 sets'), {range: [0, 1.02]})),
    annotations: [P.title(0, d.labels[o] + ': shares'), P.title(1, d.labels[o] + ': AMI of r = 8 sets')]})), CONFIG);
}"""

CSS = r"""
:root { --ink:#16161a; --accent:#2a78d6; }
.mockup-note { border:1px solid #e0c9ae; background:#f7ecdf; color:#8a4a2a; border-radius:8px; padding:8px 12px; margin:4px 0 8px; font-size:14px; }
.banner figure.arch { width:100%; }
.archscroll { overflow-x:auto; }
details.archd > summary { font-size:12.5px; margin:2px 0; }
.fig { overflow-x:auto; }
.seg { flex-wrap:wrap; }
section.tab mjx-container[jax="CHTML"]:not([display="true"]) { display:inline-block; max-width:100%; overflow-x:auto; overflow-y:hidden; vertical-align:middle; }
section.tab code { overflow-wrap:anywhere; }
.banner figure.arch figcaption { font-size:12.5px; max-width:none; }
.answer { border-left:4px solid #c2410c; background:#fff8f3; border-radius:8px; padding:12px 18px; margin:10px 0 16px; }
.answer h3 { margin-top:4px; }
.diag { border-left:4px solid #d3cdc3; padding:2px 16px; margin:10px 0 16px; }
.legend-small { font-size:13px; color:#4a463f; }
.placeholder { border:2px dashed #c08a3e; background:#fdf8ef; border-radius:8px; padding:10px 16px; margin:12px 0; }
.placeholder::before { content:"PLACEHOLDER (MOCKUP): rewritten from the real data after the run"; display:block; font-size:11.5px; letter-spacing:.05em; font-weight:700; color:#8a5a1f; margin-bottom:4px; }
.fig .fq { font-size:13.5px; color:#2a4d6e; background:#eef3f8; border-radius:6px; padding:4px 10px; margin:2px 0 8px; }
code.tk { background:#f1efe9; border:1px solid #e2ddd5; padding:0 4px; border-radius:4px; font-size:12px; white-space:pre; }
.cnt { font-size:11px; color:#6b665e; margin:0 6px 0 2px; }
.ctx { font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; word-break:break-word; }
.ctx mark { background:#ffe08a; font-weight:700; padding:0 1px; border-radius:2px; }
.ctx .aft { color:#9a958c; }
.hint { color:#6b665e; font-size:13.5px; margin:4px 0 8px; max-width:none; }
.itemcard { border:1px solid #d3cdc3; border-radius:8px; padding:10px 12px; background:white; font-size:13.5px; }
.itemcard .meta { color:#4a463f; margin-top:6px; }
.val-tag { font-size:10.5px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; background:#f1efe9; border:1px solid #d3cdc3; border-radius:4px; padding:0 5px; color:#6b665e; }
.setgrid { display:grid; gap:12px; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); }
.setcard { border:1px solid #d3cdc3; border-radius:8px; padding:8px 10px; background:white; font-size:13px; }
.setcard.on { border-color:#16161a; box-shadow:0 0 0 2px #16161a33; }
.sethead { cursor:pointer; margin-bottom:6px; }
.sethead i, .key i { display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.row { display:flex; gap:8px; align-items:center; margin:3px 0; flex-wrap:wrap; }
.row .lab { flex:0 0 96px; color:#6b665e; font-size:12px; }
.row .rv { flex:1 1 0; min-width:0; display:flex; flex-wrap:wrap; gap:3px 2px; align-items:center; }
.chip { white-space:nowrap; }
table.doc td[class^=st-] { white-space:nowrap; }
.mkbar { display:flex; height:11px; flex:1 1 auto; width:100%; min-width:120px; border-radius:3px; overflow:hidden; background:#f1efe9; }
.mkbar span { display:block; height:100%; }
.key { display:flex; flex-wrap:wrap; gap:4px 12px; font-size:12px; color:#4a463f; margin:2px 0 6px; }
ol.exs { margin:6px 0 2px; padding-left:22px; }
ol.exs li { margin:2px 0; max-width:none; }
.etas { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 10px; }
.eta { border:1px solid #e2ddd5; border-radius:6px; padding:2px 8px; font-size:12.5px; background:white; }
.eta.above { border:2px solid #16161a; }
.hleg { display:flex; flex-wrap:wrap; gap:3px 14px; font-size:12.5px; color:#3a3732; margin:6px 0 2px; align-items:center; }
.hleg span { display:inline-flex; align-items:center; gap:5px; }
.hleg .gl { font-size:14px; line-height:1; }
.hleg b { font-weight:650; color:#6b665e; }
.cols2 { display:grid; gap:16px; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); }
table.doc td.st-met, table.doc td.st-pass { color:#2f6b45; font-weight:700; }
table.doc td.st-not-met, table.doc td.st-fail { color:#b04a4a; font-weight:700; }
table.doc td.st-not-run, table.doc td.st-not-determinable { color:#8a5a1f; font-weight:700; }
table.oc td { text-align:center; font-weight:650; }
table.oc td.C { background:#e3f2e8; color:#1b7a3e; } table.oc td.E { background:#f6e6e6; color:#b04a4a; } table.oc td.U { background:#f8eedd; color:#8a5a1f; }
table.oc td.lab, table.oc th.lab { text-align:left; font-weight:500; }
.tablewrap { overflow-x:auto; }
ol.devs li { margin:6px 0; font-size:14px; }
@media (max-width: 600px) { main { padding:16px 12px 60px; } .row .lab { flex-basis:100%; } header.hero h1 { font-size:24px; } }
"""


# ----------------------------------------------------------------------------------------------- tables
def verdict_html(d: Data, real: bool) -> tuple[str, str]:
    hy, ct = target_rows(d), control_rows(d)

    def tab(rows):
        trs = "".join(f"<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(t)}</td><td>{html.escape(l)}</td>"
                      f"<td>{html.escape(v)}</td><td class='st-{s.replace(' ', '-')}'>{s}</td></tr>" for k, t, l, v, s in rows)
        return ("<div class='tablewrap'><table class='doc'><tr><th>id</th><th>target (SPEC 7, written before the run)</th>"
                f"<th>label</th><th>value</th><th>status</th></tr>{trs}</table></div>")
    n_met = sum(r[4] == "met" for r in hy)
    n_ctl_ok = sum(r[4] == "pass" for r in ct)
    rerr = [x for x in gate_values(d, "trained")[0] if fin(x)]
    near = f" The largest Sinkhorn row error, {g3(max(rerr))}, passes the 2e-12 gate narrowly." if rerr and max(rerr) > 1e-12 else ""
    legend = ("<p class='legend-small'><b>Key.</b> κ: the walk's predictive coherence (walk-weighted similarity of the "
              "model's next-token distributions); κ̃: κ above chance; κ̃|T: above what sharing the token at t gives. "
              "T̃, Ũ, W̃, C̃_c: how much more often than chance the walk stays within the same token, snippet, source or "
              "token class (0 = chance, 1 = always). AMI: agreement between hard sets and a labelling (0 = chance, 1 = "
              "identical). τ: tokens read after t. R: the tested numbers of parts {2, 3, 4, 6, 8, 12, 16, 24, 32}. "
              "Whole cell: the whole-state snapshot. Retention: how much of a set vector the later walk keeps (1 = all, "
              "about 0 = chance).</p>")
    head = (f"<p><b>{n_met} of {len(hy)} hypotheses met</b> ({', '.join(r[0] for r in hy if r[4] == 'met') or 'none'}); "
            f"not met: {', '.join(r[0] for r in hy if r[4] == 'not met') or 'none'}. Controls and gate: {n_ctl_ok} of "
            f"{len(ct)} pass; not run: {', '.join(r[0] for r in ct if r[4] == 'not run') or 'none'}; failed: "
            f"{', '.join(r[0] for r in ct if r[4] == 'fail') or 'none'}.{near}</p>") + legend
    if not real:
        head = ("<p><b>MOCKUP: these statuses evaluate the synthetic fixture, not Mamba.</b> They show that the box "
                "reads <code>targets.json</code> and the controls correctly.</p>") + head
    state = "" if not real else ("pass" if n_met == len(hy) and n_ctl_ok == len(ct) else "")
    body = head + "<h4>Hypotheses</h4>" + tab(hy) + "<h4>Controls and pilot gate</h4>" + tab(ct) + (
        "<p class='hint'>Statuses are recomputed here from the values in <code>targets.json</code> and "
        "<code>validation.json</code> (a mismatch with the stored flag is printed by the build). K1 compares two AMIs "
        "and has no null; read it with K1b.</p>")
    return body, state


def outcome_table(d: Data) -> str:
    R = d.ranks
    rows = []
    for ck in d.ck:
        for o in F3_OBJS + [f"{v}_L{l}" for v in ("point", "state") for l in d.meta["single_layers"]]:
            rr = d.ups_rows(ck, o)
            if not rr:
                continue
            by = {r_["k"]: r_ for r_ in rr}
            cells = []
            for k in R:
                r_ = by.get(k)
                if not r_:
                    cells.append("<td></td>")
                    continue
                L = {"certified": "C", "excluded": "E", "undetermined": "U"}.get(r_["outcome"], "?")
                cells.append(f"<td class='{L}' title='Υ̂ = {g3(r_['ups'])}, cap = {g3(r_['cap'])}'>{L}</td>")
            rows.append(f"<tr><td class='lab'>{html.escape(olabel(o))}</td><td class='lab'>{ck}</td>{''.join(cells)}</tr>")
    head = "<tr><th class='lab'>object</th><th class='lab'>checkpoint</th>" + "".join(f"<th>k = {k}</th>" for k in R) + "</tr>"
    return (f"<div class='tablewrap'><table class='doc oc'>{head}{''.join(rows)}</table></div><p class='hint'>C = certified "
            f"(Υ̂ ≥ θ), E = excluded (cap below θ), U = undetermined. Hover a letter for Υ̂ and the cap. Lag snapshots "
            f"are in <code>upsilon.json</code> and not shown.</p>")


def compare_table(d: Data, kinds) -> str:
    rows = []
    for r_ in d.compare:
        if r_["kind"] not in kinds:
            continue
        a = olabel(r_["a"]) + ("" if r_.get("checkpoint_a") == "trained" else f" ({r_['checkpoint_a']})")
        b = olabel(r_["b"]) + ("" if r_.get("checkpoint_b") == "trained" else f" ({r_['checkpoint_b']})")
        rows.append([html.escape(a), html.escape(b), f2(r_.get("a8")), f2(r_.get("a16")), f2(r_.get("ami8")), f2(r_.get("ami16"))])
    return "<div class='tablewrap'>" + table_html(["a", "b", "a₈", "a₁₆", "AMI r = 8", "AMI r = 16"], rows, numeric={2, 3, 4, 5}) + "</div>"


# ----------------------------------------------------------------------------------------------- page
def fig_block(tab, nb, fid, title, question, data, draw, caption, mockup, height=None, controls=None):
    tab.figure(fid, title, data=data, draw=draw, subtitle="", caption=caption, controls=controls, height=height,
               mockup=mockup)
    if question:     # the "reader should be able to say" line under the title
        lab = "Reader should be able to say" if mockup else "What this figure shows"
        blk = tab.blocks[-1]
        tab.blocks[-1] = blk.replace("</div>", f"</div><div class='fq'>{lab}: {question}</div>", 1)


def fq_text(d: Data, facts: dict) -> dict:
    """the figures' reader sentences: SPEC 9's templates on fixtures, filled from the data on real builds."""
    tmpl = {"f2": "“up the stack the residual's neighbourhoods go from [..] to [..]; each layer's state is tied to [..]”",
            "f3": "“k-way structure is certified at k = .. / excluded at k = .. / undetermined”",
            "f4": "“the residual's sets are [..]; the state's sets are [..]”",
            "f5": "“the residual's sets last [..] tokens; the state's last [..]”",
            "f6": "“the first cuts separate [..]”", "f8": "“training adds [..]”"}
    if d.fixture:
        return tmpl
    F = facts
    return {
        "f2": (f"up the stack the residual's neighbourhoods move from the class of the token just read (C̃_c "
               f"{F['share.trained.point_L4.Cc']} at L4, {F['share.trained.point_L24.Cc']} at L24) towards source "
               f"(W̃ {F['share.trained.point_L4.W']} to {F['share.trained.point_L24.W']}); each layer's state is tied "
               f"mostly to source (W̃ {F['byl.trained.state.W.range']}), unevenly across layers."),
        "f3": (f"no object is certified at the tested numbers of parts; B-point is excluded at every k under θ = "
               f"{F['theta']}; B-state is undetermined at {F['undetermined.trained.Bstate']}, the whole state at "
               f"{F['undetermined.trained.whole']}."),
        "f4": (f"the residual's sets agree most with the class of the token just read (AMI "
               f"{F['ami8.trained.Bpoint.Cc']}); the state's sets agree most with source (whole state "
               f"{F['ami8.trained.whole.W']}), as do the bag reference's ({F['ami8.trained.bag.W']})."),
        "f5": (f"B-point's set vectors fall from {F['ret.point_L16.tau0']} to {F['ret.point_L16.tau2']} within two "
               f"tokens (point L16); the whole state's mean stays at {F['ret.whole.tau16']} after sixteen, with one "
               f"vector ({F['retv.whole.min_keep16']}) as the exception."),
        "f6": (f"T1's first cut separates layout and punctuation contexts from words; T2's separates "
               f"{F['tree.T2.child0.sources']} from {F['tree.T2.child1.sources']}."),
        "f8": (f"training creates the residual's token-class organisation and predictive coherence (κ̃ at L24 "
               f"{F['tgt.K6.kappa_L24_init']} → {F['tgt.K6.kappa_L24_trained']}) and multiplies the whole state's "
               f"source share (W̃ {F['tgt.K6.W_whole_init']} → {F['tgt.K6.W_whole_trained']})."),
    }


def build(d: Data, facts: dict, out: Path):
    real = not d.fixture
    mock = not real
    tk = Toks(d)
    ex = payload_explorer(d, tk)
    f6 = payload_f6(d, tk)
    title = "Coherent sets over token contexts in mamba-130m" + ("" if real else " (MOCKUP)")
    status = ("MEASURED · 1 model · 2 checkpoints" if real
              else f"MOCKUP · synthetic fixture {d.dir.name} | nothing here is a finding about Mamba")
    nb = Notebook(title, "A follow-up to the v1 Mamba coherent-sets study: what v1 was looking at, and what the "
                  "same construction finds when the items are token contexts.", status=status,
                  kicker="CoherentModes · explore/mamba_contexts (v2)",
                  footer=f"Built {facts['build_date']} from <code>{html.escape(str(d.dir.relative_to(ROOT.parent.parent)) if d.dir.is_relative_to(ROOT.parent.parent) else str(d.dir))}</code> "
                         f"(construction {facts['fingerprint']}) and the v1 diagnosis <code>diag_v1.json</code>. "
                         "Data embedded; figures draw in the browser (Plotly), maths by MathJax.")
    nb.banner(arch_banner(d), label="Model, items and views")
    nb.style(CSS)
    js = (JS_HELPERS.replace("__CLASS_PAL__", json.dumps(CLASS_PAL)).replace("__SOURCE_PAL__", json.dumps(SOURCE_PAL))
          .replace("__SET_PAL__", json.dumps(SET_PAL)).replace("__GSTYLE__", json.dumps(GSTYLE))
          .replace("__GLABEL__", json.dumps(GLABEL)).replace("__GSHORT__", json.dumps(GSHORT)))
    nb.script(js + JS_EXPLORER)
    nb.dataset("ex", ex)
    nb.dataset("f1d", payload_f1(d))

    ov = load_prose("overview", facts, real)
    fi = load_prose("findings", facts, real)
    me = load_prose("method", facts, real)
    caps = captions(d, facts)
    FQ = fq_text(d, facts)

    # ---------------------------------------------------------------- Overview
    t = nb.tab("overview", "Overview")
    t.prose(ov.pop("_head"))
    t.prose(f"<div class='diag'>{take(ov, 'answer', 'overview')}</div>")
    fig_block(t, nb, "f1ov", "v1's sets against token class and against source · mamba-130m · v1 probe, trained",
              "“v1 scored token-level sets against document-level labels”", "@f1d", JS_F1_OV, caps["f1ov"], False)
    t.prose(take(ov, "v2", "overview"))
    t.prose(take(ov, "found", "overview"))
    t.prose(take(ov, "map", "overview"))
    t.prose("<div class='tablewrap'>" + table_html(["section", "question", "readout", "status"], experiment_map(d, facts)) + "</div></details>")
    for k in list(ov):
        sys.exit(f"prose/overview.html: section '{k}' is not placed")

    # ---------------------------------------------------------------- Findings
    t = nb.tab("findings", "Findings")
    t.prose(fi.pop("_head"))
    body, state = verdict_html(d, real)
    t.verdict(body, state=state, label="Verdict against SPEC section 7" + ("" if real else " (MOCKUP)"))
    t.prose(take(fi, "findings", "findings"))
    for k in list(fi):
        sys.exit(f"prose/findings.html: section '{k}' is not placed")

    # ---------------------------------------------------------------- Explorer (F7)
    t = nb.tab("explorer", "Explorer")
    t.prose(take(me, "explorer_intro", "method"))
    objs = d.layout_objects
    t.controls([Select("obj", "object", objs, default="Bpoint" if "Bpoint" in objs else objs[0],
                       labels={o: olabel(o) for o in objs}),
                Select("col", "colour by", ["sets", "cls", "clsn", "src", "snip", "psi1", "psi2", "psi3", "psi4"],
                       default="sets", labels={"sets": "SEBA sets", "cls": "class at t (validation)",
                                               "clsn": "class at t+1 (validation)", "src": "source (validation)",
                                               "snip": "snippet", "psi1": "ψ₁", "psi2": "ψ₂", "psi3": "ψ₃", "psi4": "ψ₄"}),
                Radio("r", "SEBA rank r", [8, 16], default=8),
                Select("k", "eigenfunction ends", list(range(1, int(d.meta.get("n_eta", 16)) + 1)), default=1,
                       labels={k: f"ψ{k}" for k in range(1, 17)})])
    fig_block(t, nb, "ex_layout", "F7 · layout of the selected object's items (UMAP of ψ₁..ψ₁₆) · trained",
              "", "@ex", "drawLayout", caps["ex_layout"], mock)
    t.figure("ex_item", "Selected item", data=None, draw="drawItem")
    t.figure("ex_sets", "Set cards (hard SEBA sets of the selected object)", data=None, draw="drawSets",
             caption=caps["ex_sets"])
    t.figure("ex_ends", "Ends of the selected eigenfunction", data=None, draw="drawEnds", caption=caps["ex_ends"])

    # ---------------------------------------------------------------- Measurements
    t = nb.tab("measurements", "Measurements")
    t.prose(take(me, "measurements_intro", "method"))
    t.h("F1. What v1 scored, and what it did not")
    fig_block(t, nb, "f1", "F1 · v1's sets scored against source, token class and token · mamba-130m, trained, v1 probe "
              f"(N = 1,000 snippets)", "“v1 scored token-level sets against document-level labels”", "@f1d", JS_F1,
              caps["f1"], False)
    t.h("F2. Excess shares by layer: what the residual and the state keep together")
    fig_block(t, nb, "f2", "F2 · excess walk shares by layer · point and state views · trained (init faint)",
              FQ["f2"],
              payload_f2(d), JS_F2, caps["f2"], mock,
              controls=[Radio("f2q", "show", ["all", "input", "output", "validation"], default="all",
                              labels={"all": "all", "input": "label-free (T, Nx, U)", "output": "network output (A, κ̃, κ̃|T)",
                                      "validation": "validation (W, C_c, C_n)"})])
    t.h("F3. Is there clear multi-way structure?")
    f3d = payload_f3(d)
    fig_block(t, nb, "f3", "F3 · spectra, gap ratios and the multi-way ratio Υ̂(k) against θ · trained",
              FQ["f3"], f3d, JS_F3, caps["f3"], mock,
              controls=[Select("f3obj", "object", f3d["order"], default="Bpoint",
                               labels={k: v["label"] for k, v in f3d["objs"].items()})])
    t.details("Outcome at every k ∈ R, for every block and listed snapshot", outcome_table(d))
    t.h("F4. What the sets group: AMI of the sets, and η² of the modes")
    fig_block(t, nb, "f4", "F4 · sets and modes against each grouping, by object · trained",
              FQ["f4"], payload_f4(d), JS_F4, caps["f4"], mock,
              controls=[Radio("f4stat", "statistic", ["a8", "a16", "eta"], default="a8",
                              labels={"a8": "AMI, r = 8", "a16": "AMI, r = 16", "eta": "η² (modes)"}),
                        Radio("f4set", "objects", ["blocks", "point", "state"], default="blocks",
                              labels={"blocks": "blocks and references", "point": "point L4..L24", "state": "state L4..L24"})])
    t.details("Agreement between objects (span agreement a_r and AMI of the hard sets)",
              "<p class='hint'>a_r: span agreement of the leading r eigenfunctions (chance about r/N; "
              f"{facts['a_null']} at r = 8). Same-checkpoint pairs among B-point, B-state, whole, bag, pred, and the "
              "three depth thirds.</p>" + compare_table(d, ("objects", "thirds")))
    t.h("F5. Along reading: how long do the sets last?")
    f5 = payload_f5(d)
    fig_block(t, nb, "f5", "F5 · retention and shares against the lag τ · point, state (L8, L16, L24) and whole · trained",
              FQ["f5"], f5, JS_F5, caps["f5"], mock,
              controls=[Select("f5view", "view for the band and the right panel", f5["views"],
                               default="point_L16" if "point_L16" in f5["views"] else f5["views"][0], labels=f5["labels"])])
    t.h("F6. Zoom trees: what the first cuts separate")
    t.figure("f6", "F6 · zoom trees T1 (B-point) and T2 (whole state) · trained", data=f6, draw=JS_F6,
             caption=caps["f6"], mockup=mock, controls=[Radio("tree", "tree", list(f6), default=list(f6)[0],
                                                              labels={k: f"{k} ({v['block']})" for k, v in f6.items()})])
    t.blocks[-1] = t.blocks[-1].replace("</div>", f"</div><div class='fq'>{'Reader should be able to say' if mock else 'What this figure shows'}: {FQ['f6']}</div>", 1)
    t.prose("<div id='f6-node'></div>")
    t.h("F8. What training adds")
    f8 = payload_f8(d)
    fig_block(t, nb, "f8", "F8 · trained against init · F2's and F4's quantities", FQ["f8"], f8, JS_F8,
              caps["f8"], mock, controls=[Select("f8obj", "object", f8["objs"], default=f8["objs"][0], labels=f8["labels"])])
    t.details("Init against trained: agreement of the same object's sets", compare_table(d, ("init_trained",)))

    # ---------------------------------------------------------------- Method
    t = nb.tab("method", "Method")
    t.prose(take(me, "construction", "method"))
    t.h("The measured quantities", 3)
    t.equation(r"\tilde S_G = \frac{S_G - S^0_G}{1 - S^0_G},\qquad "
               r"S_G = \frac1N \sum_{i \ne j,\ g(i) = g(j)} P_{ij},\qquad "
               r"\tilde\kappa_{|T} = \frac{\kappa - \kappa^{T}}{1 - \kappa^{T}},\qquad "
               r"\kappa = \frac{\sum_{i,j\ \text{diff. snippet}} P_{ij}\,\mathrm{BC}(p_i, p_j)}{\sum_{i,j\ \text{diff. snippet}} P_{ij}}",
               where=("where $P$ is a snapshot's (or block's) walk on the $N$ items, $g$ a grouping of the items "
                      "(token at $t$, token at $t+1$, snippet, the model's top-1 prediction, source, token class), "
                      "$S^0_G$ the exact chance level, $\\mathrm{BC}$ the Bhattacharyya coefficient of the trained "
                      "model's next-token distributions $p_i, p_j$, and $\\kappa^T$ the mean of $\\kappa$ when the "
                      "prediction vectors are permuted within groups of items sharing the token at $t$ "
                      f"({facts['n_perm']} permutations). $\\tilde S = 0$ is chance, $1$ means the walk never leaves the group."))

    t.prose(take(me, "labels", "method"))
    t.prose(take(me, "controls", "method"))
    t.prose(take(me, "hazards", "method"))
    t.prose(take(me, "decisions", "method"))
    t.h("Deviations from SPEC.md (quoted from README.md)", 3)
    t.prose(readme_deviations())
    t.prose(take(me, "page", "method"))
    for k in [k for k in me if k != "_head"]:
        sys.exit(f"prose/method.html: section '{k}' is not placed")

    nb.dataset("toks", tk.strs)
    nb.write(out)
    mb = out.stat().st_size / 1e6
    if mb > 15:
        log(f"warning: page is {mb:.1f} MB (target under 15 MB)")


def experiment_map(d: Data, facts: dict):
    st = "measured" if not d.fixture else "MOCKUP (fixture)"
    return [
        ["Measurements · F1", "What did v1 score its sets against?", "AMI of v1's sets with source and token class",
         "measured (v1 data)"],
        ["Measurements · F2, F4; Explorer", "(a) What do the residual's depth sets group?",
         "excess shares, κ̃ and κ̃|T by layer; AMI and η² of B-point", st],
        ["Measurements · F2, F4; Explorer", "(b) What do the state's sets group (per layer, whole)?",
         "the same for B-state and the whole cell", st],
        ["Measurements · F5", "(c) How long does each kind of set stay together along reading?",
         "retention of lag-0 set vectors; shares against τ", st],
        ["Measurements · F3", "(d) Is there clear multi-way structure?", "Υ̂(k), cap 2μₖ/μₖ₋₁, outcome under θ", st],
        ["Measurements · F6", "What do the first cuts separate?", "zoom trees T1, T2 with node descriptors", st],
        ["Measurements · F8", "What does training add?", "F2's and F4's quantities at init and trained", st],
        ["Findings", "Verdict against the pre-registered targets", "K1..K6, controls, pilot gate", st],
    ]


def captions(d: Data, facts: dict) -> dict:
    c = {}
    c["f1ov"] = (f"<b>Plotted:</b> mean AMI (bars; whiskers min to max) between v1's hard SEBA sets at r = "
                 f"{payload_f1(d)['r'][-1]} and two labellings of its items: the class of the token just read (C_c) "
                 f"and the document source (W). Depth blocks: the residual at L4..L24 at one position "
                 f"({facts['v1.n_t']} positions); reading blocks: one layer and view along t = 4..128 "
                 f"({facts['v1.n_reading']} blocks). Labels are validation only. <b>Read:</b> v1 reported the reading "
                 "blocks against source; the depth blocks' sets follow the token class, which v1 never scored.")
    c["f1"] = ("<b>Plotted:</b> (a, b) AMI between v1's hard SEBA sets (assigned items only) and four labellings, "
               "against the SEBA rank r: source W, class of the token at t (C_c) and at t+1 (C_n), token id at t (T). "
               f"Points are means, whiskers min to max over the {facts['v1.n_t']} depth blocks (one per position) or the "
               f"{facts['v1.n_reading']} reading blocks (for C_c, C_n and T also over the positions t at which the "
               "class is read). (c) Every v1 depth-block set at r = 8: share of its majority source (x) against share "
               "of its majority current-token class (y); colour = that class, size ∝ set size; hover for the commonest "
               "tokens. <b>Read:</b> sets above the diagonal are organised by token class more than by source. "
               "Source: <code>diag_v1.json</code>; labels are validation only.")
    hw = payload_f2(d)["band_hw"]
    c["f2"] = ("<b>Plotted:</b> excess walk share $\\tilde S$ of each grouping (0 = chance, 1 = the walk never leaves "
               "the group) in each point snapshot (left) and state snapshot (middle) at L1..L24, and in the blocks, the "
               "whole cell and the three references (right); predictive coherence $\\tilde\\kappa$ and its part beyond "
               "the token just read, $\\tilde\\kappa_{|T}$, in green. Faint lines and small markers: init. Hue = what "
               "groups the items (blue tokens, green the model's predictions, orange the document, purple token "
               "classes); dashes = the t+1 or conditional variant. Permutation bands "
               f"({facts['n_perm']} permutations) are at most ±{f3(hw)} in excess units and are not drawn. W, C_c, "
               "C_n are validation; A, κ̃, κ̃|T use the trained model's outputs. <b>Read:</b> a quantity that rises "
               "with depth is what the later residual keeps together; compare a layer's state with its residual.")
    c["f3"] = ("<b>Plotted:</b> left, gap ratios $\\mu_k/\\mu_{k-1}$ of $\\bar P$'s smallest non-trivial values "
               "$\\mu_j = 1-\\lambda_j$, k = 2..32, for every object (selected in black) and the structureless "
               "fixture blocks (dotted). Right, for the selected object at $k \\in R$: the multi-way ratio "
               "$\\hat\\Upsilon(k) = \\mu_k/\\hat\\rho(k)$ from the SEBA and k-means roundings (the larger is kept, "
               "large marker coloured by outcome), the cap $2\\mu_k/\\mu_{k-1}$ that no partition can exceed, "
               f"θ = {facts['theta']} ({facts['theta_rule']}, from the fixture calibration), the structureless "
               "fixture's maximum (grey) and the planted fixture's minimum at its own k (stars). v1's roots T1 and T4 "
               "(from the diagnosis, k = 2..8, SEBA rounding only) are in the object list. <b>Read:</b> certified = "
               "$\\hat\\Upsilon \\ge \\theta$; excluded = cap below θ; otherwise undetermined, which is not evidence "
               "of absence.")
    c["f4"] = ("<b>Plotted:</b> for each object, adjusted mutual information between its hard SEBA sets at r = 8 or "
               "16 (assigned items only) and each grouping; or η², the between-group share of the variance of each "
               f"of ψ₁..ψ{facts['n_eta']}, averaged with weights λ_k (needs no clusters; hover for the chance level "
               "(G−1)/(N−1) and the top of the permutation band). Hue as in F2; lighter bars mark the t+1 grouping and "
               "the model's prediction. Lag blocks are scored only against T, C_c, U, W. W, C_c, C_n are validation. "
               "<b>Read:</b> compare each bar with its chance level first (AMI: 0; η²: hover, high for the token groupings), "
               "then across groupings: the bar furthest above chance says what an object's sets (or modes) separate.")
    c["f5"] = ("<b>Plotted:</b> left, retention $f^\\top P_c f / f^\\top f$ of the centred r = 8 SEBA vectors of "
               f"B-point (blue, point lag snapshots), B-state (orange dashed, state lag snapshots) and the whole cell "
               "(black), in the lag snapshot that reads the same items τ tokens later; lines are means over the 8 "
               f"vectors, the grey band the min to max for the selected view; dotted line: chance −1/(N−1) = "
               f"{facts['ret.chance']}. Right, excess shares in the selected view's lag snapshots of the lag-0 "
               "groupings (token, class, snippet, source at t) and of the token at t+τ. <b>Read:</b> a curve near 1 "
               "keeps the lag-0 sets together; near 0 they have dissolved.")
    c["f6"] = ("<b>Plotted:</b> the zoom trees (v1's procedure: sweep cut on ψ₁..ψ₃, rebuilt on each side; depth "
               "below 3, n ≥ 64). Tile width ∝ node size; text: size, v1's two-way Υ (a two-way statistic) and the "
               f"commonest tokens at t. Dark green: Υ ≥ {facts['tree.T1.threshold']}; light blue: below; grey: leaf. "
               "Click a tile for its descriptors (μ, gap ratios, tokens at t and t+1, class and source make-up; "
               "make-ups are validation).")
    c["f8"] = ("<b>Plotted:</b> for the selected object, F2's excess shares (left) and F4's AMI at r = 8 (right), at "
               "init (faint) and trained (solid), same hues as F2. Init uses the same items and groupings, including "
               "the trained model's predictions. <b>Read:</b> the difference between the pairs is what training adds "
               "for that object.")
    c["ex_layout"] = ("<b>Plotted:</b> each point is an item (a token context), placed by UMAP of the selected "
                      "object's ψ₁..ψ₁₆ (display only; distances in the picture are not the method's), coloured by the "
                      "selected labelling. Grey = unassigned by SEBA. Hover for the context (token at t in ⟦ ⟧, the "
                      "next tokens grey); click for the item card. Labels marked validation never enter the "
                      "construction.")
    c["ex_sets"] = ("Per set: size, distinct snippets, the 8 commonest tokens at t and at t+1 with counts, token "
                    "concentration, class make-up at t and t+1 and source make-up (bars; hover for counts), and the 12 "
                    "members with the largest SEBA values (the token at t highlighted, the 4 tokens after it grey). "
                    "Lag-object sets are described with the lag-0 tokens.")
    c["ex_ends"] = ("The 12 items with the smallest and the 12 with the largest value of ψ_k, the k-th non-constant "
                    "eigenfunction of the object's operator, and η²_k of ψ_k against each grouping.")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data directory, e.g. data/m130c or data/_fixture")
    ap.add_argument("--diag", default=str(DIAG_DEFAULT))
    ap.add_argument("--bench", default=str(BENCH_DEFAULT))
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ddir = Path(a.data)
    if not ddir.is_absolute() and not ddir.exists():
        ddir = ROOT / a.data
    d = Data(ddir.resolve(), Path(a.diag).resolve(), Path(a.bench).resolve())
    facts = make_facts(d)
    if a.facts:
        for k, v in sorted(facts.items()):
            print(f"{k:56s} {v}")
        return
    if a.out:
        out = Path(a.out)
    elif d.fixture:
        out = HERE / ("mamba_contexts_MOCKUP.html" if d.fixture == "planted" else f"mamba_contexts_{d.fixture}_MOCKUP.html")
    else:
        out = HERE / "mamba_contexts.html"
    if d.fixture and "MOCKUP" not in out.name:
        sys.exit("a page built from a fixture must carry MOCKUP in its file name")
    build(d, facts, out)


if __name__ == "__main__":
    main()
