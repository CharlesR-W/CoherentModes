"""Build an explorer site from schema-v3 measurement data (../SPEC.md section 6). A generalised copy of
../../cifar_coherent/site/build_site.py (which stays untouched): K eigenfunctions and SEBA ranks from meta, byte SEBA
labels (r up to 255), SEBA amplitudes from float64 (the browser never reconstructs SEBA), per-mode stability.

    uv run --with numpy --with pillow python site/build_site.py <tag> [--clean]
    uv run --with numpy --with pillow python site/build_site.py <tag> --facts      # list {{fact:NAME}} values
    python3 -m http.server 8765 --bind 127.0.0.1 -d viewer/<tag>                   # then open http://127.0.0.1:8765/

Reads data/<tag>/ and site/ sources; writes viewer/<tag>/ (.nojekyll). Content fragments
site/content/<tag>/{overview,findings,measurements,method}.html, header.json and tours.json are read when present.

Data written to viewer/site/d/ (all little-endian; see DATA_LAYOUT below and js/data.js):
  meta.json            steps, sites, modes, kappa, nn_mass, all block spectra, layouts of the binary chunks, ...
  umap_<seed>.bin      u16 (T, S, N, 2): per-cell layout scaled to [0, 65535] (0.5-99.5 percentile box)
  points.bin           per-image diagnostics, f32 (T, N) per field (fields listed in meta.points_fields)
  labels.bin           i8 (N) CIFAR labels; sprite.png: the 2,000 probe images, 50 per row
  blk/<fam>/<idx>/<mode>/<pair>.bin.gz   one block, gzip-compressed; layout in meta.block_layout (see block_layout())
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import html
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent          # site/
ROOT = HERE.parent                              # explore/cifar_more/
OUT = ROOT / "viewer" / "site"                  # set per tag in main(): viewer/<tag>
CONTENT = HERE / "content"                      # set per tag in main(): site/content/<tag>
BLOCK_FORMAT = "blk-v3m-1"     # bump when the block chunk layout or its derivation changes
REP_PSI, REP_TOP, REP_BND = 12, 24, 12      # representative images stored per function
DEFAULT_TITLE = ("Towards Characterizing a Neural Network's Native Ontology: Coherent Sets of Data Through Layers "
                 "and Over Training")
DEFAULT_SUBTITLE = ("or, Sparse Eigenbases of the Tempered Cross-Hankel Operator for Spectral Identification of the "
                    "Kantian Categories by Neural Coherent Sets")
DEFAULT_SHORT_TITLE = "Coherent sets"
TOUR_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
POINT_FIELDS = ["loss"]   # f32; plus pred/correct


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------- SEBA (numpy)
def seba_vectors(v: np.ndarray, rot: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """SPEC section 2 recomputation; mirrors coherent.seba_reconstruct (clip=True). v: (N, r) incl. psi_0."""
    z = v @ rot
    mu = 0.99 / np.sqrt(len(v))
    s = np.sign(z) * np.maximum(np.abs(z) - mu, 0)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0)); sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return np.maximum(s[:, perm], 0)


def seba_hard(s: np.ndarray) -> np.ndarray:
    a = s.argmax(1).astype(np.int8)
    a[s.max(1) <= 0] = -1
    return a


# ----------------------------------------------------------------------------------------------- colours
def _lab(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c > .04045, ((c + .055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]])
    xyz /= np.array([.95047, 1., 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def extend_palette(base: list[str], n: int) -> list[str]:
    """Keep `base`, then add colours greedily farthest (CIELAB) from all chosen ones; mid lightness, no greys.
    Deterministic. Used for SEBA sets beyond the reference 8 and for 20 superclasses."""
    if len(base) >= n:
        return base[:n]
    g = np.arange(0, 256, 15)
    cand = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lab = _lab(cand)
    ok = (lab[:, 0] > 38) & (lab[:, 0] < 80) & (np.hypot(lab[:, 1], lab[:, 2]) > 32)
    cand, lab = cand[ok], lab[ok]
    chosen = [_lab([int(h[i:i + 2], 16) for i in (1, 3, 5)]) for h in base]
    d = np.min([np.linalg.norm(lab - c, axis=1) for c in chosen], axis=0)
    out = list(base)
    while len(out) < n:
        i = int(np.argmax(d))
        out.append("#%02x%02x%02x" % tuple(int(v) for v in cand[i]))
        d = np.minimum(d, np.linalg.norm(lab - lab[i], axis=1))
    return out


def shade(hexc: str, t: float) -> str:
    """t in [-1, 1]: mix toward black (t < 0) or white (t > 0)."""
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = c + (255 - c) * t if t > 0 else c * (1 + t)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


# ----------------------------------------------------------------------------------------------- helpers
def procrustes_chain(emb: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """emb (T, S, N, 2). Centre and scale each cell; per site rotate/reflect each checkpoint onto the previous one.
    With ref (the seed-0 result), instead rotate/reflect every cell onto the matching ref cell. Display only."""
    out = emb.astype(np.float64).copy()
    out -= out.mean(2, keepdims=True)
    out /= np.sqrt((out ** 2).sum((2, 3), keepdims=True) / out.shape[2]) + 1e-12
    T, S = out.shape[:2]
    for s in range(S):
        for t in range(T):
            target = ref[t, s] if ref is not None else (out[t - 1, s] if t > 0 else None)
            if target is None:
                continue
            u, _, vt = np.linalg.svd(out[t, s].T @ target)
            out[t, s] = out[t, s] @ (u @ vt)
    return out


def quantise_layout(emb: np.ndarray) -> np.ndarray:
    lo = np.percentile(emb, 0.5, axis=2, keepdims=True)
    hi = np.percentile(emb, 99.5, axis=2, keepdims=True)
    # keep the aspect ratio of each cell: one scale for both axes
    span = (hi - lo).max(-1, keepdims=True)
    mid = (hi + lo) / 2
    q = np.clip((emb - mid) / (span + 1e-12) + 0.5, 0, 1)
    return np.round(q * 65535).astype("<u2")


def pack_nibbles(lab: np.ndarray) -> np.ndarray:
    """lab int8 (..., N) in -1..14 -> uint8 (..., ceil(N/2)); value+1 stored, low nibble = even index."""
    v = (lab.astype(np.int16) + 1).astype(np.uint8)
    if v.shape[-1] % 2:
        v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,), np.uint8)], -1)
    return (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8)


def pad(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.full(k, 65535, np.int64)
    out[:min(k, len(idx))] = idx[:k]
    return out


def write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


# ----------------------------------------------------------------------------------------------- build
class Builder:
    def __init__(self, tag: str, layout_tag: str | None, default_mode: int | None, skip_blocks: bool, clean: bool):
        self.tag, self.layout_tag, self.skip_blocks, self.clean = tag, layout_tag, skip_blocks, clean
        self.src = ROOT / "data" / tag
        if not (self.src / "meta.json").exists():
            sys.exit(f"no {self.src}/meta.json")
        self.meta = json.loads((self.src / "meta.json").read_text())
        assert self.meta.get("schema_version") == 3, "this build_site.py expects schema_version 3"
        m = self.meta
        self.steps, self.sites, self.modes = m["steps"], m["sites"], m["modes"]
        self.T, self.S, self.M, self.n = len(self.steps), len(self.sites), len(self.modes), m["n"]
        self.seba_r = m["seba_r"]
        self.R = len(self.seba_r)
        self.K = m["n_vecs"]
        self.G = m["stability"]["guard"]
        self.n_fun = self.K + sum(self.seba_r)
        self.default_mode = default_mode if default_mode is not None else (
            m["default_mode"] if isinstance(m.get("default_mode"), int) else
            self.modes.index("half-mass (calibrated)") if "half-mass (calibrated)" in self.modes else 0)
        self.facts: dict[str, str] = {}
        self.d = OUT / "d"

    # -- layout of one block chunk (byte offsets); X = number of cells along the retention axis
    def block_layout(self, X: int):
        n, R, K = self.n, self.R, self.K
        SR = sum(self.seba_r)
        parts = [("scale", "f32", K), ("vals", "f32", K + 1), ("pn", "f32", K), ("span", "f32", K + self.G),
                 ("local", "f32", K), ("ret_psi", "f32", K * X), ("ret_seba", "f32", SR * X),
                 ("reps", "u16", K * 3 * REP_PSI + SR * (REP_TOP + REP_BND)),
                 ("vecs", "i8", K * n), ("labels", "u8", R * n), ("sval", "u8", R * n)]
        size = {"f32": 4, "u16": 2, "i16": 2, "i8": 1, "u8": 1}
        out, off = {}, 0
        for name, dt_, cnt in parts:
            out[name] = {"offset": off, "dtype": dt_, "count": cnt}
            off += size[dt_] * cnt
        return out, off

    def run(self):
        t0 = time.time()
        if OUT.exists():
            # block chunks are rebuilt incrementally (stamp per family/anchor); everything else is rewritten
            for p in OUT.iterdir():
                if p.name == "d" and not self.clean:
                    for q in p.iterdir():
                        if q.name not in ("blk", "meta.json"):
                            shutil.rmtree(q) if q.is_dir() else q.unlink()
                    continue
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        OUT.mkdir(parents=True, exist_ok=True)
        self.copy_static()
        self.probe()
        self.layouts()
        self.points()
        if not self.skip_blocks:
            self.blocks()
        self.small_json()
        self.write_meta()
        self.assemble_html()
        (OUT / ".nojekyll").write_text("")
        tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
        nfiles = sum(1 for p in OUT.rglob("*") if p.is_file())
        log(f"wrote {OUT}: {tot / 1e6:.1f} MB in {nfiles} files ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ static files
    def copy_static(self):
        for name in ["css", "js", "vendor", "assets"]:
            if (HERE / name).exists():
                shutil.copytree(HERE / name, OUT / name, dirs_exist_ok=True)
        pal = json.loads((HERE / "palette.json").read_text())
        pal["sets"] = extend_palette(pal["sets"], max(8, max(self.seba_r)))
        pal["classes"] = extend_palette(pal["classes"], max(10, len(self.meta["classes"])))
        self.palette = pal
        lines = [":root {"]
        for i, c in enumerate(pal["classes"]):
            lines.append(f"  --c-class-{i}: {c};")
        for i, c in enumerate(pal["sets"]):
            lines.append(f"  --c-set-{i}: {c};")
        for k in ["neg", "mid", "pos"]:
            lines.append(f"  --c-{k}: {pal['diverging'][k]};")
        for i, c in enumerate(pal["sequential"]):
            lines.append(f"  --c-seq-{i}: {c};")
        for i, c in enumerate(pal["ordinal"]):
            lines.append(f"  --c-ord-{i}: {c};")
        for k in ["unassigned", "accent", "bg", "surface", "fg", "mute", "line"]:
            lines.append(f"  --{k}: {pal[k]};")
        lines.append("}")
        (OUT / "css").mkdir(exist_ok=True)
        (OUT / "css" / "tokens.css").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ probe images
    def probe(self):
        p = np.load(self.src / "probe.npz")
        self.labels = p["labels"].astype(np.int8)
        write(self.d / "labels.bin", self.labels.tobytes())
        self.fine_coarse = None
        if "fine" in p.files:
            fine = p["fine"].astype(np.uint8)
            write(self.d / "fine.bin", fine.tobytes())
            fc = {}
            for f_, c_ in zip(fine.tolist(), self.labels.tolist()):
                assert fc.setdefault(f_, c_) == c_, "a fine class maps to two superclasses"
            self.fine_coarse = [fc[i] for i in range(len(self.meta["fine_classes"]))]
        imgs = p["images"]
        cols = 50
        rows = int(np.ceil(self.n / cols))
        sheet = np.zeros((rows * 32, cols * 32, 3), np.uint8)
        for i in range(self.n):
            r, c = divmod(i, cols)
            sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = imgs[i]
        buf = io.BytesIO()
        Image.fromarray(sheet).save(buf, format="WEBP", quality=92, method=4)   # ~1 MB for 2,000 CIFAR images
        write(self.d / "sprite.webp", buf.getvalue())
        self.sprite_cols = cols
        log(f"sprite {len(buf.getvalue()) / 1e6:.2f} MB")

    # ------------------------------------------------------------------ UMAP layouts
    def layouts(self):
        emb = np.stack([np.load(self.src / f"umap_{s}.npy") for s in self.steps])  # T,S,N,2
        e0 = procrustes_chain(emb)
        write(self.d / "umap_0.bin", quantise_layout(e0).tobytes())
        self.layout_info = [{"seed": 0, "tag": self.tag}]
        if self.layout_tag:
            src1 = Path(self.layout_tag) if Path(self.layout_tag).is_absolute() else ROOT / "data" / self.layout_tag
            files = [src1 / f"umap_{s}.npy" for s in self.steps]
            if all(f.exists() for f in files):
                e1 = procrustes_chain(np.stack([np.load(f) for f in files]), ref=e0)
                write(self.d / "umap_1.bin", quantise_layout(e1).tobytes())
                info = {"seed": 1, "tag": self.layout_tag}
                mpath = src1 / "meta.json"
                if mpath.exists():
                    m1 = json.loads(mpath.read_text())
                    for key in ("val_acc", "val_accuracy", "acc"):
                        if key in m1:
                            info["val_acc"] = m1[key]
                self.layout_info.append(info)
                log(f"layout from {self.layout_tag} added")
            else:
                log(f"warning: --layout-tag {self.layout_tag}: missing umap files, seed-1 layout skipped")

    # ------------------------------------------------------------------ per-image diagnostics
    def points(self):
        T, n = self.T, self.n
        fields = {}
        pred = np.zeros((T, n), np.int8)
        correct = np.zeros((T, n), np.uint8)
        present = None
        for t, step in enumerate(self.steps):
            path = self.src / f"points_{step}.npz"
            if not path.exists():
                log(f"warning: {path.name} missing")
                continue
            with np.load(path) as p:
                keys = set(p.files)
                have = [f for f in POINT_FIELDS if f in keys]
                present = have if present is None else [f for f in present if f in have]
                for f in have:
                    fields.setdefault(f, np.full((T, n), np.nan, np.float32))[t] = p[f]
                pred[t] = p["pred"]
                correct[t] = p["correct"]
        present = present or []
        self.points_fields = present
        self.points_stamp = "|".join(f"{(self.src / f'points_{s}.npz').stat().st_mtime_ns}" for s in self.steps if (self.src / f'points_{s}.npz').exists())
        blob = b"".join(f32(fields[f]).tobytes() for f in present) + pred.tobytes() + correct.tobytes()
        write(self.d / "points.bin", blob)
        log(f"points fields: {present}")

    # ------------------------------------------------------------------ blocks
    def blocks(self):
        n, R, K = self.n, self.R, self.K
        self.vals_all = {"layer": [], "time": []}
        self.hz_all = {"layer": [], "time": []}
        fams = [("layer", self.steps, self.meta["layer_pairs"], self.S), ("time", self.sites, self.meta["time_pairs"], self.T)]
        worst_agree, worst_q8 = 1.0, 0.0
        self.layouts_blk = {}
        total_bytes = raw_bytes = 0
        hz = lambda span: (lambda ok: int(ok[-1]) + 1 if len(ok) else 0)(np.flatnonzero(span >= self.meta["stability"]["threshold"]))
        for fam, names, pairs, X in fams:
            lay, size = self.block_layout(X)
            self.layouts_blk[fam] = {"fields": lay, "bytes": size, "X": X}
            for idx, name in enumerate(names):
                t0 = time.time()
                srcf = self.src / f"{fam}_{name}.npz"
                if not srcf.exists():
                    sys.exit(f"missing {srcf.name}")
                z = np.load(srcf)
                vals = z["vals"]
                self.vals_all[fam].append(np.round(vals.astype(np.float64), 5).tolist())
                self.hz_all[fam].append([[hz(z["stab_span"][mi, pi]) for pi in range(len(pairs))] for mi in range(self.M)])
                stampf = self.d / "blk" / fam / str(idx) / "stamp.txt"
                stamp = f"{BLOCK_FORMAT}|{self.tag}|{srcf.stat().st_mtime_ns}|{srcf.stat().st_size}"
                if stampf.exists() and stampf.read_text() == stamp:
                    log(f"  {fam} {name}: unchanged, kept")
                    continue
                vecs, scale = z["vecs"], z["scale"]
                assert vecs.dtype == np.int16
                # browser display precision: int8 round(127 psi / max|psi|) with a non-zero floor (nodal signs exact)
                q8 = np.round(vecs.astype(np.float64) * 127.0 / 32767.0)
                q8 = np.where((q8 == 0) & (vecs != 0), np.sign(vecs), q8)
                vecs8 = np.clip(q8, -127, 127).astype(np.int8)
                err = np.abs(vecs8.astype(np.float64) / 127 - vecs.astype(np.float64) / 32767).max()
                worst_q8 = max(worst_q8, float(err))
                assert err <= 1.01 / 127, f"int8 display error {err} exceeds the stated bound"
                lab, rot, perm, sval = z["seba_label"], z["seba_rot"], z["seba_perm"], z["seba_val"]
                rpsi, rseba, span, local = z["retention_psi"], z["retention_seba"], z["stab_span"], z["stab_local"]
                for mi in range(self.M):
                    for pi in range(len(pairs)):
                        psi = vecs[mi, pi].T.astype(np.float64) * scale[mi, pi] / 32767.0        # (N, K)
                        v = np.concatenate([np.full((n, 1), 1 / np.sqrt(n)), psi], 1)
                        pn = (psi ** 2).sum(0) ** 2 / np.maximum((psi ** 4).sum(0), 1e-300)
                        reps = []
                        for k in range(K):
                            o = np.argsort(-psi[:, k], kind="stable")
                            reps += [o[:REP_PSI], o[::-1][:REP_PSI], np.argsort(np.abs(psi[:, k]), kind="stable")[:REP_PSI]]
                        for ri, r in enumerate(self.seba_r):
                            p_ = perm[mi, pi, ri, :r].astype(int)
                            s_ = seba_vectors(v[:, :r], rot[mi, pi, ri, :r, :r].astype(np.float64), p_)
                            worst_agree = min(worst_agree, float(np.mean(seba_hard(s_) == lab[mi, pi, ri])))
                            for j in range(r):
                                mem = np.flatnonzero(lab[mi, pi, ri] == j)
                                top = mem[np.argsort(-s_[mem, j], kind="stable")]
                                other = np.delete(s_[mem], j, axis=1).max(1) if r > 1 else np.zeros(len(mem))
                                bnd = mem[np.argsort(s_[mem, j] - other, kind="stable")]
                                reps += [pad(top, REP_TOP), pad(bnd, REP_BND)]
                        ret_c = np.concatenate([rseba[mi, pi, ri, :r].ravel() for ri, r in enumerate(self.seba_r)])
                        parts = [f32(scale[mi, pi]), f32(vals[mi, pi]), f32(pn), f32(span[mi, pi]), f32(local[mi, pi]),
                                 f32(rpsi[mi, pi]), f32(ret_c), np.concatenate(reps).astype("<u2"), vecs8[mi, pi],
                                 (lab[mi, pi].astype(np.int16) + 1).astype(np.uint8), sval[mi, pi]]
                        blob = b"".join(np.ascontiguousarray(a_).tobytes() for a_ in parts)
                        assert len(blob) == size, (len(blob), size)
                        gz = gzip.compress(blob, compresslevel=6, mtime=0)   # decompressed in the browser (DecompressionStream)
                        write(self.d / "blk" / fam / str(idx) / str(mi) / f"{pi}.bin.gz", gz)
                        total_bytes += len(gz); raw_bytes += len(blob)
                stampf.write_text(stamp)
                log(f"  {fam} {name}: {time.time() - t0:.1f} s")
        if worst_agree < 0.99:
            raise AssertionError(f"SEBA recomputation (archive precision) agrees with seba_label on only {worst_agree:.4f}")
        self.q8_error = worst_q8
        log(f"blocks: {total_bytes / 1e6:.1f} MB gzipped ({raw_bytes / 1e6:.1f} MB raw) written this run; "
            f"SEBA archive recompute min agreement {worst_agree:.4f}; int8 display max error {worst_q8:.5f} of max|psi|")

    # ------------------------------------------------------------------ JSON copies + derived
    def small_json(self):
        for name in ["decod.json", "curves.json"]:
            p = self.src / name
            if p.exists():
                obj = json.loads(p.read_text())
                write(self.d / name, json.dumps(obj, separators=(",", ":"), allow_nan=False).encode())
        self.val_acc = [None] * self.T
        p = self.src / "curves.json"
        if p.exists():
            by = {c["step"]: c for c in json.loads(p.read_text()).get("checkpoints", [])}
            self.val_acc = [by.get(s, {}).get("val_acc") for s in self.steps]
        self.measure_json()

    def measure_json(self):
        """Chart objects for the Measurements tab (DESIGN.md lines / heatmap formats): stability horizons, decodability."""
        out, m, K, G = {}, self.meta, self.K, self.G
        rules = self.modes
        if getattr(self, "hz_all", None):
            lp, tp = m["layer_pairs"], m["time_pairs"]
            diag_l = [lp.index([s, s]) for s in range(self.S)]
            for mi, rule in enumerate(rules):
                key = f"rule{mi}"
                out[f"single_{key}"] = {"x": list(range(self.S)), "xticklabels": self.sites, "xlabel": "site", "ylabel": "horizon h (modes)",
                    "yrange": [0, K + G], "colors": "ordinal", "refline": {"y": K, "label": f"K = {K} stored"},
                    "series": [{"name": f"step {self.steps[t]:,}", "y": [self.hz_all["layer"][t][mi][p] for p in diag_l]} for t in range(self.T)]}
                full_t = tp.index([0, self.T - 1])
                out[f"time_full_{key}"] = {"x": list(range(self.S)), "xticklabels": self.sites, "xlabel": "site", "ylabel": "horizon h (modes)",
                    "yrange": [0, K + G], "refline": {"y": K, "label": f"K = {K} stored"},
                    "series": [{"name": f"time block, all {self.T} checkpoints", "y": [self.hz_all["time"][s][mi][full_t] for s in range(self.S)]},
                               {"name": "single snapshot, final checkpoint", "y": [self.hz_all["layer"][self.T - 1][mi][p] for p in diag_l], "dash": True}]}
                z = [[[(self.hz_all["layer"][t][mi][lp.index([a, b])] if a <= b else None) for b in range(self.S)] for a in range(self.S)] for t in range(self.T)]
                out[f"layer_heat_{key}"] = {"rows": self.sites, "cols": self.sites, "z": z, "zlabel": "horizon h", "zrange": [0, K + G],
                    "rowlabel": "first site a", "collabel": "last site b", "fmt": "int",
                    "slider": {"label": "checkpoint", "values": [f"step {s:,}" for s in self.steps]}}
            allh = {fam: np.array(self.hz_all[fam]) for fam in ("layer", "time")}     # [anchor][mode][pair]
            self.hz_summary = {rule: {fam: {"median": float(np.median(allh[fam][:, mi])), "min": int(allh[fam][:, mi].min()),
                                            "max": int(allh[fam][:, mi].max())} for fam in allh} for mi, rule in enumerate(rules)}
            out["horizon_summary"] = self.hz_summary
        p = self.src / "decod.json"
        if p.exists():
            d = json.loads(p.read_text())
            C = d.get("n_classes", len(m["classes"]))
            for view in d["views"]:
                sites = [s for s in d["sites"] if s in d["results"][view]]
                if not sites:
                    continue
                out[f"decod_{view}"] = {"x": list(range(len(sites))), "xticklabels": sites, "xlabel": "site", "ylabel": "test accuracy",
                    "yrange": [0, 1], "refline": {"y": 1 / C, "label": "chance"},
                    "series": [{"name": "ridge (closed-form fit)", "y": [d["results"][view][s]["ridge_acc"] for s in sites]},
                               {"name": "cosine 20-NN vote", "y": [d["results"][view][s]["knn_acc"] for s in sites]}]}
            self.decod = d
        write(self.d / "measure.json", json.dumps(out, separators=(",", ":"), allow_nan=False).encode())

    # ------------------------------------------------------------------ meta + facts
    def write_meta(self):
        m = self.meta
        fl = lambda key: np.array(m[key], dtype=float) if key in m else None   # None -> nan (kappa is null for rank-based rules)
        kap, mass = fl("kappa"), fl("nn_mass")
        rnd = lambda a: None if a is None else [[[None if not np.isfinite(x) else round(float(x), 4) for x in row] for row in mode] for mode in a]
        if self.skip_blocks:
            old = json.loads((self.d / "meta.json").read_text())
            self.vals_all, self.layouts_blk, self.hz_all = old["vals"], old["block_layout"], old["hz"]
        meta = {
            "tag": self.tag, "fixture": bool(m.get("fixture", False)), "steps": self.steps, "sites": self.sites,
            "modes": self.modes, "default_mode": self.default_mode, "kappa": rnd(kap), "kappa_q10": rnd(fl("kappa_q10")),
            "kappa_q90": rnd(fl("kappa_q90")), "self_weight": rnd(fl("self_weight")), "nn_mass": rnd(mass), "seba_r": self.seba_r, "n_vecs": self.K, "n": self.n,
            "model": m.get("model"), "dataset": m.get("dataset"), "fine_classes": [c.replace("_", " ") for c in m["fine_classes"]] if m.get("fine_classes") else None,
            "fine_coarse": self.fine_coarse, "stability": m["stability"], "hz": self.hz_all, "guard": self.G,
            "perplexity_unattainable": m.get("perplexity_unattainable"), "q8_error": getattr(self, "q8_error", None),
            "classes": [c.replace("_", " ") for c in m["classes"]], "rule": m.get("rule", ""), "layer_pairs": m["layer_pairs"], "time_pairs": m["time_pairs"],
            "vals": self.vals_all, "block_layout": self.layouts_blk, "n_fun": self.n_fun,
            "points_fields": self.points_fields, "layouts": self.layout_info,
            "val_acc": self.val_acc, "sprite_cols": self.sprite_cols, "palette": self.palette,
            "source_meta": {k: v for k, v in m.items() if k not in ("kappa", "kappa_q10", "kappa_q90", "self_weight", "nn_mass", "layer_pairs", "time_pairs", "steps", "sites", "modes", "classes")},
        }
        write(self.d / "meta.json", json.dumps(meta, separators=(",", ":"), allow_nan=False).encode())
        # facts for {{fact:NAME}} in content fragments
        f = self.facts
        f["n_images"] = f"{self.n:,}"
        f["n_checkpoints"] = str(self.T)
        f["n_sites"] = str(self.S)
        f["steps"] = ", ".join(f"{s:,}" for s in self.steps)
        f["final_step"] = f"{self.steps[-1]:,}"
        if self.val_acc[-1] is not None:
            f["final_val_acc"] = f"{100 * self.val_acc[-1]:.1f}%"
        mid = [i for i, s in enumerate(self.sites) if i > 0 and s not in ("pool", "logits")] or list(range(self.S))
        f["model"], f["dataset"] = m.get("model", ""), m.get("dataset", "")
        f["n_vecs"], f["seba_r"] = str(self.K), ", ".join(map(str, self.seba_r))
        f["sites"] = ", ".join(self.sites)
        f["n_classes"] = str(len(m["classes"]))
        f["stab_core"] = f"{m['stability']['core']:,}"
        f["stab_subset"] = f"{m['stability']['subset_sizes'][0]:,}"
        for rule, fams in getattr(self, "hz_summary", {}).items():
            ri = self.modes.index(rule)
            for fam, v in fams.items():
                f[f"hz.rule{ri}.{fam}.median"] = f"{v['median']:g}"
                f[f"hz.rule{ri}.{fam}.range"] = f"{v['min']}–{v['max']}"
        if getattr(self, "hz_all", None):
            lp = m["layer_pairs"]
            for mi in range(self.M):
                single = [self.hz_all["layer"][self.T - 1][mi][lp.index([s_, s_])] for s_ in range(self.S)]
                f[f"hz.rule{mi}.final_single"] = ", ".join(f"{a}: {b}" for a, b in zip(self.sites, single))
                f[f"hz.rule{mi}.final_full"] = str(self.hz_all["layer"][self.T - 1][mi][lp.index([0, self.S - 1])])
        if getattr(self, "decod", None):
            for view, res in self.decod["results"].items():
                for site, r_ in res.items():
                    f[f"decod.{view}.{site}.ridge"] = f"{100 * r_['ridge_acc']:.1f}%"
                    f[f"decod.{view}.{site}.knn"] = f"{100 * r_['knn_acc']:.1f}%"
        g = lambda v, fmt: "n/a" if not np.isfinite(v) else fmt.format(v)
        for mi, name in enumerate(self.modes):
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")      # e.g. knn_graph_k_15
            for key in (f"rule{mi}", slug):
                f[f"kappa.{key}.final.mid_median"] = g(np.nanmedian(kap[mi, -1, mid]) if np.isfinite(kap[mi, -1, mid]).any() else np.nan, "{:.2g}")
                f[f"kappa.{key}.final.min"] = g(np.nanmin(kap[mi, -1]) if np.isfinite(kap[mi, -1]).any() else np.nan, "{:.2g}")
                f[f"kappa.{key}.final.max"] = g(np.nanmax(kap[mi, -1]) if np.isfinite(kap[mi, -1]).any() else np.nan, "{:.2g}")
                f[f"nn_mass.{key}.final.mid_median"] = g(100 * np.nanmedian(mass[mi, -1, mid]), "{:.0f}%")
                f[f"nn_mass.{key}.all.min"] = g(100 * np.nanmin(mass[mi]), "{:.0f}%")
                f[f"nn_mass.{key}.all.max"] = g(100 * np.nanmax(mass[mi]), "{:.0f}%")
                f[f"mode_name.{key}"] = name
        f["default_mode"] = self.modes[self.default_mode]
        sw = fl("self_weight")
        if sw is not None:
            for mi, name in enumerate(self.modes):
                if np.isfinite(sw[mi, -1, mid]).any():
                    f[f"self_weight.rule{mi}.final.mid_median"] = f"{np.nanmedian(sw[mi, -1, mid]):.2g}"
        (self.d / "facts.json").write_text(json.dumps(f, indent=1))

    # ------------------------------------------------------------------ facts from analysis.py
    def merge_analysis_facts(self):
        """assets/questions.json["facts"] = {name: value-or-formatted-string} joins the fact table as q.<name>.
        Absent file or key: nothing is added (analysis.py may not have produced them yet). Rewrites facts.json."""
        for k in [k for k in self.facts if k.startswith("q.")]:
            del self.facts[k]
        p = OUT / "assets" / "questions.json"
        qf = json.loads(p.read_text()).get("facts") if p.exists() else None
        if qf is None:
            log("note: assets/questions.json has no 'facts' key; no q.* facts")
        elif not isinstance(qf, dict):
            sys.exit('assets/questions.json["facts"] must be a flat {name: value} object')
        else:
            for k, v in qf.items():
                if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                    sys.exit(f'assets/questions.json["facts"]["{k}"]: expected a string or number, got {type(v).__name__}')
                self.facts[f"q.{k}"] = v if isinstance(v, str) else f"{v:,}" if isinstance(v, int) else f"{v:.3g}"
            log(f"facts: {len(qf)} q.* facts from assets/questions.json")
        (self.d / "facts.json").write_text(json.dumps(self.facts, indent=1))

    # ------------------------------------------------------------------ HTML
    def sub_facts(self, text: str, where: str) -> str:
        def sub(mo):
            key = mo.group(1).strip()
            if key not in self.facts:
                sys.exit(f"{where}: unknown fact '{key}'. Known facts:\n  " + "\n  ".join(sorted(self.facts)))
            return html.escape(self.facts[key])
        return re.sub(r"\{\{\s*fact:([^}]+)\}\}", sub, text)

    def fragment(self, name: str, placeholder: str) -> str:
        p = CONTENT / f"{name}.html"
        text = self.sub_facts(p.read_text() if p.exists() else placeholder, p.name)
        # prose images load lazily so the first paint of a tab does not wait for every figure
        return re.sub(r"<img(?![^>]*\bloading=)", '<img loading="lazy" decoding="async"', text)

    def tours(self) -> str:
        """content/tours.json = [{"id", "label", "hash": "#explorer?...", "note": html}], validated and inlined
        as JSON for js/explorer.js (the Tours menu; #explorer?tour=<id>)."""
        p = CONTENT / "tours.json"
        if not p.exists():
            return "[]"
        tours = json.loads(p.read_text())
        bad = lambda msg: sys.exit(f"content/tours.json: {msg}")
        if not isinstance(tours, list):
            bad("expected a list of tours")
        seen = set()
        for i, t in enumerate(tours):
            if not isinstance(t, dict) or not all(isinstance(t.get(k), str) and t[k] for k in ("id", "label", "hash", "note")):
                bad(f"tour {i}: needs non-empty string fields id, label, hash, note")
            if not TOUR_ID.fullmatch(t["id"]) or t["id"] in seen:
                bad(f"tour {i}: id '{t['id']}' must be a unique lower-case slug (a-z, 0-9, -)")
            if not t["hash"].startswith("#explorer?") or "tour=" in t["hash"]:
                bad(f"tour '{t['id']}': hash must start with '#explorer?' and must not itself contain tour=")
            seen.add(t["id"])
            t["note"] = self.sub_facts(t["note"], f"tours.json ({t['id']})")
        # inlined in a <script type="application/json">: keep "</" out of it
        return json.dumps(tours, ensure_ascii=False).replace("</", "<\\/")

    def assemble_html(self):
        self.merge_analysis_facts()
        # header.json: {"title", "subtitle", "short_title", "date", "description"?}; a "joke" key is ignored
        hdr = {"title": DEFAULT_TITLE, "subtitle": DEFAULT_SUBTITLE, "short_title": DEFAULT_SHORT_TITLE,
               "date": dt.date.today().isoformat()}
        hp = CONTENT / "header.json"
        if hp.exists():
            hdr.update({k: v for k, v in json.loads(hp.read_text()).items() if v})
        date = dt.date.fromisoformat(hdr["date"])
        title_html = html.escape(hdr["title"])
        ph = lambda what, extra="": (f'<div class="placeholder"><b>{what}</b> will appear here: the writer\'s fragment '
                                     f'<code>site/content/{what.lower()}.html</code> has not been built yet.</div>{extra}')
        q_extra = ('\n<h2 id="decodability">Where and when is class linearly decodable?</h2>\n'
                   '<figure class="fig wide live" data-live="decod"><figcaption><b>How to read.</b> Accuracy of a closed-form '
                   'ridge fit and of a cosine kNN vote on one site\'s features (a validation fit, not part of the construction).'
                   '</figcaption></figure>\n<figure class="fig wide live" data-live="curves"><figcaption>Training curves.</figcaption></figure>')
        rep = {
            "{{TITLE}}": title_html,
            "{{TITLE_TEXT}}": html.escape(hdr["title"]),
            "{{SHORT_TITLE}}": html.escape(hdr["short_title"]),
            "{{SUBTITLE}}": html.escape(hdr["subtitle"]),
            "{{DESCRIPTION}}": html.escape(hdr.get("description", hdr["title"] + " " + hdr["subtitle"])),
            "{{DATE_ISO}}": date.isoformat(), "{{DATE}}": f"{date.day} {date:%B %Y}",
            "{{BUILD_DATE}}": dt.date.today().isoformat(), "{{TAG}}": html.escape(self.tag),
            "{{BANNER}}": html.escape(f"One network: {self.meta.get('model', '')} on {self.meta.get('dataset', '')}, one seed."),
            "{{FIXTURE_NOTE}}": " (SYNTHETIC FIXTURE: pictures carry no meaning)" if self.meta.get("fixture") else "",
            "{{EXPLORER}}": (HERE / "explorer.html").read_text(),
            "{{CONTENT_OVERVIEW}}": self.fragment("overview", ph("Overview")),
            "{{CONTENT_FINDINGS}}": self.fragment("findings", ph("Findings")),
            "{{CONTENT_MEASUREMENTS}}": self.fragment("measurements", ph("Measurements", q_extra)),
            "{{CONTENT_METHOD}}": self.fragment("method", ph("Method")),
            "{{TOURS_JSON}}": self.tours(),
        }
        page = (HERE / "index.html").read_text()
        for k, v in rep.items():
            page = page.replace(k, v)
        left = re.findall(r"\{\{[A-Z_]+\}\}", page)
        assert not left, left
        (OUT / "index.html").write_text(self.cache_bust(page))

    def cache_bust(self, page):
        """Append ?v=<hash of the built js/css> to the page's css/js URLs and to the relative module imports
        inside js/, so a returning reader never runs cached old modules against a new page (Pages: max-age=600)."""
        import hashlib
        files = sorted([*(OUT / "js").glob("*.js"), *(OUT / "css").glob("*.css")])
        v = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:10]
        imp = re.compile(r"""(["'])\./([A-Za-z0-9_-]+\.js)\1""")
        for p in (OUT / "js").glob("*.js"):
            p.write_text(imp.sub(lambda m: f"{m[1]}./{m[2]}?v={v}{m[1]}", p.read_text()))
        return re.sub(r"""((?:src|href)=")((?:js|css)/[A-Za-z0-9_-]+\.(?:js|css))(")""", rf"\1\2?v={v}\3", page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--layout-tag", default=None, help="second network whose UMAP can lay out the points (e.g. r20s1b)")
    ap.add_argument("--default-mode", type=int, default=None, help="kappa mode index selected on load")
    ap.add_argument("--facts", action="store_true", help="print the {{fact:NAME}} values and exit (needs a previous build)")
    ap.add_argument("--skip-blocks", action="store_true", help="reuse the block chunks of the previous build (fast UI iteration)")
    ap.add_argument("--only", default=None, help="development: build block chunks only for e.g. layer:7,time:5")
    ap.add_argument("--html-only", action="store_true", help="development: only re-copy css/js/assets and re-assemble index.html")
    ap.add_argument("--allow-missing", action="store_true", help="development: tolerate missing layer_/time_ files")
    ap.add_argument("--clean", action="store_true", help="discard all previous output, including block chunks")
    a = ap.parse_args()
    global OUT, CONTENT
    OUT = ROOT / "viewer" / a.tag
    CONTENT = HERE / "content" / a.tag
    if a.facts:
        p = OUT / "d" / "facts.json"
        if not p.exists():
            sys.exit("build first")
        for k, v in json.loads(p.read_text()).items():
            print(f"{k:45s} {v}")
        return
    b = Builder(a.tag, a.layout_tag, a.default_mode, a.skip_blocks, a.clean)
    b.only = set(a.only.split(",")) if a.only else None
    b.allow_missing = a.allow_missing
    if a.html_only:   # development: re-copy static files and re-assemble index.html from the previous build's data
        b.facts = json.loads((OUT / "d" / "facts.json").read_text())
        b.copy_static(); b.assemble_html()
        log("re-assembled html")
        return
    b.run()


if __name__ == "__main__":
    main()
