"""Build the static mini-writeup site from measurement data (schema_version 2, SPEC.md section 2).

    uv run --with numpy --with pillow python site/build_site.py <tag> [--layout-tag r20s1b] [--default-mode 2]
    uv run --with numpy --with pillow --with scipy --with scikit-learn python site/build_site.py r20s0b --baselines r20s0c
                                                                      # + "Sets from" data and the Baselines and nulls tab
    ... build_site.py r20s0b --baselines r20s0c --alignment ../spectral_align/data   # + the Alignment tab (site/alignment.py)
    ... build_site.py r20s0b --layout-tag r20s1b --dot-tag r20s0d   # + viewer/site/d_dot/ (dot kernel, SPEC_DOT.md section 5)
    ... build_site.py r20s0b --dot-tag r20s0d --dot-only              # rebuild d_dot/ + kernels.json only (d/ untouched)
    uv run --with numpy --with pillow python site/build_site.py <tag> --facts      # list {{fact:NAME}} values
    python3 -m http.server 8765 --bind 127.0.0.1 -d viewer/site                    # then open http://127.0.0.1:8765/

Reads data/<tag>/ (and optionally data/<layout-tag>/umap_*.npy, a second network's layouts) and site/ sources;
writes viewer/site/ (deploy root, .nojekyll). Content fragments site/content/{overview,findings,measurements,method}.html
are inlined when present (placeholders otherwise); site/content/tours.json (explorer tours) and site/content/header.json
are read when present; site/assets/ is copied verbatim, and assets/questions.json["facts"] joins the fact table as q.<name>.

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
ROOT = HERE.parent                              # explore/cifar_coherent/
OUT = ROOT / "viewer" / "site"
N_VECS = 8
BLOCK_FORMAT = "blk-v7"        # bump when the block chunk layout or its derivation changes
REP_PSI, REP_TOP, REP_BND = 12, 24, 12      # representative images stored per function
DEFAULT_TITLE = ("Towards Characterizing a Neural Network's Native Ontology: Coherent Sets of Data Through Layers "
                 "and Over Training")
DEFAULT_SUBTITLE = ("or, Sparse Eigenbases of the Tempered Cross-Hankel Operator for Spectral Identification of the "
                    "Kantian Categories by Neural Coherent Sets")
DEFAULT_SHORT_TITLE = "Coherent sets in ResNet-20"
TOUR_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
POINT_FIELDS = ["loss", "jac_fro", "grad_cos", "grad_dot", "grad_norm"]   # f32; plus pred/correct
DOT_ROOT = "d_dot"             # second data root: the dot-product (Euclidean) kernel, SPEC_DOT.md section 5
DOT_LABEL = "dot (Euclidean, keeps the norm)"
SIZE_CAP = 800e6               # whole site; above it the dot root ships rules 0-1 only (SPEC_DOT.md section 5)
COARSE_FORMAT = "coarse-v1"    # bump when d/ext, d/km or d/coarse change layout (SPEC_COARSE.md sections 3 and 5)
EXT_CAP = 300e6                # psi_9..psi_32 add-on; above it: all rules for single cells, the default rule for chains
EXT_FAMS = {"layer": "avg", "layerprodR": "prodR", "layerprodL": "prodL"}   # explorer data key -> coarse npz prefix
KEEP_D = ("blk", "meta.json", "ext", "km", "coarse")   # incremental (stamped) parts of d/ kept between builds


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
        assert self.meta.get("schema_version") == 2, "build_site.py expects schema_version 2"
        m = self.meta
        self.steps, self.sites, self.modes = m["steps"], m["sites"], m["modes"]
        self.T, self.S, self.M, self.n = len(self.steps), len(self.sites), len(self.modes), m["n"]
        self.seba_r = m["seba_r"]
        self.R = len(self.seba_r)
        self.n_fun = N_VECS + sum(self.seba_r)
        self.default_mode = default_mode if default_mode is not None else (
            m["default_mode"] if isinstance(m.get("default_mode"), int) else
            self.modes.index("half-mass (calibrated)") if "half-mass (calibrated)" in self.modes else 0)
        self.facts: dict[str, str] = {}
        self.d = OUT / "d"
        self.ref0 = None            # seed-0 layout: Procrustes target (DotBuilder: the cosine seed-0 layout); None = chain only
        self.extra_meta: dict = {}  # extra keys for meta.json (DotBuilder)
        self.coarse_meta = None     # d/ext, d/km, d/coarse description (cosine root only; see coarse())

    # -- layout of one block chunk (byte offsets); X = number of cells along the retention axis
    def block_layout(self, X: int, product: bool = False):
        n, R = self.n, self.R
        SR, SR2 = sum(self.seba_r), sum(r * r for r in self.seba_r)
        parts = [("scale", "f32", N_VECS), ("vals", "f32", N_VECS + 1), ("pn", "f32", N_VECS), ("rot", "f32", SR2),
                 ("ret_psi", "f32", N_VECS * X), ("ret_seba", "f32", SR * X), ("bcorr", "f32", SR),
                 *([("gain_psi", "f32", N_VECS * X), ("gain_seba", "f32", SR * X)] if product else []),
                 ("reps", "u16", N_VECS * 3 * REP_PSI + sum(self.seba_r) * (REP_TOP + REP_BND)),
                 ("vecs", "i8", N_VECS * n), ("labels", "u4", R * ((n + 1) // 2)), ("perm", "i8", R * 8)]
        size = {"f32": 4, "u16": 2, "i16": 2, "i8": 1, "u8": 1, "u4": 1}
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
                if p.name == DOT_ROOT:   # the dot-kernel root belongs to DotBuilder: a cosine rebuild never deletes it
                    continue
                if p.name == "d" and not self.clean:
                    for q in p.iterdir():
                        if q.name not in KEEP_D:
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
        self.coarse()
        self.small_json()
        self.write_meta()
        if self.baselines:       # site/baselines.py: d/src/ (explorer "Sets from") and d/bn/ (Baselines and nulls tab)
            import baselines
            self.facts.update(baselines.build(self, self.baselines))
        if self.alignment:       # site/alignment.py: d/al/ (Alignment tab; analysis by explore/spectral_align/analyze.py)
            import alignment
            self.facts.update(alignment.build(self, self.alignment))
        if getattr(self, "dot_tag", None):   # second data root d_dot/ (SPEC_DOT.md section 5): the explorer's Kernel switch
            DotBuilder(self.dot_tag, self.tag, self.clean, getattr(self, "dot_rules", "auto"), getattr(self, "allow_missing", False)).run()
        link_kernels()
        self.assemble_html()
        (OUT / ".nojekyll").write_text("")
        report_size(t0)

    # ------------------------------------------------------------------ static files
    def copy_static(self):
        for name in ["css", "js", "vendor", "assets"]:
            if (HERE / name).exists():
                shutil.copytree(HERE / name, OUT / name, dirs_exist_ok=True)
        pal = json.loads((HERE / "palette.json").read_text())
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
        e0 = procrustes_chain(emb, ref=self.ref0)
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
        # spectral layout: AlignedUMAP of each snapshot's own diffusion coordinates (measure_v2.spectral_embed)
        files = [self.src / f"umapspec_{s}.npy" for s in self.steps]
        if all(f.exists() for f in files):
            es = procrustes_chain(np.stack([np.load(f) for f in files]), ref=e0)
            k = len(self.layout_info)
            write(self.d / f"umap_{k}.bin", quantise_layout(es).tobytes())
            self.layout_info.append({"seed": 0, "tag": self.tag, "kind": "spectral", "mode": 0, "n_coords": 32})
            log("spectral layout added")

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
        n, R = self.n, self.R
        fams = [("layer", self.steps, self.meta["layer_pairs"], self.S), ("time", self.sites, self.meta["time_pairs"], self.T)]
        # product (composition) blocks, product.py: right (start-of-block) and left (end-of-block) singular vectors
        # all four product families or none: the explorer's Operator switch assumes every one exists
        prod = [(f + "prod" + side, names, prs, X) for f, names, prs, X in list(fams) for side in "RL"]
        have = [all((self.src / f"{f}_{nm}.npz").exists() for nm in names) for f, names, _, _ in prod]
        if all(have):
            fams += prod
        elif any(have):
            log("  product families incomplete: building without the Operator switch")
        self.vals_all = {f[0]: [] for f in fams}
        worst_agree = 1.0
        self.browser_corr = []
        self.layouts_blk = {}
        total_bytes = raw_bytes = 0
        for fam, names, pairs, X in fams:
            product = "prod" in fam
            lay, size = self.block_layout(X, product)
            self.layouts_blk[fam] = {"fields": lay, "bytes": size, "X": X}
            for idx, name in enumerate(names):
                t0 = time.time()
                if not (self.src / f"{fam}_{name}.npz").exists():
                    if not getattr(self, "allow_missing", False):
                        sys.exit(f"missing {fam}_{name}.npz (use --allow-missing for a development build)")
                    log(f"  {fam} {name}: MISSING (development build)")
                    self.vals_all[fam].append([[[None] * (N_VECS + 1)] * len(pairs)] * self.M)
                    continue
                if self.only is not None and f"{fam}:{idx}" not in self.only:
                    with np.load(self.src / f"{fam}_{name}.npz") as zz:
                        self.vals_all[fam].append(np.round(zz["vals"][:self.M].astype(np.float64), 6).tolist())
                    continue
                srcf = self.src / f"{fam}_{name}.npz"
                z = np.load(srcf)
                stampf = self.d / "blk" / fam / str(idx) / "stamp.txt"
                trim = f"|rules0-{self.M - 1}" if self.M < len(self.meta["modes"]) else ""     # DotBuilder may ship leading rules only
                stamp = f"{BLOCK_FORMAT}|{self.tag}{trim}|{srcf.stat().st_mtime_ns}|{srcf.stat().st_size}|{self.points_stamp}"
                if stampf.exists() and stampf.read_text() == stamp:
                    self.vals_all[fam].append(np.round(z["vals"][:self.M].astype(np.float64), 6).tolist())
                    log(f"  {fam} {name}: unchanged, kept")
                    continue
                n_bc0 = len(self.browser_corr)
                vals, vecs, scale = z["vals"], z["vecs"], z["scale"]
                qmax = 32767.0 if vecs.dtype == np.int16 else 127.0
                # shipped as int8 round(127 psi / max|psi|) with a non-zero floor: an entry that is not zero never rounds
                # to 0, so nodal signs stay exact (plain int8 would zero the small entries of localised modes)
                q8 = np.round(vecs.astype(np.float64) * 127.0 / qmax)
                q8 = np.where((q8 == 0) & (vecs != 0), np.sign(vecs), q8)
                vecs8 = np.clip(q8, -127, 127).astype(np.int8)
                lab, rot, perm = z["seba_label"], z["seba_rot"], z["seba_perm"]
                rpsi, rseba = z["retention_psi"], z["retention_seba"]
                gpsi, gseba = (z["gain_psi"], z["gain_seba"]) if product else (None, None)
                self.vals_all[fam].append(np.round(vals[:self.M].astype(np.float64), 6).tolist())
                for mi in range(self.M):
                    P = len(pairs)
                    rep_all, pn_all, bc_all = [], [], []
                    for pi in range(P):
                        psi = vecs[mi, pi].T.astype(np.float64) * scale[mi, pi] / qmax        # (N, 8)
                        v = np.concatenate([np.full((n, 1), 1 / np.sqrt(n)), psi], 1)
                        bcorr_pair = []
                        pn_all.append((psi ** 2).sum(0) ** 2 / np.maximum((psi ** 4).sum(0), 1e-300))
                        psi8 = vecs8[mi, pi].T.astype(np.float64) * scale[mi, pi] / 127.0           # what the browser sees
                        v8 = np.concatenate([np.full((n, 1), 1 / np.sqrt(n)), psi8], 1)
                        reps = []
                        for k in range(N_VECS):
                            o = np.argsort(-psi[:, k], kind="stable")
                            reps += [o[:REP_PSI], o[::-1][:REP_PSI], np.argsort(np.abs(psi[:, k]), kind="stable")[:REP_PSI]]
                        for ri, r in enumerate(self.seba_r):
                            p_ = perm[mi, pi, ri, :r].astype(int)
                            s_ = seba_vectors(v[:, :r], rot[mi, pi, ri, :r, :r].astype(np.float64), p_)
                            agree = float(np.mean(seba_hard(s_) == lab[mi, pi, ri]))
                            worst_agree = min(worst_agree, agree)
                            s8 = seba_vectors(v8[:, :r], rot[mi, pi, ri, :r, :r].astype(np.float64), p_)
                            for j in range(r):
                                a_, b_ = s_[:, j] - s_[:, j].mean(), s8[:, j] - s8[:, j].mean()
                                c_ = float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-300))
                                self.browser_corr.append(c_); bcorr_pair.append(c_)
                            for j in range(r):
                                mem = np.flatnonzero(lab[mi, pi, ri] == j)
                                top = mem[np.argsort(-s_[mem, j], kind="stable")]
                                other = np.delete(s_[mem], j, axis=1).max(1) if r > 1 else np.zeros(len(mem))
                                bnd = mem[np.argsort(s_[mem, j] - other, kind="stable")]
                                reps += [pad(top, REP_TOP), pad(bnd, REP_BND)]
                        rep_all.append(np.concatenate(reps).astype(np.uint16))
                        bc_all.append(np.array(bcorr_pair))
                    for pi in range(P):
                        rot_c = np.concatenate([rot[mi, pi, ri, :r, :r].ravel() for ri, r in enumerate(self.seba_r)])
                        ret_c = np.concatenate([rseba[mi, pi, ri, :r].ravel() for ri, r in enumerate(self.seba_r)])
                        gains = []
                        if product:
                            gains = [f32(gpsi[mi, pi]), f32(np.concatenate([gseba[mi, pi, ri, :r].ravel() for ri, r in enumerate(self.seba_r)]))]
                        parts = [f32(scale[mi, pi]), f32(vals[mi, pi]), f32(pn_all[pi]), f32(rot_c), f32(rpsi[mi, pi]), f32(ret_c), f32(bc_all[pi]),
                                 *gains, rep_all[pi], vecs8[mi, pi], pack_nibbles(lab[mi, pi]), perm[mi, pi].astype(np.int8)]
                        blob = b"".join(np.ascontiguousarray(a).tobytes() for a in parts)
                        assert len(blob) == size, (len(blob), size)
                        gz = gzip.compress(blob, compresslevel=6, mtime=0)   # decompressed in the browser (DecompressionStream)
                        write(self.d / "blk" / fam / str(idx) / str(mi) / f"{pi}.bin.gz", gz)
                        total_bytes += len(gz); raw_bytes += len(blob)
                bc = np.array(self.browser_corr[n_bc0:]) if len(self.browser_corr) > n_bc0 else np.array([1.0])
                (stampf.parent / "bcorr.json").write_text(json.dumps([float(bc.min()), float(np.quantile(bc, .01)), float(np.median(bc))]))
                stampf.write_text(stamp)
                log(f"  {fam} {name}: {time.time() - t0:.1f} s")
                self.int16 = qmax > 200
        side = [json.loads(p.read_text()) for p in (self.d / "blk").glob("*/*/bcorr.json")]
        if side:
            self.browser_seba = {"min": round(min(x[0] for x in side), 5), "p01": round(min(x[1] for x in side), 5), "median": round(float(np.median([x[2] for x in side])), 6)}
            log(f"browser-side SEBA vectors (from 8-bit psi) vs full precision: correlation {self.browser_seba}")
        if worst_agree < 0.99:
            msg = f"SEBA recomputation agrees with seba_label on only {worst_agree:.4f} of points"
            if getattr(self, "int16", True):
                raise AssertionError(msg)
            log("warning (int8 vecs): " + msg)
        log(f"blocks: {total_bytes / 1e6:.1f} MB gzipped ({raw_bytes / 1e6:.1f} MB raw); SEBA recompute min agreement {worst_agree:.4f}")

    # ------------------------------------------------------------------ coarse-graining add-ons (SPEC_COARSE.md 3, 5)
    def coarse(self):
        """Cosine root only. From data/<tag>/coarse/coarse_<step>_<rule>.npz and coarse_summary.json (coarse_modal.py,
        coarse_summary.py) write three lazily loaded add-ons; the blk/ format and n_vecs are untouched:
          ext/<fam>/<t>/<rule>/<pair>.bin.gz   psi_9..psi_32 of the layer average (fam "layer") and of the ordered
                                               product's start / end side ("layerprodR" / "layerprodL"); layout in
                                               meta.coarse.ext. A single cell (a == b) is stored once, under "layer"
                                               (all three operators coincide there).
          km/<step>_<rule>/<pair>.bin.gz       spectral k-means partitions, k = 2..16, start side then (chains) end side,
                                               4 bits per image, groups renumbered by decreasing size (display only)
          coarse/<step>_<rule>.json.gz         per block: spectrum (top 65), retention ceilings F_k, gap reports, the
                                               chi^2 scores of every partition method (4 significant digits)
        Which rules ship for chains is decided from the measured compressed size (EXT_CAP; recorded in meta)."""
        cdir, summ = self.src / "coarse", self.src / "coarse_summary.json"
        files = {(t, r): cdir / f"coarse_{s}_{r}.npz" for t, s in enumerate(self.steps) for r in range(self.M)}
        roots = [self.d / x for x in ("ext", "km", "coarse")]
        if not summ.exists() or not all(f.exists() for f in files.values()):
            for q in roots:
                shutil.rmtree(q, ignore_errors=True)
            log("coarse: no data/<tag>/coarse/ (or coarse_summary.json): explorer coarse-graining add-ons not built")
            return
        pairs = [tuple(p) for p in self.meta["layer_pairs"]]
        assert pairs == [(a, b) for a in range(self.S) for b in range(a, self.S)], "coarse npz pair order differs from layer_pairs"
        stamp = f"{COARSE_FORMAT}|{EXT_CAP}|{summ.stat().st_mtime_ns}|" + "|".join(f"{f.stat().st_mtime_ns}" for f in files.values())
        index = self.d / "coarse" / "index.json"
        if index.exists():
            old = json.loads(index.read_text())
            if old.get("stamp") == stamp:
                self.coarse_meta = old["meta"]
                log("coarse add-ons: unchanged, kept")
                return
        t0 = time.time()
        for q in roots:
            shutil.rmtree(q, ignore_errors=True)
        sys.path.insert(0, str(ROOT))
        from coarse_summary import block_gap        # the same descriptive gap rule as coarse_summary.json
        S_ = json.loads(summ.read_text())
        K, methods = S_["K"], S_["methods"]
        g4 = lambda x: None if x is None or not np.isfinite(x) else float(f"{float(x):.4g}")
        L4 = lambda a: [g4(x) for x in a]
        cell_of = [a == b for a, b in pairs]
        n, NE, first = self.n, 24, 9
        lay = {"vals": {"offset": 0, "dtype": "f32", "count": 33}, "scale": {"offset": 132, "dtype": "f32", "count": NE},
               "vecs": {"offset": 132 + 4 * NE, "dtype": "i8", "count": NE * n}}
        ext_bytes = 132 + 4 * NE + NE * n
        ks = list(range(2, 17))
        half = (n + 1) // 2

        def q8(v):     # int16 -> int8 with the blk/ rule: a non-zero entry never rounds to 0 (nodal signs stay exact)
            q = np.round(v.astype(np.float64) * 127.0 / 32767.0)
            return np.clip(np.where((q == 0) & (v != 0), np.sign(v), q), -127, 127).astype(np.int8)

        def nib(lab):  # (k, N) int8 in 0..15 -> (k, ceil(N/2)) uint8, low nibble = even index
            v = lab.astype(np.uint8)
            if v.shape[-1] % 2:
                v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,), np.uint8)], -1)
            return (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8)

        def by_size(lab):   # renumber groups by decreasing size (ties: first occurrence); display only
            out = np.empty_like(lab)
            for i, row in enumerate(lab):
                u, first_i, cnt = np.unique(row, return_index=True, return_counts=True)
                order = u[np.lexsort((first_i, -cnt))]
                m_ = np.zeros(int(row.max()) + 1, np.int8); m_[order] = np.arange(len(order))
                out[i] = m_[row]
            return out

        sizes = {"ext_cells": 0, "ext_chains": {}, "km_cells": 0, "km_chains": {}, "coarse": 0}
        written = []      # (path, bytes, rule, is_cell, kind) so the chains of unshipped rules can be dropped afterwards

        def put(rel, data, rule, cell, kind):
            gz = gzip.compress(data, compresslevel=6, mtime=0)
            write(self.d / rel, gz)
            written.append((self.d / rel, len(gz), rule, cell, kind))
            if cell:
                sizes[kind + "_cells"] += len(gz)
            else:
                sizes[kind + "_chains"][rule] = sizes[kind + "_chains"].get(rule, 0) + len(gz)

        for (t, r), f in files.items():
            step = self.steps[t]
            z = np.load(f)
            unit = S_["units"][f"{step}_{r}"]
            for fam, pre in EXT_FAMS.items():
                vals_all = z["avg_vals"] if pre == "avg" else z["sv"]
                vecs, scale = z[f"{pre}_vecs"], z[f"{pre}_scale"]
                assert vecs.shape[1] == NE, vecs.shape
                for pi, cell in enumerate(cell_of):
                    if cell and fam != "layer":
                        continue
                    blob = f32(vals_all[pi, :33]).tobytes() + f32(scale[pi]).tobytes() + q8(vecs[pi]).tobytes()
                    assert len(blob) == ext_bytes
                    put(f"ext/{fam}/{t}/{r}/{pi}.bin.gz", blob, r, cell, "ext")
            ls, le = z["lab_start"], z["lab_end"]
            assert ls.min() >= 0 and le.min() >= 0 and ls.max() < 16 and le.max() < 16, "k-means labels outside 0..15"
            for pi, cell in enumerate(cell_of):
                data = nib(by_size(ls[pi])).tobytes() + (b"" if cell else nib(by_size(le[pi])).tobytes())
                put(f"km/{step}_{r}/{pi}.bin.gz", data, r, cell, "km")
            blocks = []
            for pi, (a, b) in enumerate(pairs):
                u = unit[pi]
                assert (u["a"], u["b"]) == (a, b)
                cell = a == b
                rec = {"sv": L4(u["sv"]), "F": L4(u["F"]), "total": g4(u["total"]), "gap": u["gap"],
                       "avg": L4(z["avg_vals"][pi, :65]), "avg_gap": block_gap(z["avg_vals"][pi].astype(float), True)}
                keys = ["I_start", "I_end", "I_two", "keff_start", "keff_end"] + (["ncut", "maxleak"] if cell else [])
                rec["m"] = {mt: {k: L4(u[mt][k]) for k in keys} for mt in methods}
                if cell:
                    rec["kyfan"], rec["cheeger_lo"] = L4(u["kyfan"]), L4(u["cheeger_lo"])
                blocks.append(rec)
            data = json.dumps({"step": step, "rule": r, "blocks": blocks}, separators=(",", ":"), allow_nan=False).encode()
            gz = gzip.compress(data, compresslevel=9, mtime=0)
            write(self.d / "coarse" / f"{step}_{r}.json.gz", gz)
            sizes["coarse"] += len(gz)
            log(f"  coarse {step} rule {r}: {time.time() - t0:.0f} s")
        # chains: every rule if the extension fits EXT_CAP, else the default rule only (SPEC_COARSE.md section 3)
        ext_all = sizes["ext_cells"] + sum(sizes["ext_chains"].values())
        chain_rules = list(range(self.M)) if ext_all <= EXT_CAP else [self.default_mode]
        for path, nb, r, cell, kind in written:
            if not cell and r not in chain_rules:
                path.unlink()
        ext_shipped = sizes["ext_cells"] + sum(v for r, v in sizes["ext_chains"].items() if r in chain_rules)
        km_shipped = sizes["km_cells"] + sum(v for r, v in sizes["km_chains"].items() if r in chain_rules)
        note = ("all rules" if len(chain_rules) == self.M else
                f"single cells: all rules; multi-site blocks: rule {self.default_mode} ({self.modes[self.default_mode]}) only "
                f"(all rules would be {ext_all / 1e6:.0f} MB gzipped, over the {EXT_CAP / 1e6:.0f} MB target)")
        self.coarse_meta = {
            "format": COARSE_FORMAT, "K": K, "methods": methods, "ks": ks, "steps": self.steps,
            "ext": {"first": first, "last": first + NE - 1, "fields": lay, "bytes": ext_bytes, "fams": list(EXT_FAMS),
                    "cells_under": "layer", "rules_cells": list(range(self.M)), "rules_chains": chain_rules,
                    "quant": "int8 round(127 psi / max|psi|), non-zero floor; psi = vecs * scale / 127",
                    "gz_bytes": ext_shipped, "gz_bytes_all_rules": ext_all, "note": note},
            "km": {"ks": ks, "half": half, "rules_cells": list(range(self.M)), "rules_chains": chain_rules, "gz_bytes": km_shipped,
                   "note": "spectral k-means (Lloyd on the top k-1 spectral coordinates, coarse_modal.py); groups renumbered by decreasing size"},
            "summary": {"gz_bytes": sizes["coarse"], "rules": list(range(self.M)), "sig_digits": 4},
        }
        write(index, json.dumps({"stamp": stamp, "meta": self.coarse_meta}).encode())
        log(f"coarse add-ons ({time.time() - t0:.0f} s): ext {ext_shipped / 1e6:.1f} MB gz shipped (all rules {ext_all / 1e6:.1f} MB; "
            f"chain rules {chain_rules}), k-means labels {km_shipped / 1e6:.1f} MB, summaries {sizes['coarse'] / 1e6:.1f} MB")

    # ------------------------------------------------------------------ JSON copies + derived
    def small_json(self):
        for name in ["decod.json", "curves.json", "avgprod.json", "normalisation_compare.json", "sweep.json", "product_summary.json"]:
            p = self.src / name
            if p.exists():
                obj = json.loads(p.read_text())
                write(self.d / name, json.dumps(obj, separators=(",", ":"), allow_nan=False).encode())
        self.val_acc = [None] * self.T
        p = self.src / "curves.json"
        if p.exists():
            by = {c["step"]: c for c in json.loads(p.read_text()).get("checkpoints", [])}
            self.val_acc = [by.get(s, {}).get("val_acc") for s in self.steps]

    # ------------------------------------------------------------------ meta + facts
    def write_meta(self):
        m = self.meta
        fl = lambda key: np.array(m[key], dtype=float) if key in m else None   # None -> nan (kappa is null for rank-based rules)
        kap, mass = fl("kappa"), fl("nn_mass")
        rnd = lambda a: None if a is None else [[[None if not np.isfinite(x) else round(float(x), 4) for x in row] for row in mode] for mode in a]
        if self.skip_blocks:
            old = json.loads((self.d / "meta.json").read_text())
            self.vals_all, self.layouts_blk = old["vals"], old["block_layout"]
        meta = {
            "tag": self.tag, "fixture": bool(m.get("fixture", False)), "steps": self.steps, "sites": self.sites,
            "modes": self.modes, "default_mode": self.default_mode, "kappa": rnd(kap), "kappa_q10": rnd(fl("kappa_q10")),
            "kappa_q90": rnd(fl("kappa_q90")), "self_weight": rnd(fl("self_weight")), "nn_mass": rnd(mass), "seba_r": self.seba_r, "n_vecs": N_VECS, "n": self.n,
            "classes": m["classes"], "rule": m.get("rule", ""), "layer_pairs": m["layer_pairs"], "time_pairs": m["time_pairs"],
            "vals": self.vals_all, "block_layout": self.layouts_blk, "n_fun": self.n_fun, "browser_seba_corr": getattr(self, "browser_seba", None),
            "points_fields": self.points_fields, "layouts": self.layout_info, "kernel": m.get("kernel", "cosine"),
            "shared_fingerprint": shared_fingerprint(self.src, m), **self.extra_meta,
            **({"coarse": self.coarse_meta} if self.coarse_meta else {}),
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
        mid = [i for i, s in enumerate(self.sites) if re.fullmatch(r"b[1-8]", s)] or list(range(self.S))
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
        p = HERE / "assets" / "questions.json"
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
        # Coarse-graining facts (coarse_summary.py --facts-only), quoted as {{fact:c.<name>}}.
        for k in [k for k in self.facts if k.startswith("c.")]:
            del self.facts[k]
        cp = self.src / "coarse_facts.json"
        if cp.exists():
            for k, v in json.loads(cp.read_text()).items():
                self.facts[f"c.{k}"] = v if isinstance(v, str) else f"{v:,}" if isinstance(v, int) else f"{v:.3g}"
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
        p = HERE / "content" / f"{name}.html"
        text = self.sub_facts(p.read_text() if p.exists() else placeholder, p.name)
        # prose images load lazily so the first paint of a tab does not wait for every figure
        return re.sub(r"<img(?![^>]*\bloading=)", '<img loading="lazy" decoding="async"', text)

    def tours(self) -> str:
        """content/tours.json = [{"id", "label", "hash": "#explorer?...", "note": html}], validated and inlined
        as JSON for js/explorer.js (the Tours menu; #explorer?tour=<id>)."""
        p = HERE / "content" / "tours.json"
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
        hp = HERE / "content" / "header.json"
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
            "{{FIXTURE_NOTE}}": " (SYNTHETIC FIXTURE: pictures carry no meaning)" if self.meta.get("fixture") else "",
            "{{EXPLORER}}": (HERE / "explorer.html").read_text(),
            "{{CONTENT_OVERVIEW}}": self.fragment("overview", ph("Overview")),
            "{{CONTENT_FINDINGS}}": self.fragment("findings", ph("Findings")),
            "{{CONTENT_MEASUREMENTS}}": self.fragment("measurements", ph("Measurements", q_extra)),
            "{{CONTENT_METHOD}}": self.fragment("method", ph("Method")),
            "{{CONTENT_METHOD_DETAILS}}": self.fragment("method_details", ph("Method details")),
            "{{CONTENT_BASELINES}}": self.fragment("baselines", ph("Baselines")),
            "{{CONTENT_PRIMER}}": self.fragment("primer", ph("Primer")),
            # Alignment tab: shown only when d/al/ was built (--alignment); otherwise hidden, with a placeholder
            "{{CONTENT_ALIGNMENT}}": self.fragment("alignment", "") if (OUT / "d" / "al" / "index.json").exists() else
                '<div class="placeholder"><b>Alignment</b> is built with <code>build_site.py ... --alignment &lt;data dir&gt;</code>.</div>',
            "{{ALIGNMENT_TAB_HIDDEN}}": "" if (OUT / "d" / "al" / "index.json").exists() else " hidden",
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


# ----------------------------------------------------------------------------------------------- dot-kernel root
def shared_fingerprint(src: Path, m: dict) -> str:
    """sha256 over what two data roots must share to be shown side by side: the grid (steps, sites, n), every probe.npz
    array (idx, labels, images) and the per-image diagnostics (points_<step>.npz contents, which the dot root reads
    from d/). Written into each root's meta.json; kernels.json advertises d_dot/ only when it equals d/'s."""
    import hashlib
    h = hashlib.sha256(json.dumps([m["steps"], m["sites"], m["n"]]).encode())
    files = [src / "probe.npz"] + [src / f"points_{s}.npz" for s in m["steps"]]
    for f in files:
        h.update(f.name.encode())
        if not f.exists():
            h.update(b"<missing>")
            continue
        with np.load(f) as z:
            for k in sorted(z.files):
                a = np.ascontiguousarray(z[k])
                h.update(f"{k}|{a.dtype.str}|{a.shape}".encode()); h.update(a.tobytes())
    return h.hexdigest()[:20]


def tree_state(p: Path) -> list:
    return sorted((str(q.relative_to(p)), q.stat().st_size, q.stat().st_mtime_ns) for q in p.rglob("*") if q.is_file())


def tree_bytes(p: Path) -> int:
    return sum(q.stat().st_size for q in p.rglob("*") if q.is_file()) if p.exists() else 0


class DotBuilder(Builder):
    """viewer/site/d_dot/: the dot-kernel measurement (data/<dot_tag>, schema v2, meta "kernel": "dot") in d/'s formats.
    Ships meta.json, umap_<i>.bin (seed 0, Procrustes onto the cosine seed-0 layout, and the spectral layout) and blk/.
    Labels, sprite and points are kernel-independent copies: checked equal to the cosine tag's and read from d/.
    Never touches anything outside d_dot/. rules: 'auto' (all unless the site would pass SIZE_CAP, then 0-1), 'all', or a
    count of leading rules."""

    def __init__(self, tag: str, cos_tag: str, clean: bool, rules: str = "auto", allow_missing: bool = False):
        super().__init__(tag, None, None, False, clean)
        self.d = OUT / DOT_ROOT
        self.cos_src = ROOT / "data" / cos_tag
        self.only, self.allow_missing, self.baselines, self.alignment = None, allow_missing, None, None
        if self.meta.get("kernel") != "dot":
            sys.exit(f"{self.src}/meta.json: expected \"kernel\": \"dot\", got {self.meta.get('kernel')!r}")
        cm = json.loads((self.cos_src / "meta.json").read_text())
        for k in ("steps", "sites", "n", "seba_r", "n_vecs", "classes", "layer_pairs", "time_pairs"):
            if cm.get(k) != self.meta.get(k):
                sys.exit(f"--dot-tag {tag}: '{k}' differs from {cos_tag}; the two kernels must share the grid")
        if cm["modes"] != self.meta["modes"][:len(cm["modes"])]:
            sys.exit(f"--dot-tag {tag}: rule list differs from {cos_tag}")
        self.rules = rules

    def choose_rules(self) -> int:
        M = len(self.meta["modes"])
        if self.rules == "all":
            return M
        if self.rules != "auto":
            return max(1, min(M, int(self.rules)))
        # projection: the site without d_dot plus, per rule, what the cosine root's blocks for that rule weigh
        cos_d = OUT / "d" / "blk"
        per = [sum(tree_bytes(q / str(mi)) for q in cos_d.glob("*/*")) for mi in range(M)]
        rest = sum(tree_bytes(q) if q.is_dir() else q.stat().st_size for q in OUT.iterdir() if q.name != DOT_ROOT)
        proj = rest + sum(per) + 3e6
        K = M if proj <= SIZE_CAP else 2
        log(f"dot root: site without d_dot {rest / 1e6:.0f} MB; projected with all {M} dot rules {proj / 1e6:.0f} MB "
            f"(cap {SIZE_CAP / 1e6:.0f} MB) -> shipping rules 0-{K - 1}")
        return K

    def trim(self, K: int):
        self.M = K
        self.modes = self.meta["modes"][:K]
        for key in ("kappa", "kappa_q10", "kappa_q90", "self_weight", "nn_mass"):
            if key in self.meta:
                self.meta[key] = self.meta[key][:K]
        if self.default_mode >= K:
            self.default_mode = 0
        self.extra_meta = {"kernel_label": DOT_LABEL, "shared_from": "d", "modes_all": self.meta["modes"],
                           "modes_note": "" if K == len(self.meta["modes"]) else
                           f"rules {K}-{len(self.meta['modes']) - 1} not shipped (site size cap {SIZE_CAP / 1e6:.0f} MB)",
                           "layouts_note": "seed 0 and spectral only (no seed-1 dot layout); seed 0 rotated onto the cosine seed-0 layout"}

    def check_shared(self):
        """labels, images and per-image diagnostics are copies of the cosine tag's: the browser reads them from d/."""
        a, b = np.load(self.src / "probe.npz"), np.load(self.cos_src / "probe.npz")
        for k in sorted(set(a.files) | set(b.files)):
            if k not in a.files or k not in b.files or not np.array_equal(a[k], b[k]):
                sys.exit(f"{self.src}/probe.npz: '{k}' differs from {self.cos_src.name}; the kernels must share the probe")
        for step in self.steps:
            fa, fb = self.src / f"points_{step}.npz", self.cos_src / f"points_{step}.npz"
            if not fa.exists():
                continue
            with np.load(fa) as pa, np.load(fb) as pb:
                for k in set(pa.files) & set(pb.files):
                    if not np.array_equal(pa[k], pb[k], equal_nan=pa[k].dtype.kind == "f"):
                        sys.exit(f"{fa.name}: '{k}' differs from {self.cos_src.name}; per-image diagnostics are kernel-independent")
        self.points_stamp = "|".join(f"{(self.src / f'points_{s}.npz').stat().st_mtime_ns}" for s in self.steps if (self.src / f'points_{s}.npz').exists())

    def run(self):
        t0 = time.time()
        K = self.choose_rules()
        self.trim(K)
        if self.d.exists():
            if self.clean:
                shutil.rmtree(self.d)
            else:      # keep block chunks (incremental, stamped); drop rules no longer shipped
                for q in self.d.iterdir():
                    if q.name not in ("blk", "meta.json"):
                        shutil.rmtree(q) if q.is_dir() else q.unlink()
                for q in self.d.glob("blk/*/*/*"):
                    if q.is_dir() and q.name.isdigit() and int(q.name) >= K:
                        shutil.rmtree(q)
        self.d.mkdir(parents=True, exist_ok=True)
        self.check_shared()
        cm = json.loads((OUT / "d" / "meta.json").read_text())       # the cosine root, built first
        self.palette, self.sprite_cols, self.points_fields = cm["palette"], cm["sprite_cols"], cm["points_fields"]
        self.ref0 = procrustes_chain(np.stack([np.load(self.cos_src / f"umap_{s}.npy") for s in self.steps]))
        self.layouts()
        self.blocks()
        self.val_acc = cm["val_acc"]
        self.write_meta()
        log(f"wrote {self.d}: {tree_bytes(self.d) / 1e6:.1f} MB, rules {self.modes} ({time.time() - t0:.0f} s)")


def link_kernels():
    """viewer/site/kernels.json: the other data roots the explorer may offer (only d_dot/ for now), read from each root's
    own meta.json. A root is listed only if its shared_fingerprint equals d/'s (same grid, probe and per-image data),
    so a preserved d_dot/ built against another probe is never advertised. Writes nothing inside d/ or d_dot/."""
    out = {}
    mp, dp = OUT / "d" / "meta.json", OUT / DOT_ROOT / "meta.json"
    if mp.exists() and dp.exists():
        cm, dm = json.loads(mp.read_text()), json.loads(dp.read_text())
        fc, fd = cm.get("shared_fingerprint"), dm.get("shared_fingerprint")
        if fc and fc == fd:
            out["dot"] = {"root": DOT_ROOT, "tag": dm["tag"], "label": DOT_LABEL, "modes": dm["modes"], "fingerprint": fd}
            log(f"kernel switch: {DOT_ROOT}/ ({dm['tag']}, rules {len(dm['modes'])})")
        else:
            log(f"warning: {DOT_ROOT}/ ({dm.get('tag')}) does not match d/ ({cm.get('tag')}): shared fingerprint {fd} != {fc}; "
                f"not offered in the explorer (rebuild it with --dot-tag)")
    (OUT / "kernels.json").write_text(json.dumps(out, separators=(",", ":")))


def report_size(t0: float):
    tot = tree_bytes(OUT)
    nfiles = sum(1 for p in OUT.rglob("*") if p.is_file())
    log(f"wrote {OUT}: {tot / 1e6:.1f} MB in {nfiles} files (of which {DOT_ROOT}/ {tree_bytes(OUT / DOT_ROOT) / 1e6:.1f} MB) "
        f"({time.time() - t0:.0f} s)")
    if tot > SIZE_CAP * 1.02:
        log(f"warning: the site is over the {SIZE_CAP / 1e6:.0f} MB cap; rebuild with --dot-rules 2")


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
    ap.add_argument("--baselines", default=None, help="data tag of the baselines study (e.g. r20s0c): explorer 'Sets from' data + Baselines and nulls tab")
    ap.add_argument("--alignment", default=None, help="spectral-alignment data directory (e.g. ../spectral_align/data): the Alignment tab")
    ap.add_argument("--dot-tag", default=None, help=f"data tag of the dot-kernel measurement (e.g. r20s0d): second data root {DOT_ROOT}/, the explorer's Kernel switch")
    ap.add_argument("--dot-rules", default="auto", help="dot root: 'auto' (all rules unless the site would pass 800 MB, then 0-1), 'all', or a number of leading rules")
    ap.add_argument("--dot-only", action="store_true", help=f"rebuild only {DOT_ROOT}/ (needs a previous build of <tag>, the cosine root); touches nothing else")
    a = ap.parse_args()
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
    b.baselines = a.baselines
    b.alignment = a.alignment
    b.dot_tag, b.dot_rules = a.dot_tag, a.dot_rules
    if a.dot_only:
        if not a.dot_tag or not (OUT / "d" / "meta.json").exists():
            sys.exit("--dot-only needs --dot-tag and a previous build of the cosine root")
        t0 = time.time()
        before = tree_state(OUT / "d")
        DotBuilder(a.dot_tag, a.tag, a.clean, a.dot_rules, a.allow_missing).run()
        link_kernels()
        if tree_state(OUT / "d") != before:      # --dot-only must never write inside the cosine root
            sys.exit(f"--dot-only changed files under {OUT / 'd'}: this is a bug in build_site.py")
        report_size(t0)
        return
    if a.html_only:   # development: re-copy static files and re-assemble index.html from the previous build's data
        b.facts = json.loads((OUT / "d" / "facts.json").read_text())
        b.copy_static(); b.assemble_html()
        log("re-assembled html")
        return
    b.run()


if __name__ == "__main__":
    main()
