"""Build the static mini-writeup site from measurement data (schema_version 2, SPEC.md section 2).

    uv run --with numpy --with pillow python site/build_site.py <tag> [--layout-tag r20s1b] [--default-mode 2]
    uv run --with numpy --with pillow python site/build_site.py <tag> --facts      # list {{fact:NAME}} values
    python3 -m http.server 8765 --bind 127.0.0.1 -d viewer/site                    # then open http://127.0.0.1:8765/

Reads data/<tag>/ (and optionally data/<layout-tag>/umap_*.npy, a second network's layouts) and site/ sources;
writes viewer/site/ (deploy root, .nojekyll). Content fragments site/content/{overview,questions,method}.html are
inlined when present (placeholders otherwise); site/assets/ is copied verbatim.

Data written to viewer/site/d/ (all little-endian; see DATA_LAYOUT below and js/data.js):
  meta.json            steps, sites, modes, kappa, nn_mass, all block spectra, layouts of the binary chunks, ...
  umap_<seed>.bin      u16 (T, S, N, 2): per-cell layout scaled to [0, 65535] (0.5-99.5 percentile box)
  points.bin           per-image diagnostics, f32 (T, N) per field (fields listed in meta.points_fields)
  labels.bin           i8 (N) CIFAR labels; sprite.png: the 2,000 probe images, 50 per row
  chan/<t>_<s>.bin     per cell: lo f32 (C), hi f32 (C), u8 (N, C) spatially averaged channels
  blk/<fam>/<idx>/<mode>/<pair>.bin.gz   one block, gzip-compressed; layout in meta.block_layout (see block_layout())
The R^2 grid in each block chunk (in-sample OLS of the centred function on the cell's centred channels, SPEC
section 2 "Loadings") is computed here rather than in the browser so that one interaction does not need to fetch
the channels of all 96 cells; the definition is the same.
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
BLOCK_FORMAT = "blk-v6"        # bump when the block chunk layout or its derivation changes
REP_PSI, REP_TOP, REP_BND = 12, 24, 12      # representative images stored per function
DEFAULT_TITLE = ("Spectral Clustering of Tempered Cross-Hankel for Unsupervised Persistent Neural Ontology "
                 "Identification")
TITLE_JOKE = "(This name is a joke)"
POINT_FIELDS = ["loss", "jac_fro", "grad_cos", "grad_dot", "grad_norm"]   # f32; plus pred/correct


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

    # -- layout of one block chunk (byte offsets); X = number of cells along the retention axis
    def block_layout(self, X: int):
        n, R, F = self.n, self.R, self.n_fun
        SR, SR2 = sum(self.seba_r), sum(r * r for r in self.seba_r)
        parts = [("scale", "f32", N_VECS), ("vals", "f32", N_VECS + 1), ("pn", "f32", N_VECS), ("rot", "f32", SR2),
                 ("ret_psi", "f32", N_VECS * X), ("ret_seba", "f32", SR * X), ("bcorr", "f32", SR),
                 ("reps", "u16", N_VECS * 3 * REP_PSI + sum(self.seba_r) * (REP_TOP + REP_BND)),
                 ("vecs", "i8", N_VECS * n), ("labels", "u4", R * ((n + 1) // 2)), ("perm", "i8", R * 8),
                 ("r2", "u8", F * self.T * self.S)]
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
        self.points_and_channels()
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

    # ------------------------------------------------------------------ per-image diagnostics + channels
    def points_and_channels(self):
        T, n = self.T, self.n
        fields = {}
        pred = np.zeros((T, n), np.int8)
        correct = np.zeros((T, n), np.uint8)
        self.chan = {}           # (t, s) -> centred float32 (N, C)
        self.channels = []
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
                for s, site in enumerate(self.sites):
                    q, lo, hi = p[f"chan_{site}"], p[f"chan_{site}_lo"], p[f"chan_{site}_hi"]
                    if t == 0:
                        self.channels.append(int(q.shape[1]))
                    write(self.d / "chan" / f"{t}_{s}.bin", f32(lo).tobytes() + f32(hi).tobytes() + q.astype(np.uint8).tobytes())
                    x = lo + (hi - lo) * (q.astype(np.float32) / 255.0)
                    self.chan[(t, s)] = (x - x.mean(0)).astype(np.float32)
        present = present or []
        self.points_fields = present
        self.points_stamp = "|".join(f"{(self.src / f'points_{s}.npz').stat().st_mtime_ns}" for s in self.steps if (self.src / f'points_{s}.npz').exists())
        blob = b"".join(f32(fields[f]).tobytes() for f in present) + pred.tobytes() + correct.tobytes()
        write(self.d / "points.bin", blob)
        # orthonormal bases of the channel column spaces (rank-revealing), for the R^2 grid
        self.chan_basis = {}
        for key, x in self.chan.items():
            u, sv, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
            rank = int((sv > sv[0] * 1e-6).sum()) if sv.size and sv[0] > 0 else 0
            self.chan_basis[key] = u[:, :rank].astype(np.float32)
        log(f"points fields: {present}; channels per site {self.channels}")

    # ------------------------------------------------------------------ blocks
    def blocks(self):
        n, R = self.n, self.R
        self.vals_all = {"layer": [], "time": []}
        fams = [("layer", self.steps, self.meta["layer_pairs"], self.S), ("time", self.sites, self.meta["time_pairs"], self.T)]
        cells = [(t, s) for t in range(self.T) for s in range(self.S)]
        bases = [self.chan_basis[c] for c in cells]
        worst_agree = 1.0
        self.browser_corr = []
        self.layouts_blk = {}
        total_bytes = raw_bytes = 0
        for fam, names, pairs, X in fams:
            lay, size = self.block_layout(X)
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
                        self.vals_all[fam].append(np.round(zz["vals"].astype(np.float64), 6).tolist())
                    continue
                srcf = self.src / f"{fam}_{name}.npz"
                z = np.load(srcf)
                stampf = self.d / "blk" / fam / str(idx) / "stamp.txt"
                stamp = f"{BLOCK_FORMAT}|{self.tag}|{srcf.stat().st_mtime_ns}|{srcf.stat().st_size}|{self.points_stamp}"
                if stampf.exists() and stampf.read_text() == stamp:
                    self.vals_all[fam].append(np.round(z["vals"].astype(np.float64), 6).tolist())
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
                self.vals_all[fam].append(np.round(vals.astype(np.float64), 6).tolist())
                for mi in range(self.M):
                    P = len(pairs)
                    # all functions of all pairs of this (fam, idx, mode): (N, P*F)
                    funcs = np.zeros((n, P * self.n_fun), np.float32)
                    rep_all, pn_all, bc_all = [], [], []
                    for pi in range(P):
                        psi = vecs[mi, pi].T.astype(np.float64) * scale[mi, pi] / qmax        # (N, 8)
                        v = np.concatenate([np.full((n, 1), 1 / np.sqrt(n)), psi], 1)
                        cols = [psi]
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
                            cols.append(s_)
                            for j in range(r):
                                mem = np.flatnonzero(lab[mi, pi, ri] == j)
                                top = mem[np.argsort(-s_[mem, j], kind="stable")]
                                other = np.delete(s_[mem], j, axis=1).max(1) if r > 1 else np.zeros(len(mem))
                                bnd = mem[np.argsort(s_[mem, j] - other, kind="stable")]
                                reps += [pad(top, REP_TOP), pad(bnd, REP_BND)]
                        rep_all.append(np.concatenate(reps).astype(np.uint16))
                        bc_all.append(np.array(bcorr_pair))
                        funcs[:, pi * self.n_fun:(pi + 1) * self.n_fun] = np.concatenate(cols, 1)
                    funcs -= funcs.mean(0)
                    norm2 = (funcs.astype(np.float64) ** 2).sum(0) + 1e-30
                    r2 = np.zeros((len(cells), P * self.n_fun), np.float32)
                    for ci, U in enumerate(bases):
                        if U.shape[1]:
                            r2[ci] = ((U.T @ funcs) ** 2).sum(0) / norm2
                    r2 = np.clip(r2, 0, 1).reshape(len(cells), P, self.n_fun).transpose(1, 2, 0)   # P, F, T*S
                    r2q = np.round(r2 * 255).astype(np.uint8)
                    for pi in range(P):
                        rot_c = np.concatenate([rot[mi, pi, ri, :r, :r].ravel() for ri, r in enumerate(self.seba_r)])
                        ret_c = np.concatenate([rseba[mi, pi, ri, :r].ravel() for ri, r in enumerate(self.seba_r)])
                        parts = [f32(scale[mi, pi]), f32(vals[mi, pi]), f32(pn_all[pi]), f32(rot_c), f32(rpsi[mi, pi]), f32(ret_c), f32(bc_all[pi]),
                                 rep_all[pi], vecs8[mi, pi], pack_nibbles(lab[mi, pi]), perm[mi, pi].astype(np.int8), r2q[pi]]
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

    # ------------------------------------------------------------------ JSON copies + derived
    def small_json(self):
        for name in ["decod.json", "curves.json", "avgprod.json", "normalisation_compare.json", "sweep.json"]:
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
            "points_fields": self.points_fields, "channels": self.channels, "layouts": self.layout_info,
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

    # ------------------------------------------------------------------ HTML
    def fragment(self, name: str, placeholder: str) -> str:
        p = HERE / "content" / f"{name}.html"
        text = p.read_text() if p.exists() else placeholder

        def sub(mo):
            key = mo.group(1).strip()
            if key not in self.facts:
                sys.exit(f"{p.name}: unknown fact '{key}'. Known facts:\n  " + "\n  ".join(sorted(self.facts)))
            return html.escape(self.facts[key])
        text = re.sub(r"\{\{\s*fact:([^}]+)\}\}", sub, text)
        # prose images load lazily so the first paint of a tab does not wait for every figure
        return re.sub(r"<img(?![^>]*\bloading=)", '<img loading="lazy" decoding="async"', text)

    def assemble_html(self):
        hdr = {"title": DEFAULT_TITLE, "joke": TITLE_JOKE,
               "subtitle": "Which images does a trained network treat as alike, and does that grouping persist across "
                           "layers and through training? A label-free spectral look at ResNet-20 on CIFAR-10.",
               "date": dt.date.today().isoformat()}
        hp = HERE / "content" / "header.json"
        if hp.exists():
            user = json.loads(hp.read_text())
            if "joke" not in user and "joke" in user.get("subtitle", "").lower() + user.get("title", "").lower():
                user["joke"] = ""          # the writer already says it
            hdr.update(user)
        date = dt.date.fromisoformat(hdr["date"])
        title_html = html.escape(hdr["title"]) + (f' <span class="joke">{html.escape(hdr["joke"])}</span>' if hdr.get("joke") else "")
        ph = lambda what, extra="": (f'<div class="placeholder"><b>{what}</b> will appear here: the writer\'s fragment '
                                     f'<code>site/content/{what.lower()}.html</code> has not been built yet.</div>{extra}')
        q_extra = ('\n<h2 id="decodability">Where and when is class linearly decodable?</h2>\n'
                   '<figure class="fig wide live" data-live="decod"><figcaption><b>How to read.</b> Accuracy of a closed-form '
                   'ridge fit and of a cosine kNN vote on one site\'s features (a validation fit, not part of the construction).'
                   '</figcaption></figure>\n<figure class="fig wide live" data-live="curves"><figcaption>Training curves.</figcaption></figure>')
        rep = {
            "{{TITLE}}": title_html,
            "{{TITLE_TEXT}}": html.escape(hdr["title"] + (" " + hdr["joke"] if hdr.get("joke") else "")),
            "{{SHORT_TITLE}}": html.escape(hdr.get("short_title", "Coherent sets in ResNet-20")),
            "{{SUBTITLE}}": html.escape(hdr["subtitle"]),
            "{{DESCRIPTION}}": html.escape(hdr["subtitle"]),
            "{{DATE_ISO}}": date.isoformat(), "{{DATE}}": f"{date.day} {date:%B %Y}",
            "{{BUILD_DATE}}": dt.date.today().isoformat(), "{{TAG}}": html.escape(self.tag),
            "{{FIXTURE_NOTE}}": " (SYNTHETIC FIXTURE: pictures carry no meaning)" if self.meta.get("fixture") else "",
            "{{EXPLORER}}": (HERE / "explorer.html").read_text(),
            "{{CONTENT_OVERVIEW}}": self.fragment("overview", ph("Overview")),
            "{{CONTENT_QUESTIONS}}": self.fragment("questions", ph("Questions", q_extra)),
            "{{CONTENT_METHOD}}": self.fragment("method", ph("Method")),
        }
        page = (HERE / "index.html").read_text()
        for k, v in rep.items():
            page = page.replace(k, v)
        left = re.findall(r"\{\{[A-Z_]+\}\}", page)
        assert not left, left
        (OUT / "index.html").write_text(page)


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
