"""Export the Mamba-130m token-context study for the explorer, validated against the source arrays.

    uv run -q --with numpy python -I explore/ncs_report/build_mamba.py

Reads ../mamba_contexts/data/m130c (read-only) and writes viewer/site/d/mamba/:
  items.json       per context: snippet U, position t, token at t (T), next token (Nx), token class at t / t+1
                   (Cc, Cn), source (W), the trained model's top-1 next token (A) and its probability (p1)
  ctx.bin          u16 (N, 28): the 24 tokens up to and including t, then the 4 after (65535 = none)
  vocab.json       {id: decoded string} for every id shown
  layouts.bin      u16 (L, N, 2): the stored UMAP layouts of psi_1..psi_16 of L objects (display only)
  obj/<ck>/<o>.bin.gz   one spectral object (layout in meta "obj_layout"): vals f32 (17: lambda_0..16),
                   pn f32 (16), vecs i8 (16, N) = round(127 psi_k / max|psi_k|) with a non-zero floor,
                   labels i8 (9, N) (the stored SEBA hard labels, -1 unassigned), memb u8 (9, N) = round(255 s),
                   s the item's SEBA value for its own set (0 if unassigned), ends u16 (16, 2, 12) (the stored
                   12 largest / 12 smallest entries of psi_k)
  obj/<ck>/<o>.json     examples[r][j]: 12 members of set j in decreasing SEBA value (the stored list for r = 8, 16)
  meta.json        object list and names, ranks, classes, sources, layouts, palette, validation summary

Checks (the build fails if one does not hold): SEBA labels reconstructed from the stored int16 eigenvectors and
rotations (the convention of mc_schema.validate: dequantise, prepend the constant, threshold 0.99/sqrt(N), unit
norm, sign, max-scale, reliability permutation, clamp) agree with the stored labels on >= 99 % of items for every
object and r; set sizes, the commonest tokens at t and t+1, class and source make-up recomputed from the exported
arrays equal sets_<ck>.json exactly; the stored examples are members of their set; the int8 eigenvectors correlate
>= 0.98 with the int16 ones (worst reported; low only for localised modes), move no item by more than one of the explorer's 41 colour levels, and keep the
sign pattern (nodal domains) exactly; the exported context's token at t is
the stored T and the next is the stored Nx.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "mamba_contexts" / "data" / "m130c"
OUT = HERE / "viewer" / "site" / "d" / "mamba"
NV = 16                 # eigenfunctions shipped per object
CTX_T = 23              # index of the token at t inside ctx

NAMES = {"Bpoint": "residual, all 24 layers", "Bearly": "residual, layers 1–8", "Bmid": "residual, layers 9–16",
         "Blate": "residual, layers 17–24", "Bk8": "residual, all layers, k = 8", "Bk30": "residual, all layers, k = 30",
         "Bstate": "per-layer state, all 24 layers", "whole": "whole state (all 48 state constituents)",
         "Blagpoint": "residual L16 along reading (τ = 0 … 16)", "Blagwhole": "whole state along reading (τ = 0 … 16)",
         "bag": "mean input embedding of the prefix (reference)", "recent": "mean input embedding of the last 16 tokens (reference)",
         "pred": "the model's next-token distribution (reference)"}
SHORT = {"Bpoint": "residual L1–24", "Bearly": "residual L1–8", "Bmid": "residual L9–16", "Blate": "residual L17–24",
         "Bstate": "state L1–24", "whole": "whole state", "Blagpoint": "residual L16, τ 0–16",
         "Blagwhole": "whole state, τ 0–16", "bag": "bag (reference)", "recent": "recent (reference)", "pred": "pred (reference)"}


# neighbour count of each object's kNN graph: the run's k, except the two stored sensitivity blocks (meta["k_sens"])
K_OF = {"Bk8": 8, "Bk30": 30}


def name_of(o: str) -> str:
    if o in NAMES:
        return NAMES[o]
    view, rest = o.split("_", 1)
    v = {"point": "residual", "state": "state"}.get(view, view)
    if "_lag" in rest:
        L, lag = rest.split("_lag")
        return f"{'whole state' if view == 'whole' else v + ' ' + L} read {lag} token{'s' if lag != '1' else ''} further (τ = {lag})"
    if view == "whole":
        n = rest[3:]
        return f"whole state read {n} token{'s' if n != '1' else ''} further"
    return f"{v} {rest}"


def group_of(o: str) -> str:
    if o in ("bag", "recent", "pred"):
        return "references"
    if "lag" in o:
        return "along reading"
    if o.startswith("point_") or o in ("Bpoint", "Bearly", "Bmid", "Blate", "Bk8", "Bk30"):
        return "residual"
    return "state"


def log(*a):
    print(*a, flush=True)


def fail(msg):
    sys.exit(f"build_mamba: {msg}")


def seba_soft(v, rot, perm):
    """v (N, r) incl. the constant column; mirrors cifar_coherent/seba.py seba_reconstruct (clip=True)."""
    z = v @ rot
    s = np.sign(z) * np.maximum(np.abs(z) - 0.99 / np.sqrt(len(v)), 0)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0)); sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return np.maximum(s[:, perm], 0)


def hard(s):
    a = s.argmax(1)
    a[s.max(1) <= 0] = -1
    return a


def quantise_layout(e):
    lo, hi = np.percentile(e, 0.5, axis=0), np.percentile(e, 99.5, axis=0)
    span, mid = (hi - lo).max(), (hi + lo) / 2
    return np.round(np.clip((e - mid) / span + 0.5, 0, 1) * 65535).astype("<u2")


def main():
    t0 = time.time()
    meta = json.loads((SRC / "meta.json").read_text())
    if sorted(K_OF.values()) != sorted(meta["k_sens"]):
        fail(f"sensitivity blocks' k {K_OF} disagree with meta k_sens {meta['k_sens']}")
    R, N = meta["ranks"], meta["n"]
    vocab = json.loads((SRC / "vocab.json").read_text())["tokens"]
    it = np.load(SRC / "items.npz")
    ctx = it["ctx"]
    g = {k: it[k].astype(np.int64) for k in ("T", "Nx", "Cc", "Cn", "W", "U", "A")}
    if not (np.array_equal(ctx[:, CTX_T], g["T"]) and np.array_equal(ctx[:, CTX_T + 1], g["Nx"])):
        fail("ctx[:, 23] is not the token at t (or ctx[:, 24] not the next token)")
    if ctx.max() >= 65535:
        fail("token id does not fit u16")
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.rglob("*"):
        if p.is_file():
            p.unlink()
    (OUT / "ctx.bin").write_bytes(np.where(ctx < 0, 65535, ctx).astype("<u2").tobytes())
    used = set(np.unique(ctx[ctx >= 0]).tolist()) | set(np.unique(it["top5"]).tolist())
    miss = [i for i in used if str(i) not in vocab]
    if miss:
        fail(f"{len(miss)} token ids have no decoded string, e.g. {miss[:5]}")
    (OUT / "vocab.json").write_text(json.dumps({str(i): vocab[str(i)] for i in sorted(used)}, ensure_ascii=False, separators=(",", ":")))
    items = {k: g[k].tolist() for k in ("T", "Nx", "Cc", "Cn", "W", "U", "A")}
    items["t"] = it["t"].astype(int).tolist()
    items["p1"] = np.round(it["top5_p"][:, 0].astype(float), 3).tolist()
    (OUT / "items.json").write_text(json.dumps(items, separators=(",", ":")))
    lay = np.load(SRC / "layouts.npz")
    lnames = meta["layout_objects"]
    L = np.stack([quantise_layout(lay[o].astype(np.float64)) for o in lnames])
    if not np.isfinite(np.stack([lay[o] for o in lnames])).all():
        fail("non-finite layout")
    (OUT / "layouts.bin").write_bytes(L.tobytes())

    parts = [("vals", "f32", NV + 1), ("pn", "f32", NV), ("vecs", "i8", NV * N), ("labels", "i8", len(R) * N),
             ("memb", "u8", len(R) * N), ("ends", "u16", NV * 2 * 12)]
    size = {"f32": 4, "i8": 1, "u8": 1, "u16": 2}
    layout, off = {}, 0
    for nm, dt, cnt in parts:
        layout[nm] = {"offset": off, "dtype": dt, "count": cnt}
        off += size[dt] * cnt
    nbytes = off
    worst = {"seba_agree": 1.0, "vec_corr": 1.0, "example_overlap": 12, "examples_checked": 0, "summaries_checked": 0}
    objects = {}
    for ck in meta["checkpoints"]:
        sets = json.loads((SRC / f"sets_{ck}.json").read_text())
        z = np.load(SRC / f"spectra_{ck}.npz")
        for o in meta["objects"][ck]:
            q, scale = z[f"{o}__vecs"], z[f"{o}__scale"]
            vfull = q.T.astype(np.float64) * scale / 32767.0                       # (N, 32)
            v = np.concatenate([np.full((N, 1), 1 / np.sqrt(N)), vfull], 1)
            labs, rots, perms = z[f"{o}__seba_label"], z[f"{o}__seba_rot"], z[f"{o}__seba_perm"]
            memb = np.zeros((len(R), N), np.uint8)
            examples = {}
            for ri, r in enumerate(R):
                s = seba_soft(v[:, :r], rots[ri, :r, :r].astype(np.float64), perms[ri, :r].astype(int))
                lab = labs[ri].astype(np.int64)
                agree = float(np.mean(hard(s) == lab))
                worst["seba_agree"] = min(worst["seba_agree"], agree)
                if agree < 0.99:
                    fail(f"{ck}/{o} r={r}: reconstructed SEBA labels agree with the stored ones on {agree:.4f} of items")
                own = np.where(lab >= 0, s[np.arange(N), np.maximum(lab, 0)], 0.0)
                memb[ri] = np.round(np.clip(own, 0, 1) * 255).astype(np.uint8)
                ex = []
                stored = {d["set"]: d for d in sets.get(o, {}).get(str(r), [])}
                for j in range(r):
                    mem = np.flatnonzero(lab == j)
                    mine = mem[np.argsort(-s[mem, j], kind="stable")][:12].tolist()
                    if j in stored:
                        d = stored[j]
                        st = d.get("examples", [])
                        if not set(st) <= set(mem.tolist()):
                            fail(f"{ck}/{o} r={r} set {j}: a stored example is not a member")
                        if len(mem) >= 12:
                            worst["example_overlap"] = min(worst["example_overlap"], len(set(st) & set(mine)))
                        worst["examples_checked"] += 1
                        ex.append(st)
                        # summaries recomputed from the exported arrays must equal the stored descriptors
                        if d["size"] != len(mem):
                            fail(f"{ck}/{o} r={r} set {j}: size {len(mem)} vs stored {d['size']}")
                        for key, name in (("T", "tok_t"), ("Nx", "tok_next")):
                            ids, cnt = np.unique(g[key][mem], return_counts=True)
                            oo = np.argsort(-cnt, kind="stable")[:len(d[name])]
                            if [[int(ids[k]), int(cnt[k])] for k in oo] != d[name]:
                                fail(f"{ck}/{o} r={r} set {j}: {name} differs from the stored descriptor")
                        for key, name, n_ in (("Cc", "class_t", 12), ("Cn", "class_next", 12), ("W", "source", 5)):
                            if np.bincount(g[key][mem], minlength=n_).tolist() != d[name]:
                                fail(f"{ck}/{o} r={r} set {j}: {name} differs from the stored descriptor")
                        worst["summaries_checked"] += 1
                    else:
                        ex.append(mine)
                examples[str(r)] = ex
            # eigenfunctions psi_1..psi_16 as int8 with a non-zero floor (nodal signs exact)
            psi = vfull[:, :NV]
            mx = np.abs(psi).max(0)
            q8 = np.round(psi / mx * 127)
            q8 = np.where((q8 == 0) & (psi != 0), np.sign(psi), q8)
            vecs8 = np.clip(q8, -127, 127).astype(np.int8).T                       # (16, N)
            for k in range(NV):
                a, b = psi[:, k] - psi[:, k].mean(), vecs8[k].astype(np.float64)
                b = b - b.mean()
                c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                worst["vec_corr"] = min(worst["vec_corr"], c)
                # the explorer colours psi / max|psi| in 41 diverging levels: a level may move by at most one
                lv = lambda x: np.round((x + 1) / 2 * 40)
                dl = np.abs(lv(vecs8[k] / 127.0) - lv(psi[:, k] / mx[k])).max()
                worst["colour_level_shift"] = max(worst.get("colour_level_shift", 0), float(dl))
                if c < 0.98 or dl > 1 or not np.array_equal(np.sign(vecs8[k]), np.sign(psi[:, k])):
                    fail(f"{ck}/{o} psi_{k + 1}: int8 export correlation {c:.5f}, colour level shift {dl}, or sign pattern differs")
            ends = z[f"{o}__ends"].astype(np.int64)
            # the stored ends are the 12 largest / smallest of the float64 vectors; check they are extreme here too
            for k in range(NV):
                top = set(np.argsort(-psi[:, k], kind="stable")[:24].tolist())
                bot = set(np.argsort(psi[:, k], kind="stable")[:24].tolist())
                if len(set(ends[k, 0].tolist()) & top) < 10 or len(set(ends[k, 1].tolist()) & bot) < 10:
                    fail(f"{ck}/{o} psi_{k + 1}: a stored end is not extreme in the exported vector")
            vals = z[f"{o}__vals"][:NV + 1].astype(np.float32)
            pnv = ((psi ** 2).sum(0) ** 2 / (psi ** 4).sum(0)).astype(np.float32)
            blob = b"".join(np.ascontiguousarray(a).tobytes() for a in (
                vals.astype("<f4"), pnv.astype("<f4"), vecs8, labs.astype(np.int8), memb, ends.astype("<u2")))
            assert len(blob) == nbytes, (len(blob), nbytes)
            d = OUT / "obj" / ck
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{o}.bin.gz").write_bytes(gzip.compress(blob, compresslevel=6, mtime=0))
            (d / f"{o}.json").write_text(json.dumps({"examples": examples}, separators=(",", ":")))
            ent = objects.setdefault(o, {"name": name_of(o), "short": SHORT.get(o, name_of(o)), "group": group_of(o), "ck": [],
                                         "k": K_OF.get(o, int(meta["k"]))})
            ent["ck"].append(ck)
            ent.setdefault("sizes", {})[ck] = {str(r): np.bincount(labs[ri][labs[ri] >= 0], minlength=r).tolist() for ri, r in enumerate(R)}
        z.close()
    if worst["example_overlap"] < 8:
        fail(f"stored examples and the reconstructed ordering overlap in only {worst['example_overlap']} of 12")
    out = {"n": N, "ranks": R, "n_vecs": NV, "ctx_len": int(ctx.shape[1]), "ctx_t": CTX_T, "classes": meta["classes"],
           "sources": meta["sources"], "layouts": lnames, "layout_names": {o: SHORT[o] for o in lnames},
           "objects": objects, "order": [o for o in meta["objects"]["trained"]], "obj_layout": layout, "obj_bytes": nbytes,
           "k": meta["k"], "lags": meta["lags"], "validation": worst, "model": meta["model"]}
    (OUT / "meta.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    log(f"mamba: {len(objects)} objects, {tot / 1e6:.1f} MB; validation {worst} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
