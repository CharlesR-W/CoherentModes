"""Figures and gifs for the Overview and Method tabs (matplotlib, real data, fixed axes, palette.json colours).

    uv run --with numpy --with matplotlib --with pillow python -I site/make_figures.py <tag> [--only NAME,...]

Reads data/<tag>/ (schema_version 2, SPEC section 2), site/figdata/<kernel-tag>_kernel.npz (from figdata.py) and,
for the pipeline figure, the float16 Gram matrices written by figdata.py to /var/tmp/cm-figs/.  Writes
site/assets/fig/:

  pipeline.png         how one block is computed, on real matrices (recomputed here from the Grams: Sinkhorn,
                       block average, eigenvectors, SEBA; cross-checked against the stored vectors when present);
                       superseded by steps.png, kept until the Overview switches over
  steps.png, step1..5.png   the Overview's procedure in five steps, same data as pipeline.png: a strip of five
                       numbered panels and one image per step (--only steps)
  static_dynamic.png   psi_1 of a single-cell block vs psi_1 of a layer block containing it, with retention strips
  timeblock.gif/_strip.png   one site through training, coloured by a time block's coherent sets
  hero.png             1200 x 630 preview image: the time block's site at all checkpoints, coloured by its sets
  kappa.png            (Method) kernel weight and cumulative kernel mass against neighbour rank under five
                       single-kappa rules, achieved 15-NN mass per site; all from figdata.py's local recomputation
  rules.png            (Overview) label-free diagnostics from sweep.json against one global kappa, per-image rules marked
  sweep.png            (Method) the full sweep, including class agreement as validation
  seba.png             from eigenfunctions to coherent sets: eigen-coordinates with threshold lines, sparse vectors, colours
  howto.gif/_still.png slow annotated walk through the explorer (a schematic of its layout drawn with real data)
  avgprod.png          (Method) agreement of the average with the product operator over layer pairs (avgprod.json)

Example choices (blocks, sites) are CLI options with defaults set after looking at r20s0b; they are printed.
Nothing here feeds back into the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/var/tmp/cm-figs/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "assets" / "fig"
PAL = json.loads((HERE / "palette.json").read_text())
FG, MUTE, LINE, BG, ACC = PAL["fg"], PAL["mute"], PAL["line"], PAL["bg"], PAL["accent"]
SURF = PAL.get("surface", "#ffffff")
DIV = LinearSegmentedColormap.from_list("div", [PAL["diverging"]["neg"], PAL["diverging"]["mid"], PAL["diverging"]["pos"]])
SEQ = LinearSegmentedColormap.from_list("seq", PAL["sequential"])
GREY = PAL["unassigned"]
RULE_STYLE = [  # colour, dash; rules are not ordered, so ink greys + accent rather than data colours
    (MUTE, (0, (1, 1.5))), (PAL["ordinal"][2], (0, (5, 2))), (ACC, "-"), (PAL["ordinal"][4], "-"), (PAL["ordinal"][7], "-")]
RULE_SHORT = ["kNN scale", "half-mass (heuristic)", "half-mass (calibrated)", "global 5", "global 30"]

from matplotlib import font_manager  # noqa: E402

for _f in sorted((HERE / "figdata" / "fonts").glob("Inter-*.ttf")):  # static instances of the vendored Inter
    font_manager.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "text.color": FG, "axes.labelcolor": FG,
    "axes.edgecolor": LINE, "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlecolor": FG, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "mathtext.fontset": "dejavusans"})


# ----------------------------------------------------------------------------------------------- data access
def procrustes_chain(emb):
    """(T,S,N,2): centre/scale each layout; per site rotate/reflect onto the previous checkpoint (as build_viewer)."""
    out = emb.astype(np.float64).copy()
    out -= out.mean(2, keepdims=True)
    out /= np.sqrt((out ** 2).sum((2, 3), keepdims=True) / out.shape[2])
    for s in range(out.shape[1]):
        for t in range(1, out.shape[0]):
            u, _, vt = np.linalg.svd(out[t, s].T @ out[t - 1, s])
            out[t, s] = out[t, s] @ (u @ vt)
    lo = np.percentile(out, 0.5, axis=2, keepdims=True)
    hi = np.percentile(out, 99.5, axis=2, keepdims=True)
    return np.clip((out - lo) / (hi - lo + 1e-12), -0.02, 1.02)


def seba_reconstruct(v, rot, perm):
    """numpy mirror of coherent.seba_reconstruct (SPEC 2): Z = V R, soft threshold, normalise, sign, max, permute."""
    z = v @ rot
    s = np.sign(z) * np.clip(np.abs(z) - 0.99 / np.sqrt(len(v)), 0, None)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0))
    sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return s[:, perm]


def seba_np(v, r, max_iter=5000, tol=1e-14):
    """numpy port of coherent.seba (reference Algorithm 3.1); returns (clipped vectors, transform, perm)."""
    orig = v[:, :r].astype(np.float64)
    q, tri = np.linalg.qr(orig)
    rng = np.random.default_rng(0)
    for j in range(r):
        if q[:, j].max() - q[:, j].min() < 1e-14:
            q[:, j] += (rng.random(len(q)) - 0.5) * 1e-12
    rot = np.eye(r)
    mu = 0.99 / np.sqrt(len(q))
    for it in range(max_iter):
        z = q @ rot
        s = np.sign(z) * np.clip(np.abs(z) - mu, 0, None)
        s /= np.linalg.norm(s, axis=0)
        u, _, wt = np.linalg.svd(q.T @ s)
        new = u @ wt
        if np.linalg.norm(new - rot) <= tol:
            break
        if it != max_iter - 1:
            rot = new
    transform = np.linalg.solve(tri, rot)
    raw = seba_reconstruct(orig, transform, np.arange(r))
    perm = np.argsort(-raw.min(0), kind="stable")
    return np.clip(raw[:, perm], 0, None), transform, perm


def labels_from(s):
    lab = s.argmax(1)
    lab[s.max(1) <= 0] = -1
    return lab


class Data:
    def __init__(self, tag):
        self.tag = tag
        self.root = ROOT / "data" / tag
        self.meta = json.loads((self.root / "meta.json").read_text())
        m = self.meta
        self.steps, self.sites, self.modes = m["steps"], m["sites"], m["modes"]
        self.T, self.S, self.N = len(self.steps), len(self.sites), m["n"]
        self.seba_r = m["seba_r"]
        self.lp = {tuple(p): i for i, p in enumerate(m["layer_pairs"])}
        self.tp = {tuple(p): i for i, p in enumerate(m["time_pairs"])}
        probe = np.load(self.root / "probe.npz")
        self.labels, self.images = probe["labels"], probe["images"]
        self._z = {}
        self._umap = None
        self.fixture = bool(m.get("fixture", False))
        cur = self.root / "curves.json"
        self.curves = json.loads(cur.read_text())["checkpoints"] if cur.exists() else []

    def blk(self, fam, idx):
        key = (fam, idx)
        if key not in self._z:
            f = f"layer_{self.steps[idx]}.npz" if fam == "layer" else f"time_{self.sites[idx]}.npz"
            z = np.load(self.root / f)
            self._z[key] = {k: z[k] for k in z.files}
        return self._z[key]

    def pidx(self, fam, a, b):
        return (self.lp if fam == "layer" else self.tp)[(a, b)]

    def vals(self, fam, idx, m, a, b):
        return self.blk(fam, idx)["vals"][m, self.pidx(fam, a, b)]

    def psi(self, fam, idx, m, a, b, k):
        z, p = self.blk(fam, idx), self.pidx(fam, a, b)
        q = 32767.0 if z["vecs"].dtype == np.int16 else 127.0
        return z["vecs"][m, p, k - 1].astype(np.float64) * z["scale"][m, p, k - 1] / q

    def retention_psi(self, fam, idx, m, a, b, k):
        return self.blk(fam, idx)["retention_psi"][m, self.pidx(fam, a, b), k - 1]

    def seba(self, fam, idx, m, a, b, r):
        z, p = self.blk(fam, idx), self.pidx(fam, a, b)
        ri = self.seba_r.index(r)
        v = np.column_stack([np.full(self.N, 1 / np.sqrt(self.N))] + [self.psi(fam, idx, m, a, b, k) for k in range(1, r)])
        s = seba_reconstruct(v, z["seba_rot"][m, p, ri, :r, :r].astype(np.float64), z["seba_perm"][m, p, ri, :r])
        return np.clip(s, 0, None), z["seba_label"][m, p, ri].astype(int)

    def umap(self):
        if self._umap is None:
            self._umap = procrustes_chain(np.stack([np.load(self.root / f"umap_{s}.npy") for s in self.steps]))
        return self._umap

    def val_acc(self, step):
        for c in self.curves:
            if c.get("step") == step:
                return c.get("val_acc")
        return None


# ----------------------------------------------------------------------------------------------- drawing helpers
def set_colors(lab):
    sets = np.array([to_rgb(c) for c in PAL["sets"]])
    out = np.tile(np.array(to_rgb(GREY)), (len(lab), 1))
    ok = lab >= 0
    out[ok] = sets[lab[ok] % len(sets)]
    return out


def div_colors(f):
    return DIV(0.5 + 0.5 * f / (np.abs(f).max() + 1e-12))[:, :3]


def class_colors(y):
    return np.array([to_rgb(PAL["classes"][c]) for c in y])


def cell(ax, xy, col, order=None, s=1.6, frame=LINE, lw=0.8):
    """One UMAP cell. order: draw order (grey first by default when colours are set colours)."""
    o = np.arange(len(xy)) if order is None else order
    ax.scatter(xy[o, 0], xy[o, 1], c=col[o], s=s, linewidths=0, rasterized=True)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(frame)
        sp.set_linewidth(lw)


def grey_first(lab):
    return np.concatenate([np.where(lab < 0)[0], np.where(lab >= 0)[0]])


def save(fig, name, dpi=110):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=dpi)
    plt.close(fig)
    print("wrote", OUT / name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def fig_to_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def write_gif(frames, durations, name, colors=96):
    """Shared adaptive palette from a mosaic of frames; no dithering (keeps points crisp and the file small)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pick = frames[:: max(1, len(frames) // 8)]
    w, h = pick[0].size
    mosaic = Image.new("RGB", (w, h * len(pick)))
    for i, f in enumerate(pick):
        mosaic.paste(f, (0, i * h))
    pal = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(OUT / name, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print("wrote", OUT / name, f"{(OUT / name).stat().st_size / 1e6:.2f} MB, {len(frames)} frames, "
          f"{sum(durations) / 1000:.0f} s")


def stepname(s):
    return f"step {s:,}"


# ----------------------------------------------------------------------------------------------- (e) kappa
def fig_kappa(d: Data, kern, t_idx, site):
    k = int(kern["k"])
    ks = list(kern["steps"])
    sites = list(kern["sites"])
    ti = ks.index(d.steps[t_idx]) if d.steps[t_idx] in ks else len(ks) - 1
    si = sites.index(d.sites[site])
    dr = kern["dist_rank"][ti, si].astype(np.float64)
    ranks = np.arange(1, len(dr) + 1)
    kap = kern["kappa"][ti, si]  # local recomputation (figdata.py); its five single-kappa rules, not meta.json's modes
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.32})
    ax = axs[0]
    for m in range(5):
        c, ls = RULE_STYLE[m]
        ax.plot(ranks, np.exp(-kap[m] * (dr - dr[0])), color=c, ls=ls, lw=1.8,
                label=f"{RULE_SHORT[m]} (κ = {kap[m]:.1f})")
    ax.axvline(k, color=MUTE, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(1, len(dr))
    ax.set_xlabel("neighbour rank m (log)")
    ax.set_ylabel("kernel weight, relative to nearest")
    ax.set_title(f"Weight of the m-th neighbour ({d.sites[site]}, {stepname(d.steps[t_idx])})", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6)
    ax.text(k * 1.1, 1.0, "m = 15", color=MUTE, fontsize=8.5, va="top")
    ax = axs[1]
    cr = kern["cum_ranks"]
    for m in range(5):
        c, ls = RULE_STYLE[m]
        ax.plot(cr, kern["cum_mass"][ti, si, m], color=c, ls=ls, lw=1.8)
    ax.axvline(k, color=MUTE, lw=0.8, ls=":")
    ax.axhline(0.5, color=MUTE, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, len(dr))
    ax.set_xlabel("m (log)")
    ax.set_ylabel("share of an image's kernel mass in its m nearest")
    ax.set_title("Cumulative kernel mass (median over images)", loc="left")
    ax = axs[2]
    x = np.arange(d.S)
    for m in range(5):
        c, ls = RULE_STYLE[m]
        ax.plot(x, [kern["nn_mass"][ti, sites.index(d.sites[s]), m] for s in range(d.S)], color=c, ls=ls, lw=1.8,
                marker="o", ms=3.5, label=RULE_SHORT[m])
    ax.axhline(0.5, color=MUTE, lw=0.8, ls=":")
    ax.set_xticks(x, d.sites, rotation=45, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("median 15-NN share of kernel mass")
    ax.set_title(f"Achieved 15-NN mass by site ({stepname(d.steps[t_idx])})", loc="left")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.83)
    save(fig, "kappa.png")
    i500 = min(len(dr), 1000) - 1
    print(f"  kappa cell {d.sites[site]}: d1={dr[0]:.3f} d15={dr[k - 1]:.3f} d1000={dr[i500]:.3f}; "
          f"knn-scale weight at rank 1000 rel. nearest = {np.exp(-kap[0] * (dr[i500] - dr[0])):.2f}, "
          f"rel. 15th = {np.exp(-kap[0] * (dr[i500] - dr[k - 1])):.2f}")


# ----------------------------------------------------------------------------------------------- (a) pipeline
def kernel_np(c, rule, kappa=None, k=15):
    """Zero-diagonal kernel of one cell under a named rule (mirrors sweep_math.kernel_rules / SPEC 2 'Kernel rules')."""
    d = 1.0 - c.astype(np.float64)
    np.fill_diagonal(d, np.inf)
    if "kNN graph" in rule or "union kNN" in rule:
        idx = np.argpartition(d, k, axis=1)[:, :k]
        mask = np.zeros(d.shape, bool)
        np.put_along_axis(mask, idx, True, axis=1)
        mask |= mask.T
        kmat = mask + 1e-6
    elif "perplexity" in rule:
        perp = float(rule.split()[-1])
        ds = d - d.min(1, keepdims=True)
        dm = np.where(np.isfinite(ds), ds, 0.0)
        lo, hi = np.zeros(len(d)), np.ones(len(d))

        def cond(beta):
            lg = -beta[:, None] * ds
            mx = lg.max(1, keepdims=True)
            lz = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
            lp = lg - lz[:, None]
            return lz + beta * (np.exp(lp) * dm).sum(1), lp
        for _ in range(60):
            ent, _ = cond(hi)
            if not (ent > np.log(perp)).any():
                break
            hi[ent > np.log(perp)] *= 2
        for _ in range(50):
            beta = (lo + hi) / 2
            ent, _ = cond(beta)
            lo = np.where(ent > np.log(perp), beta, lo)
            hi = np.where(ent > np.log(perp), hi, beta)
        _, lp = cond((lo + hi) / 2)
        pc = np.exp(lp)
        kmat = (pc + pc.T) / 2
    else:
        kmat = np.exp(-kappa * d)
    np.fill_diagonal(kmat, 0.0)
    return kmat


def sinkhorn_np(kmat, tol=1e-10):
    """SPEC 2: symmetric Sinkhorn of a zero-diagonal kernel to doubly stochastic P; returns M = (I + P)/2."""
    v = 1 / np.sqrt(kmat.sum(1))
    for _ in range(400000):
        kv = kmat @ v
        if np.abs(v * kv - 1).max() < tol:
            break
        v = np.sqrt(v / kv)
    else:
        raise RuntimeError("Sinkhorn did not converge")
    p = v[:, None] * kmat * v[None, :]
    return 0.5 * (np.eye(len(kmat)) + p)


def blockmean(a, b=8):
    n = a.shape[0] // b * b
    return a[:n, :n].reshape(n // b, b, n // b, b).mean((1, 3))


def pipeline_data(d: Data, grams_path, t_idx, a, b, mode, r):
    """Recompute one block from the Grams (kernel, Sinkhorn, average, eigenvectors, SEBA) for the pipeline figures."""
    grams = np.load(grams_path, mmap_mode="r")
    rule = d.modes[mode]
    kap = [d.meta["kappa"][mode][t_idx][s] for s in range(a, b + 1)]
    ms, kmats = [], []
    for s, kv in zip(range(a, b + 1), kap):
        km = kernel_np(np.asarray(grams[s], dtype=np.float64), rule, kv)
        ms.append(sinkhorn_np(km))
        kmats.append(km)
    mbar = sum(ms) / len(ms)
    w, v = np.linalg.eigh(mbar)
    w, v = w[::-1][:9], v[:, ::-1][:, :9]
    v *= np.sign(v[np.abs(v).argmax(0), np.arange(v.shape[1])])
    v[:, 0] = 1 / np.sqrt(d.N)
    s_vec, _, _ = seba_np(v, r)
    lab = labels_from(s_vec)
    if not d.fixture:  # cross-check against the stored measurement
        stored = np.column_stack([d.psi("layer", t_idx, mode, a, b, k) for k in range(1, r)])
        qa, _ = np.linalg.qr(v[:, 1:r])
        qb, _ = np.linalg.qr(stored)
        agree = float(np.square(qa.T @ qb).sum() / (r - 1))
        _, slab = d.seba("layer", t_idx, mode, a, b, r)
        print(f"  pipeline cross-check: span agreement with stored psi_1..psi_{r - 1} = {agree:.4f}; "
              f"SEBA label agreement = {(slab == lab).mean():.4f}; eigenvalues here {np.round(w[:5], 4)} "
              f"stored {np.round(d.vals('layer', t_idx, mode, a, b)[:5], 4)}")
    # order images: by set, then by decreasing membership; unassigned last, by psi_1
    keys = []
    for j in range(r):
        idx = np.where(lab == j)[0]
        keys.append(idx[np.argsort(-s_vec[idx, j])])
    un = np.where(lab < 0)[0]
    keys.append(un[np.argsort(v[un, 1])])
    order = np.concatenate(keys)
    bounds = np.cumsum([len(x) for x in keys])[:-1]
    mid = a + (b - a) // 2
    cmid = np.asarray(grams[mid], dtype=np.float64)
    return dict(rule=rule, kap=kap, ms=ms, kmats=kmats, mbar=mbar, w=w, v=v, s_vec=s_vec, lab=lab, order=order,
                bounds=bounds, mid=mid, cmid=cmid, t_idx=t_idx, a=a, b=b, mode=mode, r=r)


def fig_pipeline(d: Data, ctx):
    """The original two-row pipeline figure (pipeline.png), kept until the Overview switches to the step strip."""
    g = ctx
    rule, kap, ms, kmats, mbar, w, v = g["rule"], g["kap"], g["ms"], g["kmats"], g["mbar"], g["w"], g["v"]
    s_vec, lab, order, bounds, mid, cmid = g["s_vec"], g["lab"], g["order"], g["bounds"], g["mid"], g["cmid"]
    t_idx, a, b, r = g["t_idx"], g["a"], g["b"], g["r"]

    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.42, left=0.03, right=0.965, top=0.9, bottom=0.07)
    top = gs[0].subgridspec(1, 4, wspace=0.42, width_ratios=[1, 1, 1, 1.25])
    bot = gs[1].subgridspec(1, 5, wspace=0.36, width_ratios=[1.05, 1, 1.25, 1.25, 1.1])

    def mat(ax, m, title, sub, norm=None, cmap=SEQ, vmin=None, vmax=None):
        img = blockmean(m[np.ix_(order, order)])
        im = ax.imshow(img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for bd in bounds:
            ax.axhline(bd / 8 - 0.5, color=SURF, lw=0.5, alpha=0.8)
            ax.axvline(bd / 8 - 0.5, color=SURF, lw=0.5, alpha=0.8)
        ax.set_title(title, loc="left")
        ax.text(0, -0.06, sub, transform=ax.transAxes, va="top", fontsize=8.6, color=MUTE)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.outline.set_visible(False)
        if isinstance(norm, LogNorm):
            from matplotlib.ticker import LogLocator
            cb.ax.yaxis.set_major_locator(LogLocator(numticks=4))
            cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        cb.ax.tick_params(labelsize=7.5, colors=MUTE)
        return ax

    # top row: one site
    ax0 = fig.add_subplot(top[0])
    h = d.images[order[:: max(1, d.N // 64)]][:36]
    sheet = np.concatenate([np.concatenate(list(h[i * 6:(i + 1) * 6]), 1) for i in range(6)], 0)
    ax0.imshow(sheet)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("2,000 probe images", loc="left")
    ax0.text(0, -0.06, f"activations at site {d.sites[mid]}: centre over images,\nscale each to unit length",
             transform=ax0.transAxes, va="top", fontsize=8.6, color=MUTE)
    lim = np.percentile(np.abs(cmid), 99.5)
    mat(fig.add_subplot(top[1]), cmid, f"cosine Gram C ({d.sites[mid]})", "C_ij = cos(h_i, h_j)", cmap=DIV, vmin=-lim, vmax=lim)
    km = kmats[mid - a]
    kpos = km[km > 0]
    ktitle = ("kernel K: 15-NN graph" if "kNN graph" in rule else "kernel K (per-image perplexity)"
              if "perplexity" in rule else "kernel K = exp(κ(C − 1))")
    ksub = (f"rule: {rule}; K_ii = 0; log colour scale" if kap[mid - a] is None else
            f"median κ_i = {kap[mid - a]:.1f} ({rule}); log colour" if "perplexity" in rule else
            f"κ = {kap[mid - a]:.1f} ({rule}); log colour scale")
    mat(fig.add_subplot(top[2]), km, ktitle, ksub,
        norm=LogNorm(vmin=np.percentile(kpos, 50), vmax=np.percentile(kpos, 99.99)))
    mm = ms[mid - a]
    offd = mm - np.diag(np.diag(mm))
    pos = offd[offd > 0]
    mnorm = LogNorm(vmin=np.percentile(pos, 50), vmax=np.percentile(pos, 99.99))
    mat(fig.add_subplot(top[3]), offd, "walk M = (I + P)/2",
        "P = Sinkhorn(K): rows, columns sum to 1;\noff-diagonal shown, log colour scale", norm=mnorm)
    # bottom row: block
    sub = bot[0].subgridspec(2, 2, wspace=0.08, hspace=0.25)
    cells = list(range(a, b + 1))
    pick = cells if len(cells) <= 4 else [cells[i] for i in np.linspace(0, len(cells) - 1, 4).round().astype(int)]
    for i, s in enumerate(pick):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        ms_off = ms[s - a] - np.diag(np.diag(ms[s - a]))
        img = blockmean(ms_off[np.ix_(order, order)])
        ax.imshow(img, cmap=SEQ, norm=mnorm, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"M ({d.sites[s]})", fontsize=8.8, loc="left", pad=2)
    fig.text(0.03, 0.47, f"every site of the block {d.sites[a]}–{d.sites[b]}, {stepname(d.steps[t_idx])}",
             fontsize=10.5, weight="semibold")
    mat(fig.add_subplot(bot[1]), mbar - np.diag(np.diag(mbar)), "average M̄", f"mean of {len(cells)} walks", norm=mnorm)
    ax = fig.add_subplot(bot[2])
    xs = np.arange(d.N)
    for k in range(1, 4):
        ax.plot(xs, v[order, k] * np.sqrt(d.N) + (3 - k) * 4.5, color=FG if k == 1 else MUTE, lw=0.6)
        ax.text(d.N * 1.01, (3 - k) * 4.5, f"ψ{k}\nλ={w[k]:.3f}", fontsize=8, color=MUTE, va="center")
    for bd in bounds:
        ax.axvline(bd, color=LINE, lw=0.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, d.N)
    ax.spines["left"].set_visible(False)
    ax.set_title("leading eigenvectors of M̄", loc="left")
    ax.text(0, -0.06, "images in the same order as the matrices", transform=ax.transAxes, va="top", fontsize=8.6, color=MUTE)
    ax = fig.add_subplot(bot[3])
    setc = PAL["sets"]
    for j in range(r):
        ax.fill_between(xs, (r - 1 - j) * 1.15, (r - 1 - j) * 1.15 + s_vec[order, j], color=setc[j], lw=0, step="mid")
        ax.text(d.N * 1.01, (r - 1 - j) * 1.15 + 0.4, f"set {j + 1}", fontsize=8, color=MUTE, va="center")
    for bd in bounds:
        ax.axvline(bd, color=LINE, lw=0.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, d.N)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"SEBA: {r} sparse vectors from ψ0…ψ{r - 1}", loc="left")
    ax.text(0, -0.06, "argmax per image gives its coherent set; grey (unassigned) if all are 0", transform=ax.transAxes, va="top",
            fontsize=8.6, color=MUTE)
    ax = fig.add_subplot(bot[4])
    xy = d.umap()[t_idx, b]
    cell(ax, xy, set_colors(lab), grey_first(lab), s=2.2)
    ax.set_title(f"painted on the UMAP of {d.sites[b]}", loc="left")
    ax.text(0, -0.06, "UMAP is display only", transform=ax.transAxes, va="top", fontsize=8.6, color=MUTE)
    fig.text(0.03, 0.965, "How a block is computed, on real data", fontsize=13, weight="semibold")
    fig.text(0.03, 0.935, "Matrices are 2,000 × 2,000, block-averaged 8 × 8 for display; images ordered by the "
             "coherent set they end up in (the white lines).", fontsize=9.5, color=MUTE)
    for x0, x1, y in [(0.255, 0.275, 0.73), (0.49, 0.51, 0.73), (0.72, 0.74, 0.73), (0.215, 0.235, 0.27),
                      (0.39, 0.41, 0.27), (0.6, 0.62, 0.27), (0.8, 0.82, 0.27)]:
        fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=12, color=MUTE, lw=1.2))
    save(fig, "pipeline.png", dpi=105)
    return lab


# ----------------------------------------------------------------------------------------------- (a2) five-step strip
STEP_W = [1.0, 1.75, 1.85, 1.3, 1.9]  # relative panel widths of steps 1..5
SMALL = 13  # secondary text in the step figures
STEP_RC = {"font.size": 14, "axes.titlesize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12}


def _step_head(fig, spec, k, title, sub=None, inner=False):
    """Number badge, title and optional subtitle above the panel occupying `spec` (offsets in points)."""
    from matplotlib.transforms import offset_copy
    bb = spec.get_position(fig)
    tr = lambda dy: offset_copy(fig.transFigure, fig=fig, x=0, y=dy, units="points")  # noqa: E731
    y0 = 24  # leave room for the panels' own sub-titles (same height for every step)
    fig.text(bb.x0, bb.y1, f" {k} ", transform=tr(y0 + 26), fontsize=17, weight="bold", color=SURF,
             va="bottom", bbox=dict(boxstyle="circle,pad=0.18", fc=ACC, ec="none"))
    fig.text(bb.x0, bb.y1, "       " + title, transform=tr(y0 + 28), fontsize=17, weight="semibold",
             va="bottom", color=FG)
    if sub:
        fig.text(bb.x0, bb.y1, sub, transform=tr(y0 + 4), fontsize=13, va="bottom", color=MUTE)


def _set_bar(fig, ax, lab_sorted, side="left", w=0.045):
    """Thin bar beside a matrix showing each (8-image) row's coherent set; the images' display order."""
    pos = ax.get_position()
    cols = set_colors(lab_sorted[::8][: (len(lab_sorted) // 8)])
    if side == "left":
        bx = fig.add_axes([pos.x0 - pos.width * (w + 0.02), pos.y0, pos.width * w, pos.height])
    else:
        bx = fig.add_axes([pos.x0, pos.y1 + pos.height * 0.02, pos.width, pos.height * w])
    bx.imshow(cols[:, None, :] if side == "left" else cols[None, :, :], aspect="auto", interpolation="nearest")
    bx.set_axis_off()


def _mat(ax, img, bounds, **kw):
    ax.imshow(img, interpolation="nearest", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(LINE)
    for bd in bounds:
        ax.axhline(bd / 8 - 0.5, color=SURF, lw=0.7, alpha=0.9)
        ax.axvline(bd / 8 - 0.5, color=SURF, lw=0.7, alpha=0.9)


def _norms(g):
    km = g["kmats"][g["mid"] - g["a"]]
    kpos = km[km > 0]
    p = 2 * g["ms"][g["mid"] - g["a"]] - np.eye(len(km))
    ppos = p[p > 0]
    off = [m - np.diag(np.diag(m)) for m in g["ms"]]
    mpos = off[g["mid"] - g["a"]]
    mpos = mpos[mpos > 0]
    return (LogNorm(vmin=np.percentile(kpos, 50), vmax=np.percentile(kpos, 99.99)),
            LogNorm(vmin=np.percentile(ppos, 50), vmax=np.percentile(ppos, 99.99)),
            LogNorm(vmin=np.percentile(mpos, 50), vmax=np.percentile(mpos, 99.99)), p, off)


def _step1(fig, spec, d, g):
    order, lab = g["order"], g["lab"]
    ax = fig.add_subplot(spec)
    lim = np.percentile(np.abs(g["cmid"]), 99.5)
    _mat(ax, blockmean(g["cmid"][np.ix_(order, order)]), g["bounds"], cmap=DIV, vmin=-lim, vmax=lim)
    ax.text(0, -0.04, "red > 0 > blue", transform=ax.transAxes, va="top", fontsize=SMALL, color=MUTE)
    _step_head(fig, spec, 1, "Cosine Gram",
               f"$C_{{ij}} = \\cos(h_i, h_j)$; {d.sites[g['mid']]}, {stepname(d.steps[g['t_idx']])}")
    return [ax]


def _step2(fig, spec, d, g):
    order = g["order"]
    knorm, pnorm, _, p, _ = _norms(g)
    sub = spec.subgridspec(1, 2, wspace=0.28)
    ka, pa = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    km = g["kmats"][g["mid"] - g["a"]]
    _mat(ka, blockmean(km[np.ix_(order, order)]), g["bounds"], cmap=SEQ, norm=knorm)
    _mat(pa, blockmean((p - np.diag(np.diag(p)))[np.ix_(order, order)]), g["bounds"], cmap=SEQ, norm=pnorm)
    rule = g["rule"].replace("kNN graph, k = 15", "15-NN graph")
    ka.set_title(f"kernel $K_c$ ({rule})", loc="left", fontsize=13, weight="normal", pad=5)
    pa.set_title("walk $P_c$ = Sinkhorn($K_c$)", loc="left", fontsize=13, weight="normal", pad=5)
    pa.text(0, -0.04, "rows, columns sum to 1", transform=pa.transAxes, va="top", fontsize=SMALL, color=MUTE)
    ka.text(0, -0.04, "log colour", transform=ka.transAxes, va="top", fontsize=SMALL, color=MUTE)
    _step_head(fig, spec, 2, "Kernel, then walk", "same cell; lazy walk $M_c = (I + P_c)/2$", inner=True)
    return [ka, pa]


def _step3(fig, spec, d, g):
    order, a, b = g["order"], g["a"], g["b"]
    _, _, mnorm, _, off = _norms(g)
    sub = spec.subgridspec(1, 2, width_ratios=[0.85, 1], wspace=0.32)
    cells = list(range(a, b + 1))
    pick = cells if len(cells) <= 4 else [cells[i] for i in np.linspace(0, len(cells) - 1, 4).round().astype(int)]
    small = sub[0].subgridspec(2, 2, wspace=0.12, hspace=0.3)
    axs = []
    for i, s in enumerate(pick):
        ax = fig.add_subplot(small[i // 2, i % 2])
        _mat(ax, blockmean(off[s - a][np.ix_(order, order)]), [], cmap=SEQ, norm=mnorm)
        ax.set_title(f"$M_c$: {d.sites[s]}", loc="left", fontsize=SMALL, weight="normal", pad=3)
        axs.append(ax)
    ax = fig.add_subplot(sub[1])
    mb = g["mbar"]
    _mat(ax, blockmean((mb - np.diag(np.diag(mb)))[np.ix_(order, order)]), g["bounds"], cmap=SEQ, norm=mnorm)
    ax.set_title("$\\bar M$ = mean of the $M_c$", loc="left", fontsize=13, weight="normal", pad=5)
    ax.text(0, -0.04, "off-diagonal, log colour", transform=ax.transAxes, va="top", fontsize=SMALL, color=MUTE)
    _step_head(fig, spec, 3, "Average over a block",
               f"layer block {d.sites[a]}–{d.sites[b]}, {stepname(d.steps[g['t_idx']])}", inner=True)
    return axs + [ax]


def _step4(fig, spec, d, g):
    v, w, order, ms, a, b, r = g["v"], g["w"], g["order"], g["ms"], g["a"], g["b"], g["r"]
    sub = spec.subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.3)
    ax = fig.add_subplot(sub[0])
    n = len(order)
    xs = np.arange(n)
    for k in range(1, r):
        off = (r - 1 - k) * 4.2
        ax.plot(xs, v[order, k] * np.sqrt(n) + off, color=FG, lw=0.7)
        ax.text(-n * 0.02, off, f"$\\psi_{k}$", ha="right", va="center", fontsize=14)
    for bd in g["bounds"]:
        ax.axvline(bd, color=LINE, lw=0.9)
    ax.set_xlim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    # retention table: rho_c(psi_k) = psi_k^T M_c psi_k for each cell of the block, and lambda_k = their mean
    ret = np.array([[v[:, k] @ ms[c] @ v[:, k] for c in range(len(ms))] for k in range(1, r)])
    tab = np.column_stack([ret, w[1:r]])
    bx = fig.add_subplot(sub[1])
    nc = ret.shape[1]
    img = SEQ(np.clip((tab - 0.5) / 0.5, 0, 1))
    img[:, nc] = to_rgb(BG) + (1,)
    bx.imshow(img, aspect="auto", interpolation="nearest")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            dark = j < nc and (tab[i, j] - 0.5) / 0.5 > 0.55
            bx.text(j, i, f"{tab[i, j]:.2f}", ha="center", va="center", fontsize=SMALL,
                    color=SURF if dark else FG, weight="semibold" if j == nc else "normal")
    bx.axvline(nc - 0.5, color=SURF, lw=3)
    bx.set_xticks(range(nc + 1), [d.sites[c] for c in range(a, b + 1)] + ["$\\lambda_k$"], fontsize=SMALL)
    bx.set_yticks(range(r - 1), [f"$\\psi_{k}$" for k in range(1, r)], fontsize=13)
    bx.tick_params(length=0)
    bx.xaxis.tick_top()
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_xlabel("retention $\\rho_c(\\psi_k) = \\psi_k^\\top M_c \\psi_k$;  $\\lambda_k$ = mean", fontsize=SMALL,
                  color=MUTE)
    _step_head(fig, spec, 4, "Eigenfunctions", "leading eigenvectors of $\\bar M$")
    return [ax, bx], ret


def _step5(fig, spec, d, g):
    s_vec, lab, order, r, t_idx, b = g["s_vec"], g["lab"], g["order"], g["r"], g["t_idx"], g["b"]
    sub = spec.subgridspec(1, 2, width_ratios=[1, 1], wspace=0.34)
    ax = fig.add_subplot(sub[0])
    n = len(order)
    xs = np.arange(n)
    nun = int((lab < 0).sum())
    for j in range(r):
        base = (r - 1 - j) * 1.2
        ax.fill_between(xs, base, base + s_vec[order, j], color=PAL["sets"][j], lw=0, step="mid")
        ax.text(-n * 0.03, base + 0.35, f"{j + 1}", ha="right", va="center", fontsize=14, color=FG)
    ax.axvspan(n - nun, n, color=GREY, alpha=0.45, lw=0)
    ax.text(n - nun / 2, -0.15, "unassigned", ha="right", va="top", fontsize=SMALL, color=MUTE)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.75, r * 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"SEBA vectors $s_1,\\ldots,s_{r}$", loc="left", fontsize=13, weight="normal", pad=5)
    ax.set_xlabel("image (same order)", fontsize=SMALL, color=MUTE)
    bx = fig.add_subplot(sub[1])
    cell(bx, d.umap()[t_idx, b], set_colors(lab), grey_first(lab), s=4.5)
    bx.set_title(f"coherent sets (layout: {d.sites[b]})", loc="left", fontsize=13, weight="normal", pad=5)
    bx.text(0, -0.04, "set = largest $s_j$; grey: all 0", transform=bx.transAxes, va="top", fontsize=SMALL,
            color=MUTE)
    _step_head(fig, spec, 5, "Sparse vectors, coherent sets", f"SEBA on $\\psi_0,\\ldots,\\psi_{r - 1}$ (r = {r})",
               inner=True)
    return [ax, bx]


STEP_DRAW = [_step1, _step2, _step3, _step4, _step5]


def fig_steps(d: Data, g):
    """The Overview's five-step strip (steps.png) and one image per step (step1.png .. step5.png), same data as
    pipeline.png: images in every matrix and trace are ordered by the coherent set they end up in."""
    with plt.rc_context(STEP_RC):
        unit, H = 2.9, 4.7
        gap = 0.62  # inches between panels
        Wt = unit * sum(STEP_W) + gap * 4 + 1.1
        fig = plt.figure(figsize=(Wt, H))
        lefts, x = [], 0.75
        for wd in STEP_W:
            lefts.append(x)
            x += unit * wd + gap
        specs = []
        for k in range(5):
            gs = fig.add_gridspec(1, 1, left=lefts[k] / Wt, right=(lefts[k] + unit * STEP_W[k]) / Wt, top=0.7,
                                  bottom=0.17)
            specs.append(gs[0])
            out = STEP_DRAW[k](fig, gs[0], d, g)
            if k == 3:
                out, ret = out
            if k == 0:
                _set_bar(fig, out[0], g["lab"][g["order"]])
        for k in range(4):
            x0 = (lefts[k] + unit * STEP_W[k] + 0.08) / Wt
            x1 = (lefts[k + 1] - 0.22) / Wt
            fig.patches.append(FancyArrowPatch((x0, 0.44), (x1, 0.44), transform=fig.transFigure, arrowstyle="-|>",
                                               mutation_scale=22, color=MUTE, lw=1.8))
        fig.text(0.75 / Wt, 0.02, "Matrices are 2,000 × 2,000, shown as means of 8 × 8 tiles. In every matrix and "
                 "trace the images are ordered by the coherent set they end up in (colour bar in step 1; white lines).",
                 fontsize=SMALL, color=MUTE)
        save(fig, "steps.png", dpi=100)
        for k in range(5):
            wk = unit * STEP_W[k] + 1.05
            fig = plt.figure(figsize=(wk, H))
            gs = fig.add_gridspec(1, 1, left=0.75 / wk, right=1 - 0.3 / wk, top=0.7, bottom=0.14)
            out = STEP_DRAW[k](fig, gs[0], d, g)
            if k == 3:
                out = out[0]
            if k == 0:
                _set_bar(fig, out[0], g["lab"][g["order"]])
            save(fig, f"step{k + 1}.png", dpi=150)
    a, b, r = g["a"], g["b"], g["r"]
    if not d.fixture:
        stored = np.array([d.retention_psi("layer", g["t_idx"], g["mode"], a, b, k)[a:b + 1] for k in range(1, r)])
        print(f"  steps: retention recomputed vs stored, max |diff| = {np.abs(stored - ret).max():.2e}")
    lab = g["lab"]
    print(f"  steps: block {d.sites[a]}-{d.sites[b]}, {d.steps[g['t_idx']]}, rule {g['rule']}, r={r}; "
          f"lambda_1..{r - 1} = {np.round(g['w'][1:r], 4)}, gap after lambda_{r - 1}: {g['w'][r - 1] - g['w'][r]:.4f}; "
          f"retention per cell (rows psi_k, cols sites):\n{np.round(ret, 3)}\n"
          f"  set sizes {[int((lab == j).sum()) for j in range(r)]}, unassigned {int((lab < 0).sum())}")


# ----------------------------------------------------------------------------------------------- (c) static vs dynamic
def pick_static_dynamic(d: Data, t_idx, mode, minw=3):
    """Block (a,b) and site s in it maximising (block lambda_1) - (mean retention over the block of s's static psi_1)."""
    best = None
    for (a, b), _ in d.lp.items():
        if b - a + 1 < minw:
            continue
        lam = d.vals("layer", t_idx, mode, a, b)[1]
        for s in range(a, b + 1):
            st = d.retention_psi("layer", t_idx, mode, s, s, 1)[a:b + 1].mean()
            gap = lam - st
            if best is None or gap > best[0]:
                best = (gap, a, b, s)
    return best


def fig_static_dynamic(d: Data, t_idx, mode, a, b, s):
    fs = d.psi("layer", t_idx, mode, s, s, 1)
    fb = d.psi("layer", t_idx, mode, a, b, 1)
    rs = d.retention_psi("layer", t_idx, mode, s, s, 1)
    rb = d.retention_psi("layer", t_idx, mode, a, b, 1)
    cells = list(range(a, b + 1))
    if len(cells) > 5:
        cells = sorted(set([a, s, b] + [cells[i] for i in np.linspace(0, len(cells) - 1, 5).round().astype(int)]))[:6]
    nc = len(cells)
    fig = plt.figure(figsize=(2.15 * nc + 4.8, 5.3))
    gs = fig.add_gridspec(2, nc + 2, width_ratios=[1] * nc + [0.35, 2.2], wspace=0.08, hspace=0.22, left=0.07,
                          right=0.985, top=0.86, bottom=0.11)
    xy = d.umap()[t_idx]
    for row, (f, name) in enumerate([(fs, f"single-cell block {d.sites[s]}: ψ1"),
                                     (fb, f"layer block {d.sites[a]}–{d.sites[b]}: ψ1")]):
        col = div_colors(f)
        order = np.argsort(np.abs(f))
        for i, c in enumerate(cells):
            ax = fig.add_subplot(gs[row, i])
            cell(ax, xy[c], col, order, s=1.5, frame=ACC if (row == 0 and c == s) else LINE, lw=1.6 if (row == 0 and c == s) else 0.8)
            if row == 0:
                ax.set_title(d.sites[c], fontsize=9.5, color=MUTE, weight="normal")
            if i == 0:
                ax.set_ylabel(name, fontsize=9.5, color=FG)
    ax = fig.add_subplot(gs[:, nc + 1])
    x = np.arange(d.S)
    ax.axvspan(a - 0.5, b + 0.5, color=LINE, alpha=0.7, lw=0)
    ax.plot(x, rs, color=MUTE, marker="o", ms=4, lw=1.6, label=f"ψ1 of single-cell block {d.sites[s]}")
    ax.plot(x, rb, color=ACC, marker="o", ms=4, lw=1.8, label=f"ψ1 of layer block {d.sites[a]}–{d.sites[b]}")
    lam = d.vals("layer", t_idx, mode, a, b)[1]
    ax.hlines(lam, a - 0.5, b + 0.5, color=ACC, lw=0.9, ls="--")
    ax.text(a + 0.6, lam + 0.02, f"λ1 = {lam:.3f}: mean of ρ_c over the block", color=ACC, fontsize=8,
            va="bottom")
    ax.hlines(rs[a:b + 1].mean(), a - 0.5, b + 0.5, color=MUTE, lw=0.9, ls="--")
    ax.text(a - 0.4, rs[a:b + 1].mean() - 0.012, f"mean over the block: {rs[a:b + 1].mean():.3f}", color=MUTE,
            fontsize=8, va="top")
    ax.set_xticks(x, d.sites, rotation=45, fontsize=8.5)
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("retention ρ_c(ψ1) = ψ1ᵀ M_c ψ1")
    ax.set_title(f"Retention at every site, {stepname(d.steps[t_idx])}", loc="left")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.text(0.07, 0.95, f"ψ1 of a single-cell block vs ψ1 of a layer block ({stepname(d.steps[t_idx])}, "
             f"kernel rule: {d.modes[mode]})",
             fontsize=12, weight="semibold")
    save(fig, "static_dynamic.png")
    print(f"  static/dynamic: block {d.sites[a]}-{d.sites[b]}, site {d.sites[s]}; block lambda1 {lam:.4f}; "
          f"static retention in block mean {rs[a:b + 1].mean():.4f}, at own site {rs[s]:.4f}; "
          f"block retention min/max {rb[a:b + 1].min():.4f}/{rb[a:b + 1].max():.4f}; "
          f"|corr(static, block)| = {abs(np.corrcoef(fs, fb)[0, 1]):.3f}")


# ----------------------------------------------------------------------------------------------- (d) time block
def fig_timeblock(d: Data, site, c, dd, mode, r):
    _, lab = d.seba("time", site, mode, c, dd, r)
    col = set_colors(lab)
    order = grey_first(lab)
    U = d.umap()[:, site]
    fig, axs = plt.subplots(1, d.T, figsize=(1.75 * d.T, 2.55))
    for t in range(d.T):
        inb = c <= t <= dd
        cell(axs[t], U[t], col, order, s=1.0, frame=ACC if inb else LINE, lw=1.6 if inb else 0.8)
        acc = d.val_acc(d.steps[t])
        axs[t].set_title(f"{d.steps[t]:,}" + (f"\nval acc {100 * acc:.0f}%" if acc is not None else ""), fontsize=8.5,
                         color=FG if inb else MUTE, weight="normal")
    fig.text(0.01, 0.97, f"Site {d.sites[site]} at each checkpoint, coloured by the coherent sets (r = {r}) of the "
             f"time block steps {d.steps[c]:,}–{d.steps[dd]:,} (outlined)", fontsize=10, weight="semibold", va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.02, wspace=0.06)
    save(fig, "timeblock_strip.png")
    # gif: one large cell, layout interpolated between checkpoints (display only)
    frames, durs = [], []
    ret = d.blk("time", site)["retention_seba"][mode, d.pidx("time", c, dd), d.seba_r.index(r), :r]
    for t in range(d.T):
        nxt = min(t + 1, d.T - 1)
        sub = [0.0] if t == d.T - 1 else [0.0, 0.2, 0.4, 0.6, 0.8]
        for u in sub:
            e = u * u * (3 - 2 * u)
            xy = (1 - e) * U[t] + e * U[nxt]
            fig = plt.figure(figsize=(5.6, 6.2), dpi=90)
            ax = fig.add_axes([0.04, 0.2, 0.92, 0.7])
            inb = c <= t <= dd and u == 0
            cell(ax, xy, col, order, s=3.2, frame=ACC if inb else LINE, lw=2 if inb else 1)
            acc = d.val_acc(d.steps[t])
            fig.text(0.04, 0.955, f"{d.sites[site]} at step {d.steps[t]:,}" + (" → …" if u > 0 else "") +
                     (f"   (val acc {100 * acc:.0f}%)" if acc is not None and u == 0 else ""), fontsize=12,
                     weight="semibold")
            fig.text(0.04, 0.915, f"colours: coherent sets (r = {r}) of the time block steps {d.steps[c]:,}–"
                     f"{d.steps[dd]:,}, fixed throughout",
                     fontsize=9, color=MUTE)
            bx = fig.add_axes([0.08, 0.06, 0.84, 0.08])
            bx.set_xlim(-0.5, d.T - 0.5)
            bx.set_ylim(0, 1)
            bx.axvspan(c - 0.5, dd + 0.5, color=ACC, alpha=0.15, lw=0)
            bx.plot([t + e], [0.5], marker="v", color=FG, ms=9)
            bx.set_yticks([])
            bx.set_xticks(range(d.T), [f"{s:,}" for s in d.steps], fontsize=7.5)
            bx.spines["left"].set_visible(False)
            bx.set_title("checkpoint (shaded: the time block)", fontsize=8, color=MUTE, loc="left", weight="normal", pad=2)
            frames.append(fig_to_image(fig))
            plt.close(fig)
            durs.append(1700 if u == 0 else 90)
    durs[-1] = 3000
    write_gif(frames, durs, "timeblock.gif", colors=64)
    sizes = [(lab == j).sum() for j in range(r)]
    print(f"  timeblock: site {d.sites[site]}, block {d.steps[c]}-{d.steps[dd]}, r={r}, set sizes {sizes}, "
          f"unassigned {(lab < 0).sum()}; SEBA retention per set over checkpoints:\n{np.round(ret, 3)}")


def fig_hero(d: Data, site, c, dd, mode, r):
    """1200 x 630 social-preview image: one site at every checkpoint (2 x 4), coloured by a time block's sets."""
    _, lab = d.seba("time", site, mode, c, dd, r)
    col, order = set_colors(lab), grey_first(lab)
    U = d.umap()[:, site]
    fig = plt.figure(figsize=(12, 6.3))
    nc = (d.T + 1) // 2
    gs = fig.add_gridspec(2, nc, left=0.02, right=0.98, top=0.755, bottom=0.015, wspace=0.05, hspace=0.27)
    for t in range(d.T):
        ax = fig.add_subplot(gs[t // nc, t % nc])
        inb = c <= t <= dd
        cell(ax, U[t], col, order, s=3.2, frame=ACC if inb else LINE, lw=2.2 if inb else 1)
        ax.set_title(f"step {d.steps[t]:,}", fontsize=15, color=FG if inb else MUTE, weight="normal", pad=4)
    fig.text(0.02, 0.965, f"One layer ({d.sites[site]}) of a ResNet-20 at {d.T} checkpoints in training",
             fontsize=21, weight="semibold", va="top")
    fig.text(0.02, 0.9, f"Each dot is one of the same {d.N:,} test images in every panel. Colour: its coherent set "
             f"(r = {r}) in the time block\nsteps {d.steps[c]:,}–{d.steps[dd]:,} (outlined); grey: unassigned. "
             "Layouts (UMAP) are for display only.", fontsize=13.5, color=MUTE, va="top", linespacing=1.4)
    save(fig, "hero.png", dpi=100)


# ----------------------------------------------------------------------------------------------- (f) SEBA explainer
def fig_seba(d: Data, t_idx, mode, a, b):
    r = 3
    p1, p2 = d.psi("layer", t_idx, mode, a, b, 1), d.psi("layer", t_idx, mode, a, b, 2)
    s_vec, lab = d.seba("layer", t_idx, mode, a, b, r)
    z, p = d.blk("layer", t_idx), d.pidx("layer", a, b)
    ri = d.seba_r.index(r)
    rot = z["seba_rot"][mode, p, ri, :r, :r].astype(np.float64)
    perm = z["seba_perm"][mode, p, ri, :r]
    n = d.N
    mu = 0.99 / np.sqrt(n)
    fig = plt.figure(figsize=(14.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 1], wspace=0.25, left=0.05, right=0.985, top=0.83, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    o = grey_first(lab)
    ax.scatter(p1[o] * np.sqrt(n), p2[o] * np.sqrt(n), c=set_colors(lab)[o], s=3, linewidths=0, rasterized=True)
    # threshold lines: column j of V R is c0/sqrt(n) + c1 psi1 + c2 psi2; |.| = mu, drawn in sqrt(n)-scaled coords
    xl = np.array([p1.min(), p1.max()]) * np.sqrt(n) * 1.1
    yl = np.array([p2.min(), p2.max()]) * np.sqrt(n) * 1.1
    gx, gy = np.meshgrid(np.linspace(*xl, 300), np.linspace(*yl, 300))
    for jj in range(r):
        j = perm[jj]
        c0, c1, c2 = rot[:, j]
        zz = c0 / np.sqrt(n) + (c1 * gx + c2 * gy) / np.sqrt(n)
        zi = (c0 + c1 * p1 * np.sqrt(n) + c2 * p2 * np.sqrt(n)) / np.sqrt(n)
        sg = np.sign((np.sign(zi) * np.clip(np.abs(zi) - mu, 0, None)).sum()) or 1.0
        ax.contour(gx, gy, sg * zz, levels=[mu], colors=[PAL["sets"][jj]], linewidths=1.3, linestyles="--")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.axhline(0, color=LINE, lw=0.8)
    ax.axvline(0, color=LINE, lw=0.8)
    ax.set_xlabel("ψ1 × √N")
    ax.set_ylabel("ψ2 × √N")
    ax.set_title("1. each image at (ψ1, ψ2)", loc="left")
    ax.text(0.0, -0.17, "dashed: where a rotated vector crosses the threshold;\nbeyond it, that vector is non-zero",
            transform=ax.transAxes, fontsize=8.4, color=MUTE, va="top")
    ax = fig.add_subplot(gs[1])
    keys = [np.where(lab == j)[0][np.argsort(-s_vec[lab == j, j])] for j in range(r)]
    un = np.where(lab < 0)[0]
    keys.append(un)
    order = np.concatenate(keys)
    xs = np.arange(n)
    for j in range(r):
        base = (r - 1 - j) * 1.2
        ax.fill_between(xs, base, base + s_vec[order, j], color=PAL["sets"][j], lw=0, step="mid")
        ax.text(n * 1.01, base + 0.45, f"set {j + 1}\n{(lab == j).sum()} imgs", fontsize=8.2, color=MUTE, va="center")
    ax.axvspan(n - len(un), n, color=GREY, alpha=0.35, lw=0)
    ax.text(n - len(un) / 2, -0.25, f"unassigned\n({len(un)})", ha="center", va="top", fontsize=8.2, color=MUTE)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.6, r * 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"2. the {r} SEBA vectors (each scaled to max 1), images sorted by set", loc="left")
    ax = fig.add_subplot(gs[2])
    cell(ax, d.umap()[t_idx, b], set_colors(lab), o, s=2.2)
    ax.set_title(f"3. set = largest vector;\ngrey = unassigned (layout: {d.sites[b]})", loc="left")
    fig.text(0.05, 0.95, f"From eigenfunctions to coherent sets: layer block {d.sites[a]}–{d.sites[b]}, "
             f"{stepname(d.steps[t_idx])}, r = 3 (ψ0, ψ1, ψ2)", fontsize=12, weight="semibold")
    save(fig, "seba.png")
    print(f"  seba: block {d.sites[a]}-{d.sites[b]} r=3 sizes {[len(k) for k in keys[:-1]]} unassigned {len(un)}; "
          f"class purity of sets {[np.bincount(d.labels[k], minlength=10).max() / max(len(k), 1) for k in keys[:-1]]}")


# ----------------------------------------------------------------------------------------------- avgprod (Method)
def fig_avgprod(d: Data, t_idx, r):
    """avgprod.json: agreement of the average's leading span with the product's singular vectors, one row per rule."""
    rows = json.loads((d.root / "avgprod.json").read_text())["rows"]
    step = d.steps[t_idx]
    modes = sorted({row.get("mode", 0) for row in rows})
    keys = [("right", "Q = M_b⋯M_a: right sing. vectors"), ("left", "Q: left sing. vectors"),
            ("end_right", "M_b M_a: right"), ("end_left", "M_b M_a: left")]
    fig, axs = plt.subplots(len(modes), 4, figsize=(15, 4.1 * len(modes)), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.9, wspace=0.3, hspace=0.5)
    for ri_, mode in enumerate(modes):
        for ax, (k, title) in zip(axs[ri_], keys):
            z = np.full((d.S, d.S), np.nan)
            for row in rows:
                if row["step"] == step and row["r"] == r and row.get("mode", 0) == mode:
                    z[row["a"], row["b"]] = row[k]
            im = ax.imshow(z, cmap=SEQ, vmin=0, vmax=1)
            ax.set_xticks(range(d.S), d.sites, rotation=90, fontsize=7.5)
            ax.set_yticks(range(d.S), d.sites, fontsize=7.5)
            ax.set_xlabel("last site b")
            ax.set_ylabel("first site a")
            ax.set_title(title, loc="left", fontsize=9.5)
            v = [row[k] for row in rows if row["step"] == step and row["r"] == r and row.get("mode", 0) == mode]
            if v:
                print(f"  avgprod {d.modes[mode]} {k}: median {np.median(v):.3f}, min {np.min(v):.3f}, "
                      f"max {np.max(v):.3f} over {len(v)} pairs")
    for ri_, mode in enumerate(modes):
        pos = axs[ri_, 0].get_position()
        fig.text(0.005, pos.y0 + pos.height / 2, d.modes[mode], rotation=90, va="center", fontsize=10.5,
                 weight="semibold")
    cax = fig.add_axes([0.94, 0.2, 0.012, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean cos² of principal angles", color=MUTE)
    cb.outline.set_visible(False)
    fig.suptitle(f"Average vs product: agreement of leading spans (r = {r}, {stepname(step)}, constant projected out)",
                 x=0.01, ha="left", fontsize=11.5, weight="semibold")
    save(fig, "avgprod.png")


# ----------------------------------------------------------------------------------------------- kernel rules (Overview)
def fig_rules(d: Data, step=None):
    """Label-free diagnostics from sweep.json against a single global kappa, per site, with the per-image rules and
    the calibrated half-mass rule marked: participation number of psi_1, 1 - lambda_1, smallest of 8 SEBA sets."""
    rows = json.loads((d.root / "sweep.json").read_text())["rows"]
    step = step or max(r["step"] for r in rows)
    sites = [x for x in d.sites if any(r["site"] == x and r["step"] == step for r in rows)]
    cols = [PAL["ordinal"][int(i)] for i in np.linspace(1, len(PAL["ordinal"]) - 1, len(sites))]
    marks = ["o", "s", "^", "D", "v", "P"]
    adapt = [("union kNN 15", "kNN\ngraph"), ("perplexity 100", "perp.\n100"), ("perplexity 300", "perp.\n300")]
    metrics = [("ψ1 participation number", lambda r: r["participation"][0], "log", (1.5, 3000)),
               ("1 − λ1 (lower = slower ψ1)", lambda r: r["gaps"][0], "linear", (0, 0.5)),
               ("smallest of 8 coherent sets (images)", lambda r: max(min(r["seba"]["8"]["sizes"]), 0.8), "log", (0.8, 1000))]
    fig = plt.figure(figsize=(13.6, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.06, right=0.99, top=0.8, bottom=0.2)
    for mi, (name, fn_, scale, lim) in enumerate(metrics):
        sub = gs[mi].subgridspec(1, 2, width_ratios=[3, 1.35], wspace=0.06)
        ax, bx = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        for si, site in enumerate(sites):
            get = {r["rule"]: r for r in rows if r["site"] == site and r["step"] == step}
            gl = sorted((r["kappa"], fn_(r)) for n, r in get.items() if r["family"] == "global")
            ax.plot([g[0] for g in gl], [g[1] for g in gl], color=cols[si], marker=marks[si], ms=3.5, lw=1.6,
                    label=site)
            hm = get.get("mass 0.5")
            if hm is not None:
                ax.scatter([hm["kappa"]], [fn_(hm)], marker="x", s=40, color=ACC, zorder=5, linewidths=1.6,
                           label="half-mass (calibrated)" if si == 0 else None)
            bx.plot(range(len(adapt)), [fn_(get[a]) for a, _ in adapt], color=cols[si], marker=marks[si], ms=4.5,
                    lw=0, alpha=0.95)
        for a_ in (ax, bx):
            a_.set_yscale(scale)
            a_.set_ylim(*lim)
        if mi == 0:
            ax.axhline(2, color=MUTE, lw=0.8, ls="--")
            ax.text(2.05, 2.25, "a pair", color=MUTE, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1.8, 50)
        ax.set_xticks([2, 3, 5, 8, 12, 20, 30, 45], ["2", "3", "5", "8", "12", "20", "30", "45"])
        ax.minorticks_off()
        ax.set_xlabel("one global κ for the cell")
        ax.set_title(name, loc="left")
        bx.set_xlim(-0.5, len(adapt) - 0.5)
        bx.set_xticks(range(len(adapt)), [l for _, l in adapt], fontsize=7.8)
        bx.tick_params(axis="y", labelleft=False)
        bx.spines["left"].set_color(LINE)
        bx.set_facecolor(BG)
        bx.set_title("per-image", loc="left", fontsize=9, color=MUTE, weight="normal")
    h, l = ax.get_legend_handles_labels()
    order = [i for i, x in enumerate(l) if x in sites] + [i for i, x in enumerate(l) if x not in sites]
    fig.legend([h[i] for i in order], [l[i] for i in order], ncol=len(sites) + 1, loc="upper left", fontsize=9,
               bbox_to_anchor=(0.06, 0.93))
    fig.text(0.99, 0.02, "kNN graph = default rule; perp. = per-image perplexity; a smallest set of 0 is drawn at the "
             "floor", ha="right", fontsize=8, color=MUTE)
    fig.text(0.06, 0.975, f"Label-free diagnostics against one global κ, one cell per site ({stepname(step)}); "
             "per-image kernel rules at right", fontsize=11.5, weight="semibold", va="top")
    save(fig, "rules.png")


# ----------------------------------------------------------------------------------------------- sweep (Method)
def fig_sweep(d: Data, step=None):
    """Kernel-rule sweep (sweep.json): spread, slowness and class agreement of the leading modes, per rule and site."""
    sw = json.loads((d.root / "sweep.json").read_text())
    rows = sw["rows"]
    step = step or max(r["step"] for r in rows)
    sites = [x for x in d.sites if any(r["site"] == x and r["step"] == step for r in rows)]
    fam_order = ["global", "mass", "perplexity", "union_knn"]
    rules = []
    for f in fam_order:
        rs = sorted({(r.get("kappa") if f == "global" else r.get("target") if f == "mass" else
                      r.get("perplexity") if f == "perplexity" else r.get("k"), r["rule"]) for r in rows
                     if r["family"] == f and r["step"] == step})
        rules += [(f, name) for _, name in rs]
    x = np.arange(len(rules))
    lab = {"global": "global κ", "mass": "15-NN mass target", "perplexity": "perplexity (per image)",
           "union_knn": "kNN graph"}
    short = [n.split(" ")[-1] for _, n in rules]
    cols = [PAL["ordinal"][int(i)] for i in np.linspace(1, len(PAL["ordinal"]) - 1, len(sites))]
    marks = ["o", "s", "^", "D", "v", "P"]
    fam_of = [f for f, _ in rules]
    segs = [[i for i in range(len(rules)) if fam_of[i] == f] for f in fam_order if f in fam_of]
    fig, axs = plt.subplots(3, 1, figsize=(12.5, 8.8), sharex=True)
    for si, site in enumerate(sites):
        get = {r["rule"]: r for r in rows if r["site"] == site and r["step"] == step}
        pn = np.array([np.median(get[n]["participation"]) for _, n in rules])
        pmin = np.array([np.min(get[n]["participation"]) for _, n in rules])
        g1 = np.array([get[n]["gaps"][0] for _, n in rules])
        j8 = np.array([get[n]["seba"]["8"]["mean_matched_jaccard"] for _, n in rules])
        for gi, seg in enumerate(segs):
            kw = dict(color=cols[si], lw=1.8, marker=marks[si % len(marks)], ms=4, label=site if gi == 0 else None)
            axs[0].plot(x[seg], pn[seg], **kw)
            axs[0].plot(x[seg], pmin[seg], color=cols[si], lw=0.7, ls=":", alpha=0.7)
            axs[1].plot(x[seg], g1[seg], **kw)
            axs[2].plot(x[seg], j8[seg], **kw)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1.5, 2000)
    axs[0].axhline(2, color=MUTE, lw=0.8, ls="--")
    axs[0].text(len(rules) - 0.4, 2.2, "a pair", color=MUTE, fontsize=8.5, ha="right", va="bottom")
    axs[0].set_ylabel("participation number\nof ψ1–ψ8 (median; dotted: min)")
    axs[1].set_ylim(0, 0.5)
    axs[1].set_ylabel("1 − λ1 of M\n(lower = slower leading mode)")
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("8 coherent sets vs classes\n(matched Jaccard; validation)")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=len(sites), loc="upper right", fontsize=9.5,
               bbox_to_anchor=(0.99, 0.995), title="site", title_fontsize=9)
    starts = [i for i in range(len(rules)) if i == 0 or rules[i][0] != rules[i - 1][0]]
    for ax in axs:
        for i in starts[1:]:
            ax.axvline(i - 0.5, color=LINE, lw=1)
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < len(starts) else len(rules)
        axs[0].text((st + en - 1) / 2, 2600, lab[rules[st][0]], ha="center", va="bottom", fontsize=9.5, color=FG)
    axs[2].set_xticks(x, short, fontsize=8.5)
    axs[2].set_xlabel("rule parameter")
    fig.suptitle(f"Kernel-rule sweep at {stepname(step)}: label-free diagnostics (top two), agreement with classes "
                 "(validation, bottom)", x=0.01, ha="left",
                 fontsize=12, weight="semibold")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.08, hspace=0.12)
    save(fig, "sweep.png")


# ----------------------------------------------------------------------------------------------- (b) how-to gif
def fig_howto(d: Data, mode, t_idx, a, b, site_time, c_time, d_time, hover_img, r=4):
    U = d.umap()
    T, S = d.T, d.S
    W, H = 10.4, 6.6
    gx0, gy0, gw, gh = 0.075, 0.2, 0.62, 0.64  # grid area in figure coordinates
    cw, ch = gw / S, gh / T
    cls = class_colors(d.labels)
    f1 = d.psi("layer", t_idx, mode, a, b, 1)
    eig = div_colors(f1)
    eig_order = np.argsort(np.abs(f1))
    _, slab = d.seba("layer", t_idx, mode, a, b, r)
    sebc = set_colors(slab)
    _, tlab = d.seba("time", site_time, mode, c_time, d_time, r)
    tc = set_colors(tlab)
    vals = d.vals("layer", t_idx, mode, a, b)
    tvals = d.vals("time", site_time, mode, c_time, d_time)
    ret = d.retention_psi("layer", t_idx, mode, a, b, 1)
    tret = d.retention_psi("time", site_time, mode, c_time, d_time, 1)
    neutral = np.tile(np.array(to_rgb("#9a9aa0")), (d.N, 1))

    def base(colors, order, block=None, text="", sub="", hl=None, hover=False, side=None, step_no=None):
        fig = plt.figure(figsize=(W, H), dpi=88)
        fig.patch.set_facecolor(SURF)
        for t in range(T):
            fig.text(gx0 - 0.006, gy0 + gh - (t + 0.5) * ch, f"{d.steps[t]:,}", ha="right", va="center", fontsize=7,
                     color=MUTE)
            for s in range(S):
                ax = fig.add_axes([gx0 + s * cw + 0.002, gy0 + gh - (t + 1) * ch + 0.002, cw - 0.004, ch - 0.004])
                inb = block is not None and block(t, s)
                cell(ax, U[t, s], colors, order, s=0.35, frame=ACC if inb else LINE, lw=1.6 if inb else 0.5)
                if hover:
                    ax.scatter(*U[t, s, hover_img], s=26, facecolors="none", edgecolors=FG, linewidths=0.9)
        for s in range(S):
            fig.text(gx0 + (s + 0.5) * cw, gy0 + gh + 0.008, d.sites[s], ha="center", va="bottom", fontsize=7, color=MUTE)
        fig.text(gx0 - 0.06, gy0 + gh + 0.008, "step", fontsize=7, color=MUTE)
        fig.text(gx0, gy0 + gh + 0.04, "sites (depth) →", fontsize=8, color=MUTE)
        fig.text(0.01, gy0 + gh / 2, "training →", fontsize=8, color=MUTE, rotation=-90, va="center")
        if hl is not None:
            for (t0, s0, t1, s1) in hl:
                fig.patches.append(Rectangle((gx0 + s0 * cw - 0.003, gy0 + gh - (t1 + 1) * ch - 0.003),
                                             (s1 - s0 + 1) * cw + 0.006, (t1 - t0 + 1) * ch + 0.006, fill=False,
                                             ec=FG, lw=2.2, transform=fig.transFigure, ls=(0, (3, 2))))
        # caption band
        fig.patches.append(Rectangle((0, 0), 1, 0.15, color=BG, transform=fig.transFigure, zorder=-1))
        if step_no is not None:
            fig.text(0.02, 0.105, f"{step_no}", fontsize=17, weight="bold", color=ACC, va="center")
        fig.text(0.07, 0.115, text, fontsize=12.2, weight="semibold", va="center", color=FG)
        fig.text(0.07, 0.05, sub, fontsize=10, va="center", color=MUTE)
        if side:
            side(fig)
        return fig

    def spectrum(v, title, hk=None, nsel=None):
        def draw(fig):
            ax = fig.add_axes([0.765, 0.58, 0.215, 0.24])
            lo = min(0.5, float(np.min(v)) - 0.02)
            cols = [ACC if (hk is not None and k == hk) else (PAL["sets"][0] if (nsel and k < nsel) else "#b8c4d6")
                    for k in range(len(v))]
            ax.bar(range(len(v)), v - lo, bottom=lo, color=cols, width=0.7)
            ax.set_ylim(lo, 1.0)
            ax.set_xticks(range(len(v)), [f"ψ{k}" for k in range(len(v))], fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_title(title, fontsize=8.5, loc="left")
        return draw

    def strip(rv, labels_, lo, hi, title):
        def draw(fig):
            ax = fig.add_axes([0.765, 0.25, 0.215, 0.2])
            x = np.arange(len(rv))
            ax.axvspan(lo - 0.5, hi + 0.5, color=ACC, alpha=0.13, lw=0)
            ax.plot(x, rv, color=ACC, marker="o", ms=3, lw=1.4)
            ax.set_ylim(0.5, 1)
            ax.set_xticks(x, labels_, rotation=90, fontsize=6.5)
            ax.tick_params(labelsize=7)
            ax.set_title(title, fontsize=8.5, loc="left")
        return draw

    def both(*fs):
        def draw(fig):
            for f in fs:
                f(fig)
        return draw

    def thumb(fig):
        ax = fig.add_axes([0.80, 0.38, 0.13, 0.2])
        ax.imshow(d.images[hover_img])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"image #{hover_img}\n({d.meta['classes'][d.labels[hover_img]]})", fontsize=8.5)

    lb = lambda t, s: t == t_idx and a <= s <= b  # noqa: E731
    tb = lambda t, s: s == site_time and c_time <= t <= d_time  # noqa: E731
    bl = f"{d.sites[a]}–{d.sites[b]} at step {d.steps[t_idx]:,}"
    spec_l = spectrum(vals, f"spectrum of the block\n{bl}", hk=1)
    spec_s = spectrum(vals, f"spectrum of the block\n{bl}", nsel=r)
    strip_l = strip(ret, d.sites, a, b, "retention of ψ1 in each site")
    steps_ = [  # (colors, order, block, text, sub, highlight, hover, side, ms)
        (neutral, None, None, "How to read the explorer (slowly)",
         "Ten steps, five to eight seconds each. The real thing is in the Explorer tab.", None, False, None, 4500),
        (neutral, None, None, "Each square is one cell: one checkpoint (row), one site (column).",
         "Every cell shows the same 2,000 test images, laid out by UMAP of that cell's activations. Display only.",
         [(t_idx, 4, t_idx, 4)], False, None, 6500),
        (neutral, None, None, "A dot is the same image in every cell.",
         "Hover an image and it is ringed everywhere, so you can follow it through depth and training.",
         None, True, thumb, 6500),
        (cls, None, None, "Colour by class, for orientation.",
         "The class labels never enter the computation; they only colour plots and check results.",
         None, False, None, 5500),
        (cls, None, lb, f"Pick a block: sites {bl}.",
         "Their walks M_c are averaged into the block operator M̄. Its eigenfunctions ψ_k vary least, on average.",
         None, False, spec_l, 7000),
        (eig, eig_order, lb, "Colour by eigenfunction ψ1: red > 0, blue < 0, grey ≈ 0.",
         "Computed from the outlined block only, then painted on every cell. Its sign is a convention.",
         None, False, spec_l, 7000),
        (eig, eig_order, lb, "Retention strip: ρ_c(ψ1) = ψ1ᵀ M_c ψ1 for each site.",
         "The eigenvalue λ1 is its mean over the block. A lower ρ_c: ψ1 varies more between that cell's neighbours.",
         None, False, both(spec_l, strip_l), 8000),
        (sebc, grey_first(slab), lb, f"Colour by coherent sets (SEBA, here r = {r}).",
         f"ψ0…ψ{r - 1} are rotated into {r} sparse vectors; each image takes its largest one; grey = unassigned.",
         None, False, spec_s, 7500),
        (sebc, grey_first(slab), lb, "The block's colours are painted on every cell.",
         "Along the row: the other sites at this checkpoint. Up a column: the same site at earlier checkpoints.",
         [(0, b, t_idx, b), (t_idx, 0, t_idx, S - 1)], False, spec_s, 8000),
        (tc, grey_first(tlab), tb, f"Time block: fix a site ({d.sites[site_time]}), average over checkpoints.",
         f"Here steps {d.steps[c_time]:,}–{d.steps[d_time]:,}. The retention strip then runs over checkpoints.",
         None, False, both(spectrum(tvals, f"spectrum ({d.sites[site_time]}, steps {d.steps[c_time]:,}–{d.steps[d_time]:,})", nsel=r),
                           strip(tret, [f"{s:,}" for s in d.steps], c_time, d_time, "retention of ψ1 per checkpoint")), 8000),
        (tc, grey_first(tlab), tb, "Your turn: open the Explorer tab.",
         "Also try the kernel rule, r, and the channel-loadings panel. Colours are always from the selected block.",
         None, False, None, 5000),
    ]
    frames, durs = [], []
    for i, (cols, order, block, text, sub, hl, hov, side, ms) in enumerate(steps_):
        fig = base(cols, order, block, text, sub, hl, hov, side, step_no=None if i == 0 else i)
        frames.append(fig_to_image(fig))
        if i == 7:
            fig.savefig(OUT / "howto_still.png", dpi=88)
            print("wrote", OUT / "howto_still.png")
        plt.close(fig)
        durs.append(ms)
    write_gif(frames, durs, "howto.gif", colors=128)


def kappa_table(d: Data, t_idx):
    """HTML for the Method tab: kappa / achieved 15-NN mass share per site and rule at one checkpoint."""
    m = d.meta
    head = "".join(f'<th class="num">{n}</th>' for n in d.modes)
    rows = []
    for s in range(d.S):
        def fmt(r):
            k = m["kappa"][r][t_idx][s]
            nm = m["nn_mass"][r][t_idx][s]
            return f'<td class="num">{"–" if k is None else f"{k:.1f}"} / {"–" if nm is None else f"{nm:.2f}"}</td>'
        cells = "".join(fmt(r) for r in range(len(d.modes)))
        rows.append(f"<tr><td>{d.sites[s]}</td>{cells}</tr>")
    return (f'<table class="data">\n<tr><th>site</th>{head}</tr>\n' + "\n".join(rows) +
            f'\n</table>\n<p class="byline">κ (median κ<sub>i</sub> for the perplexity rules; none for the kNN graph) / median '
            f'15-NN share of kernel mass, step {d.steps[t_idx]:,}; '
            f'source data/{d.tag}/meta.json.</p>')


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--only", default="")
    ap.add_argument("--mode", type=int, default=0, help="kernel rule index for examples (default 0: kNN graph)")
    ap.add_argument("--kernel", default=None, help="figdata kernel npz (default: site/figdata/<tag>_kernel.npz, "
                                                    "falling back to r20s0)")
    ap.add_argument("--grams", default=None, help="float16 Gram stack (S,N,N) of the final checkpoint")
    ap.add_argument("--block", default="7,10", help="layer block a,b at the final checkpoint for pipeline/seba/howto")
    ap.add_argument("--sd", default="auto", help="static-vs-dynamic example a,b,s or auto")
    ap.add_argument("--tblock", default="9,4,7", help="time block site,c,d")
    ap.add_argument("--kcell", default="5", help="site index for the kappa figure (final checkpoint)")
    ap.add_argument("--hover", type=int, default=1234)
    ap.add_argument("--kappa-table", action="store_true", help="print the Method tab's kappa table (HTML) and exit")
    a = ap.parse_args()
    only = set(a.only.split(",")) - {""}
    want = lambda n: not only or n in only  # noqa: E731
    d = Data(a.tag)
    T = d.T - 1
    if a.kappa_table:
        print(kappa_table(d, T))
        return
    ba, bb = map(int, a.block.split(","))
    print(f"tag {a.tag} (fixture={d.fixture}); mode {a.mode} = {d.modes[a.mode]}; block {d.sites[ba]}-{d.sites[bb]}")
    if want("kappa"):
        kp = Path(a.kernel) if a.kernel else HERE / "figdata" / f"{a.tag}_kernel.npz"
        if not kp.exists():
            kp = HERE / "figdata" / "r20s0_kernel.npz"
        fig_kappa(d, dict(np.load(kp)), T, int(a.kcell))
    if want("pipeline"):
        gp = Path(a.grams) if a.grams else Path(f"/var/tmp/cm-figs/grams_{a.tag}_{d.steps[T]}.npy")
        if not gp.exists():
            gp = Path(f"/var/tmp/cm-figs/grams_r20s0_{d.steps[T]}.npy")
        if gp.exists():
            fig_pipeline(d, pipeline_data(d, gp, T, ba, bb, a.mode, 4))
        else:
            print("  pipeline: no Gram file; run figdata.py first")
    if want("steps"):
        gp = Path(a.grams) if a.grams else Path(f"/var/tmp/cm-figs/grams_{a.tag}_{d.steps[T]}.npy")
        if not gp.exists():
            gp = Path(f"/var/tmp/cm-figs/grams_r20s0_{d.steps[T]}.npy")
        if gp.exists():
            fig_steps(d, pipeline_data(d, gp, T, ba, bb, a.mode, 4))
        else:
            print("  steps: no Gram file; run figdata.py --grams-only first")
    if want("static_dynamic"):
        if a.sd == "auto":
            gap, sa, sb, ss = pick_static_dynamic(d, T, a.mode)
            print(f"  auto static/dynamic pick: gap {gap:.4f}")
        else:
            sa, sb, ss = map(int, a.sd.split(","))
        fig_static_dynamic(d, T, a.mode, sa, sb, ss)
    if want("timeblock"):
        ts, tc, td = map(int, a.tblock.split(","))
        fig_timeblock(d, ts, tc, td, a.mode, 4)
    if want("hero"):
        ts, tc, td = map(int, a.tblock.split(","))
        fig_hero(d, ts, tc, td, a.mode, 4)
    if want("seba"):
        fig_seba(d, T, a.mode, ba, bb)
    if want("avgprod") and (d.root / "avgprod.json").exists():
        fig_avgprod(d, T, 3)
    if want("sweep") and (d.root / "sweep.json").exists():
        fig_sweep(d)
    if want("rules") and (d.root / "sweep.json").exists():
        fig_rules(d)
    if want("howto"):
        ts, tc, td = map(int, a.tblock.split(","))
        fig_howto(d, a.mode, T, ba, bb, ts, tc, td, a.hover)


if __name__ == "__main__":
    main()
