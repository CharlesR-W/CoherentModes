"""Static figures for the LM toy page (matplotlib; palette and type of the ResNet site).

    uv run --with numpy --with matplotlib --with pillow python -I page/make_figures.py <tag>

Reads explore/lm_coherent/data/<tag>/ and writes page/build/<tag>/fig/:

  hero.png        F1: the explorer grid at the final step (bag row + six layers x six positions, AlignedUMAP
                  layouts), coloured by the r = 4 coherent sets of T1's block (reading block, L8); that row outlined
  step1..5.png    the Overview's procedure, on the stored thumbnails of figdata.npz (cell (L8, 128); T1's block)
                  and T1's block record: cosine Gram, kernel and walk, block average, eigenfunctions with retention,
                  SEBA vectors and sets
  cube.png        F10: the 5 x 6 x 6 cells as a lattice with one reading, one depth and one training block marked

build_page.py runs this automatically when the PNGs are missing or older than the data. Nothing here feeds back
into a measurement. Layout coordinates are quantised exactly as on the page (build_page.layout_u8).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/var/tmp/cm-lm/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_page import RESNET, ROOT, layout_u8  # noqa: E402

PAL = json.loads((RESNET / "palette.json").read_text())
FG, MUTE, LINE, BG, ACC, SURF = PAL["fg"], PAL["mute"], PAL["line"], PAL["bg"], PAL["accent"], PAL["surface"]
DIV = LinearSegmentedColormap.from_list("div", [PAL["diverging"]["neg"], PAL["diverging"]["mid"], PAL["diverging"]["pos"]])
SEQ = LinearSegmentedColormap.from_list("seq", PAL["sequential"])
GREY = PAL["unassigned"]
for _f in sorted((RESNET / "figdata" / "fonts").glob("Inter-*.ttf")):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "text.color": FG, "axes.labelcolor": FG,
    "axes.edgecolor": LINE, "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlecolor": FG, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "mathtext.fontset": "dejavusans"})
SMALL = 13
STEP_RC = {"font.size": 14, "axes.titlesize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12}


def set_colors(lab):
    lab = np.asarray(lab)
    out = np.array([to_rgb(GREY)] * len(lab))
    for j in range(lab.max(initial=-1) + 1):
        out[lab == j] = to_rgb(PAL["sets"][j % len(PAL["sets"])])
    return out


def grey_first(lab):
    return np.argsort(np.asarray(lab) >= 0, kind="stable")


def seba_vectors(v, rot, perm):
    """SEBA vectors from the stored rotation (mirrors the ResNet site's recomputation, clip=True)."""
    z = v @ rot
    mu = 0.99 / np.sqrt(len(v))
    s = np.sign(z) * np.maximum(np.abs(z) - mu, 0)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0))
    sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return np.maximum(s[:, perm], 0)


class D:
    def __init__(self, tag):
        self.tag = tag
        self.src = ROOT / "data" / tag
        self.meta = json.loads((self.src / "meta.json").read_text())
        self.steps, self.layers, self.positions = self.meta["steps"], self.meta["layers"], self.meta["positions"]
        self.final = self.steps[-1]
        self.fixture = bool(self.meta.get("fixture"))
        self.n = self.meta["n"]
        self.blocks = np.load(self.src / f"blocks_{self.final}.npz")
        self.umap = np.load(self.src / "umap.npz")
        self.fig = np.load(self.src / "figdata.npz")
        self.li8 = self.layers.index(8)
        self.r4 = self.meta["seba_r"].index(4)
        b, li = self.blocks, self.li8
        self.lab4 = b["reading_seba_label"][li, self.r4].astype(int)
        q = 32767.0 if b["reading_vecs"].dtype == np.int16 else 127.0
        self.psi = (b["reading_vecs"][li].astype(np.float64) * b["reading_scale"][li][:, None] / q).T   # (N, 8)
        self.vals = b["reading_vals"][li].astype(np.float64)
        self.ret = b["reading_retention_psi"][li].astype(np.float64)    # (8, positions)
        v = np.concatenate([np.full((self.n, 1), 1 / np.sqrt(self.n)), self.psi[:, :3]], 1)
        perm = b["reading_seba_perm"][li, self.r4, :4].astype(int)
        self.svec = seba_vectors(v, b["reading_seba_rot"][li, self.r4, :4, :4].astype(np.float64), perm)
        self.mock = " (MOCKUP)" if self.fixture else ""


def save(fig, out, name, dpi=150):
    """PNG, then quantised to a 256-colour palette (the page embeds every figure; this roughly halves the bytes)."""
    from PIL import Image
    fig.savefig(out / name, dpi=dpi)
    plt.close(fig)
    im = Image.open(out / name).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out / name, optimize=True)
    print(f"  wrote {name} ({(out / name).stat().st_size / 1e3:.0f} kB)")


def panel(ax, xy, cols, order, s=2.2):
    ax.scatter(xy[order, 0], 1 - xy[order, 1], c=cols[order], s=s, lw=0, rasterized=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(LINE)


# ----------------------------------------------------------------------------------------------- F1 hero
def fig_hero(d: D, out: Path):
    grid = layout_u8(d.umap["grid"]).astype(np.float64) / 255   # [7, 6, N, 2]
    cols, order = set_colors(d.lab4), grey_first(d.lab4)
    rows = ["bag"] + [f"L{x}" for x in d.layers]
    R, C = grid.shape[:2]
    fig = plt.figure(figsize=(10.5, 12.4))
    gs = fig.add_gridspec(R, C, left=0.08, right=0.99, top=0.9, bottom=0.03, wspace=0.06, hspace=0.06)
    for r in range(R):
        for c in range(C):
            ax = fig.add_subplot(gs[r, c])
            panel(ax, grid[r, c], cols, order, s=1.6)
            if r == 0:
                ax.set_title(f"t = {d.positions[c]}", fontsize=12, weight="semibold", color=FG)
            if c == 0:
                ax.set_ylabel(rows[r], rotation=0, ha="right", va="center", fontsize=12, labelpad=8,
                              color=FG if r - 1 == d.li8 else MUTE, weight="bold" if r - 1 == d.li8 else "normal")
    # outline T1's row (L8)
    r = d.li8 + 1
    bb0, bb1 = fig.axes[r * C].get_position(), fig.axes[r * C + C - 1].get_position()
    fig.patches.append(FancyBboxPatch((bb0.x0 - 0.004, bb0.y0 - 0.003), bb1.x1 - bb0.x0 + 0.008, bb0.height + 0.006,
                                      boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
                                      fill=False, ec=FG, lw=2.2))
    sizes = [int((d.lab4 == j).sum()) for j in range(4)]
    fig.text(0.08, 0.965, f"{d.meta['model'].split('/')[-1]}, step {d.final:,}: coherent sets of the reading block "
             f"L8, t = 4 → 128 (outlined){d.mock}", fontsize=14, weight="semibold")
    keys = ", ".join(str(x) for x in sizes) + f"; {int((d.lab4 < 0).sum())} unassigned (grey)"
    fig.text(0.08, 0.94, "Colours: the r = 4 sets of the outlined row, painted on every panel (same snippet, same colour). "
             "Dot positions: display only (AlignedUMAP per row). Set sizes " + keys + ".",
             fontsize=11, color=MUTE)
    save(fig, out, "hero.png", dpi=110)


# ----------------------------------------------------------------------------------------------- steps
def _head(fig, ax, k, title, sub=None):
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + 0.17, f" {k} ", fontsize=17, weight="bold", color=SURF, va="bottom",
             bbox=dict(boxstyle="circle,pad=0.18", fc=ACC, ec="none"))
    fig.text(bb.x0 + 0.075, bb.y1 + 0.172, title, fontsize=17, weight="semibold", va="bottom", color=FG)
    if sub:
        fig.text(bb.x0, bb.y1 + 0.1, sub, fontsize=SMALL, va="bottom", color=MUTE)


def _mat(ax, img, **kw):
    ax.imshow(img, interpolation="nearest", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(LINE)


def _bar(fig, ax, lab_sorted):
    pos = ax.get_position()
    cols = set_colors(lab_sorted)
    bx = fig.add_axes([pos.x0 - pos.width * 0.065, pos.y0, pos.width * 0.045, pos.height])
    bx.imshow(cols[:, None, :], aspect="auto", interpolation="nearest")
    bx.set_axis_off()


def _lognorm(m):
    v = m[m > 0]
    return LogNorm(vmin=np.percentile(v, 50), vmax=np.percentile(v, 99.9))


def fig_steps(d: D, out: Path):
    fdat = d.fig
    order = fdat["order"].astype(int)
    cell = fdat["cell"].tolist()
    lab_sorted = d.lab4[order]
    with plt.rc_context(STEP_RC):
        # 1. cosine Gram
        fig = plt.figure(figsize=(4.6, 5.2))
        ax = fig.add_axes([0.16, 0.1, 0.78, 0.6])
        C = fdat["C"].astype(np.float64)
        lim = np.percentile(np.abs(C), 99.5)
        _mat(ax, C, cmap=DIV, vmin=-lim, vmax=lim)
        _bar(fig, ax, lab_sorted)
        ax.text(0, -0.03, "red > 0 > blue", transform=ax.transAxes, va="top", fontsize=SMALL, color=MUTE)
        _head(fig, ax, 1, "Cosine Gram", f"$C_{{ij}} = \\cos(h_i, h_j)$; cell (L{cell[0]}, t = {cell[1]})")
        save(fig, out, "step1.png")
        # 2. kernel, then walk
        fig = plt.figure(figsize=(7.4, 5.2))
        ka = fig.add_axes([0.06, 0.1, 0.42, 0.6])
        pa = fig.add_axes([0.55, 0.1, 0.42, 0.6])
        K, Pm = fdat["K"].astype(np.float64), fdat["P"].astype(np.float64)
        _mat(ka, K, cmap=SEQ, norm=_lognorm(K))
        _mat(pa, Pm, cmap=SEQ, norm=_lognorm(Pm))
        ka.set_title("kernel $K_c$ (union 15-NN graph)", loc="left", fontsize=13, weight="normal", pad=5)
        pa.set_title("walk $P_c$ = Sinkhorn($K_c$)", loc="left", fontsize=13, weight="normal", pad=5)
        ka.text(0, -0.03, "log colour", transform=ka.transAxes, va="top", fontsize=SMALL, color=MUTE)
        pa.text(0, -0.03, "rows, columns sum to 1", transform=pa.transAxes, va="top", fontsize=SMALL, color=MUTE)
        _head(fig, ka, 2, "Kernel, then walk", "same cell; lazy walk $M_c = (I + P_c)/2$")
        save(fig, out, "step2.png")
        # 3. block average (with the cells' own walks when figdata carries them)
        M = fdat["Mbar"].astype(np.float64)
        Moff = M.copy()
        np.fill_diagonal(Moff, 0)
        mnorm = _lognorm(Moff)
        if "M_cells" in fdat.files:
            fig = plt.figure(figsize=(7.6, 5.2))
            cts = [int(x) for x in fdat["cells_t"]] if "cells_t" in fdat.files else d.positions
            pick = [0, 2, 4, 5] if len(cts) >= 6 else list(range(min(4, len(cts))))
            for k, ci in enumerate(pick):
                ax_ = fig.add_axes([0.05 + (k % 2) * 0.21, 0.42 - (k // 2) * 0.32, 0.19, 0.27])
                mc = fdat["M_cells"][ci].astype(np.float64)
                np.fill_diagonal(mc, 0)
                _mat(ax_, mc, cmap=SEQ, norm=mnorm)
                ax_.set_title(f"$M_c$: t = {cts[ci]}", loc="left", fontsize=SMALL, weight="normal", pad=3)
                if k == 0:
                    first = ax_
            ax = fig.add_axes([0.52, 0.1, 0.44, 0.6])
            _mat(ax, Moff, cmap=SEQ, norm=mnorm)
            ax.set_title("$\\bar M$ = mean of the 6 cells", loc="left", fontsize=13, weight="normal", pad=5)
            ax.text(0, -0.03, "off-diagonal, log colour", transform=ax.transAxes, va="top", fontsize=SMALL, color=MUTE)
            _head(fig, first, 3, "Average over a block", "reading block L8, t = 4 to 128 (four of its six cells shown)")
        else:
            fig = plt.figure(figsize=(4.6, 5.2))
            ax = fig.add_axes([0.12, 0.1, 0.8, 0.6])
            _mat(ax, Moff, cmap=SEQ, norm=mnorm)
            ax.text(0, -0.03, "off-diagonal, log colour", transform=ax.transAxes, va="top", fontsize=SMALL, color=MUTE)
            _head(fig, ax, 3, "Average over a block", f"$\\bar M$ = mean of 6 cells: L8, t = 4 to 128")
        save(fig, out, "step3.png")
        # 4. eigenfunctions and retention
        fig = plt.figure(figsize=(6.4, 6.0))
        ax = fig.add_axes([0.14, 0.47, 0.82, 0.3])
        n = d.n
        xs = np.arange(n)
        for k in (1, 2, 3):
            off = (3 - k) * 4.2
            ax.plot(xs, d.psi[order, k - 1] * np.sqrt(n) + off, color=FG, lw=0.7)
            ax.text(-n * 0.02, off, f"$\\psi_{k}$", ha="right", va="center", fontsize=14)
        ax.set_xlim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        tab = np.column_stack([d.ret[:3], d.vals[1:4]])
        nc = d.ret.shape[1]
        bx = fig.add_axes([0.14, 0.1, 0.82, 0.25])
        img = SEQ(np.clip((tab - 0.5) / 0.5, 0, 1))
        img[:, nc] = to_rgb(BG) + (1,)
        bx.imshow(img, aspect="auto", interpolation="nearest")
        for i in range(3):
            for j in range(nc + 1):
                dark = j < nc and (tab[i, j] - 0.5) / 0.5 > 0.55
                bx.text(j, i, f"{tab[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        color=SURF if dark else FG, weight="semibold" if j == nc else "normal")
        bx.axvline(nc - 0.5, color=SURF, lw=3)
        bx.set_xticks(range(nc + 1), [str(t) for t in d.positions] + ["$\\lambda_k$"], fontsize=SMALL)
        bx.set_yticks(range(3), [f"$\\psi_{k}$" for k in (1, 2, 3)], fontsize=13)
        bx.tick_params(length=0)
        bx.xaxis.tick_top()
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.set_xlabel("retention $\\rho_c(\\psi_k)$ at each position t;  $\\lambda_k$ = mean", fontsize=SMALL, color=MUTE)
        _head(fig, ax, 4, "Eigenfunctions", "leading eigenvectors of $\\bar M$, items in the same order")
        save(fig, out, "step4.png")
        # 5. SEBA vectors and sets
        fig = plt.figure(figsize=(8.6, 5.4))
        ax = fig.add_axes([0.07, 0.14, 0.4, 0.56])
        so = np.lexsort((-d.svec.max(1), np.where(d.lab4 < 0, 99, d.lab4)))
        nun = int((d.lab4 < 0).sum())
        for j in range(4):
            base = (3 - j) * 1.2
            ax.fill_between(xs, base, base + d.svec[so, j], color=PAL["sets"][j], lw=0, step="mid")
            ax.text(-n * 0.03, base + 0.35, f"{j + 1}", ha="right", va="center", fontsize=14)
        if nun:
            ax.axvspan(n - nun, n, color=GREY, alpha=0.45, lw=0)
        ax.set_xlim(0, n)
        ax.set_ylim(-0.4, 4 * 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title("SEBA vectors $s_1,\\ldots,s_4$", loc="left", fontsize=13, weight="normal", pad=5)
        ax.set_xlabel("item (sorted by set)", fontsize=SMALL, color=MUTE)
        bx = fig.add_axes([0.56, 0.14, 0.4, 0.56])
        xy = layout_u8(d.umap["grid"][d.li8 + 1, -1]).astype(np.float64) / 255
        panel(bx, xy, set_colors(d.lab4), grey_first(d.lab4), s=4)
        bx.set_title("coherent sets (layout: L8, t = 128)", loc="left", fontsize=13, weight="normal", pad=5)
        bx.text(0, -0.03, "set = largest $s_j$; grey: all 0", transform=bx.transAxes, va="top", fontsize=SMALL,
                color=MUTE)
        _head(fig, ax, 5, "Sparse vectors, coherent sets", "SEBA on $\\psi_0,\\ldots,\\psi_3$ (r = 4)")
        save(fig, out, "step5.png")


# ----------------------------------------------------------------------------------------------- F10 cube
def fig_cube(d: D, out: Path):
    T, L, S = len(d.positions), len(d.layers), len(d.steps)
    ox, oy = 0.62, 0.42            # oblique offset per checkpoint

    def P(ti, li, si):          # the final checkpoint is the front face; earlier ones recede
        return ti + ox * (S - 1 - si), li + oy * (S - 1 - si)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    pts = np.array([P(t, l, s) for s in range(S) for l in range(L) for t in range(T)])
    ax.scatter(pts[:, 0], pts[:, 1], s=16, color="#c9c8c1", zorder=1, lw=0)
    li8, fin = d.li8, S - 1
    blocks = [
        ("reading block: L8, t = 4 → 128, step " + f"{d.final:,}", [P(t, li8, fin) for t in range(T)], PAL["sets"][0]),
        ("depth block: L2 → L12 at t = 16, step " + f"{d.final:,}", [P(2, l, fin) for l in range(L)], PAL["sets"][1]),
        ("training block: L8, t = 128, steps 0 → " + f"{d.final:,}", [P(T - 1, li8, s) for s in range(S)], PAL["sets"][2]),
    ]
    for name, pp, col in blocks:
        pp = np.array(pp)
        ax.plot(pp[:, 0], pp[:, 1], color=col, lw=9, alpha=0.28, solid_capstyle="round", zorder=2)
        ax.scatter(pp[:, 0], pp[:, 1], s=34, color=col, zorder=3, lw=0)
    # axes labels
    fr = np.array([P(-0.5, -0.5, S - 1), P(T - 0.5, -0.5, S - 1), P(T - 0.5, L - 0.5, S - 1), P(-0.5, L - 0.5, S - 1),
                   P(-0.5, -0.5, S - 1)])
    ax.plot(fr[:, 0], fr[:, 1], color=LINE, lw=1.2, zorder=0)
    for t in range(T):
        x, y = P(t, -0.9, S - 1)
        ax.text(x, y, str(d.positions[t]), ha="center", va="center", fontsize=10, color=MUTE)
    for l in range(L):
        x, y = P(-0.65, l, S - 1)
        ax.text(x, y, f"L{d.layers[l]}", ha="right", va="center", fontsize=10, color=MUTE)
    for s in range(S):
        x, y = P(T - 1 + 0.45, -0.55, s)
        ax.text(x, y, f"{d.steps[s]:,}" if d.steps[s] < 1000 else f"{d.steps[s] // 1000}k", ha="left", va="center",
                fontsize=9.5, color=MUTE)
    ax.text(*P(2.5, -1.6, S - 1), "position t (tokens read) →", ha="center", va="center", fontsize=11, color=FG)
    ax.text(*P(-1.65, 2.5, S - 1), "layer →", ha="center", va="center", fontsize=11, color=FG, rotation=90)
    ax.text(*P(T + 0.15, -1.45, 1.6), "training step (front: final)", ha="left", va="center", fontsize=11,
            color=FG, rotation=34)
    for i, (name, pp, col) in enumerate(blocks):
        ax.text(0.0, 1.0 - 0.07 * i, "●  " + name, transform=ax.transAxes, color=col, fontsize=11.5,
                weight="semibold", va="top")
    ax.set_axis_off()
    ax.set_xlim(-2.2, T - 1 + ox * (S - 1) + 2.2)
    ax.set_ylim(-2.0, L - 1 + oy * (S - 1) + 2.4)
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, out, "cube.png", dpi=130)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    tag = argv[0]
    d = D(tag)
    out = HERE / "build" / tag / "fig"
    out.mkdir(parents=True, exist_ok=True)
    print(f"make_figures: {tag} -> {out}")
    fig_hero(d, out)
    fig_steps(d, out)
    fig_cube(d, out)


if __name__ == "__main__":
    main()
