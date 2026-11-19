"""Static summary figures for the Overview (matplotlib, stored data only, the site's palette and fonts).

    uv run -q --with numpy --with matplotlib --with pillow python -I explore/ncs_report/make_figures.py

Writes site/assets/fig/:
  hero_vit.png    four "mini explorer views" of DINOv2 ViT-S/14: the r = 8 coherent sets of the single-snapshot
                  blocks L1, L6, L9 and of the layer block L9..L12, each on a UMAP panel, with five member images
                  per set (largest SEBA value first; computed here from the stored eigenvectors and rotations)
  hero_mamba.png  three views of Mamba-130m on the stored UMAP of the all-layer residual block: the r = 8 sets of
                  residual L4 and L24 and the r = 16 sets of the all-layer residual block, with each set's size and
                  commonest token at t and at t+1
Positions are display only. Labels are not used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncs-mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
OUT = SITE / "assets" / "fig"
EXP = HERE.parent
VIT = EXP / "vit_coherent" / "data" / "vits_b2k"
MC = EXP / "mamba_contexts" / "data" / "m130c"
PAL = json.loads((SITE / "palette.json").read_text())
FG, MUTE, LINE, SURF = PAL["fg"], PAL["mute"], PAL["line"], PAL.get("surface", "#ffffff")
GREY = PAL["unassigned"]
for f in sorted((EXP / "cifar_coherent" / "site" / "figdata" / "fonts").glob("Inter-*.ttf")):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "text.color": FG, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": LINE})
MONO = "DejaVu Sans Mono"


def set_colour(j):
    if j < 8:
        return PAL["sets"][j]
    return PAL["sets_dark"][j - 8] if j < 16 else "#8d8c86"


def seba_soft(v, rot, perm):
    z = v @ rot
    s = np.sign(z) * np.maximum(np.abs(z) - 0.99 / np.sqrt(len(v)), 0)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0)); sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return np.maximum(s[:, perm], 0)


def norm_layout(e):
    lo, hi = np.percentile(e, 0.5, axis=0), np.percentile(e, 99.5, axis=0)
    span, mid = (hi - lo).max(), (hi + lo) / 2
    return np.clip((e - mid) / span + 0.5, -0.05, 1.05)


def scatter(ax, xy, lab, soft, r):
    order = np.argsort(np.where(lab < 0, -1, soft))
    cols = np.array([GREY if l < 0 else set_colour(l) for l in lab])
    ax.scatter(xy[order, 0], xy[order, 1], s=2.2, c=cols[order], linewidths=0, rasterized=True)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)


VIT_VIEWS = [("L1", (0, 0), 0), ("L6", (5, 5), 5), ("L9", (8, 8), 8), ("L9–L12 (block)", (8, 11), 11)]


def vit_sets(views):
    """For each (title, (a, b), layout site): the r = 8 labels, the soft SEBA vectors and the UMAP of that site."""
    meta = json.loads((VIT / "meta.json").read_text())
    lp = {tuple(p): i for i, p in enumerate(meta["layer_pairs"])}
    z = np.load(VIT / "layer_0.npz")
    umap = np.load(VIT / "umap_0.npy")                      # (12, N, 2)
    N = umap.shape[1]
    ri = meta["seba_r"].index(8)
    out = []
    for title, (a, b), site in views:
        pi = lp[(a, b)]
        lab = z["seba_label"][0, pi, ri].astype(int)
        psi = z["vecs"][0, pi].T.astype(np.float64) * z["scale"][0, pi] / 32767.0
        v = np.concatenate([np.full((N, 1), 1 / np.sqrt(N)), psi], 1)[:, :8]
        s = seba_soft(v, z["seba_rot"][0, pi, ri, :8, :8].astype(np.float64), z["seba_perm"][0, pi, ri, :8].astype(int))
        own = s[np.arange(N), np.maximum(lab, 0)] * (lab >= 0)
        out.append((title, lab, s, own, norm_layout(umap[site])))
    return out


def hero_vit_fig(views, n_img, fname, figsize, W, gap, top, umap_h, row_h, img_w, title, tsize, lsize, lab_w, row_gap):
    """One column per view: a UMAP panel coloured by its r = 8 sets, then one row per set (largest first): a colour
    chip, its size, and its n_img members with the largest SEBA value. All geometry in figure fractions."""
    imgs = np.load(VIT / "probe.npz")["images"]
    fig = plt.figure(figsize=figsize, dpi=200)
    asp = figsize[0] / figsize[1]
    for c, (vt, lab, s, own, xy) in enumerate(vit_sets(views)):
        x0 = 0.02 + c * (W + gap)
        uw = umap_h / asp
        ax = fig.add_axes([x0 + (W - uw) / 2, top - umap_h, uw, umap_h])
        scatter(ax, xy, lab, own, 8)
        ax.set_title(f"{vt}, r = 8", fontsize=tsize, fontweight="semibold", pad=4)
        sizes = np.bincount(lab[lab >= 0], minlength=8)
        for q, j in enumerate(sorted(range(8), key=lambda j: -sizes[j])):
            y = top - umap_h - row_gap - q * row_h
            ih = row_h * 0.86
            fig.patches.append(plt.Rectangle((x0, y + ih * 0.55), 0.008 * 12 / figsize[0], ih * 0.2, color=set_colour(j), transform=fig.transFigure, figure=fig))
            fig.text(x0, y + ih * 0.08, f"{sizes[j]}", fontsize=lsize, color=MUTE)
            mem = np.flatnonzero(lab == j)
            top_i = mem[np.argsort(-s[mem, j], kind="stable")][:n_img]
            lx = x0 + lab_w
            for t, i in enumerate(top_i):
                axi = fig.add_axes([lx + t * (img_w + 0.003), y, img_w, ih])
                axi.imshow(imgs[i], interpolation="lanczos"); axi.set_axis_off()
    fig.text(0.02, 0.992, title, fontsize=tsize + 1.5, fontweight="bold", va="top")
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def hero_vit():
    hero_vit_fig(VIT_VIEWS, 5, "hero_vit.png", (12, 8.6), W=0.235, gap=0.012, top=0.928, umap_h=0.245, row_h=0.073,
                 img_w=0.04, title="DINOv2 ViT-S/14: coherent sets of the same 2,000 images at four depths", tsize=11, lsize=7.5,
                 lab_w=0.02, row_gap=0.098)
    # phone: the first and the last view only, bigger thumbnails and type (shown at 390 px by <picture>)
    hero_vit_fig([VIT_VIEWS[0], VIT_VIEWS[-1]], 4, "phone/hero_vit.png", (7.1, 10.2), W=0.47, gap=0.02, top=0.935,
                 umap_h=0.27, row_h=0.075, img_w=0.098, title="DINOv2 ViT-S/14: sets at L1 and at L9–L12", tsize=13, lsize=11,
                 lab_w=0.058, row_gap=0.095)


def tok_short(s, n=8):
    s = s.replace("\n", "↵").replace("\t", "⇥")
    if s.strip() == "" and len(s) > 1:
        return f"{len(s)}sp"
    s = s if len(s) <= n else s[:n - 1] + "…"
    return "“" + s + "”"


MAMBA_VIEWS = [("residual layer 4", "point_L4", 8), ("residual, all 24 layers averaged", "Bpoint", 16), ("residual layer 24", "point_L24", 8)]


def hero_mamba():
    """Desktop: three panels side by side. Phone: the three stacked, UMAP left and the largest sets right."""
    meta = json.loads((MC / "meta.json").read_text())
    vocab = json.loads((MC / "vocab.json").read_text())["tokens"]
    sets = json.loads((MC / "sets_trained.json").read_text())
    lay = norm_layout(np.load(MC / "layouts.npz")["Bpoint"].astype(np.float64))
    z = np.load(MC / "spectra_trained.npz")
    R = meta["ranks"]
    N = len(lay)
    data = []
    for title, o, r in MAMBA_VIEWS:
        lab = z[f"{o}__seba_label"][R.index(r)].astype(int)
        data.append((title, r, lab, sorted(sets[o][str(r)], key=lambda d: -d["size"])))
    head = f"Mamba-130m: the same {N:,} token contexts at the same positions, three colourings"
    sub = "Positions: one UMAP (of the all-layer residual block), identical in all three panels. Colour: the coherent sets of the object above each panel."

    def toks(d, nt, nn, n):
        t = " ".join(tok_short(vocab[str(a)], n) for a, _ in d["tok_t"][:nt])
        return t, " ".join(tok_short(vocab[str(a)], n) for a, _ in d["tok_next"][:nn])

    def row(fig, R, x, yp, fy, PX, d, nt, nn, n, fs, cs, chip):
        """One set's row from x (figure fraction): colour chip, size, tokens at t, then → tokens at t+1, each placed
        after the measured end of the previous text."""
        fig.patches.append(plt.Rectangle((x, fy(yp) - 0.0008), chip / PX, chip / fig.bbox.height * 1.0, color=set_colour(d["set"]),
                                         transform=fig.transFigure, figure=fig))
        tx = fig.text(x + (chip + 5) / PX, fy(yp), f"{d['size']:,}", fontsize=cs, color=MUTE)
        probe = fig.text(0, 0, "0,000", fontsize=cs); wnum = probe.get_window_extent(R).width; probe.remove()
        x2 = max(tx.get_window_extent(R).x1 + 9, x * PX + chip + 5 + wnum + 9) / PX
        t, nx = toks(d, nt, nn, n)
        tt = fig.text(x2, fy(yp), t, fontsize=fs, family=MONO, color=FG)
        tn = fig.text(tt.get_window_extent(R).x1 / PX + 7 / PX, fy(yp), "→ " + nx, fontsize=fs, family=MONO, color=MUTE)
        return tn.get_window_extent(R).x1

    # ---- desktop
    PX = 1320
    H = 1010
    fig = plt.figure(figsize=(PX / 110, H / 110), dpi=110)
    R = fig.canvas.get_renderer()
    fy = lambda px: 1 - px / H                      # figure fraction from a pixel offset from the top
    W, G = 0.322, 0.0117
    for c, (title, r, lab, desc) in enumerate(data):
        x0 = 0.015 + c * (W + G)
        ax = fig.add_axes([x0 + (W - 360 / PX) / 2, fy(100 + 360), 360 / PX, 360 / H])
        scatter(ax, lay, lab, np.ones(len(lab)), r)
        ax.set_title(f"coloured by: {title}, r = {r}", fontsize=11, fontweight="semibold", pad=5)
        fig.text(x0, fy(500), "size · commonest tokens at t → at t+1", fontsize=9, color=MUTE)
        for q, d in enumerate(desc):
            end = row(fig, R, x0, 528 + q * 29, fy, PX, d, 3, 2, 7, 9.5, 9, 11)
            if end > (x0 + W) * PX + 2:
                raise SystemExit(f"hero_mamba: row {q} of {title} runs past its column ({end:.0f} px)")
    fig.text(0.015, fy(18), head, fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.015, fy(46), sub, fontsize=10.5, color=MUTE, va="top")
    fig.text(0.015, fy(H - 14), "Tokens in quotes keep their leading space; ↵ newline, Nsp = N spaces.", fontsize=9, color=MUTE)
    fig.savefig(OUT / "hero_mamba.png", dpi=110)
    plt.close(fig)

    # ---- phone: 780 px wide (shown at 390 css px), the three stacked; at most 8 sets listed per panel
    PX, rows_max, row_px, umap_px = 780, 8, 44, 250
    block = 52 + 30 + rows_max * row_px + 40
    H = 170 + 3 * block
    fig = plt.figure(figsize=(PX / 110, H / 110), dpi=110)
    R = fig.canvas.get_renderer()
    fy = lambda px: 1 - px / H
    fig.text(14 / PX, fy(12), "Mamba-130m: same contexts, same", fontsize=19, fontweight="bold", va="top")
    fig.text(14 / PX, fy(48), "positions, three colourings", fontsize=19, fontweight="bold", va="top")
    fig.text(14 / PX, fy(92), "One UMAP, identical in every panel; colour:", fontsize=15, color=MUTE, va="top")
    fig.text(14 / PX, fy(120), "the sets of the object named above it.", fontsize=15, color=MUTE, va="top")
    for c, (title, r, lab, desc) in enumerate(data):
        y0 = 170 + c * block
        fig.text(14 / PX, fy(y0), f"{title}, r = {r}", fontsize=17, fontweight="semibold", va="top")
        ax = fig.add_axes([14 / PX, fy(y0 + 48 + umap_px), umap_px / PX, umap_px / H])
        scatter(ax, lay, lab, np.ones(len(lab)), r)
        x1 = (umap_px + 26) / PX
        fig.text(x1, fy(y0 + 50), "size · at t → at t+1", fontsize=13, color=MUTE, va="top")
        for q, d in enumerate(desc[:rows_max]):
            end = row(fig, R, x1, y0 + 100 + q * row_px, fy, PX, d, 2, 1, 6, 14, 13, 15)
            if end > PX - 4:
                raise SystemExit(f"phone hero_mamba: row {q} of {title} runs past the image ({end:.0f} px)")
        if len(desc) > rows_max:
            rest = desc[rows_max:]
            fig.text(x1, fy(y0 + 100 + rows_max * row_px), f"+ {len(rest)} smaller sets ({sum(d['size'] for d in rest):,} contexts)", fontsize=13, color=MUTE)
    fig.savefig(OUT / "phone" / "hero_mamba.png", dpi=110)
    plt.close(fig)


def phone_views():
    """Phone versions of the explorer screenshots: the side panel only (the card at the right of each capture),
    which is readable at 390 px where the whole capture is not. The card's left edge is found as the first column
    right of the middle whose pixels are the card's border colour over most of the card's height."""
    from PIL import Image
    for f in sorted(OUT.glob("view_*.png")):
        im = Image.open(f).convert("RGB")
        a = np.asarray(im).astype(int)
        h, w, _ = a.shape
        rows = slice(30, h - 10)
        bg = a[h // 2, w - 2]                      # page background right of the card

        def is_line(x, colour, tol):
            return (np.abs(a[rows, x] - colour).sum(1) <= tol).mean() > 0.6

        x_r = next((x for x in range(w - 2, w // 2, -1) if not is_line(x, bg, 6)), None)
        if x_r is None:
            raise SystemExit(f"phone_views: no side card in {f.name}")
        border = a[h // 2, x_r]
        x_l = next((x for x in range(x_r - 120, w // 2, -1) if is_line(x, border, 8) and is_line(x - 1, bg, 6)), None)
        if x_l is None:
            raise SystemExit(f"phone_views: no left edge of the side card in {f.name}")
        im.crop((x_l - 4, 0, min(w, x_r + 5), min(h, 560))).save(OUT / "phone" / f.name, optimize=True)   # the top of the card: enough to read at 390 px without a long scroll


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phone").mkdir(exist_ok=True)
    hero_vit()
    hero_mamba()
    phone_views()
    print("figures:", ", ".join(sorted(str(p.relative_to(OUT)) for p in list(OUT.glob("hero_*.png")) + list(OUT.glob("phone/*.png")))))


if __name__ == "__main__":
    main()
