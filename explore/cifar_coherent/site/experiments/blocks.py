"""X4 x-blocks: which layer's structure does a whole-path block reflect?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/blocks.py r20s0b
    python3 site/check_fragment.py blocks

Reads data/<tag>/layer_<final>.npz (stored eigenpairs of every layer block, schema_v2.py: vals = lazy eigenvalues of
the block's mean operator M = mean_c (I + P_c)/2, vals[..., 0] = 1; vecs x scale / 32767 = the 8 leading nontrivial
unit eigenvectors; retention_psi[rule, pair, j, c] = psi_j^T M_c psi_j), data/<tag>/meta.json, and the cut
diagnostics of the spectral-alignment measurement (../spectral_align/data: cells.json, span_0.npy = span_k of every
cell for kernel rule 0, eig_0.npy = its eigenvalues; SPEC.md "Cut reproducibility per cell"). The alignment cells of
resnet20-relu-seed0 are the same operators as this archive's single sites (checked below: eigenvalues agree).

Computed here (seconds, local):
  overlap_k(a, c) = ||U_k(block a..logits)^T U_k(site c)||_F^2 / k, the mean cos^2 of the k principal angles between
  the leading k nontrivial eigenvectors of the averaged block a..logits and of the single site c (chance for random
  subspaces orthogonal to the constant: k / (N - 1)).
  Resolution of the cut k|k+1: a single site is resolved when span_k >= 0.5 (the alignment criterion). A block has no
  subsample rebuild; it is flagged unresolved when its eigenvalue gap lambda_k - lambda_{k+1} is no larger than the
  largest gap at which a single site's cut (k = 1..4, same checkpoint and rule) failed span_k >= 0.5. This is a proxy,
  stated as one in the fragment.
  Medoid of a block: the site c in a..logits with the largest mean overlap_k(c, c') over the sites c' of the block.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

RULE = 0                    # default kernel rule; others only in the rule comparison
KS = (1, 2, 3, 4)           # cuts computed; k = 1 (single mode) and k = 3 (span) are reported
K_SPAN = 3
SPAN_OK = 0.5               # SPEC.md (spectral_align): a cut with span_k < 0.5 is unresolved
SA = xlib.ROOT.parent / "spectral_align" / "data"
NET = "resnet20-relu-seed0"


def decode(z):
    return z["vecs"].astype(np.float64) * z["scale"][..., None].astype(np.float64) / 32767


def overlap(A, B, k):
    return float(np.linalg.norm(A[:k] @ B[:k].T) ** 2 / k)


def runs(names, starts):
    """'b6 for the blocks starting at stem and b1, b7 for b2 to b5, ...' from per-start best sites."""
    out, i = [], 0
    while i < len(names):
        j = i
        while j + 1 < len(names) and names[j + 1] == names[i]:
            j += 1
        s = starts[i] if i == j else (f"{starts[i]} and {starts[j]}" if j == i + 1 else f"{starts[i]} to {starts[j]}")
        out.append(f"{names[i]} for {s}")
        i = j + 1
    return "; ".join(out)


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    S = meta["sites"]
    assert S == xlib.SITES
    lp = meta["layer_pairs"]
    lp = json.loads(lp) if isinstance(lp, str) else lp
    idx = {tuple(p): i for i, p in enumerate(lp)}
    fin = meta["steps"][-1]
    z = np.load(src / f"layer_{fin}.npz")
    V, L, RP = decode(z), z["vals"].astype(np.float64), z["retention_psi"].astype(np.float64)
    N = V.shape[-1]
    nS, last = len(S), len(S) - 1
    starts = list(range(last))                       # a = stem .. pool; the block a..logits has >= 2 sites
    nA = len(starts)
    rules = list(range(V.shape[0]))
    rname = meta["modes"]
    blk = lambda r, a: V[r, idx[(a, last)]]
    one = lambda r, c: V[r, idx[(c, c)]]

    # ------------------------------------------------------------------ overlaps (every rule) and site x site
    O = np.array([[[[overlap(blk(r, a), one(r, c), k) for c in range(nS)] for a in starts] for k in KS] for r in rules])
    SS = np.array([[[[overlap(one(r, c), one(r, d), k) for d in range(nS)] for c in range(nS)] for k in KS] for r in rules])
    kx = {k: i for i, k in enumerate(KS)}

    # ------------------------------------------------------------------ resolution (rule 0 only: span_k is rule 0)
    cells = json.loads((SA / "cells.json").read_text())
    span, eig = np.load(SA / "span_0.npy"), np.load(SA / "eig_0.npy")
    cell = {x["site_name"]: x["index"] for x in cells if x["net"] == NET and x["step"] == fin}
    if sorted(cell) != sorted(S):
        raise SystemExit("alignment cells do not cover every site at the final checkpoint")
    for c in range(nS):       # same operators: archive vals (lazy) = alignment eigenvalues
        if np.max(np.abs(L[RULE, idx[(c, c)], :9] - eig[cell[S[c]], :9])) > 1e-4:
            raise SystemExit(f"alignment eigenvalues differ from the archive at {S[c]}: not the same operator")
    span_k = np.array([[span[cell[S[c]], k - 1] for c in range(nS)] for k in KS])           # (k, site)
    site_ok = span_k >= SPAN_OK
    gap = lambda pair, k: L[RULE, pair, k] - L[RULE, pair, k + 1]
    site_gap = np.array([[gap(idx[(c, c)], k) for c in range(nS)] for k in KS])
    failed = site_gap[~site_ok]
    if failed.size == 0:
        raise SystemExit("no single-site cut failed: the block proxy threshold is undefined; rewrite the text")
    g_thr = float(failed.max())
    kf, cf = np.argwhere((site_gap == g_thr) & ~site_ok)[0]
    blk_gap = np.array([[gap(idx[(a, last)], k) for a in starts] for k in KS])
    blk_ok = blk_gap > g_thr
    ok = site_ok[:, None, :] & blk_ok[:, :, None]                                             # (k, a, c)

    # ------------------------------------------------------------------ best site, runner-up, medoid (rule 0)
    def best(k, resolved=True, r=RULE):
        o = O[r, kx[k]].copy()
        if resolved:
            o[~ok[kx[k]]] = -1
        return [int(np.argmax(o[i])) for i in range(nA)]

    def medoid(k, r=RULE):
        """Among the block's resolved sites, the one with the largest mean overlap with the block's other resolved
        sites (unresolved sites are excluded both as candidates and from the average)."""
        out = []
        for a in starts:
            keep = np.flatnonzero(site_ok[kx[k], a:])
            m = SS[r, kx[k], a:, a:][np.ix_(keep, keep)].mean(1)
            out.append(a + int(keep[np.argmax(m)]))
        return out

    b1, b3, b3raw = best(1), best(K_SPAN), best(K_SPAN, False)
    m1, m3 = medoid(1), medoid(K_SPAN)
    if not ok[kx[1]].all():
        raise SystemExit("some k = 1 cell is unresolved: the text says every single-mode match is resolved")
    O3 = O[RULE, kx[K_SPAN]]
    # first run of starts with the same resolved best site (the early starts)
    e = 0
    while e + 1 < nA and b3[e + 1] == b3[0]:
        e += 1
    early = list(range(e + 1))
    late = list(range(e + 1, nA))
    # by construction: the block's psi_1 retains less at a than a's own psi_1 (lambda_1 of site a)
    ret_blk = np.array([RP[RULE, idx[(a, last)], 0, a] for a in starts])
    lam_own = np.array([L[RULE, idx[(a, a)], 1] for a in starts])
    n_below = int((ret_blk < lam_own + 1e-9).sum())
    if n_below != nA:
        raise SystemExit("block retention at a exceeds a's own lambda_1 somewhere: the 'by construction' line is wrong")

    # ------------------------------------------------------------------ lead figure (static: hatching + markers)
    plt, P = xlib.mpl_style()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("seq", P["sequential"])
    rows = [f"{S[a]}{xlib.ARROW}logits" for a in starts]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    for ax, k, bb, mm in ((axes[0], 1, b1, m1), (axes[1], K_SPAN, b3, m3)):
        Z = O[RULE, kx[k]]
        im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(nA):
            for c in range(nS):
                ax.text(c, i, f"{Z[i, c]:.2f}", ha="center", va="center", fontsize=6.3,
                        color=P["surface"] if Z[i, c] > 0.6 else P["fg"])
                if not ok[kx[k], i, c]:
                    ax.add_patch(Rectangle((c - .5, i - .5), 1, 1, fill=False, hatch="////", edgecolor=P["mute"], lw=0))
            ax.add_patch(Rectangle((bb[i] - .47, i - .47), .94, .94, fill=False, edgecolor=P["accent"], lw=1.6))
            ax.plot(mm[i] + .33, i - .3, marker="o", ms=3.2, color=P["accent"], mec=P["surface"], mew=.5)
        ax.set_xticks(range(nS), S, rotation=45, ha="right")
        ax.set_yticks(range(nA), rows if k == 1 else [""] * nA)
        ax.set_xlabel("single site c")
        if k == 1:
            ax.set_ylabel("averaged block a→logits")
        ax.set_title("k = 1: leading eigenvector" if k == 1 else f"k = {k}: span of the leading {k} eigenvectors")
    fig.colorbar(im, ax=axes, shrink=.8, label="span overlap")
    out = xlib.Out("blocks")
    fig.savefig(out.fig_path("lead"))
    plt.close(fig)

    # ------------------------------------------------------------------ rule comparison (static: categorical y)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), constrained_layout=True, sharey=True)
    off = np.linspace(-.24, .24, len(rules))
    for ax, k in zip(axes, (1, K_SPAN)):
        ax.plot([-.5, nA - .5], [-.5, nA - .5], color=P["mute"], ls=":", lw=1, label="c = a (the start site)")
        for r in rules:
            y = best(k, resolved=False, r=r)
            ax.plot(np.arange(nA) + off[r], y, ls="-" if r == RULE else "none", lw=.8, marker="o",
                    ms=5 if r == RULE else 4, color=P["sets"][r], label=rname[r] + (" (figures above)" if r == RULE else ""))
        ax.set_xticks(range(nA), [S[a] for a in starts], rotation=45, ha="right")
        ax.set_yticks(range(nS), S)
        ax.set_ylim(-.6, nS - .4)
        ax.set_xlabel("start site a of the block a→logits")
        ax.set_title(f"k = {k}: site c of largest overlap")
    axes[0].set_ylabel("single site c")
    axes[1].legend(loc="lower right", fontsize=6.5)
    fig.savefig(out.fig_path("rules"))
    plt.close(fig)

    # ------------------------------------------------------------------ disclosure chart: retention profiles
    ser = [dict(name=f"ψ₁ of {S[a]}{xlib.ARROW}logits", y=[RP[RULE, idx[(a, last)], 0, c] for c in range(nS)],
                color=P["ordinal"][min(len(P["ordinal"]) - 1, int(round(a * (len(P["ordinal"]) - 1) / (nA - 1))))],
                width=1.6) for a in (0, 2, 4, 6, 8, 10)]
    ser.append(dict(name="each site's own λ₁ (largest possible retention at that site)",
                    y=[L[RULE, idx[(c, c)], 1] for c in range(nS)], color=P["fg"], dash=True, width=1.4))
    out.chart("retention", xlib.lines(list(range(nS)), ser, "site c", "retention ψ₁ᵀ M_c ψ₁ (lazy scale)",
                                      yrange=[float(np.floor(min(min(s["y"]) for s in ser) * 20) / 20), 1.0],
                                      xticklabels=S))

    # ------------------------------------------------------------------ facts
    f = out.fact
    nm = lambda ii: [S[i] for i in ii]
    f("rule", rname[RULE])
    f("n_starts", nA)
    f("chance3", K_SPAN / (N - 1), 4)
    f("span1_min", span_k[kx[1]].min(), 2)
    f("blockgap1_min", blk_gap[kx[1]].min(), 4)
    f("gap_thr", g_thr, 4)
    f("gap_thr_where", f"{S[cf]}, cut {KS[kf]}|{KS[kf] + 1}")
    f("span_thr_value", span_k[kf, cf], 2)
    bad3 = [S[c] for c in range(nS) if not site_ok[kx[K_SPAN], c]]
    if not bad3:
        raise SystemExit("no single site fails the 3|4 cut: rewrite the hatching sentence")
    f("k3.unresolved_sites", ", ".join(bad3))
    f("k3.unresolved_span", ", ".join(xlib.fmt(span_k[kx[K_SPAN], S.index(s)], 2) for s in bad3))
    if not blk_ok[kx[K_SPAN]].all():
        raise SystemExit("a block's 3|4 cut is flagged: the text says every block's 3|4 cut passes the proxy")
    f("k3.blockgap_min", blk_gap[kx[K_SPAN]].min(), 4)
    f("k3.blockgap_min_where", f"{S[starts[int(np.argmin(blk_gap[kx[K_SPAN]]))]]}{xlib.ARROW}logits")
    f("k1.best", runs(nm(b1), [f"{S[a]}{xlib.ARROW}logits" for a in starts]))
    f("k3.best", runs(nm(b3), [f"{S[a]}{xlib.ARROW}logits" for a in starts]))
    f("k3.early_site", S[b3[0]])
    f("k3.early_first", S[early[0]])
    f("k3.early_last", S[early[-1]])
    f("k3.early_best_lo", min(O3[i, b3[i]] for i in early), 2)
    f("k3.early_best_hi", max(O3[i, b3[i]] for i in early), 2)
    f("k3.early_own_lo", min(O3[i, a] for i, a in enumerate(early)), 2)
    f("k3.early_own_hi", max(O3[i, a] for i, a in enumerate(early)), 2)
    second = lambda i: np.sort(np.where(ok[kx[K_SPAN], i], O3[i], -1))[-2]
    f("k3.early_margin_max", max(O3[i, b3[i]] - second(i) for i in early), 2)
    run_sites = sorted({S[c] for i in early for c in range(nS)
                        if ok[kx[K_SPAN], i, c] and O3[i, b3[i]] - O3[i, c] <= 0.02 + 1e-9}, key=S.index)
    f("k3.early_within2", ", ".join(run_sites))
    f("k3.late_first", S[late[0]])
    f("k3.late_sites", " and ".join(sorted({S[b3[i]] for i in late}, key=S.index)))
    f("k3.late_best_lo", min(O3[i, b3[i]] for i in late), 2)
    f("k3.late_best_hi", max(O3[i, b3[i]] for i in late), 2)
    f("k3.late_raw_site", ", ".join(sorted({S[b3raw[i]] for i in late}, key=S.index)))
    f("k3.late_raw_hi", max(O3[i, b3raw[i]] for i in late), 2)
    f("k1.n_med_agree", sum(int(x == y) for x, y in zip(b1, m1)))
    f("k3.n_med_agree", sum(int(x == y) for x, y in zip(b3, m3)))
    f("k3.early_med_sites", " and ".join(sorted({S[m3[i]] for i in early}, key=S.index)))
    f("k3.early_med_lo", min(O3[i, m3[i]] for i in early), 2)
    f("k3.early_med_hi", max(O3[i, m3[i]] for i in early), 2)
    f("n_below", n_below)
    # rule comparison
    f("n_rules", len(rules))
    f("rules.k3.stem", "; ".join(f"{rname[r]}: {S[best(K_SPAN, False, r)[0]]}" for r in rules))
    f("rules.k1.stem", "; ".join(f"{rname[r]}: {S[best(1, False, r)[0]]}" for r in rules))
    r3 = 3
    f("rule3", rname[r3])
    f("rule3.lam_early_lo", min(L[r3, idx[(c, c)], 1] for c in range(S.index("pool"))), 2)
    f("rule3.lam_early_hi", max(L[r3, idx[(c, c)], 1] for c in range(S.index("pool"))), 2)
    f("rule3.lam_pool", L[r3, idx[(S.index("pool"),) * 2], 1], 2)
    f("rule3.lam_logits", L[r3, idx[(last, last)], 1], 2)
    f("rule0.lam_early_lo", min(L[RULE, idx[(c, c)], 1] for c in range(S.index("b9"))), 2)
    f("rule0.lam_early_hi", max(L[RULE, idx[(c, c)], 1] for c in range(S.index("b9"))), 2)

    # ------------------------------------------------------------------ to App 4 (Reference charts)
    app4 = {}
    unres = lambda k: ", ".join([S[c] for c in range(nS) if not site_ok[kx[k], c]]
                                + [f"{S[a]}{xlib.ARROW}logits" for i, a in enumerate(starts) if not blk_ok[kx[k], i]]) or "none"
    app4["overlap_k"] = dict(xlib.heat(
        [O[RULE, kx[k]] for k in KS], rows, S, "span overlap (mean cos² of principal angles)",
        "averaged block a→logits", "single site c", slider=dict(label="leading k", values=list(KS)), fmt="2f"),
        caption=f"<b>Plotted:</b> mean cos² of the k principal angles between the leading k nontrivial eigenvectors "
                f"of the averaged block a→logits (rows) and of the single site c (columns), k = 1 to {KS[-1]} "
                f"(slider), final checkpoint, kernel rule {rname[RULE]}. Unresolved cuts are not marked here; they "
                + "are, by k: " + "; ".join(f"k = {k}: {unres(k)}" for k in KS) + ".")
    app4["site_site"] = dict(xlib.heat(
        [SS[RULE, kx[k]] for k in KS], S, S, "span overlap (mean cos² of principal angles)", "single site c",
        "single site c′", slider=dict(label="leading k", values=list(KS)), fmt="2f"),
        caption=f"<b>Plotted:</b> mean cos² of the k principal angles between the leading k nontrivial eigenvectors "
                f"of single sites c and c′, k = 1 to {KS[-1]} (slider), final checkpoint, kernel rule {rname[RULE]}; "
                "the medoid of a block a→logits is the row, among a to logits, with the largest mean over columns "
                "a to logits, both restricted to sites whose cut k|k+1 is resolved.")
    app4["cuts"] = dict(xlib.lines(
        list(range(nS)), [dict(name=f"span_{k} (cut {k}|{k + 1})", y=span_k[kx[k]], color=P["sets"][i])
                          for i, k in enumerate(KS)],
        "single site c", "span_k (min cos² between subsample rebuilds)", xticklabels=S,
        refline={"y": SPAN_OK, "label": "resolution threshold"}),
        caption=f"<b>Plotted:</b> cut reproducibility span_k of each single site (minimum cos² principal angle "
                f"between the leading k spans of two overlapping label-free subsample rebuilds; spectral-alignment "
                f"measurement), k = 1 to {KS[-1]}, final checkpoint, kernel rule {rname[RULE]}; reference line at "
                f"{SPAN_OK}.")
    app4["block_gaps"] = dict(xlib.lines(
        list(range(nA)), [dict(name=f"block gap λ_{k} − λ_{k + 1}", y=blk_gap[kx[k]], color=P["sets"][i])
                          for i, k in enumerate(KS)],
        "start site a of the block a→logits", "eigenvalue gap (lazy scale)", xticklabels=[S[a] for a in starts],
        yrange=xlib.nice_range([blk_gap], step=0.005), refline={"y": g_thr, "label": "proxy threshold"}),
        caption=f"<b>Plotted:</b> eigenvalue gaps λ_k − λ_(k+1) of the averaged block a→logits (lazy eigenvalues of "
                f"the mean of (I + P_c)/2), k = 1 to {KS[-1]}, final checkpoint, kernel rule {rname[RULE]}; reference "
                f"line: the largest single-site gap whose cut failed span_k ≥ {SPAN_OK} "
                f"({xlib.fmt(g_thr, 4)}, {S[cf]}, cut {KS[kf]}|{KS[kf] + 1}).")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
