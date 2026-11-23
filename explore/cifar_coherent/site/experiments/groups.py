"""X1 x-groups: what groups of images does the last stage of the network treat alike at the end of training?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/groups.py r20s0b
    python3 site/check_fragment.py groups

Reads data/<tag>/layer_<final>.npz (schema v2: vals, vecs, scale, seba_label, seba_rot, seba_perm, retention_seba),
data/<tag>/meta.json, data/<tag>/probe.npz (labels: validation only) and the baselines archive
data/<baselines>/reps.json (null spectra of the representative block b7..b9; default r20s0c). Writes
assets/x/groups.json, assets/x/groups.facts.json and assets/fig/x_groups_ranks.png. Local computation only: class
make-up, matched Jaccard (with a label-permutation null and the size ceiling), conditional entropies between the stored
SEBA labels at different ranks, and an exact permutation reference for retention.

Scales: the archive stores lazy eigenvalues lambda_M of M = (I + P)/2 (vals[..., 0] = 1 is the trivial mode); every
spectrum here is converted to lambda_P = 2 lambda_M - 1, the scale of reps.json and of the bn-spectra chart.
Membership always comes from the stored seba_label (int8, -1 = unassigned; ri indexes meta.seba_r), never from the
int16 vecs. The vecs are decoded only for the retention reference, which needs the values of each SEBA vector.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

RULE = 0                 # default kernel rule (kNN graph, k = 15)
A, B = 7, 10             # the block b7..pool (site indices)
NULL_A, NULL_B = 7, 9    # the representative block b7..b9, fixed before class outcomes; the only one with null draws
R_MAIN = 4               # the resolution read here (a resolution, not a discovered number; see the gap facts)
N_PERM = 1000            # label permutations for the matched-Jaccard null
GRID_N = 22              # images per set in the image grids
BASELINES = "r20s0c"
NULLS = {"N3": "trajectory shuffle", "N2a": "spectrum-matched Gaussian"}   # reps.json ids -> names (js/baselines.js)


def entropy(a):
    c = np.bincount(a).astype(float)
    c = c[c > 0] / len(a)
    return float(-(c * np.log2(c)).sum())


def cond_entropy(a, b):
    """H(a | b) in bits; a, b non-negative ints."""
    joint = a.astype(np.int64) * (int(b.max()) + 1) + b
    return entropy(np.unique(joint, return_inverse=True)[1]) - entropy(np.unique(b, return_inverse=True)[1])


def seba_values(z, pair, ri, r, N):
    """The stored SEBA vectors (clipped, max 1), decoded as site/make_figures.py does (seba_reconstruct)."""
    psi = [z["vecs"][RULE, pair, k - 1].astype(np.float64) * z["scale"][RULE, pair, k - 1] / 32767.0 for k in range(1, r)]
    v = np.column_stack([np.full(N, 1 / np.sqrt(N))] + psi)
    zz = v @ z["seba_rot"][RULE, pair, ri, :r, :r].astype(np.float64)
    s = np.sign(zz) * np.clip(np.abs(zz) - 0.99 / np.sqrt(N), 0, None)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0))
    sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return np.clip(s[:, z["seba_perm"][RULE, pair, ri, :r]], 0, None)


def alluvial(path, L, ranks, r_col, N, P):
    """Static alluvial of the stored labels across ranks; ribbons coloured by the set at rank r_col."""
    plt, _ = xlib.mpl_style()
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    gap, barw = 26.0, 0.16
    ci = ranks.index(r_col)
    col4 = L[ci]
    unc = P["unassigned"]
    colour = lambda c: P["sets"][c] if c >= 0 else unc
    # segment order in each column: assigned sets by the mean rank-r_col set of their members (fewest crossings
    # with the coloured column), unassigned last
    order, pos = [], []
    for i, r in enumerate(ranks):
        l = L[i]
        keys = []
        for j in range(r):
            mem = col4[l == j]
            mem = mem[mem >= 0]
            keys.append((mem.mean() if mem.size else r_col, j))
        o = [j for _, j in sorted(keys)] + [-1]
        y, p = 0.0, {}
        for j in o:
            n = int((l == j).sum())
            p[j] = (y, y + n)
            y += n + gap
        order.append(o)
        pos.append(p)
    H = max(p[order[i][-1]][1] for i, p in enumerate(pos))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(ranks), dtype=float)
    for i in range(len(ranks) - 1):
        l0, l1 = L[i], L[i + 1]
        cols = [c for c in range(r_col)] + [-1]
        # out-order at the left: destination order, then colour; in-order at the right: source order, then colour
        out_y = {j: pos[i][j][0] for j in order[i]}
        in_y = {j: pos[i + 1][j][0] for j in order[i + 1]}
        ribbons = []
        for s in order[i]:
            for d in order[i + 1]:
                for c in cols:
                    n = int(((l0 == s) & (l1 == d) & (col4 == c)).sum())
                    if n:
                        ribbons.append((s, d, c, n))
        left = {}
        for s, d, c, n in ribbons:                                   # already in (s, d-order, c) order
            left[(s, d, c)] = out_y[s]
            out_y[s] += n
        rib_in = sorted(ribbons, key=lambda t: (order[i + 1].index(t[1]), order[i].index(t[0]), cols.index(t[2])))
        right = {}
        for s, d, c, n in rib_in:
            right[(s, d, c)] = in_y[d]
            in_y[d] += n
        x0, x1 = xs[i] + barw / 2, xs[i + 1] - barw / 2
        t = np.linspace(0, 1, 40)
        sm = t * t * (3 - 2 * t)
        for s, d, c, n in ribbons:
            ya, yb = left[(s, d, c)], right[(s, d, c)]
            top = np.column_stack([x0 + (x1 - x0) * t, ya + (yb - ya) * sm])
            bot = np.column_stack([x1 - (x1 - x0) * t, (yb + n) + ((ya + n) - (yb + n)) * sm])
            verts = np.vstack([top, bot, top[:1]])
            codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour(c), edgecolor="none",
                                   alpha=0.55 if c >= 0 else 0.8, zorder=1))
    for i, r in enumerate(ranks):
        for j in order[i]:
            y0, y1 = pos[i][j]
            if y1 <= y0:
                continue
            ax.add_patch(plt.Rectangle((xs[i] - barw / 2, y0), barw, y1 - y0, facecolor=P["fg"] if j >= 0 else unc,
                                       edgecolor=P["surface"], linewidth=0.6, zorder=2))
            if y1 - y0 > 55:
                ax.text(xs[i], (y0 + y1) / 2, str(j + 1) if j >= 0 else "u", ha="center", va="center",
                        fontsize=7, color=P["surface"] if j >= 0 else P["fg"], zorder=3)
    ax.set_xlim(-0.35, len(ranks) - 0.65)
    ax.set_ylim(H + 10, -10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"r = {r}" for r in ranks])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0)
    hand = [plt.Rectangle((0, 0), 1, 1, facecolor=P["sets"][c], alpha=0.55) for c in range(r_col)] + \
           [plt.Rectangle((0, 0), 1, 1, facecolor=unc)]
    ax.legend(hand, [f"set {c + 1} at r = {r_col}" for c in range(r_col)] + [f"unassigned at r = {r_col}"],
              loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=r_col + 1, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites, ranks = meta["steps"], meta["sites"], meta["seba_r"]
    assert sites == xlib.SITES, "unexpected site list"
    lp = meta["layer_pairs"]
    lp = json.loads(lp) if isinstance(lp, str) else lp
    fin = steps[-1]
    z = np.load(src / f"layer_{fin}.npz")
    pair, pair_null = lp.index([A, B]), lp.index([NULL_A, NULL_B])
    labels = np.load(src / "probe.npz")["labels"].astype(int)           # validation only
    classes = meta["classes"]
    N = int(meta["n"])
    L = z["seba_label"][RULE, pair].astype(int)                          # (len(ranks), N), -1 = unassigned
    ri = ranks.index(R_MAIN)
    l4 = L[ri]
    P = xlib.PALETTE
    out = xlib.Out("groups")
    blk = f"{sites[A]}..{sites[B]}"
    blk_null = f"{sites[NULL_A]}..{sites[NULL_B]}"
    out.fact("rule", meta["modes"][RULE])
    out.fact("grid_n", GRID_N)

    # ------------------------------------------------------------------ lead: class make-up of the r = 4 sets
    rows = [l4 == j for j in range(R_MAIN)] + [l4 < 0]
    Z = np.array([[int((m & (labels == c)).sum()) for c in range(len(classes))] for m in rows])
    per_class = np.bincount(labels, minlength=len(classes))
    assert (per_class == per_class[0]).all(), "probe is not label-stratified"
    q = json.loads((xlib.SITE / "assets" / "questions.json").read_text())["q_classmix"]["named"]
    ref = q.get(f"layer_t{len(steps) - 1}_b7-pool_r{R_MAIN}")
    if ref is not None and ref["z"] != Z.tolist():
        raise SystemExit("class make-up differs from assets/questions.json: check the pair / rule indices")
    out.chart("lead", xlib.heat(
        Z.tolist(), [f"set {j + 1}" for j in range(R_MAIN)] + ["unassigned"], classes, "images", "SEBA set",
        "CIFAR class (validation)", zrange=[0, int(per_class[0])], fmt="int",
        title=f"{blk}, step {fin:,}, r = {R_MAIN}"))
    sizes = [int(m.sum()) for m in rows[:-1]]
    for j, s in enumerate(sizes):
        out.fact(f"set{j + 1}.size", s)
    out.fact("n_unassigned", int(rows[-1].sum()))
    out.fact("per_class", int(per_class[0]))
    # classes with a majority of their probe images in one set (a count; no set is named)
    maj = int(sum(Z[:R_MAIN, c].max() > per_class[c] / 2 for c in range(len(classes))))
    out.fact("n_classes_majority", maj)
    out.fact("n_classes", len(classes))

    # matched Jaccard against the classes, size ceiling, label-permutation null (validation)
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(N) for _ in range(N_PERM)])
    # matched-Jaccard nulls permute the labels among the assigned images only: set sizes and which images are
    # unassigned are both kept (the unassigned mask is fixed)
    asg = np.flatnonzero(l4 >= 0)
    perms_a = np.tile(np.arange(N), (N_PERM, 1))
    for p_ in perms_a:
        p_[asg] = asg[rng.permutation(len(asg))]
    assert ((l4[perms_a] < 0) == (l4 < 0)[None]).all()
    mj, _, null = xlib.matched_jaccard(l4, labels, R_MAIN, len(classes), perms=perms_a)
    ceil = xlib.size_ceiling(l4, labels, R_MAIN, len(classes))
    nm, nlo, nhi = xlib.null_summary(null)
    out.fact("vs_classes", mj, 2)
    out.fact("vs_classes_ceiling", ceil, 2)
    out.fact("vs_classes_null_mean", nm, 2)
    out.fact("vs_classes_null_q975", nhi, 2)
    out.fact("n_perm", N_PERM)

    # ------------------------------------------------------------------ persistence: retention in each snapshot
    ret = z["retention_seba"][RULE, pair, ri, :R_MAIN]                    # (sets, 12 sites), lazy M_c
    inblk = ret[:, A:B + 1]
    out.fact("retention_min", float(inblk.min()), 2)
    out.fact("retention_max", float(inblk.max()), 2)
    # Exact permutation reference: for a site walk P that is doubly stochastic with zero diagonal (rule 0), the mean
    # of f^T M f / f^T f over uniform relabellings of the images is 1/2 + (S1^2 - S2) / (2 (N - 1) S2), S1 = sum f,
    # S2 = sum f^2. It depends only on the values of f, so it is the same at every site.
    S = seba_values(z, pair, ri, R_MAIN, N)
    hard = np.where(S.max(1) > 0, S.argmax(1), -1)
    if not (hard == l4).all():
        raise SystemExit("decoded SEBA vectors do not reproduce the stored labels")
    S1, S2 = S.sum(0), (S ** 2).sum(0)
    ret_ref = 0.5 + (S1 ** 2 - S2) / (2 * (N - 1) * S2)
    out.fact("retention_ref_max", float(ret_ref.max()), 2)
    out.fact("retention_ref_min", float(ret_ref.min()), 2)
    # how far the null-tested block's own sets match this block's (same rule, same r, label-free)
    l4n = z["seba_label"][RULE, pair_null, ri].astype(int)
    mjn, _, nulln = xlib.matched_jaccard(l4, l4n, R_MAIN, R_MAIN, perms=perms_a)
    out.fact("vs_null_blk", mjn, 2)
    out.fact("vs_null_blk_null_q975", xlib.null_summary(nulln)[2], 2)
    # seed reproducibility: the same block, rule 0, r = 4, on the second training seed (baselines archive, source I3;
    # kernel rule 0, same probe, same layer-pair order)
    meta_c = json.loads((xlib.DATA / BASELINES / "meta_c.json").read_text())
    if meta_c["kernel_rule"] != RULE or meta_c["seba_r"] != ranks:
        raise SystemExit("seed-1 archive uses a different rule or rank list")
    zs = np.load(xlib.DATA / BASELINES / f"I3_seed1_layer_{fin}.npz")
    l4s = zs["seba_label"][pair, ri].astype(int)
    mjs, _, nulls_ = xlib.matched_jaccard(l4, l4s, R_MAIN, R_MAIN, perms=perms_a)
    out.fact("vs_seed1", mjs, 2)
    out.fact("vs_seed1_null_q975", xlib.null_summary(nulls_)[2], 2)
    out.fact("seed1.sizes", ", ".join(f"{int((l4s == j).sum()):,}" for j in range(R_MAIN)))
    out.fact("seed1.n_unassigned", int((l4s < 0).sum()))

    # ------------------------------------------------------------------ spectra (lambda_P) side by side
    lamP = lambda p: 2.0 * z["vals"][RULE, p, 1:].astype(float) - 1.0     # 8 non-trivial values
    sp_main, sp_null = lamP(pair), lamP(pair_null)
    reps = json.loads((xlib.DATA / BASELINES / "reps.json").read_text())[f"{sites[NULL_A]}-{sites[NULL_B]}"]["nulls"]
    if reps["cells"] != [f"{fin}_{c}" for c in range(NULL_A, NULL_B + 1)]:
        raise SystemExit("reps.json b7-b9 is not the final-checkpoint block b7..b9")
    if np.abs(np.array(reps["real"]["lambda_P"]) - sp_null).max() > 1e-4:
        raise SystemExit("archive spectrum of b7..b9 (2 lambda_M - 1) differs from reps.json lambda_P")
    x = list(range(1, 9))
    ticks = [f"λ{k}" for k in x]
    yr = (0, 1)
    out.chart("spectrum_main", xlib.lines(
        x, [dict(name=f"{blk}, real block", y=sp_main, color=P["fg"], width=2.2, dots=3)],
        "", "λ_P = 2λ_M − 1", yrange=yr, xticklabels=ticks))
    ser = [dict(name=f"{blk_null}, real block", y=sp_null, color=P["fg"], width=2.2, dots=3)]
    band_hi = {}
    for (k, name), col in zip(NULLS.items(), (P["sets"][0], P["sets"][1])):
        bd = reps["bands"][k]["lambda_P"]
        n = len(reps[k])
        ser.append(dict(name=f"{name} (median of {n} draws)", y=[b["q50"] for b in bd], color=col, dash=True,
                        width=1.6, band=[[b["q05"] for b in bd], [b["q95"] for b in bd]]))
        band_hi[k] = max(b["q95"] for b in bd)
        out.fact(f"null.{k.lower()}.n", n)
        out.fact(f"null.{k.lower()}.q95_max", band_hi[k], 2)
    out.chart("spectrum_null", xlib.lines(
        x, ser, "", "λ_P = 2λ_M − 1", yrange=yr, xticklabels=ticks, band_mode="each",
        band_label="bands: 5th to 95th percentile over the null draws"))
    out.fact("main.lam1", sp_main[0], 2)
    out.fact("main.lam8", sp_main[-1], 2)
    out.fact("null_blk.lam1", sp_null[0], 2)
    out.fact("null_blk.lam8", sp_null[-1], 2)
    out.fact("null_blk.min_excess", float(sp_null.min() - max(band_hi.values())), 2)
    # within-class trajectory shuffle (uses the labels: validation null), for the caveat
    n4 = reps["bands"]["N4"]["lambda_P"]
    out.fact("null.n4.lam1_median", n4[0]["q50"], 2)
    out.fact("null.n4.n", len(reps["N4"]))
    # gaps on the lambda_P scale: r sets use the trivial mode and lambda_1..lambda_{r-1}, so r cuts after lambda_{r-1}
    gaps = -np.diff(sp_main)
    for r in (2, 3, 4):
        out.fact(f"gap_r{r}", gaps[r - 2], 3)
    out.fact("gap_max", gaps.max(), 3)
    out.fact("gap_max_after", f"λ{int(gaps.argmax()) + 1}")

    # ------------------------------------------------------------------ overlap across ranks
    alluvial(out.fig_path("ranks"), L, ranks, R_MAIN, N, P)
    cons = list(zip(range(len(ranks) - 1), range(1, len(ranks))))
    frac_all, frac_both, cover, frac_perm = [], [], [], []
    for i, j in cons:
        a, b = L[i] + 1, L[j] + 1                                          # unassigned = its own category (0)
        frac_all.append(cond_entropy(a, b) / entropy(a))
        both = (L[i] >= 0) & (L[j] >= 0)
        frac_both.append(cond_entropy(L[i][both], L[j][both]) / entropy(L[i][both]))
        cover.append(both.mean())
        pr = [cond_entropy(a, b[perms[p]]) / entropy(a) for p in range(200)]
        frac_perm.append(float(np.percentile(pr, 2.5)))
    xl = [f"{ranks[i]} | {ranks[j]}" for i, j in cons]
    out.chart("rank_entropy", xlib.lines(
        list(range(len(cons))), [
            dict(name="every image (unassigned as its own category)", y=frac_all, color=P["sets"][0], width=2.2, dots=3),
            dict(name="images assigned at both ranks", y=frac_both, color=P["sets"][1], width=2.2, dots=3),
            dict(name="finer labels permuted (2.5th percentile of 200)", y=frac_perm, color=P["mute"], dash="dot", width=1.4),
        ], "coarser rank r | finer rank r′", "H(sets at r | sets at r′) / H(sets at r)", yrange=(0, 1), xticklabels=xl))
    out.fact("rank.all_min", min(frac_all), 2)
    out.fact("rank.all_max", max(frac_all), 2)
    out.fact("rank.both_min", min(frac_both), 2)
    out.fact("rank.both_max", max(frac_both), 2)
    out.fact("rank.both_max_pair", f"r = {xl[int(np.argmax(frac_both))].replace(' | ', ' given r′ = ')}")
    out.fact("rank.cover_min", min(cover), 0, pct=True)
    out.fact("rank.cover_max", max(cover), 0, pct=True)
    out.fact("rank.perm_min", min(frac_perm), 2)
    # nesting seen from the finer side: for each set at the finer rank, the share of its members that are assigned at
    # the coarser rank lying in a single coarser set; and the share of its members unassigned at the coarser rank
    nest = []
    for i, j in cons:
        for f in range(ranks[j]):
            m = L[j] == f
            bc = np.bincount(L[i][m] + 1, minlength=ranks[i] + 1)
            nest.append((bc[1:].max() / bc[1:].sum(), ranks[i], ranks[j], f, bc[0] / m.sum()))
    nest.sort()
    worst = nest[0]
    out.fact("rank.nest_worst", f"set {worst[3] + 1} at r = {worst[2]}")
    out.fact("rank.nest_worst_share", worst[0], 0, pct=True)
    out.fact("rank.nest_worst_coarse", f"r = {worst[1]}")
    out.fact("rank.nest_others_min", nest[1][0], 0, pct=True)
    big_u = max(nest, key=lambda t: t[4])
    out.fact("rank.most_from_unassigned", f"set {big_u[3] + 1} at r = {big_u[2]}")
    out.fact("rank.most_from_unassigned_share", big_u[4], 0, pct=True)
    out.fact("rank.most_from_unassigned_coarse", f"r = {big_u[1]}")
    # the r = 3 -> r = 4 step, as counts
    i3, i4 = ranks.index(3), ranks.index(R_MAIN)
    C34 = np.array([[int(((L[i3] == a) & (L[i4] == b)).sum()) for b in range(R_MAIN)] for a in range(3)])
    src = int(np.argmax([(row > 0.2 * row.sum()).sum() for row in C34]))
    parts = np.argsort(-C34[src])[:2]
    out.fact("r3to4.src", f"set {src + 1}")
    out.fact("r3to4.parts", f"set {parts[0] + 1} ({C34[src, parts[0]]:,} images) and set {parts[1] + 1} "
                            f"({C34[src, parts[1]]:,})")
    out.fact("ranks", ", ".join(str(r) for r in ranks))
    for r in ranks:
        out.fact(f"r{r}.n_unassigned", int((L[ranks.index(r)] < 0).sum()))

    # ------------------------------------------------------------------ to App 4 (Reference charts)
    app4 = {}
    rser = [dict(name=f"set {j + 1}: retention", y=ret[j], color=P["sets"][j], width=2) for j in range(R_MAIN)]
    rser += [dict(name=f"set {j + 1}: permutation reference", y=[float(ret_ref[j])] * len(sites), color=P["sets"][j],
                  dash=True, width=1.2) for j in range(R_MAIN)]
    app4["retention_by_site"] = dict(xlib.lines(
        list(range(len(sites))), rser, "site (snapshot at the final checkpoint)", "retention f^T M_c f / f^T f",
        yrange=xlib.nice_range([ret, ret_ref], step=0.05), xticklabels=sites),
        caption=f"<b>Plotted:</b> retention $f^\\top M_c f / f^\\top f$ (y) of each SEBA vector $f$ of layer block "
                f"{blk} at $r = {R_MAIN}$ (colour: set 1 to {R_MAIN}), in the lazy walk $M_c$ of each single snapshot "
                f"$c$ (x: site), step {fin:,}, kernel rule {meta['modes'][RULE]}; the block's snapshots are "
                f"{', '.join(sites[A:B + 1])}. Dashed: the mean retention of the same vector over uniform relabellings "
                "of the images, exact for a doubly stochastic walk with zero diagonal "
                "($\\tfrac12 + (S_1^2 - S_2)/(2(N-1)S_2)$, $S_1 = \\sum f$, $S_2 = \\sum f^2$).")
    nr = len(ranks)
    Hm = [[None] * nr for _ in range(nr)]
    Hb = [[None] * nr for _ in range(nr)]
    for i in range(nr):
        for j in range(i + 1, nr):
            a, b = L[i] + 1, L[j] + 1
            Hm[i][j] = cond_entropy(a, b) / entropy(a)
            both = (L[i] >= 0) & (L[j] >= 0)
            Hb[i][j] = cond_entropy(L[i][both], L[j][both]) / entropy(L[i][both])
    for key, M_, what in (("rank_entropy_all", Hm, "every image, unassigned as its own category"),
                          ("rank_entropy_assigned", Hb, "images assigned at both ranks")):
        app4[key] = dict(xlib.heat(M_, [f"r = {r}" for r in ranks], [f"r′ = {r}" for r in ranks],
                                   "H(r | r′) / H(r)", "coarser rank", "finer rank", fmt="2f"),
                         caption=f"<b>Plotted:</b> conditional entropy of the stored SEBA labels of layer block {blk} "
                                 f"at rank $r$ (rows) given those at rank $r'$ (columns), divided by the entropy at $r$, "
                                 f"for every $r < r'$ ({what}); step {fin:,}, kernel rule {meta['modes'][RULE]}. "
                                 "Zero: every set at $r'$ lies inside one set at $r$. Blank: not computed ($r \\ge r'$).")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
