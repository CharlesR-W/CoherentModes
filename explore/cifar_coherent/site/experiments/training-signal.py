"""X5 x-training-signal: do the images grouped alike in activations also have similar loss gradients?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/training-signal.py r20s0b
    python3 site/check_fragment.py training-signal --strict

Reads (all local, no new Modal run):
  data/<tag>/gradgram_<step>.npy   cosine Gram of per-image cross-entropy loss gradients over ALL trainable
                                   parameters (measure_v2.py, vmap(grad(loss))); one per checkpoint, float16.
                                   These do not depend on the site: a chart over sites changes only the grouping.
  data/<tag>/time_<site>.npz       seba_label[rule, time pair, ri, n] (int8, -1 = unassigned), ri over meta.seba_r
  data/<tag>/probe.npz             labels (validation only: the class-union reference and the within-class null)
  data/<tag>/points_<step>.npz     grad_cos (cosine of each image's gradient with the mean probe gradient; App 4)
  data/kernels/compare_ggn.json    span overlaps of each site's leading modes with the leading modes of the
                                   GGN / NTK / loss-gradient sample kernels (SPEC_DOT.md sections 2 and 4)

Computed here (local, about a minute): the within-minus-between gradient cosine of the time-block sets, of their
class-union grouping on the same assigned images, and three nulls (set-label permutation among the assigned images,
for the sets and for the class union; within-class permutation of the set labels, which keeps each class's per-set
counts and the unassigned mask). Writes assets/x/training-signal.json and assets/x/training-signal.facts.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

RULE = 0                 # kernel rule of the sets and of the site operators (kNN graph, k = 15)
R_MAIN, R_ALT = 4, 2     # SEBA ranks; r = 4 is the site's default rank, r = 2 is reported beside it
SITE_MAIN = "b9"
N_PERM = 200             # permutations per null at b9 (lead); every-site chart uses N_PERM_SITES
N_PERM_SITES = 100
SEED = 0
GGN_R = (4, 10)          # span ranks stored in compare_ggn.json


def within_between(G, L, r):
    """Pair-weighted within / between mean of G for each labeling row of L (P, N), -1 excluded (diagonal excluded).
    Same definition as xlib.grad_within_between, batched over arbitrary labelings. Returns diff (P,)."""
    P, N = L.shape
    A = (L >= 0)
    Af = A.T.astype(np.float32)                                   # (N, P)
    H = np.zeros((N, P * r), np.float32)
    ii, pp = np.nonzero(A.T)
    H[ii, pp * r + L[pp, ii]] = 1
    GH = G @ H
    win = (H * GH).sum(0, dtype=np.float64).reshape(P, r).sum(1)
    tot = (Af * (G @ Af)).sum(0, dtype=np.float64)
    dg = np.diagonal(G).astype(np.float64) @ A.T.astype(np.float64)
    na = A.sum(1).astype(np.float64)
    sq = np.array([(np.bincount(l[l >= 0], minlength=r).astype(np.float64) ** 2).sum() for l in L])
    within = (win - dg) / (sq - na)
    between = (tot - win) / (na * na - sq)
    return within - between


def perm_assigned(l, rng, P):
    """P relabelings: the labels of the assigned images permuted among the assigned images (sizes, mask kept)."""
    I = np.flatnonzero(l >= 0)
    out = np.repeat(l[None], P, 0)
    for p in range(P):
        out[p, I] = l[rng.permutation(I)]
    return out


def perm_within_class(l, cls, rng, P):
    """P relabelings: within each class, the set labels of its assigned images permuted among them. Keeps each
    class's per-set counts (hence every set's class make-up) and the unassigned mask."""
    out = np.repeat(l[None], P, 0)
    groups = [np.flatnonzero((cls == c) & (l >= 0)) for c in np.unique(cls)]
    for p in range(P):
        for I in groups:
            out[p, I] = l[rng.permutation(I)]
    return out


def class_union(l, cls, r):
    """Each class given wholly to the set holding most of its images; restricted to the images l assigns."""
    cnt = np.zeros((r, cls.max() + 1), int)
    for j in range(r):
        cnt[j] = np.bincount(cls[l == j], minlength=cls.max() + 1)
    owner = cnt.argmax(0)
    return np.where(l >= 0, owner[cls], -1), cnt, owner


def q(v):
    return np.percentile(v, [2.5, 97.5])


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites, R = meta["steps"], meta["sites"], meta["seba_r"]
    assert sites == xlib.SITES
    T = len(steps)
    tp = {tuple(p): i for i, p in enumerate(meta["time_pairs"])}
    pall = tp[(0, T - 1)]
    with np.load(src / "probe.npz") as p:
        cls = p["labels"].astype(np.int64)
    N = len(cls)
    TL = {s: np.load(src / f"time_{s}.npz")["seba_label"][RULE, pall].astype(np.int64) for s in sites}   # (ri, N)
    rng = np.random.default_rng(SEED)
    P = xlib.PALETTE
    C_SETS, C_UNION, C_NULL, C_LAB = P["sets"][0], P["sets"][1], P["mute"], P["fg"]
    out = xlib.Out("training-signal")
    rule_name = meta["modes"][RULE]

    # ------------------------------------------------------------------ labelings and null banks (fixed across steps)
    lab = {}
    for r in (R_ALT, R_MAIN):
        l = TL[SITE_MAIN][R.index(r)]
        lu, cnt, owner = class_union(l, cls, r)
        lab[r] = dict(l=l, lu=lu, cnt=cnt, owner=owner,
                      perm=perm_assigned(l, rng, N_PERM), perm_u=perm_assigned(lu, rng, N_PERM),
                      wcls=perm_within_class(l, cls, rng, N_PERM))
    site_lab = {}
    for s in sites:
        l = TL[s][R.index(R_MAIN)]
        lu = class_union(l, cls, R_MAIN)[0]
        site_lab[s] = dict(l=l, lu=lu, perm=perm_assigned(l, rng, N_PERM_SITES),
                           perm_u=perm_assigned(lu, rng, N_PERM_SITES),
                           wcls=perm_within_class(l, cls, rng, N_PERM_SITES))

    # ------------------------------------------------------------------ per checkpoint
    res = {r: {k: [] for k in ("sets", "union", "perm", "perm_u", "wcls")} for r in lab}
    cls_diff, mean_cos = [], []
    site_fin = {}
    align = {r: np.full((r + 1, T), np.nan) for r in lab}
    for t, st in enumerate(steps):
        G = np.load(src / f"gradgram_{st}.npy").astype(np.float32)
        assert G.shape == (N, N) and np.allclose(np.diagonal(G), 1, atol=1e-2), "gradgram is not a cosine Gram"
        cls_diff.append(within_between(G, cls[None], 10)[0])
        mean_cos.append(float((G.sum(dtype=np.float64) - np.trace(G)) / (N * (N - 1))))
        for r, d in lab.items():
            v = within_between(G, np.vstack([d["l"][None], d["lu"][None]]), r)
            res[r]["sets"].append(v[0])
            res[r]["union"].append(v[1])
            for k in ("perm", "perm_u", "wcls"):
                res[r][k].append(within_between(G, d[k], r))
            with np.load(src / f"points_{st}.npz") as pt:
                gc = pt["grad_cos"].astype(np.float64)
            for j in range(r):
                align[r][j, t] = gc[d["l"] == j].mean()
            align[r][r, t] = gc.mean()
        if t == T - 1:
            for s, d in site_lab.items():
                v = within_between(G, np.vstack([d["l"][None], d["lu"][None]]), R_MAIN)
                site_fin[s] = dict(sets=v[0], union=v[1], perm=q(within_between(G, d["perm"], R_MAIN)),
                                   perm_u=q(within_between(G, d["perm_u"], R_MAIN)),
                                   wcls=q(within_between(G, d["wcls"], R_MAIN)))
        # cross-check the batched statistic against the copied helper (observed labeling only)
        if t in (0, T - 1):
            w, b = xlib.grad_within_between(G, lab[R_MAIN]["l"], R_MAIN)
            assert abs((w[0] - b[0]) - res[R_MAIN]["sets"][-1]) < 1e-5, "batched within-between disagrees"
        print(f"  step {st} done", flush=True)
    for r in res:
        for k in ("sets", "union"):
            res[r][k] = np.array(res[r][k])
        for k in ("perm", "perm_u", "wcls"):
            a = np.array(res[r][k])                                       # (T, P)
            res[r][k] = dict(mean=a.mean(1), lo=np.percentile(a, 2.5, 1), hi=np.percentile(a, 97.5, 1))
    cls_diff = np.array(cls_diff)

    # ------------------------------------------------------------------ lead: b9, r = 4, over training
    def over_training(r):
        d = res[r]
        return xlib.lines(steps, [
            dict(name=f"sets of the time block {SITE_MAIN} (r = {r})", y=d["sets"], color=C_SETS, width=2.2),
            dict(name="class-union grouping, same images (reference)", y=d["union"], color=C_UNION, width=2.2),
            dict(name="sets, labels permuted within each class (null)", y=d["wcls"]["mean"], color=C_SETS, dash=True,
                 width=1.4, band=[d["wcls"]["lo"], d["wcls"]["hi"]]),
            dict(name="sets, labels permuted among assigned images (null)", y=d["perm"]["mean"], color=C_NULL,
                 dash=True, width=1.2, band=[d["perm"]["lo"], d["perm"]["hi"]]),
        ], "training step", "gradient cosine: within − between", xscale="symlog",
            yrange=xlib.nice_range([d["sets"], d["union"], d["wcls"]["hi"], d["perm"]["lo"]], step=0.02),
            band_mode="each", refline=dict(y=0, label="no difference"))
    out.chart("lead", over_training(R_MAIN))

    # ------------------------------------------------------------------ kept: label-free GGN / NTK overlap by site
    cg = json.loads((xlib.DATA / "kernels" / "compare_ggn.json").read_text())
    assert cg["sites"] == sites and cg["n"] == N
    fin = steps[-1]
    rows = {(r_["step"], r_["site"], r_["rule"], r_["reference"]): r_ for r_ in cg["rows"]}
    ov = lambda ref, rr, form="cos", st=fin: np.array([rows[(st, s, RULE, ref)][form][str(rr)]["agreement"] for s in sites])
    gap = lambda ref, rr, form="cos", st=fin, k="gap_b": np.array([rows[(st, s, RULE, ref)][form][str(rr)][k] for s in sites])
    xs = list(range(len(sites)))
    rN = {rr: rr / N for rr in GGN_R}
    out.chart("ggn", xlib.lines(xs, [
        dict(name="GGN-cos sample kernel", y=ov("GGN-cos", 4), color=P["sets"][2], width=2.2, dots=3),
        dict(name="NTK-cos sample kernel", y=ov("NTK-cos", 4), color=P["sets"][3], width=2.2, dots=3),
        dict(name="r/N: expected overlap of random spans (scale reference)", y=[rN[4]] * len(sites), color=C_NULL,
             dash="dot", width=1.2),
    ], "site", "overlap of leading spans (mean cos² of principal angles)", yrange=(0, 1), xticklabels=sites))

    # ------------------------------------------------------------------ facts (only those the fragment quotes)
    k = T - 1
    sv = np.array([site_fin[s]["sets"] for s in sites])
    wlo = np.array([site_fin[s]["wcls"][0] for s in sites])
    whi = np.array([site_fin[s]["wcls"][1] for s in sites])
    phi = np.array([site_fin[s]["perm"][1] for s in sites])
    uv = np.array([site_fin[s]["union"] for s in sites])

    # Sentences in the fragment that are checked here rather than typed:
    def need(cond, what):
        if not cond:
            raise SystemExit(f"the fragment's claim no longer holds: {what}")
    for r in lab:
        d = res[r]
        need(all(d["union"] > d["perm_u"]["hi"]), f"r={r}: class union above its own null at every checkpoint "
                                                  "(ratios are quoted)")
        need(all(d["sets"] > d["perm"]["hi"]), f"r={r}: sets above the permutation band at every checkpoint")
    d4, d2 = res[R_MAIN], res[R_ALT]
    need(all(d4["sets"][1:] < d4["wcls"]["lo"][1:]) and d4["wcls"]["lo"][0] <= d4["sets"][0] <= d4["wcls"]["hi"][0],
         "r=4 below the within-class band after the first checkpoint, inside it at the first")
    need(all(d4["sets"] < d4["union"]), "r=4 sets below the class union at every checkpoint")
    above2 = d2["sets"] > d2["wcls"]["hi"]
    below2 = d2["sets"] < d2["wcls"]["lo"]
    need(above2[0] and above2[k] and all(below2[1:k]), "r=2 above the within-class band at the first and final "
                                                       "checkpoints only, below it at the others")
    need(all(sv > phi), "every site's sets above their permutation band")
    need(sites[int(np.argmin([rows[(fin, s, RULE, "GGN-cos")]["cos"]["4"]["agreement"] for s in sites]))] == "stem",
         "GGN overlap lowest at the stem")

    out.fact("rule", rule_name)
    out.fact("n_perm", N_PERM)
    out.fact("first_step", steps[0])
    out.fact("n_steps", T)
    for r in lab:
        d, pre = res[r], f"r{r}"
        out.fact(f"{pre}.n_assigned", int((lab[r]["l"] >= 0).sum()))
        for key in ("sets", "union", "wcls_lo", "wcls_hi", "ratio"):
            val = {"sets": d["sets"][k], "union": d["union"][k], "wcls_lo": d["wcls"]["lo"][k],
                   "wcls_hi": d["wcls"]["hi"][k], "ratio": d["sets"][k] / d["union"][k]}[key]
            out.fact(f"{pre}.final.{key}", val, 2 if key == "ratio" else (4 if r == R_ALT else 3))   # r = 2 differs in the 4th
    out.fact("r2.n_below_wcls", int(below2.sum()))
    out.fact("r4.final.perm_hi", d4["perm"]["hi"][k], 3)
    for key, val in (("sets", d4["sets"][0]), ("union", d4["union"][0]), ("wcls_lo", d4["wcls"]["lo"][0]),
                     ("wcls_hi", d4["wcls"]["hi"][0])):
        out.fact(f"r4.first.{key}", val, 3)
    out.fact("r4.ratio_min", (d4["sets"] / d4["union"]).min(), 2)
    out.fact("r4.ratio_max", (d4["sets"] / d4["union"]).max(), 2)
    out.fact("classes.first", cls_diff[0], 3)
    out.fact("classes.final", cls_diff[k], 3)
    out.fact("mean_cos.first", mean_cos[0], 3)
    out.fact("mean_cos.final", mean_cos[k], 3)
    # every site (final step, r = 4): the groupings change, the gradients do not
    out.fact("sites.n", len(sites))
    out.fact("sites.sets_min", sv.min(), 3)
    out.fact("sites.sets_max", sv.max(), 3)
    out.fact("sites.n_below_wcls", int((sv < wlo).sum()))
    out.fact("sites.n_above_wcls", int((sv > whi).sum()))
    # span overlaps (final step, rule 0, cosine site kernel). GGN / NTK label-free; loss-gradient label-dependent.
    b9 = sites.index(SITE_MAIN)
    out.fact("ggn.r4.b9", ov("GGN-cos", 4)[b9], 2)
    out.fact("ntk.r4.b9", ov("NTK-cos", 4)[b9], 2)
    out.fact("ggn.r10.b9", ov("GGN-cos", 10)[b9], 2)
    out.fact("ntk.r10.b9", ov("NTK-cos", 10)[b9], 2)
    out.fact("ggn.r4.stem", ov("GGN-cos", 4)[0], 2)
    out.fact("ntk.r4.stem", ov("NTK-cos", 4)[0], 2)
    out.fact("lossgrad.r4.b9", ov("loss-gradient-cos", 4)[b9], 2)
    out.fact("ggn.r4.gap", gap("GGN-cos", 4)[0], 4)          # reference eigengap at the cutoff (same for every site)
    out.fact("ntk.r4.gap", gap("NTK-cos", 4)[0], 4)
    out.fact("sitegap.r4.b9", gap("GGN-cos", 4, k="gap_a")[b9], 4)
    out.fact("rn.r4", rN[4], 3)
    out.fact("rn.r10", rN[10], 3)
    out.fact("ggn_dotsite.r4.b9", ov("GGN-cos", 4, "dot")[b9], 2)
    out.fact("ntk_dotsite.r4.b9", ov("NTK-cos", 4, "dot")[b9], 2)

    # ------------------------------------------------------------------ App 4 (Reference charts)
    app4 = {}
    app4["r2_over_training"] = dict(
        over_training(R_ALT),
        caption=f"<b>Plotted:</b> as the lead figure of this experiment, for the r = {R_ALT} sets of the time block "
                f"{SITE_MAIN} over all {T} checkpoints, kernel rule {rule_name}: within-minus-between cosine of "
                "whole-model per-image loss gradients (y) at each checkpoint (x, symlog), on the "
                f"{int((lab[R_ALT]['l'] >= 0).sum()):,} images the sets assign. Solid blue: the sets; solid orange: "
                "their class-union grouping on the same images; dashed blue with band: within-class permutation null "
                f"(mean, 2.5 to 97.5 percentiles of {N_PERM}); dashed grey with band: permutation of the set labels "
                "among the assigned images.")
    app4["every_site"] = dict(xlib.lines(xs, [
        dict(name=f"sets of each site's time block (r = {R_MAIN})", y=sv, color=C_SETS, width=2.2, dots=3),
        dict(name="class-union grouping, same images", y=uv, color=C_UNION, width=2.2, dots=3),
        dict(name="within-class permutation null", y=(wlo + whi) / 2, color=C_SETS, dash=True, width=1.2,
             band=[wlo, whi]),
        dict(name="permutation among assigned images (97.5th percentile)", y=phi, color=C_NULL, dash=True, width=1.2),
    ], "site whose time block defines the grouping", "gradient cosine: within − between",
        yrange=xlib.nice_range([sv, uv, whi], step=0.005), xticklabels=sites, band_mode="each"),
        caption=f"<b>Plotted:</b> groupings from each site's time block, whole-model gradients. For each site (x), "
                f"the r = {R_MAIN} sets of that site's time block over all {T} checkpoints (kernel rule {rule_name}) "
                f"and the within-minus-between cosine (y) of the same per-image loss gradients (all parameters) at "
                f"step {steps[-1]:,}. The gradients are identical in every column; only the grouping changes. Blue: "
                "the sets; orange: their class-union grouping on the same assigned images; dashed blue with band: "
                f"within-class permutation null ({N_PERM_SITES} draws, 2.5 to 97.5 percentiles); dashed grey: 97.5th "
                "percentile of the permutation of set labels among assigned images.")
    app4["setalign"] = dict(xlib.lines(steps, [
        dict(name=f"set {j + 1}", y=align[R_MAIN][j], color=P["sets"][j]) for j in range(R_MAIN)] + [
        dict(name="all probe images", y=align[R_MAIN][R_MAIN], color=C_NULL, dash=True)],
        "training step", "mean cosine with the mean probe gradient", xscale="symlog",
        yrange=xlib.nice_range([align[R_MAIN]], step=0.05)),
        caption=f"<b>Plotted:</b> for each r = {R_MAIN} set of the time block {SITE_MAIN} over all {T} checkpoints "
                f"(kernel rule {rule_name}), the mean over its members of the cosine between the image's loss "
                "gradient (all parameters) and the mean loss gradient of the probe (y), at each checkpoint (x, "
                "symlog); dashed: the mean over all probe images.")
    app4["class_counts"] = dict(xlib.heat(
        lab[R_MAIN]["cnt"].tolist(), [f"set {j + 1}" for j in range(R_MAIN)], meta["classes"], "images",
        "set", "class", zrange=[0, int(np.bincount(cls).max())], fmt="int"),
        caption=f"<b>Plotted:</b> number of probe images of each class (columns) in each r = {R_MAIN} set (rows) of the "
                f"time block {SITE_MAIN} over all {T} checkpoints, kernel rule {rule_name}; unassigned images are "
                "not counted. The class-union grouping gives each class (column) to the row with its largest count.")
    app4["lossgrad_overlap"] = dict(xlib.lines(xs, [
        dict(name="r = 4", y=ov("loss-gradient-cos", 4), color=P["sets"][0], width=2.2, dots=3),
        dict(name="r = 10", y=ov("loss-gradient-cos", 10), color=P["sets"][1], width=2.2, dots=3),
        dict(name="r/N, r = 4 (scale reference)", y=[rN[4]] * len(sites), color=C_NULL, dash="dot"),
        dict(name="r/N, r = 10 (scale reference)", y=[rN[10]] * len(sites), color=C_NULL, dash="dot", width=0.8),
    ], "site", "overlap of leading spans (mean cos² of principal angles)", yrange=(0, 1), xticklabels=sites),
        caption=f"<b>Plotted:</b> label-dependent reference. Overlap (mean squared cosine of principal angles) of "
                f"each site's leading r modes (single snapshot, step {fin:,}, kernel rule {rule_name}, cosine site "
                "kernel) with the leading r modes of the loss-gradient-cosine sample kernel through the same rule "
                "(the loss uses the labels). Dotted: r/N, the expected overlap of random spans (scale reference, not "
                "a control).")
    app4["ggn_r10"] = dict(xlib.lines(xs, [
        dict(name="GGN-cos sample kernel", y=ov("GGN-cos", 10), color=P["sets"][2], width=2.2, dots=3),
        dict(name="NTK-cos sample kernel", y=ov("NTK-cos", 10), color=P["sets"][3], width=2.2, dots=3),
        dict(name="r/N (scale reference)", y=[rN[10]] * len(sites), color=C_NULL, dash="dot", width=1.2),
    ], "site", "overlap of leading spans (mean cos² of principal angles)", yrange=(0, 1), xticklabels=sites),
        caption=f"<b>Plotted:</b> as the GGN/NTK figure of this experiment, at r = 10: overlap of each site's "
                f"leading ten modes (single snapshot, step {fin:,}, kernel rule {rule_name}, cosine site kernel) with "
                "the leading ten modes of the GGN-cos and NTK-cos sample kernels (label-free) through the same rule. "
                "Dotted: r/N (scale reference).")
    app4["ggn_dotsite"] = dict(xlib.lines(xs, [
        dict(name="GGN-cos, cosine site kernel", y=ov("GGN-cos", 4), color=P["sets"][2], width=2.2),
        dict(name="GGN-cos, dot site kernel", y=ov("GGN-cos", 4, "dot"), color=P["sets"][2], dash=True, width=1.6),
        dict(name="NTK-cos, cosine site kernel", y=ov("NTK-cos", 4), color=P["sets"][3], width=2.2),
        dict(name="NTK-cos, dot site kernel", y=ov("NTK-cos", 4, "dot"), color=P["sets"][3], dash=True, width=1.6),
    ], "site", "overlap of leading spans (mean cos² of principal angles)", yrange=(0, 1), xticklabels=sites),
        caption=f"<b>Plotted:</b> overlap of each site's leading four modes with the leading four modes of the GGN-cos "
                f"and NTK-cos sample kernels (label-free), step {fin:,}, kernel rule {rule_name}; solid: the site "
                "kernel in its default cosine form; dashed: the dot (Euclidean) site kernel of Appendix A.")
    ggn_t = lambda ref: [rows[(st, SITE_MAIN, RULE, ref)]["cos"]["4"]["agreement"] for st in steps]
    app4["ggn_over_training"] = dict(xlib.lines(steps, [
        dict(name="GGN-cos sample kernel", y=ggn_t("GGN-cos"), color=P["sets"][2], width=2.2),
        dict(name="NTK-cos sample kernel", y=ggn_t("NTK-cos"), color=P["sets"][3], width=2.2),
        dict(name="loss-gradient-cos (uses labels)", y=ggn_t("loss-gradient-cos"), color=C_LAB, dots=3, noline=True),
        dict(name="r/N (scale reference)", y=[rN[4]] * T, color=C_NULL, dash="dot", width=1.2),
    ], "training step", "overlap of leading spans (mean cos² of principal angles)", xscale="symlog", yrange=(0, 1)),
        caption=f"<b>Plotted:</b> overlap of {SITE_MAIN}'s leading four modes (single snapshot at each checkpoint, "
                f"kernel rule {rule_name}, cosine site kernel) with the leading four modes of the GGN-cos and NTK-cos "
                "sample kernels at the same checkpoint (lines, label-free) and of the loss-gradient-cosine kernel "
                "(dots, label-dependent), against training step (x, symlog). Dotted: r/N (scale reference).")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
