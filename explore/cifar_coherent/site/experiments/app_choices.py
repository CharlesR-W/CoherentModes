"""Appendix A (`#choices`, content/choices.html): construction choices.

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/app_choices.py r20s0b

Writes (only) site/assets/x/app-choices.json (chart objects, data-src="assets/x/app-choices.json#<key>"),
site/assets/x/app-choices.facts.json ({{fact:x.app-choices.<name>}}) and site/assets/fig/app_choices_*.png.

Reads, all existing exports (no new spectra; labels only in validation numbers):
    data/<tag>/sweep.json            the 19-rule kernel sweep (sweep_math.py)
    data/<tag>/avgprod.json          average vs product spans (r = 3, 6)
    data/<tag>/product_summary.json  average vs product sets and spans by block (product_analysis.py)
    data/<tag>/layer*_<step>.npz     sets, spectra, step gains (schema v2) of the cosine construction
    data/r20s0d/layer_<step>.npz     the same for the dot (Euclidean) construction (SPEC_DOT.md)
    data/kernels/compare_single.json, compare_blocks.json, compare_ggn.json   cosine vs dot comparisons
Scale convention: archive eigenvalues are lazy, lambda_M of M = (I + P)/2; every quantity here stays on that scale.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

XID = "app-choices"
RULE = 0
N_PERM = 1000
EARLY = ["stem"] + [f"b{i}" for i in range(1, 9)]          # stem..b8
LATE = ["b9", "pool", "logits"]


def figpath(name):
    p = xlib.SITE / "assets" / "fig" / f"app_choices_{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def rng_txt(lo, hi, d=2):
    a, b = (xlib.fmt(v, d).replace("-", "−") for v in (lo, hi))
    if a == b:
        return a
    return f"{a} to {b}" if lo < 0 or hi < 0 else f"{a}–{b}"


def listing(items):
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def main(tag):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites, N = meta["steps"], meta["sites"], meta["n"]
    assert sites == xlib.SITES
    fin = steps[-1]
    T1 = len(steps) - 1
    lp = {tuple(p): i for i, p in enumerate(meta["layer_pairs"])}
    tp = {tuple(p): i for i, p in enumerate(meta["time_pairs"])}
    rix = {r: i for i, r in enumerate(meta["seba_r"])}
    sidx = {s: i for i, s in enumerate(sites)}
    P = xlib.PALETTE
    C0, C1, C2, C3 = P["sets"][0], P["sets"][1], P["sets"][2], P["sets"][3]
    out = xlib.Out(XID)
    out.fact("final_step", fin)

    # ================================================================== 1. the 19-rule sweep (caption numbers)
    sw = json.loads((src / "sweep.json").read_text())["rows"]
    rules = list(dict.fromkeys(r["rule"] for r in sw))
    out.fact("sweep.n_rules", len(rules))
    out.fact("sweep.steps", listing(f"{s:,}" for s in sorted({r["step"] for r in sw})))
    sw_sites = [s for s in sites if s in {r["site"] for r in sw}]
    out.fact("sweep.sites", listing(sw_sites))
    fam_par = {"global": "kappa", "mass": "target", "perplexity": "perplexity", "union_knn": "k"}
    for fam, key in fam_par.items():
        vals = list(dict.fromkeys(r[key] for r in sw if r["family"] == fam))
        out.fact(f"sweep.{fam}", ", ".join(f"{v:g}" for v in vals))
    mid = ("b2", "b5", "b8")
    g = [r["gaps"][0] for r in sw if r["step"] == fin and r["site"] in mid and r["rule"] in ("global 2", "global 3", "global 5")]
    out.fact("sweep.gap1.global_small", rng_txt(min(g), max(g)))
    per_image = ("union kNN 15", "perplexity 100", "perplexity 300")
    g = [r["gaps"][0] for r in sw if r["step"] == fin and r["site"] in mid and r["rule"] in per_image]
    out.fact("sweep.gap1.per_image", rng_txt(min(g), max(g)))

    # ================================================================== 2. cosine vs dot (Euclidean)
    kd = xlib.DATA / "kernels"
    cs = json.loads((kd / "compare_single.json").read_text())
    cb = json.loads((kd / "compare_blocks.json").read_text())
    cg = json.loads((kd / "compare_ggn.json").read_text())
    assert cs["sites"] == sites and cs["steps"] == steps
    row = {(r["step"], r["site"], r["rule"]): r for r in cs["rows"]}
    xs = list(range(len(sites)))
    span = lambda st, s, ru, r: row[(st, s, ru)]["span"][str(r)]["agreement"]
    out.chart("dot_span", xlib.lines(
        xs, [dict(name=f"rule 0 ({meta['modes'][0]}), r = 4", y=[span(fin, s, 0, 4) for s in sites], color=C0, width=2.2),
             dict(name="rule 0, r = 10", y=[span(fin, s, 0, 10) for s in sites], color=C0, dash=True),
             dict(name=f"rule 1 ({meta['modes'][1]}), r = 4", y=[span(fin, s, 1, 4) for s in sites], color=C1, width=2.2),
             dict(name="rule 1, r = 10", y=[span(fin, s, 1, 10) for s in sites], color=C1, dash=True)],
        "site (single snapshot)", "cosine–dot span agreement (mean cos²)", yrange=(0, 1), xticklabels=sites))
    hub = lambda s, k: row[(fin, s, 0)]["hubness"][k]["max_indegree"]
    out.chart("dot_indeg", xlib.lines(
        xs, [dict(name="cosine, 15-NN", y=[hub(s, "cos") for s in sites], color=C0, width=2.2),
             dict(name="dot (Euclidean), 15-NN", y=[hub(s, "dot") for s in sites], color=C1, width=2.2)],
        "site (single snapshot)", f"largest 15-NN in-degree (of {N:,} images)", yrange=(0, N), xticklabels=sites))
    cls = lambda s, ru, k: row[(fin, s, ru)]["validation"][k]["span"]["10"]["agreement"]
    out.chart("dot_class", xlib.lines(
        xs, [dict(name="cosine, rule 0", y=[cls(s, 0, "cos") for s in sites], color=C0, width=2.2),
             dict(name="dot, rule 0", y=[cls(s, 0, "dot") for s in sites], color=C0, dash=True),
             dict(name="cosine, rule 1", y=[cls(s, 1, "cos") for s in sites], color=C1, width=2.2),
             dict(name="dot, rule 1", y=[cls(s, 1, "dot") for s in sites], color=C1, dash=True)],
        "site (single snapshot)", "agreement with the class-indicator span (r = 10)", yrange=(0, 1), xticklabels=sites))
    # over training (heatmaps site x step), rules 0 and 1, r = 4
    for ru in (0, 1):
        out.chart(f"dot_span_steps_rule{ru}", xlib.heat(
            [[span(st, s, ru, 4) for st in steps] for s in sites], sites, [xlib.step_label(st) for st in steps],
            "cosine–dot span agreement, r = 4", "site", "training step", fmt="2f"))
    # facts: single snapshots, final step
    e0 = [span(fin, s, 0, 4) for s in EARLY]
    out.fact("dot.early_sites", f"{EARLY[0]}–{EARLY[-1]}")
    out.fact("dot.late_sites", listing(LATE))
    out.fact("dot.r4.rule0.early", rng_txt(min(e0), max(e0)))
    e1 = [span(fin, s, 1, 4) for s in EARLY]
    out.fact("dot.r4.rule1.early", rng_txt(min(e1), max(e1)))
    lt = [span(fin, s, ru, 4) for s in LATE for ru in (0, 1)]
    out.fact("dot.r4.late", rng_txt(min(lt), max(lt)))
    hd = [hub(s, "dot") for s in EARLY]
    hc = [hub(s, "cos") for s in EARLY]
    out.fact("dot.indeg.dot.early", f"{min(hd):,}–{max(hd):,}")
    out.fact("dot.indeg.cos.early", f"{min(hc):,}–{max(hc):,}")
    hl = [hub(s, k) for s in LATE for k in ("cos", "dot")]
    out.fact("dot.indeg.late", f"{min(hl):,}–{max(hl):,}")
    no = [row[(fin, s, 0)]["neighbour_overlap"] for s in EARLY]
    out.fact("dot.nn_overlap.early", rng_txt(min(no), max(no)))
    out.fact("n_images", N)
    c_med = lambda ru, k: float(np.median([cls(s, ru, k) for s in sites]))
    for ru in (0, 1):
        for k in ("cos", "dot"):
            out.fact(f"dot.class10.rule{ru}.{k}.median", c_med(ru, k), 2)

    # participation number of the leading eigenvectors (unit psi), single snapshots, final step, both constructions
    def pn(tag_):
        with np.load(xlib.DATA / tag_ / f"layer_{fin}.npz") as z:
            vecs, scale = z["vecs"], z["scale"]
        res = {}
        for ru in (0, 1):
            for s in sites:
                p = lp[(sidx[s], sidx[s])]
                psi = vecs[ru, p].astype("f8") * scale[ru, p][:, None] / 32767       # (8, N), unit norm
                psi /= np.linalg.norm(psi, axis=1, keepdims=True)
                res[(ru, s)] = 1.0 / (psi ** 4).sum(1)
        return res
    pn_c, pn_d = pn(tag), pn("r20s0d")
    plt, _ = xlib.mpl_style()
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, ru in zip(axs, (0, 1)):
        for res, col, lab in ((pn_c, C0, "cosine"), (pn_d, C1, "dot (Euclidean)")):
            ax.plot(xs, [res[(ru, s)][0] for s in sites], color=col, lw=2, marker="o", ms=3, label=f"{lab}: PN(ψ₁)")
            ax.plot(xs, [np.median(res[(ru, s)]) for s in sites], color=col, lw=1.3, ls="--",
                    label=f"{lab}: median PN(ψ₁..ψ₈)")
        ax.axhline(2, color=P["mute"], lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xticks(xs, sites, rotation=45)
        ax.set_title(f"rule {ru}: {meta['modes'][ru]}", loc="left")
    axs[0].set_ylabel("participation number (log)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figpath("pn"))
    plt.close(fig)
    d_early = [pn_d[(0, s)][0] for s in EARLY]
    c_all = [pn_c[(0, s)][0] for s in sites]
    out.fact("dot.pn1.rule0.dot.early", rng_txt(min(d_early), max(d_early), 0))
    out.fact("dot.pn1.rule0.cos.all", rng_txt(min(c_all), max(c_all), 0))
    out.fact("dot.pn1.rule0.dot.early_n_le3", sum(v <= 3 for v in d_early))
    out.fact("dot.n_early", len(EARLY))
    v = [np.median(pn_d[(1, s)]) for s in EARLY]
    out.fact("dot.pnmed.rule1.dot.early", rng_txt(min(v), max(v), 0))
    v = [np.median(pn_c[(1, s)]) for s in EARLY]
    out.fact("dot.pnmed.rule1.cos.early", rng_txt(min(v), max(v), 0))

    # blocks: heatmaps first x last site, slider over checkpoints
    brow = {(r["family"], r["name"], r["a"], r["b"], r["rule"]): r for r in cb["rows"]}
    def bheat(ru, what, r):
        Z = []
        for st in steps:
            M = np.full((len(sites), len(sites)), np.nan)
            for a in range(len(sites)):
                for b in range(a, len(sites)):
                    rr = brow[("layer", st, a, b, ru)]
                    M[a, b] = rr["span"][str(r)]["agreement"] if what == "span" else rr["seba"][str(r)]["mean"]
            Z.append(M)
        return Z
    labs = {"span": "cosine–dot span agreement", "sets": "cosine–dot matched Jaccard of SEBA sets"}
    for ru in (0, 1):
        for what in ("span", "sets"):
            out.chart(f"dot_blocks_{what}_rule{ru}", xlib.heat(
                bheat(ru, what, 4), sites, sites, f"{labs[what]}, r = 4", "first site a", "last site b",
                slider=dict(label="training step", values=[f"{s:,}" for s in steps]), fmt="2f"))
    full = lambda ru: brow[("layer", fin, 0, len(sites) - 1, ru)]
    for ru in (0, 1):
        out.fact(f"dot.full.rule{ru}.span4", full(ru)["span"]["4"]["agreement"], 2)
        out.fact(f"dot.full.rule{ru}.sets4", full(ru)["seba"]["4"]["mean"], 2)
        out.fact(f"dot.full.rule{ru}.sets8", full(ru)["seba"]["8"]["mean"], 2)

    # dot sites vs cosine sites against fixed gradient-side references (label-free ones: GGN, NTK)
    refs = [r for r in cg["references"] if not r.lower().startswith("loss")]
    grow = {(r["step"], r["site"], r["rule"], r["reference"]): r for r in cg["rows"]}
    ser = []
    for i, ref in enumerate(refs):
        ser.append(dict(name=f"reference {ref}", y=[grow[(fin, s, RULE, ref)]["delta"]["10"] for s in sites],
                        color=P["sets"][i % len(P["sets"])], width=2, dots=3))
    out.chart("dot_ggn", xlib.lines(xs, ser, "site (single snapshot)",
                                    "agreement(dot site, R) − agreement(cosine site, R), r = 10",
                                    yrange=xlib.nice_range([s["y"] for s in ser], step=0.05), xticklabels=sites,
                                    refline=dict(y=0, label="no difference")))
    de = [grow[(fin, s, RULE, ref)]["delta"]["10"] for s in EARLY for ref in refs]
    dl = [grow[(fin, s, RULE, ref)]["delta"]["10"] for s in LATE for ref in refs]
    out.fact("dot.ggn.refs", listing(refs))
    out.fact("dot.ggn.d10.early", rng_txt(min(de), max(de)))
    out.fact("dot.ggn.d10.late", rng_txt(min(dl), max(dl)))
    out.fact("dot.ggn.early_neg", sum(v < 0 for v in de))
    out.fact("dot.ggn.early_total", len(de))
    bd = [r["delta"][k] for r in cg["block_rows"] if r["step"] == fin and r["rule"] == RULE and r["reference"] in refs
          for k in ("4", "10")]
    out.fact("dot.ggn.block", rng_txt(min(bd), max(bd)))

    # raw dot ranking (maximum inner product) diagnostic
    mp = [m for m in cs["mips"] if m["step"] == fin]
    out.fact("mips.steps", listing(f"{s:,}" for s in sorted({m["step"] for m in cs["mips"]})))
    out.fact("mips.nn_cos", float(np.median([m["neighbour_overlap"]["cos"] for m in mp])), 2)
    out.fact("mips.nn_dot", float(np.median([m["neighbour_overlap"]["dot"] for m in mp])), 2)
    msite = {m["site"]: m for m in mp}
    mid_sites = [s for s in sites if s not in ("stem", "pool", "logits")]
    out.fact("mips.mid_sites", f"{mid_sites[0]}–{mid_sites[-1]}")
    for k in ("cos", "dot"):
        v = [msite[s]["span"][k]["4"]["agreement"] for s in mid_sites]
        out.fact(f"mips.span_{k}.mid", rng_txt(min(v), max(v)))
        v = [msite[s]["span"][k]["4"]["agreement"] for s in ("stem", "pool", "logits")]
        out.fact(f"mips.span_{k}.ends", rng_txt(min(v), max(v)))
    v = [msite[s]["hubness"]["max_indegree"] for s in sites]
    out.fact("mips.indeg", f"{min(v):,}–{max(v):,}")
    out.chart("mips", xlib.lines(
        xs, [dict(name="15-NN overlap with cosine", y=[msite[s]["neighbour_overlap"]["cos"] for s in sites], color=C0, width=2),
             dict(name="15-NN overlap with dot (Euclidean)", y=[msite[s]["neighbour_overlap"]["dot"] for s in sites], color=C1, width=2),
             dict(name="span agreement with cosine, r = 4", y=[msite[s]["span"]["cos"]["4"]["agreement"] for s in sites], color=C0, dash=True),
             dict(name="span agreement with dot, r = 4", y=[msite[s]["span"]["dot"]["4"]["agreement"] for s in sites], color=C1, dash=True)],
        "site (single snapshot)", "agreement of the raw-dot-ranking rule 0 with …", yrange=(0, 1), xticklabels=sites))

    # ================================================================== 3. average vs product
    ap = json.loads((src / "avgprod.json").read_text())["rows"]
    out.fact("avgprod.n_pairs", len({(r["a"], r["b"]) for r in ap if r["step"] == fin and r["mode"] == 0 and r["r"] == 3}))
    for mode in (0, 1):
        for key in ("right", "left", "end_right", "end_left"):
            v = np.array([r[key] for r in ap if r["step"] == fin and r["mode"] == mode and r["r"] == 3])
            out.fact(f"avgprod.rule{mode}.r3.{key}", f"{xlib.fmt(np.median(v), 3)} ({xlib.fmt(v.min(), 3)})")
    v = [np.median([r[key] for r in ap if r["step"] == fin and r["mode"] == mode and r["r"] == 6])
         for mode in (0, 1) for key in ("right", "left", "end_right", "end_left")]
    out.fact("avgprod.r6.median_range", rng_txt(min(v), max(v)))

    ps = json.loads((src / "product_summary.json").read_text())["rows"]
    lay = [r for r in ps if r["fam"] == "layer" and r["anchor"] == T1]
    lens = list(range(2, len(sites) + 1))
    def by_len(mode, key):
        return [float(np.mean([r[key] for r in lay if r["mode"] == mode and r["b"] - r["a"] + 1 == L])) for L in lens]
    ser = [dict(name="sets: average vs product · start", y=by_len(0, "avg_start_r4"), color=C0, width=2.2),
           dict(name="sets: average vs product · end", y=by_len(0, "avg_end_r4"), color=C1, width=2.2),
           dict(name="spans: average vs product · start", y=by_len(0, "span_start_r4"), color=C0, dash="dot"),
           dict(name="spans: average vs product · end", y=by_len(0, "span_end_r4"), color=C1, dash="dot"),
           dict(name=f"sets, rule 1 ({meta['modes'][1]}) · start", y=by_len(1, "avg_start_r4"), color=C0, dash=True, width=1.4),
           dict(name="sets, rule 1 · end", y=by_len(1, "avg_end_r4"), color=C1, dash=True, width=1.4)]
    out.chart("product_by_len", xlib.lines(lens, ser, "sites in the layer block (b − a + 1)",
                                           "agreement, r = 4 (mean over blocks of that length)", yrange=(0, 1)))
    s_ = by_len(0, "avg_start_r4")
    e_ = by_len(0, "avg_end_r4")
    for L in (2, 4, 6, 8, 10):
        out.fact(f"prod.len{L}.sets_start", s_[lens.index(L)], 2)
        out.fact(f"prod.len{L}.sets_end", e_[lens.index(L)], 2)
    out.fact("prod.n_blocks", len([r for r in lay if r["mode"] == 0]))
    fb = next(r for r in lay if r["mode"] == 0 and r["a"] == 0 and r["b"] == len(sites) - 1)
    out.fact("prod.full.sets_start", fb["avg_start_r4"], 2)
    out.fact("prod.full.sets_end", fb["avg_end_r4"], 2)
    out.fact("prod.full.span_start", fb["span_start_r4"], 2)
    out.fact("prod.full.span_end", fb["span_end_r4"], 2)
    out.fact("prod.full.sigma", ", ".join(xlib.fmt(v, 2) for v in fb["sigma"]))
    out.fact("prod.full.lam", ", ".join(xlib.fmt(v, 2) for v in fb["lam"]))
    for r in (4, 8):
        out.fact(f"prod.full.cls_avg_r{r}", fb[f"cls_avg_r{r}"], 2)
        out.fact(f"prod.full.cls_end_r{r}", fb[f"cls_end_r{r}"], 2)
        out.fact(f"prod.full.cls_start_r{r}", fb[f"cls_start_r{r}"], 2)
    # step gains multiply to sigma: check on every exported product block
    worst, nblk = 0.0, 0
    files = [f"layerprod{s}_{st}.npz" for s in "RL" for st in steps] + [f"timeprod{s}_{si}.npz" for s in "RL" for si in sites]
    for fn in files:
        with np.load(src / fn) as z:
            g, vals = z["gain_psi"].astype("f8"), z["vals"].astype("f8")
        prod = np.where(np.isnan(g), 1.0, g).prod(-1)                  # (mode, pair, k)
        ok = ~np.all(np.isnan(g), -1) & np.isfinite(vals[..., 1:])
        d = np.abs(prod - vals[..., 1:])[ok]
        if d.size:
            worst = max(worst, float(d.max()))
            nblk += int(ok.any(-1).sum())
    mant, ex = f"{worst:.1e}".split("e")
    out.fact("prod.gain_check", f"{mant} \\times 10^{{{int(ex)}}}")          # quoted inside $...$
    out.fact("prod.gain_check_n", nblk)

    # ================================================================== 4. robustness of the b7..pool sets (X1)
    labels = {}
    with np.load(src / f"layer_{fin}.npz") as z:
        LL = z["seba_label"]
    with np.load(src / "time_b9.npz") as z:
        TL = z["seba_label"]
    a7, p10 = sidx["b7"], sidx["pool"]
    cols = [(f"rule {m}", lambda ri, m=m: LL[m, lp[(a7, p10)], ri]) for m in range(1, len(meta["modes"]))]
    cols += [("b9 alone", lambda ri: LL[RULE, lp[(sidx["b9"], sidx["b9"])], ri]),
             (f"b9, steps {steps[len(steps) // 2]:,}–{fin:,}", lambda ri: TL[RULE, tp[(len(steps) // 2, T1)], ri])]
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(N) for _ in range(N_PERM)])
    ser, robust = [], {}
    for i, r in enumerate((2, 3, 4)):
        ri = rix[r]
        ref = LL[RULE, lp[(a7, p10)], ri].astype(np.int64)
        y, lo, hi = [], [], []
        for name, fn in cols:
            other = fn(ri).astype(np.int64)
            sc, _, null = xlib.matched_jaccard(ref, other, r, r, perms)
            _, l_, h_ = xlib.null_summary(null)
            y.append(sc); lo.append(l_); hi.append(h_)
        robust[r] = (y, hi)
        ser.append(dict(name=f"r = {r}", y=y, color=P["sets"][i], dots=4, noline=True, band=[lo, hi]))
    out.chart("robust", xlib.lines(list(range(len(cols))), ser, "compared with (b7..pool, rule 0, final step)",
                                   "matched Jaccard", yrange=(0, 1), xticklabels=[c[0] for c in cols],
                                   band_mode="each", band_label="permutation null, 2.5–97.5th percentile"))
    y4, hi4 = robust[4]
    names = ["rule1", "rule2", "rule3", "rule4", "b9_alone", "b9_time_late"]
    for nm, v in zip(names, y4):
        out.fact(f"robust.r4.{nm}", v, 2)
    out.fact("robust.r4.null_hi_max", max(hi4), 2)
    out.fact("robust.late_from", steps[len(steps) // 2])
    out.fact("robust.n_perm", N_PERM)

    # ================================================================== 5. choice of r: gaps at each SEBA rank
    with np.load(src / f"layer_{fin}.npz") as z:
        Lv = z["vals"][RULE].astype("f8")                     # (pair, 9): lambda_0 = 1, lambda_1..8 (lazy)
    rows_ = sites + ["b7..pool"]
    pidx = [lp[(i, i)] for i in range(len(sites))] + [lp[(a7, p10)]]
    rs = list(range(2, 9))
    G = [[float(Lv[p, r - 1] - Lv[p, r]) for r in rs] for p in pidx]
    out.chart("gaps", xlib.heat(G, rows_, [str(r) for r in rs], "gap λ(r−1) − λ(r), lazy scale", "operator (final step, rule 0)",
                                "SEBA rank r", zrange=[0, float(np.ceil(np.max(G) * 100) / 100)], fmt="3f"))
    Garr = np.array(G)
    best = Garr.argmax(1)
    out.fact("gaps.n_rows", len(rows_))
    from collections import Counter
    cnt = Counter(rs[b] for b in best)
    out.fact("gaps.argmax_counts", listing(f"r = {r} at {n}" for r, n in sorted(cnt.items())))
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
