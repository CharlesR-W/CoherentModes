"""through-depth: one operator on (image, site) whose functions may change from site to site.

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/through-depth.py r20s0b
    python3 site/check_fragment.py through-depth

Reads data/<tag>/spacetime/st_<step>_<rule>.npz (spacetime_stage.compute, SPEC_SPACETIME.md; 8 checkpoints x rules
0 and 1), data/<tag>/meta.json and data/<tag>/probe.npz (class labels: validation only, the flow figure's class
make-up). Writes assets/x/through-depth.json (chart objects), assets/x/through-depth.facts.json and
assets/fig/x_through-depth_{tradeoff,sets,sets_rule1,flow,flow_k10}.png. No new computation on the operators: every
value is a re-plot, a mean, a ratio or a count of stored arrays.

Stored layout used here (spacetime_stage.py): leading axes (scale 2: 0 = relative site units 'own', 1 = native walk
units) x (coupling 6: t = t* x FACTORS). Low space = the first K = 16 modes. Set readout: k-means on the rank-8
space, k in (5, 10), 'a' = shape-aligned rows, 's' = shared coordinates; scores per site (info, ceiling, null) and per
link (churn, churn_matched). References ref_<name>_set<ki>_* do not depend on scale or coupling.

Summaries (all defined here once and quoted in the captions):
  information kept  = mean over sites of info_l / ceiling_l (sites without labels excluded)
  null              = the same mean of the exact random-relabelling mean null_l / ceiling_l
  churn             = mean over the 11 links of churn_matched (share of images whose set changes, after the optimal
                      relabelling of each link)
  TV(w, v)          = half the L1 distance between two depth profiles over the 12 sites
  directions (90%)  = number of eigenvalues of R = sum_l G_l G_l^T (trace K) needed to reach 90% of the trace
  between-site share = t B / (A + t B) for the first K modes (S6: A + t B = sum of the first K eigenvalues)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

XID = "through-depth"
RULE, ALT = 0, 1
REL, NAT = 0, 1
FSTAR = 3                      # index of t = t* in FACTORS
K = 16
KSETS = (5, 10)
FLAB = ["t*/64", "t*/16", "t*/4", "t*", "4t*", "16t*"]
REFS = [("site", "independent per site"), ("thirds", "four blocks of three sites"),
        ("avg", "12-site block average"), ("avg_rel", "12-site block average, relative units")]


def sig(v, n=3):
    """n significant figures as a plain string (no thousands separator below 1000)."""
    v = float(v)
    if v == 0:
        return "0"
    d = max(0, n - 1 - int(np.floor(np.log10(abs(v)))))
    return f"{v:.{d}f}"


def n90(r):
    c = np.cumsum(np.sort(np.asarray(r, float))[::-1])
    return int(np.searchsorted(c / c[-1], 0.9 - 1e-12) + 1)


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())


def score(z, pre, ki, idx=()):
    """Summaries of one stored labelling's scores: info share, null share, churn (matched), churn as labelled, NMI."""
    g = lambda key: np.asarray(z[f"{pre}{ki}_{key}"][idx], float)
    info, ceil, null = g("info"), g("ceiling"), g("null")
    return dict(info=float(np.nanmean(info / ceil)), null=float(np.nanmean(null / ceil)),
                churn=float(np.nanmean(g("churn_matched"))), churn_lab=float(np.nanmean(g("churn"))),
                nmi=float(np.nanmean(g("nmi"))), info_site=info / ceil, null_site=null / ceil,
                churn_link=g("churn_matched"))


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites = meta["steps"], meta["sites"]
    assert sites == xlib.SITES, "unexpected site list"
    nl = len(sites)
    fin = steps[-1]
    Z = {(s, r): np.load(src / "spacetime" / f"st_{s}_{r}.npz") for s in steps for r in (RULE, ALT)}
    z = Z[(fin, RULE)]
    factors = np.asarray(z["factors"])
    assert np.allclose(factors, [1 / 64, 1 / 16, 1 / 4, 1, 4, 16]) and list(z["ksets"]) == list(KSETS)
    assert z["occupancy"].shape == (2, 6, nl)
    probe = np.load(src / "probe.npz")
    if not np.array_equal(probe["idx"], z["probe_idx"]):
        raise SystemExit("probe.npz and the spacetime probe indices differ")
    classes = np.asarray(probe["labels"])
    cnames = meta["classes"]
    P = xlib.PALETTE
    out = xlib.Out(XID)
    xs = list(range(nl))
    rule_name = meta["modes"][RULE]
    out.fact("rule", rule_name)
    out.fact("rule_alt", meta["modes"][ALT])
    out.fact("k_modes", K)
    out.fact("n_links", nl - 1)

    # ------------------------------------------------------------------ couplings
    for s, nm in ((REL, "rel"), (NAT, "nat")):
        out.fact(f"{nm}.tstar", sig(z["t"][s, FSTAR]))
        out.fact(f"{nm}.t_list", ", ".join(sig(v) for v in z["t"][s]))
    sc = np.asarray(z["scales_own"])
    out.fact("scale_mid_lo", sig(sc[1:9].min(), 2))
    out.fact("scale_mid_hi", sig(sc[1:9].max(), 2))
    out.fact("scale_pool", sig(sc[nl - 2], 2))
    out.fact("scale_logits", sig(sc[nl - 1], 2))
    out.fact("scale_ratio", f"{sc[1:9].mean() / sc[nl - 1]:.0f}")

    # ------------------------------------------------------------------ lead: occupancy per site
    occ = np.asarray(z["occupancy"]) / K                       # (scale, coupling, site), rows sum to 1
    assert np.allclose(occ.sum(-1), 1, atol=1e-6)
    pocc = np.asarray(z["paper_occupancy"]) / K
    ymax = float(np.ceil(max(occ.max(), pocc.max()) * 20 + 0.5) / 20)
    for s, key, nm in ((REL, "occ_rel", "relative site units"), (NAT, "occ_nat", "native walk units")):
        ser = [dict(name=f"t = {FLAB[f]}", y=occ[s, f], width=2.4 if f == FSTAR else 1.6) for f in range(6)]
        ser.append(dict(name="paper's matrix (no coupling)", y=pocc, color=P["fg"], dash=True, width=1.4))
        out.chart(key, xlib.lines(xs, ser, "site", f"share of the low space at the site ({nm})", yrange=(0, ymax),
                                  colors="ordinal", xticklabels=sites,
                                  refline={"y": 1 / nl, "label": "even share 1/12"}))
    # the ordinal colours are assigned by the renderer to all 7 series; give the 6 couplings explicit ones instead
    ords = P["ordinal"]
    pick = [ords[i] for i in (0, 2, 3, 4, 6, 7)]
    for key in ("occ_rel", "occ_nat"):
        for f in range(6):
            out.charts[key]["series"][f]["color"] = pick[f]
    tail = lambda s, f: float(occ[s, f, nl - 3:].sum())            # b9, pool, logits
    head = lambda s, f: float(occ[s, f, :4].sum())                 # stem..b3
    for s, nm in ((REL, "rel"), (NAT, "nat")):
        for f, fl in ((0, "f0"), (2, "f2"), (FSTAR, "fstar"), (5, "f5")):
            out.fact(f"{nm}.{fl}.occ_tail", tail(s, f), 2)
            out.fact(f"{nm}.{fl}.occ_head", head(s, f), 2)
        out.fact(f"{nm}.fstar.occ_min", occ[s, FSTAR].min(), 3)
        out.fact(f"{nm}.fstar.occ_max", occ[s, FSTAR].max(), 3)
        out.fact(f"{nm}.fstar.occ_argmin", sites[int(occ[s, FSTAR].argmin())])
        out.fact(f"{nm}.fstar.occ_argmax", sites[int(occ[s, FSTAR].argmax())])
        out.fact(f"{nm}.f5.occ_min", occ[s, 5].min(), 3)
        out.fact(f"{nm}.f5.occ_max", occ[s, 5].max(), 3)
    out.fact("nat.f0.n_empty", int((occ[NAT, 0] < 1e-3).sum()))
    out.fact("paper.occ_tail", float(pocc[nl - 3:].sum()), 2)
    out.fact("paper.occ_min", pocc.min(), 3)
    out.fact("paper.occ_max", pocc.max(), 3)
    out.fact("even_share", 1 / nl, 3)

    # ------------------------------------------------------------------ mode 1 against the gap-only control (S5)
    w1 = np.asarray(z["weight"])[:, :, 0, :]                    # (scale, coupling, site)
    sw = np.asarray(z["scalar_weight"])
    cols3 = [P["sets"][0], P["sets"][1], P["sets"][2]]
    for s, key in ((REL, "mode1_rel"), (NAT, "mode1_nat")):
        ser = []
        for c, f in zip(cols3, (2, FSTAR, 4)):
            ser.append(dict(name=f"mode 1, t = {FLAB[f]}", y=w1[s, f], color=c, width=2.2))
            ser.append(dict(name=f"gap-only control, t = {FLAB[f]}", y=sw[s, f], color=c, dash=True, width=1.5))
        out.chart(key, xlib.lines(xs, ser, "site", "depth weight (share of the squared norm at the site)",
                                  yrange=xlib.nice_range([w1[s, 2:5], sw[s, 2:5]], step=0.05), xticklabels=sites,
                                  refline={"y": 1 / nl, "label": "even share 1/12"}))
    tvm = np.array([[tv(w1[s, f], sw[s, f]) for f in range(6)] for s in (REL, NAT)])
    tvu = np.array([[tv(sw[s, f], np.full(nl, 1 / nl)) for f in range(6)] for s in (REL, NAT)])
    tvm1u = np.array([[tv(w1[s, f], np.full(nl, 1 / nl)) for f in range(6)] for s in (REL, NAT)])
    out.chart("mode1_tv", xlib.lines(
        list(range(6)), [dict(name="mode 1 vs gap-only control, relative units", y=tvm[REL], color=P["sets"][0], dots=3),
                        dict(name="mode 1 vs gap-only control, native units", y=tvm[NAT], color=P["sets"][1], dots=3),
                        dict(name="mode 1 vs even share, relative units (scale)", y=tvm1u[REL], color=P["sets"][0], dash="dot"),
                        dict(name="mode 1 vs even share, native units (scale)", y=tvm1u[NAT], color=P["sets"][1], dash="dot")],
        "coupling t", "total-variation distance between depth profiles", xticklabels=FLAB,
        yrange=xlib.nice_range([tvm, tvm1u], step=0.05)))
    out.fact("tv.rel.fstar", tvm[REL, FSTAR], 3)
    out.fact("tv.nat.fstar", tvm[NAT, FSTAR], 3)
    out.fact("tv.rel.max", tvm[REL].max(), 3)
    out.fact("tv.rel.max_at", FLAB[int(tvm[REL].argmax())])
    out.fact("tv.nat.max", tvm[NAT].max(), 3)
    out.fact("tv.rel.min", tvm[REL].min(), 3)
    out.fact("tvu.rel.fstar", tvm1u[REL, FSTAR], 2)
    out.fact("tvu.nat.f0", tvm1u[NAT, 0], 2)
    out.fact("tv.nat.f0", tvm[NAT, 0], 3)
    out.fact("mode1.nat.f0.tail", float(w1[NAT, 0, nl - 3:].sum()), 3)          # mode 1 on b9, pool, logits
    allt = np.array([[[[tv(Z[(st, r)]["weight"][s, f, 0], Z[(st, r)]["scalar_weight"][s, f]) for f in range(6)]
                       for s in (REL, NAT)] for r in (RULE, ALT)] for st in steps])          # (step, rule, scale, f)
    out.fact("tv.all.max", allt.max(), 3)
    out.fact("tv.all.median", float(np.median(allt)), 3)
    out.fact("tv.all.n", int(allt.size))
    out.fact("tv.all.n_below_05", int((allt < 0.05).sum()))
    i = np.unravel_index(allt.argmax(), allt.shape)
    out.fact("tv.all.max_where", f"step {steps[i[0]]:,}, rule {meta['modes'][(RULE, ALT)[i[1]]]}, "
                                 f"{('relative', 'native')[i[2]]} units, t = {FLAB[i[3]]}")
    se = np.asarray(z["scalar_energy"])
    v1 = np.asarray(z["vals"])[:, :, 0]
    assert (se <= v1 + 1e-10).all()
    out.fact("rel.fstar.lam1", v1[REL, FSTAR], 3)
    out.fact("rel.fstar.scalar_energy", se[REL, FSTAR], 3)

    # ------------------------------------------------------------------ how many image directions; neighbour frames
    rec = np.asarray(z["recurrent"])
    nd = np.array([[n90(rec[s, f]) for f in range(6)] for s in (REL, NAT)])
    pnd = n90(z["paper_recurrent"])
    out.chart("directions", xlib.lines(
        list(range(6)), [dict(name="relative units", y=nd[REL], color=P["sets"][0], dots=3.5),
                        dict(name="native units", y=nd[NAT], color=P["sets"][1], dots=3.5)],
        "coupling t", "image directions for 90% of the trace (of 16)", xticklabels=FLAB, yrange=(0, K),
        refline={"y": pnd, "label": f"paper's matrix: {pnd}"}))
    for s, nm in ((REL, "rel"), (NAT, "nat")):
        for f in range(6):
            out.fact(f"{nm}.n90.f{f}", int(nd[s, f]))
    out.fact("paper.n90", pnd)
    fc = np.asarray(z["frame_cos2"])                           # (scale, coupling, link)
    pfc = np.asarray(z["paper_frame_cos2"])
    pairs = xlib.pair_labels(sites)
    ser = [dict(name=f"t = {FLAB[f]}", y=fc[REL, f], color=pick[f], width=2.4 if f == FSTAR else 1.6) for f in range(6)]
    ser.append(dict(name="paper's matrix", y=pfc, color=P["fg"], dash=True, width=1.4))
    out.chart("frames", xlib.lines(list(range(nl - 1)), ser, "adjacent pair of sites",
                                   "mean cos² of principal angles, leading 4 local directions", yrange=(0, 1),
                                   xticklabels=pairs))
    ser = [dict(name=f"t = {FLAB[f]}", y=fc[NAT, f], color=pick[f], width=2.4 if f == FSTAR else 1.6) for f in range(6)]
    out.chart("frames_nat", xlib.lines(list(range(nl - 1)), ser, "adjacent pair of sites",
                                       "mean cos² of principal angles, leading 4 local directions", yrange=(0, 1),
                                       xticklabels=pairs))
    for s, nm in ((REL, "rel"), (NAT, "nat")):
        out.fact(f"{nm}.frame.fstar.mean", fc[s, FSTAR].mean(), 2)
        out.fact(f"{nm}.frame.fstar.min", fc[s, FSTAR].min(), 2)
        out.fact(f"{nm}.frame.fstar.min_at", pairs[int(fc[s, FSTAR].argmin())])
        out.fact(f"{nm}.frame.mean_lo", fc[s].mean(-1).min(), 2)
        out.fact(f"{nm}.frame.mean_hi", fc[s].mean(-1).max(), 2)
    # frame_cos2 compares the leading 4 local directions; it is ill-conditioned where a site's 4th and 5th local
    # strengths tie. tie(link) = the smaller over the link's two sites of (s4 - s5) / s4.
    stg_all = np.asarray(z["strengths"])                       # (scale, coupling, site, K)
    rel_gap = (stg_all[..., 3] - stg_all[..., 4]) / stg_all[..., 3]
    tie = np.minimum(rel_gap[..., :-1], rel_gap[..., 1:])      # (scale, coupling, link)
    weak = fc < 0.5
    out.fact("frame.n_weak", int(weak.sum()))
    out.fact("frame.n_links_all", int(weak.size))
    assert weak.any() and (tie > 0.05).any()
    out.fact("frame.weak_tie_max", float(tie[weak].max()), 1, pct=True)
    out.fact("frame.untied_min", float(fc[tie > 0.05].min()), 2)
    out.fact("frame.n_untied", int((tie > 0.05).sum()))
    out.fact("frame.tie_cut", 0.05, 0, pct=True)
    lk0 = int(fc[REL, FSTAR].argmin())
    out.fact("rel.frame.fstar.min_s45", f"{sig(stg_all[REL, FSTAR, lk0, 3], 3)} and {sig(stg_all[REL, FSTAR, lk0, 4], 3)} at "
                                       f"{sites[lk0]}, {sig(stg_all[REL, FSTAR, lk0 + 1, 3], 3)} and "
                                       f"{sig(stg_all[REL, FSTAR, lk0 + 1, 4], 3)} at {sites[lk0 + 1]}")
    out.fact("paper.frame.mean", pfc.mean(), 2)
    out.fact("paper.frame.min", pfc.min(), 2)
    out.fact("paper.frame.min_at", pairs[int(pfc.argmin())])
    # the local strength behind a weak frame link: share of the space at the two sites and their 4th strength
    lk = int(fc[REL, FSTAR].argmin())
    stg = np.asarray(z["strengths"])[REL, FSTAR]                 # (site, K) eigenvalues of G_l^T G_l
    out.fact("rel.frame.fstar.min_s4", f"{sig(stg[lk, 3], 2)} and {sig(stg[lk + 1, 3], 2)}")
    out.fact("rel.frame.fstar.min_s1", f"{sig(stg[lk, 0], 2)} and {sig(stg[lk + 1, 0], 2)}")

    # ------------------------------------------------------------------ the exact trade-off (S6)
    tr = np.asarray(z["tradeoff"])                             # (scale, coupling, K in 4 8 16, (A, B))
    kt = list(z["ktrade"])
    tt = np.asarray(z["t"])
    for s in (REL, NAT):
        assert (np.diff(tr[s, :, :, 0], axis=0) >= -1e-7).all() and (np.diff(tr[s, :, :, 1], axis=0) <= 1e-7).all()
    share = tt[:, :, None] * tr[..., 1] / (tr[..., 0] + tt[:, :, None] * tr[..., 1])      # (scale, coupling, K)
    for s, nm in ((REL, "rel"), (NAT, "nat")):
        for f, fl in ((0, "f0"), (FSTAR, "fstar"), (5, "f5")):
            out.fact(f"{nm}.{fl}.a16", tr[s, f, 2, 0], 3 if s == NAT else 2)
            out.fact(f"{nm}.{fl}.b16", tr[s, f, 2, 1], 3)
        out.fact(f"{nm}.fstar.share16", share[s, FSTAR, 2], 2)
        out.fact(f"{nm}.fstar.share4", share[s, FSTAR, 0], 2)
        out.fact(f"{nm}.b_ratio", tr[s, 0, 2, 1] / tr[s, 5, 2, 1], 0)
        out.fact(f"{nm}.a_ratio", tr[s, 5, 2, 0] / tr[s, 0, 2, 0], 1)
        out.fact(f"{nm}.fstar.b_drop", 1 - tr[s, FSTAR, 2, 1] / tr[s, 0, 2, 1], 2, pct=False)
        out.fact(f"{nm}.fstar.a_rise", tr[s, FSTAR, 2, 0] / tr[s, 0, 2, 0], 2)
    out.chart("tradeoff_share", xlib.lines(
        list(range(6)), [dict(name=f"K = {k}, relative units", y=share[REL, :, j], color=P["sets"][j], dots=3)
                        for j, k in enumerate(kt)] +
                       [dict(name=f"K = {k}, native units", y=share[NAT, :, j], color=P["sets"][j], dash=True)
                        for j, k in enumerate(kt)],
        "coupling t", "between-site share tB / (A + tB)", xticklabels=FLAB, yrange=(0, 1)))
    trade_png(out, tr, kt, factors)

    # ------------------------------------------------------------------ pilot controls (relative units, t*)
    bv, shv = np.asarray(z["base_vals"]), np.asarray(z["shuffle_vals"])
    bound = float(z["shuffle_scalar_energy"])
    out.chart("ctrl_vals", xlib.lines(
        list(range(1, K + 1)), [dict(name="probe images matched across sites (the operator)", y=bv, color=P["sets"][0], dots=3),
                                dict(name="images relabelled independently at every site (control)", y=shv,
                                     color=P["sets"][0], dash=True, dots=3)],
        "mode", "eigenvalue on mean-zero slices", yrange=(0, float(np.ceil(shv.max() * 2) / 2)),
        refline={"y": bound, "label": "gap-only lower bound on the first eigenvalue"}))
    out.fact("ctrl.base_l1", bv[0], 3)
    out.fact("ctrl.shuffle_l1", shv[0], 3)
    out.fact("ctrl.bound", bound, 3)
    out.fact("ctrl.base_excess", bv[0] - bound, 3)
    out.fact("ctrl.shuffle_excess", shv[0] - bound, 2)
    out.fact("ctrl.excess_ratio", (shv[0] - bound) / (bv[0] - bound), 0)
    out.fact("ctrl.base_l16", bv[-1], 2)
    out.fact("ctrl.shuffle_l16", shv[-1], 2)
    out.fact("ctrl.shuffle_tstar", sig(z["shuffle_tstar"]))
    bt, st_ = np.asarray(z["base_tradeoff"]), np.asarray(z["shuffle_tradeoff"])
    out.fact("ctrl.base_a", bt[0], 1)
    out.fact("ctrl.base_b", bt[1], 3)
    out.fact("ctrl.shuffle_a", st_[0], 1)
    out.fact("ctrl.shuffle_b", st_[1], 3)
    assert np.allclose(bt, tr[REL, FSTAR, 2], rtol=1e-6), "base_tradeoff differs from the stored t* trade-off"
    out.fact("ctrl.base_adjcos", float(np.asarray(z["adj_cos"])[REL, FSTAR].mean()), 2)
    out.fact("ctrl.shuffle_adjcos", float(np.asarray(z["shuffle_adj_cos"]).mean()), 2)
    for r in (4, 16):
        out.fact(f"ctrl.rank{r}.tstar", sig(z[f"rank{r}_tstar"]))
        out.fact(f"ctrl.rank{r}.share", float(z[f"rank{r}_space_share"]), 3)
    out.fact("ctrl.rank8.tstar", sig(z["tstar_own"]))
    base_occ = np.asarray(z["occupancy"])[REL, FSTAR]
    out.fact("ctrl.base.occ_body", base_occ[:nl - 2].sum() / K, 2)
    for nm in ("drop_pool", "drop_logits", "dup_logits"):
        out.fact(f"ctrl.{nm}.tstar", sig(z[f"{nm}_tstar"]))
        out.fact(f"ctrl.{nm}.share", float(z[f"{nm}_space_share"]), 3)
        o = np.asarray(z[f"{nm}_occupancy"])
        body = o[:nl - 2].sum() / K                     # stem..b9 are the first ten sites in every variant
        out.fact(f"ctrl.{nm}.occ_body", body, 2)
        out.fact(f"ctrl.{nm}.smin", float(z[f"{nm}_restricted_smin"]), 2)

    # ------------------------------------------------------------------ sets through depth (post-hoc readout)
    S = {}                                                     # (step, rule, k, method) -> summary; method strings below
    for (st, r), zz in Z.items():
        for ki, k in enumerate(KSETS):
            for tagr in "as":
                for s in (REL, NAT):
                    for f in range(6):
                        S[(st, r, k, f"{tagr}{s}{f}")] = score(zz, f"set_{tagr}", ki, (s, f))
                S[(st, r, k, f"paper_{tagr}")] = score(zz, f"paper_set_{tagr}", ki)
            for nm, _ in REFS:
                S[(st, r, k, nm)] = score(zz, f"ref_{nm}_set", ki)
    sets_png(out, S, fin, RULE, "sets", meta)
    sets_png(out, S, fin, ALT, "sets_rule1", meta)
    for k in KSETS:
        g = lambda m, st=fin, r=RULE: S[(st, r, k, m)]
        for f in range(6):
            for tagr in "as":
                out.fact(f"sets.k{k}.{tagr}.f{f}.info", g(f"{tagr}{REL}{f}")["info"], 2)
                out.fact(f"sets.k{k}.{tagr}.f{f}.churn", g(f"{tagr}{REL}{f}")["churn"], 2)
        # native units: at small couplings the rank-8 space sits on a few late sites; the other sites get no labels
        # (-1) and the means above run over the labelled sites only. Compare with the thirds on the same sites.
        lab_n = np.asarray(z[f"labels_a{KSETS.index(k)}"])[NAT]                 # (coupling, site, N)
        nlive = (lab_n >= 0).all(-1).sum(-1)
        out.fact(f"sets.k{k}.nat.nlive", ", ".join(str(int(v)) for v in nlive))
        out.fact(f"sets.k{k}.nat.nlive_min", int(nlive.min()))
        f1 = 1
        live1 = (lab_n[f1] >= 0).all(-1)
        out.fact(f"sets.k{k}.nat.f1.nlive", int(live1.sum()))
        out.fact(f"sets.k{k}.nat.f1.first_live", sites[int(np.flatnonzero(live1)[0])])
        out.fact(f"sets.k{k}.a.nat.f1.info", g(f"a{NAT}{f1}")["info"], 2)
        out.fact(f"sets.k{k}.a.nat.f1.churn", g(f"a{NAT}{f1}")["churn"], 2)
        th = g("thirds")
        out.fact(f"sets.k{k}.thirds.nat_f1_sites.info", float(np.nanmean(th["info_site"][live1])), 2)
        lk = live1[1:] & live1[:-1]
        out.fact(f"sets.k{k}.thirds.nat_f1_sites.churn", float(np.nanmean(th["churn_link"][lk])), 2)
        for nm, _ in REFS + [("paper_a", ""), ("paper_s", "")]:
            out.fact(f"sets.k{k}.{nm}.info", g(nm)["info"], 2 if nm != "avg" else 3)
            out.fact(f"sets.k{k}.{nm}.churn", g(nm)["churn"], 2)
        out.fact(f"sets.k{k}.a.fstar.churn_lab", g(f"a{REL}{FSTAR}")["churn_lab"], 2)
        out.fact(f"sets.k{k}.s.f0.churn_lab", g(f"s{REL}0")["churn_lab"], 2)
        out.fact(f"sets.k{k}.a.fstar.info3", g(f"a{REL}{FSTAR}")["info"], 3)
        nulls = [g(m)["null"] for m in [f"{t_}{REL}{f}" for t_ in "as" for f in range(6)] + [n for n, _ in REFS] + ["paper_a", "paper_s"]]
        out.fact(f"sets.k{k}.null_lo", min(nulls), 2)
        out.fact(f"sets.k{k}.null_hi", max(nulls), 2)
        out.fact(f"sets.k{k}.churn_ratio", g("thirds")["churn"] / g(f"a{REL}{FSTAR}")["churn"], 1)
        out.fact(f"sets.k{k}.info_cost", g("thirds")["info"] - g(f"a{REL}{FSTAR}")["info"], 2)
        out.fact(f"sets.k{k}.info_over_avg", g(f"a{REL}{FSTAR}")["info"] - g("avg")["info"], 3)
        best = max(range(6), key=lambda f: g(f"a{REL}{f}")["info"])
        out.fact(f"sets.k{k}.a.best_f", FLAB[best])
        out.fact(f"sets.k{k}.a.best_info", g(f"a{REL}{best}")["info"], 2)
        out.fact(f"sets.k{k}.a.best_churn", g(f"a{REL}{best}")["churn"], 2)
        # thirds: churn sits at the three block boundaries
        cl = g("thirds")["churn_link"]
        out.fact(f"sets.k{k}.thirds.boundary_churn", float(np.nanmean(cl[[2, 5, 8]])), 2)
        # occupied sets per site (the ceiling bounds up to k sets; the null uses the realised number)
        ki = KSETS.index(k)
        out.fact(f"sets.k{k}.a.fstar.groups_min", int(np.asarray(z[f"set_a{ki}_groups"])[REL, FSTAR].min()))
        out.fact(f"sets.k{k}.s.f0.groups_min", int(np.asarray(z[f"set_s{ki}_groups"])[REL, 0].min()))
        # where the shared-coordinates readout keeps more than the shape-aligned one (relative units, both rules, final)
        ahead = [f"{FLAB[f]} ({meta['modes'][r]}, $k = {k}$)" for r in (RULE, ALT) for f in range(6)
                 if S[(fin, r, k, f"s{REL}{f}")]["info"] > S[(fin, r, k, f"a{REL}{f}")]["info"]]
        out.fact(f"sets.k{k}.s_ahead", "; ".join(ahead) if ahead else "none")
        out.fact(f"sets.k{k}.s_ahead_n", len(ahead))
    # alignment strength of the shape-aligned readout: sigma_min / sigma_max of each link's cross matrix
    lnk = np.asarray(z["link"])[REL, FSTAR]
    out.fact("sets.link_min", f"{float(np.nanmin(lnk)):.1e}".replace("e-0", "e-"))
    out.fact("sets.link_min_at", pairs[int(np.nanargmin(lnk))])
    out.fact("sets.link_n_below_01", int((lnk < 0.01).sum()))
    sa_lead = max(S[(fin, r, k, f"s{REL}{f}")]["info"] - S[(fin, r, k, f"a{REL}{f}")]["info"]
                  for r in (RULE, ALT) for k in KSETS for f in range(6))
    out.fact("sets.s_lead_max", sa_lead, 3)
    n_ahead = sum(S[(fin, r, k, f"s{REL}{f}")]["info"] > S[(fin, r, k, f"a{REL}{f}")]["info"]
                  for r in (RULE, ALT) for k in KSETS for f in range(6))
    out.fact("sets.s_ahead_total", int(n_ahead))
    out.fact("sets.n_settings_final", 2 * len(KSETS) * 6)
    # per-site view at t* (details)
    for k in KSETS:
        g = lambda m: S[(fin, RULE, k, m)]
        ser = [dict(name="through-depth sets, shape-aligned, t = t*", y=g(f"a{REL}{FSTAR}")["info_site"], color=P["sets"][0], width=2.4),
               dict(name="through-depth sets, shared coordinates, t = t*", y=g(f"s{REL}{FSTAR}")["info_site"], color=P["sets"][1]),
               dict(name="independent per site", y=g("site")["info_site"], color=P["sets"][2]),
               dict(name="four blocks of three sites", y=g("thirds")["info_site"], color=P["sets"][3]),
               dict(name="12-site block average", y=g("avg")["info_site"], color=P["mute"]),
               dict(name="random relabelling, same sizes (null, through-depth sets)", y=g(f"a{REL}{FSTAR}")["null_site"],
                    color=P["sets"][0], dash="dot")]
        out.chart(f"sets_site_info_k{k}", xlib.lines(xs, ser, "site", "information kept / ceiling", yrange=(0, 1), xticklabels=sites))
        ser = [dict(name="through-depth sets, shape-aligned, t = t*", y=g(f"a{REL}{FSTAR}")["churn_link"], color=P["sets"][0], width=2.4),
               dict(name="through-depth sets, shared coordinates, t = t*", y=g(f"s{REL}{FSTAR}")["churn_link"], color=P["sets"][1]),
               dict(name="independent per site", y=g("site")["churn_link"], color=P["sets"][2]),
               dict(name="four blocks of three sites", y=g("thirds")["churn_link"], color=P["sets"][3])]
        out.chart(f"sets_site_churn_k{k}", xlib.lines(list(range(nl - 1)), ser, "adjacent pair of sites",
                                                       "churn (share of images changing set)", yrange=(0, 1), xticklabels=pairs))

    # across checkpoints and rule 1, at t*, shape-aligned readout, relative units
    for r, rn in ((RULE, "r0"), (ALT, "r1")):
        for k in KSETS:
            g = lambda st, m: S[(st, r, k, m)]
            sets_i = np.array([g(st, f"a{REL}{FSTAR}")["info"] for st in steps])
            sets_c = np.array([g(st, f"a{REL}{FSTAR}")["churn"] for st in steps])
            th_i = np.array([g(st, "thirds")["info"] for st in steps])
            th_c = np.array([g(st, "thirds")["churn"] for st in steps])
            av_i = np.array([g(st, "avg")["info"] for st in steps])
            out.fact(f"train.{rn}.k{k}.n_less_churn", int((sets_c < th_c).sum()))
            out.fact(f"train.{rn}.k{k}.churn_ratio_lo", (th_c / sets_c).min(), 1)
            out.fact(f"train.{rn}.k{k}.churn_ratio_hi", (th_c / sets_c).max(), 1)
            out.fact(f"train.{rn}.k{k}.cost_lo", (th_i - sets_i).min(), 3)
            out.fact(f"train.{rn}.k{k}.cost_hi", (th_i - sets_i).max(), 2)
            out.fact(f"train.{rn}.k{k}.cost_hi_at", f"{steps[int((th_i - sets_i).argmax())]:,}")
            out.fact(f"train.{rn}.k{k}.over_avg_lo", (sets_i - av_i).min(), 3)
            out.fact(f"train.{rn}.k{k}.over_avg_hi", (sets_i - av_i).max(), 3)
            out.fact(f"train.{rn}.k{k}.n_below_thirds", int((sets_i < th_i).sum()))
            out.fact(f"train.{rn}.k{k}.n_above_avg", int((sets_i > av_i).sum()))
            out.fact(f"train.{rn}.k{k}.sets_churn_lo", sets_c.min(), 2)
            out.fact(f"train.{rn}.k{k}.sets_churn_hi", sets_c.max(), 2)
            out.fact(f"train.{rn}.k{k}.thirds_churn_lo", th_c.min(), 2)
            out.fact(f"train.{rn}.k{k}.thirds_churn_hi", th_c.max(), 2)
        gi = lambda st, k, m: S[(st, r, k, m)]["info"]
        gc = lambda st, k, m: S[(st, r, k, m)]["churn"]
        sfx = "" if r == RULE else "_r1"
        out.chart(f"train_info{sfx}", xlib.lines(
            steps, [dict(name=f"through-depth sets (t*), k = {k}", y=[gi(st, k, f'a{REL}{FSTAR}') for st in steps], color=P["sets"][0], dash=(k == 10), dots=3) for k in KSETS] +
                   [dict(name=f"four blocks of three sites, k = {k}", y=[gi(st, k, 'thirds') for st in steps], color=P["sets"][3], dash=(k == 10), dots=3) for k in KSETS] +
                   [dict(name=f"12-site block average, k = {k}", y=[gi(st, k, 'avg') for st in steps], color=P["mute"], dash=(k == 10), dots=3) for k in KSETS],
            "training step", "information kept / ceiling (mean over sites)", xscale="symlog", yrange=(0.4, 0.9)))
        out.chart(f"train_churn{sfx}", xlib.lines(
            steps, [dict(name=f"through-depth sets (t*), k = {k}", y=[gc(st, k, f'a{REL}{FSTAR}') for st in steps], color=P["sets"][0], dash=(k == 10), dots=3) for k in KSETS] +
                   [dict(name=f"four blocks of three sites, k = {k}", y=[gc(st, k, 'thirds') for st in steps], color=P["sets"][3], dash=(k == 10), dots=3) for k in KSETS],
            "training step", "churn (mean over the 11 links)", xscale="symlog", yrange=(0, 0.2)))
    for k in KSETS:
        out.fact(f"r1.k{k}.a.fstar.info", S[(fin, ALT, k, f"a{REL}{FSTAR}")]["info"], 2)
        out.fact(f"r1.k{k}.a.fstar.churn", S[(fin, ALT, k, f"a{REL}{FSTAR}")]["churn"], 2)
        for nm in ("thirds", "avg"):
            out.fact(f"r1.k{k}.{nm}.info", S[(fin, ALT, k, nm)]["info"], 2 if nm != "avg" else 3)
        out.fact(f"r1.k{k}.thirds.churn", S[(fin, ALT, k, "thirds")]["churn"], 2)

    # flow of set membership (final, rule 0, relative units, t*, shape-aligned)
    for k, name in ((5, "flow"), (10, "flow_k10")):
        lab = np.asarray(z[f"labels_a{KSETS.index(k)}"][REL, FSTAR]).astype(int)
        flow_png(out, name, lab, classes, cnames, sites, k)
    lab5 = np.asarray(z["labels_a0"][REL, FSTAR]).astype(int)
    assert (lab5 >= 0).all()
    order = np.argsort(-np.array([np.bincount(lab5[l], minlength=5) for l in range(nl)]).sum(0))   # as flow_png
    sz = lambda l: ", ".join(str(int(np.sum(lab5[l] == j))) for j in order)
    out.fact("flow.sizes_stem", sz(0))
    out.fact("flow.sizes_logits", sz(nl - 1))
    out.fact("flow.same_end_to_end", float((lab5[0] == lab5[-1]).mean()), 2)
    out.fact("flow.never_change", float((lab5 == lab5[:1]).all(0).mean()), 2)
    nmi = np.asarray(z["set_a0_nmi"])[REL, FSTAR]
    out.fact("flow.nmi_stem", nmi[0], 2)
    out.fact("flow.nmi_b9", nmi[nl - 3], 2)
    out.fact("flow.nmi_logits", nmi[nl - 1], 2)
    out.fact("flow.nmi_site_logits", float(np.asarray(z["ref_site_set0_nmi"])[nl - 1]), 2)

    # ------------------------------------------------------------------ across training (rule 0 and rule 1)
    tst = {(r, s): np.array([float(Z[(st, r)]["t"][s, FSTAR]) for st in steps]) for r in (RULE, ALT) for s in (REL, NAT)}
    out.chart("train_tstar", xlib.lines(
        steps, [dict(name=f"relative units, rule {meta['modes'][r]}", y=tst[(r, REL)], color=P["sets"][0], dash=(r == ALT), dots=3)
                for r in (RULE, ALT)],
        "training step", "balance coupling t* (relative units)", xscale="symlog",
        yrange=(0, float(np.ceil(max(tst[(RULE, REL)].max(), tst[(ALT, REL)].max()) + 0.5)))))
    out.chart("train_tstar_nat", xlib.lines(
        steps, [dict(name=f"native units, rule {meta['modes'][r]}", y=tst[(r, NAT)], color=P["sets"][1], dash=(r == ALT), dots=3)
                for r in (RULE, ALT)],
        "training step", "balance coupling t* (native units)", xscale="symlog",
        yrange=(0, float(np.ceil(max(tst[(RULE, NAT)].max(), tst[(ALT, NAT)].max()) * 4 + 0.5) / 4))))
    for r, rn in ((RULE, "r0"), (ALT, "r1")):
        for s, nm in ((REL, "rel"), (NAT, "nat")):
            out.fact(f"train.{rn}.{nm}.tstar0", sig(tst[(r, s)][0]))
            out.fact(f"train.{rn}.{nm}.tstar_fin", sig(tst[(r, s)][-1]))
    occt = {s: np.array([np.asarray(Z[(st, RULE)]["occupancy"])[s, FSTAR] / K for st in steps]) for s in (REL, NAT)}
    for s, key in ((REL, "train_occ"), (NAT, "train_occ_nat")):
        out.chart(key, xlib.lines(
            xs, [dict(name=f"step {xlib.step_label(st)}", y=occt[s][i]) for i, st in enumerate(steps)],
            "site", "share of the low space at the site, t = t*", yrange=(0, float(np.ceil(occt[s].max() * 20 + 0.5) / 20)),
            colors="ordinal", xticklabels=sites, refline={"y": 1 / nl, "label": "even share 1/12"}))
    out.fact("train.rel.occ_tail0", occt[REL][0, nl - 3:].sum(), 2)
    out.fact("train.rel.occ_head0", occt[REL][0, :4].sum(), 2)
    out.fact("train.rel.occ_tail_fin", occt[REL][-1, nl - 3:].sum(), 2)
    out.fact("train.rel.occ_head_fin", occt[REL][-1, :4].sum(), 2)
    out.fact("train.rel.occ_max0", occt[REL][0].max(), 3)
    out.fact("train.rel.occ_max0_at", sites[int(occt[REL][0].argmax())])
    shr = {(r, s): np.array([(lambda zz: float(zz["t"][s, FSTAR] * zz["tradeoff"][s, FSTAR, 2, 1] /
                                             (zz["tradeoff"][s, FSTAR, 2, 0] + zz["t"][s, FSTAR] * zz["tradeoff"][s, FSTAR, 2, 1])))(Z[(st, r)])
                             for st in steps]) for r in (RULE, ALT) for s in (REL, NAT)}
    out.chart("train_share", xlib.lines(
        steps, [dict(name=f"{('relative', 'native')[s]} units, rule {meta['modes'][r]}", y=shr[(r, s)],
                     color=P["sets"][s], dash=(r == ALT), dots=3) for s in (REL, NAT) for r in (RULE, ALT)],
        "training step", "between-site share tB / (A + tB), K = 16, t = t*", xscale="symlog", yrange=(0, 0.5)))
    for r, rn in ((RULE, "r0"), (ALT, "r1")):
        for s, nm in ((REL, "rel"), (NAT, "nat")):
            out.fact(f"train.{rn}.{nm}.share_lo", shr[(r, s)].min(), 2)
            out.fact(f"train.{rn}.{nm}.share_hi", shr[(r, s)].max(), 2)
    nd_t = {(r, s): [n90(Z[(st, r)]["recurrent"][s, FSTAR]) for st in steps] for r in (RULE, ALT) for s in (REL, NAT)}
    for r, rn in ((RULE, "r0"), (ALT, "r1")):
        out.fact(f"train.{rn}.rel.n90", ", ".join(str(v) for v in nd_t[(r, REL)]))
    out.fact("train.r1.rel.occ_tail_fin", float(np.asarray(Z[(fin, ALT)]["occupancy"])[REL, FSTAR, nl - 3:].sum() / K), 2)
    out.fact("train.r1.rel.occ_head_fin", float(np.asarray(Z[(fin, ALT)]["occupancy"])[REL, FSTAR, :4].sum() / K), 2)
    out.fact("train.r1.tv.rel.fstar", tv(Z[(fin, ALT)]["weight"][REL, FSTAR, 0], Z[(fin, ALT)]["scalar_weight"][REL, FSTAR]), 3)
    # numerical gates of the stored units (spacetime_stage.accept), re-read here so the page can say they hold
    out.fact("max_resid", f"{max(float(np.asarray(zz['resid']).max()) for zz in Z.values()):.0e}".replace("e-0", "e-"))
    out.fact("n_units", len(Z))
    out.write()


# ---------------------------------------------------------------------------------------------- static figures
def trade_png(out, tr, kt, factors):
    plt, P = xlib.mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, s, ttl in zip(axes, (REL, NAT), ("relative site units", "native walk units")):
        for j, k in enumerate(kt):
            a, b = tr[s, :, j, 0], tr[s, :, j, 1]
            ax.plot(b, a, "-", color=P["sets"][j], lw=1.6, label=f"first K = {k} modes", zorder=2)
            ax.scatter(b, a, s=[46 if f == FSTAR else 18 for f in range(6)], color=P["sets"][j],
                       edgecolor=P["surface"], linewidth=1.0, zorder=3)
            if k == kt[-1]:
                for f in range(6):
                    ax.annotate(FLAB[f], (b[f], a[f]), xytext=(5, 4), textcoords="offset points", fontsize=7,
                                color=P["fg"], fontweight="bold" if f == FSTAR else "normal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("B: total between-site change, Σ |f_{l+1} − f_l|²")
        ax.set_ylabel("A: total within-site roughness" + (" (÷ c_l)" if s == REL else ""))
        ax.set_title(f"final checkpoint · rule 0 · {ttl}", loc="left")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out.fig_path("tradeoff"))
    plt.close(fig)


def sets_png(out, S, step, rule, name, meta):
    plt, P = xlib.mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    marks = {"site": ("s", P["sets"][2]), "thirds": ("D", P["sets"][3]), "avg": ("^", P["mute"]), "avg_rel": ("v", P["mute"])}
    for ax, k in zip(axes, KSETS):
        g = lambda m: S[(step, rule, k, m)]
        nulls = [g(m)["null"] for m in [f"{t_}{REL}{f}" for t_ in "as" for f in range(6)] + [n for n, _ in REFS] + ["paper_a", "paper_s"]]
        ax.axvspan(min(nulls), max(nulls), color=P["fg"], alpha=0.08, lw=0)
        ax.text(max(nulls) + 0.01, 0.005, "random relabelling\n(same sizes)", fontsize=6.5, color=P["mute"], va="bottom")
        for tagr, col, lab in (("a", P["sets"][0], "through-depth sets, shape-aligned"),
                               ("s", P["sets"][1], "through-depth sets, shared coordinates")):
            xi = [g(f"{tagr}{REL}{f}")["info"] for f in range(6)]
            yc = [g(f"{tagr}{REL}{f}")["churn"] for f in range(6)]
            ax.plot(xi, yc, "-", color=col, lw=1.5, label=lab + " (t*/64 … 16t*)", zorder=2)
            ax.scatter(xi, yc, s=[52 if f == FSTAR else 20 for f in range(6)], color=col, edgecolor=P["surface"], lw=1, zorder=3)
            for f in ((0, FSTAR, 5) if tagr == "a" else (0, FSTAR)):
                ax.annotate(FLAB[f], (xi[f], yc[f]), xytext=(5, 3) if tagr == "a" else (-5, -10), textcoords="offset points",
                            fontsize=6.5, color=P["fg"], ha="left" if tagr == "a" else "right")
        for nm, lab in REFS:
            mk, col = marks[nm]
            ax.scatter([g(nm)["info"]], [g(nm)["churn"]], marker=mk, s=44, color=col if nm != "avg_rel" else "none",
                       edgecolor=col, lw=1.4, label=lab, zorder=4)
        for nm, mk, lab in (("paper_a", "*", "paper's matrix, shape-aligned"), ("paper_s", "X", "paper's matrix, shared coordinates")):
            ax.scatter([g(nm)["info"]], [g(nm)["churn"]], marker=mk, s=70 if mk == "*" else 46, color=P["fg"],
                       edgecolor=P["surface"], lw=0.8, label=lab, zorder=4)
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0, 0.4)
        ax.set_xlabel("information kept / ceiling (mean over sites)")
        ax.set_title(f"k = {k} · step {step:,} · rule {meta['modes'][rule]} · relative units", loc="left")
    axes[0].set_ylabel("churn: share of images changing set (mean over links)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out.fig_path(name))
    plt.close(fig)


def flow_png(out, name, lab, classes, cnames, sites, k):
    """Alluvial of set membership across the sites (top) and each set's class make-up (bottom, validation)."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath
    plt, P = xlib.mpl_style()
    nl, n = lab.shape
    live = (lab >= 0).all(1)
    sizes = np.array([np.bincount(lab[l][lab[l] >= 0], minlength=k) for l in range(nl)])      # (site, set)
    order = np.argsort(-sizes.sum(0))                          # set numbering: by total size over the sites
    rank = np.empty(k, int); rank[order] = np.arange(k)
    L = np.where(lab >= 0, rank[np.clip(lab, 0, None)], -1)
    sizes = sizes[:, order]
    gap = 0.012 * n
    col = lambda j: P["sets"][j % len(P["sets"])]
    W = 0.16                                                   # node width in site units
    tops = np.zeros((nl, k))                                   # y of each node's top (stacked downwards from 0)
    for l in range(nl):
        y = 0.0
        for j in range(k):
            tops[l, j] = y
            y += sizes[l, j] + gap
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.4, 7.6), gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.28))
    for l in range(nl - 1):
        if not (live[l] and live[l + 1]):
            continue
        c = np.zeros((k, k), int)
        np.add.at(c, (L[l], L[l + 1]), 1)
        out_off = tops[l].copy()
        in_off = tops[l + 1].copy()
        for i in range(k):
            for j in range(k):
                if c[i, j] == 0:
                    continue
                y0a, y0b = out_off[i], out_off[i] + c[i, j]
                y1a, y1b = in_off[j], in_off[j] + c[i, j]
                out_off[i] += c[i, j]
                in_off[j] += c[i, j]
                x0, x1 = l + W / 2, l + 1 - W / 2
                xm = (x0 + x1) / 2
                verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                         MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), facecolor=col(i), alpha=0.42 if i != j else 0.30, lw=0))
    for l in range(nl):
        for j in range(k):
            if sizes[l, j]:
                ax.add_patch(Rectangle((l - W / 2, tops[l, j]), W, sizes[l, j], facecolor=col(j), edgecolor=P["surface"], lw=0.8))
                if l == 0 or l == nl - 1:
                    ax.text(l + (-W / 2 - 0.06 if l == 0 else W / 2 + 0.06), tops[l, j] + sizes[l, j] / 2,
                            f"set {j + 1} ({sizes[l, j]})", ha="right" if l == 0 else "left", va="center", fontsize=7)
    ymax = tops[:, -1].max() + sizes[:, -1].max() + gap
    for a in (ax, bx):
        a.set_xlim(-1.15, nl + 0.15)
        a.set_ylim(ymax, -gap)
        a.set_xticks(range(nl)); a.set_xticklabels(sites)
        a.set_yticks([]); a.grid(False)
        a.spines["left"].set_visible(False)
    ax.set_title(f"set membership across sites · k = {k} · shape-aligned readout · t = t* · relative units · final "
                 "checkpoint · rule 0 (flows coloured by the set at the left site)", loc="left")
    # class make-up of every node (validation)
    for l in range(nl):
        for j in range(k):
            m = L[l] == j
            if not m.any():
                continue
            cc = np.bincount(classes[m], minlength=10)
            y = tops[l, j]
            for ci in range(10):
                if cc[ci]:
                    bx.add_patch(Rectangle((l - 0.3, y), 0.6, cc[ci], facecolor=P["classes"][ci], lw=0))
                    y += cc[ci]
            bx.add_patch(Rectangle((l - 0.3, tops[l, j]), 0.6, sizes[l, j], facecolor="none", edgecolor=P["fg"], lw=0.6))
            if l == 0:
                bx.text(-0.3 - 0.08, tops[l, j] + sizes[l, j] / 2, f"set {j + 1}", ha="right", va="center", fontsize=7)
    bx.set_title("the same nodes, each split by CIFAR-10 class (validation only; classes enter no operator)", loc="left")
    hs = [Rectangle((0, 0), 1, 1, facecolor=P["classes"][ci]) for ci in range(10)]
    bx.legend(hs, cnames, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=10, fontsize=7, handlelength=1.0)
    fig.savefig(out.fig_path(name), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
