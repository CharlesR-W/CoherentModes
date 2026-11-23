"""X3 x-depth: where along depth does the network redraw its grouping?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/depth.py r20s0b
    python3 site/check_fragment.py depth

Two measures of change between neighbouring sites, on one chart along depth (EXPERIMENTS_PLAN.md section 2, X3):
  sets       1 - matched Jaccard between adjacent sites' single-snapshot r = 4 SEBA sets, with the permutation null
             (site assets/questions.json q_carry["0_4"]; analysis.py docstring, Q2);
  operators  projector distance d_3 between adjacent sites' operators (explore/spectral_align/data pairs_0.npz `proj`,
             SPEC.md there), with the seed floor (d_3 between seeds 0 and 1 at the same site and checkpoint) and the
             cut diagnostics (span_0.npy; a cut k|k+1 is resolved when span_k >= 0.5 in both operators).
k = 3 because SEBA at r = 4 rotates [constant, psi_1, psi_2, psi_3] (schema_v2.py): d_3 compares exactly the spans the
r = 4 sets are cut from. The generator checks that the study's operators are the archive's (equal leading eigenvalues).
Final checkpoint and step 0 (control: an untrained network's neighbouring sites share structure through the shortcuts
and the image statistics). Writes assets/x/depth.json and assets/x/depth.facts.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import xlib  # noqa: E402

RULE, R_SETS, K_OP, K_ALT = 0, 4, 3, 4
STAB = 0.5
K_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64)
SA = xlib.ROOT.parent / "spectral_align" / "data"
NET0, NET1 = "resnet20-relu-seed0", "resnet20-relu-seed1"


def ord_color(t):
    """The ordinal ramp of js/colors.js (Palette.ord): palette.json 'ordinal' stops, linear in RGB."""
    stops = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in xlib.PALETTE["ordinal"]]
    x = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    c = [a + (b - a) * (x - i) for a, b in zip(stops[i], stops[i + 1])]
    return "rgb({},{},{})".format(*(round(v) for v in c))


def join(names):
    names = list(names)
    return "none" if not names else names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites = meta["steps"], meta["sites"]
    assert sites == xlib.SITES
    S, fin = len(sites), steps[-1]
    pairs = xlib.pair_labels(sites)
    npair = len(pairs)
    x = list(range(npair))
    Q = json.loads((xlib.SITE / "assets" / "questions.json").read_text())
    base = json.loads((xlib.SITE / "build_inputs" / "facts_base.json").read_text())
    P = xlib.PALETTE
    C_SET, C_OP = P["sets"][0], P["sets"][1]
    out = xlib.Out("depth")

    # ------------------------------------------------------------------ sets: adjacent matched Jaccard, r = 4
    C = Q["q_carry"][f"{RULE}_{R_SETS}"]
    assert C["adjacent"]["xticklabels"] == pairs
    names = [s["name"] for s in C["adjacent"]["series"]]
    t0, tf = names.index(f"step {steps[0]}"), names.index(f"step {fin}")
    J = {t: np.array(C["adjacent"]["series"][t]["y"], float) for t in (t0, tf)}
    Jlo = {t: np.array(C["adjacent_null"]["series"][t]["band"][0], float) for t in (t0, tf)}
    Jhi = {t: np.array(C["adjacent_null"]["series"][t]["band"][1], float) for t in (t0, tf)}
    jf, j0 = J[tf], J[t0]
    assert xlib.fmt(jf.min(), 2) == base[f"q.carry.r{R_SETS}.final.adjacent_min"]
    assert pairs[int(jf.argmin())] == base[f"q.carry.r{R_SETS}.final.adjacent_min_pair"]

    # ------------------------------------------------------------------ operators: adjacent d_k and seed floors
    cells = json.loads((SA / "cells.json").read_text())
    idx = {(c["net"], c["step"], c["site_index"]): c["index"] for c in cells}
    for c in cells:
        if c["net"] in (NET0, NET1):
            assert c["site_name"] == sites[c["site_index"]]
    stage = {c["site_index"]: c["stage"] for c in cells if c["net"] == NET0}
    proj = np.load(SA / "pairs_0.npz")["proj"]
    span = np.load(SA / "span_0.npy")
    eig = np.load(SA / "eig_0.npy", mmap_mode="r")
    lp = {tuple(p): i for i, p in enumerate(meta["layer_pairs"])}
    for st in (steps[0], fin):
        vals = np.load(src / f"layer_{st}.npz")["vals"]
        for s in range(S):
            assert np.allclose(vals[RULE, lp[(s, s)], :6], eig[idx[(NET0, st, s)], :6], atol=1e-5), (st, s)

    def profile(net, step, k):
        ki = K_GRID.index(k)
        c = [idx[(net, step, s)] for s in range(S)]
        ok = lambda i: bool(span[i, k - 1] >= STAB)
        d = np.array([proj[c[s], c[s + 1], ki] for s in range(npair)], float)
        res = np.array([ok(c[s]) and ok(c[s + 1]) for s in range(npair)])
        return d, res

    def floors(step, k):
        ki = K_GRID.index(k)
        ok = lambda i: bool(span[i, k - 1] >= STAB)
        fl, fok = np.zeros(S), np.zeros(S, bool)
        for s in range(S):
            a, b = idx[(NET0, step, s)], idx[(NET1, step, s)]
            fl[s], fok[s] = proj[a, b, ki], ok(a) and ok(b)
        lo = np.array([min([fl[s] for s in (p, p + 1) if fok[s]], default=np.nan) for p in range(npair)])
        hi = np.array([max([fl[s] for s in (p, p + 1) if fok[s]], default=np.nan) for p in range(npair)])
        return fl, fok, lo, hi

    df, rf = profile(NET0, fin, K_OP)
    d0, r0 = profile(NET0, steps[0], K_OP)
    fl, fok, flo, fhi = floors(fin, K_OP)
    dfr, d0r = np.where(rf, df, np.nan), np.where(r0, d0, np.nan)

    # ------------------------------------------------------------------ seed 1: final rule-0 SEBA sets (baselines run I3)
    z1 = np.load(xlib.DATA / "r20s0c" / "I3_seed1_layer_64000.npz")
    L1, v1 = z1["seba_label"].astype(np.int64), z1["vals"]                      # (pairs, ri, N), (pairs, 9)
    ri = meta["seba_r"].index(R_SETS)
    for s_ in range(S):    # same operators as the alignment study's seed-1 cells: pair order and probe order agree
        assert np.allclose(v1[lp[(s_, s_)], :6], eig[idx[(NET1, fin, s_)], :6], atol=1e-5), s_
    L0 = np.load(src / f"layer_{fin}.npz")["seba_label"][RULE, :, ri].astype(np.int64)
    lab0 = lambda s_: L0[lp[(s_, s_)]]
    lab1 = lambda s_: L1[lp[(s_, s_)], ri]
    for p_ in range(npair):        # seed 0 recomputed here matches q_carry
        assert abs(xlib.matched_jaccard(lab0(p_), lab0(p_ + 1), R_SETS, R_SETS)[0] - jf[p_]) < 1e-3
    perms = np.stack([np.random.default_rng(20261119 + i).permutation(meta["n"]) for i in range(200)])
    j1, j1hi, xs, xshi = np.zeros(npair), np.zeros(npair), np.zeros(S), np.zeros(S)
    for p_ in range(npair):
        sc, _, nl = xlib.matched_jaccard(lab1(p_), lab1(p_ + 1), R_SETS, R_SETS, perms)
        j1[p_], j1hi[p_] = sc, np.percentile(nl, 97.5)
    for s_ in range(S):
        sc, _, nl = xlib.matched_jaccard(lab1(s_), lab0(s_), R_SETS, R_SETS, perms)
        xs[s_], xshi[s_] = sc, np.percentile(nl, 97.5)
    out.fact("seed1.sets.min", j1.min(), 3)
    out.fact("seed1.sets.min_pair", pairs[int(j1.argmin())])
    out.fact("seed1.sets.max", j1.max(), 2)
    out.fact("seed1.sets.null_hi_max", j1hi.max(), 2)
    b2, b3 = sites.index("b2"), sites.index("b3")
    out.fact("xseed.b2", xs[b2], 3)
    out.fact("xseed.b3", xs[b3], 3)
    out.fact("xseed.min", xs.min(), 2)
    out.fact("xseed.min_site", sites[int(xs.argmin())])
    out.fact("xseed.max", xs.max(), 2)
    out.fact("xseed.max_site", sites[int(xs.argmax())])
    out.fact("xseed.null_hi_max", xshi.max(), 2)
    if not max(jf[b2], j1[b2]) < min(xs[b2], xs[b3]):
        raise SystemExit("b2->b3 agreement within a seed is not below the cross-seed agreement at b2 and b3; rewrite")
    if int(j1.argmin()) != int(jf.argmin()):
        raise SystemExit("seed 1's least-agreeing adjacent pair differs from seed 0's; rewrite the seed-1 sentence")

    out.chart("lead", xlib.lines(
        x, [
            dict(name="sets (r = 4): 1 − matched Jaccard, final checkpoint", y=1 - jf, color=C_SET, width=2.4,
                 band=[1 - Jhi[tf], 1 - Jlo[tf]]),
            dict(name="sets (r = 4): 1 − matched Jaccard, step 0 (control)", y=1 - j0, color=C_SET, dash=True, width=1.6),
            dict(name="sets (r = 4): 1 − matched Jaccard, seed 1, final checkpoint", y=1 - j1, color=C_SET, dash="dot", width=1.6),
            dict(name="operators: d₃, final checkpoint", y=dfr, color=C_OP, width=2.4, band=[flo, fhi]),
            dict(name="operators: d₃, step 0 (control)", y=d0r, color=C_OP, dash=True, width=1.6),
            dict(name="operators: d₃ at an unresolved cut, final checkpoint (sensitivity value)",
                 y=np.where(rf, np.nan, df), color=C_OP, dots=4, noline=True),
            dict(name="operators: d₃ at an unresolved cut, step 0 (sensitivity value)",
                 y=np.where(r0, np.nan, d0), color=P["mute"], dots=4, noline=True),
        ],
        "adjacent pair of sites (s → s+1)", "change between neighbouring sites (0 = identical)", yrange=(0, 1),
        xticklabels=pairs))

    # ------------------------------------------------------------------ the claims the text makes, checked
    rank_desc = lambda v: [int(i) for i in np.argsort(-np.nan_to_num(v, nan=-1))]
    of, o0 = rank_desc(dfr), rank_desc(d0r)
    if int(np.argmin(jf)) != of[0]:
        raise SystemExit("final: the lowest set agreement and the largest resolved d_3 are at different pairs; rewrite the answer")
    if int(np.argmin(j0)) != o0[0]:
        raise SystemExit("step 0: the lowest set agreement and the largest resolved d_3 are at different pairs; rewrite the answer")
    if int(np.argmax(jf)) != int(np.argmin(j0)):
        raise SystemExit("the step-0 low point is not the final high point of set agreement; rewrite the answer")
    if stage[of[0]] != stage[of[0] + 1]:
        raise SystemExit("the largest final change is at a stage boundary; rewrite 'inside the first stage'")
    if not (jf > Jhi[tf]).all() or not (j0 > Jhi[t0]).all():
        raise SystemExit("some adjacent set agreement is inside its null band; the text says every pair is above it")

    out.fact("rule", meta["modes"][RULE])
    out.fact("n_pairs", npair)
    # sets
    out.fact("sets.final.min", jf.min(), 2)
    out.fact("sets.final.min_pair", pairs[int(jf.argmin())])
    out.fact("sets.final.max", jf.max(), 2)
    out.fact("sets.final.max_pair", pairs[int(jf.argmax())])
    out.fact("sets.null_hi_max", max(Jhi[tf].max(), Jhi[t0].max()), 2)
    out.fact("sets.step0.min", j0.min(), 2)
    out.fact("sets.step0.min_pair", pairs[int(j0.argmin())])
    out.fact("sets.step0.mean", j0.mean(), 2)
    out.fact("sets.final.mean", jf.mean(), 2)
    i0 = int(j0.argmin())
    out.fact("sets.final.at_step0_min_pair", jf[i0], 2)
    # operators, k = 3
    out.fact("op.final.max", dfr[of[0]], 3)
    out.fact("op.final.max_pair", pairs[of[0]])
    out.fact("op.final.second", dfr[of[1]], 3)
    out.fact("op.final.second_pair", pairs[of[1]])
    out.fact("op.final.third", dfr[of[2]], 3)
    out.fact("op.final.third_pair", pairs[of[2]])
    out.fact("op.step0.max", d0r[o0[0]], 3)
    out.fact("op.final.at_step0_max_pair", df[o0[0]], 3)
    out.fact("op.final.at_step0_max_pair_resolved", "resolved" if rf[o0[0]] else "unresolved")
    out.fact("op.final.unresolved", join(pairs[i] for i in range(npair) if not rf[i]))
    out.fact("op.step0.unresolved", join(pairs[i] for i in range(npair) if not r0[i]))
    above = [i for i in range(npair) if rf[i] and np.isfinite(fhi[i]) and dfr[i] > fhi[i]]
    out.fact("op.final.above_floor", join(pairs[i] for i in above))
    out.fact("op.final.n_above_floor", len(above))
    out.fact("op.final.n_resolved", int(rf.sum()))
    out.fact("op.floor_min", fl[fok].min(), 3)
    out.fact("op.floor_max", fl[fok].max(), 3)
    out.fact("op.floor_unresolved", join(sites[s] for s in range(S) if not fok[s]))
    out.fact("op.chance", 1 - K_OP / (meta["n"] - 1), 4)
    # agreement of the two profiles (Spearman over the pairs whose cut is resolved)
    for nm, j, d, r in (("final", jf, df, rf), ("step0", j0, d0, r0)):
        rho = spearmanr(1 - j[r], d[r]).statistic
        out.fact(f"rho.{nm}", rho, 2)
        out.fact(f"rho.{nm}.n", int(r.sum()))
    # stage boundaries (stride-2 blocks): the first site of a new residual stage
    bnd = [p for p in range(npair) if 1 <= stage[p] and stage[p + 1] != stage[p] and stage[p + 1] <= 3]
    out.fact("stage_bounds", join(pairs[p] for p in bnd))
    out.fact("stage_bounds.sets", join(xlib.fmt(jf[p], 2) for p in bnd))
    out.fact("stage_bounds.op", join(xlib.fmt(df[p], 3) for p in bnd))
    # robustness: k = 4 and seed 1
    d4, r4 = profile(NET0, fin, K_ALT)
    o4 = rank_desc(np.where(r4, d4, np.nan))
    out.fact("k4.final.max_pair", pairs[o4[0]])
    out.fact("k4.final.max", d4[o4[0]], 3)
    out.fact("k4.final.second_pair", pairs[o4[1]])
    out.fact("k4.final.second", d4[o4[1]], 3)
    out.fact("k4.final.n_resolved", int(r4.sum()))
    d40, r40 = profile(NET0, steps[0], K_ALT)
    out.fact("k4.step0.max_pair", pairs[rank_desc(np.where(r40, d40, np.nan))[0]])
    if of[0] not in o4[:2]:
        raise SystemExit("at k = 4 the pair the sets single out is not among the two largest operator changes")
    out.fact("op.final.max_pair_stage", f"stage {stage[of[0]]}")
    d1, r1 = profile(NET1, fin, K_OP)
    o1 = rank_desc(np.where(r1, d1, np.nan))
    out.fact("seed1.final.max_pair", pairs[o1[0]])
    out.fact("seed1.final.max", d1[o1[0]], 3)
    out.fact("seed1.final.second_pair", pairs[o1[1]])

    # ------------------------------------------------------------------ App 4 (Reference charts)
    app4 = {}
    m = C["matrix"]
    app4["matrix"] = dict(xlib.heat(m["z"], sites, sites, "matched Jaccard between the two sites' r = 4 sets", "site", "site",
                                    slider=m["slider"], fmt="2f"),
                          caption=f"<b>Plotted:</b> matched Jaccard between the single-snapshot r = 4 SEBA sets of every "
                                  f"pair of sites at one checkpoint (slider), kernel rule {meta['modes'][RULE]}; no null.")
    ad = C["adjacent"]
    app4["sets_all_steps"] = dict(xlib.lines(
        x, [dict(name=s["name"], y=s["y"], color=ord_color(i / (len(ad["series"]) - 1)), width=1.6) for i, s in enumerate(ad["series"])],
        "adjacent pair of sites (s → s+1)", "matched Jaccard, site vs next site (r = 4)", yrange=(0, 1), xticklabels=pairs,
        refline=dict(y=float(max(Jhi[tf].max(), Jhi[t0].max())), label="largest null 97.5th percentile")),
        caption="<b>Plotted:</b> matched Jaccard between adjacent sites' r = 4 SEBA sets, one line per checkpoint "
                "(light: step 0, dark: final); grey line: the largest 97.5th percentile of the permutation null over pairs.")
    for w in (2, 3):
        o = C[f"w{w}"]
        app4[f"w{w}"] = dict(xlib.lines(
            o["x"], [dict(name=s["name"], y=s["y"], band=s["band"], color=ord_color(i / (len(ad["series"]) - 1)), width=1.6)
                     for i, s in enumerate(o["series"])], o["xlabel"], "matched Jaccard, block vs next block (r = 4)",
            yrange=(0, 1), xticklabels=o["xticklabels"], band_mode="envelope",
            band_label="shaded: permutation null, 2.5th–97.5th percentile (envelope)"),
            caption=f"<b>Plotted:</b> matched Jaccard between the r = 4 SEBA sets of a layer block of {w} adjacent sites "
                    f"and of the next {w} sites, one line per checkpoint (light: step 0, dark: final); shaded: "
                    "permutation-null envelope.")
    fl4, fok4, flo4, fhi4 = floors(fin, K_ALT)
    app4["operators_k4"] = dict(xlib.lines(
        x, [dict(name="d₄, final checkpoint", y=np.where(r4, d4, np.nan), color=C_OP, width=2.2, band=[flo4, fhi4]),
            dict(name="d₄, step 0", y=np.where(r40, d40, np.nan), color=C_OP, dash=True, width=1.6),
            dict(name=f"d₃, seed 1, final checkpoint", y=np.where(r1, d1, np.nan), color=P["mute"], dash="dot", width=1.6)],
        "adjacent pair of sites (s → s+1)", "projector distance between neighbouring sites", yrange=(0, 1), xticklabels=pairs),
        caption="<b>Plotted:</b> projector distance between adjacent sites' operators: d₄ for seed 0 at the final "
                "checkpoint (solid; shaded: the two sites' seed floors d₄ between seeds 0 and 1, lower to higher) and at "
                "step 0 (dashed), and d₃ for seed 1 at the final checkpoint (dotted). Unresolved cuts are gaps.")
    app4["cross_seed_sets"] = dict(xlib.lines(
        list(range(S)), [dict(name="seed 0 vs seed 1, same site, final checkpoint", y=xs, color=C_SET, width=2, band=[np.zeros(S), xshi])],
        "site", "matched Jaccard between the two seeds' r = 4 sets", yrange=(0, 1), xticklabels=sites),
        caption="<b>Plotted:</b> matched Jaccard between the single-snapshot r = 4 SEBA sets of seed 0 and seed 1 at the same "
                f"site, final checkpoint, kernel rule {meta['modes'][RULE]}; shaded up to the 97.5th percentile of a "
                "permutation null (200 permutations of seed 1's labels).")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
