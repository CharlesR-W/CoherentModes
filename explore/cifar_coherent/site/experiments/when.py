"""X2 x-when: when in training does each layer's grouping take its final form?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/when.py r20s0b
    python3 site/check_fragment.py when

Three operationalizations, three separate answers (EXPERIMENTS_PLAN.md section 2, X2):
  (a) agreement of each site's single-snapshot r = 4 SEBA sets with the same site's final sets (label-free);
  (b) alignment of the r = 2 sets with the post-hoc class bipartition {airplane, automobile, ship, truck} / other six
      (validation): normalised excess (score - null mean) / (ceiling - null mean);
  (c) projector distance d_4 between each site's operator and its final operator, against the two-seed floor
      (label-free).

Reads site/assets/questions.json (q_when, q_va, q_classmix; written by analysis.py / analysis_findings.py, whose
docstrings define every statistic), data/<tag>/curves.json (validation accuracy, display only) and the spectral-
alignment measurement explore/spectral_align/data (pairs_0.npz `proj`, span_0.npy, cells.json; SPEC.md there).
Writes assets/x/when.json and assets/x/when.facts.json. The milestones of (a) and (b) are recomputed here from the
stored arrays with the definitions of analysis.py and analysis_findings.py and checked against the shared facts; the
settling steps of (c) are recomputed with analyze.py's definition and checked against al.w2.r20s0.rule0.proj_k4.settled.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

RULE, R_SETS, R_REF, K_OP = 0, 4, 2, 4       # default kernel rule; r = 4 sets (a); r = 2 sets (b); cut k = 4 (c)
THRESH = 0.5                                  # milestone thresholds of analysis.py (a) and analysis_findings.py (b)
STAB = 0.5                                    # spectral_align: a cut k|k+1 is resolved when span_k >= 0.5
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


def step_txt(s):
    return "never" if s is None else f"{int(s):,}"


def group_sites(vals: dict, sites, fmt=step_txt, prefix="step "):
    """{site: value} -> 'step 1,024 at stem–b2 and b5–logits; step 2,560 at b3 and b4' (runs of consecutive sites)."""
    order = []
    for s in sites:
        if s not in vals:
            continue
        v = vals[s]
        if v not in order:
            order.append(v)
    if all(v is None or isinstance(v, (int, float)) for v in order):      # numeric steps: ascending, "never" last
        order.sort(key=lambda v: (v is None, v or 0))
    parts = []
    for v in order:
        idx = [i for i, s in enumerate(sites) if s in vals and vals[s] == v]
        runs, cur = [], [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        names = []
        for r in runs:
            names += [f"{sites[r[0]]}–{sites[r[-1]]}"] if len(r) >= 3 else [sites[i] for i in r]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        label = fmt(v)
        parts.append(f"{'' if label == 'never' else prefix}{label} at {joined}")
    return "; ".join(parts)


def first_stable(ok):
    return next((t for t in range(len(ok)) if np.all(ok[t:])), None)


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites = meta["steps"], meta["sites"]
    assert sites == xlib.SITES
    T, S = len(steps), len(sites)
    Q = json.loads((xlib.SITE / "assets" / "questions.json").read_text())
    base = json.loads((xlib.SITE / "build_inputs" / "facts_base.json").read_text())
    P = xlib.PALETTE
    out = xlib.Out("when")
    fin = steps[-1]

    # validation accuracy at the checkpoints (display only)
    cur = {c["step"]: c["val_acc"] for c in json.loads((src / "curves.json").read_text())["checkpoints"]}
    acc = [cur[s] for s in steps]

    # ================================================================== (a) agreement with the final sets
    W = Q["q_when"][f"{RULE}_{R_SETS}"]
    vf = np.array(W["vs_final"]["z"], float)                                     # (S, T)
    lv = W["lines_vs_final"]
    assert [s["name"] for s in lv["series"]] == sites and lv["x"] == steps
    vf_lo = np.array([s["band"][0] for s in lv["series"]], float)
    vf_hi = np.array([s["band"][1] for s in lv["series"]], float)
    ok = (vf > THRESH) & (vf > vf_hi)
    first = {sites[s]: next((steps[t] for t in range(T) if ok[s, t]), None) for s in range(S)}
    settled = {sites[s]: (lambda i: None if i is None else steps[i])(first_stable(ok[s])) for s in range(S)}
    for s in sites:      # same definition as analysis.py summary.when -> shared facts q.when.r4.<site>.*
        assert step_txt(first[s]) == base[f"q.when.r{R_SETS}.{s}.first_step"], s
        assert step_txt(settled[s]) == base[f"q.when.r{R_SETS}.{s}.settled_step"], s

    ser = [dict(name=sites[s], y=vf[s], band=[vf_lo[s], vf_hi[s]], color=ord_color(s / (S - 1)),
                width=1.8) for s in range(S)]
    ser.append(dict(name="validation accuracy (all 10 classes; display only)", y=acc, color=P["fg"], dash="dot", width=1.4))
    out.chart("lead", xlib.lines(
        steps, ser, "training step (symmetric-log axis)", "matched Jaccard with the site's final sets (r = 4)",
        xscale="symlog", yrange=(0, 1), refline=dict(y=THRESH, label="milestone threshold 0.5"),
        band_mode="envelope", band_label="shaded: permutation null of each site, 2.5th–97.5th percentile (envelope over sites)"))

    tix = {s: i for i, s in enumerate(steps)}
    out.fact("a.first", group_sites(first, sites))
    diff = [s for s in sites if settled[s] != first[s]]
    out.fact("a.settled_diff", "; ".join(f"{s} (above it at every later checkpoint only from step {step_txt(settled[s])})" for s in diff) if diff else "none")
    out.fact("a.null_hi_max", vf_hi[:, :-1].max(), 2)
    out.fact("a.step0_min", vf[:, 0].min(), 2)
    out.fact("a.step0_max", vf[:, 0].max(), 2)
    out.fact("a.step0_above_null", int((vf[:, 0] > vf_hi[:, 0]).sum()))
    t26 = tix[25600]
    out.fact("a.step25600_min", vf[:, t26].min(), 2)
    out.fact("a.step25600_max", vf[:, t26].max(), 2)
    # dips after the first milestone (sites whose agreement falls back to <= 0.5 after first exceeding it)
    dips = []
    for s in range(S):
        f = tix[first[sites[s]]] if first[sites[s]] is not None else None
        if f is not None:
            low = [t for t in range(f + 1, T) if not ok[s, t]]
            if low:
                dips.append(f"{sites[s]} ({', '.join(xlib.fmt(vf[s, t], 2) + ' at step ' + step_txt(steps[t]) for t in low)})")
    out.fact("a.dips", "; ".join(dips) if dips else "none")
    for s in (1024, 2560):
        out.fact(f"acc.step{s}", acc[tix[s]], pct=True)
    out.fact("acc.final", acc[-1], pct=True)
    out.fact("n_sites", S)
    out.fact("n_steps", T)
    out.fact("steps", ", ".join(step_txt(s) for s in steps))
    out.fact("rule", meta["modes"][RULE])

    # ================================================================== (b) the post-hoc class bipartition, r = 2
    V = Q["q_va"]["by_mode"][str(RULE)]
    z = lambda k: np.array(V[k]["z"], float)
    score, mu, lo, hi, nmax, ceil, nexc = (z("score"), z("null_mean"), None, z("null_hi"), z("null_max"), z("ceiling"),
                                           z("norm_excess"))
    assert np.allclose(nexc, (score - mu) / (ceil - mu), atol=2e-3, equal_nan=True)
    half = {sites[s]: (lambda i: None if i is None else steps[i])(first_stable(nexc[s] >= THRESH)) for s in range(S)}
    for s in sites:      # analysis_findings.py half_step -> q.va.<site>.half_step
        assert step_txt(half[s]).replace("never", "none") == base[f"q.va.{s}.half_step"], s
    beat0 = int((score[:, 0] > nmax[:, 0]).sum())
    assert base["q.va.onset_step.all_sites"] == "0" and beat0 == S
    pix = np.array(Q["q_va"]["pixel_baseline"]["score"]["z"], float)
    above_pix = {sites[s]: (lambda i: None if i is None else steps[i])(first_stable(score[s] > pix[s])) for s in range(S)}
    out.fact("b.half", group_sites(half, sites))
    out.fact("b.step0_beat_max", beat0)
    out.fact("b.step0_nexc_min", nexc[:, 0].min(), 2)
    out.fact("b.step0_nexc_max", nexc[:, 0].max(), 2)
    out.fact("b.final_nexc_min", nexc[:, -1].min(), 2)
    out.fact("b.final_nexc_max", nexc[:, -1].max(), 2)
    never = [i for i in range(S) if half[sites[i]] is None]
    out.fact("b.never", ", ".join(sites[i] for i in never[:-1]) + (" and " if len(never) > 1 else "") + sites[never[-1]])
    out.fact("b.never_final_nexc", f"{xlib.fmt(nexc[never, -1].min(), 2)} to {xlib.fmt(nexc[never, -1].max(), 2)}")
    # the text says (b) is ordered by depth (deeper sites reach the milestone no later) and (a) is not
    key = lambda v: np.inf if v is None else v
    hs = [key(half[s]) for s in sites]
    if any(hs[i] < hs[i + 1] for i in range(S - 1)):
        raise SystemExit("(b) milestone is not non-increasing with depth; rewrite 'deep sites first'")
    fs_ = [key(first[s]) for s in sites]
    if all(fs_[i] >= fs_[i + 1] for i in range(S - 1)) or all(fs_[i] <= fs_[i + 1] for i in range(S - 1)):
        raise SystemExit("(a) milestone is monotone in depth; rewrite 'no depth order'")
    out.fact("b.above_pixel", group_sites(above_pix, sites))
    out.fact("b.step0_pixel_wins", int((pix[:, 0] >= score[:, 0]).sum()))
    b9 = sites.index("b9")
    for nm, arr in (("score", score), ("null_mean", mu), ("ceiling", ceil), ("pixel", pix)):
        out.fact(f"b.b9.step0.{nm}", arr[b9, 0], 2)
        out.fact(f"b.b9.final.{nm}", arr[b9, -1], 2)
    rank = np.array(Q["q_va"]["best_bipartition"]["va_rank"]["z"], float)
    out.fact("b.rank_final_n1", int((rank[:, -1] == 1).sum()))
    out.fact("b.rank_step0_min", int(rank[:, 0].min()))
    out.fact("b.rank_step0_max", int(rank[:, 0].max()))

    out.chart("nexc", xlib.lines(
        steps, [dict(name=sites[s], y=nexc[s], color=ord_color(s / (S - 1)), width=1.8) for s in range(S)],
        "training step (symmetric-log axis)", "normalised excess (0 = null mean, 1 = size ceiling)", xscale="symlog",
        yrange=(-0.2, 1), refline=dict(y=THRESH, label="milestone threshold 0.5")))
    out.chart("pixel_b9", xlib.lines(
        steps, [dict(name="b9 sets (r = 2) vs the class bipartition", y=score[b9], color=P["sets"][0], width=2.2),
                dict(name="permutation null mean (band: 2.5th–97.5th percentile)", y=mu[b9], color=P["mute"], dash=True, width=1.4,
                     band=[np.array(V["lines"]["series"][b9]["band"][0]), hi[b9]]),
                dict(name="size ceiling", y=ceil[b9], color=P["mute"], dash="dot", width=1.4),
                dict(name="best single pixel-statistic split of the same sizes (chosen with the labels)", y=pix[b9],
                     color=P["sets"][1], dash=True, width=2)],
        "training step (symmetric-log axis)", "matched Jaccard with the class bipartition", xscale="symlog", yrange=(0, 1)))
    assert V["lines"]["series"][b9]["name"] == "b9"

    # ================================================================== (c) operator distance to the final operator
    cells = json.loads((SA / "cells.json").read_text())
    idx = {(c["net"], c["step"], c["site_index"]): c["index"] for c in cells}
    for c in cells:
        if c["net"] == NET0:
            assert c["site_name"] == sites[c["site_index"]]
    proj = np.load(SA / "pairs_0.npz")["proj"]                                    # (cells, cells, len(K_GRID))
    span = np.load(SA / "span_0.npy")
    eig = np.load(SA / "eig_0.npy", mmap_mode="r")
    # the study's operators are the archive's: same leading eigenvalues at every site and checkpoint
    for t, st in enumerate(steps):
        vals = np.load(src / f"layer_{st}.npz")["vals"]
        lp = {tuple(p): i for i, p in enumerate(meta["layer_pairs"])}
        for s in range(S):
            assert np.allclose(vals[RULE, lp[(s, s)], :6], eig[idx[(NET0, st, s)], :6], atol=1e-5), (st, s)
    def op_settle(k):
        """analyze.py W2 'settled' at cut k: per site, D (S, T), unresolved points, floor, floor resolved, settled."""
        ki = K_GRID.index(k)
        res = lambda c: bool(span[c, k - 1] >= STAB)
        D = np.full((S, T), np.nan)
        unres = np.zeros((S, T), bool)
        floor = np.full(S, np.nan)
        floor_ok = np.zeros(S, bool)
        settled_c = {}
        for s in range(S):
            f, g = idx[(NET0, fin, s)], idx[(NET1, fin, s)]
            floor[s], floor_ok[s] = proj[f, g, ki], res(f) and res(g)
            for t, st in enumerate(steps):
                c = idx[(NET0, st, s)]
                D[s, t] = proj[c, f, ki]
                unres[s, t] = t < T - 1 and not (res(c) and res(f))
            if not floor_ok[s]:
                settled_c[sites[s]] = "n/a (floor cut unresolved)"
                continue
            i = first_stable(D[s] <= floor[s])
            late = [t for t in range(T - 1) if unres[s, t] and (i is None or t >= i)]
            settled_c[sites[s]] = (f"n/a (cut unresolved at step {steps[late[-1]]:,})" if late else
                                   ("never" if i is None else f"{steps[i]:,}"))
        return D, unres, floor, floor_ok, settled_c

    D, unres, floor, floor_ok, settled_c = op_settle(K_OP)
    # robustness: k = 3 (the span the r = 4 sets are cut from) and the share of resolved snapshot cuts per k
    _, _, _, _, set3 = op_settle(3)
    out.fact("c.k3.settled", "; ".join(f"{s} {v}" for s, v in set3.items()).replace(f"{fin:,}", f"{fin:,} (final only)"))
    own = [c["index"] for c in cells if c["net"] == NET0]
    for k in (3, K_OP, 9):
        out.fact(f"c.resolved_share_k{k}", float((span[own, k - 1] >= STAB).mean()), pct=True)
    # (a) at the other stored ranks: range of the first milestone over sites
    for r in (2, 3, 6, 8):
        Wr = Q["q_when"].get(f"{RULE}_{r}")
        if Wr is None:
            continue
        v = np.array(Wr["vs_final"]["z"], float)
        h = np.array([s_["band"][1] for s_ in Wr["lines_vs_final"]["series"]], float)
        okr = (v > THRESH) & (v > h)
        fs = [next((steps[t] for t in range(T) if okr[s_, t]), None) for s_ in range(S)]
        out.fact(f"a.r{r}.first_range", f"{min(fs):,} to {max(fs):,}" if min(fs) != max(fs) else f"{min(fs):,} at every site")
    assert "; ".join(f"{s} {settled_c[s]}" for s in sites) == base[f"al.w2.r20s0.rule{RULE}.proj_k{K_OP}.settled"]
    stepped = {s: v for s, v in settled_c.items() if not v.startswith("n/a")}
    only_final = [s for s in sites if stepped.get(s) == f"{fin:,}"]
    before = {k: v for k, v in settled_c.items() if not v.startswith("n/a") and v != f"{fin:,}"}
    out.fact("c.settled_before", group_sites(before, sites, fmt=lambda v: v))
    L = only_final
    out.fact("c.only_final", L[0] if len(L) == 1 else (f"{L[0]}–{L[-1]}" if [sites.index(x) for x in L] ==
             list(range(sites.index(L[0]), sites.index(L[-1]) + 1)) else ", ".join(L)))
    out.fact("c.unresolved_floor", ", ".join(sites[i] for i in range(S) if not floor_ok[i]) or "none")
    out.fact("c.floor_min", floor[floor_ok].min(), 3)
    out.fact("c.floor_max", floor[floor_ok].max(), 3)
    out.fact("c.step0_min", D[:, 0].min(), 2)
    out.fact("c.step0_max", D[:, 0].max(), 2)
    out.fact("c.chance", 1 - K_OP / (meta["n"] - 1), 4)
    late = [sites.index(s) for s in only_final]
    out.fact("c.late_step25600_min", D[late, t26].min(), 3)
    out.fact("c.late_step25600_max", D[late, t26].max(), 3)
    out.fact("c.late_floor_min", floor[late].min(), 3)
    out.fact("c.late_floor_max", floor[late].max(), 3)
    out.fact("c.n_unresolved_points", int(unres.sum()))
    out.fact("c.n_points", int(S * (T - 1)))
    out.fact("c.lr_drops", " and ".join(f"{m:,}" for m in json.loads((src / "curves.json").read_text())["config"]["lr_schedule"]["milestones"]))

    Dr = np.where(unres, np.nan, D)
    um = [(s, t) for s in range(S) for t in range(T) if unres[s, t]]
    # unresolved cuts are gaps (d_k there is a sensitivity value, not a property of the operators); listed in the caption
    ser = [dict(name=sites[s], y=Dr[s], color=ord_color(s / (S - 1)), width=1.8) for s in range(S)]
    fl_lo, fl_hi = floor[floor_ok].min(), floor[floor_ok].max()
    out.chart("operator", xlib.lines(
        steps, ser, "training step (symmetric-log axis)", f"projector distance d_{K_OP} to the site's final operator",
        xscale="symlog", yrange=(0, 1), band=[[fl_lo] * T, [fl_hi] * T],
        band_label=f"shaded: seed floor d_{K_OP} at the final checkpoint, range over the sites whose floor cut is resolved"))
    out.fact("c.unresolved_list", "; ".join(f"{sites[s]} at step {steps[t]:,}" for s, t in um) or "none")

    # ================================================================== App 4 (Reference charts)
    app4 = {}
    vc = W["lines_vs_class"]
    ceil_c = np.array(W["vs_class_ceiling"]["z"], float)
    app4["vs_final_heat"] = dict(xlib.heat(vf.tolist(), sites, steps, "matched Jaccard with the site's final sets (r = 4)", "site",
                                           "checkpoint (step)", fmt="2f"),
                                 caption=f"<b>Plotted:</b> matched Jaccard of each site's single-snapshot r = 4 SEBA sets (rows) at each "
                                         f"checkpoint (columns) with the same site's sets at step {fin:,}; kernel rule {meta['modes'][RULE]}. "
                                         "The last column is a self-comparison.")
    app4["vs_class_lines"] = dict(xlib.lines(
        steps, [dict(name=s["name"], y=s["y"], band=s["band"], color=ord_color(i / (S - 1)), width=1.6)
                for i, s in enumerate(vc["series"])]
        + [dict(name="size ceiling (mean over sites)", y=ceil_c.mean(0), color=P["mute"], dash=True, width=1.4)],
        "training step (symmetric-log axis)", "matched Jaccard with the ten classes (r = 4)", xscale="symlog", yrange=(0, 1),
        band_mode="envelope", band_label="shaded: permutation null, 2.5th–97.5th percentile (envelope over sites)"),
        caption="<b>Plotted:</b> matched Jaccard of each site's r = 4 SEBA sets with the ten CIFAR-10 classes (validation) "
                "against training step, one line per site (light: stem, dark: logits); dashed: the size ceiling, mean over "
                "sites; shaded: permutation-null envelope.")
    app4["rank511"] = dict(xlib.heat(rank.tolist(), sites, steps, "rank of {airplane, automobile, ship, truck} / other six among the 511 class bipartitions (1 = best match)",
                                     "site", "checkpoint (step)", zrange=[1, 12], fmt="int"),
                           caption="<b>Plotted:</b> for the r = 2 SEBA sets of each site (rows) at each checkpoint (columns), the rank "
                                   "of the post-hoc reference bipartition among all 511 two-sided splits of the ten classes, ordered "
                                   "by matched Jaccard with the sets (1 = the reference matches best). Descriptive only: the "
                                   "reference was chosen after seeing the final split, and no selection-adjusted test is made. "
                                   "Colour scale clipped at 12.")
    cp = Q["q_va"]["class_placement"]
    app4["class_placement"] = dict(xlib.heat(cp["in_matched"]["z"], cp["in_matched"]["rows"], cp["in_matched"]["cols"],
                                             "fraction of the class in the r = 2 set matched to its side of the reference",
                                             "site", "CIFAR-10 class (validation)", slider=cp["in_matched"]["slider"], fmt="2f"),
                                   caption="<b>Plotted:</b> for each site (rows) and class (columns), the fraction of the "
                                           "class's images that lie in the r = 2 SEBA set matched (Hungarian) to the class's "
                                           "side of the reference bipartition {airplane, automobile, ship, truck} / other six; "
                                           "slider: checkpoint. Unassigned images count as not placed.")
    cm = Q["q_classmix"]["cells"]["2"]["b9"]
    app4["classmix_b9"] = dict(xlib.heat(cm["z"], cm["rows"], cm["cols"], "images", "SEBA set (r = 2)", "CIFAR-10 class (validation)",
                                         zrange=[0, 200], slider=cm["slider"], fmt="int"),
                               caption="<b>Plotted:</b> number of probe images of each class (columns) in each r = 2 SEBA set "
                                       "of b9 and among the unassigned (rows); slider: checkpoint; 200 images per class.")
    app4["operator_floor_ratio"] = dict(xlib.lines(
        steps, [dict(name=sites[s], y=(Dr[s] / floor[s]) if floor_ok[s] else [None] * T,
                     color=ord_color(s / (S - 1)), width=1.6) for s in range(S)],
        "training step (symmetric-log axis)", f"d_{K_OP} to the final operator / the site's seed floor", xscale="symlog",
        yrange=xlib.nice_range([np.where(floor_ok[:, None], Dr / floor[:, None], np.nan)], step=1),
        refline=dict(y=1, label="seed floor")),
        caption=f"<b>Plotted:</b> projector distance d_{K_OP} between each site's operator and its final operator divided "
                "by that site's seed floor (d_4 between seeds 0 and 1 at the final checkpoint); sites whose floor cut is "
                "unresolved and points whose cut is unresolved are omitted.")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
