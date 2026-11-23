"""X8 x-across-nets: in the tested ResNets, do different seeds and depths go through matching sequences of operators?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/across-nets.py r20s0b
    python3 site/check_fragment.py across-nets

Reads the spectral-alignment analysis (explore/spectral_align/analysis/data/, written by
explore/spectral_align/analyze.py from explore/spectral_align/data/; the same files build_site.py --alignment copies
to d/al/): index.json (networks, cross-network comparison list), summary_r{0,1}.json (per distance and comparison:
DTW cost, null quantiles, seed-floor costs, unresolved share, inversions, stage ends) and diag_r0.bin (per-snapshot
cut reproducibility span_k). For the reference chart of seed-0 vs seed-1 set agreement it reads
data/r20s0c/sources_layer_<final>.npz (agree_I3; explore/cifar_baselines) and data/r20s0c/meta_c.json.
Writes assets/x/across-nets.json and assets/x/across-nets.facts.json. No new computation beyond medians, ranges and
the "below the null from" step, which is recomputed here for every distance quoted (the alignment build exports it
only for some cuts).

Summary row fields (analyze.py): cost = normalised DTW cost; n05/n50/n95 = permutation-null quantiles; pnull =
fraction of null draws with cost <= observed; const = constant-matrix reference; unres = share of the DTW path whose
cut is unresolved in either snapshot (projector distance only); floor_a/floor_b = DTW cost between the two seeds of
A's depth / B's depth at the same checkpoint; inv_r/inv_c = inversion fraction of the row/column argmins; ends =
stage ends passed (of 3); gap = mean relative-depth gap from the architectural reference.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

AL = xlib.ROOT.parent / "spectral_align" / "analysis" / "data"
RULE = 0                       # default kernel rule (union 15-NN graph); rule 1 only in Caveats
K_LO, K_HI = 3, 9              # the two cuts reported (see the fragment's "How I measure it")
LEAD = (20, 56)                # depth pair of the lead figure: the largest depth difference tested
DEPTH_PAIRS = [(20, 32), (20, 56), (32, 56)]
SEED_PAIRS = [(20, 20), (32, 32), (56, 56)]


N_STAGE_PERM = 1000           # draws of the stage-preserving shuffle (computed here; analyze.py has only the global one)


def dtw_cost_batch(d):
    """Copy of explore/spectral_align/analyze.py dtw_cost_batch: normalized symmetric2 DTW optimum for a batch of cost
    matrices d (R, LA, LB), pinned at both corners; g[-1,-1] / (LA + LB - 1) = align.dtw_alignment's 'cost'."""
    d = np.moveaxis(np.asarray(d, np.float64), 0, -1)
    LA, LB = d.shape[:2]
    g = np.empty_like(d)
    g[0, 0] = d[0, 0]
    for j in range(1, LB):
        g[0, j] = g[0, j - 1] + d[0, j]
    for i in range(1, LA):
        g[i, 0] = g[i - 1, 0] + d[i, 0]
        a = np.minimum(g[i - 1, 1:] + d[i, 1:], g[i - 1, :-1] + 2 * d[i, 1:])
        for j in range(1, LB):
            g[i, j] = np.minimum(a[j - 1], g[i, j - 1] + d[i, j])
    return g[-1, -1] / (LA + LB - 1)


def stage_perms(stages, n, seed=0):
    """n permutations of B's sites that shuffle residual blocks only within their own stage (stem, pool, logits and
    stage membership fixed)."""
    rng = np.random.default_rng(seed)
    st = np.asarray(stages)
    groups = [np.flatnonzero(st == s) for s in (1, 2, 3)]
    out = np.tile(np.arange(len(st)), (n, 1))
    for r in range(n):
        for g in groups:
            out[r, g] = g[rng.permutation(len(g))]
    return out


def main(tag: str):
    I = json.loads((AL / "index.json").read_text())
    if I["fixture"]:
        raise SystemExit(f"{AL} is a synthetic fixture; the experiment needs the measured alignment data")
    S = {r: json.loads((AL / f"summary_r{r}.json").read_text()) for r in I["rules"]}
    nets, steps, fin = I["nets"], I["main_steps"], I["final"]
    F = S[RULE]["fields"]
    P = xlib.PALETTE
    out = xlib.Out("across-nets")

    # ------------------------------------------------------------------ comparison lookup
    cross = {}
    for ci, c in enumerate(I["cross"]):
        if c["kind"] == "step":
            cross[(c["a"], c["b"], c["step"])] = ci
    nid = {n["id"]: i for i, n in enumerate(nets)}

    def pairings(da, db, same_seed=False):
        """Network index pairs (a, b) of depths da, db; different-seed pairings unless same_seed."""
        res = []
        for (a, b, st) in cross:
            if st != fin:
                continue
            A, B = nets[a], nets[b]
            if sorted((A["depth"], B["depth"])) != sorted((da, db)):
                continue
            if A["depth"] == da and B["depth"] == db or da == db:
                if (A["seed"] == B["seed"]) == same_seed:
                    res.append((a, b))
        return sorted(set(res))

    def row(dist, a, b, step, rule=RULE):
        ci = cross.get((a, b, step))
        r = S[rule]["dists"][dist][ci] if ci is not None else None
        if r is None:
            return None
        d = dict(zip(F, r))
        return d if d["cost"] is not None and np.isfinite(d["cost"]) else None

    def name(a, b):
        A, B = nets[a], nets[b]
        return f"ResNet-{A['depth']} seed {A['seed']} – ResNet-{B['depth']} seed {B['seed']}"

    def series_over(dist, a, b, f):
        return [(row(dist, a, b, s) or {}).get(f) for s in steps]

    def below_from(dist, prs, rule=RULE):
        """Earliest checkpoint from which on every pairing's cost is below its null's 5th percentile (None: never)."""
        ok = [all((r := row(dist, a, b, s, rule)) is not None and r["cost"] < r["n05"] for a, b in prs) for s in steps]
        for i in range(len(steps)):
            if all(ok[i:]):
                return steps[i]
        return None

    dpairs = {dp: pairings(*dp) for dp in DEPTH_PAIRS}
    spairs = {sp: pairings(*sp) for sp in SEED_PAIRS}
    assert all(len(v) == 2 for v in dpairs.values()) and all(len(v) == 1 for v in spairs.values()), (dpairs, spairs)
    cross_diff = [p for v in dpairs.values() for p in v]          # six cross-depth, different-seed pairings
    seed_pairs = [p for v in spairs.values() for p in v]          # three same-depth (two-seed) pairs

    # ------------------------------------------------------------------ cut reproducibility (label-free) at the final step
    W = I["n_diag_eig"] + I["n_span"]
    q = np.fromfile(AL / f"diag_r{RULE}.bin", dtype=np.uint16).reshape(-1, W).astype(float)
    q[q == 65535] = np.nan
    span = q[:, I["n_diag_eig"]:] / 65534
    fin_groups = [g for g in I["groups"] if g["step"] == fin and g["main"]]
    fin_cells = [c for g in fin_groups for c in g["cells"]]
    res = lambda cells, k: np.nan_to_num(span[cells, k - 1], nan=-1) >= I["stab_threshold"]
    out.fact("n_final_snapshots", len(fin_cells))
    for k in (K_LO, K_HI):
        out.fact(f"resolved_share.k{k}", res(fin_cells, k).mean(), 0, pct=True)
    # K_LO is described as the largest cut of the grid resolved in at least nine tenths of the final snapshots
    big = [k for k in I["k_grid"] if res(fin_cells, k).mean() >= 0.9]
    if max(big) != K_LO:
        raise SystemExit(f"largest cut resolved in >= 90% of final snapshots is {max(big)}, not {K_LO}: fix the text")
    out.fact("resolved_share.k_lo_min", min(res(fin_cells, k).mean() for k in range(1, K_LO + 1)), 0, pct=True)
    # sites whose cut K_HI is resolved in every network at the final step (label-free; stated in the text)
    always = [s for s in ("pool", "logits") if all(res([g["cells"][nets[g["net"]]["sites"].index(s)]], K_HI)[0]
                                                    for g in fin_groups)]
    if always != ["pool", "logits"]:
        raise SystemExit(f"cut {K_HI} is not resolved at pool and logits of every network: fix the text's justification")

    # ------------------------------------------------------------------ facts: final step, rule 0
    # Per-distance summaries are computed for every distance below, but only the names in QUOTED are written as facts
    # (the fragment quotes these; the rest would be unused facts).
    QUOTED = {
        "k3": {"depth.cost", "seed.cost", "depth.null05", "depth.ratio_max", "depth.unres", "seed.unres",
               "depth.inv_hi", "depth.const", "depth.ends_lo", "depth.n_all_ends", "depth.gap", "below_from.20_32",
               "below_from.20_56", "below_from.32_56", "step0.n_below"},
        "k9": {"depth.cost", "seed.cost", "depth.null05", "depth.ratio_max", "depth.unres", "seed.unres",
               "depth.inv_hi", "depth.n_all_ends", "step0.n_below"},
        "h16": {"depth.cost", "seed.cost", "depth.null05", "depth.ratio_max", "below_from.depth_max"},
        "cka": {"depth.cost", "seed.cost", "depth.inv_hi"},
    }

    def put(key, name, v, d=3, pct=False, kind="value"):
        """kind: value (one fact) or range (name_lo, name_hi)."""
        if name not in QUOTED[key]:
            return
        if kind == "range":
            out.fact(f"{key}.{name}_lo", min(v), 0 if pct else d, pct=pct)
            out.fact(f"{key}.{name}_hi", max(v), 0 if pct else d, pct=pct)
        else:
            out.fact(f"{key}.{name}", v, d, pct=pct)

    def put_range(name, v, d=3, pct=False):
        out.fact(name + "_lo", min(v), 0 if pct else d, pct=pct)
        out.fact(name + "_hi", max(v), 0 if pct else d, pct=pct)

    below_all = {}
    for dist, key in ((f"proj_k{K_LO}", f"k{K_LO}"), (f"proj_k{K_HI}", f"k{K_HI}"), ("heat_t16", "h16"), ("cka", "cka")):
        R = [row(dist, a, b, fin) for a, b in cross_diff]
        Rs = [row(dist, a, b, fin) for a, b in seed_pairs]
        assert all(R) and all(Rs), f"missing final-step summaries for {dist}"
        rg = lambda name, v, d=3, pct=False: put(key, name, v, d, pct, kind="range")
        rg("depth.cost", [r["cost"] for r in R])
        rg("seed.cost", [r["cost"] for r in Rs])          # = the seed floors at the final step
        rg("depth.null05", [r["n05"] for r in R])
        put(key, "depth.ratio_max", max(r["cost"] / max(r["floor_a"], r["floor_b"]) for r in R), 2)
        if dist.startswith("proj") and max(r["pnull"] for r in R + Rs) > 0:   # the text says "none of the null draws"
            raise SystemExit(f"{dist}: some null draw reaches the observed cost at the final step; fix the answer")
        put(key, "depth.inv_hi", max(max(r["inv_r"], r["inv_c"]) for r in R), 4)
        if dist.startswith("proj"):
            rg("depth.unres", [r["unres"] for r in R], pct=True)
            rg("seed.unres", [r["unres"] for r in Rs], pct=True)
            rg("depth.const", [r["const"] for r in R])
            ends = [r["ends"] for r in R]
            put(key, "depth.ends_lo", min(ends), None)
            put(key, "depth.n_all_ends", sum(e == 3 for e in ends), None)
            rg("depth.gap", [r["gap"] for r in R], 2)
        # checkpoints: earliest step from which on every different-seed pairing is below its null's 5th percentile
        bf = {dp: below_from(dist, dpairs[dp]) for dp in DEPTH_PAIRS}
        bs = {sp: below_from(dist, spairs[sp]) for sp in SEED_PAIRS}
        if any(v is None for v in list(bf.values()) + list(bs.values())):
            raise SystemExit(f"{dist}: some pairing is not below the null at the final step; rewrite the answer")
        for dp, v in bf.items():
            put(key, f"below_from.{dp[0]}_{dp[1]}", v, None)
        put(key, "below_from.depth_max", max(list(bf.values()) + list(bs.values())), None)
        below_all[key] = max(list(bf.values()) + list(bs.values()))
        # step 0 (untrained reference): how many of the nine different-seed comparisons are already below the null
        put(key, "step0.n_below", sum(1 for a, b in cross_diff + seed_pairs
                                      if (r := row(dist, a, b, steps[0])) and r["cost"] < r["n05"]), None)
    # ---- matches and the stage-preserving shuffle (final step, cross-depth different-seed pairings, rule 0)
    for k in (K_LO, K_HI):
        dist = f"proj_k{k}"
        paths = json.loads((AL / "s" / f"{dist}_r{RULE}.json").read_text())
        import gzip
        qv = np.frombuffer(gzip.decompress((AL / "s" / f"{dist}_r{RULE}.bin.gz").read_bytes()), np.uint16).astype(float)
        vals = np.where(qv == 65535, np.nan, qv * I["scale"][f"{dist}_r{RULE}"] / 65534)
        cover_ab, cover_ba, sn05, sbelow, spn = [], [], [], 0, []
        for a, b in cross_diff:
            ci = cross[(a, b, fin)]
            sg = I["segs"][I["cross"][ci]["seg"]]
            D = vals[sg["o"]: sg["o"] + sg["nr"] * sg["nc"]].reshape(sg["nr"], sg["nc"])
            LA, LB = len(nets[a]["sites"]), len(nets[b]["sites"])
            assert D.shape == (LA, LB) and np.isfinite(D).all()
            obs = float(dtw_cost_batch(D[None])[0])
            stored = row(dist, a, b, fin)["cost"]
            if abs(obs - stored) > 1e-3:
                raise SystemExit(f"{dist} {name(a, b)}: recomputed DTW cost {obs} != stored {stored}")
            pr = paths[ci]
            assert len(pr["ra"]) == LA and len(pr["ca"]) == LB
            cover_ab.append(len(set(pr["ra"])) / LA)        # distinct nearest sites of B hit by A's LA sites
            cover_ba.append(len(set(pr["ca"])) / LA)        # sites of A (the shallower) hit by B's nearest matches
            perms = stage_perms(nets[b]["stages"], N_STAGE_PERM)
            null = dtw_cost_batch(D[:, perms].transpose(1, 0, 2))
            q05 = float(np.percentile(null, 5))
            sn05.append(q05)
            sbelow += obs < q05
            spn.append(float(np.mean(null <= obs)))
        out.fact(f"k{k}.match.ab_lo", min(cover_ab), 0, pct=True)
        out.fact(f"k{k}.match.ab_hi", max(cover_ab), 0, pct=True)
        out.fact(f"k{k}.match.ba_lo", min(cover_ba), 0, pct=True)
        out.fact(f"k{k}.match.ba_hi", max(cover_ba), 0, pct=True)
        out.fact(f"k{k}.stage_null.null05_lo", min(sn05), 3)
        out.fact(f"k{k}.stage_null.null05_hi", max(sn05), 3)
        out.fact(f"k{k}.stage_null.n_below", int(sbelow))
        if max(spn) > 0:          # the text says "none of the stage-preserving draws reaches the observed cost"
            raise SystemExit(f"{dist}: a stage-preserving draw reaches the observed cost; fix the text")
    out.fact("n_stage_perm", N_STAGE_PERM)
    # the answer's "at both cuts ... from step X on": X = the later of the two cuts' steps
    out.fact("below_from.both_cuts", max(below_all[f"k{K_LO}"], below_all[f"k{K_HI}"]))
    out.fact("n_cross_diff", len(cross_diff))
    out.fact("n_compared", len(cross_diff) + len(seed_pairs))
    out.fact("n_nets", len(nets))
    meta_r = json.loads((xlib.DATA / tag / "meta.json").read_text())
    out.fact("rule", meta_r["modes"][RULE])          # same rule as the alignment study's rule 0 (I["rule_names"])
    out.fact("rule1", I["rule_names"]["1"])
    out.fact("lead_pairings", " and ".join(name(a, b) for a, b in dpairs[LEAD]))

    # rule 1 (perplexity 100), final step, k = K_LO
    R1 = [row(f"proj_k{K_LO}", a, b, fin, 1) for a, b in cross_diff]
    R1s = [row(f"proj_k{K_LO}", a, b, fin, 1) for a, b in seed_pairs]
    assert all(R1) and all(R1s), "rule-1 final-step summaries missing"
    put_range(f"r1.k{K_LO}.depth.cost", [r["cost"] for r in R1])
    put_range(f"r1.k{K_LO}.seed.cost", [r["cost"] for r in R1s])
    put_range(f"r1.k{K_LO}.depth.null05", [r["n05"] for r in R1])
    put_range(f"r1.k{K_LO}.depth.unres", [r["unres"] for r in R1], pct=True)
    out.fact("r1.n_missing_cross", len(I["missing_cross"]["1"]))

    # ------------------------------------------------------------------ chart helpers
    C_FLOOR = [P["mute"], P["fg"]]

    def cost_chart(dist, da, db, same_seed=False):
        prs = pairings(da, db, same_seed)
        ser = []
        for q_, (a, b) in enumerate(prs):
            ser.append(dict(name=name(a, b) + (" (same seed: shared initial draws)" if same_seed else ""),
                            y=series_over(dist, a, b, "cost"), color=P["sets"][q_], width=2.2,
                            band=[series_over(dist, a, b, "n05"), series_over(dist, a, b, "n95")]))
        a0, b0 = prs[0]
        ser.append(dict(name=f"seed floor ResNet-{da} (its two seeds)", y=series_over(dist, a0, b0, "floor_a"),
                        color=C_FLOOR[0], dash=True, width=1.6))
        if da != db:
            ser.append(dict(name=f"seed floor ResNet-{db} (its two seeds)", y=series_over(dist, a0, b0, "floor_b"),
                            color=C_FLOOR[1], dash=True, width=1.6))
        return xlib.lines(steps, ser, "training step", "normalised DTW cost", xscale="symlog",
                          yrange=xlib.nice_range([s["y"] for s in ser] + [s["band"][1] for s in ser if "band" in s],
                                                 step=0.1),
                          band_mode="each" if len(prs) <= 2 else "envelope",
                          band_label="band: block-permutation null of that pairing, 5th to 95th percentile")

    def unres_chart(da, db):
        ser = []
        for q_, (a, b) in enumerate(pairings(da, db)):
            ser.append(dict(name=f"{name(a, b)}, k = {K_LO}", y=series_over(f"proj_k{K_LO}", a, b, "unres"),
                            color=P["sets"][q_], width=2.2, dots=2.5))
            ser.append(dict(name=f"{name(a, b)}, k = {K_HI}", y=series_over(f"proj_k{K_HI}", a, b, "unres"),
                            color=P["sets"][q_], dash=True, width=1.6, dots=2.5))
        return xlib.lines(steps, ser, "training step", "share of the DTW path with an unresolved cut",
                          xscale="symlog", yrange=(0, 1))

    # ------------------------------------------------------------------ lead (two panels) and kept panel
    out.chart("lead", cost_chart(f"proj_k{K_LO}", *LEAD))
    out.chart("lead_unres", unres_chart(*LEAD))
    out.chart("k9", cost_chart(f"proj_k{K_HI}", *LEAD))
    # values quoted under the lead: final-step costs of the lead pairings
    for key, dist in ((f"k{K_LO}", f"proj_k{K_LO}"), (f"k{K_HI}", f"proj_k{K_HI}")):
        R = [row(dist, a, b, fin) for a, b in dpairs[LEAD]]
        out.fact(f"lead.{key}.cost", " and ".join(xlib.fmt(r["cost"], 3) for r in R))
        out.fact(f"lead.{key}.null05", " and ".join(xlib.fmt(r["n05"], 3) for r in R))
        out.fact(f"lead.{key}.floors", f"{xlib.fmt(R[0]['floor_a'], 3)} (ResNet-{LEAD[0]}) and "
                                       f"{xlib.fmt(R[0]['floor_b'], 3)} (ResNet-{LEAD[1]})")
        out.fact(f"lead.{key}.unres", " and ".join(xlib.fmt(r["unres"], 0, pct=True) for r in R))
        R0 = [row(dist, a, b, steps[0]) for a, b in dpairs[LEAD]]
        if key == f"k{K_LO}":
            out.fact(f"lead.{key}.step0.cost", " and ".join(xlib.fmt(r["cost"], 3) for r in R0))
            out.fact(f"lead.{key}.step0.null05", " and ".join(xlib.fmt(r["n05"], 3) for r in R0))

    # ------------------------------------------------------------------ App 4 (Reference charts)
    app4 = {}
    rule_txt = f"kernel rule {RULE} ({I['rule_names'][str(RULE)]})"
    cap_cost = ("<b>Plotted:</b> normalised DTW cost (y) against training step (x, symmetric-log) for {what}, "
                "projector distance $d_{{{k}}}$, " + rule_txt + ". Solid: the different-seed pairings, each with "
                "its block-permutation null (band, 5th to 95th percentile, 1,000 permutations). Dashed: the seed "
                "floors (DTW cost between the two seeds of one depth at the same checkpoint).")
    for da, db in [dp for dp in DEPTH_PAIRS if dp != LEAD]:
        for k in (K_LO, K_HI):
            app4[f"cost_{da}_{db}_k{k}"] = dict(cost_chart(f"proj_k{k}", da, db),
                                                caption=cap_cost.format(what=f"ResNet-{da} against ResNet-{db}", k=k))
    for da, db in [dp for dp in DEPTH_PAIRS if dp != LEAD]:
        app4[f"unres_{da}_{db}"] = dict(unres_chart(da, db), caption=(
            f"<b>Plotted:</b> share of the DTW path whose cut is unresolved in either snapshot (y) against training "
            f"step (x, symmetric-log), ResNet-{da} against ResNet-{db}, different-seed pairings; solid $k = {K_LO}$, "
            f"dashed $k = {K_HI}$; {rule_txt}."))
    # the two seeds of one depth, each against its own null
    ser = []
    for q_, (a, b) in enumerate(seed_pairs):
        ser.append(dict(name=name(a, b), y=series_over(f"proj_k{K_LO}", a, b, "cost"), color=P["sets"][q_], width=2.0,
                        band=[series_over(f"proj_k{K_LO}", a, b, "n05"), series_over(f"proj_k{K_LO}", a, b, "n95")]))
    app4["seed_pairs_k3"] = dict(xlib.lines(steps, ser, "training step", "normalised DTW cost", xscale="symlog",
                                            yrange=xlib.nice_range([s["y"] for s in ser] + [s["band"][1] for s in ser], step=0.1),
                                            band_mode="each",
                                            band_label="band: block-permutation null of that pair, 5th to 95th percentile"),
                                 caption=f"<b>Plotted:</b> normalised DTW cost (y) between the two seeds of each depth "
                                         f"against training step (x, symmetric-log), projector distance $d_{{{K_LO}}}$, "
                                         f"{rule_txt}; bands: each pair's block-permutation null, 5th to 95th percentile. "
                                         "These costs are the seed floors of the depth comparisons.")
    # same-seed pairings of different depth (shared initial draws)
    app4["same_seed_20_56_k3"] = dict(cost_chart(f"proj_k{K_LO}", *LEAD, same_seed=True), caption=(
        f"<b>Plotted:</b> as the lead figure (ResNet-{LEAD[0]} against ResNet-{LEAD[1]}, $d_{{{K_LO}}}$, {rule_txt}) "
        "for the two same-seed pairings, whose networks draw the initial weights of the stem and first blocks "
        "identically; not used in any summary."))
    # cut-free distance and the CKA baseline over training
    for dist, lab in (("heat_t16", "diffusion-kernel distance $h_{16}$ (no cut)"),
                      ("cka", "CKA distance on the centred-cosine Grams (baseline, not an output of the method)")):
        app4[f"cost_{LEAD[0]}_{LEAD[1]}_{dist}"] = dict(cost_chart(dist, *LEAD), caption=(
            f"<b>Plotted:</b> as the lead figure (ResNet-{LEAD[0]} against ResNet-{LEAD[1]}, different-seed "
            f"pairings, null bands, seed floors dashed) with the {lab}" +
            ("" if dist == "cka" else f", {rule_txt}") + "."))
    # sensitivity to the cut at the final step: medians over the six cross-depth pairings
    kg = I["k_grid"]
    med = lambda f, prs: [float(np.median([row(f"proj_k{k}", a, b, fin)[f] for a, b in prs])) for k in kg]
    fl = [float(np.median([max(row(f"proj_k{k}", a, b, fin)["floor_a"], row(f"proj_k{k}", a, b, fin)["floor_b"])
                           for a, b in cross_diff])) for k in kg]
    chance = [1 - k / (I["n"] - 1) for k in kg]
    ser = [dict(name="DTW cost, cross-depth different-seed pairings (median)", y=med("cost", cross_diff), color=P["sets"][0], width=2.2),
           dict(name="null 5th percentile (median)", y=med("n05", cross_diff), color=P["sets"][0], dash=True, width=1.6),
           dict(name="larger seed floor of the two depths (median)", y=fl, color=P["mute"], dash=True, width=1.6),
           dict(name="share of the path with an unresolved cut (median)", y=med("unres", cross_diff), color=P["sets"][1], width=2.0),
           dict(name="chance 1 − k/(N−1) for one pair of snapshots", y=chance, color=P["mute"], dash="dot", width=1.2)]
    app4["by_k_final"] = dict(xlib.lines(kg, ser, "cut k (projector distance d_k)", "value", xscale="log", yrange=(0, 1)),
                              caption=f"<b>Plotted:</b> at the final checkpoint (step {fin:,}), {rule_txt}, against the cut "
                                      f"$k$ of the projector distance (x, log): the median over the {len(cross_diff)} "
                                      "cross-depth different-seed pairings of the normalised DTW cost (solid blue), of "
                                      "the null's 5th percentile (dashed blue), of the larger of the two seed floors "
                                      "(dashed grey) and of the share of the path whose cut is unresolved (orange); "
                                      "dotted: the chance value of $d_k$ for two independent random spans.")
    # seed-0 vs seed-1 set agreement per site (ResNet-20, final checkpoint, single-snapshot blocks)
    c_dir = xlib.DATA / "r20s0c"
    meta_b = json.loads((xlib.DATA / tag / "meta.json").read_text())
    meta_c = json.loads((c_dir / "meta_c.json").read_text())
    fld = {f_: i for i, f_ in enumerate(meta_c["agree_fields"])}
    with np.load(c_dir / f"sources_layer_{fin}.npz") as z:
        A = z["agree_I3"]
    lp = [tuple(p) for p in meta_b["layer_pairs"]]
    sites = meta_b["sites"]
    ser = []
    for q_, r in enumerate((3, 4)):
        ri = meta_c["seba_r"].index(r)
        pi = [lp.index((s, s)) for s in range(len(sites))]
        ser.append(dict(name=f"matched Jaccard, r = {r}", y=[float(A[p, ri, fld["mj_max"]]) for p in pi], color=P["sets"][q_], width=2.0, dots=3))
        ser.append(dict(name=f"permutation null 95th percentile, r = {r}", y=[float(A[p, ri, fld["mj_null_q95"]]) for p in pi],
                        color=P["sets"][q_], dash=True, width=1.4))
    app4["seed_sets_r20"] = dict(xlib.lines(list(range(len(sites))), ser, "site", "matched Jaccard", yrange=(0, 1),
                                            xticklabels=sites),
                                 caption=f"<b>Plotted:</b> ResNet-20, final checkpoint, {rule_txt}: matched Jaccard (y) "
                                         "between the SEBA sets of seed 0 and of seed 1 at each site's single-snapshot "
                                         "block (x), at $r = 3$ and $r = 4$ (solid), with the 95th percentile of its "
                                         f"size-preserving permutation null ({meta_c['n_perm']:,} permutations; dashed).")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
