"""X7 x-markov: is the constructed walk through the layers approximately Markov under the tested partitions?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/markov.py r20s0b
    python3 site/check_fragment.py markov

Reads data/<tag>/markov_summary.json (markov_summary.py; SPEC_COARSE.md section 6), data/<tag>/coarse_summary.json
(spectral gap reports, for the choice of k) and data/<tag>/meta.json. Writes assets/x/markov.json (chart objects) and
assets/x/markov.facts.json ({{fact:x.markov.<name>}}). No new computation: every value is a re-plot or a median of
exported values. Exported -1 entries are undefined (zero denominator, or a partition that was not scored, e.g. the
classes at k != 10) and become gaps, never values.

Array layout of markov_summary.json, per unit "<step>_<rule>":
    defect_frac[world][method][k][pair s]   pair s = (site s, site s+1), 11 pairs
    mem_frac[world][method][k][triple s]    triple s = (s, s+1, s+2), 10 triples
    ck_by_h[h] = [ck[world][method][k], ck_ind[world][method][k]]   mean over start sites a with b - a = h
    nmi[method][k][pair s]                  normalised mutual information of neighbouring partitions
worlds = [real, uniform]; methods = [spectral, pca, shared, random, labels].
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

RULE = 0          # the default kernel rule (kNN graph, k = 15) for every figure; other rules only as a range
K_MAIN, K_ALT = 10, 4


def main(tag: str):
    src = xlib.DATA / tag
    ms = json.loads((src / "markov_summary.json").read_text())
    meta = json.loads((src / "meta.json").read_text())
    cs = json.loads((src / "coarse_summary.json").read_text())
    steps, sites, ks, methods = ms["steps"], ms["sites"], ms["k"], ms["methods"]
    assert sites == xlib.SITES and ms["worlds"] == ["real", "uniform"], "unexpected markov_summary layout"
    M = {m: i for i, m in enumerate(methods)}
    fin = steps[-1]
    rules = sorted({int(u.split("_")[1]) for u in ms["units"] if u.startswith(f"{fin}_")})
    unit = lambda step, rule: ms["units"][f"{step}_{rule}"]
    arr = lambda step, rule, key: xlib.missing(unit(step, rule)[key])          # -1 -> NaN
    ki = ks.index(K_MAIN)
    P = xlib.PALETTE
    C_OWN, C_SHARED, C_PCA, C_RAND, C_LAB = P["sets"][0], P["sets"][1], P["sets"][2], P["mute"], P["fg"]
    pairs = xlib.pair_labels(sites)
    out = xlib.Out("markov")

    # ------------------------------------------------------------------ lead: defect fraction per adjacent pair
    D = arr(fin, RULE, "defect_frac")          # (world, method, k, pair)
    own, own_u = D[0, M["spectral"], ki], D[1, M["spectral"], ki]
    sh, sh_u = D[0, M["shared"], ki], D[1, M["shared"], ki]
    lab = D[0, M["labels"], ki]
    x = list(range(len(pairs)))
    out.chart("lead", xlib.lines(
        x, [
            dict(name="each site's own partition, constructed walk", y=own, color=C_OWN, width=2.2),
            dict(name="each site's own partition, lazy uniform walk (control)", y=own_u, color=C_OWN, dash=True, width=1.6),
            dict(name="b9's partition at every site, constructed walk", y=sh, color=C_SHARED, width=2.2),
            dict(name="b9's partition at every site, lazy uniform walk (control)", y=sh_u, color=C_SHARED, dash=True, width=1.6),
            dict(name="classes at every site, constructed walk (validation)", y=lab, color=C_LAB, dots=4, noline=True),
        ],
        "adjacent pair of sites (s → s+1)", "defect fraction δₛ / I(Zₛ; Cₛ₊₁)", yrange=(0, 1),
        xticklabels=pairs))

    # ------------------------------------------------------------------ kept: Chapman-Kolmogorov error vs horizon
    H = sorted(int(h) for h in unit(fin, RULE)["ck_by_h"])
    ck = np.array([xlib.missing(unit(fin, RULE)["ck_by_h"][str(h)], np.nan) for h in H])   # (h, 2, world, method, k)
    ck_own, ind_own, ck_own_u = ck[:, 0, 0, M["spectral"], ki], ck[:, 1, 0, M["spectral"], ki], ck[:, 0, 1, M["spectral"], ki]
    ck_sh, ind_sh = ck[:, 0, 0, M["shared"], ki], ck[:, 1, 0, M["shared"], ki]
    out.chart("ck", xlib.lines(
        H, [
            dict(name="own partitions: CK error", y=ck_own, color=C_OWN, width=2.2),
            dict(name="own partitions: distance from independence", y=ind_own, color=C_OWN, dash="dot", width=1.6),
            dict(name="own partitions, lazy uniform walk (control): CK error", y=ck_own_u, color=C_OWN, dash=True, width=1.6),
            dict(name="b9's partition: CK error", y=ck_sh, color=C_SHARED, width=2.2),
            dict(name="b9's partition: distance from independence", y=ind_sh, color=C_SHARED, dash="dot", width=1.6),
        ],
        "horizon h = b − a (sites)", "mass-weighted total variation", yrange=xlib.nice_range([ck_own, ind_own, ck_sh, ind_sh], step=0.05)))

    # The text says the control is exactly zero for a partition shared by every site (by construction: on the lazy
    # uniform walk the next block's distribution depends only on the current block). Check it rather than quote it.
    Mem = arr(fin, RULE, "mem_frac")
    for what, v in (("defect, shared", sh_u), ("defect, classes", D[1, M["labels"], ki]),
                     ("CK error, shared", ck[:, 0, 1, M["shared"], ki]), ("memory, shared", Mem[1, M["shared"], ki])):
        if np.nanmax(np.abs(v)) > 1e-4:
            raise SystemExit(f"control not zero for {what}: the 'exactly zero by construction' sentence is wrong")

    # ------------------------------------------------------------------ facts (rule 0, final step, k = 10 unless named)
    med = lambda v: float(np.nanmedian(v))
    out.fact("rule", meta["modes"][RULE])
    out.fact("n_pairs", len(pairs))
    out.fact("own.real.defect", med(own), 2)
    out.fact("own.real.defect_min", np.nanmin(own), 2)
    out.fact("own.real.defect_max", np.nanmax(own), 2)
    out.fact("own.uniform.defect", med(own_u), 2)
    out.fact("own.uniform.defect_min", np.nanmin(own_u), 2)
    out.fact("own.uniform.defect_max", np.nanmax(own_u), 2)
    out.fact("own.real.defect_last2", f"{xlib.fmt(own[-2], 2)} and {xlib.fmt(own[-1], 2)}")
    out.fact("own.uniform.defect_last2", f"{xlib.fmt(own_u[-2], 2)} and {xlib.fmt(own_u[-1], 2)}")
    out.fact("shared.real.defect", med(sh), 2)
    out.fact("shared.real.defect_min", np.nanmin(sh), 2)
    out.fact("shared.real.defect_max", np.nanmax(sh), 2)
    out.fact("labels.real.defect", med(lab), 2)
    out.fact("own.nmi", med(np.array(unit(fin, RULE)["nmi"], float)[M["spectral"], ki]), 2)
    for h in (2, H[-1]):
        j = H.index(h)
        out.fact(f"h{h}.own.real.ck", ck_own[j], 3)
        out.fact(f"h{h}.own.real.ckind", ind_own[j], 3)
        out.fact(f"h{h}.own.uniform.ck", ck_own_u[j], 3)
        out.fact(f"h{h}.shared.real.ck", ck_sh[j], 3)
        out.fact(f"h{h}.shared.real.ckind", ind_sh[j], 3)
    out.fact("h_max", H[-1])
    # Unnormalised one-step defect and next-block information in nats, from the raw per-unit archive
    # (markov_<step>_<rule>.npz, step[world, method, k, pair, field]; field 0 = delta_s = I(Z_s; C_{s+1} | C_s),
    # field 1 = I(Z_s; C_{s+1})). The plotted fractions have different denominators on the two walks, so the text
    # compares amounts as well as fractions. Checked against the exported fractions.
    st = np.load(src / "coarse" / f"markov_{fin}_{RULE}.npz")["step"]
    for w in (0, 1):
        for m in ("spectral", "shared"):
            num, den = st[w, M[m], ki, :, 0], st[w, M[m], ki, :, 1]
            frac = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1), np.nan)
            if not np.allclose(frac, D[w, M[m], ki], atol=1e-4, equal_nan=True):
                raise SystemExit(f"markov npz and markov_summary.json disagree ({m}, world {w})")
    out.fact("own.real.defect_nats", med(st[0, M["spectral"], ki, :, 0]), 2)
    out.fact("own.uniform.defect_nats", med(st[1, M["spectral"], ki, :, 0]), 2)
    out.fact("own.real.info_nats", med(st[0, M["spectral"], ki, :, 1]), 2)
    out.fact("own.uniform.info_nats", med(st[1, M["spectral"], ki, :, 1]), 2)
    out.fact("shared.real.defect_nats", med(st[0, M["shared"], ki, :, 0]), 2)
    # coarse memory (text only; its charts go to App 4)
    out.fact("own.real.mem", med(Mem[0, M["spectral"], ki]), 3)
    out.fact("own.uniform.mem", med(Mem[1, M["spectral"], ki]), 3)
    out.fact("shared.real.mem", med(Mem[0, M["shared"], ki]), 3)
    # across kernel rules: range over rules of the median over pairs
    by_rule = lambda w, m: [med(arr(fin, r, "defect_frac")[w, M[m], ki]) for r in rules]
    out.fact("n_rules", len(rules))
    for w, wn in ((0, "real"), (1, "uniform")):
        v = by_rule(w, "spectral")
        out.fact(f"rules.own.{wn}.defect_lo", min(v), 2)
        out.fact(f"rules.own.{wn}.defect_hi", max(v), 2)
    v = by_rule(0, "shared")
    out.fact("rules.shared.real.defect_lo", min(v), 2)
    out.fact("rules.shared.real.defect_hi", max(v), 2)
    # k = 4
    k4 = ks.index(K_ALT)
    out.fact("k4.own.real.defect", med(D[0, M["spectral"], k4]), 2)
    out.fact("k4.own.uniform.defect", med(D[1, M["spectral"], k4]), 2)
    out.fact("k4.shared.real.defect", med(D[0, M["shared"], k4]), 2)
    # choice of k: chains with a clear spectral gap after K_MAIN modes (coarse_summary.block_gap; rule 0, final step)
    gap_chains = [f"{sites[b['a']]}{xlib.ARROW}{sites[b['b']]}" for b in cs["units"][f"{fin}_{RULE}"]
                  if b["a"] < b["b"] and b["gap"]["clear"] and b["gap"]["k"] == K_MAIN]
    if not gap_chains:
        raise SystemExit(f"no chain has a clear gap after {K_MAIN}: the text's justification of k does not hold")
    out.fact("gap_chains", ", ".join(gap_chains[:-1]) + (" and " if len(gap_chains) > 1 else "") + gap_chains[-1])

    # ------------------------------------------------------------------ to App 4 (Reference charts): chart + caption
    app4 = {}
    kx = ks
    medk = lambda w, m: [med(D[w, M[m], q]) for q in range(len(ks))]
    app4["defect_by_k"] = dict(xlib.lines(
        kx, [dict(name="own partitions (spectral k-means)", y=medk(0, "spectral"), color=C_OWN),
             dict(name="own partitions, control", y=medk(1, "spectral"), color=C_OWN, dash=True),
             dict(name="PCA k-means per site", y=medk(0, "pca"), color=C_PCA),
             dict(name="PCA k-means per site, control", y=medk(1, "pca"), color=C_PCA, dash=True),
             dict(name="b9's partition at every site", y=medk(0, "shared"), color=C_SHARED),
             dict(name="independent random partitions", y=medk(0, "random"), color=C_RAND)],
        "number of blocks k", "defect fraction (median over adjacent pairs)", yrange=(0, 1)),
        caption=f"<b>Plotted:</b> median over the {len(pairs)} adjacent pairs of the one-step defect fraction "
                f"against the number of blocks k, final checkpoint, kernel rule {meta['modes'][RULE]}; solid: "
                "constructed walk, dashed: lazy uniform walk with the same partitions (control). The b9 and random "
                "controls are omitted (b9's is exactly zero; random's equals its real value to two decimals).")
    Mk = arr(fin, RULE, "mem_frac")
    memk = lambda w, m: [med(Mk[w, M[m], q]) for q in range(len(ks))]
    app4["mem_by_k"] = dict(xlib.lines(
        kx, [dict(name="own partitions", y=memk(0, "spectral"), color=C_OWN),
             dict(name="own partitions, control", y=memk(1, "spectral"), color=C_OWN, dash=True),
             dict(name="PCA k-means per site", y=memk(0, "pca"), color=C_PCA),
             dict(name="b9's partition at every site", y=memk(0, "shared"), color=C_SHARED),
             dict(name="independent random partitions", y=memk(0, "random"), color=C_RAND)],
        "number of blocks k", "memory fraction (median over triples)", yrange=xlib.nice_range([memk(0, "spectral"), memk(0, "pca")], step=0.01)),
        caption=f"<b>Plotted:</b> median over the {len(pairs) - 1} adjacent triples (s, s+1, s+2) of "
                "I(C_{s+2}; C_s | C_{s+1}) / H(C_{s+2} | C_{s+1}) against k, final checkpoint, kernel rule "
                f"{meta['modes'][RULE]}; dashed: lazy uniform walk with the same partitions.")
    memt = lambda w, m: [med(arr(st, RULE, "mem_frac")[w, M[m], ki]) for st in steps]
    deft = lambda w, m: [med(arr(st, RULE, "defect_frac")[w, M[m], ki]) for st in steps]
    for key, fn, yl in (("mem_over_training", memt, "memory fraction (median over triples)"),
                        ("defect_over_training", deft, "defect fraction (median over adjacent pairs)")):
        ser = [dict(name="own partitions", y=fn(0, "spectral"), color=C_OWN),
               dict(name="own partitions, control", y=fn(1, "spectral"), color=C_OWN, dash=True),
               dict(name="b9's partition at every site", y=fn(0, "shared"), color=C_SHARED),
               dict(name="classes at every site (validation)", y=fn(0, "labels"), color=C_LAB, dots=3, noline=True)]
        app4[key] = dict(xlib.lines(steps, ser, "training step", yl, xscale="symlog",
                                    yrange=xlib.nice_range([s["y"] for s in ser], step=0.01)),
                         caption=f"<b>Plotted:</b> {yl} at k = {K_MAIN} against training step (symlog axis), kernel "
                                 f"rule {meta['modes'][RULE]}; dashed: lazy uniform walk with the same partitions; "
                                 "b9's partition is b9's own spectral k-means at each checkpoint.")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
