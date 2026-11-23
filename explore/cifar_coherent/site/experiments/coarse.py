"""X6 x-coarse: how much of each layer's walk can k groups keep, and does the spectrum prefer a k?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/coarse.py r20s0b
    python3 site/check_fragment.py coarse

Reads data/<tag>/coarse_summary.json (coarse_summary.py; definitions in SPEC_COARSE.md sections 1-2) and
data/<tag>/meta.json. Writes assets/x/coarse.json (chart objects) and assets/x/coarse.facts.json
({{fact:x.coarse.<name>}}). No new computation beyond ratios, medians and ceilings recomputed from the exported
singular values: every value is a re-plot of exported scores.

Layout of coarse_summary.json, per unit "<step>_<rule>": a list of 78 blocks {a, b, ...}; a == b is a single snapshot
(cell, Q = M), a < b the layer chain Q = M_b ... M_a. Per block: total = I(X;Y) (chi^2), sv = singular values
sigma_1 = 1, sigma_2, ... (65 stored), gap = coarse_summary.block_gap (1 - sigma scale for cells,
sigma_k / sigma_{k+1} for chains; "clear" = largest ratio >= 1.15 and > 1.1 x runner-up; gap after k modes,
constant included), and per method {spectral, seba, pca, random, labels, pred} the scores I_start = I(c(X);Y),
I_end = I(X;c'(Y)), I_two = I(c(X);c'(Y)), keff_start, keff_end, each a list over K = [2..16, 20, 24, 32] (None where a
method was not scored at that k: SEBA only at r in {2,3,4,6,8,10,12,16}, classes and predictions only at 10).

Conventions here (EXPERIMENTS_PLAN.md X6): single snapshots use the one-sided I_start (cells are symmetric, so
I_start = I_end); chains use the two-sided I_two. Every method is drawn at its k_eff (the number of nonempty groups;
for chains min(keff_start, keff_end)), and ceilings are F_{k_eff} = sigma_2^2 + ... + sigma_{k_eff}^2.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

RULE = 0                  # the default kernel rule (kNN graph, k = 15); other rules only in caveats and App D
K_MAIN = 10
CELL_SITE = "b9"          # worked examples named in the plan (EXPERIMENTS_PLAN.md X6)
CHAIN = ("b9", "logits")


def up_to(vals, step):
    """[0, hi]: a y range from zero to just above the largest finite value (shares and counts are never negative)."""
    v = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in vals])
    v = v[np.isfinite(v)]
    return [0, float(np.ceil(1.06 * v.max() / step) * step)]


def two_sided(Q_or_alpha, la, lb, N):
    """I(c(X); c'(Y)) = sum_{j,l} T_lj^2 / (|A_j| |B_l|) - 1 with T_lj = sum_{x in A_j, y in B_l} Q[y, x] (SPEC_COARSE.md 1).
    Q_or_alpha: a dense Q, or the scalar alpha^L of the lazy uniform chain U^L = alpha^L I + (1 - alpha^L) J / N."""
    ka, kb = la.max() + 1, lb.max() + 1
    na = np.bincount(la, minlength=ka).astype(float)
    nb = np.bincount(lb, minlength=kb).astype(float)
    if np.ndim(Q_or_alpha) == 0:
        C = np.zeros((kb, ka))
        np.add.at(C, (lb, la), 1.0)
        T = Q_or_alpha * C + (1 - Q_or_alpha) * np.outer(nb, na) / N
    else:
        T = np.eye(kb)[lb].T @ Q_or_alpha @ np.eye(ka)[la]
    m = np.outer(nb, na) > 0
    return float((T[m] ** 2 / np.outer(nb, na)[m]).sum() - 1)


def ceiling(sv, k):
    return float((np.asarray(sv[1:int(k)], float) ** 2).sum())


def main(tag: str):
    src = xlib.DATA / tag
    cs = json.loads((src / "coarse_summary.json").read_text())
    meta = json.loads((src / "meta.json").read_text())
    K, steps, sites = cs["K"], cs["steps"], cs["sites"]
    assert sites == xlib.SITES and not cs["missing"], "unexpected coarse_summary layout"
    fin = steps[-1]
    rules = sorted({int(u.split("_")[1]) for u in cs["units"] if u.startswith(f"{fin}_")})
    N = int(meta["n"])
    ki = K.index(K_MAIN)
    SI = {s: i for i, s in enumerate(sites)}
    P = xlib.PALETTE
    C_SPEC, C_SEBA, C_PCA, C_RAND, C_CEIL = P["sets"][0], P["sets"][1], P["sets"][2], P["mute"], P["fg"]
    C_LAB, C_PRED = P["fg"], P["sets"][4]
    out = xlib.Out("coarse")

    def blk(step, rule, a, b):
        return next(x for x in cs["units"][f"{step}_{rule}"] if x["a"] == a and x["b"] == b)

    def score(x, m, cell):
        """Retained information of method m at each k, with its k_eff. Cells: one-sided; chains: two-sided."""
        s = "I_start" if cell else "I_two"
        I = [np.nan if v is None else v for v in x[m][s]]
        ks_ = x[m]["keff_start"] if cell else [None if a is None or b is None else min(a, b)
                                               for a, b in zip(x[m]["keff_start"], x[m]["keff_end"])]
        return np.array(I, float), [None if k is None else int(k) for k in ks_]

    def at_keff(x, m, cell, frac):
        """Series over K with each value placed at its own k_eff (as a fraction of the total if frac)."""
        I, ke = score(x, m, cell)
        y = [np.nan] * len(K)
        for v, k in zip(I, ke):
            if k is None or not np.isfinite(v):
                continue
            if k not in K:
                raise SystemExit(f"k_eff {k} of {m} is not on the k axis")
            j = K.index(k)
            if np.isfinite(y[j]):
                raise SystemExit(f"two values of {m} share k_eff {k}: decide how to draw them")
            y[j] = v / x["total"] if frac else v
        return np.array(y, float)

    def eta(x, m, cell, k=K_MAIN):
        """I / F_{k_eff} at the value scored for k (tightness)."""
        I, ke = score(x, m, cell)
        j = K.index(k)
        return I[j] / ceiling(x["sv"], ke[j]) if ke[j] is not None and np.isfinite(I[j]) else np.nan

    pct = lambda v: 100 * np.asarray(v, float)

    # ------------------------------------------------------------------ lead: retained information vs k, % of total
    # Reference for chains: the same spectral k-means partitions (stored labels, k = 2..16) scored on the lazy uniform
    # chain U^L, L = b - a + 1 factors (coarse.uniform_lazy: U = (I + (J - I)/(N - 1))/2, no geometry). It keeps the
    # stay-put path and the overlap of the two end partitions, and nothing else.
    labz = np.load(src / "coarse" / f"coarse_{fin}_{RULE}.npz")
    PAIRS = [(a, b) for a in range(len(sites)) for b in range(a, len(sites))]
    KL = list(range(2, 17))                      # k of the stored labels
    alpha = 0.5 - 0.5 / (N - 1)                  # U = alpha I + (1 - alpha) J / N

    def unif_chain(a, b, k):
        bi = PAIRS.index((a, b))
        la, lb = labz["lab_start"][bi, KL.index(k)].astype(int), labz["lab_end"][bi, KL.index(k)].astype(int)
        if la.min() < 0 or lb.min() < 0:
            raise SystemExit("unassigned spectral k-means label")
        return two_sided(alpha ** (b - a + 1), la, lb, N), la, lb

    # check the closed form of U^L against a dense power once (b9 -> logits, k = 10)
    a0, b0 = SI[CHAIN[0]], SI[CHAIN[1]]
    U = np.full((N, N), 0.5 / (N - 1)); np.fill_diagonal(U, 0.5)
    v, la, lb = unif_chain(a0, b0, K_MAIN)
    dense = two_sided(np.linalg.matrix_power(U, b0 - a0 + 1), la, lb, N)
    if abs(v - dense) > 1e-8 * max(1.0, dense):
        raise SystemExit(f"uniform-chain closed form {v} disagrees with the dense power {dense}")
    del U

    def lead(x, cell, name):
        T = x["total"]
        ceil = [ceiling(x["sv"], k) / T for k in K]
        extra = [] if cell else [dict(
            name="same spectral partitions on the lazy uniform chain (reference)",
            y=pct([unif_chain(x["a"], x["b"], k)[0] / T if k in KL else np.nan for k in K]),
            color=C_SPEC, dash=True, width=1.6)]
        ser = extra + [dict(name="ceiling F_k (no k-block partition keeps more)", y=pct(ceil), color=C_CEIL, dash="dot", width=1.6),
               dict(name="spectral k-means", y=pct(at_keff(x, "spectral", cell, True)), color=C_SPEC, width=2.2),
               dict(name="random balanced partitions (null)" if cell else "independent balanced endpoint partitions (null)",
                    y=pct(at_keff(x, "random", cell, True)), color=C_RAND, dash=True, width=1.6),
               dict(name="SEBA, r = 2, 3, 4, 6, 8, 10, 12, 16", y=pct(at_keff(x, "seba", cell, True)), color=C_SEBA, dots=4, noline=True),
               dict(name="classes (validation)", y=pct(at_keff(x, "labels", cell, True)), color=C_LAB, dots=4.5, noline=True),
               dict(name="predictions (validation)", y=pct(at_keff(x, "pred", cell, True)), color=C_PRED, dots=3, noline=True)]
        side = "one-sided I(c(X); Y)" if cell else "two-sided I(c(X); c′(Y))"
        out.chart(name, xlib.lines(K, ser, "number of groups k", f"{side}, % of total I(X;Y)",
                                   yrange=up_to([s["y"] for s in ser], step=0.5)))
        return ser

    xc = blk(fin, RULE, SI[CELL_SITE], SI[CELL_SITE])
    xh = blk(fin, RULE, SI[CHAIN[0]], SI[CHAIN[1]])
    lead(xc, True, "lead_cell")
    lead(xh, False, "lead_chain")

    # facts for the answer and captions
    out.fact("rule", meta["modes"][RULE])
    for nm, x, cell in (("cell", xc, True), ("chain", xh, False)):
        T = x["total"]
        I, ke = score(x, "spectral", cell)
        out.fact(f"{nm}.total", T, 1 if T >= 10 else 3)
        out.fact(f"{nm}.ceil", ceiling(x["sv"], ke[ki]), 2)
        out.fact(f"{nm}.spec", I[ki], 2)
        out.fact(f"{nm}.ceil_pct", ceiling(x["sv"], ke[ki]) / T, 1, pct=True)
        out.fact(f"{nm}.spec_pct", I[ki] / T, 1, pct=True)
        for m in ("labels", "pred"):
            v, k = score(x, m, cell)
            if ke[ki] != k[ki]:
                raise SystemExit(f"{m} k_eff differs from {K_MAIN} at the final step: the caption's wording is wrong")
            out.fact(f"{nm}.{m}_pct", v[ki] / T, 1, pct=True)
        r, _ = score(x, "random", cell)
        if cell:
            out.fact(f"{nm}.rand_pct", r[ki] / T, 2, pct=True)
        else:
            out.fact(f"{nm}.rand", r[ki], 4)
            out.fact(f"{nm}.unif", unif_chain(x["a"], x["b"], K_MAIN)[0], 3)
        out.fact(f"{nm}.ceil32_pct", ceiling(x["sv"], 32) / T, 1, pct=True)

    shares = []
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            real = score(blk(fin, RULE, a, b), "spectral", False)[0][ki]
            shares.append((unif_chain(a, b, K_MAIN)[0] / real, b - a))
    if max(shares)[1] != 1:
        raise SystemExit("largest uniform-chain share is not at adjacent sites: rewrite the text")
    shares = [v for v, _ in shares]
    out.fact("unif.share_max", max(shares), 1, pct=True)
    out.fact("unif.share_med", float(np.median(shares)), 1, pct=True)

    # single snapshots at every site (final step): totals, ceiling, spectral share
    cells = [blk(fin, RULE, i, i) for i in range(len(sites))]
    floor = N / 4 - 1
    all_cells = [b for u in cs["units"].values() for b in u if b["a"] == b["b"]]
    if min(b["total"] for b in all_cells) < floor - 1e-3:
        raise SystemExit("a single snapshot has I(X;Y) < N/4 - 1: the lazy-half sentence is wrong")
    out.fact("lazy_floor", int(round(floor)))
    out.fact("cells.total_min", min(b["total"] for b in cells), 0)
    out.fact("cells.total_max", max(b["total"] for b in cells), 0)
    c10 = [ceiling(b["sv"], K_MAIN) / b["total"] for b in cells]
    c32 = [ceiling(b["sv"], 32) / b["total"] for b in cells]
    out.fact("cells.ceil_pct_min", min(c10), 1, pct=True)
    out.fact("cells.ceil_pct_max", max(c10), 1, pct=True)
    out.fact("cells.ceil32_pct_max", max(c32), 1, pct=True)
    # without the stay-put half: the same partitions scored on P = 2M - I (stored I_start_P, total_P, F_P)
    out.fact("cell.p.ceil_pct", xc["F_P"][ki] / xc["total_P"], 1, pct=True)
    out.fact("cell.p.spec_pct", xc["spectral"]["I_start_P"][ki] / xc["total_P"], 1, pct=True)
    out.fact("cell.p.rand_pct", xc["random"]["I_start_P"][ki] / xc["total_P"], 1, pct=True)

    # chains by length: ceiling share for adjacent pairs vs stem -> logits
    adj = [blk(fin, RULE, a, a + 1) for a in range(len(sites) - 1)]
    full = blk(fin, RULE, 0, len(sites) - 1)
    out.fact("adj.ceil_pct", float(np.median([ceiling(b["sv"], K_MAIN) / b["total"] for b in adj])), 1, pct=True)
    out.fact("adj.total", float(np.median([b["total"] for b in adj])), 0)
    out.fact("full.ceil_pct", ceiling(full["sv"], K_MAIN) / full["total"], 0, pct=True)
    out.fact("full.total", full["total"], 2)

    # ------------------------------------------------------------------ kept: tightness I / F_{k_eff} per site, k = 10
    ser = []
    for m, nm_, col, kw in (("spectral", "spectral k-means", C_SPEC, dict(width=2.2)),
                            ("seba", "SEBA, r = 10", C_SEBA, dict(width=1.6)),
                            ("pca", "PCA k-means (baseline, no operator)", C_PCA, dict(width=1.6)),
                            ("random", "random balanced partitions (null)", C_RAND, dict(dash=True, width=1.6)),
                            ("labels", "classes (validation)", C_LAB, dict(dots=4.5, noline=True)),
                            ("pred", "predictions (validation)", C_PRED, dict(dots=3, noline=True))):
        ser.append(dict(name=nm_, y=[eta(b, m, True) for b in cells], color=col, **kw))
    out.chart("tight", xlib.lines(list(range(len(sites))), ser, "site (single snapshot)",
                                  "I(c(X); Y) / F_k_eff", yrange=(0, 1), xticklabels=sites))
    E = {s["name"]: np.array(s["y"], float) for s in ser}
    out.fact("tight.spec_min", E["spectral k-means"].min(), 2)
    out.fact("tight.spec_max", E["spectral k-means"].max(), 2)
    out.fact("tight.rand_min", E["random balanced partitions (null)"].min(), 2)
    out.fact("tight.rand_max", E["random balanced partitions (null)"].max(), 2)

    # ------------------------------------------------------------------ gap position: cells (site x step), chains (a x b, step slider)
    def gap_k(x):
        return x["gap"]["k"] if x["gap"]["clear"] else None

    zc = [[gap_k(blk(st, RULE, i, i)) for st in steps] for i in range(len(sites))]
    zch = [[[gap_k(blk(st, RULE, a, b)) if a < b else None for b in range(1, len(sites))]
            for a in range(len(sites) - 1)] for st in steps]
    vals = [v for row in zc for v in row if v] + [v for t in zch for row in t for v in row if v]
    zr = [0, max(vals)]          # from 0, so that k = 2 is distinguishable from the grey "no clear gap"
    out.chart("gap_cells", xlib.heat(zc, sites, steps, "number of groups k before the clear gap (grey: no clear gap)",
                                     "site", "training step", zrange=zr, fmt="int"))
    out.chart("gap_chains", xlib.heat(zch, sites[:-1], sites[1:],
                                      "number of groups k before the clear gap (grey: no clear gap or a ≥ b)",
                                      "start site a", "end site b", zrange=zr, fmt="int",
                                      slider=dict(label="training step", values=[f"{s:,}" for s in steps])))

    fin_cells = cells
    n_c2 = sum(1 for b in fin_cells if gap_k(b) == 2)
    out.fact("gap.cells_n", len(fin_cells))
    out.fact("gap.cells_k2", n_c2)
    allc = [blk(st, RULE, i, i) for st in steps for i in range(len(sites))]
    out.fact("gap.cells_all_n", len(allc))
    out.fact("gap.cells_all_k2", sum(1 for b in allc if gap_k(b) == 2))
    out.fact("gap.cells_all_none", sum(1 for b in allc if gap_k(b) is None))
    out.fact("gap.cells_all_other", sum(1 for b in allc if gap_k(b) not in (2, None)))
    run10 = [sites[b["a"]] for b in fin_cells if b["gap"]["k2"] == K_MAIN]
    out.fact("gap.cells_runner10", ", ".join(run10[:-1]) + (" and " if len(run10) > 1 else "") + run10[-1] if run10 else "none")

    chains_fin = [b for b in cs["units"][f"{fin}_{RULE}"] if b["a"] < b["b"]]
    name = lambda b: f"{sites[b['a']]}{xlib.ARROW}{sites[b['b']]}"
    g10 = [b for b in chains_fin if gap_k(b) == K_MAIN]
    if not g10 or not any((sites[b["a"]], sites[b["b"]]) == CHAIN for b in g10):
        raise SystemExit("b9 -> logits is no longer among the chains with a clear gap after 10: rewrite the text")
    lst = [name(b) for b in g10]
    out.fact("gap.chains10", ", ".join(lst[:-1]) + (" and " if len(lst) > 1 else "") + lst[-1])
    out.fact("gap.chains10_n", len(g10))
    out.fact("gap.chains_n", len(chains_fin))
    out.fact("gap.chains_k2", sum(1 for b in chains_fin if gap_k(b) == 2))
    out.fact("gap.chains_none", sum(1 for b in chains_fin if gap_k(b) is None))
    ch10_steps = [st for st in steps if any(gap_k(b) == K_MAIN for b in cs["units"][f"{st}_{RULE}"] if b["a"] < b["b"])]
    out.fact("gap.chains10_steps", ", ".join(f"{s:,}" for s in ch10_steps))
    out.fact("gap.chain.s10", xh["sv"][K_MAIN - 1], 3)
    out.fact("gap.chain.s11", xh["sv"][K_MAIN], 3)
    out.fact("gap.chain.ratio", xh["gap"]["ratio"], 2)
    out.fact("gap.chain.ratio2", xh["gap"]["ratio2"], 2)
    out.fact("gap.chain.k2", xh["gap"]["k2"])
    # across rules (final step)
    per_rule = [sum(1 for b in cs["units"][f"{fin}_{r}"] if b["a"] < b["b"] and gap_k(b) == K_MAIN) for r in rules]
    out.fact("gap.rules_n", len(rules))
    out.fact("gap.rules.chains10_lo", min(per_rule))
    out.fact("gap.rules.chains10_hi", max(per_rule))
    out.fact("gap.rules.chain_clear", sum(1 for r in rules if gap_k(blk(fin, r, SI[CHAIN[0]], SI[CHAIN[1]])) == K_MAIN))
    out.fact("gap.rules.cells_k2_lo", min(sum(1 for i in range(len(sites)) if gap_k(blk(fin, r, i, i)) == 2) for r in rules))
    out.fact("gap.rules.cells_k2_hi", max(sum(1 for i in range(len(sites)) if gap_k(blk(fin, r, i, i)) == 2) for r in rules))

    # ------------------------------------------------------------------ kept: sigma_k / sigma_{k+1} for the chains in the list
    kr = list(range(2, 33))
    ser = []
    for j, b in enumerate(g10):
        sv = np.asarray(b["sv"], float)
        ratio = [sv[k - 1] / sv[k] if sv[k] > 1e-8 else np.nan for k in kr]
        if abs(np.nanmax(ratio) - b["gap"]["ratio"]) > 1e-3 * b["gap"]["ratio"]:
            raise SystemExit(f"ratio recomputed from sv disagrees with block_gap for {name(b)}")
        ser.append(dict(name=name(b), y=ratio, color=P["sets"][j % len(P["sets"])],
                        width=2.2 if (sites[b["a"]], sites[b["b"]]) == CHAIN else 1.6))
    out.chart("ratio", xlib.lines(kr, ser, "number of groups k (gap between σ_k and σ_k+1)", "σ_k / σ_k+1",
                                  yrange=[0.9, up_to([s["y"] for s in ser], step=0.05)[1]],
                                  refline=dict(y=1.15, label="1.15")))

    # ------------------------------------------------------------------ to App 4 (Reference charts): chart + caption
    app4 = {}
    rn = meta["modes"][RULE]
    for nm, x, cell, lab in (("eta_vs_k_cell", xc, True, f"single snapshot {CELL_SITE}, one-sided"),
                             ("eta_vs_k_chain", xh, False, f"chain {CHAIN[0]}{xlib.ARROW}{CHAIN[1]}, two-sided")):
        def eta_k(m):
            I, ke = score(x, m, cell)
            y = [np.nan] * len(K)
            for v, k in zip(I, ke):
                if k is not None and np.isfinite(v):
                    y[K.index(k)] = v / ceiling(x["sv"], k)
            return y
        ser = [dict(name="spectral k-means", y=eta_k("spectral"), color=C_SPEC),
               dict(name="PCA k-means (baseline)", y=eta_k("pca"), color=C_PCA),
               dict(name="random (null)", y=eta_k("random"), color=C_RAND, dash=True),
               dict(name="SEBA", y=eta_k("seba"), color=C_SEBA, dots=4, noline=True),
               dict(name="classes (validation)", y=eta_k("labels"), color=C_LAB, dots=4.5, noline=True),
               dict(name="predictions (validation)", y=eta_k("pred"), color=C_PRED, dots=3, noline=True)]
        app4[nm] = dict(xlib.lines(K, ser, "number of groups k", "I / F_k_eff", yrange=(0, 1)),
                        caption=f"<b>Plotted:</b> retained χ² information divided by the ceiling F at the partition's "
                                f"number of nonempty groups (tightness), against k; {lab}; final checkpoint, kernel rule {rn}. "
                                "Dashed: random balanced partitions. Markers: SEBA at its ranks, classes and predictions "
                                "(validation) at their number of nonempty groups.")
    # chains: one-sided start and end side for b9 -> logits (secondary per plan)
    T = xh["total"]
    one = lambda m, s: [np.nan if v is None else v / T for v in xh[m][s]]
    ser = [dict(name="ceiling F_k", y=pct([ceiling(xh["sv"], k) / T for k in K]), color=C_CEIL, dash="dot"),
           dict(name="spectral k-means, start side I(c(X); Y)", y=pct(one("spectral", "I_start")), color=C_SPEC),
           dict(name="spectral k-means, end side I(X; c′(Y))", y=pct(one("spectral", "I_end")), color=C_SPEC, dash=True),
           dict(name="spectral k-means, two-sided", y=pct(one("spectral", "I_two")), color=C_SEBA),
           dict(name="random, start side", y=pct(one("random", "I_start")), color=C_RAND)]
    app4["chain_one_sided"] = dict(xlib.lines(K, ser, "number of groups k", "% of total I(X;Y)",
                                              yrange=up_to([s["y"] for s in ser], step=0.5)),
                                   caption=f"<b>Plotted:</b> one-sided (start side, end side) and two-sided retained χ² "
                                           f"information as % of the total, chain {CHAIN[0]}{xlib.ARROW}{CHAIN[1]}, final "
                                           f"checkpoint, kernel rule {rn}; dotted: ceiling F_k.")
    # per-site table as a chart: % of total at k = 10, single snapshots
    ser = [dict(name="ceiling F_10", y=pct(c10), color=C_CEIL, dash="dot"),
           dict(name="spectral k-means", y=pct([score(b, "spectral", True)[0][ki] / b["total"] for b in cells]), color=C_SPEC),
           dict(name="PCA k-means (baseline)", y=pct([score(b, "pca", True)[0][ki] / b["total"] for b in cells]), color=C_PCA),
           dict(name="random (null)", y=pct([score(b, "random", True)[0][ki] / b["total"] for b in cells]), color=C_RAND, dash=True),
           dict(name="classes (validation)", y=pct([score(b, "labels", True)[0][ki] / b["total"] for b in cells]), color=C_LAB, dots=4, noline=True)]
    app4["cells_by_site"] = dict(xlib.lines(list(range(len(sites))), ser, "site (single snapshot)", "one-sided, % of total I(X;Y)",
                                            yrange=up_to([s["y"] for s in ser], step=0.5), xticklabels=sites),
                                 caption=f"<b>Plotted:</b> one-sided retained χ² information I(c(X); Y) at k = {K_MAIN} as % of "
                                         f"the snapshot's total I(X;Y), per site, final checkpoint, kernel rule {rn}; dotted: "
                                         "ceiling F_10; dashed: random balanced partitions.")
    # without the stay-put half: b9 scored on P = 2M - I
    ser = [dict(name="ceiling (singular values of P)", y=pct(np.array(xc["F_P"]) / xc["total_P"]), color=C_CEIL, dash="dot"),
           dict(name="spectral k-means", y=pct([np.nan if v is None else v / xc["total_P"] for v in xc["spectral"]["I_start_P"]]), color=C_SPEC),
           dict(name="random (null)", y=pct([np.nan if v is None else v / xc["total_P"] for v in xc["random"]["I_start_P"]]), color=C_RAND, dash=True)]
    app4["cell_P"] = dict(xlib.lines(K, ser, "number of groups k", "one-sided, % of total I(X;Y) under P",
                                     yrange=up_to([s["y"] for s in ser], step=1)),
                          caption=f"<b>Plotted:</b> the same partitions of the {CELL_SITE} snapshot scored on P = 2M − I "
                                  "(no stay-put half): one-sided retained χ² information as % of the total under P, "
                                  f"against k; final checkpoint, kernel rule {rn}; dotted: ceiling from the singular values "
                                  "of P (ordered by magnitude).")
    # two-sided retained share of every chain, k = 10, final step
    zz = [[score(blk(fin, RULE, a, b), "spectral", False)[0][ki] / blk(fin, RULE, a, b)["total"] if a < b else None
           for b in range(1, len(sites))] for a in range(len(sites) - 1)]
    zf = [[ceiling(blk(fin, RULE, a, b)["sv"], K_MAIN) / blk(fin, RULE, a, b)["total"] if a < b else None
           for b in range(1, len(sites))] for a in range(len(sites) - 1)]
    app4["chains_two_sided"] = dict(xlib.heat([zz, zf], sites[:-1], sites[1:], "% of total I(X;Y)", "start site a",
                                              "end site b", zrange=[0, 1], fmt="pct",
                                              slider=dict(label="quantity", values=["spectral k-means, two-sided", "ceiling F_10"])),
                                    caption=f"<b>Plotted:</b> for every chain a→b at the final checkpoint, kernel rule {rn}, "
                                            f"k = {K_MAIN}: the two-sided retained χ² information of spectral k-means "
                                            "(slider position 1) and the ceiling F_10 (position 2), each as a share of the "
                                            "chain's total I(X;Y).")
    # over training: b9 snapshot and b9 -> logits, k = 10
    for nm, a, b, cell in (("over_training_cell", SI[CELL_SITE], SI[CELL_SITE], True),
                           ("over_training_chain", SI[CHAIN[0]], SI[CHAIN[1]], False)):
        bs = [blk(st, RULE, a, b) for st in steps]
        f = lambda m: pct([score(x, m, cell)[0][ki] / x["total"] for x in bs])
        ser = [dict(name="ceiling F_10", y=pct([ceiling(x["sv"], K_MAIN) / x["total"] for x in bs]), color=C_CEIL, dash="dot"),
               dict(name="spectral k-means", y=f("spectral"), color=C_SPEC),
               dict(name="random (null)", y=f("random"), color=C_RAND, dash=True),
               dict(name="classes (validation)", y=f("labels"), color=C_LAB, dots=4, noline=True)]
        lab = f"single snapshot {CELL_SITE}, one-sided" if cell else f"chain {CHAIN[0]}{xlib.ARROW}{CHAIN[1]}, two-sided"
        app4[nm] = dict(xlib.lines(steps, ser, "training step", "% of total I(X;Y)", xscale="symlog",
                                   yrange=up_to([s["y"] for s in ser], step=0.5)),
                        caption=f"<b>Plotted:</b> retained χ² information at k = {K_MAIN} as % of the total, {lab}, against "
                                f"training step (symlog axis), kernel rule {rn}. Predictions are omitted: at step 0 they "
                                "occupy fewer than ten classes.")
    # gap-after-10 count by rule and step
    zr_ = [[sum(1 for b in cs["units"][f"{st}_{r}"] if b["a"] < b["b"] and gap_k(b) == K_MAIN) for st in steps] for r in rules]
    app4["chains10_by_rule"] = dict(xlib.heat(zr_, [meta["modes"][r] for r in rules], steps,
                                              f"chains with a clear gap after {K_MAIN}", "kernel rule", "training step",
                                              zrange=[0, max(max(r) for r in zr_) or 1], fmt="int", ml=170),
                                    caption=f"<b>Plotted:</b> number of the {len(chains_fin)} chains a→b (a < b) whose "
                                            f"spectrum has a clear gap after {K_MAIN} groups (largest σ_k/σ_k+1 ≥ 1.15 and "
                                            "> 1.1 × the runner-up), per kernel rule and checkpoint.")
    # leakage against the Ky Fan bound (single snapshots, k = 10) -- App C material, offered here as a chart
    ser = [dict(name=f"rule {r}: {meta['modes'][r]}", y=[blk(fin, r, i, i)["spectral"]["ncut"][ki] / blk(fin, r, i, i)["kyfan"][ki]
                                                          for i in range(len(sites))]) for r in rules]
    app4["ncut_kyfan"] = dict(xlib.lines(list(range(len(sites))), ser, "site (single snapshot)", "Ncut / Ky Fan bound",
                                         yrange=up_to([s["y"] for s in ser], step=0.5),
                                         xticklabels=sites),
                              caption=f"<b>Plotted:</b> summed leakage Σ_j φ(A_j) of spectral k-means at k = {K_MAIN} "
                                      "divided by its Ky Fan lower bound Σ_{i≤k}(1 − λ_i), per site, final checkpoint, one "
                                      "line per kernel rule (1 = the bound is attained).")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
