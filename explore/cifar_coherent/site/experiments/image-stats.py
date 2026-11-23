"""X9 x-image-stats: how does the association of the leading eigenfunctions with twelve pixel statistics change
over training?

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/image-stats.py r20s0b
    python3 site/check_fragment.py image-stats

Reads data/<tag>/meta.json, data/<tag>/probe.npz (pixels; labels only for the within-class validation),
data/<tag>/layer_<step>.npz (psi_1..psi_8 of every single-snapshot block, default kernel rule),
site/assets/questions.json#q_imgstats (the stored fit, nulls and heatmaps of analysis_findings.py section 3) and the
baselines data data/r20s0c/sources_layer_<step>.npz (agreement of each block's sets with the method run on the
network's input, I1, and on the same block at step 0, I2).

Local computation (about a minute; numpy only), all closed-form fits on already-computed eigenfunctions, nothing in
the construction:
  - the twelve statistics are recomputed from the pixels (a copy of analysis_findings.image_stats) and checked
    against the stored per-image values;
  - R^2 of psi_k on the statistics is recomputed and checked against the stored r2_psi (3 decimals);
  - within-class R^2: psi_k and the statistics each have their class mean removed, then the same fit; null: images
    permuted within each class (1,000 draws), 95th percentile pooled over k = 1..8, per snapshot;
  - span R^2: the largest R^2 of any unit combination of psi_1..psi_8 (the squared first canonical correlation
    between span(psi_1..psi_8) and the statistics' span), with its own image-permutation null (1,000 draws).
Writes assets/x/image-stats.json and assets/x/image-stats.facts.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import numpy as np  # noqa: E402

import xlib  # noqa: E402

BASELINES = "r20s0c"        # baselines data (sources I1 = pixels, I2 = untrained network)
R_SETS = 4                  # the site's default rank for set comparisons (Appendix C, defaults)
N_PERM = 1000
SEED = 20261119
STATS = ['mean_r', 'mean_g', 'mean_b', 'luminance', 'contrast', 'saturation', 'bc_r', 'bc_g', 'bc_b', 'bc_lum',
         'bc_dist', 'hf_frac']


def image_stats(images):
    """Copy of site/analysis_findings.py image_stats (Appendix C, image statistics). Returns (N, 12)."""
    x = images.astype(np.float64) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0.0)
    H, W = Y.shape[1:]
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ring = np.minimum(np.minimum(ii, jj), np.minimum(H - 1 - ii, W - 1 - jj)) < 4
    centre = (ii >= 8) & (ii < 24) & (jj >= 8) & (jj < 24)
    bc = lambda A: A[:, ring].mean(1) - A[:, centre].mean(1)
    Y0 = Y - Y.mean((1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(Y0)) ** 2
    f = np.fft.fftfreq(H) * H
    rad = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    tot = P[:, rad > 0].sum(1)
    hf = np.where(tot > 0, P[:, rad > 8].sum(1) / np.where(tot > 0, tot, 1), 0.0)
    out = dict(mean_r=R.mean((1, 2)), mean_g=G.mean((1, 2)), mean_b=B.mean((1, 2)), luminance=Y.mean((1, 2)),
               contrast=Y.std((1, 2)), saturation=sat.mean((1, 2)), bc_r=bc(R), bc_g=bc(G), bc_b=bc(B), bc_lum=bc(Y),
               hf_frac=hf)
    out['bc_dist'] = np.sqrt(out['bc_r'] ** 2 + out['bc_g'] ** 2 + out['bc_b'] ** 2)
    return np.stack([out[k] for k in STATS], 1)


def zscore(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


def span_basis(Z):
    """Orthonormal basis of the column span of Z (rank-revealing SVD; the statistics are collinear)."""
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, s > 1e-8 * s[0]]


def main(tag: str):
    src = xlib.DATA / tag
    meta = json.loads((src / "meta.json").read_text())
    steps, sites, rule = meta["steps"], meta["sites"], meta["default_mode"]
    assert sites == xlib.SITES
    q = json.loads((xlib.SITE / "assets" / "questions.json").read_text())["q_imgstats"]
    assert q["stats"] == STATS and q["mode"] == rule
    T, S = len(steps), len(sites)
    fin = steps[-1]
    lp = meta["layer_pairs"]
    cells = [lp.index([s, s]) for s in range(S)]

    with np.load(src / "probe.npz") as p:
        images, cls = p["images"], p["labels"].astype(np.int64)
    X = image_stats(images)
    N = len(X)
    stored_pi = np.stack([np.array(q["per_image"][s], float) for s in STATS], 1)
    if np.abs(X - stored_pi).max() > 6e-5:
        raise SystemExit("recomputed image statistics differ from questions.json per_image")

    def within(M):
        M = np.array(M, float, copy=True)
        for c in range(10):
            M[cls == c] -= M[cls == c].mean(0)
        return M

    Q = span_basis(zscore(X))                   # (N, rank) statistics' span (centred)
    Qw = span_basis(within(X))                  # within-class: class means removed
    rank, rank_w = Q.shape[1], Qw.shape[1]
    rng = np.random.default_rng(SEED)
    pbank = np.stack([rng.permutation(N) for _ in range(N_PERM)])
    idx_c = [np.flatnonzero(cls == c) for c in range(10)]
    wbank = np.empty((N_PERM, N), np.int64)
    for d in range(N_PERM):
        for ic in idx_c:
            wbank[d, ic] = ic[rng.permutation(len(ic))]
    Qp = Q[pbank].transpose(1, 0, 2).reshape(N, N_PERM * rank)
    Qwp = Qw[wbank].transpose(1, 0, 2).reshape(N, N_PERM * rank_w)

    r2 = np.zeros((T, S, 8)); r2w = np.zeros((T, S, 8)); span = np.zeros((T, S))
    nul = np.zeros((T, S)); nulw = np.zeros((T, S)); nuls = np.zeros((T, S))
    for t, st in enumerate(steps):
        with np.load(src / f"layer_{st}.npz") as z:
            V, Sc = z["vecs"][rule][cells], z["scale"][rule][cells]
        for s in range(S):
            psi = V[s].T.astype(np.float64) * (Sc[s].astype(np.float64) / 32767.0)    # (N, 8), Appendix C
            pc = psi - psi.mean(0)
            den = (pc ** 2).sum(0)
            r2[t, s] = ((Q.T @ pc) ** 2).sum(0) / den
            nul[t, s] = np.percentile(((Qp.T @ pc) ** 2).reshape(N_PERM, rank, 8).sum(1) / den, 95)
            pw = within(psi)
            denw = (pw ** 2).sum(0)
            r2w[t, s] = ((Qw.T @ pw) ** 2).sum(0) / denw
            nulw[t, s] = np.percentile(((Qwp.T @ pw) ** 2).reshape(N_PERM, rank_w, 8).sum(1) / denw, 95)
            U = span_basis(pc)                                                          # span(psi_1..psi_8)
            span[t, s] = np.linalg.svd(Q.T @ U, compute_uv=False)[0] ** 2
            sn = [np.linalg.svd(Qp[:, d * rank:(d + 1) * rank].T @ U, compute_uv=False)[0] ** 2 for d in range(N_PERM)]
            nuls[t, s] = np.percentile(sn, 95)
    stored = np.array(q["cells"]["r2_psi"], float)
    if np.abs(r2 - stored).max() > 6e-4:
        raise SystemExit(f"recomputed R^2 differs from questions.json r2_psi by {np.abs(r2 - stored).max():.4f}")
    stored_null = float(np.max(q["cells"]["r2_null_hi"]))      # analysis_findings' own bank: must match ours closely
    if abs(nul.max() - stored_null) > 0.002:
        raise SystemExit(f"image-permutation null {nul.max():.4f} differs from the stored {stored_null:.4f}")

    # ------------------------------------------------ agreement with the pixel and untrained-network sources (r = 4)
    mc = json.loads((xlib.DATA / BASELINES / "meta_c.json").read_text())
    assert mc["based_on"] == tag and mc["kernel_rule"] == rule
    F = mc["agree_fields"]
    ri = mc["seba_r"].index(R_SETS)
    ag = {k: np.zeros((T, S)) for k in ("I1", "I2")}
    agn = {k: np.zeros((T, S)) for k in ("I1", "I2")}
    cov = {k: np.zeros((T, S)) for k in ("I1", "I2")}
    for t, st in enumerate(steps):
        with np.load(xlib.DATA / BASELINES / f"sources_layer_{st}.npz") as z:
            for k in ("I1", "I2"):
                A = z["agree_" + k]
                ag[k][t] = A[cells, ri, F.index("ami_joint")]
                agn[k][t] = A[cells, ri, F.index("ami_joint_null_q95")]
                cov[k][t] = A[cells, ri, F.index("joint_coverage")]

    # ------------------------------------------------ charts
    out = xlib.Out("image-stats")
    sl = lambda Y: [dict(name=sites[s], y=Y[:, s]) for s in range(S)]
    out.chart("lead", xlib.lines(
        steps, sl(r2[:, :, 0]), "training step", "R² of ψ₁ on the twelve statistics", xscale="symlog",
        yrange=(0, 1), colors="ordinal",
        refline=dict(y=round(float(nul.max()), 3), label="image-permutation null, 95th percentile (largest over snapshots)")))
    out.chart("within", xlib.lines(
        steps, sl(r2w[:, :, 0]), "training step", "within-class R² of ψ₁", xscale="symlog",
        yrange=(0, 1), colors="ordinal",
        refline=dict(y=round(float(nulw.max()), 3), label="within-class permutation null, 95th percentile (largest over snapshots)")))
    out.chart("pixels", xlib.lines(
        steps, sl(ag["I1"]), "training step", "joint AMI with the method on the input (r = 4)", xscale="symlog",
        yrange=(0, 1), colors="ordinal",
        refline=dict(y=round(float(agn["I1"].max()), 3), label="label-permutation null, 95th percentile (largest over snapshots)")))

    # ------------------------------------------------ facts
    i = {s: k for k, s in enumerate(sites)}
    out.fact("rule", meta["modes"][rule])
    out.fact("rank", rank)
    out.fact("n_perm", N_PERM)
    out.fact("r_sets", R_SETS)
    out.fact("null_max", nul.max(), 3)
    out.fact("wc_null_max", nulw.max(), 3)
    out.fact("span_null_max", nuls.max(), 3)
    # step 0
    out.fact("step0.lo", r2[0, :, 0].min(), 2)
    out.fact("step0.hi", r2[0, :, 0].max(), 2)
    out.fact("step0.lo_site", sites[int(r2[0, :, 0].argmin())])
    out.fact("step0.wc_lo", r2w[0, :, 0].min(), 2)
    out.fact("step0.wc_hi", r2w[0, :, 0].max(), 2)
    # final: falls with depth from stem to b8, then flat
    rf = r2[-1, :, 0]
    if not np.all(np.diff(np.round(rf[:i["b8"] + 1], 2)) < 0):
        raise SystemExit("final R^2 of psi_1 is not decreasing from stem to b8: rewrite the answer")
    for s in ("stem", "b4", "b8"):
        out.fact(f"final.{s}", rf[i[s]], 2)
    late = [i["b9"], i["pool"], i["logits"]]
    out.fact("final.late_lo", rf[late].min(), 2)
    out.fact("final.late_hi", rf[late].max(), 2)
    if rf[late].max() > rf[i["b7"]] or rf[late].min() < rf[i["b8"]] - 0.005:
        raise SystemExit("b9, pool, logits are not between b8 and b7 at the final step: rewrite the answer")
    # when: b9, pool, logits by step 1,024; b7, b8 by step 2,560; b4..b6 still high at 1,024
    t1k, t2k = steps.index(1024), steps.index(2560)
    out.fact("t1k", steps[t1k])
    out.fact("t2k", steps[t2k])
    out.fact("t1k.late_hi", r2[t1k, late, 0].max(), 2)
    out.fact("t1k.mid_lo", r2[t1k, i["b4"]:i["b6"] + 1, 0].min(), 2)
    out.fact("t2k.b7b8_hi", r2[t2k, [i["b7"], i["b8"]], 0].max(), 2)
    out.fact("t1k.b7b8_lo", r2[t1k, [i["b7"], i["b8"]], 0].min(), 2)
    if r2[t1k:, late, 0].max() > r2[t1k, late, 0].max() + 1e-9:
        raise SystemExit("b9..logits rise above their step-1,024 value later: rewrite the 'from step 1,024 on' claim")
    # within-class
    rw = r2w[-1, :, 0]
    out.fact("wc.final.stem", rw[0], 2)
    out.fact("wc.final.late_hi", rw[late].max(), 2)
    out.fact("wc.final.b7b8_hi", rw[[i["b7"], i["b8"]]].max(), 2)
    # max over k and the span of psi_1..psi_8
    mk = r2[-1].max(1)
    out.fact("maxk.final.late_hi", mk[late].max(), 2)
    out.fact("maxk.final.b7", mk[i["b7"]], 2)
    out.fact("maxk.final.b7_k", int(r2[-1, i["b7"]].argmax()) + 1)
    out.fact("span.step0.lo", span[0].min(), 2)
    out.fact("span.step0.hi", span[0].max(), 2)
    out.fact("span.final.stem", span[-1, 0], 2)
    out.fact("span.final.late_lo", span[-1, late].min(), 2)
    out.fact("span.final.late_hi", span[-1, late].max(), 2)
    # pixel source (I1) and untrained network (I2), r = 4
    for k, nm in (("I1", "pix"), ("I2", "untr")):
        out.fact(f"{nm}.final.stem", ag[k][-1, 0], 2)
        out.fact(f"{nm}.final.late_hi", ag[k][-1, late].max(), 2)
    if not np.all(ag["I1"][-1] < ag["I1"][0]):
        raise SystemExit("pixel-source agreement is not lower at the final step than at step 0 at every site")
    out.fact("pix.null_max", agn["I1"].max(), 3)
    out.fact("pix.cov_lo", cov["I1"].min(), 2)
    out.fact("pix.step0.stem", ag["I1"][0, 0], 2)
    out.fact("pix.step0.b9", ag["I1"][0, i["b9"]], 2)

    # ------------------------------------------------ App D (Reference charts)
    app4 = {}
    app4["r2_maxk"] = dict(xlib.lines(
        steps, sl(r2.max(2)), "training step", "max over k = 1..8 of R² of ψₖ", xscale="symlog", yrange=(0, 1),
        colors="ordinal", refline=dict(y=round(float(nul.max()), 3), label="image-permutation null, 95th percentile")),
        caption="<b>Plotted:</b> for each site's single-snapshot block (one line per site, ordinal colour) the largest "
                "over k = 1..8 of the R² of a least-squares fit of ψₖ on the twelve standardised image statistics plus "
                f"an intercept, against training step (symlog axis); kernel rule {meta['modes'][rule]}. Reference line: "
                "95th percentile of the same fit with the images permuted, pooled over k, largest over snapshots.")
    app4["r2_span"] = dict(xlib.lines(
        steps, sl(span), "training step", "largest R² over span(ψ₁..ψ₈)", xscale="symlog", yrange=(0, 1),
        colors="ordinal", refline=dict(y=round(float(nuls.max()), 3), label="image-permutation null, 95th percentile")),
        caption="<b>Plotted:</b> for each site's single-snapshot block the largest R² of any unit-norm combination of "
                "ψ₁..ψ₈ fitted on the twelve statistics (the squared first canonical correlation of the two spans), "
                "against training step (symlog axis). Reference line: 95th percentile of the same quantity with the "
                f"images permuted ({N_PERM} draws), largest over snapshots.")
    app4["r2_within_maxk"] = dict(xlib.lines(
        steps, sl(r2w.max(2)), "training step", "max over k of within-class R² of ψₖ", xscale="symlog", yrange=(0, 1),
        colors="ordinal", refline=dict(y=round(float(nulw.max()), 3), label="within-class permutation null, 95th percentile")),
        caption="<b>Plotted:</b> as the within-class chart of this experiment, the largest over k = 1..8; class means "
                "of ψₖ and of each statistic removed before the fit (labels used for validation only). Reference line: "
                "95th percentile with images permuted within each class, pooled over k, largest over snapshots.")
    app4["untrained"] = dict(xlib.lines(
        steps, sl(ag["I2"]), "training step", f"joint AMI with the step-0 sets (r = {R_SETS})", xscale="symlog",
        yrange=(0, 1), colors="ordinal",
        refline=dict(y=round(float(agn["I2"].max()), 3), label="label-permutation null, 95th percentile")),
        caption=f"<b>Plotted:</b> for each site's single-snapshot block, the AMI between its SEBA sets (r = {R_SETS}) "
                "and the sets of the same block at step 0 (the untrained network), on the images both assign, "
                "against training step (symlog axis; 1 at step 0 by construction). Reference line: 95th percentile of "
                "the AMI with one labelling permuted within the jointly assigned images, largest over snapshots.")
    for key, cap in (("heat_r2_psi", "R² of ψₖ (rows k = 1..8) on the twelve statistics, per site (columns); slider: "
                                     "checkpoint."),
                     ("heat_maxabs_psi", "max over k = 1..8 of |corr(statistic, ψₖ)| (rows: the twelve statistics; "
                                         "columns: sites; slider: checkpoint)."),
                     ("heat_maxabs_psi_wc", "as the previous chart with within-class correlations (class means "
                                            "removed from ψₖ and the statistic).")):
        app4[key] = dict(q[key], caption=f"<b>Plotted:</b> {cap} Single-snapshot blocks, kernel rule "
                                         f"{meta['modes'][rule]}; computed over the {N:,} probe images.")
    out.chart("app4", app4)
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
