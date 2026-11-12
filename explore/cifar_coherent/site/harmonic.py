"""Harmonicity test (site/REFRAME_PLAN.md, "Harmonicity test (design v3)").

Usage (CPU, ~15 min):
    uv run --with torch --with numpy --with scipy python -I site/harmonic.py r20s0b [--ckpt-dir /var/tmp/cm-harm/ck]

Writes `data/<tag>/harmonic.json` ({"q_harm": ..., "facts": {...}}) and merges it into `site/assets/questions.json`
(keys `q_harm.interp`, `q_harm.smooth`; facts `harm.*`, exposed by the build as `q.harm.*`). `site/analysis.py`
re-merges the cached file on every run (`merge_into`), so analysis.py does not need torch.

What this is. Part B is a labelled **mechanism-validation intervention**: it runs the network's own suffix on
activations that no image produced (straight lines between two real activations). Nothing here enters the
construction; the coherent sets are the stored hard SEBA sets. It is a finite-resolution test of sharp downstream
transitions associated with coherent sets, not proof of literal discontinuities or of harmonic label propagation.
The ResNet-20 (ReLU, BatchNorm in eval mode) is continuous and piecewise linear in its input, so every path z(tau)
is continuous and piecewise linear: a "jump" below means a transition narrower than the sampling grid.

Inputs and provenance
---------------------
Checkpoints step_0.pt and step_64000.pt of runs/resnet20-relu-seed0 (SHA-256 checked against data/<tag>/curves.json);
model code ~/code/ModelZoo/src (zoo.models.build_model); probe pixels `probe.npz['images']` (uint8), normalised as
zoo.data does ((x - 255 mean) / (255 std), CIFAR-10 mean/std). Activations are re-extracted in float32 (the model's
precision) on CPU; Grams and operators in float64 with the repo's own code (coherent.centred_cosine_gram,
coherent.symmetric_balance; the kNN rule copied from sweep_math.final_kernel_bank, rule 0). Checks (asserted):
re-extracted logits and channel means at the used sites agree with the stored quantised `points_<step>.npz` values
to within half a quantisation step (+1e-4); predicted classes agree exactly; the rebuilt single-snapshot operators'
leading 9 eigenvalues agree with the stored `vals` to 1e-4 and |cos(psi_1, stored psi_1)| > 0.99
(the measured values are stored in q_harm.interp.provenance and the harm.prov.* facts).
Sites are indexed as in meta.sites; site b_k is the output of residual block k; its suffix is blocks k+1..9, global
average pooling and the linear head (for b9 only pooling and the head: an exact affine map, the negative control).

Part B: interpolation (q_harm.interp)
-------------------------------------
Variants: "main" (step 64,000; sets of the single-snapshot block at the same site, kNN rule, r = 4), "perp100"
(same, perplexity-100 rule; robustness), "step0" (step-0 network, its own activations and its own step-0 sets;
reference). Sites: b2, b5, b8 (predeclared) and b9 (control; not in the decision).
Pairs: among assigned images, 200,000 random ordered pairs (i != j; numpy default_rng(SEED + 100 + 10 v + s) for
variant v and site s) are split into within-set and crossing-set candidates (first 20,000 of each). Matching: both
features, endpoint cosine distance d = 1 - C_ij in the site's view (centred, row-normalised activations) and endpoint
output distance ||z_i - z_j|| (logits), are cut at the quintiles of the pooled candidates (5 x 5 bins); in bin b,
n_b = min(within_b, crossing_b); k_b = n_b * 300 / sum n_b rounded down, remainders to the largest fractions; k_b
pairs are drawn without replacement from each group in each bin. Stratum "same_pred": the selected pairs whose two
endpoints have the same predicted class (descriptive).
Paths: h(tau) = (1 - tau) h_i + tau h_j at tau = k / 256, k = 0..256; z(tau) = suffix(h(tau)) (eval mode). The
T = 64 path is every fourth point of the T = 256 path. m_T = max_k ||z(tau_{k+1}) - z(tau_k)|| at resolution T.
    sigma (sharpness index) = log(m_64 / m_256) / log 4    (0 for a jump; 1 for any path smooth at this scale);
    S (jump magnitude)      = m_256 / median over the variant-site's selected pairs of ||z(1) - z(0)||;
    descriptive: EV = sum_k ||dz_k|| / ||z(1) - z(0)|| - 1; n_class_changes = number of k where argmax z changes;
    tau_star = (argmax_k ||dz_k|| + 1/2) / 256; width = tau-span from the step where the cumulative sum of ||dz||
    first reaches 10 % of the total to where it first reaches 90 % (inclusive, in units of 1/256).
Decision rule (predeclared, variant "main"; the same rule is reported for perp100 and step0):
    premise(site) = fraction of crossing-set paths with sigma < 0.5; below 0.10 the site is "inapplicable".
    diff(site) = median S (within) - median S (crossing); 95 % percentile bootstrap CI (2,000 resamples of each
    group separately, default_rng(SEED + 200 + ...)).
    per site: inapplicable; else "consistent" if diff < 0 and CI upper < 0; "inconsistent" if diff >= 0;
    otherwise "inconclusive".
    overall: H-consistent if every applicable predeclared site is consistent; H-inconsistent if diff >= 0 at two or
    more applicable sites; otherwise inconclusive; "inapplicable" if no predeclared site is applicable.
Calibration: analytic planted 1-d paths z = g(tau) (step at tau0 = 0.5 + 0.3/256; logistic of width w in {1/256,
0.01, 0.05, 0.2}, rescaled to g(0) = 0, g(1) = 1; linear) through the same sigma and S code (S in units of
|z(1) - z(0)| = 1). Empirical: b9 (affine; sigma must be 1 up to float error); the step-0 network; and a resolution
check: 50 main-variant paths at b2, b5, b8 with sigma < 0.5 (default_rng(SEED + 300)), recomputed at T = 1,024:
sigma_hi = log(m_256 / m_1024) / log 4 on that path (0 = still a jump at 4x finer resolution).

Part A: downstream smoothness, descriptive (q_harm.smooth), cut to essentials
---------------------------------------------------------------------------
Step 64,000, sites s in {b2, b5, b8, b9}, operator M_s = (I + P_s)/2 of the single-snapshot block (kNN rule),
rebuilt in float64 (and checked as above). Targets F (centred over the probe): the logits; and, for s before b9,
the top-10 principal-component scores of b9's centred, row-normalised view (each scaled to unit variance; the
retained variance share is stored). Quantities:
    E_s(F) = tr(F^T (I - M_s) F) / tr(F^T F)   (equivalent to spectral concentration; only E is reported);
    edge energy sum_ij P_ij ||F_i - F_j||^2 split into the share on within-set edges (both ends assigned, same set),
    crossing edges (both assigned, different sets) and edges touching an unassigned image; sets = the site's own
    single-snapshot sets (sites strictly before the target);
    coverage = assigned fraction; var_explained = 1 - SS_within / SS_total with the unassigned images as one more group.
Calibration on the same graph: planted piecewise-constant F = one-hot of the site's set (unassigned images get a
uniformly random set), with a fraction l in {0, 0.1, 0.3} of all images reassigned to a uniformly random set
(default_rng(SEED + 400)); affine F = A W (A = raw flattened activations at s, W Gaussian D x 10,
default_rng(SEED + 401)); smooth nonlinear F = tanh of the standardised columns of A W' (W' default_rng(SEED + 402)).
E and the shares are scale-free, so variance matching is automatic. Exact identity check: the head applied to the
re-extracted pool activations reproduces the re-extracted logits (max abs difference stored). Neighbourhood scale:
the median 15-NN cosine distance per site (k = 50 diagnostics were cut for time).
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

SITE_DIR = Path(__file__).resolve().parent
ROOT = SITE_DIR.parent
SEED = 20261112
PRIMARY = ['b2', 'b5', 'b8']
CONTROL = 'b9'
N_PAIRS, N_CAND, N_DRAW, NBINS = 300, 20000, 200000, 5
T_HI, T_LO, T_CHECK, N_CHECK = 256, 64, 1024, 50
N_BOOT = 2000
R_SETS = 4
ZOO_SRC = Path.home() / 'code/ModelZoo/src'
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _analysis():
    spec = importlib.util.spec_from_file_location('analysis_helpers', SITE_DIR / 'analysis.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_into(out, data_dir):
    """Merge a cached harmonic.json into a questions.json dict (q_harm key, harm.* facts). Returns True if merged."""
    p = Path(data_dir) / 'harmonic.json'
    if not p.exists():
        return False
    h = json.loads(p.read_text())
    out['q_harm'] = h['q_harm']
    facts = out.setdefault('facts', {})
    for k in [k for k in facts if k.startswith('harm.')]:
        del facts[k]
    facts.update(h['facts'])
    return True


# ---------------------------------------------------------------- path statistics (pure numpy)

def path_stats(z, T=T_HI, lo=T_LO):
    """z (P, T+1, K) -> dict of per-path arrays. Assumes T % lo == 0."""
    d = np.linalg.norm(np.diff(z, axis=1), axis=2)  # (P, T)
    m_hi = d.max(1)
    zl = z[:, ::T // lo]
    m_lo = np.linalg.norm(np.diff(zl, axis=1), axis=2).max(1)
    sigma = np.log(m_lo / m_hi) / np.log(T // lo)
    end = np.linalg.norm(z[:, -1] - z[:, 0], axis=1)
    tot = d.sum(1)
    cs = np.cumsum(d, 1) / tot[:, None]
    k10 = (cs >= 0.1).argmax(1)
    k90 = (cs >= 0.9).argmax(1)
    if z.shape[2] > 1:
        cls = z.argmax(2)
        changes = (np.diff(cls, axis=1) != 0).sum(1)
    else:
        changes = np.zeros(len(z), int)
    return dict(m_hi=m_hi, m_lo=m_lo, sigma=sigma, end=end, ev=tot / end - 1, changes=changes,
                tau_star=(d.argmax(1) + 0.5) / T, width=(k90 - k10 + 1) / T)


def analytic_calibration():
    tau = np.arange(T_HI + 1) / T_HI
    t0 = 0.5 + 0.3 / T_HI
    rows = []
    g_step = (tau >= t0).astype(float)
    rows.append(('step', g_step))
    for w, name in ((1 / 256, 'logistic w=1/256'), (0.01, 'logistic w=0.01'), (0.05, 'logistic w=0.05'),
                    (0.2, 'logistic w=0.2')):
        L = lambda x: 1 / (1 + np.exp(-(x - t0) / w))
        rows.append((name, (L(tau) - L(0)) / (L(1) - L(0))))
    rows.append(('linear', tau.copy()))
    out = []
    for name, g in rows:
        st = path_stats(g[None, :, None])
        out.append(dict(path=name, sigma=float(st['sigma'][0]), S=float(st['m_hi'][0] / st['end'][0]),
                        width=float(st['width'][0])))
    return out


# ---------------------------------------------------------------- pair selection (pure numpy)

def select_pairs(labels, C, z, rng):
    """Matched within/crossing pairs. Returns (within (n,2), crossing (n,2), balance dict)."""
    A = np.flatnonzero(labels >= 0)
    i = A[rng.integers(0, len(A), N_DRAW)]
    j = A[rng.integers(0, len(A), N_DRAW)]
    keep = i != j
    i, j = i[keep], j[keep]
    same = labels[i] == labels[j]
    W = np.stack([i[same], j[same]], 1)[:N_CAND]
    X = np.stack([i[~same], j[~same]], 1)[:N_CAND]
    f = lambda P: (1 - C[P[:, 0], P[:, 1]], np.linalg.norm(z[P[:, 0]] - z[P[:, 1]], axis=1))
    (wd, wz), (xd, xz) = f(W), f(X)
    q = np.linspace(0, 1, NBINS + 1)[1:-1]
    ed, ez = np.quantile(np.r_[wd, xd], q), np.quantile(np.r_[wz, xz], q)
    wb = np.searchsorted(ed, wd) * NBINS + np.searchsorted(ez, wz)
    xb = np.searchsorted(ed, xd) * NBINS + np.searchsorted(ez, xz)
    nb = np.array([min((wb == b).sum(), (xb == b).sum()) for b in range(NBINS * NBINS)])
    target = min(N_PAIRS, int(nb.sum()))
    raw = nb * target / max(nb.sum(), 1)
    kb = np.floor(raw).astype(int)
    for b in np.argsort(-(raw - kb))[:target - kb.sum()]:
        kb[b] += 1
    kb = np.minimum(kb, nb)
    sw, sx = [], []
    for b in range(NBINS * NBINS):
        if kb[b]:
            sw += list(rng.choice(np.flatnonzero(wb == b), kb[b], replace=False))
            sx += list(rng.choice(np.flatnonzero(xb == b), kb[b], replace=False))
    sw, sx = np.array(sw, int), np.array(sx, int)
    bal = dict(n=int(len(sw)), cand_within=int(len(W)), cand_crossing=int(len(X)),
               median_dcos_within=float(np.median(wd[sw])), median_dcos_crossing=float(np.median(xd[sx])),
               median_dz_within=float(np.median(wz[sw])), median_dz_crossing=float(np.median(xz[sx])))
    return W[sw], X[sx], bal


def boot_diff(a, b, rng):
    ia = rng.integers(0, len(a), (N_BOOT, len(a)))
    ib = rng.integers(0, len(b), (N_BOOT, len(b)))
    d = np.median(a[ia], 1) - np.median(b[ib], 1)
    return float(np.median(a) - np.median(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_median_ci(a, rng):
    m = np.median(a[rng.integers(0, len(a), (N_BOOT, len(a)))], 1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def site_label(premise, diff, hi):
    if premise < 0.10:
        return 'inapplicable'
    if diff >= 0:
        return 'inconsistent'
    return 'consistent' if hi < 0 else 'inconclusive'


def overall_label(rows):
    app = [r for s, r in rows.items() if s in PRIMARY and r['label'] != 'inapplicable']
    if not app:
        return 'inapplicable'
    if all(r['label'] == 'consistent' for r in app):
        return 'H-consistent'
    if sum(r['diff'] >= 0 for r in app) >= 2:
        return 'H-inconsistent'
    return 'inconclusive'


# ---------------------------------------------------------------- torch parts

def load_model(ck_dir, step, data_dir):
    import torch
    sys.path.insert(0, str(ZOO_SRC))
    from zoo.models import build_model
    p = Path(ck_dir) / f'step_{step}.pt'
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    man = {c['step']: c['sha256'] for c in json.loads((data_dir / 'curves.json').read_text())['checkpoints']}
    assert sha == man[step], (p, sha, man[step])
    ck = torch.load(p, map_location='cpu', weights_only=False)
    model = build_model(ck['config'])
    model.load_state_dict(ck['model_state'])
    return model.eval(), sha


def extract(model, images):
    import torch
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1) * 255.0
    x = (x - mean) / std
    out = None
    with torch.inference_mode():
        for a in range(0, len(x), 250):
            h = model.act(model.bn1(model.conv1(x[a:a + 250])))
            acts = [h]
            for blk in model.blocks:
                h = blk(h)
                acts.append(h)
            pool = h.mean((2, 3))
            acts += [pool, model.fc(pool)]
            out = [[a_] for a_ in acts] if out is None else [o + [a_] for o, a_ in zip(out, acts)]
    return [torch.cat(o).numpy() for o in out]


def operator(act):
    """Rebuilt single-snapshot operator, kNN rule (rule 0): returns (C, M) float64 numpy."""
    import torch
    sys.path.insert(0, str(ROOT))
    from coherent import centred_cosine_gram, symmetric_balance
    c = centred_cosine_gram(torch.from_numpy(act))
    d = 1 - c.double()
    d.diagonal().fill_(float('inf'))
    near = d.topk(15, largest=False, dim=1)
    mask = torch.zeros_like(c, dtype=torch.bool)
    mask.scatter_(1, near.indices, True)
    mask = mask | mask.T
    w = torch.where(mask, torch.ones_like(c), torch.full_like(c, 1e-6))
    w.diagonal().fill_(0)
    p = symmetric_balance(torch.log(w))
    p = p[0] if isinstance(p, tuple) else p
    m = p * 0.5
    m.diagonal().add_(0.5)
    return c.numpy(), m.numpy(), float(near.values[:, -1].median())


def run_paths(model, act, pairs, site_k, T):
    """z (P, T+1, 10) float64 for straight lines between act[i] and act[j], suffix from site index site_k."""
    import torch
    blocks = list(model.blocks)[site_k:]
    tau = torch.arange(T + 1, dtype=torch.float32) / T
    A = torch.from_numpy(act)
    per = max(1, 2048 // (T + 1))
    out = np.empty((len(pairs), T + 1, 10))
    with torch.inference_mode():
        for a in range(0, len(pairs), per):
            pr = pairs[a:a + per]
            hi, hj = A[pr[:, 0]], A[pr[:, 1]]
            shp = (len(pr), T + 1) + (1,) * (hi.dim() - 1)
            h = (1 - tau).view(1, -1, *([1] * (hi.dim() - 1))) * hi[:, None] + \
                tau.view(1, -1, *([1] * (hi.dim() - 1))) * hj[:, None]
            h = h.reshape(-1, *hi.shape[1:])
            for blk in blocks:
                h = blk(h)
            z = model.fc(h.mean((2, 3)))
            out[a:a + len(pr)] = z.reshape(len(pr), T + 1, 10).double().numpy()
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--ckpt-dir', default='/var/tmp/cm-harm/ck')
    ap.add_argument('--no-merge', action='store_true')
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    A = _analysis()
    data_dir = ROOT / 'data' / args.tag
    meta = json.loads((data_dir / 'meta.json').read_text())
    sites, steps = meta['sites'], meta['steps']
    sidx = {s: i for i, s in enumerate(sites)}
    lp = {tuple(p): i for i, p in enumerate(meta['layer_pairs'])}
    with np.load(data_dir / 'probe.npz') as p:
        images = p['images']
    ri = meta['seba_r'].index(R_SETS)
    use_sites = PRIMARY + [CONTROL]
    prov, nets = {}, {}
    for step in (64000, 0):
        model, sha = load_model(args.ckpt_dir, step, data_dir)
        acts = extract(model, images)
        with np.load(data_dir / f'points_{step}.npz') as pt:
            def deq(key):
                q, lo, hi = pt[key].astype(np.float64), pt[key + '_lo'], pt[key + '_hi']
                return lo + (hi - lo) * q / 255.0, (hi - lo) / 510.0
            zq, tol = deq('chan_logits')
            err_logit = float((np.abs(acts[-1] - zq) - tol).max())
            pred_ok = bool((acts[-1].argmax(1) == pt['pred']).all())
            err_chan = {}
            for s in use_sites + ['pool']:
                v, tl = deq(f'chan_{s}')
                a = acts[sidx[s]]
                cm = a.mean((2, 3)) if a.ndim == 4 else a
                err_chan[s] = float((np.abs(cm - v) - tl).max())
        assert err_logit < 1e-4 and pred_ok and max(err_chan.values()) < 1e-4, (step, err_logit, pred_ok, err_chan)
        with np.load(data_dir / f'layer_{step}.npz') as L:
            vals, vecs, scale, labs = L['vals'], L['vecs'], L['scale'], L['seba_label']
        ops = {}
        spec_err, psi_cos = {}, {}
        for s in use_sites:
            k = sidx[s]
            C, M, d15 = operator(acts[k])
            ev, evec = np.linalg.eigh(M)
            ev, evec = ev[::-1][:9], evec[:, ::-1][:, :9]
            pi = lp[(k, k)]
            spec_err[s] = float(np.abs(ev - vals[0, pi]).max())
            u = vecs[0, pi, 0].astype(np.float64) * scale[0, pi, 0] / 32767
            psi_cos[s] = float(abs(evec[:, 1] @ u) / np.linalg.norm(u))
            ops[s] = dict(C=C, M=M, d15=d15, sets={0: labs[0, pi, ri].astype(int), 1: labs[1, pi, ri].astype(int)})
        assert max(spec_err.values()) < 1e-4 and min(psi_cos.values()) > 0.99, (step, spec_err, psi_cos)
        head_err = float(np.abs(model.fc(torch.from_numpy(acts[sidx['pool']])).detach().numpy() - acts[-1]).max())
        prov[step] = dict(sha256=sha, logit_excess_over_half_quant_step=err_logit, preds_match=pred_ok,
                          channel_mean_excess=err_chan, eig_max_abs_diff=spec_err, psi1_abs_cos=psi_cos,
                          head_identity_max_abs_diff=head_err)
        keep = {sidx[s] for s in use_sites} | {sidx['pool'], len(acts) - 1}
        acts = [a if i in keep else None for i, a in enumerate(acts)]  # memory: drop unused sites
        for s in use_sites:
            ops[s]['C'] = ops[s]['C'].astype(np.float32)
            if step != 64000:
                ops[s]['M'] = None  # Part A uses step 64,000 only
        nets[step] = dict(model=model, acts=acts, ops=ops)
        print(f'step {step}: provenance ok ({time.time() - t0:.0f}s): eig diff {max(spec_err.values()):.1e}, '
              f'logit excess {err_logit:.1e}', flush=True)

    # ------------------------------------------------ Part B
    variants = {'main': (64000, 0, use_sites), 'perp100': (64000, 1, PRIMARY), 'step0': (0, 0, use_sites)}
    cache = {}
    interp = dict(definitions='see site/harmonic.py docstring, Part B', intervention=True, T=[T_LO, T_HI],
                  n_pairs_target=N_PAIRS, sites=use_sites, primary_sites=PRIMARY, control_site=CONTROL,
                  provenance={str(k): v for k, v in prov.items()}, variants={})
    raw = {}
    for vi, (vname, (step, mode, vsites)) in enumerate(variants.items()):
        net = nets[step]
        z_img = net['acts'][-1].astype(np.float64)
        pred = z_img.argmax(1)
        rows = {}
        for si_, s in enumerate(vsites):
            k = sidx[s]
            op = net['ops'][s]
            lab = op['sets'][mode]
            rng = np.random.default_rng(SEED + 100 + 10 * vi + si_)
            Wp, Xp, bal = select_pairs(lab, op['C'], z_img, rng)
            allp = np.r_[Wp, Xp]
            need = [tuple(p) for p in allp if (step, s, tuple(p)) not in cache]
            if need:
                zz = run_paths(net['model'], net['acts'][k], np.array(need), k, T_HI)
                for p, zp in zip(need, zz):
                    cache[(step, s, p)] = zp
            Z = np.stack([cache[(step, s, tuple(p))] for p in allp])
            end_err = float(np.abs(Z[:, 0] - z_img[allp[:, 0]]).max())
            st = path_stats(Z)
            norm = float(np.median(st['end']))
            S = st['m_hi'] / norm
            nw = len(Wp)
            w_, x_ = slice(0, nw), slice(nw, None)
            rngb = np.random.default_rng(SEED + 200 + 10 * vi + si_)
            diff, lo, hi = boot_diff(S[w_], S[x_], rngb)
            premise = float((st['sigma'][x_] < 0.5).mean())
            sp = pred[allp[:, 0]] == pred[allp[:, 1]]
            spw, spx = sp[w_], sp[x_]
            same = None
            if spw.sum() >= 10 and spx.sum() >= 10:
                d2, l2, h2 = boot_diff(S[w_][spw], S[x_][spx], rngb)
                same = dict(n_within=int(spw.sum()), n_crossing=int(spx.sum()), diff=d2, ci=[l2, h2])
            q = lambda a: [float(np.quantile(a, v)) for v in (0.1, 0.25, 0.5, 0.75, 0.9)]
            rows[s] = dict(
                step=step, mode=mode, balance=bal, endpoint_check_max_abs=end_err, norm_median_endpoint_dz=norm,
                premise=premise, frac_within_sigma_lt_half=float((st['sigma'][w_] < 0.5).mean()),
                median_S_within=float(np.median(S[w_])), median_S_crossing=float(np.median(S[x_])),
                ci_median_S_within=boot_median_ci(S[w_], rngb), ci_median_S_crossing=boot_median_ci(S[x_], rngb),
                diff=diff, ci=[lo, hi], same_pred=same,
                sigma_q_within=q(st['sigma'][w_]), sigma_q_crossing=q(st['sigma'][x_]),
                sigma_min=float(st['sigma'].min()), sigma_max=float(st['sigma'].max()),
                ev_median_within=float(np.median(st['ev'][w_])), ev_median_crossing=float(np.median(st['ev'][x_])),
                changes_median_within=float(np.median(st['changes'][w_])),
                changes_median_crossing=float(np.median(st['changes'][x_])),
                width_median_within=float(np.median(st['width'][w_])),
                width_median_crossing=float(np.median(st['width'][x_])))
            rows[s]['label'] = 'control' if s == CONTROL else site_label(premise, diff, hi)
            raw[(vname, s)] = dict(sigma=st['sigma'], S=S, nw=nw, pairs=allp, step=step)
            print(f'  {vname} {s}: n={nw}+{len(Xp)} premise={premise:.3f} medS w/x={np.median(S[w_]):.4f}/'
                  f'{np.median(S[x_]):.4f} diff={diff:.4f} [{lo:.4f},{hi:.4f}] -> {rows[s]["label"]} '
                  f'({time.time() - t0:.0f}s)', flush=True)
        interp['variants'][vname] = dict(step=step, rule=meta['modes'][mode], sites=rows, verdict=overall_label(rows))

    # resolution check at T = 1024
    rng = np.random.default_rng(SEED + 300)
    cand = [(s, i) for s in PRIMARY for i in np.flatnonzero(raw[('main', s)]['sigma'] < 0.5)]
    pick = [cand[c] for c in rng.choice(len(cand), min(N_CHECK, len(cand)), replace=False)] if cand else []
    hi_sig = []
    for s, i in pick:
        pr = raw[('main', s)]['pairs'][i][None]
        zf = run_paths(nets[64000]['model'], nets[64000]['acts'][sidx[s]], pr, sidx[s], T_CHECK)
        m1024 = np.linalg.norm(np.diff(zf[0], axis=0), axis=1).max()
        m256 = np.linalg.norm(np.diff(zf[0, ::4], axis=0), axis=1).max()
        hi_sig.append(float(np.log(m256 / m1024) / np.log(4)))
    hi_sig = np.array(hi_sig)
    interp['resolution_check'] = dict(
        n=len(pick), n_candidates=len(cand), sites=[s for s, _ in pick], sigma_256_1024=hi_sig,
        median=float(np.median(hi_sig)) if len(hi_sig) else None,
        frac_below_half=float((hi_sig < 0.5).mean()) if len(hi_sig) else None)
    interp['calibration_analytic'] = analytic_calibration()
    print(f'resolution check done ({time.time() - t0:.0f}s)', flush=True)

    # charts
    grid = np.round(np.arange(-0.2, 1.401, 0.02), 2)
    for vname in variants:
        vs = variants[vname][2]
        ser = []
        for s in vs:
            r = raw[(vname, s)]
            for part, sl, dash in (('within', slice(0, r['nw']), True), ('crossing', slice(r['nw'], None), False)):
                sg = np.sort(r['sigma'][sl])
                ser.append(dict(name=f'{s} {part}', y=np.searchsorted(sg, grid, side='right') / len(sg), dash=dash))
        interp['variants'][vname]['chart_sigma_cdf'] = A.lines(
            list(grid), ser, 'sharpness index σ (0 = jump, 1 = smooth at T = 64 vs 256)',
            'fraction of paths with σ ≤ x', 'linear', (0, 1), colors='categorical')
        rows = interp['variants'][vname]['sites']
        ym = max(max(r['ci_median_S_crossing'][1], r['ci_median_S_within'][1]) for r in rows.values())
        interp['variants'][vname]['chart_S'] = A.lines(
            list(range(len(vs))),
            [dict(name='within-set pairs', y=[rows[s]['median_S_within'] for s in vs],
                  band=[[rows[s]['ci_median_S_within'][0] for s in vs], [rows[s]['ci_median_S_within'][1] for s in vs]]),
             dict(name='crossing-set pairs', y=[rows[s]['median_S_crossing'] for s in vs],
                  band=[[rows[s]['ci_median_S_crossing'][0] for s in vs],
                        [rows[s]['ci_median_S_crossing'][1] for s in vs]])],
            'site (suffix = the rest of the network)', 'median jump magnitude S (bootstrap 95 % CI)', 'linear',
            (0, float(np.ceil(ym * 11) / 10)), xticklabels=vs)

    # ------------------------------------------------ Part A
    net = nets[64000]
    F_log = net['acts'][-1].astype(np.float64)
    xb9 = net['acts'][sidx['b9']].reshape(2000, -1).astype(np.float64)
    xb9 = xb9 - xb9.mean(0)
    xb9 /= np.linalg.norm(xb9, axis=1, keepdims=True)
    xb9 -= xb9.mean(0)
    U, sv, _ = np.linalg.svd(xb9, full_matrices=False)
    F_pc = U[:, :10] * np.sqrt(2000)
    pc_var = float((sv[:10] ** 2).sum() / (sv ** 2).sum())

    def energy(F, M, lab):
        F = F - F.mean(0)
        E = float(np.trace(F.T @ F - F.T @ (M @ F)) / np.trace(F.T @ F))
        P = 2 * M - np.eye(len(M))
        sq = (F ** 2).sum(1)
        D2 = sq[:, None] + sq[None] - 2 * F @ F.T
        Wt = P * D2
        a = lab >= 0
        inside = (lab[:, None] == lab[None]) & a[:, None] & a[None]
        cross = (lab[:, None] != lab[None]) & a[:, None] & a[None]
        tot = Wt.sum()
        g = np.where(a, lab, lab.max() + 1)
        ssw = sum(((F[g == c] - F[g == c].mean(0)) ** 2).sum() for c in np.unique(g))
        return dict(E=E, within_share=float(Wt[inside].sum() / tot), crossing_share=float(Wt[cross].sum() / tot),
                    unassigned_share=float(Wt[~(inside | cross)].sum() / tot),
                    var_explained=float(1 - ssw / (F ** 2).sum()))

    smooth = dict(definitions='see site/harmonic.py docstring, Part A (descriptive, cut to essentials)',
                  b9_pc10_variance_retained=pc_var, sites={})
    rng_pl = np.random.default_rng(SEED + 400)
    for si_, s in enumerate(use_sites):
        op = net['ops'][s]
        M, lab = op['M'], op['sets'][0]
        Xr = net['acts'][sidx[s]].reshape(2000, -1).astype(np.float64)
        r = R_SETS
        row = dict(coverage=float((lab >= 0).mean()), median_15nn_distance=op['d15'], logits=energy(F_log, M, lab))
        if s != CONTROL:
            row['b9_pc10'] = energy(F_pc, M, lab)
        for leak in (0.0, 0.1, 0.3):
            g = np.where(lab >= 0, lab, rng_pl.integers(0, r, 2000))
            flip = rng_pl.random(2000) < leak
            g = np.where(flip, rng_pl.integers(0, r, 2000), g)
            row[f'planted_{leak}'] = energy(np.eye(r)[g], M, lab)
        Wr = np.random.default_rng(SEED + 401 + si_).standard_normal((Xr.shape[1], 10))
        row['affine'] = energy(Xr @ Wr, M, lab)
        pre = Xr @ np.random.default_rng(SEED + 402 + si_).standard_normal((Xr.shape[1], 10))
        pre = (pre - pre.mean(0)) / pre.std(0)
        row['tanh'] = energy(np.tanh(pre), M, lab)
        smooth['sites'][s] = row
    names = [('logits', 'logits'), ('b9_pc10', 'b9 top-10 PC scores'), ('planted_0.0', 'planted on sets, leak 0'),
             ('planted_0.1', 'planted, leak 0.1'), ('planted_0.3', 'planted, leak 0.3'),
             ('affine', 'affine of activations'), ('tanh', 'random tanh features')]
    for q_, ylab in (('E', 'E_s(F) = tr(Fᵀ(I − M_s)F) / tr(FᵀF)'),
                     ('crossing_share', 'share of edge energy on crossing-set edges')):
        ys = [[smooth['sites'][s][n][q_] if n in smooth['sites'][s] else None for s in use_sites] for n, _ in names]
        top = max(v for y in ys for v in y if v is not None)
        smooth[f'chart_{q_}'] = A.lines(list(range(len(use_sites))),
                                        [dict(name=lbl, y=y, dash=n not in ('logits', 'b9_pc10'))
                                         for (n, lbl), y in zip(names, ys)],
                                        'site (operator M_s and sets of its single-snapshot block)', ylab, 'linear',
                                        (0, float(np.ceil(top * 11) / 10)), xticklabels=use_sites)

    # ------------------------------------------------ facts
    f2, f3 = (lambda x: 'n/a' if x is None else f'{x:.2f}'), (lambda x: 'n/a' if x is None else f'{x:.3f}')
    facts = {}
    for vname, v in interp['variants'].items():
        facts[f'harm.interp.{vname}.verdict'] = v['verdict']
        for s, r in v['sites'].items():
            pre = f'harm.interp.{vname}.{s}'
            facts[f'{pre}.label'] = r['label']
            facts[f'{pre}.premise'] = f2(r['premise'])
            facts[f'{pre}.frac_within_sigma_lt_half'] = f2(r['frac_within_sigma_lt_half'])
            facts[f'{pre}.median_S_within'] = f3(r['median_S_within'])
            facts[f'{pre}.median_S_crossing'] = f3(r['median_S_crossing'])
            facts[f'{pre}.diff'] = f3(r['diff'])
            facts[f'{pre}.ci_lo'], facts[f'{pre}.ci_hi'] = f3(r['ci'][0]), f3(r['ci'][1])
            facts[f'{pre}.n_pairs'] = str(r['balance']['n'])
            facts[f'{pre}.sigma_median_within'] = f2(r['sigma_q_within'][2])
            facts[f'{pre}.sigma_median_crossing'] = f2(r['sigma_q_crossing'][2])
            facts[f'{pre}.sigma_min'], facts[f'{pre}.sigma_max'] = f3(r['sigma_min']), f3(r['sigma_max'])
            facts[f'{pre}.width_median_crossing'] = f3(r['width_median_crossing'])
            facts[f'{pre}.changes_median_crossing'] = f'{r["changes_median_crossing"]:g}'
            if r['same_pred']:
                facts[f'{pre}.same_pred.diff'] = f3(r['same_pred']['diff'])
                facts[f'{pre}.same_pred.ci_lo'] = f3(r['same_pred']['ci'][0])
                facts[f'{pre}.same_pred.ci_hi'] = f3(r['same_pred']['ci'][1])
                facts[f'{pre}.same_pred.n'] = f"{r['same_pred']['n_within']}+{r['same_pred']['n_crossing']}"
    rc = interp['resolution_check']
    facts['harm.interp.check1024.n'] = str(rc['n'])
    facts['harm.interp.check1024.median_sigma'] = f2(rc['median'])
    facts['harm.interp.check1024.frac_below_half'] = f2(rc['frac_below_half'])
    for c in interp['calibration_analytic']:
        key = c['path'].replace(' ', '_').replace('=', '').replace('/', '_')
        facts[f'harm.calib.{key}.sigma'] = f2(c['sigma'])
        facts[f'harm.calib.{key}.S'] = f3(c['S'])
    for step, pv in prov.items():
        facts[f'harm.prov.step{step}.eig_max_abs_diff'] = f'{max(pv["eig_max_abs_diff"].values()):.1e}'
        facts[f'harm.prov.step{step}.psi1_abs_cos_min'] = f'{min(pv["psi1_abs_cos"].values()):.4f}'
        facts[f'harm.prov.step{step}.head_identity_max_abs_diff'] = f'{pv["head_identity_max_abs_diff"]:.1e}'
    facts['harm.T_lo'], facts['harm.T_hi'], facts['harm.n_pairs_target'] = str(T_LO), str(T_HI), str(N_PAIRS)
    facts['harm.smooth.b9_pc10_variance_retained'] = f2(pc_var)
    for s, row in smooth['sites'].items():
        for n in ('logits', 'b9_pc10', 'planted_0.0', 'planted_0.1', 'planted_0.3', 'affine', 'tanh'):
            if n in row:
                facts[f'harm.smooth.{s}.{n}.E'] = f3(row[n]['E'])
                facts[f'harm.smooth.{s}.{n}.crossing_share'] = f2(row[n]['crossing_share'])
        facts[f'harm.smooth.{s}.coverage'] = f2(row['coverage'])

    res = A.clean(dict(q_harm=dict(interp=interp, smooth=smooth), facts=facts))
    (data_dir / 'harmonic.json').write_text(json.dumps(res, separators=(',', ':'), allow_nan=False,
                                                        ensure_ascii=False))
    print(f'wrote {data_dir / "harmonic.json"} ({time.time() - t0:.0f}s)', flush=True)
    if not args.no_merge:
        qp = SITE_DIR / 'assets' / 'questions.json'
        out = json.loads(qp.read_text())
        merge_into(out, data_dir)
        qp.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False, ensure_ascii=False))
        print(f'merged into {qp}', flush=True)
    print(json.dumps({k: v for k, v in facts.items() if k.endswith(('verdict', '.label', 'premise', '.diff'))},
                     indent=0, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
