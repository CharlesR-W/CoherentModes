#!/usr/bin/env python3
"""CPU data generator for PLAN.md. Run from the repository root:
uv run --with numpy --with torch python -I viz/toy.py

The planted model is an architectural test fixture, not a discovered mechanism.
Labels enter only the loss, validation contrasts, and export ordering.
"""
import os
# Set BEFORE importing numerical libraries: this small dense job is faster with
# one BLAS thread on this four-vCPU host (and avoids oversubscription).
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
SEED = 0
GAP_RATIO = 0.1
MAX_BLOCK = 6
BETAS = np.geomspace(0.1, 100, 12)


def unit(a):
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero views must be removed jointly, not normalized silently')
    return a / norms


def sites(per_cluster=200, seed=SEED):
    rng = np.random.default_rng(seed)
    angles = np.arange(6) * np.pi / 3
    centers = 2 * np.column_stack((np.cos(angles), np.sin(angles)))
    labels = np.repeat(np.arange(6), per_cluster)
    x = centers[labels] + rng.normal(0, .25, (len(labels), 2))
    # The split line is radial: t = 2 at cluster 2's fixed center. Radial
    # variation is deliberately nearly invisible in the initial cosine view.
    tangent = np.array([-.5, np.sqrt(3)/2])
    half = x @ tangent >= 2
    target = np.select([labels < 2, labels == 2, labels < 5],
                       [0, np.where(half, 1, 2), 3], default=1)
    return x, labels, half, target.astype(np.int64)


class Toy(nn.Module):
    def __init__(self, planted=True):
        super().__init__()
        self.input = nn.Linear(2, 3)
        self.w1 = nn.ModuleList([nn.Linear(3, 16) for _ in range(2)])
        self.w2 = nn.ModuleList([nn.Linear(16, 3, bias=False) for _ in range(2)])
        self.output = nn.Linear(3, 4, bias=False)
        if planted:
            self.plant()

    @torch.no_grad()
    def plant(self):
        for p in self.parameters():
            p.zero_()
        e = torch.tensor([np.sqrt(3)/2, .5, 0.])
        t = torch.tensor([-.5, np.sqrt(3)/2, 0.])
        self.input.weight[:2] = torch.eye(2)
        self.input.bias[2] = .1
        # Block 1: write h=q e+t_coord t+0.1 z. With s=12(q-.85),
        # t_new=t_coord - [ReLU(t_coord+s)-ReLU(-t_coord+s)]/2.
        # In the right-hand wedge s>|t_coord| this is EXACTLY zero,
        # putting clusters 0 and 1 on the same narrow ray; outside the
        # wedge s<-|t_coord| it is unchanged. No cluster labels are used.
        self.w1[0].weight[0] = t + 12*e
        self.w1[0].weight[1] = -t + 12*e
        self.w1[0].bias[:2] = -12*.85
        self.w2[0].weight[:, 0] = -.5*t
        self.w2[0].weight[:, 1] = .5*t
        # Block 2: a narrow continuous ramp across t_coord=2 adds +/-1.5
        # in z. It is -1.5 + 300[ReLU(t-2+.005)-ReLU(t-2-.005)].
        # Cluster 2 straddles this radial line; its two halves become
        # separated directions. All other cluster centers lie below it.
        # A constant hidden unit implements -1.5 without an output bias.
        eps = .005
        self.w1[1].weight[0] = t
        self.w1[1].weight[1] = t
        self.w1[1].bias[0] = -2 + eps
        self.w1[1].bias[1] = -2 - eps
        self.w1[1].bias[2] = 1
        self.w2[1].weight[2, :3] = torch.tensor([1.5/eps, -1.5/eps, -1.5])
        self.output.weight[:] = torch.tensor([[1., 0., -1.],
                                              [-.5, 1., 1.],
                                              [-.5, 1., -1.],
                                              [-1., -1., -.5]])

    def forward(self, x, retain=False):
        h = self.input(x)
        states, pre = [h], []
        for a, b in zip(self.w1, self.w2):
            z = a(h)
            if retain:
                z.retain_grad()
            pre.append(z)
            h = h + b(torch.relu(z))
            states.append(h)
        return self.output(h), states, pre


def kernel(h, beta):
    z = unit(h)
    return np.exp(beta * (np.clip(z @ z.T, -1, 1) - 1))


def sinkhorn(k, tol=1e-10, max_iter=10000):
    """Symmetric scaling, stopped on actual row sums (reference convention)."""
    v = np.ones(len(k))
    for iteration in range(max_iter):
        kv = k @ v
        error = np.max(np.abs(v * kv - 1))
        if error < tol * .25:
            p = k * v[:, None] * v[None, :]
            assert np.max(np.abs(p.sum(1) - 1)) < tol
            return p, iteration, float(error)
        v = np.sqrt(v / kv)
    raise RuntimeError(f'Sinkhorn failed: row error {error}')


def centered(p):
    return p - p.mean(0)[None, :] - p.mean(1)[:, None] + p.mean()


def signs(v):
    s = np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(v.shape[1])])
    return v * np.where(s == 0, 1, s)


def eigenbasis(p):
    # Householder restriction to 1^perp avoids choosing an arbitrary "constant"
    # inside the large numerical zero eigenspace of CPC. Still a dense eigh
    # of CPC in an orthonormal coordinate system, no approximate eigenpairs.
    n = len(p)
    a = np.ones(n) / np.sqrt(n)
    a[0] -= 1
    a /= np.linalg.norm(a)
    c = centered(p)
    ca = c @ a
    transformed = c - 2*np.outer(a, ca) - 2*np.outer(ca, a) + 4*(a@ca)*np.outer(a, a)
    values, vectors = np.linalg.eigh(transformed[1:, 1:])
    values, vectors = values[::-1], vectors[:, ::-1]
    v = np.vstack((np.zeros(n-1), vectors))
    v -= 2*np.outer(a, a@v)
    v = signs(v)
    return values, np.column_stack((np.ones(n)/np.sqrt(n), v))


def isolated_blocks(values, ratio=GAP_RATIO, max_dim=None):
    """Maximal runs satisfying the plan's relative-gap rule.

    Both existing external gaps must exceed every internal gap / ratio.
    The top of the CENTERED spectrum has no upper neighbour. A 1e-10
    numerical gap floor prevents numerical nullspace eigenvectors being
    called isolated. Missing bottom boundary is not an isolating gap:
    the unresolved tail is always an explicit complement.
    """
    gaps = np.maximum(0., values[:-1] - values[1:])
    candidates = []
    for start in range(len(gaps)):
        upper = gaps[start-1] if start else np.inf
        if upper <= 1e-10:
            continue
        # End is exclusive. For a singleton, the maximum internal gap is 0.
        internal = np.r_[0., np.maximum.accumulate(gaps[start:-1])]
        external = np.minimum(upper, gaps[start:])
        valid = np.flatnonzero((external > 1e-10) & (internal < ratio*external))
        if len(valid):
            end = start + int(valid[-1]) + 1
            candidates.append((start, end, float(external[valid[-1]])))
    # Remove runs properly contained in another valid run BEFORE applying
    # the display dimension cap; a small sub-run cannot stand in for an
    # actual isolated block of dimension >6.
    maximal, furthest = [], -1
    for block in candidates:
        if block[1] > furthest:
            maximal.append(block)
            furthest = block[1]
    return [b for b in maximal if max_dim is None or b[1]-b[0] <= max_dim]


def top_block(values):
    blocks = isolated_blocks(values)
    top = next((b for b in blocks if b[0] == 0), None)
    if top is None or top[1] > MAX_BLOCK:
        return (0, min(MAX_BLOCK, len(values)), 0.), False
    return top, True


def contrasts(labels, half):
    merge = np.where(labels == 0, 1., np.where(labels == 1, -1., 0.))
    split = np.where(labels == 2, np.where(half, 1., -1.), 0.)
    return {'merge': merge/np.linalg.norm(merge), 'split': split/np.linalg.norm(split)}


def gates(p0, p2, cs):
    out = {}
    for name, c in cs.items():
        n0, n2, nq = [float(np.linalg.norm(a)) for a in (p0@c, p2@c, p2@(p0@c))]
        passed = n0 > .9 and nq < .2 if name == 'merge' else n0 < .2 and n2 > .9
        out[name] = dict(p0_norm=n0, p2_norm=n2, product_norm=nq, realized=bool(passed))
    return out


def downsample(a, labels, size=120):
    order = np.argsort(labels, kind='stable')
    assert len(a) % size == 0
    width = len(a)//size
    return a[np.ix_(order, order)].reshape(size, width, size, width).mean((1, 3))


def varimax(v, iterations=100):
    r = np.eye(v.shape[1])
    for _ in range(iterations):
        w = v @ r
        u, _, vt = np.linalg.svd(v.T @ (w**3 - w @ np.diag((w*w).sum(0))/len(v)))
        new = u @ vt
        if np.linalg.norm(new-r) < 1e-10:
            r = new
            break
        r = new
    rotated = v@r
    s = np.sign(rotated[np.argmax(np.abs(rotated), axis=0), np.arange(v.shape[1])])
    r *= np.where(s == 0, 1, s)
    return v@r, r


def serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


class Timer:
    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.times = {}

    def stage(self, name):
        now = time.perf_counter()
        self.times[name] = now-self.last
        self.last = now
        print(f'{name:30s} {self.times[name]:8.3f} s', flush=True)


def groups_for(values):
    """All resolved leading maximal blocks, then one complete complement.

    Stop at the first unresolved interval: this keeps cumulative ranks true
    spectral prefixes, so truncation never cuts a block or skips a mode.
    """
    groups = [dict(name='constant', start=0, stop=1, dimension=1, gap=None)]
    cursor = 0
    for start, stop, gap in isolated_blocks(values):
        if start != cursor:
            break
        groups.append(dict(name=f'block_{len(groups)}', start=start+1,
                           stop=stop+1, dimension=stop-start, gap=gap))
        cursor = stop
    groups.append(dict(name='complement', start=cursor+1, stop=len(values)+1,
                       dimension=len(values)-cursor, gap=None))
    return groups


def gradient_data(model, x, target, psi, groups, display_count):
    model.zero_grad(set_to_none=True)
    logits, states, pre = model(torch.tensor(x), retain=True)
    loss = nn.functional.cross_entropy(logits, torch.tensor(target), reduction='mean')
    loss.backward()
    u = states[1].detach().numpy()
    delta = pre[1].grad.detach().numpy()
    grad = model.w1[1].weight.grad.detach().numpy()
    direct = delta.T @ u
    assert np.allclose(direct, grad, atol=1e-10, rtol=1e-10)
    b, g = psi.T @ u, psi.T @ delta
    pieces, pressures, ranks, residuals = [], [], [], []
    pre_coeff = psi.T @ pre[1].detach().numpy()
    post_coeff = psi.T @ torch.relu(pre[1]).detach().numpy()
    pre_energy, post_energy = [], []
    summed = np.zeros_like(grad)
    for group in groups:
        sl = slice(group['start'], group['stop'])
        piece = g[sl].T @ b[sl]
        pieces.append(piece)
        pressures.append(float(np.linalg.norm(piece)))
        summed += piece
        ranks.append(group['stop']-1)
        residuals.append(float(np.linalg.norm(grad-summed)))
        pre_energy.append((pre_coeff[sl]**2).sum(0))
        post_energy.append((post_coeff[sl]**2).sum(0))
    assert np.allclose(summed, grad, atol=1e-9, rtol=1e-10)
    assert np.allclose(psi @ b, u, atol=1e-10)
    assert np.allclose(psi @ g, delta, atol=1e-10)
    piece_matrix = np.asarray(pieces).reshape(len(pieces), -1)
    # A deliberately constructed single eigenfunction, not a learned neuron.
    rect_mode = groups[2]['start'] if len(groups) > 3 else 1
    before = psi[:, rect_mode]
    after = np.maximum(before, 0)
    coeff_before, coeff_after = psi.T @ before, psi.T @ after
    assert np.allclose(psi@coeff_after, after, atol=1e-10)
    rect_pre, rect_post = [], []
    for group in groups:
        sl = slice(group['start'], group['stop'])
        rect_pre.append(float(np.sum(coeff_before[sl]**2)))
        rect_post.append(float(np.sum(coeff_after[sl]**2)))
    assert np.isclose(sum(rect_post), after@after, atol=1e-10)
    # Only show learned pre/post pairs when rectification actually increases
    # block participation, rather than implying all neurons exhibit it.
    pe, qe = np.array(pre_energy).T, np.array(post_energy).T
    def participation(a):
        return np.divide(a.sum(1)**2, (a*a).sum(1),
                         out=np.zeros(len(a)), where=(a*a).sum(1)>0)
    effect = np.flatnonzero(participation(qe) > participation(pe)+.25)
    return dict(loss=float(loss.detach()), groups=groups,
                pressures=np.array(pressures), interactions=piece_matrix@piece_matrix.T,
                truncation=dict(r=np.array(ranks), residual_fro=np.array(residuals)),
                input_loadings=b[:display_count+1], error_loadings=g[:display_count+1],
                gradient=grad, pre_block_energy=pe, post_block_energy=qe,
                learned_effect_neurons=effect,
                rectification=dict(mode_index=rect_mode, before=before, after=after,
                                   before_block_energy=np.array(rect_pre),
                                   after_block_energy=np.array(rect_post)),
                checks=dict(autograd_max_error=float(np.max(np.abs(direct-grad))),
                            block_sum_max_error=float(np.max(np.abs(summed-grad))),
                            input_reconstruction_max_error=float(np.max(np.abs(psi@b-u)))))


def loading_data(h, psi, values, groups, block, isolated):
    start, stop, gap = block
    v = psi[:, start+1:stop+1]
    a = psi.T @ h
    rotated, rotation = varimax(v)
    rotated_loadings = rotation.T @ a[start+1:stop+1]
    assert np.allclose(rotated@rotated_loadings, v@a[start+1:stop+1], atol=1e-10)
    assert np.allclose(psi@a, h, atol=1e-10)
    r, residuals, tail_sums = [], [], []
    for group in groups:
        end = group['stop']
        residual = float(np.linalg.norm(h-psi[:, :end]@a[:end])**2)
        tail = float(np.sum(a[end:]**2))
        assert np.isclose(residual, tail, atol=1e-8, rtol=1e-9)
        r.append(end-1)
        residuals.append(residual)
        tail_sums.append(tail)
    return dict(mode_indices=np.arange(start+1, stop+1), eigenvalues=values[start:stop],
                modes=v, loadings=a[start+1:stop+1], constant_mode=psi[:, 0],
                constant_loading=a[0], isolated=isolated, gap=gap,
                basis_label='isolated block eigenfunctions' if isolated else 'no isolated block: top-6 subspace',
                rotated_modes=rotated, rotated_loadings=rotated_loadings,
                rotation=rotation, rotated_label='rotated block basis (not eigenfunctions)',
                reconstruction=dict(r=np.array(r), residual_squared=np.array(residuals),
                                    tail_sum=np.array(tail_sums)),
                reconstruction_max_error=float(np.max(np.abs(psi@a-h))))


def nearest_beta(h, k=10):
    z = unit(h)
    d2 = np.maximum(0, 2-2*(z@z.T))
    np.fill_diagonal(d2, np.inf)
    median = float(np.median(np.partition(d2, k-1, axis=1)[:, k-1]))
    return 1/median if median else None, median


def noise_scale(model, optimizer, x, target, generator):
    """Eight one-step resamples, each restoring parameters AND Adam state."""
    model_state = copy.deepcopy(model.state_dict())
    adam_state = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        _, initial, _ = model(x)
        initial = [unit(h.numpy()) for h in initial]
    displacements = []
    for _ in range(8):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(adam_state))
        ix = torch.randint(len(x), (128,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(x[ix])
        nn.functional.cross_entropy(logits, target[ix]).backward()
        optimizer.step()
        with torch.no_grad():
            _, hs, _ = model(x)
        displacements.append(np.stack([unit(h.numpy())-base for h, base in zip(hs, initial)]))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(adam_state)
    d = np.stack(displacements)
    d -= d.mean(0, keepdims=True)  # remove mean stochastic displacement per site
    # Mean squared distance over distinct pairs of resamples. Factor 2n/(n-1)
    # converts the centered second moment to pairwise differences.
    pairwise = 2*len(d)/(len(d)-1) * np.mean(np.sum(d*d, axis=-1), axis=(0, 2))
    return dict(beta_by_layer=np.divide(1., pairwise).tolist(),
                pairwise_squared_by_layer=pairwise.tolist(), resamples=8)


def train_variant(x, labels, half, target, beta, timer):
    torch.manual_seed(1)
    model = Toy(planted=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(1)
    noise_generator = torch.Generator().manual_seed(2)
    xt, yt = torch.tensor(x), torch.tensor(target)
    dense = []  # states are saved every ten steps, but never exported as weights
    noise, losses, step150 = [], [], None
    for step in range(1, 301):
        ix = torch.randint(len(x), (128,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(xt[ix])
        loss = nn.functional.cross_entropy(logits, yt[ix])
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            dense.append((step, copy.deepcopy(model.state_dict())))
            losses.append([step, float(loss.detach())])
        if step in (50, 150, 300):
            noise.append(dict(step=step, **noise_scale(model, optimizer, xt, yt, noise_generator)))
        if step == 150:
            step150 = copy.deepcopy(model.state_dict())
    # Checkpoint slider/projector matching is explicitly deferred. Save dense
    # checkpoints in memory for training provenance; only endpoints needed by
    # first-pass panels are analysed (never an unrequested 30-checkpoint scan).
    with torch.no_grad():
        logits, hs, _ = model(xt)
        hs = [h.numpy() for h in hs]
        accuracy = float((logits.argmax(1)==yt).double().mean())
    p0, _, _ = sinkhorn(kernel(hs[0], beta))
    p2, _, _ = sinkhorn(kernel(hs[2], beta))
    verdicts = gates(p0, p2, contrasts(labels, half))
    beta_nn, distance = nearest_beta(hs[2])
    model.load_state_dict(step150)
    with torch.no_grad():
        _, h150, _ = model(xt)
    p150, _, _ = sinkhorn(kernel(h150[2].numpy(), beta))
    val, psi = eigenbasis(p150)
    block, isolated = top_block(val)
    gradient = gradient_data(model, x, target, psi, groups_for(val), block[1])
    timer.stage('optional_training_and_analysis')
    return dict(seed=1, noise_seed=2, steps=300, learning_rate=.01, batch_size=128,
                checkpoint_steps=[s for s, _ in dense], losses=np.array(losses),
                noise_marks=noise, display_beta=beta, beta_nn=beta_nn,
                nn_median_squared_distance=distance, gates=verdicts,
                final_accuracy=accuracy, raw=np.stack(hs), normalized=np.stack([unit(h) for h in hs]),
                gradient_step150=gradient,
                gradient_layer1_normalized_step150=unit(h150[1].numpy()),
                gradient_basis_modes=psi[:, :block[1]+1],
                gradient_basis_eigenvalues=val[:block[1]], gradient_basis_isolated=isolated)


def generate(train=False, output=None):
    timer = Timer()
    torch.manual_seed(SEED)
    x, labels, half, target = sites()
    model = Toy()
    with torch.no_grad():
        _, states, _ = model(torch.tensor(x))
    hs = [h.numpy() for h in states]
    timer.stage('sites_and_planted_model')
    indicator = np.eye(6)[labels] / np.sqrt(np.bincount(labels))[None, :]
    scan, cached = [], []
    for beta in BETAS:
        k = kernel(hs[2], beta)
        p, iterations, error = sinkhorn(k)
        values, psi = eigenbasis(p)
        block, isolated = top_block(values)
        start, stop, gap = block
        angles = np.arccos(np.clip(np.linalg.svd(psi[:, start+1:stop+1].T@indicator,
                                               compute_uv=False), 0, 1))
        scan.append(dict(beta=float(beta), eigenvalues=values[:12],
                         block_dimension=stop-start, isolated=isolated, gap=gap,
                         principal_angles_radians=angles, sinkhorn_iterations=iterations,
                         sinkhorn_row_error=error))
        # Only twelve operators (~140MB) and selected eigenpairs are cached;
        # the complete eigenbasis is recomputed once at the display beta.
        cached.append((k, p))
    display_index = int(np.argmax([s['gap'] if s['isolated'] else -1 for s in scan]))
    beta = float(BETAS[display_index])
    k2, p2 = cached[display_index]
    values, psi = eigenbasis(p2)
    block, isolated = top_block(values)
    # Low, selected display, high: all three are actual scan temperatures.
    illustrative_indices = sorted(set([0, display_index, len(BETAS)-1]))
    if len(illustrative_indices) < 3:
        illustrative_indices = sorted(set(illustrative_indices + [len(BETAS)//2]))
    heatmaps = [dict(beta=float(BETAS[i]), kernel=downsample(cached[i][0], labels),
                     operator=downsample(cached[i][1], labels)) for i in illustrative_indices]
    beta_nn, nn_distance = nearest_beta(hs[2])
    timer.stage('layer2_scan_and_three_heatmaps')
    k0 = kernel(hs[0], beta)
    p0, it0, error0 = sinkhorn(k0)
    p1, it1, error1 = sinkhorn(kernel(hs[1], beta))
    cs = contrasts(labels, half)
    verdicts = gates(p0, p2, cs)
    assert all(v['realized'] for v in verdicts.values()), verdicts
    q = p2 @ p0
    # Exact dense SVD; center to remove the common constant singular pair.
    left, singular, right = np.linalg.svd(centered(q), full_matrices=False)
    right = right.T
    sg = np.sign(right[np.argmax(np.abs(right[:, :3]), axis=0), np.arange(3)])
    left, right = left[:, :3]*sg, right[:, :3]*sg
    top_sv = float(np.linalg.svd(q, compute_uv=False)[0])
    def degree(k):
        d = k.sum(1)
        return k / np.sqrt(d[:, None]*d[None, :])
    degree_sv = float(np.linalg.svd(degree(k2)@degree(k0), compute_uv=False)[0])
    assert abs(top_sv-1) < 1e-9
    assert degree_sv < 1
    assert np.allclose(q@right, left*singular[:3], atol=1e-9)
    directional = {name: dict(contrast=c, p0_contrast=p0@c,
                              p2_contrast=p2@c, product_contrast=q@c,
                              **verdicts[name]) for name, c in cs.items()}
    timer.stage('coherence_gates_and_normalization')
    groups = groups_for(values)
    loadings = loading_data(hs[2], psi, values, groups, block, isolated)
    gradient = gradient_data(model, x, target, psi, groups, block[1])
    timer.stage('loadings_and_gradient_identities')
    data = dict(
        meta=dict(schema_version='1.0', N=len(x), seeds=dict(data=0, planted=0, training=1, noise=2),
                  model='planted', dtype='float64', threads=1,
                  temperatures=dict(scan=BETAS, display=beta, display_index=display_index,
                                    illustrative=BETAS[illustrative_indices], beta_nn=beta_nn,
                                    nn_k=10, nn_median_squared_distance=nn_distance,
                                    selection='largest isolated top-block boundary gap at layer 2; scan-index tie break'),
                  block_rule=dict(gap_ratio=GAP_RATIO, display_max_dimension=MAX_BLOCK,
                                  numerical_gap_floor=1e-10,
                                  rule='maximal runs: max internal gap < 0.1 times each existing neighbouring gap; top centered boundary is absent; unresolved tail is complement'),
                  gates=verdicts, timings_seconds=timer.times,
                  sinkhorn=dict(tolerance=1e-10, layer0_error=error0, layer1_error=error1,
                                layer0_iterations=it0, layer1_iterations=it1),
                  assertions=dict(sinkhorn_product_top_sv=top_sv, degree_product_top_sv=degree_sv,
                                  exact_reconstruction=True, autograd_gradient=True,
                                  complete_gradient_sum=True, gates_passed=True),
                  construction=dict(merge_wedge_slope=12., merge_wedge_threshold=.85,
                                    split_line_t=2., split_ramp_halfwidth=.005, split_amplitude=1.5,
                                    input_z=.1),
                  notes=[
                      'Planted geometry, not a discovered or learned mechanism.',
                      'Labels define validation contrasts and loss only, never kernels or bases.',
                      'At beta infinity identity requires distinct normalized vectors; repeated directions leave averaging blocks.',
                      'Counting measure; loadings of raw activations, not unit directions; rank at most 3.',
                      'Block profiles and gradient pieces use the layer-2 display-temperature eigenbasis.',
                      'Split contrast uses +/-1/sqrt(200); unequal halves are not silently recentered.',
                      'Scan eigenvalues are sorted at each beta; individual modes are not tracked across crossings.',
                      'Deferred slider, second seed, output view, and attention toy are omitted.']),
        sites=dict(input=x, cluster=labels, split_half=half, target=target,
                   raw=np.stack(hs), normalized=np.stack([unit(h) for h in hs]),
                   norms=np.stack([np.linalg.norm(h, axis=1) for h in hs])),
        scale=dict(scan=scan, heatmaps=heatmaps,
                   heatmap_cluster_order=np.repeat(np.arange(6), 20),
                   diffusion=dict(times=np.array([1, 4, 16]),
                                  eigenvalues=values[:12][None, :]**np.array([1, 4, 16])[:, None]),
                   normalization=dict(sinkhorn_product_top_sv=top_sv, degree_product_top_sv=degree_sv)),
        coherence=dict(beta=beta, singular_values=singular[:3], left_modes=left,
                       right_modes=right, directional=directional),
        loadings=loadings, gradient=gradient)
    if train:
        data['trained'] = train_variant(x, labels, half, target, beta, timer)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        attach_manifest(data)
        # Time serialization, then record it in the final serialized output.
        text = json.dumps(data, default=serial, separators=(',', ':'), allow_nan=False)
        output.write_text(text+'\n')
        timer.stage('json_serialization_and_write')
        timer.times['total'] = time.perf_counter()-timer.start
        data['meta']['budget_seconds'] = 60
        data['meta']['within_budget'] = timer.times['total'] < 60
        attach_manifest(data)
        output.write_text(json.dumps(data, default=serial, separators=(',', ':'), allow_nan=False)+'\n')
        write_schema(data, output.with_name('SCHEMA.md'))
        timer.stage('schema_and_final_metadata')
        timer.times['total'] = time.perf_counter()-timer.start
        data['meta']['within_budget'] = timer.times['total'] < 60
        attach_manifest(data)
        output.write_text(json.dumps(data, default=serial, separators=(',', ':'), allow_nan=False)+'\n')
        write_schema(data, output.with_name('SCHEMA.md'))
    else:
        timer.times['total'] = time.perf_counter()-timer.start
    print('display beta:', beta, 'illustrative:', BETAS[illustrative_indices].tolist(), flush=True)
    print('gates:', json.dumps(verdicts), flush=True)
    print(f"total measured computation/export: {timer.times['total']:.3f} s", flush=True)
    if not train:
        assert timer.times['total'] < 60, 'Default computation exceeded approved 60 s budget'
    return data

# Field-level descriptions are emitted both into the JSON manifest and SCHEMA.
# [] denotes an array of records; it never denotes an omitted matrix axis.
DESCRIPTIONS = {
    'meta': 'Provenance, numerical conventions, temperatures, gates, checks, and timing.',
    'meta.schema_version': 'Version of this data contract.',
    'meta.N': 'Number of common sites, 1200 in the default run.',
    'meta.seeds': 'Independent reproducibility seeds; unused optional seeds remain documented.',
    'meta.seeds.data': 'NumPy generator seed for Gaussian input draws.',
    'meta.seeds.planted': 'Torch seed before constructing deterministic planted weights.',
    'meta.seeds.training': 'Torch initialization and minibatch generator seed for optional training.',
    'meta.seeds.noise': 'Separate generator seed for optional stochastic resamples.',
    'meta.model': 'Model shown by the main five sections: planted.',
    'meta.dtype': 'Numerical dtype used by NumPy and Torch.',
    'meta.threads': 'BLAS/Torch CPU thread count.',
    'meta.temperatures': 'Temperature choices; all main panels use the display value.',
    'meta.temperatures.scan': 'Twelve log-spaced layer-2 scan beta values, in increasing order.',
    'meta.temperatures.display': 'Scan beta maximizing the isolated top-block boundary gap.',
    'meta.temperatures.display_index': 'Zero-based selected entry in the beta scan.',
    'meta.temperatures.illustrative': 'Three actual scan betas used for kernel/operator heatmaps.',
    'meta.temperatures.beta_nn': 'Layer-2 1/median squared tenth-neighbour chordal distance; null if distance is zero.',
    'meta.temperatures.nn_k': 'Nearest-neighbour rank, excluding the site itself.',
    'meta.temperatures.nn_median_squared_distance': 'Median squared chordal distance used for beta_nn.',
    'meta.temperatures.selection': 'Declared display-only selection rule, not an inferred physical temperature.',
    'meta.block_rule': 'Operational interpretation of the approved relative-gap rule.',
    'meta.block_rule.gap_ratio': 'Internal gaps must be strictly below this fraction of each existing boundary gap.',
    'meta.block_rule.display_max_dimension': 'Largest isolated top block whose individual modes are displayed.',
    'meta.block_rule.numerical_gap_floor': 'Gaps below this absolute numerical threshold cannot establish isolation.',
    'meta.block_rule.rule': 'Maximality, boundary convention, and complement policy in words.',
    'meta.gates': 'Acceptance-gate numbers for the planted model at display beta.',
    'meta.timings_seconds': 'Wall-clock duration per named compute/export stage, excluding dependency installation and module imports.',
    'meta.sinkhorn': 'Actual convergence checks for layers 0 and 1; layer-2 checks occur in scale.scan.',
    'meta.sinkhorn.tolerance': 'Required maximum absolute row-sum error.',
    'meta.assertions': 'Summary of checks that raise AssertionError if violated.',
    'meta.assertions.exact_reconstruction': 'Complete constant-plus-nonconstant basis reconstructs raw activations.',
    'meta.assertions.autograd_gradient': 'Delta transpose U agrees with Torch mean-loss W1 gradient.',
    'meta.assertions.complete_gradient_sum': 'Constant, all resolved blocks, and complement sum to the gradient.',
    'meta.assertions.gates_passed': 'Both planted gate inequalities hold at display beta.',
    'meta.construction': 'Constants defining the constructed residual network, not learned fitting parameters.',
    'meta.construction.merge_wedge_slope': 'Slope in s=12(q-0.85).',
    'meta.construction.merge_wedge_threshold': 'q offset in the merge wedge.',
    'meta.construction.split_line_t': 'Fixed radial split threshold t=2.',
    'meta.construction.split_ramp_halfwidth': 'Halfwidth of the continuous piecewise-linear split ramp.',
    'meta.construction.split_amplitude': 'Magnitude of the added positive/negative z plateau.',
    'meta.construction.input_z': 'Constant third coordinate of h0.',
    'meta.notes': 'Caption caveats and conventions that must not be silently dropped.',
    'meta.budget_seconds': 'Approved default compute budget.',
    'meta.within_budget': 'Whether the measured run was below the default budget; optional training is not budget-asserted.',
    'meta.key_manifest': 'Exact field inventory: records with path, shape, meaning. Self entry does not recurse.',
    'sites': 'Figure 1: same ordered sites at each layer, no scatter downsampling.',
    'sites.input': 'Original two-dimensional Gaussian inputs, site by xy coordinate.',
    'sites.cluster': 'Planted cluster IDs 0 through 5, validation/colouring only.',
    'sites.split_half': 'True iff original input lies on the positive side of t=2; only cluster 2 defines the split contrast.',
    'sites.target': 'Four cross-entropy labels: 0/1->A=0; cluster2 positive->B=1, negative->C=2; 3/4->D=3; 5->B=1.',
    'sites.raw': 'Raw h0,h1,h2: axes [layer,site,xyz].',
    'sites.normalized': 'Row-normalized h0,h1,h2 on S2: axes [layer,site,xyz].',
    'sites.norms': 'Dropped Euclidean activation norms: axes [layer,site], for point size.',
    'scale': 'Figure 2: beta scan, downsampled heatmaps, diffusion spectra, and normalization comparison.',
    'scale.scan': 'One record per beta, increasing beta; sorted eigenvalues, not mode matching across crossings.',
    'scale.scan[].beta': 'Kernel inverse bandwidth beta at this scan entry.',
    'scale.scan[].eigenvalues': 'Leading twelve NONCONSTANT eigenvalues of CPC, descending, not clipped.',
    'scale.scan[].block_dimension': 'Dimension of the isolated top block, or six-dimensional fallback.',
    'scale.scan[].isolated': 'False means label the panel no isolated block: top-6 subspace.',
    'scale.scan[].gap': 'Minimum existing boundary gap of the isolated top block; zero for fallback.',
    'scale.scan[].principal_angles_radians': 'Principal angles to the six uncentered normalized planted indicator columns; size min(block dimension,6).',
    'scale.scan[].sinkhorn_iterations': 'Number of symmetric scaling updates before row-sum convergence.',
    'scale.scan[].sinkhorn_row_error': 'Actual maximum absolute row-sum deviation from one.',
    'scale.heatmaps': 'Three layer-2 kernel/operator pairs, low/display/high scan values.',
    'scale.heatmaps[].beta': 'Temperature of this pair.',
    'scale.heatmaps[].kernel': 'Shifted exp(beta(dot-1)) kernel, stable cluster-sorted 10-by-10 block averages; axes [row-bin,column-bin].',
    'scale.heatmaps[].operator': 'Doubly stochastic P, same bin averages; averaged rows sum to 0.1, NOT 1 (not a re-normalized quotient operator).',
    'scale.heatmap_cluster_order': 'Cluster ID of each of the 120 heatmap bins; 20 bins per cluster.',
    'scale.diffusion': 'Display-temperature leading twelve eigenvalues raised to diffusion time.',
    'scale.diffusion.times': 'Diffusion exponents [1,4,16].',
    'scale.diffusion.eigenvalues': 'Axes [diffusion-time index,nonconstant mode], lambda_i^t.',
    'scale.normalization': 'Uncentered layer product P2 P0: exact dense top singular values.',
    'coherence': 'Figure 3: preserved singular pairs and directional validation contrasts.',
    'coherence.beta': 'Common display beta for both layers.',
    'coherence.singular_values': 'Top three NONTRIVIAL singular values of centered P2 P0.',
    'coherence.left_modes': 'Layer-2 left singular functions u, axes [site,pair], Euclidean unit norm.',
    'coherence.right_modes': 'Layer-0 right singular functions v, axes [site,pair]; largest-absolute site of v is positive; same sign applied to u.',
    'coherence.directional': 'Separate merge and split contrast records; these are validation, not discovered modes.',
    'loadings': 'Figure 4: layer-2 raw-activation loadings, eigenfunctions, rotated basis, and exact reconstruction tails.',
    'loadings.mode_indices': 'One-based nonconstant complete-basis indices of displayed block, with 0 reserved for constant.',
    'loadings.eigenvalues': 'Eigenvalues associated with displayed actual eigenfunctions.',
    'loadings.modes': 'Displayed eigenfunctions, axes [site,mode], signs fixed at largest-absolute site.',
    'loadings.loadings': 'Raw H2 transpose psi coefficients as rows, axes [mode,xyz]; not normalized or orthogonal arrows.',
    'loadings.constant_mode': 'Exactly 1/sqrt(N) at every site.',
    'loadings.constant_loading': 'Raw H2 transpose 1/sqrt(N), xyz.',
    'loadings.isolated': 'Whether displayed modes form an isolated block rather than fallback top-six subspace.',
    'loadings.gap': 'Existing-neighbour boundary gap; zero when fallback.',
    'loadings.basis_label': 'Required label distinguishing isolated eigenfunctions from no-isolated-block fallback.',
    'loadings.rotated_modes': 'Varimax rotated displayed basis, axes [site,basis-vector]; NOT eigenfunctions.',
    'loadings.rotated_loadings': 'Consistently rotated raw coefficients, axes [basis-vector,xyz].',
    'loadings.rotation': 'Orthogonal right rotation R: rotated_modes=modes R and rotated_loadings=R transpose loadings.',
    'loadings.rotated_label': 'Mandatory warning that this rotated basis is not eigenfunctions.',
    'loadings.reconstruction': 'Constant included at every truncation; cumulative ranks are only complete block boundaries, ending at complete complement.',
    'loadings.reconstruction.r': 'Number of nonconstant modes included; r=0 includes constant, final r=N-1 is exact.',
    'loadings.reconstruction.residual_squared': 'Direct squared Frobenius error ||H2-Hr||^2.',
    'loadings.reconstruction.tail_sum': 'Sum of squared omitted loading coefficients, asserted equal to direct residual.',
    'loadings.reconstruction_max_error': 'Maximum entrywise reconstruction error in the complete eigenbasis.',
    'gradient': 'Figure 5: block-2 W1 cross-entropy gradient, using the layer-2 eigenbasis and h1 inputs; not an optimizer update.',
    'gradient.loss': 'Mean-over-all-sites cross-entropy at the model weights.',
    'gradient.groups': 'Disjoint contiguous complete-basis groups, constant first and complement last; jointly cover all N modes.',
    'gradient.groups[].name': 'constant, block_j, or complement.',
    'gradient.groups[].start': 'Inclusive index in the complete basis [constant, descending nonconstant eigenfunctions].',
    'gradient.groups[].stop': 'Exclusive index in the complete basis.',
    'gradient.groups[].dimension': 'Number of basis functions in this group.',
    'gradient.groups[].gap': 'Minimum existing boundary gap for isolated blocks; null for constant/complement.',
    'gradient.pressures': '||G_B||_F per group, including constant and complement.',
    'gradient.interactions': 'Signed Frobenius products <G_B,G_Bprime>, axes [group,group].',
    'gradient.truncation': 'Direct gradient residual, never a sum of per-mode norms.',
    'gradient.truncation.r': 'Cumulative nonconstant-mode ranks at complete group boundaries.',
    'gradient.truncation.residual_fro': '||G-G_r||_F computed directly from the sum of retained gradient matrices.',
    'gradient.input_loadings': 'b_i=U transpose psi_i, constant then displayed modes; axes [mode,xyz], U=h1.',
    'gradient.error_loadings': 'g_i=Delta transpose psi_i, constant then displayed modes; axes [mode,16 hidden neurons]; Delta includes loss factor 1/N.',
    'gradient.gradient': 'Torch W1 gradient matrix, axes [16 hidden neurons,3 stream inputs].',
    'gradient.pre_block_energy': 'Squared pre-ReLU profile coefficients summed within groups, axes [16 neurons,group].',
    'gradient.post_block_energy': 'Squared post-ReLU profile coefficients summed within groups, axes [16 neurons,group].',
    'gradient.learned_effect_neurons': 'Zero-based neuron indices for which rectification raises inverse-participation block count by >0.25; show paired learned profiles only for these.',
    'gradient.rectification': 'CONSTRUCTED inset: one eigenfunction and its ReLU, expanded in the SAME complete basis.',
    'gradient.rectification.mode_index': 'Complete-basis index of the constructed eigenfunction: first mode of the second resolved block, or mode1 if none.',
    'gradient.rectification.before': 'The chosen eigenfunction at all sites.',
    'gradient.rectification.after': 'Pointwise max(before,0).',
    'gradient.rectification.before_block_energy': 'Squared expansion coefficients summed over each complete group before rectification.',
    'gradient.rectification.after_block_energy': 'Squared expansion coefficients summed over each complete group after rectification, including constant/complement.',
    'gradient.checks': 'Maximum absolute numerical errors of asserted identities.',
    'gradient.checks.autograd_max_error': 'max abs(Delta transpose U - autograd W1 gradient).',
    'gradient.checks.block_sum_max_error': 'max abs(sum_B G_B - autograd W1 gradient).',
    'gradient.checks.input_reconstruction_max_error': 'max abs(Psi Psi transpose U-U).',
    'trained': 'Optional first-pass trained variant; absent by default. Gates reported honestly without assertion.',
    'trained.seed': 'Initialization and minibatch generator seed.',
    'trained.noise_seed': 'Independent noise-resample generator seed.',
    'trained.steps': 'Adam update count.',
    'trained.learning_rate': 'Adam learning rate.',
    'trained.batch_size': 'Minibatch size sampled with replacement.',
    'trained.checkpoint_steps': 'Steps saved in memory every ten updates; no deferred slider data exported.',
    'trained.losses': 'Axes [checkpoint,2], columns step and minibatch mean cross-entropy.',
    'trained.noise_marks': 'Noise scales at steps 50,150,300; each resample restores model AND Adam state.',
    'trained.noise_marks[].step': 'Checkpoint around which the eight stochastic one-step resamples were taken.',
    'trained.noise_marks[].beta_by_layer': 'Inverse mean pairwise squared chordal noise for layers 0,1,2; plot layer2 value on beta scan.',
    'trained.noise_marks[].pairwise_squared_by_layer': 'Mean squared distance between distinct resample displacements, after removing mean displacement; layers 0,1,2.',
    'trained.noise_marks[].resamples': 'Number of independent one-step resamples, eight.',
    'trained.display_beta': 'Same planted-selected beta held fixed for trained diagnostics; not re-selected.',
    'trained.beta_nn': 'Final trained layer-2 nearest-neighbour temperature mark.',
    'trained.nn_median_squared_distance': 'Final trained layer-2 median squared tenth-neighbour distance.',
    'trained.gates': 'Final step300 validation gates, may be false; never caption an unrealized change as realized.',
    'trained.final_accuracy': 'Full-site accuracy at step300.',
    'trained.raw': 'Final h0,h1,h2, axes [layer,site,xyz].',
    'trained.normalized': 'Final normalized clouds, axes [layer,site,xyz].',
    'trained.gradient_step150': 'Same contract as gradient, evaluated at trained step150 using its own layer-2 eigenbasis at fixed display beta.',
    'trained.gradient_layer1_normalized_step150': 'Step150 layer1 unit-sphere cloud, axes [site,xyz], for the step150 gradient loading panel; DO NOT use the final step300 cloud.',
    'trained.gradient_basis_modes': 'Step150 constant and displayed eigenfunctions, axes [site,mode], accompanying trained gradient loadings.',
    'trained.gradient_basis_eigenvalues': 'Step150 nonconstant eigenvalues of displayed modes.',
    'trained.gradient_basis_isolated': 'Whether step150 modes are an isolated block or a labelled fallback.',
}

COMMON_DESCRIPTIONS = {
    'merge': 'Validation contrast between planted clusters 0 and 1.',
    'split': 'Validation contrast between two radial halves of cluster 2.',
    'p0_norm': 'Euclidean norm ||P0 c|| at the declared common beta.',
    'p2_norm': 'Euclidean norm ||P2 c|| at the declared common beta.',
    'product_norm': 'Euclidean norm ||P2 P0 c||.',
    'realized': 'Merge: p0_norm>0.9 AND product_norm<0.2. Split: p0_norm<0.2 AND p2_norm>0.9.',
    'contrast': 'c: +/-1/sqrt(support size), zero outside; uncentered so the stated planted contrast is unchanged.',
    'p0_contrast': 'P0 c on sites.',
    'p2_contrast': 'P2 c on sites.',
    'product_contrast': 'P2 P0 c on sites.',
    'sinkhorn_product_top_sv': 'Top uncentered singular value of P2 P0; asserted within 1e-9 of one.',
    'degree_product_top_sv': 'Top singular value of separately degree-normalized layer2 times layer0 kernel; asserted strictly below one.',
    'layer0_error': 'Actual maximum absolute layer0 row-sum error at display beta.',
    'layer1_error': 'Actual maximum absolute layer1 row-sum error at display beta.',
    'layer0_iterations': 'Symmetric scaling update count for layer0.',
    'layer1_iterations': 'Symmetric scaling update count for layer1.',
}


def description(path):
    alias = path.replace('trained.gradient_step150', 'gradient')
    if alias in DESCRIPTIONS:
        return DESCRIPTIONS[alias]
    if path.startswith('meta.timings_seconds.'):
        return ('Measured total of compute/export stages in seconds.' if path.endswith('.total')
                else 'Wall seconds for stage '+path.rsplit('.', 1)[-1]+'.')
    leaf = path.rsplit('.', 1)[-1]
    if leaf in COMMON_DESCRIPTIONS:
        return COMMON_DESCRIPTIONS[leaf]
    raise KeyError('Missing schema description: '+path)


def inventory(obj, prefix=''):
    entries = []
    for key, value in obj.items():
        path = prefix+'.'+key if prefix else key
        if path == 'meta.key_manifest':
            shape = 'list of {path:string, shape:string, meaning:string}; self-described, not recursively inventoried'
        elif isinstance(value, dict):
            shape = 'object'
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            shape = f'[{len(value)}] records (nested [] shapes may vary per record)'
        elif isinstance(value, (list, tuple, np.ndarray)):
            shape = str(list(np.shape(value)))
        elif value is None:
            shape = 'null'
        else:
            shape = type(value).__name__
        entries.append(dict(path=path, shape=shape, meaning=description(path)))
        if isinstance(value, dict):
            entries += inventory(value, path)
        elif (isinstance(value, (list, tuple)) and value and isinstance(value[0], dict)
              and path != 'meta.key_manifest'):
            # Union fields across records; annotate all observed shapes, not
            # just the first block/scan record's variable mode dimension.
            by_path = {}
            for record in value:
                for entry in inventory(record, path+'[]'):
                    old = by_path.get(entry['path'])
                    if old and entry['shape'] not in old['shape'].split(' or '):
                        old['shape'] += ' or '+entry['shape']
                    elif old is None:
                        by_path[entry['path']] = entry
            entries += list(by_path.values())
    return entries


def attach_manifest(data):
    data['meta']['key_manifest'] = []
    data['meta']['key_manifest'] = inventory(data)


def write_schema(data, path):
    header = '''# Toy visualization JSON contract

Generated by `viz/toy.py`; data only, for the page builder. Run from the repository root:

```sh
uv run --with numpy --with torch python -I viz/toy.py
uv run --with numpy --with torch --with pytest python -I -m pytest viz/test_toy.py
# Optional, adds a trained-model panel (overwrites the chosen output):
uv run --with numpy --with torch python -I viz/toy.py --train
```

## Global conventions and required caption caveats

- All arrays are row-major JSON arrays, float64 computation. N=1200; site order is
  fixed across all layers, cluster 0's 200 sites first, then 1 through 5. No site
  scatter or mode-vector downsampling. Layers are 0,1,2; xyz coordinates have length 3.
- Euclidean/counting-measure basis normalization. Complete-basis index 0 is the
  exact constant 1/sqrt(N); indices 1..N-1 are actual descending eigenfunctions of
  CPC. We compute dense `eigh` in the exact Householder coordinates of 1-perp,
  avoiding arbitrary mixing of the constant with the numerical nullspace.
- Kernel is exp(beta(dot-1)); Sinkhorn is symmetric scaling stopped on row errors
  below 1e-10, not a fixed iteration count. Beta scan is primary. The displayed
  temperature maximizes the boundary gap of the isolated TOP block, with first
  scan index breaking ties. It is a display choice, not a noise estimate.
- Block rule: a maximal run whose largest internal gap is strictly below 0.1 of
  BOTH existing external gaps. The top of the centered spectrum has no upper
  neighbour; singleton runs have no internal gap. An absolute 1e-10 gap floor
  suppresses numerical nullspace isolation. Maximality is tested BEFORE applying
  the dimension <=6 display limit. No qualifying top block means explicitly
  "no isolated block: top-6 subspace". All resolved leading blocks are retained
  in order; at the first unresolved interval, the entire remaining complete
  basis is the explicit complement. Truncation curves never cut through blocks.
- Rotated basis is varimax, labelled **rotated block basis (not eigenfunctions)**.
  Loadings are raw-activation coefficients, not orthogonal directions; rank <=3.
  Constant + blocks + complement reconstruct exactly. Gradient is of the actual
  mean cross-entropy at block-2 W1; it is NOT an Adam update. Input U=h1; delta
  already includes 1/N. The SAME layer-2 basis is used throughout figures 4 and 5.
- Contrast gates are unchanged: merge ||P0c||>0.9 and ||P2P0c||<0.2; split
  ||P0c||<0.2 and ||P2c||>0.9. Contrasts are +/-1/sqrt(support size), not silently
  recentered or balanced. Cluster2 half counts need not match; retain its constant
  component. Labels enter validation/loss/ordering only, never the spectral method.
- Identity-limit captions MUST say "for distinct normalized vectors; repeated
  directions leave averaging blocks". Singular pairs show preservation, not
  direction of splitting/merging; the contrasts are a separate validation panel.
- Matrix heatmaps are 120x120 arithmetic block averages after stable cluster
  sorting, 10 sites per bin. P heatmap row sums are 0.1, not 1: do NOT interpret
  the display as a newly normalized 120-site operator. Interaction and unit-profile
  heatmaps have their own group/neuron axes and are not site-space heatmaps.
- The rectification inset is CONSTRUCTED from an eigenfunction, not learned.
  Learned pre/post comparisons are shown only for `learned_effect_neurons`.
- Optional training uses a random initialization of the same architecture,
  Adam lr .01, 300 steps, batch128. Dense checkpoints every10 are kept in memory.
  Only step150 gradient and step300 gate endpoints are analysed in this first
  pass; the eight-checkpoint slider/projector analysis is deferred as approved.
  Noise marks at 50,150,300 use eight single-step resamples restoring model AND
  Adam state; subtract mean displacement per site and use mean squared pairwise
  differences over distinct resamples. Main display beta is held fixed.
- No deferred second seed, output geometry, or attention toy. No kernel/basis
  fitting, no neural training in the default run, no full NxN matrices in JSON.
- Construction refinement during prototype: the first +/-4 z split plateau made
  a single north/south mode dominate the scan-selected temperature. Reducing the
  plateau to +/-1.5 balanced between-cluster directions; the unchanged largest-gap
  selection and both unchanged gates then passed. This changes weights, not gates.
- No specification deviation. The boundary/singleton/numerical-floor conventions
  above make the plan's relative-gap rule numerically explicit. Optional checkpoint
  analysis is omitted with the deferred slider, not silently shown as completed.
- Timing excludes first-time package download and interpreter/module startup.
  The exact stated uv command may install the default CUDA-enabled Torch wheel,
  but ALL computation is CPU. A preinstalled CPU-only Torch also works.

## Sections for page integration

`sites` = Figure 1 (THEORY 1–2, A1–A2); `scale` = Figure 2 (THEORY 2–4,10);
`coherence` = Figure 3 (THEORY 4.1); `loadings` = Figure 4 (THEORY 1.3,5);
`gradient` = Figure 5 (THEORY 7.1–7.2, A3). `trained` is absent by default.
The field table is exhaustive for the generated file. Array-of-record paths use
`[]`; its child shape is PER RECORD, with all observed variable shapes listed.
The exact same inventory is embedded at `meta.key_manifest`; each manifest record
has `path` (string JSON path), `shape` (string shape/type), and `meaning` (string).

## Every generated JSON key

| Key | Shape / type | Meaning |
|---|---|---|
'''
    rows = ['| `'+e['path']+'` | `'+e['shape']+'` | '+e['meaning'].replace('|', '\\|')+' |'
            for e in data['meta']['key_manifest']]
    optional = '\n\n## Optional fields (when --train is used)\n\n'
    optional += 'The generated table above includes these fields when present. Their meanings are:\n\n'
    optional += '\n'.join('- `'+k+'`: '+v for k, v in DESCRIPTIONS.items() if k.startswith('trained'))
    optional += '\n\n`trained.gradient_step150` recursively uses the exact `gradient` contract above.\n'
    path.write_text(header+'\n'.join(rows)+optional)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Include optional Adam-trained first-pass panels')
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'toy.json')
    args = parser.parse_args()
    generate(train=args.train, output=args.output)


if __name__ == '__main__':
    main()
