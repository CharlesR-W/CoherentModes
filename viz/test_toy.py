"""Fast identity/regression checks; no 12-temperature scan or output writes.
uv run --with numpy --with torch --with pytest python -I -m pytest viz/test_toy.py
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('coherent_modes_toy', Path(__file__).with_name('toy.py'))
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)
np, torch = toy.np, toy.torch


def test_planted_acceptance_gates_and_architecture():
    x, labels, half, _ = toy.sites()
    model = toy.Toy()
    with torch.no_grad():
        logits, hs, pre = model(torch.tensor(x))
    assert logits.shape == (1200, 4)
    assert all(z.shape == (1200, 16) for z in pre)
    hs = [h.numpy() for h in hs]
    # This is the approved scan's selected beta, not an independently tuned
    # gate temperature. The full generator asserts gates at its own selection.
    beta = toy.BETAS[8]
    p0, _, _ = toy.sinkhorn(toy.kernel(hs[0], beta))
    p2, _, _ = toy.sinkhorn(toy.kernel(hs[2], beta))
    result = toy.gates(p0, p2, toy.contrasts(labels, half))
    assert result['merge']['realized'] and result['split']['realized']
    # Folding and splitting really happen in the implemented network.
    tangent = np.array([-.5, np.sqrt(3)/2, 0.])
    q = hs[0] @ np.array([np.sqrt(3)/2, .5, 0.])
    t = hs[0] @ tangent
    wedge = 12*(q-.85) > np.abs(t)
    assert np.max(np.abs(hs[1][wedge]@tangent)) < 1e-12
    # Gaussian tails need not all fall inside the wedge; the unchanged
    # acceptance gate above is the specified population-level criterion.
    assert np.mean(wedge[labels < 2]) > .99
    assert np.mean(hs[2][(labels == 2) & half, 2]) > 1.4
    assert np.mean(hs[2][(labels == 2) & ~half, 2]) < -1.3
    assert np.max(np.abs(p0.sum(1)-1)) < 1e-10
    assert np.max(np.abs(p2.sum(1)-1)) < 1e-10


def test_symmetric_sinkhorn_centering_eigenpairs_and_products():
    rng = np.random.default_rng(19)
    h0, h2 = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
    k0, k2 = toy.kernel(h0, 2), toy.kernel(h2, 2)
    p0, _, _ = toy.sinkhorn(k0)
    p2, _, _ = toy.sinkhorn(k2)
    assert np.max(np.abs(p0.sum(1)-1)) < 1e-10
    assert np.allclose(p0, p0.T, atol=1e-14)
    values, psi = toy.eigenbasis(p2)
    assert np.allclose(psi.T@psi, np.eye(40), atol=1e-12)
    assert np.allclose(psi[:, 0], np.ones(40)/np.sqrt(40), atol=1e-14)
    assert np.allclose(toy.centered(p2)@psi[:, 1:], psi[:, 1:]*values, atol=1e-12)
    assert abs(np.linalg.svd(p2@p0, compute_uv=False)[0]-1) < 1e-9
    def degree(k):
        d = k.sum(1)
        return k/np.sqrt(d[:, None]*d[None, :])
    assert np.linalg.svd(degree(k2)@degree(k0), compute_uv=False)[0] < 1
    # No overflow, diagonal exactly one; shifted and unshifted normalization agree.
    assert np.all(np.diag(k0) > 1-1e-14)
    unshifted, _, _ = toy.sinkhorn(k0*np.exp(2))
    assert np.allclose(p0, unshifted, atol=1e-10)


def test_reconstruction_loadings_rotations_and_autograd_decomposition():
    x, _, _, target = toy.sites(per_cluster=10)
    model = toy.Toy()
    with torch.no_grad():
        _, hs, _ = model(torch.tensor(x))
    h2 = hs[2].numpy()
    p, _, _ = toy.sinkhorn(toy.kernel(h2, 15))
    values, psi = toy.eigenbasis(p)
    block, isolated = toy.top_block(values)
    groups = toy.groups_for(values)
    assert groups[0]['start'] == 0 and groups[-1]['stop'] == len(x)
    assert all(a['stop'] == b['start'] for a, b in zip(groups, groups[1:]))
    loads = toy.loading_data(h2, psi, values, groups, block, isolated)
    grads = toy.gradient_data(model, x, target, psi, groups, block[1])
    assert loads['reconstruction_max_error'] < 1e-10
    assert grads['checks']['autograd_max_error'] < 1e-10
    assert grads['checks']['block_sum_max_error'] < 1e-9
    assert grads['truncation']['residual_fro'][-1] < 1e-9
    assert np.allclose(loads['rotation'].T@loads['rotation'], np.eye(block[1]), atol=1e-12)
    assert np.allclose(grads['interactions'], grads['interactions'].T, atol=1e-12)
    assert np.allclose(np.diag(grads['interactions']), grads['pressures']**2)
    assert np.isclose(grads['rectification']['before_block_energy'].sum(), 1)
    assert grads['rectification']['after_block_energy'][0] > 0


def test_blocks_are_maximal_before_dimension_cap():
    # Three nearly equal leading eigenvalues form ONE maximal block, not
    # three arbitrarily isolated singletons.
    vals = np.array([.999, .998, .997, .3, .1, .01, 0.])
    b, isolated = toy.top_block(vals)
    assert isolated and b[:2] == (0, 3)
    assert not any(start == 0 and end < 3 for start, end, _ in toy.isolated_blocks(vals))
    # A seven-dimensional true top block must trigger the explicit fallback.
    vals = np.r_[np.linspace(.999, .993, 7), .2, .01, 0.]
    b, isolated = toy.top_block(vals)
    assert not isolated and b[:2] == (0, 6)
    b, isolated = toy.top_block(np.zeros(20))
    assert not isolated and b[1] == 6


def test_heatmap_averages_are_not_renormalized():
    a = np.arange(24*24).reshape(24, 24)
    labels = np.tile(np.arange(6), 4)
    b = toy.downsample(a, labels, size=12)
    order = np.argsort(labels, kind='stable')
    assert np.isclose(b[0, 0], a[np.ix_(order[:2], order[:2])].mean())
    p = np.ones((24, 24))/24
    assert np.allclose(toy.downsample(p, labels, size=12).sum(1), .5)


def test_noise_resampling_restores_model_and_adam():
    torch.manual_seed(3)
    model = toy.Toy(planted=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    x, _, _, target = toy.sites(per_cluster=4)
    x, target = torch.tensor(x), torch.tensor(target)
    logits, _, _ = model(x)
    toy.nn.functional.cross_entropy(logits, target).backward()
    optimizer.step()
    weights = toy.copy.deepcopy(model.state_dict())
    state = toy.copy.deepcopy(optimizer.state_dict())
    noise = toy.noise_scale(model, optimizer, x, target, torch.Generator().manual_seed(4))
    assert len(noise['beta_by_layer']) == 3
    assert np.all(np.isfinite(noise['beta_by_layer']))
    for key, value in weights.items():
        assert torch.equal(value, model.state_dict()[key])
    after = optimizer.state_dict()
    assert state['param_groups'] == after['param_groups']
    for key, entry in state['state'].items():
        for field, value in entry.items():
            assert torch.equal(value, after['state'][key][field])
