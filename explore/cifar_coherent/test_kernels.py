"""Small CPU regressions for the gradient-kernel estimands."""
import torch

from kernels_modal import centre_gram, jacobian_grams, jacobians, row_cosine, rule_zero


def test_ggn_and_ntk_against_tiny_mlp():
    torch.manual_seed(23)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3)).eval()
    x = torch.randn(6, 3)
    j, logits = jacobians(model, x, batch=4)
    ggn, ntk = jacobian_grams(j, logits, tile=5)
    js, factors = [], []
    for xi in x:
        z = model(xi)
        grads = torch.stack([torch.cat([v.flatten() for v in torch.autograd.grad(
            z[k], tuple(model.parameters()), retain_graph=True)]) for k in range(len(z))]).double()
        p = z.detach().double().softmax(0)
        h = p.diag() - p.outer(p)
        val, vec = torch.linalg.eigh(h)
        root = vec @ torch.diag(val.clamp_min(0).sqrt()) @ vec.T
        torch.testing.assert_close(root @ root, h, atol=1e-14, rtol=1e-13)
        js.append(grads)
        factors.append(root @ grads)
    expected_ggn = torch.stack([torch.stack([(fa * fb).sum() for fb in factors]) for fa in factors])
    expected_ntk = torch.stack([torch.stack([(ja * jb).sum() for jb in js]) for ja in js])
    for got, expected in ((ggn, expected_ggn), (ntk, expected_ntk)):
        torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-6)
        assert float(torch.linalg.eigvalsh(got).min()) >= -1e-13


def test_double_centring_agrees_with_explicit_features():
    x = torch.randn(10, 5, generator=torch.Generator().manual_seed(14), dtype=torch.float64) + 3
    h = torch.eye(len(x), dtype=x.dtype) - 1 / len(x)
    g = x @ x.T
    xc = x - x.mean(0)
    got = centre_gram(g)
    torch.testing.assert_close(got, h @ g @ h, atol=1e-13, rtol=0)
    torch.testing.assert_close(got, xc @ xc.T, atol=1e-13, rtol=0)
    xc /= xc.norm(dim=1, keepdim=True)
    torch.testing.assert_close(row_cosine(got), xc @ xc.T, atol=1e-13, rtol=0)


def test_mips_union_affinity():
    x = torch.tensor([[1., 2.], [2., 4.], [1., -2.], [3., -1.]], dtype=torch.float64)
    log_k, ix = rule_zero(x @ x.T, k=1, raw=True)
    assert ix[:, 0].tolist() == [1, 0, 3, 2]
    expected = torch.tensor([[0, 1, 1e-6, 1e-6], [1, 0, 1e-6, 1e-6],
                             [1e-6, 1e-6, 0, 1], [1e-6, 1e-6, 1, 0]], dtype=x.dtype)
    torch.testing.assert_close(log_k.exp(), expected, atol=1e-20, rtol=1e-14)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
    print('all kernel tests passed')
