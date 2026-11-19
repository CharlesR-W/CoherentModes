"""SPEC_DOT.md section 1 identities (CPU, small N).

    uv run --with torch --with numpy python test_dot.py
"""
import torch
from coherent import centred_cosine_gram, centred_dot_gram, dot_effective_cosine, symmetric_balance
from sweep_math import final_kernel_bank

torch.manual_seed(0)


def test_unit_rows_reduce_to_cosine():
    # After probe-centring, rescale rows to unit norm: c_eff must equal the centred cosine Gram.
    x = torch.randn(60, 7, dtype=torch.float64); x = x - x.mean(0)
    x = x / x.norm(dim=1, keepdim=True)
    g = x @ x.T
    assert torch.allclose(dot_effective_cosine(g), (x @ x.T), atol=1e-12)


def test_scale_and_translation_invariance_of_rules():
    x = torch.randn(400, 5, dtype=torch.float64) * torch.rand(400, 1) * 3  # N > perplexity 300
    a = final_kernel_bank(dot_effective_cosine(centred_dot_gram(x)))
    b = final_kernel_bank(dot_effective_cosine(centred_dot_gram(7.3 * x + 2.0)))
    for (la, _), (lb, _) in zip(a, b):
        pa, pb = symmetric_balance(la), symmetric_balance(lb)
        assert float((pa - pb).abs().max()) < 1e-9


def test_exponentiated_dot_equals_gaussian_after_scaling():
    x = torch.randn(50, 4, dtype=torch.float64) * torch.rand(50, 1) * 2
    g = x @ x.T
    kappa = 1.7
    d = (g.diagonal()[:, None] + g.diagonal()[None, :]) / 2 - g
    l1 = kappa * g; l1.diagonal().fill_(float('-inf'))
    l2 = -kappa * d; l2.diagonal().fill_(float('-inf'))
    assert float((symmetric_balance(l1) - symmetric_balance(l2)).abs().max()) < 1e-10


def test_norm_changes_dot_but_not_cosine():
    x = torch.randn(40, 6, dtype=torch.float64); x = x - x.mean(0)
    y = x * torch.linspace(.2, 3, 40, dtype=torch.float64)[:, None]
    assert torch.allclose(centred_cosine_gram(x - x.mean(0)), centred_cosine_gram(x), atol=1e-12)
    assert not torch.allclose(dot_effective_cosine(centred_dot_gram(x)), dot_effective_cosine(centred_dot_gram(y)), atol=1e-3)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
    print('all dot tests passed')
