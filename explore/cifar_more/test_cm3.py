"""Local synthetic tests for cm3 (SPEC.md section 9). Run from this directory:
OMP_NUM_THREADS=2 uv run --with torch --with numpy --with scipy python -I test_cm3.py [--big]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import cm3
from coherent import centred_cosine_gram, seba_reconstruct, seba_labels
import schema_v2
from sweep_math import final_kernel_bank


def clusters(n, k, d=40, sep=3.0, noise=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(k, d, generator=g) * sep
    lab = torch.arange(n) % k
    return centres[lab] + noise * torch.randn(n, d, generator=g), lab


def test_kernel_bank_matches_reference():
    x, _ = clusters(600, 4)
    c = centred_cosine_gram(x)
    ref = final_kernel_bank(c)
    for (lk, info), (lr, ir) in zip(cm3.kernel_bank(c), ref[:2]):
        assert torch.equal(lk, lr)
        for key in ("kappa", "kappa_q10", "kappa_q90", "nn_mass"):
            assert info[key] == ir[key], (key, info[key], ir[key])


def test_perplexity_ties():
    """Coincident images: unattainable rows take the uniform-over-ties limit; other rows match the reference."""
    from sweep_math import perplexity_logkernel as ref
    x, _ = clusters(600, 4)
    c = centred_cosine_gram(x)
    a, ia = cm3.perplexity_logkernel(c, 100)
    b, ib = ref(c, 100)
    assert torch.equal(a, b) and ia["kappa_median"] == ib["kappa_median"] and ia["perplexity_unattainable"] == 0
    x2 = x.clone()
    x2[:150] = x2[0]                     # 150 identical images: 149 ties at distance 0 > 100
    c2 = centred_cosine_gram(x2)
    try:
        ref(c2, 100)
        raise AssertionError("reference should fail on this input")
    except (ValueError, AssertionError) as e:
        assert "reference should fail" not in str(e)
    lk, info = cm3.perplexity_logkernel(c2, 100)
    assert info["perplexity_unattainable"] == 150 and torch.isfinite(info["kappa_median"] * torch.ones(1)).all()
    m = cm3.lazy_operators(c2)
    for op, _ in m:
        assert float((op.sum(1) - 1).abs().max()) < 1e-11
    # every row unattainable: one fully collapsed cloud, and two collapsed groups
    for x3 in (x[:1].repeat(300, 1), torch.cat([x[:1].repeat(150, 1), x[1:2].repeat(150, 1)])):
        lk, info = cm3.perplexity_logkernel(centred_cosine_gram(x3 + 0 * x3), 100)
        assert info["perplexity_unattainable"] == 300 and info["kappa_median"] is None
        for op, inf in cm3.lazy_operators(centred_cosine_gram(x3)):
            assert float((op.sum(1) - 1).abs().max()) < 1e-11


def test_block_record_matches_v2():
    xs = [clusters(600, 4, seed=s)[0] for s in range(3)]
    ms = [cm3.lazy_operators(centred_cosine_gram(x))[0][0] for x in xs]
    for a, b in [(0, 0), (0, 2), (1, 2)]:
        new, _ = cm3.block_record(ms, a, b, 8, schema_v2.RANKS)
        old, _ = schema_v2.block_record(ms, a, b)
        for key in old:
            assert np.array_equal(np.asarray(new[key]), np.asarray(old[key]), equal_nan=True), key


def _stab(x, K):
    s1, s2, c1, c2 = cm3.split(len(x))
    m1 = cm3.lazy_operators(centred_cosine_gram(x[s1]))[0][0]
    m2 = cm3.lazy_operators(centred_cosine_gram(x[s2]))[0][0]
    return cm3.stability_record([m1], [m2], 0, 0, K, c1, c2)


def test_stability_planted_and_noise():
    torch.manual_seed(0)
    x, _ = clusters(900, 6, d=60, sep=4.0)
    n = len(x)
    s1, s2, c1, c2 = cm3.split(n)
    assert len(set(s1) & set(s2)) == n // 3 and np.array_equal(s1[c1], s2[c2])
    assert np.array_equal(cm3.split(n)[0], s1)          # deterministic, label-free
    rec = _stab(x, 16)
    span = rec["stab_span"]
    print("planted span", np.round(span, 2), "horizon", cm3.horizon(span))
    assert span[4] > .9 and cm3.horizon(span) == 5, span   # cuts inside the 5-dim group need not be reproducible
    noise = _stab(torch.randn(900, 200), 16)["stab_span"]
    print("noise span", np.round(noise[:8], 2), "horizon", cm3.horizon(noise))
    assert cm3.horizon(noise) <= 1, noise


def test_stability_wide_degenerate_group():
    """12 symmetric clusters: an 11-dim near-degenerate stable span, wider than the local window and crossing K=8."""
    g = torch.Generator().manual_seed(3)
    centres = torch.linalg.qr(torch.randn(80, 12, generator=g)).Q.T * 5      # equidistant: symmetric group
    lab = torch.arange(1800) % 12
    x = centres[lab] + torch.randn(1800, 80, generator=g)
    rec = _stab(x, 8)
    span, local = rec["stab_span"], rec["stab_local"]
    print("degenerate span", np.round(span[:14], 2), "local", np.round(local, 2), "horizon", cm3.horizon(span))
    assert span[10] > .9 and cm3.horizon(span) == 11, span     # resolved through the guard band, beyond K = 8
    assert len(span) == 8 + cm3.STAB_GUARD and len(local) == 8


def test_family_end_to_end():
    xs = [clusters(600, 5, seed=s)[0] for s in range(3)]
    s1, s2, _, _ = cm3.split(600)
    grams = [(centred_cosine_gram(x), centred_cosine_gram(x[s1]), centred_cosine_gram(x[s2])) for x in xs]
    arrays, infos = cm3.compute_family(grams, 12, [2, 4, 6, 10], log=lambda *a: None)
    assert arrays["vecs"].shape == (2, 6, 12, 600) and np.isfinite(arrays["stab_span"]).all()
    assert (arrays["seba_val"][:, :, 0] > 0).any() and arrays["seba_lead"].max() <= 255
    assert infos[0][0]["kappa"] is None and infos[0][1]["kappa"] > 0


def test_seba_r40_int16(n=5000):
    x, _ = clusters(n, 60, d=80, sep=3.0)
    m = cm3.lazy_operators(centred_cosine_gram(x))[0][0]
    rec, _ = cm3.block_record([m], 0, 0, 48, [40])
    v = np.concatenate([np.full((n, 1), 1 / n ** .5), rec["vecs"].T.astype("f8") * rec["scale"] / 32767], 1)
    s = seba_reconstruct(torch.from_numpy(v[:, :40]), torch.from_numpy(rec["seba_rot"][0].astype("f8")),
                         torch.from_numpy(rec["seba_perm"][0].astype("i8")))
    agree = float(np.mean(seba_labels(s).numpy() == rec["seba_label"][0]))
    print("r=40 int16 agreement", agree)
    assert agree >= .99


def test_augment_matches_slicing():
    from nets import augment
    torch.manual_seed(5)
    x = torch.randint(0, 256, (16, 3, 32, 32), dtype=torch.uint8)
    out, (oy, ox, flip) = augment(x)
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    for i in range(16):
        ref = padded[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
        ref = ref.flip(-1) if flip[i] else ref
        assert torch.equal(out[i], ref)


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_kernel_bank_matches_reference(); print("kernel bank ok")
    test_perplexity_ties(); print("perplexity ties ok")
    test_block_record_matches_v2(); print("block record ok")
    test_stability_planted_and_noise(); print("stability ok")
    test_stability_wide_degenerate_group(); print("degenerate group ok")
    test_family_end_to_end(); print("family ok")
    test_augment_matches_slicing(); print("augment ok")
    if "--big" in sys.argv:
        test_seba_r40_int16(); print("seba r40 ok")
    print("all tests passed")
