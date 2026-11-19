"""Tests for analyze.py (CPU, seconds). Run with the torch/dtw environment given in analyze.py's docstring:
    .../uv run ... python -m pytest -q explore/spectral_align/test_analyze.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import align  # noqa: E402


@pytest.mark.parametrize("shape", [(12, 12), (12, 30), (18, 12), (5, 7), (2, 9)])
def test_batched_dtw_equals_align(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(5):
        d = rng.random(shape)
        assert abs(analyze.dtw_cost_batch(d[None])[0] - align.dtw_alignment(d)["cost"]) < 1e-12


def test_batched_null_equals_align():
    rng = np.random.default_rng(3)
    for shape in [(12, 30), (12, 12), (18, 30)]:
        d = rng.random(shape)
        ours = analyze.permutation_null(d, 60, seed=0)
        ref = align.block_permutation_null(d, 60, seed=0)
        assert np.allclose(ours, ref, atol=1e-12, rtol=0)


def test_constant_matrix_cost_and_reviewer_matrix():
    assert abs(analyze.dtw_cost_batch(np.full((1, 6, 9), .3))[0] - .3) < 1e-12
    d = np.array([[1, .1], [1, 1]])
    assert abs(analyze.dtw_cost_batch(d[None])[0] - align.dtw_alignment(d)["cost"]) < 1e-12


def test_q16_roundtrip_and_nan():
    v = np.array([0, .25, .5, 1, np.nan, np.inf])
    q = analyze.q16(v, 1.0)
    assert q[4] == 65535 and q[5] == 65535 and q[0] == 0 and q[3] == 65534
    back = analyze.dq16(q, 1.0)
    assert np.allclose(back[:4], v[:4], atol=1 / 65534) and np.isnan(back[4:]).all()


def test_int16_vectors_keep_projector_distances():
    rng = np.random.default_rng(0)
    n, k = 2000, 64
    x = rng.standard_normal((n, 2 * k))
    x -= x.mean(0)
    q = np.linalg.qr(x)[0]
    ua, ub = q[:, :k], np.linalg.qr(q[:, :k] + .5 * q[:, k:])[0]
    sa, qa = analyze.quantize_vectors(ua)
    sb, qb = analyze.quantize_vectors(ub)
    full = analyze.projector_dk(ua, ub, align.K_GRID)
    quant = analyze.projector_dk(qa * (sa / 32767), qb * (sb / 32767), align.K_GRID)
    assert np.abs(full - quant).max() < 1e-3


def test_horizon_and_resolved():
    span = np.array([.9, .8, .3, .6, .2] + [.1] * 75)
    assert analyze.horizon(span) == 4                      # largest k with span >= .5, gaps allowed (cm3.horizon)
    assert analyze.horizon(np.full(80, np.nan)) == 0
    r = analyze.resolved(np.stack([span, np.full(80, np.nan)]), [1, 3, 4])
    assert r.tolist() == [[True, False, True], [False, False, False]]


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    x = rng.random((300, 4))
    x[:, 2] = np.round(x[:, 0] * 5)                      # ties
    ours = analyze.spearman_matrix(x)
    ref = spearmanr(x).correlation
    assert np.allclose(ours, ref, atol=1e-12)


def test_first_stable():
    assert analyze.first_stable([False, True, False, True, True]) == 3
    assert analyze.first_stable([True, True]) == 0
    assert analyze.first_stable([True, False]) is None


# ------------------------------------------------------------------------------------------------ end to end
def tiny_data(tmp: Path):
    """Three networks (r20 s0, r20 s1, r32 s0), two checkpoints, an off-schedule iso-accuracy checkpoint."""
    from nets import site_names, site_stage
    nets = [("resnet20-relu-seed0", 20, 0), ("resnet20-relu-seed1", 20, 1), ("resnet32-relu-seed0", 32, 0)]
    cells = []
    for net, d, s in nets:
        for step in ([0, 64000] if d == 20 else [0, 3000, 64000]):
            for si, nm in enumerate(site_names(d)):
                cells.append(dict(index=len(cells), net=net, depth=d, seed=s, step=step, is_iso_accuracy=step == 3000,
                                  site_index=si, site_name=nm, stage=site_stage(d)[si], val_acc=.5))
    nc, n = len(cells), 80
    rng = np.random.default_rng(0)
    for r in (0, 1):
        x = rng.random((nc, nc, 19)).astype(np.float32)
        x = (x + x.transpose(1, 0, 2)) / 2
        x[np.arange(nc), np.arange(nc)] = 0
        np.savez(tmp / f"pairs_{r}.npz", proj=x, eigd=x * .3, heat=x[:, :, :5])
        np.save(tmp / f"eig_{r}.npy", np.sort(rng.random((nc, n)), 1)[:, ::-1].astype(np.float32))
        sp = rng.random((nc, 80)).astype(np.float32)
        sp[0] = np.nan
        np.save(tmp / f"span_{r}.npy", sp)
    c = rng.random((nc, nc)).astype(np.float32)
    c = (c + c.T) / 2
    np.fill_diagonal(c, 0)
    np.save(tmp / "cka.npy", c)
    q = np.linalg.qr(rng.standard_normal((n, 64)))[0]
    np.save(tmp / "U_resnet20-relu-seed0_64000.npy", np.stack([q] * 12).astype(np.float32))
    np.savez(tmp / "probe.npz", idx=np.arange(n), labels=np.arange(n) % 10)
    (tmp / "cells.json").write_text(json.dumps(cells))
    (tmp / "iso_accuracy.json").write_text(json.dumps({"resnet32-relu-seed0": dict(target=.5, available=True, step=3000,
                                                                                    accuracy=.501, overshoot=.001)}))
    (tmp / "accuracies.json").write_text(json.dumps({nm: .9 for nm, _, _ in nets}))
    return cells


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cells = tiny_data(data)
    out = analyze.run(data, tmp_path / "out", perms=20, workers=1)
    idx = json.loads((out / "index.json").read_text())
    # every shipped value equals the source tensor entry (within quantization)
    with np.load(data / "pairs_0.npz") as z:
        P = z["proj"]
    q = np.frombuffer(gzip.decompress((out / "s" / "proj_k8_r0.bin.gz").read_bytes()), "<u2")
    assert len(q) == idx["n_entries"]
    vals = analyze.dq16(q, idx["scale"]["proj_k8_r0"])
    k8 = idx["k_grid"].index(8)
    for s in idx["segs"]:
        rows = idx["groups"][s["group"]]["cells"] if s["kind"] == "self" else None
        if s["kind"] == "cross":
            rows, cols = idx["groups"][s["ga"]]["cells"], idx["groups"][s["gb"]]["cells"]
        elif s["kind"] == "self":
            cols = rows
        else:
            n = idx["nets"][s["net"]]
            rows = cols = [next(c["index"] for c in cells if c["net"] == n["name"] and c["step"] == st and c["site_index"] == s["site"])
                           for st in idx["main_steps"]]
        got = vals[s["o"]:s["o"] + s["nr"] * s["nc"]].reshape(s["nr"], s["nc"])
        assert np.allclose(got, P[np.ix_(rows, cols)][:, :, k8], atol=1 / 60000)
    # one iso comparison (r20 final vs r32 at 3000) per r20 seed, and the DTW records line up with the comparisons
    assert sum(x["kind"] == "iso" for x in idx["cross"]) == 2
    paths = json.loads((out / "s" / "proj_k8_r0.json").read_text())
    summ = json.loads((out / "summary_r0.json").read_text())
    assert len(paths) == len(idx["cross"]) == len(summ["dists"]["proj_k8"])
    for x, p in zip(idx["cross"], paths):
        s = idx["segs"][x["seg"]]
        pts = np.array(p["p"]).reshape(-1, 2)
        assert tuple(pts[0]) == (0, 0) and tuple(pts[-1]) == (s["nr"] - 1, s["nc"] - 1)
    # W4 matrices are square over all distances, with ones on the diagonal
    z = np.array(idx["w4"]["within_r0"]["z"], float)
    assert z.shape == (len(idx["w4_dists"]),) * 2 and np.allclose(np.diag(z), 1)
    facts = json.loads((out / "facts.json").read_text())
    assert all(k.startswith("al.") and isinstance(v, str) for k, v in facts.items())
    assert (out / "U" / "r20s0_64000_0.bin").stat().st_size == 64 * 4 + 80 * 64 * 2


def test_rule1_unavailable_snapshots_are_missing_not_zero(tmp_path):
    """Snapshots without a rule-1 operator: NaN in, 65535 (missing) out, no DTW for comparisons that need them,
    horizon -1, counted in the facts; rule 0 untouched."""
    data = tmp_path / "data"
    data.mkdir()
    cells = tiny_data(data)
    bad = [c["index"] for c in cells if c["net"] == "resnet32-relu-seed0" and c["step"] == 0 and c["site_index"] < 2]
    with np.load(data / "pairs_1.npz") as z:
        arrs = {k: z[k].copy() for k in z.files}
    for a in arrs.values():
        a[bad] = np.nan
        a[:, bad] = np.nan
    np.savez(data / "pairs_1.npz", **arrs)
    for name in ("eig_1.npy", "span_1.npy"):
        x = np.load(data / name)
        x[bad] = np.nan
        np.save(data / name, x)
    (data / "rule1_unavailable.json").write_text(json.dumps([dict(index=i) for i in bad]))
    out = analyze.run(data, tmp_path / "out", perms=10, workers=1)
    idx = json.loads((out / "index.json").read_text())
    facts = json.loads((out / "facts.json").read_text())
    assert idx["unavailable"]["1"] == bad and idx["unavailable"]["0"] == []
    assert all(idx["horizon"]["1"][c] == -1 for c in bad)
    q = np.frombuffer(gzip.decompress((out / "s" / "proj_k8_r1.bin.gz").read_bytes()), "<u2")
    rows = np.concatenate([np.repeat(s_rows, len(s_cols)) for s_rows, s_cols in
                           [(analyze_rows(idx, cells, s), analyze_cols(idx, cells, s)) for s in idx["segs"]]])
    cols = np.concatenate([np.tile(analyze_cols(idx, cells, s), len(analyze_rows(idx, cells, s))) for s in idx["segs"]])
    touches = np.isin(rows, bad) | np.isin(cols, bad)
    assert (q[touches] == 65535).all() and (q[~touches] != 65535).all()
    summ = json.loads((out / "summary_r1.json").read_text())
    need = [xi for xi, x in enumerate(idx["cross"]) if set(bad) & set(idx["groups"][x["ga"]]["cells"] + idx["groups"][x["gb"]]["cells"])]
    assert need and all(summ["dists"]["proj_k8"][xi][0] is None for xi in need)
    assert facts["al.missing.rule1.snapshots"] == str(len(bad)) and facts["al.missing.rule1.comparisons"] == str(len(need))
    assert all(np.isfinite(np.array(r[0], float)) for r in json.loads((out / "summary_r0.json").read_text())["dists"]["proj_k8"])


def analyze_rows(idx, cells, s):
    if s["kind"] == "self":
        return idx["groups"][s["group"]]["cells"]
    if s["kind"] == "cross":
        return idx["groups"][s["ga"]]["cells"]
    n = idx["nets"][s["net"]]
    return [next(c["index"] for c in cells if c["net"] == n["name"] and c["step"] == st and c["site_index"] == s["site"]) for st in idx["main_steps"]]


def analyze_cols(idx, cells, s):
    return idx["groups"][s["gb"]]["cells"] if s["kind"] == "cross" else analyze_rows(idx, cells, s)
