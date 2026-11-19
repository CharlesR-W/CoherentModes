"""Structural validation of a schema-v3 directory (SPEC.md section 6).

    uv run --with torch --with numpy --with scipy python -I validate3.py data/<tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import cm3
from coherent import seba_labels, seba_reconstruct


def validate(root, require_umap=True):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    n, K, ranks = meta["n"], meta["n_vecs"], meta["seba_r"]
    steps, sites = meta["steps"], meta["sites"]
    T, S, M, R, rmax = len(steps), len(sites), len(meta["modes"]), len(ranks), max(ranks)
    assert meta["schema_version"] == 3 and meta["modes"] == cm3.MODES
    assert meta["construction_fingerprint"] == cm3.fingerprint(K, ranks, n)
    assert meta["layer_pairs"] == [list(p) for p in cm3.pairs(S)] and meta["time_pairs"] == [list(p) for p in cm3.pairs(T)]
    for key in ("kappa", "kappa_q10", "kappa_q90", "nn_mass"):
        a = np.asarray(meta[key], dtype=object)
        assert a.shape == (M, T, S), key
        if key != "nn_mass":
            assert all(v is None for v in a[0].ravel()), key
            assert all(v is None or np.isfinite(v) for v in a[1].ravel()), key      # None: every row unattainable
    assert np.max(np.asarray(meta["sinkhorn_row_error"], dtype=float)) < 2e-12
    torch.set_num_threads(4)
    min_agree, blocks, horizons = 1.0, 0, {"layer": [], "time": []}
    for fam, names, X in (("layer", steps, S), ("time", sites, T)):
        P = X * (X + 1) // 2
        expected = cm3.allocate(M, P, n, X, K, ranks)
        for name in names:
            with np.load(root / f"{fam}_{name}.npz") as z:
                assert set(z.files) == set(expected), (name, set(z.files) ^ set(expected))
                for key, tmpl in expected.items():
                    assert z[key].shape == tmpl.shape and z[key].dtype == tmpl.dtype, (fam, name, key, z[key].shape)
                    if key != "retention_seba":
                        assert np.isfinite(z[key]).all(), (fam, name, key)
                vals = z["vals"]
                assert np.max(np.abs(vals[..., 0] - 1)) < 1e-6 and np.all(np.diff(vals, axis=-1) <= 1e-6)
                assert np.all(z["stab_span"] >= -1e-6) and np.all(z["stab_span"] <= 1 + 1e-6)
                vecs, scale, rot, perm, lab, ret = (z[k] for k in ("vecs", "scale", "seba_rot", "seba_perm", "seba_label", "retention_seba"))
                sval, slead = z["seba_val"], z["seba_lead"]
                for mi in range(M):
                    for pi in range(P):
                        horizons[fam].append(cm3.horizon(z["stab_span"][mi, pi]))
                        v = np.concatenate([np.full((n, 1), 1 / n ** .5), vecs[mi, pi].T.astype("f8") * scale[mi, pi] / 32767], 1)
                        assert np.max(np.abs(np.linalg.norm(v[:, 1:], axis=0) - 1)) < 2e-5
                        assert np.all(np.abs(vecs[mi, pi]).max(axis=1) == 32767)
                        for ri, r in enumerate(ranks):
                            L = lab[mi, pi, ri]
                            assert ((L >= -1) & (L < r)).all()
                            assert np.all(rot[mi, pi, ri, r:, :] == 0) and np.all(rot[mi, pi, ri, :, r:] == 0)
                            p = perm[mi, pi, ri, :r]
                            assert sorted(p) == list(range(r)) and (perm[mi, pi, ri, r:] == -1).all()
                            assert np.isfinite(ret[mi, pi, ri, :r]).all() and np.isnan(ret[mi, pi, ri, r:]).all()
                            assert np.all(sval[mi, pi, ri][L < 0] == 0) and np.all(slead[mi, pi, ri] <= sval[mi, pi, ri])
                            s = seba_reconstruct(torch.from_numpy(v[:, :r]), torch.from_numpy(rot[mi, pi, ri, :r, :r].astype("f8")),
                                                 torch.from_numpy(p.astype("i8")))
                            agree = float(np.mean(seba_labels(s).numpy() == L))
                            min_agree = min(min_agree, agree)
                            assert agree >= .99, (fam, name, mi, pi, r, agree)
                        blocks += 1
    for step in steps:
        with np.load(root / f"points_{step}.npz") as p:
            assert p["loss"].shape == (n,) and p["pred"].shape == (n,) and p["correct"].dtype == bool
            assert np.isfinite(p["loss"]).all()
        if require_umap:
            u = np.load(root / f"umap_{step}.npy")
            assert u.shape == (S, n, 2) and np.isfinite(u).all()
    with np.load(root / "probe.npz") as p:
        assert p["idx"].shape == (n,) and p["labels"].shape == (n,) and p["images"].shape == (n, 32, 32, 3)
        assert int(p["labels"].max()) < len(meta["classes"])
    man = json.loads((root / "curves.json").read_text())
    assert {r["step"] for r in man["checkpoints"]} >= set(steps)
    summary = {fam: dict(median=float(np.median(h)), min=int(min(h)), max=int(max(h))) for fam, h in horizons.items()}
    report = dict(complete=True, blocks=blocks, minimum_seba_label_agreement=min_agree, horizon_summary=summary,
                  decod=(root / "decod.json").exists())
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    validate(sys.argv[1], require_umap="--no-umap" not in sys.argv)
