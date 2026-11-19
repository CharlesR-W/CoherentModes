"""Activations of DINOv2 ViT-S/14 for PLAN §1 and the CLS-position decomposition of PLAN §4.

Runs inside a Modal GPU container (see pipeline.py). The model is evaluated twice:
  * the hub model in float32 through its own `get_intermediate_layers(norm=True)` (the standard readout),
  * a manual float64 re-implementation of every block, which yields the components
    (per-head outputs, MLP neuron activations, biases) and the views used downstream.
The float64 views are the ones used everywhere; their max deviation from the float32 hub readout is
reported (`readout_vs_hub`).  The decomposition identity of PLAN §4,
    h_l = e + sum_{l'<=l} [ sum_h A_{l',h} + b_attn_{l'} + sum_n M_{l',n} + b_mlp_{l'} ],
and the view identity v_l = J_l h_l + o_l are asserted numerically on the probe set.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_LAYERS, D, N_HEADS, D_MLP, N_PATCH = 12, 384, 6, 1536, 256


def transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])


def to_tensor(pils):
    import torch
    x = torch.stack([torch.from_numpy(np.asarray(p, dtype=np.float32) / 255.0).permute(2, 0, 1) for p in pils])
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s


HF_DIR = "/root/hf_dinov2s"


def load_model():
    """dinov2_vits14 architecture from the official hub code, weights from HF `facebook/dinov2-small`
    (the official weight host's TLS certificate expired on 2026-11-11; see DEVIATIONS.md).
    Returns (model, mapped-key count)."""
    import torch
    from safetensors.torch import load_file

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
    hf = load_file(f"{HF_DIR}/model.safetensors")
    sd = {"cls_token": hf["embeddings.cls_token"], "pos_embed": hf["embeddings.position_embeddings"],
          "mask_token": hf["embeddings.mask_token"].reshape(model.mask_token.shape),
          "patch_embed.proj.weight": hf["embeddings.patch_embeddings.projection.weight"],
          "patch_embed.proj.bias": hf["embeddings.patch_embeddings.projection.bias"],
          "norm.weight": hf["layernorm.weight"], "norm.bias": hf["layernorm.bias"]}
    for i in range(N_LAYERS):
        p, q = f"encoder.layer.{i}.", f"blocks.{i}."
        a = p + "attention.attention."
        sd[q + "attn.qkv.weight"] = torch.cat([hf[a + n + ".weight"] for n in ("query", "key", "value")], 0)
        sd[q + "attn.qkv.bias"] = torch.cat([hf[a + n + ".bias"] for n in ("query", "key", "value")], 0)
        sd[q + "attn.proj.weight"] = hf[p + "attention.output.dense.weight"]
        sd[q + "attn.proj.bias"] = hf[p + "attention.output.dense.bias"]
        sd[q + "ls1.gamma"] = hf[p + "layer_scale1.lambda1"]
        sd[q + "ls2.gamma"] = hf[p + "layer_scale2.lambda1"]
        for n in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
            sd[q + n + ".weight"] = hf[p + n + ".weight"]
            sd[q + n + ".bias"] = hf[p + n + ".bias"]
    model.load_state_dict(sd, strict=True)
    return model.eval(), len(sd)


def parity_check(model, x518) -> float:
    """Max |difference| of the final-normed tokens between the hub architecture (loaded from HF weights)
    and transformers' Dinov2Model on the same weights, at 518 px (no positional interpolation)."""
    import torch
    from transformers import Dinov2Model
    hfm = Dinov2Model.from_pretrained(HF_DIR).eval().to(x518.device)
    with torch.no_grad():
        a = model.forward_features(x518)
        b = hfm(pixel_values=x518).last_hidden_state
    ours = torch.cat([a["x_norm_clstoken"][:, None], a["x_norm_patchtokens"]], 1)
    return float((ours - b).abs().max())


def manual_forward(model, x, want_components: bool, want_attn: bool):
    """Float64 forward through the blocks of `model` (already .double()).

    Returns dict with
      ln_cls   [B, 13, D]  final-LayerNorm(h_l)[CLS], l = 0..12
      ln_mp    [B, 13, D]  mean over patch tokens of final-LayerNorm(h_l)
      h_cls    [B, 13, D]  raw residual at CLS, l = 0..12
    and if want_components:
      heads    [B, 12, 6, 64]  per-head attention output at CLS before the output projection
      neurons  [B, 12, 1536]   post-GELU MLP activations at CLS
    and if want_attn:
      attn     [B, 12, 6, 257] CLS-row attention weights (softmax over all 257 tokens)
    """
    import torch
    import torch.nn.functional as F

    out = {k: [] for k in ("ln_cls", "ln_mp", "h_cls", "heads", "neurons", "attn")}
    h = model.prepare_tokens_with_masks(x)
    norm = model.norm

    def readout(h):
        z = norm(h)
        out["ln_cls"].append(z[:, 0])
        out["ln_mp"].append(z[:, 1:].mean(1))
        out["h_cls"].append(h[:, 0])

    readout(h)
    for blk in model.blocks:
        a = blk.attn
        z = blk.norm1(h)
        B, T, C = z.shape
        qkv = a.qkv(z).reshape(B, T, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * a.scale, qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)).softmax(dim=-1)            # [B, H, T, T]
        o = att @ v                                                  # [B, H, T, 64]
        if want_components:
            out["heads"].append(o[:, :, 0, :])
        if want_attn:
            out["attn"].append(att[:, :, 0, :])
        o = o.transpose(1, 2).reshape(B, T, C)
        h = h + blk.ls1(a.proj(o))
        z2 = blk.norm2(h)
        g = F.gelu(blk.mlp.fc1(z2))                                  # nn.GELU() default = exact erf GELU
        if want_components:
            out["neurons"].append(g[:, 0])
        h = h + blk.ls2(blk.mlp.fc2(g))
        readout(h)
    res = {k: torch.stack(v, 1) for k, v in out.items() if v}
    return res


def weights(model) -> dict:
    """Float64 numpy copies of everything the decomposition needs."""
    W = {"cls_e": (model.cls_token[0, 0] + model.pos_embed[0, 0]).detach().cpu().double().numpy(),
         "norm_w": model.norm.weight.detach().cpu().double().numpy(),
         "norm_b": model.norm.bias.detach().cpu().double().numpy(),
         "norm_eps": float(model.norm.eps)}
    for nm in ("proj_w", "proj_b", "ls1", "fc2_w", "fc2_b", "ls2"):
        W[nm] = []
    for blk in model.blocks:
        W["proj_w"].append(blk.attn.proj.weight.detach().cpu().double().numpy())     # [D, D]
        W["proj_b"].append(blk.attn.proj.bias.detach().cpu().double().numpy())
        W["ls1"].append(blk.ls1.gamma.detach().cpu().double().numpy())
        W["fc2_w"].append(blk.mlp.fc2.weight.detach().cpu().double().numpy())        # [D, 1536]
        W["fc2_b"].append(blk.mlp.fc2.bias.detach().cpu().double().numpy())
        W["ls2"].append(blk.ls2.gamma.detach().cpu().double().numpy())
    for nm in ("proj_w", "proj_b", "ls1", "fc2_w", "fc2_b", "ls2"):
        W[nm] = np.stack(W[nm])
    return W


def head_outputs(heads: np.ndarray, W: dict) -> np.ndarray:
    """A_{l,h}(x) in raw residual coordinates: ls1 ⊙ (proj_w[:, h-cols] @ o_h).  heads [B,12,6,64] -> [B,12,6,D]."""
    pw = W["proj_w"].reshape(N_LAYERS, D, N_HEADS, 64)                # [L, D, H, 64]
    A = np.einsum("blhc,ldhc->blhd", heads, pw)
    return A * W["ls1"][None, :, None, :]


def neuron_dirs(W: dict) -> np.ndarray:
    """u_{l,n} = LayerScale_l ⊙ W_out[:, n]: [12, 1536, D]."""
    return np.transpose(W["fc2_w"], (0, 2, 1)) * W["ls2"][:, None, :]


def baseline_bias(W: dict) -> np.ndarray:
    """Cumulative constant part of h_l at CLS: e + sum_{l'<=l} (b_attn + b_mlp), l = 0..12: [13, D]."""
    b = W["ls1"] * W["proj_b"] + W["ls2"] * W["fc2_b"]
    return W["cls_e"][None] + np.concatenate([np.zeros((1, D)), np.cumsum(b, 0)], 0)


def reconstruct_h(heads: np.ndarray, neurons: np.ndarray, W: dict) -> np.ndarray:
    """Sum of components: [B, 13, D]."""
    A = head_outputs(heads, W).sum(2)                                 # [B, 12, D]
    M = np.einsum("bln,lnd->bld", neurons, neuron_dirs(W))
    per = A + M
    cum = np.concatenate([np.zeros((per.shape[0], 1, D)), np.cumsum(per, 1)], 1)
    return cum + baseline_bias(W)[None]


def view_maps(h_cls: np.ndarray, W: dict, mean: np.ndarray):
    """Exact view map at CLS (PLAN §4): v_l = J_l h_l + o_l with
    J_l(x) u = gamma ⊙ (u - mean(u)) / (sigma_l rho_l),  o_l = (beta - m_l)/rho_l.
    h_cls [B, 13, D], mean [13, D] = probe mean of LN(h_l)[CLS].
    Returns v [B,13,D], sigma [B,13], rho [B,13]."""
    mu = h_cls.mean(-1, keepdims=True)
    var = ((h_cls - mu) ** 2).mean(-1)
    sigma = np.sqrt(var + W["norm_eps"])
    ln = W["norm_w"] * (h_cls - mu) / sigma[..., None] + W["norm_b"]
    c = ln - mean[None]
    rho = np.linalg.norm(c, axis=-1)
    J = lambda u: W["norm_w"] * (u - u.mean(-1, keepdims=True)) / (sigma * rho)[..., None]  # noqa: E731
    v_sum = J(h_cls) + (W["norm_b"] - mean[None]) / rho[..., None]
    return c / rho[..., None], v_sum, sigma, rho


def run(root: Path, parquet_dir: Path, plan: dict, batch: int = 250, limit: int | None = None) -> dict:
    """Compute and save activations for the probe and held-out sets.  `plan` holds the sampled rows."""
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    t0 = time.time()
    dev = "cuda"
    torch.backends.cudnn.allow_tf32 = False          # exact float32 for the hub-readout comparison
    torch.backends.cuda.matmul.allow_tf32 = False
    model32, n_keys = load_model()
    model32 = model32.to(dev)
    import copy
    model64 = copy.deepcopy(model32).double()
    W = weights(model64)
    tf = transform()

    # load the needed rows: plan["probe"] / plan["heldout"] are lists of [file, row]
    def load(rows):
        by_file: dict[str, list[int]] = {}
        for f, r in rows:
            by_file.setdefault(f, []).append(r)
        got = {}
        for f, rs in by_file.items():
            t = pq.read_table(parquet_dir / f, columns=["image"])
            col = t.column("image")
            for r in rs:
                got[(f, r)] = col[r].as_py()["bytes"]
        return [tf(Image.open(io.BytesIO(got[(f, r)])).convert("RGB")) for f, r in rows]

    out = {"weights": {"source": "HF facebook/dinov2-small", "n_keys": n_keys}}
    for split in ("probe", "heldout"):
        rows = plan[split] if limit is None else plan[split][:limit]
        t1 = time.time()
        pils = load(rows)
        t_load = time.time() - t1
        if split == "probe":
            from torchvision import transforms as T
            big = [T.Resize(518, interpolation=T.InterpolationMode.BICUBIC)(T.CenterCrop(224)(p)) for p in pils[:4]]
            par = parity_check(model32, to_tensor(big).to(dev))
            out["weights"]["parity_vs_transformers_518px"] = par
            assert par < 1e-3, f"weight mapping parity failed: {par}"
            thumbs = np.stack([np.asarray(p.resize((64, 64), Image.LANCZOS)) for p in pils])
            np.save(root / "thumbs64.npy", thumbs)
        acc = {}
        hub_err = 0.0
        attn_idx = set(plan["attn_idx"]) if split == "probe" else set()
        attn_rows, attn_list = [], []
        for s in range(0, len(pils), batch):
            x = to_tensor(pils[s:s + batch]).to(dev)
            with torch.no_grad():
                hub = model32.get_intermediate_layers(x, n=N_LAYERS, return_class_token=True, norm=True)
                hub_cls = torch.stack([c for _, c in hub], 1).double()                 # [B,12,D]
                res = manual_forward(model64, x.double(), split == "probe", False)
                hub_err = max(hub_err, float((res["ln_cls"][:, 1:] - hub_cls).abs().max()))
                sel = [i for i in range(s, min(s + batch, len(pils))) if i in attn_idx]
                if sel:
                    ra = manual_forward(model64, x[[i - s for i in sel]].double(), False, True)
                    attn_rows += sel
                    attn_list.append(ra["attn"].cpu().numpy())
            for k, v in res.items():
                acc.setdefault(k, []).append(v.cpu().numpy())
        acc = {k: np.concatenate(v) for k, v in acc.items()}
        out[split] = {"n": len(pils), "load_s": round(t_load, 1), "readout_vs_hub": hub_err}
        np.save(root / f"{split}_ln_cls.npy", acc["ln_cls"])                       # f64 [N,13,D]
        np.save(root / f"{split}_ln_mp.npy", acc["ln_mp"])
        if split == "probe":
            np.save(root / "probe_h_cls.npy", acc["h_cls"])
            np.save(root / "probe_heads.npy", acc["heads"].astype(np.float64))
            np.save(root / "probe_neurons.npy", acc["neurons"].astype(np.float32))
            # decomposition identity, float64
            rec = reconstruct_h(acc["heads"], acc["neurons"].astype(np.float64), W)
            err_h = float(np.abs(rec - acc["h_cls"]).max())
            scale_h = float(np.abs(acc["h_cls"]).max())
            mean = acc["ln_cls"].mean(0)
            v, v_sum, _, _ = view_maps(acc["h_cls"], W, mean)
            v_direct = (acc["ln_cls"] - mean) / np.linalg.norm(acc["ln_cls"] - mean, axis=-1, keepdims=True)
            err_v = float(np.abs(v_sum[:, 1:] - v_direct[:, 1:]).max())
            err_v2 = float(np.abs(v[:, 1:] - v_direct[:, 1:]).max())
            out["reconstruction"] = {"max_abs_err_h": err_h, "max_abs_h": scale_h,
                                     "max_abs_err_view_identity": err_v, "max_abs_err_view_ln": err_v2}
            assert err_h < 1e-8 * max(1.0, scale_h), out["reconstruction"]
            assert err_v < 1e-10, out["reconstruction"]
            order = np.argsort(attn_rows)
            np.save(root / "attn_idx.npy", np.asarray(attn_rows)[order])
            np.save(root / "attn.npy", np.concatenate(attn_list)[order].astype(np.float32))
            np.savez(root / "weights.npz", **{k: np.asarray(v) for k, v in W.items()})
    out["seconds"] = round(time.time() - t0, 1)
    (root / "acts.json").write_text(json.dumps(out, indent=1))
    return out
