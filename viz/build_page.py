"""Build pages/toy.html from viz/data/toy.json with the report engine in viz/report/.

    uv run --with numpy python -I viz/build_page.py
    bash viz/report/verify_page.sh pages/toy.html /tmp/toy-shots

Five sections follow viz/PLAN.md "Figures, first pass"; each is preceded by its
explanatory diagram from viz/diagrams/ (D1, D2, D5, D3, D4+D7; D6 in the closing
note). Palette and type follow viz/diagrams/STYLE.md: the engine's default style and
palette strings are replaced in memory, the engine file is not edited. Every number
in a caption is read from toy.json, never typed in. Display coordinates are rounded
to 4 decimals for embedding (the engine itself never rounds); nothing is recomputed.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viz" / "report"))
sys.path.insert(0, str(ROOT / "viz" / "diagrams"))
import report  # noqa: E402  (viz/report/report.py)
from make_diagrams import div_color, seq_color  # noqa: E402  (palette functions, STYLE.md)

DATA = ROOT / "viz" / "data" / "toy.json"
OUT = ROOT / "pages" / "toy.html"
DIAG = ROOT / "viz" / "diagrams"

# ------------------------------------------------------------------ style (STYLE.md)
SURFACE, PAGE, INK, INK2, MUTED, GRID, AXISC = "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DIV = [[t, div_color(2 * t - 1, 1)] for t in np.linspace(0, 1, 11).tolist()]
SEQ = [[t, seq_color(t)] for t in np.linspace(0, 1, 11).tolist()]

report.STYLE = f"""
* {{box-sizing:border-box}}
:root {{--ink:{INK};--dim:{INK2};--border:{GRID};--bg:{PAGE};--soft:#f0efec}}
body {{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 {FONT}}}
main {{max-width:1180px;margin:auto;padding:26px 28px 70px}}
h1 {{font-size:30px;line-height:1.2;margin:4px 0 10px}}
h2 {{font-size:22px;margin:40px 0 8px;padding-top:14px;border-top:1px solid var(--border)}}
h3 {{font-size:16px;margin:0 0 8px}}
p {{max-width:84ch}}
.subtitle,figcaption {{color:var(--dim)}}
figure {{border:1px solid var(--border);border-radius:8px;background:{SURFACE};padding:14px;margin:16px 0;overflow:hidden}}
figure.diagram {{padding:0;overflow:hidden}}
figure.diagram svg {{display:block;width:100%;height:auto}}
.plot {{width:100%;height:420px}}
figcaption {{font-size:13.5px;margin-top:8px;max-width:110ch}}
.controls {{position:sticky;top:0;z-index:10;background:var(--soft);border:1px solid var(--border);
 border-radius:8px;padding:12px;display:flex;flex-wrap:wrap;align-items:center;gap:12px}}
pre {{white-space:pre-wrap;overflow-wrap:anywhere}}
#error {{color:#d03b3b;white-space:pre-wrap}}
@media(max-width:600px) {{main{{padding:18px 12px 40px}}figure{{padding:10px}}.plot{{height:360px}}h1{{font-size:25px}}}}
"""
_script = report.SCRIPT
for old, new in [
    ("const PALETTE = ['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377','#bbbbbb'];",
     "const PALETTE = " + json.dumps(CAT) + ";"),
    ("gridcolor:'#eeebe5',zerolinecolor:'#e2ddd5',linecolor:'#d3cdc3'",
     f"gridcolor:'{GRID}',zerolinecolor:'{AXISC}',linecolor:'{AXISC}'"),
    ("paper_bgcolor:'white',plot_bgcolor:'white'", f"paper_bgcolor:'{SURFACE}',plot_bgcolor:'{SURFACE}'"),
    ("font:{family:'system-ui,sans-serif',size:13,color:'#2b2a27'}",
     "font:{family:" + json.dumps(FONT) + f",size:13,color:'{INK}'}}"),
]:
    assert old in _script, old
    _script = _script.replace(old, new)
report.SCRIPT = _script

# ------------------------------------------------------------------ helpers
def r4(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()

def f2(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")

SP, SPZ = 2.7, 3.2   # spacing between panels (columns, rows) inside one 3D scene

def off(i, ncols):
    """Offset of panel i in a grid laid out inside ONE scene: columns along x, rows down z.
    One WebGL context per figure (Chrome keeps only ~16 alive), so small multiples share a scene."""
    return np.array([(i % ncols) * SP, 0.0, -(i // ncols) * SPZ])

def grid_scene(ncols=1, nrows=1, size=2.3):
    """One orthographic scene holding an ncols x nrows grid of unit-radius panels; the aspect
    ratio follows the grid's extent so the panels are drawn large and undistorted."""
    ax = dict(visible=False, showgrid=False, zeroline=False, showbackground=False)
    span = np.array([(ncols - 1) * SP + 2.6, 2.6, (nrows - 1) * SPZ + 2.9])
    ratio = span / span.max() * size
    return dict(domain=dict(x=[0, 1], y=[0, 1]), aspectmode="manual", dragmode="turntable",
                aspectratio=dict(x=float(ratio[0]), y=float(ratio[1]), z=float(ratio[2])),
                camera=dict(projection=dict(type="orthographic"), eye=dict(x=0.3, y=-0.95, z=0.6), up=dict(x=0, y=0, z=1)),
                xaxis=ax, yaxis=ax, zaxis=ax)

def triad(o, sc, labels=False, length=1.15):
    xs, ys, zs = [], [], []
    for k in range(3):
        e = np.eye(3)[k] * length
        xs += [o[0], o[0] + e[0], None]; ys += [o[1], o[1] + e[1], None]; zs += [o[2], o[2] + e[2], None]
    tr = [dict(type="scatter3d", mode="lines", scene=sc, x=xs, y=ys, z=zs,
               line=dict(color=AXISC, width=2), showlegend=False, hoverinfo="skip")]
    if labels:
        tip = o[None, :] + np.eye(3) * (length + 0.12)
        tr.append(dict(type="scatter3d", mode="text", scene=sc, x=r4(tip[:, 0]), y=r4(tip[:, 1]), z=r4(tip[:, 2]),
                       text=["e1", "e2", "e3"], textfont=dict(size=11, color=MUTED), showlegend=False, hoverinfo="skip"))
    return tr

def label3d(o, text_, sc, dz=1.3):
    return dict(type="scatter3d", mode="text", scene=sc, x=[float(o[0])], y=[float(o[1])], z=[float(o[2] + dz)],
                text=[text_], textfont=dict(size=13, color=INK), showlegend=False, hoverinfo="skip")

def sphere_trace(sc, o=np.zeros(3)):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    return dict(type="surface", scene=sc, x=r4(o[0] + np.cos(u) * np.sin(v)), y=r4(o[1] + np.sin(u) * np.sin(v)),
                z=r4(o[2] + np.cos(v)), opacity=0.08, showscale=False, hoverinfo="skip",
                colorscale=[[0, MUTED], [1, MUTED]])

def cloud_by_cluster(X, cluster, sc, showlegend, o=np.zeros(3), size=2.4, sizes=None):
    X = X + o
    tr = []
    for c in range(6):
        m = cluster == c
        mk = dict(color=CAT[c], size=(r4(sizes[m]) if sizes is not None else size), opacity=0.85, line=dict(width=0))
        tr.append(dict(type="scatter3d", mode="markers", scene=sc, x=r4(X[m, 0]), y=r4(X[m, 1]), z=r4(X[m, 2]),
                       name=f"cluster {c}", legendgroup=f"c{c}", showlegend=showlegend, marker=mk,
                       hovertemplate=f"cluster {c}<extra></extra>"))
    cen = np.array([X[cluster == c].mean(0) for c in range(6)])
    tr.append(dict(type="scatter3d", mode="text", scene=sc, x=r4(cen[:, 0]), y=r4(cen[:, 1]), z=r4(cen[:, 2] + 0.14),
                   text=[str(c) for c in range(6)], textfont=dict(size=13, color=INK), showlegend=False, hoverinfo="skip"))
    return tr

def cloud_by_value(X, vals, sc, vmax, name, colorbar=None, size=2.6, o=np.zeros(3)):
    X = X + o
    mk = dict(color=r4(vals), colorscale=DIV, cmin=-vmax, cmax=vmax, size=size, line=dict(width=0))
    if colorbar is not None:
        mk["colorbar"] = colorbar
        mk["showscale"] = True
    else:
        mk["showscale"] = False
    return dict(type="scatter3d", mode="markers", scene=sc, x=r4(X[:, 0]), y=r4(X[:, 1]), z=r4(X[:, 2]),
                name=name, showlegend=False, marker=mk, hovertemplate="%{marker.color:.4f}<extra>" + name + "</extra>")

def arrows3d(vecs, labels, sc, colors, o=np.zeros(3)):
    tr = []
    for v, lab, col in zip(vecs, labels, colors):
        tip = o + np.asarray(v)
        tr.append(dict(type="scatter3d", mode="lines+markers+text", scene=sc, x=r4([o[0], tip[0]]), y=r4([o[1], tip[1]]),
                       z=r4([o[2], tip[2]]), line=dict(color=col, width=5), marker=dict(size=[0, 4], color=col),
                       text=["", lab], textposition="top center", textfont=dict(size=13, color=INK),
                       showlegend=False, hoverinfo="skip"))
    return tr

def annot(text, x, y, xref="paper", yref="paper", size=13, color=INK, **kw):
    return dict(text=text, x=x, y=y, xref=xref, yref=yref, showarrow=False, font=dict(size=size, color=color), **kw)

def cbar(title, x=1.0, length=0.8):
    return dict(title=dict(text=title, side="right", font=dict(size=12)), x=x, len=length, thickness=12, outlinewidth=0)

# ------------------------------------------------------------------ build
def build():
    d = json.loads(DATA.read_text())
    meta, sites, scale, coh, load, grad = (d[k] for k in ("meta", "sites", "scale", "coherence", "loadings", "gradient"))
    N = meta["N"]
    T = meta["temperatures"]
    beta_d = T["display"]
    cluster = np.array(sites["cluster"])
    raw = np.array(sites["raw"]); nrm = np.array(sites["normalized"]); norms = np.array(sites["norms"])
    br = meta["block_rule"]
    tall = {}   # figure number -> plot height override (px)
    secs = []
    fig_no = [0]
    scene_no = [0]

    def fig(title, caption, data, layout, height=None):
        # Plotly gives each 3D subplot a DOM id equal to its scene name, so scene names
        # must be unique across the whole page: renumber this figure's scenes globally.
        local = sorted((k for k in layout if re.fullmatch(r"scene\d*", k)), key=lambda k: int(k[5:] or 1))
        ren = {}
        for k in local:
            scene_no[0] += 1
            ren[k] = "scene" if scene_no[0] == 1 else f"scene{scene_no[0]}"
        layout = {ren.get(k, k): v for k, v in layout.items()}
        for tr in data:
            if "scene" in tr:
                tr["scene"] = ren[tr["scene"]]
        # Pin numeric cartesian axes to "linear": Plotly's type auto-detection mis-typed some
        # numeric axes as categorical when several figures share the page.
        for ax_key, d_key in (("xaxis", "x"), ("yaxis", "y")):
            arrs = [tr[d_key] for tr in data if tr.get("type") in ("scatter", "bar") and d_key in tr]
            if arrs and all(all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a) for a in arrs):
                axd = dict(layout.get(ax_key, {}))
                axd.setdefault("type", "linear")
                layout = dict(layout, **{ax_key: axd})
        if height:
            tall[fig_no[0]] = height
        fig_no[0] += 1
        return dict(title=title, caption=caption, data=data, layout=layout)

    disp_scan = scale["scan"][T["display_index"]]
    # ============================================================ intro
    intro = dict(
        title="What this page is",
        caption=(f"A picture, not an experiment: the construction of THEORY.md on a residual network whose stream has "
                 f"d = 3, so every activation is a point in R^3 and, after cosine normalization, on the sphere S^2. "
                 f"N = {N} sites (six planted Gaussian clusters on a ring, 200 points each). The weights are planted, "
                 f"not trained: block 1 merges clusters 0 and 1, block 2 splits cluster 2 along a line; both planted "
                 f"gates are realized (Section 3). "
                 f"Declared choices. Display temperature: beta = {beta_d:.1f}, the value in the {len(T['scan'])}-point "
                 f"scan ({T['scan'][0]:g} to {T['scan'][-1]:g}, log-spaced) with the largest isolated-block gap at layer 2; "
                 f"this is a display choice, not an inferred temperature. Block rule: a block is a maximal run of "
                 f"eigenvalues whose internal gaps are all below {br['gap_ratio']:g} times each neighbouring gap (the top of "
                 f"the centered spectrum has no upper neighbour; gaps below {br['numerical_gap_floor']:g} cannot isolate); if "
                 f"no block of dimension <= {br['display_max_dimension']} is isolated the page says so and shows the top-6 "
                 f"subspace; whatever is not resolved into blocks is the complement. "
                 f"Known limits of the toy. d = 3 cannot show superposition of many features: the toy shows the operator "
                 f"machinery, not the dictionary regime. Only the dense path is exercised; no sparse-graph or Nystrom "
                 f"estimand. The merge and split are constructed; whether training produces them is checked separately on "
                 f"a trained variant (Section 3b), which may not. Each data section is preceded by an "
                 f"explanatory diagram of the objects it uses (D1 to D7; those numbers come from small separate toys)."))
    secs.append(intro)

    # ============================================================ 1 sites
    s1 = []
    rmax = float(np.linalg.norm(raw, axis=2).max())
    data, lay = [], dict(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.06, x=0), scene=grid_scene(3, 1, 3.0))
    for L in range(3):
        o = off(L, 3)
        data += triad(o, "scene", labels=L == 0)
        data += cloud_by_cluster(raw[L] / rmax, cluster, "scene", L == 0, o=o)
        data.append(label3d(o, f"h{L} (raw)", "scene"))
    s1.append(fig("Raw residual stream h0, h1, h2 in R^3, coloured by planted cluster",
                  f"Each point is one site; the same {N} sites appear at every layer [A1]. Colour and the digit at each "
                  f"cluster's centroid give the planted cluster (validation only; labels never enter a kernel). All three "
                  f"layers are drawn at one common scale (divided by the largest norm, {rmax:.2f}); grey lines are the "
                  f"coordinate axes e1, e2, e3 of each panel, and dragging rotates the three panels together. Between h0 "
                  f"and h1 clusters 0 and 1 are mapped onto a common direction (planted merge); between h1 and h2 cluster 2 "
                  f"is cut into two directions (planted split).", data, lay, 430))
    data, lay = [], dict(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.06, x=0), scene=grid_scene(3, 1, 3.0))
    smax = norms.max()
    for L in range(3):
        o = off(L, 3)
        data.append(sphere_trace("scene", o))
        data += cloud_by_cluster(nrm[L], cluster, "scene", L == 0, o=o, sizes=1.2 + 4.0 * norms[L] / smax)
        data.append(label3d(o, f"h{L} / ||h{L}||", "scene"))
    s1.append(fig("Normalized stream on the unit sphere S^2; point size is the dropped norm",
                  f"The geometric view keeps only the direction h/||h|| (faint sphere); marker size encodes the norm "
                  f"||h(x)|| that the cosine discards (diameter linear in the norm, from 1.2 px at zero to 5.2 px at the largest, {smax:.2f}). This normalization is the single geometric "
                  f"commitment [A2]: the kernels and operators downstream see only these normalized directions; the loading and "
                  f"gradient panels use raw activations where they say so.", data, lay, 430))
    secs.append(dict(title="1. Sites and the metric choice",
                     caption="[A1] the site is the atom; [A2] the dot product is the one geometric commitment. "
                             "THEORY Sections 1-2. The same ordered sites are drawn at every layer; only their "
                             "activations change.", figures=s1, diagram="D1"))

    # ============================================================ 2 kernel -> operator -> scale
    s2 = []
    order = np.array(scale["heatmap_cluster_order"])
    ticks = [int(np.mean(np.where(order == c)[0])) for c in range(6)]
    names = {0: "near the linear limit", 1: "display temperature", 2: "toward the identity limit"}
    for k, hm in enumerate(scale["heatmaps"]):
        Kd, Pd = np.array(hm["kernel"]), np.array(hm["operator"])
        ax = lambda dom: dict(domain=list(dom), tickvals=ticks, ticktext=[str(c) for c in range(6)], showgrid=False,
                              title=dict(text="planted cluster (bins)", font=dict(size=12)))
        lay = dict(margin=dict(l=60, r=30, t=40, b=60),
                   xaxis=ax((0, 0.40)), yaxis=dict(ax((0, 1)), autorange="reversed", scaleanchor="x"),
                   xaxis2=dict(ax((0.55, 0.95)), anchor="y2"), yaxis2=dict(ax((0, 1)), autorange="reversed", anchor="x2", scaleanchor="x2"),
                   annotations=[annot("kernel K (shifted, exp(beta(dot - 1)))", 0.20, 1.07),
                                annot("operator P (Sinkhorn)", 0.75, 1.07)])
        data = [dict(type="heatmap", z=r4(Kd), colorscale=SEQ, zmin=0, zmax=1, colorbar=cbar("K entry", 0.42),
                     hovertemplate="K %{z:.4f}<extra></extra>"),
                dict(type="heatmap", z=np.round(Pd, 7).tolist(), xaxis="x2", yaxis="y2", colorscale=SEQ, zmin=0,
                     zmax=float(Pd.max()), colorbar=cbar("P bin average", 0.97),
                     hovertemplate="P %{z:.2e}<extra></extra>")]
        s2.append(fig(f"Layer 2 kernel and operator at beta = {hm['beta']:.3g} ({names[k]})",
                      "Sites sorted by planted cluster and averaged in 10 x 10 bins (120 x 120). The kernel is drawn shifted, "
                      "exp(beta(dot - 1)) = e^-beta K_beta, which leaves P unchanged. The operator panel is "
                      "a bin average of P, so its rows sum to 0.1, not 1; it is not a re-normalized 120-site operator."
                      + (" Here the kernel is nearly constant (every entry close to 1): as beta -> 0 the structure is in its small centered part, which after division by beta tends to the centered Gram matrix of the normalized activations (linear structure)."
                         if k == 0 else "") +
                      (" At beta -> infinity the operator tends to the identity for distinct normalized vectors; repeated "
                       "directions leave averaging blocks." if k == 2 else ""), data, lay, 440))
    # eigenvalue scan
    betas = [s["beta"] for s in scale["scan"]]
    ev = np.array([s["eigenvalues"] for s in scale["scan"]])
    data = []
    for i in range(ev.shape[1]):
        data.append(dict(type="scatter", mode="lines+markers", x=r4(betas), y=r4(ev[:, i]), name=f"lambda_{i + 1}",
                         showlegend=False, line=dict(color=INK2 if i < 5 else AXISC, width=1.6), marker=dict(size=5),
                         hovertemplate=f"lambda_{i + 1} = %{{y:.4f}} at beta %{{x:.3g}}<extra></extra>"))
    bnn = T["beta_nn"]
    lay = dict(xaxis=dict(type="log", dtick=1, range=[-1.2, math.log10(bnn) + 0.25], title=dict(text="beta (log scale)")),
               yaxis=dict(range=[0, 1.05], title=dict(text="eigenvalue of C P C")),
               shapes=[dict(type="line", x0=beta_d, x1=beta_d, y0=0, y1=1, yref="paper", line=dict(color=INK, dash="dash", width=1.5)),
                       dict(type="line", x0=bnn, x1=bnn, y0=0, y1=1, yref="paper", line=dict(color=MUTED, dash="dot", width=1.5))],
               annotations=[annot(f"display beta = {beta_d:.1f} ", math.log10(beta_d), 1.04, xref="x", xanchor="right"),
                            annot(f"beta_nn = {bnn:.0f} ", math.log10(bnn), 1.04, xref="x", xanchor="right", color=INK2),
                            annot("dark: lambda_1 to lambda_5; light: lambda_6 to lambda_12", 0.01, 0.97, xanchor="left", color=INK2, size=12)])
    s2.append(fig("Leading twelve nonconstant eigenvalues of C P C across the beta scan (layer 2)",
                  f"C = I - 11^T/N is the centering projector, so C P C drops the constant mode. Eigenvalues are sorted at each beta and not tracked across crossings. Small beta: all eigenvalues near "
                  f"zero (divided by beta they tend to the centered Gram limit, rank at most 3 in d = 3); large beta: all tend to one (identity limit for "
                  f"distinct normalized vectors). Clustering is the finite window between, where a block of "
                  f"{disp_scan['block_dimension']} eigenvalues sits near one with a gap of {disp_scan['gap']:.2f} below it at the "
                  f"display beta. beta_nn = 1/(median squared distance to the {T['nn_k']}th neighbour) = {bnn:.0f} lies beyond "
                  f"the scan and is marked for reference only.", data, lay))
    # principal angles + block dimension
    data = []
    xs, ys = [], []
    for s in scale["scan"]:
        for a in s["principal_angles_radians"]:
            xs.append(s["beta"]); ys.append(a)
    data.append(dict(type="scatter", mode="markers", x=r4(xs), y=r4(ys), marker=dict(color=INK2, size=8),
                     name="principal angle", showlegend=False, hovertemplate="%{y:.3f} rad at beta %{x:.3g}<extra></extra>"))
    lay = dict(xaxis=dict(type="log", dtick=1, range=[-1.15, 2.15], title=dict(text="beta (log scale)")),
               yaxis=dict(range=[-0.05, 1.8], title=dict(text="principal angle (radians)")),
               shapes=[dict(type="line", x0=0, x1=1, xref="paper", y0=math.pi / 2, y1=math.pi / 2, line=dict(color=AXISC, dash="dot"))],
               annotations=[annot(f"dim {s['block_dimension']}" + ("" if s["isolated"] else "*"), math.log10(s["beta"]), 1.72,
                                  xref="x", yref="y", size=12, color=INK2) for s in scale["scan"]]
               + [annot("pi/2", 0.995, math.pi / 2 + 0.06, xanchor="right", yref="y", size=12, color=MUTED)])
    nfall = sum(not s["isolated"] for s in scale["scan"])
    s2.append(fig("Indicator-subspace recovery: principal angles between the top block and the planted cluster indicators",
                  f"For each beta, the angles between the isolated top block (its dimension, dim, written above each column"
                  + ("; * = no isolated block, top-6 subspace" if nfall else "; every scan value has an isolated top block")
                  + ") and the span of the six planted cluster indicators. At the display beta the block has dimension "
                  f"{disp_scan['block_dimension']} and four angles are below 0.1 rad; one angle is near pi/2 because at layer 2 "
                  f"clusters 0 and 1 have merged and cluster 2 has split, so one layer-2 contrast is not a planted layer-0 "
                  f"cluster indicator (THEORY Section 4.3).", data, lay))
    # normalization
    nn_ = scale["normalization"]
    data = [dict(type="scatter", mode="markers+text", x=[nn_["sinkhorn_product_top_sv"], nn_["degree_product_top_sv"]],
                 y=["Sinkhorn (bistochastic) P2 P0", "degree-normalized D^-1/2 K D^-1/2, layer 2 times layer 0"],
                 text=[f"{nn_['sinkhorn_product_top_sv']:.10f}", f"{nn_['degree_product_top_sv']:.4f}"],
                 textposition="middle left", marker=dict(size=12, color=INK), showlegend=False,
                 hovertemplate="%{x:.10f}<extra></extra>")]
    lay = dict(xaxis=dict(range=[0.9984, 1.0003], title=dict(text="top singular value of the uncentered layer product")),
               yaxis=dict(automargin=True), margin=dict(l=20, r=30, t=20, b=60),
               shapes=[dict(type="line", x0=1, x1=1, y0=0, y1=1, yref="paper", line=dict(color=AXISC, dash="dot"))])
    s2.append(fig("Why the shared measure matters: top singular value of the two-layer product",
                  f"With Sinkhorn scaling every layer's operator is Markov for the same counting measure, so the product "
                  f"P2 P0 has top singular value exactly one ({nn_['sinkhorn_product_top_sv']:.10f}); with the usual degree "
                  f"normalization each layer has its own stationary vector and the product falls below one "
                  f"({nn_['degree_product_top_sv']:.4f}), which would read as loss of coherence [M3]. Points, not bars: "
                  f"the axis is zoomed.", data, lay, 260))
    # diffusion time
    dif = scale["diffusion"]
    data = []
    for k, (t, row) in enumerate(zip(dif["times"], dif["eigenvalues"])):
        data.append(dict(type="scatter", mode="lines+markers", x=list(range(1, len(row) + 1)), y=r4(row),
                         name=f"t = {t}", line=dict(color=CAT[k], width=2), marker=dict(size=7),
                         hovertemplate=f"t = {t}: " + "lambda_%{x}^t = %{y:.4f}<extra></extra>"))
    lay = dict(xaxis=dict(title=dict(text="nonconstant mode index i"), dtick=1),
               yaxis=dict(range=[0, 1.05], title=dict(text="lambda_i^t")),
               annotations=[annot(f"t = {t}", 5.15, row[4], xref="x", yref="y", xanchor="left", size=12)
                            for t, row in zip(dif["times"], dif["eigenvalues"])])
    s2.append(fig(f"Diffusion-time control at the display beta: lambda_i^t for t = {', '.join(map(str, dif['times']))}",
                  f"Raising P to the power t damps fine modes smoothly. At t = {dif['times'][-1]} the block's eigenvalues "
                  f"have decayed to between {min(dif['eigenvalues'][-1][:5]):.2f} and {max(dif['eigenvalues'][-1][:5]):.2f} "
                  f"while the modes past the gap are at most {max(dif['eigenvalues'][-1][5:]):.1e}, so the gap survives and "
                  f"time and truncation agree here. Without a gap they would "
                  "disagree and the smooth knob is preferred (THEORY Section 3.2).", data, lay))
    secs.append(dict(title="2. Kernel, operator and scale",
                     caption="THEORY Sections 2-4 and 10. The two limits: the centered Gram matrix at beta -> 0 (after centering and division by beta), and the "
                             "identity at beta -> infinity for distinct normalized vectors (repeated directions leave "
                             "averaging blocks). Clustering is a finite window, not a limit; blocks are delimited by gaps; "
                             "and the shared measure (Sinkhorn) is what lets layer operators compose.",
                     figures=s2, diagram="D2"))

    # ============================================================ 3 coherence
    s3 = []
    U_, V_ = np.array(coh["left_modes"]), np.array(coh["right_modes"])
    sv = coh["singular_values"]
    vmax = float(max(np.abs(U_).max(), np.abs(V_).max()))
    data = []
    for i in range(3):
        o0, o2 = off(2 * i, 2), off(2 * i + 1, 2)
        data += [sphere_trace("scene", o0), cloud_by_value(nrm[0], V_[:, i], "scene", vmax, f"v_{i + 1}", o=o0,
                                                           colorbar=cbar("value", 1.0, 0.5) if i == 0 else None),
                 sphere_trace("scene", o2), cloud_by_value(nrm[2], U_[:, i], "scene", vmax, f"u_{i + 1}", o=o2),
                 label3d(o0, f"v_{i + 1} on layer 0", "scene"), label3d(o2, f"u_{i + 1} on layer 2, sigma_{i + 1} = {sv[i]:.3f}", "scene")]
    s3.append(fig("Preserved structure: the top three nontrivial singular pairs of Q = P2 P0",
                  f"Each row is one singular pair: v_i drawn as colour on the layer-0 cloud, u_i on the layer-2 cloud "
                  f"(blue negative, red positive, one symmetric scale for all six; the same sites in both columns). "
                  f"sigma = {sv[0]:.3f}, {sv[1]:.3f}, {sv[2]:.3f}: near one means slow at both layers, kept together. A "
                  f"singular pair shows preservation, not the direction of a merge or split.", data,
                  dict(scene=grid_scene(2, 3, 1.8), margin=dict(l=0, r=0, t=10, b=0)), 860))
    data = []
    cmax = max(float(np.abs(np.array(coh["directional"][n]["contrast"])).max()) for n in ("merge", "split"))
    texts = {}
    for row, name in enumerate(("merge", "split")):
        rec = coh["directional"][name]
        c = np.array(rec["contrast"])
        o0, o2 = off(2 * row, 2), off(2 * row + 1, 2)
        verdict = "realized" if rec["realized"] else "not realized"
        data += [sphere_trace("scene", o0), cloud_by_value(nrm[0], c, "scene", cmax, "c", o=o0,
                                                           colorbar=cbar("c", 1.0, 0.5) if row == 0 else None),
                 sphere_trace("scene", o2), cloud_by_value(nrm[2], c, "scene", cmax, "c", o=o2),
                 label3d(o0, f"{name} contrast on layer 0", "scene"), label3d(o2, f"on layer 2: gate {verdict}", "scene")]
        if name == "merge":
            texts[name] = ("Top row, merge: clusters 0 (+) and 1 (-), separate at layer 0 and one direction at layer 2; "
                           f"slow at layer 0 (||P0 c|| = {rec['p0_norm']:.3f}) and nearly annihilated after layer 2 "
                           f"(||P2 P0 c|| = {rec['product_norm']:.3f}).")
        else:
            npos, nneg = int((c > 0).sum()), int((c < 0).sum())
            texts[name] = (f"Bottom row, split: the two halves of cluster 2 ({npos} sites +, {nneg} sites -, both at "
                           f"1/sqrt({npos + nneg}), not rebalanced or recentered, so c keeps a constant component), one "
                           "blob at layer 0 and two directions at "
                           f"layer 2; not slow at layer 0 (||P0 c|| = {rec['p0_norm']:.3f}) but slow at layer 2 "
                           f"(||P2 c|| = {rec['p2_norm']:.3f}).")
    s3.append(fig("Planted merge and split contrasts c, each drawn on the layer-0 and the layer-2 cloud",
                  "c is +/- 1/sqrt(support size) on two planted site groups and zero elsewhere (grey). " + texts["merge"]
                  + " " + texts["split"] + " The contrasts are planted validation checks, not modes the method discovered.",
                  data, dict(scene=grid_scene(2, 2, 1.8), margin=dict(l=0, r=0, t=10, b=0)), 640))
    g = meta["gates"]
    labels = ["||P0 c||", "||P2 c||", "||P2 P0 c||"]
    data = []
    for k, name in enumerate(("merge", "split")):
        vals = [g[name]["p0_norm"], g[name]["p2_norm"], g[name]["product_norm"]]
        data.append(dict(type="scatter", mode="markers+text", x=r4(vals), y=[f"{name}: {l}" for l in labels],
                         text=[f"{v:.3f}" for v in vals], textposition="middle right", name=f"{name} contrast",
                         marker=dict(size=12, color=CAT[k]), hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    lay = dict(xaxis=dict(range=[0, 1.12], title=dict(text="norm (||c|| = 1)")), yaxis=dict(autorange="reversed"),
               margin=dict(l=20, r=30, t=40, b=60), legend=dict(orientation="h", y=-0.28, x=0),
               shapes=[dict(type="line", x0=v, x1=v, y0=0, y1=1, yref="paper", line=dict(color=MUTED, dash="dot")) for v in (0.2, 0.9)],
               annotations=[annot("0.2", 0.2, 1.04, xref="x", size=12, color=INK2), annot("0.9", 0.9, 1.04, xref="x", size=12, color=INK2)])
    s3.append(fig("Acceptance gates for the planted merge and split",
                  f"Merge realized when ||P0 c|| > 0.9 and ||P2 P0 c|| < 0.2: {g['merge']['p0_norm']:.3f} and "
                  f"{g['merge']['product_norm']:.3f}, {'realized' if g['merge']['realized'] else 'not realized'}. Split "
                  f"realized when ||P0 c|| < 0.2 and ||P2 c|| > 0.9: {g['split']['p0_norm']:.3f} and "
                  f"{g['split']['p2_norm']:.3f}, {'realized' if g['split']['realized'] else 'not realized'}. Dotted lines "
                  f"mark the two thresholds. The split gate compares ||P0 c|| with ||P2 c||, as the plan specifies; the "
                  f"reverse product ||P0 P2 c|| (reading Q^T) is not in the data file and is not shown.", data, lay, 330))
    secs.append(dict(title="3. Preservation, merge and split",
                     caption="THEORY Section 4.1. Merge: a contrast slow at L and annihilated at L'. Sigma alone cannot "
                             "give the direction, because Q and its transpose share singular values; the direction comes "
                             "from the contrast norms. The planted contrasts are validation, not discovery.",
                     figures=s3, diagram="D5"))

    # ============================================================ 3b trained variant
    tr = d.get("trained")
    if tr is not None:
        s3b = []
        tn = np.array(tr["normalized"])
        data = []
        for k, L in enumerate((0, 2)):
            o = off(k, 2)
            data.append(sphere_trace("scene", o))
            data += cloud_by_cluster(tn[L], cluster, "scene", k == 0, o=o)
            data.append(label3d(o, f"trained h{L} / ||h{L}||, step {tr['steps']}", "scene"))
        s3b.append(fig("Trained variant: final normalized layer-0 and layer-2 clouds, coloured by planted cluster",
                       f"The same architecture trained from a random initialization (Adam, learning rate {tr['learning_rate']:g}, "
                       f"{tr['steps']} steps, batch {tr['batch_size']}, seed {tr['seed']}) on the coarsened labels (clusters 0 and 1 "
                       f"share a label; cluster 2's halves get different labels). Final full-site accuracy "
                       f"{100 * tr['final_accuracy']:.1f}%. Compare the planted clouds in Section 1: here clusters 0 and 1 end "
                       f"near each other at layer 2 but remain distinct directions, and cluster 2 is spread out rather than cut "
                       f"into two directions, consistent with the gate numbers below.", data,
                       dict(scene=grid_scene(2, 1, 2.4), margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.06, x=0)), 430))
        ls = np.array(tr["losses"])
        data = [dict(type="scatter", mode="lines+markers", x=ls[:, 0].astype(int).tolist(), y=r4(ls[:, 1]), name="minibatch cross-entropy",
                     line=dict(color=INK, width=2), marker=dict(size=6), hovertemplate="step %{x}: %{y:.4f}<extra></extra>")]
        lay = dict(xaxis=dict(title=dict(text="Adam step")), yaxis=dict(rangemode="tozero", title=dict(text="mean cross-entropy on the step's minibatch")))
        s3b.append(fig("Trained variant: training loss",
                       f"Minibatch mean cross-entropy at the {len(tr['checkpoint_steps'])} checkpoints (every 10 steps), one "
                       f"minibatch of {tr['batch_size']} per point. The data file holds no held-out or test loss, so none is "
                       f"shown; the endpoint is summarized by the full-site accuracy, {100 * tr['final_accuracy']:.1f}%.", data, lay, 340))
        gp, gt = meta["gates"], tr["gates"]
        rows = []
        for name in ("merge", "split"):
            for key, lab in (("p0_norm", "||P0 c||"), ("p2_norm", "||P2 c||"), ("product_norm", "||P2 P0 c||")):
                rows.append((f"{name}: {lab}", gp[name][key], gt[name][key]))
        data = []
        for k, (mname, idx) in enumerate((("planted", 1), ("trained", 2))):
            data.append(dict(type="scatter", mode="markers+text", x=r4([r[idx] for r in rows]), y=[r[0] for r in rows],
                             text=[f"{r[idx]:.3f}" for r in rows], textposition="bottom center" if k else "top center",
                             name=f"{mname} model", marker=dict(size=12, color=CAT[k], symbol="circle" if k == 0 else "diamond"),
                             hovertemplate=mname + " %{y}: %{x:.4f}<extra></extra>"))
        verdict = lambda g_, n: "realized" if g_[n]["realized"] else "not realized"
        lay = dict(xaxis=dict(range=[-0.12, 1.15], title=dict(text="norm (||c|| = 1)")), yaxis=dict(autorange="reversed"),
                   margin=dict(l=20, r=30, t=40, b=60), legend=dict(orientation="h", y=-0.25, x=0),
                   shapes=[dict(type="line", x0=v, x1=v, y0=0, y1=1, yref="paper", line=dict(color=MUTED, dash="dot")) for v in (0.2, 0.9)],
                   annotations=[annot("0.2", 0.2, 1.04, xref="x", size=12, color=INK2), annot("0.9", 0.9, 1.04, xref="x", size=12, color=INK2)])
        lay["yaxis"] = dict(autorange="reversed")
        s3b.append(fig("Acceptance gates, planted versus trained (same contrasts, same beta)",
                       f"Same contrasts and the same display beta = {tr['display_beta']:.1f} for both models (not re-selected for "
                       f"the trained one). Merge needs ||P0 c|| > 0.9 and ||P2 P0 c|| < 0.2: planted {gp['merge']['p0_norm']:.3f} and "
                       f"{gp['merge']['product_norm']:.3f}, {verdict(gp, 'merge')}; trained {gt['merge']['p0_norm']:.3f} and "
                       f"{gt['merge']['product_norm']:.3f}, {verdict(gt, 'merge')}. Split needs ||P0 c|| < 0.2 and ||P2 c|| > 0.9: "
                       f"planted {gp['split']['p0_norm']:.3f} and {gp['split']['p2_norm']:.3f}, {verdict(gp, 'split')}; trained "
                       f"{gt['split']['p0_norm']:.3f} and {gt['split']['p2_norm']:.3f}, {verdict(gt, 'split')}. Circles: planted; "
                       f"diamonds: trained.", data, lay, 420))
        secs.append(dict(title="3b. Trained variant: does training produce the merge and split?",
                         caption=f"No. The same architecture trained to {100 * tr['final_accuracy']:.1f}% accuracy on labels that "
                                 f"coarsen the clusters exactly as the planted merge and split do, yet at the same beta the merge "
                                 f"gate is {verdict(gt, 'merge')} (the contrast between clusters 0 and 1 is only partly removed by "
                                 f"layer 2) and the split gate is {verdict(gt, 'split')} (the halves of cluster 2 are only partly "
                                 f"separated). This is the plan's warning made concrete: classification does not force cosine "
                                 f"collapse or separation, so a merge or split is captioned only where its gate passes. THEORY "
                                 f"Section 4.1; the trained model's step-150 gradient decomposition is in the data file but not "
                                 f"shown in this first pass.", figures=s3b))

    # ============================================================ 4 loadings
    s4 = []
    modes, A = np.array(load["modes"]), np.array(load["loadings"])
    rmodes, rA = np.array(load["rotated_modes"]), np.array(load["rotated_loadings"])
    a0 = np.array(load["constant_loading"])
    ev4 = load["eigenvalues"]
    for rot in (False, True):
        M, AA = (rmodes, rA) if rot else (modes, A)
        vmax = float(np.abs(M).max())
        data = []
        nc = 3 if rot else 4
        k0 = 0 if rot else 1                       # eigenfunction figure starts with the constant mode
        if not rot:
            cm = np.array(load["constant_mode"])
            data += [sphere_trace("scene", off(0, nc)),
                     cloud_by_value(nrm[2], cm, "scene", vmax, "psi_0", None, size=2.2, o=off(0, nc)),
                     label3d(off(0, nc), f"psi_0 = 1/sqrt(N) = {cm[0]:.4f}, lambda = 1", "scene")]
        for j in range(5):
            o = off(j + k0, nc)
            data.append(sphere_trace("scene", o))
            data.append(cloud_by_value(nrm[2], M[:, j], "scene", vmax, ("r" if rot else "psi_") + str(j + 1),
                                       cbar("value", 1.0, 0.45) if j == 0 else None, size=2.2, o=o))
            lab = (f"rotated vector {j + 1}" if rot else f"psi_{load['mode_indices'][j]}, lambda = {ev4[j]:.4f}")
            data.append(label3d(o, lab, "scene"))
        o = off(5 + k0, nc)
        vecs = [a0] + [AA[j] for j in range(5)]
        ascale = 1.0 / float(max(np.linalg.norm(v) for v in [a0] + list(A) + list(rA)))
        labs = ["a_0"] + [(f"r{j + 1}" if rot else f"a_{j + 1}") for j in range(5)]
        data += triad(o, "scene", labels=True)
        data += arrows3d([v * ascale for v in vecs], labs, "scene", [MUTED] + [INK2] * 5, o=o)
        data.append(label3d(o, f"raw loadings x {ascale:.3f}", "scene"))
        lay = dict(scene=grid_scene(nc, 2, 2.6), margin=dict(l=0, r=0, t=10, b=0))
        if rot:
            s4.append(fig(load["rotated_label"].capitalize() + ": varimax rotation of the same block",
                          "A localizing rotation of the same five-dimensional block, with loadings rotated consistently "
                          "(rotated_loadings = R^T loadings). The span is the same canonical object; these individual vectors "
                          "are a declared display choice [P4] and are not eigenfunctions. Last panel: the rotated raw "
                          f"loadings, drawn scaled by {ascale:.3f}.", data, lay, 640))
        else:
            s4.append(fig(f"Constant mode and the isolated top block at layer 2 ({load['basis_label']}), with raw loadings",
                          f"First panel: the constant mode psi_0 (uniform). Next five: each eigenfunction of the block (gap to the rest of the spectrum {load['gap']:.2f}) as "
                          f"colour on the layer-2 sphere (one symmetric scale), signs fixed so the largest-|value| site is "
                          f"positive. Last panel: the loadings a_i = H2^T psi_i of the raw activations, with a_0 for the "
                          f"constant mode, drawn scaled by {ascale:.3f}; they are coefficients, not orthogonal directions, "
                          f"and in d = 3 at most three are independent. Panels rotate together.", data, lay, 640))
    rec = load["reconstruction"]
    r_ = np.array(rec["r"]); res = np.array(rec["residual_squared"]); tail = np.array(rec["tail_sum"])
    keep = r_ <= 40
    data = [dict(type="scatter", mode="lines+markers", x=r_[keep].tolist(), y=r4(res[keep]), name="direct ||H - H_r||_F^2",
                 line=dict(color=INK, width=2), marker=dict(size=6), hovertemplate="r = %{x}: %{y:.4f}<extra>direct</extra>"),
            dict(type="scatter", mode="markers", x=r_[keep].tolist(), y=r4(tail[keep]), name="tail sum of ||a_i||^2",
                 marker=dict(size=11, color="rgba(0, 0, 0, 0)", line=dict(color=CAT[1], width=2)),
                 hovertemplate="r = %{x}: %{y:.4f}<extra>tail sum</extra>")]
    lay = dict(xaxis=dict(title=dict(text="r = nonconstant modes kept (constant always kept; cuts only at block boundaries)")),
               yaxis=dict(type="log", dtick=1, title=dict(text="squared Frobenius error (log scale)")), legend=dict(orientation="h", y=1.1, x=0))
    after = res[r_ == r_[keep].max()][0]
    s4.append(fig("Reconstruction tail at layer 2: direct residual equals the sum of omitted squared loadings",
                  f"The exact expansion H2 = sum_i psi_i a_i^T over the complete basis means the error after keeping the "
                  f"constant mode and r further modes is the tail sum; dots (direct) and rings (tail sum) coincide. Cuts "
                  f"are made only at block boundaries. At r = 0 the error is {res[0]:.2f}; after the five-mode block "
                  f"(r = 5) it is {res[r_ == 5][0] if (r_ == 5).any() else float('nan'):.4f}; at r = {r_[keep].max()} it is "
                  f"{after:.2e}, and at r = {r_.max()} the reconstruction is exact (max entry error "
                  f"{load['reconstruction_max_error']:.1e}).", data, lay))
    secs.append(dict(title="4. Loadings and reconstruction",
                     caption=f"THEORY Sections 1.3 and 5, at the display beta = {beta_d:.1f}, layer 2. The expansion over the "
                             "complete basis is exact and the truncation error is the tail sum. Blocks are canonical; "
                             "individual vectors inside a block are not, which is why the rotated basis is shown separately.",
                     figures=s4, diagram="D3"))

    # ============================================================ 5 gradient and units
    s5 = []
    groups = grad["groups"]
    nshow = 11                                   # constant + first 10 blocks, then complement
    sel = list(range(nshow)) + [len(groups) - 1]
    gname = lambda gi: groups[gi]["name"].replace("block_", "B_") + (f" (dim {groups[gi]['dimension']})" if groups[gi]["dimension"] > 1 else "")
    press = np.array(grad["pressures"])
    rest_max = float(press[nshow:-1].max()) if len(groups) > nshow + 1 else 0.0
    from collections import Counter
    dims = Counter(gg["dimension"] for gg in groups[nshow:-1])
    omitted_dims = ", ".join(f"{n} of dimension {dd}" for dd, n in sorted(dims.items()))
    data = [dict(type="scatter", mode="markers", x=[gname(i) for i in sel], y=np.round(press[sel], 8).tolist(),
                 marker=dict(color=INK2, size=11), name="block pressure", hovertemplate="%{x}: %{y:.3e}<extra></extra>")]
    lay = dict(margin=dict(l=64, r=32, t=36, b=120),
               xaxis=dict(title=dict(text="group in the layer-2 basis (constant, blocks in spectral order, complement)", standoff=28)),
               yaxis=dict(type="log", dtick=1, title=dict(text="block pressure ||G_B||_F (log scale)")))
    s5.append(fig("Block pressure on W1 of the second residual MLP: ||G_B||_F for the constant mode, the first spectral blocks and the complement",
                  f"G = Delta^T U is the gradient of the mean cross-entropy (loss {grad['loss']:.3f}) with respect to W1 of the "
                  f"second residual MLP (h2 = h1 + W2 ReLU(W1 h1 + b)) at the planted weights, so U = h1 and Delta = dL/dz at "
                  f"z = W1 h1 + b. Any complete orthonormal basis on the sites decomposes G exactly; the layer-2 basis is used "
                  f"so that Sections 4 and 5 share one basis (a display choice). G_B = Delta^T Pi_B U is the part carried by block B of the layer-2 "
                  f"basis, invariant to rotation inside the block. The {len(groups) - nshow - 1} further blocks not drawn "
                  f"({omitted_dims}) each have pressure at most {rest_max:.2e}. This is the gradient, not an optimizer update.",
                  data, lay))
    I = np.array(grad["interactions"])[np.ix_(sel, sel)]
    dg = np.sqrt(np.clip(np.diag(I), 0, None))
    cosI = I / np.outer(dg, dg)
    data = [dict(type="heatmap", z=r4(cosI), customdata=np.round(I, 8).tolist(), x=[gname(i) for i in sel], y=[gname(i) for i in sel],
                 colorscale=DIV, zmin=-1, zmax=1, colorbar=cbar("cosine of <G_B, G_B'>_F"),
                 hovertemplate="%{y} x %{x}: cosine %{z:.3f}, raw %{customdata:.2e}<extra></extra>")]
    lay = dict(yaxis=dict(autorange="reversed"), margin=dict(l=110, r=30, t=30, b=90))
    s5.append(fig("Signed interactions between the same groups, as cosines <G_B, G_B'>_F / (||G_B||_F ||G_B'||_F)",
                  "Red: the two groups' gradient contributions align (cooperative); blue: they oppose (interference); grey: "
                  "orthogonal. Shown as cosines so that small groups are visible next to B_1; hover gives the raw signed "
                  "interaction <G_B, G_B'>_F, whose diagonal is the squared pressure. This is the alignment of spectral contributions, not "
                  "in general of site-group loss gradients (THEORY Section 7.2).", data, lay, 460))
    tr_ = grad["truncation"]
    data = [dict(type="scatter", mode="lines+markers", x=tr_["r"], y=r4(tr_["residual_fro"]), line=dict(color=INK, width=2),
                 marker=dict(size=5), name="||G - G_r||_F", hovertemplate="r = %{x}: %{y:.5f}<extra></extra>")]
    lay = dict(xaxis=dict(type="log", dtick=1, title=dict(text="r = nonconstant modes kept (log scale; constant always kept)")),
               yaxis=dict(title=dict(text="||G - G_r||_F, computed directly")))
    xr = [r for r in tr_["r"] if r > 0]
    data[0]["x"] = xr
    data[0]["y"] = r4(np.array(tr_["residual_fro"])[[i for i, r in enumerate(tr_["r"]) if r > 0]])
    rf0 = tr_["residual_fro"][0]
    s5.append(fig("Gradient truncation residual ||G - G_r||_F (computed, not summed)",
                  f"The rank-one pieces are not Frobenius-orthogonal, so the residual is computed directly from the "
                  f"truncated sum at each block boundary rather than added up from piece norms. With only the constant mode "
                  f"kept (r = 0, not drawn on the log axis) it is {rf0:.4f}; it reaches zero with the complete basis.", data, lay))
    # input loadings on the layer-1 sphere + error loadings
    b = np.array(grad["input_loadings"]); gl = np.array(grad["error_loadings"])
    bscale = 1.0 / float(np.linalg.norm(b, axis=1).max())
    data = [sphere_trace("scene"),
            dict(type="scatter3d", mode="markers", scene="scene", x=r4(nrm[1][:, 0]), y=r4(nrm[1][:, 1]), z=r4(nrm[1][:, 2]),
                 marker=dict(size=1.6, color=AXISC), showlegend=False, hoverinfo="skip")]
    blabs = ["b_0"] + [f"b_{k}" for k in load["mode_indices"]]
    data += arrows3d([v * bscale for v in b], blabs, "scene", [MUTED] + [INK2] * 5)
    data += triad(np.zeros(3), "scene", labels=True)
    gm = float(np.abs(gl).max())
    quiet = [k for k in range(gl.shape[1]) if np.abs(gl[:, k]).max() == 0.0]
    data.append(dict(type="heatmap", z=np.round(gl, 8).tolist(), x=[f"n{k}" for k in range(gl.shape[1])],
                     y=["g_0 (constant)"] + [f"g_{k}" for k in load["mode_indices"]], xaxis="x", yaxis="y",
                     colorscale=DIV, zmin=-gm, zmax=gm, colorbar=cbar("g_i", 1.0, 0.8), hovertemplate="%{y}, %{x}: %{z:.2e}<extra></extra>"))
    lay = dict(scene=dict(grid_scene(1, 1, 1.6), domain=dict(x=[0, 0.4], y=[0, 1])), xaxis=dict(domain=[0.48, 0.93], title=dict(text="hidden neuron of the second MLP")),
               yaxis=dict(autorange="reversed"), margin=dict(l=0, r=20, t=40, b=60),
               annotations=[annot(f"b_i = U^T psi_i on the layer-1 sphere (x {bscale:.3f})", 0.0, 1.06, size=12, xanchor="left"),
                            annot("g_i = Delta^T psi_i per hidden neuron", 0.48, 1.06, size=12, xanchor="left")])
    s5.append(fig("The two factors of each rank-one piece g_i b_i^T: input loadings and error loadings",
                  "Left: b_i, the loading of the W1 input U = h1 on mode psi_i (constant mode and the five block modes), drawn "
                  "as arrows among the normalized layer-1 sites (grey); right: g_i, the loading of the backpropagated error "
                  "Delta = dL/dz on the same modes, per hidden neuron (shared symmetric scale). Individual pieces inside the "
                  "five-mode block depend on the basis; only their block sum is reported as invariant."
                  + (f" Neurons {', '.join('n' + str(k) for k in quiet)} have zero error loading on all six modes "
                     f"(no backpropagated error reaches them on these sites)." if quiet else ""), data, lay, 440))
    # unit profiles
    post = np.array(grad["post_block_energy"]); pre = np.array(grad["pre_block_energy"])
    def compress(E):
        E = np.asarray(E)
        out = np.concatenate([E[:, :nshow], E[:, nshow:].sum(1, keepdims=True)], axis=1)
        tot = out.sum(1, keepdims=True)
        return np.divide(out, tot, out=np.full_like(out, np.nan), where=tot > 0)
    cols = [gname(i) for i in range(nshow)] + ["all other modes"]
    Pp = compress(post)
    dead = [k for k in range(Pp.shape[0]) if np.isnan(Pp[k]).all()]
    zP = [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in Pp]
    data = [dict(type="heatmap", z=zP, x=cols, y=[f"n{k}" for k in range(Pp.shape[0])], colorscale=SEQ, zmin=0, zmax=1,
                 colorbar=cbar("share of energy"), hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>")]
    lay = dict(yaxis=dict(autorange="reversed", title=dict(text="hidden neuron")), margin=dict(l=70, r=30, t=30, b=90))
    s5.append(fig("Units as functions on sites: block profiles of the 16 hidden neurons (post-ReLU)",
                  "Each row is one neuron's activation over the sites, expanded in the layer-2 basis; cells are the share of "
                  "its squared norm in each group (rows sum to one). This is association with the basis at this scale, not a "
                  "mechanism; spread across blocks can come from rectifying a single spectral function."
                  + (f" Empty rows: neuron{'s' if len(dead) > 1 else ''} {', '.join('n' + str(k) for k in dead)} never "
                     f"activate{'' if len(dead) > 1 else 's'} on these sites (zero post-ReLU response), so no share is defined."
                     if dead else ""), data, lay, 460))
    eff = grad["learned_effect_neurons"]
    data = []
    for k, n in enumerate(eff):
        for which, E, dash in (("pre-ReLU", compress(pre)[n], None), ("post-ReLU", compress(post)[n], None)):
            data.append(dict(type="bar", x=cols, y=r4(E), name=f"n{n} {which}",
                             marker=dict(color=CAT[k], opacity=1.0 if which == "pre-ReLU" else 0.45,
                                         line=dict(color=CAT[k], width=1.5)),
                             hovertemplate=f"n{n} {which}, " + "%{x}: %{y:.3f}<extra></extra>"))
    lay = dict(barmode="group", yaxis=dict(type="linear", range=[0, 1], title=dict(text="share of squared norm")),
               xaxis=dict(title=dict(text="group")), legend=dict(orientation="h", y=1.12, x=0))
    s5.append(fig("Planted-model neurons where rectification spreads the profile: pre- versus post-ReLU",
                  f"Shown only for the neurons where rectification raises the effective number of groups, 1 / sum_B share_B^2, by more than 0.25 "
                  f"(n{', n'.join(map(str, eff))}; the selection rule is the data file's). Solid: the pre-activation's profile; light: "
                  f"the rectified response's. For these neurons rectification lowers the constant-mode share ("
                  + "; ".join(f"n{n}: {compress(pre)[n][0]:.2f} to {compress(post)[n][0]:.2f}" for n in eff)
                  + ") and redistributes the profile across more groups (THEORY Section 7.1).", data, lay))
    rc = grad["rectification"]
    before = np.array(rc["before_block_energy"]); after_ = np.array(rc["after_block_energy"])
    bc = np.concatenate([before[:nshow], [before[nshow:].sum()]]); ac = np.concatenate([after_[:nshow], [after_[nshow:].sum()]])
    data = [dict(type="bar", x=cols, y=r4(bc / bc.sum()), name=f"psi_{rc['mode_index']} (before)", marker=dict(color=CAT[0])),
            dict(type="bar", x=cols, y=r4(ac / ac.sum()), name=f"ReLU(psi_{rc['mode_index']}) (after)", marker=dict(color=CAT[1]))]
    lay = dict(barmode="group", yaxis=dict(type="linear", range=[0, 1], title=dict(text="share of squared norm")),
               xaxis=dict(title=dict(text="group")), legend=dict(orientation="h", y=1.12, x=0))
    s5.append(fig(f"Constructed rectification inset: one eigenfunction psi_{rc['mode_index']} and ReLU(psi_{rc['mode_index']}) in the same basis",
                  f"Constructed, not learned: psi_{rc['mode_index']} has all its energy in its own group; after pointwise "
                  f"rectification the energy spreads into the constant mode and many other groups, "
                  f"keeping only {ac[[i for i in range(nshow) if groups[i]['start'] <= rc['mode_index'] < groups[i]['stop']][0]] / ac.sum():.2f} "
                  f"in the original one. This is why 'detects one coherent structure' is stated for the pre-rectification input.",
                  data, lay))
    secs.append(dict(title="5. Gradient and units",
                     caption=f"THEORY Sections 7.1-7.2, [A3]: only the training loss and the training sites enter. The "
                             f"decomposition is of the gradient, not of an optimizer update, and it is exact at block "
                             f"granularity: constant + blocks + complement sum to the autograd gradient (max error "
                             f"{grad['checks']['block_sum_max_error']:.1e}).",
                     figures=s5, diagram=["D4", "D7"]))
    # ============================================================ closing note
    secs.append(dict(title="Deferred: context, carried versus read",
                     caption="Not computed on this toy. The context construction of THEORY Sections 6.3-6.4 (two "
                             "targets for the same past, a regularized canonical-correlation SVD, and the explained-variance "
                             "ladder) needs a sequence model and is deferred with the training slider, the second seed, the "
                             "output view and the attention toy; the diagram below states its objects and cautions.",
                     diagram_after="D6"))
    return secs, tall

SUB = str.maketrans("0123456789ijkr", "₀₁₂₃₄₅₆₇₈₉ᵢⱼₖᵣ")
TEXT_KEYS = {"title", "caption", "text", "name", "hovertemplate", "x", "y", "ticktext"}

def prettify(t):
    """Plain-text ASCII math -> THEORY notation in Unicode (β, λ, ψ, σ, ‖·‖, subscripts, ᵀ)."""
    for a_, b_ in (("beta_nn", "β_nn"), ("||", "‖"), ("pi/2", "π/2"), ("R^3", "ℝ³"), ("S^2", "S²"), ("^T", "ᵀ"),
                   ("D^-1/2", "D^−1/2"), ("lambda_i^t", "λᵢᵗ"), ("+/-", "±"), ("->", "→"), ("dL/dz", "∂ℒ/∂z"), ("<G_B, G_B'>", "⟨G_B, G_B′⟩")):
        t = t.replace(a_, b_)
    for a_, b_ in (("beta", "β"), ("lambda", "λ"), ("psi", "ψ"), ("sigma", "σ"), ("Delta", "Δ"), ("Pi_B", "Π_B")):
        t = re.sub(r"(?<![A-Za-z])" + a_ + r"(?![a-z])", b_, t)
    t = re.sub(r"(?<=[A-Za-zβλψσΔΠ])_(\d+|[ijkr])(?![A-Za-z0-9])", lambda m: m.group(1).translate(SUB), t)
    return t

def pretty_spec(o, key=None):
    if isinstance(o, dict):
        return {k: pretty_spec(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [pretty_spec(v, key) for v in o]
    if isinstance(o, str) and key in TEXT_KEYS:
        return prettify(o)
    return o

def svg_block(tag):
    path = next(DIAG.glob(f"{tag}_*.svg"))
    svg = path.read_text()
    svg = re.sub(r'<svg xmlns="http://www.w3.org/2000/svg" width="\d+" height="\d+"',
                 '<svg xmlns="http://www.w3.org/2000/svg"', svg, count=1)
    return f'<figure class="diagram" aria-label="explanatory diagram {tag}">{svg}</figure>'

def main():
    secs, tall = build()
    spec = dict(title="Seeing the construction on a d = 3 residual stream",
                subtitle="Planted toy model, CPU, all identities asserted in viz/toy.py; explanatory diagrams D1-D7 "
                         "precede the data. THEORY.md v1.5.",
                sections=[{k: v for k, v in s.items() if k not in ("diagram", "diagram_after")} for s in secs])
    spec = pretty_spec(spec)
    secs = [dict(s, **pretty_spec({k: s[k] for k in ("title", "caption") if k in s})) for s in secs]
    # the engine needs every section to be either prose-only or to have figures
    tmp = OUT.with_suffix(".tmp.html")
    report.write_report(spec, tmp)
    page = tmp.read_text()
    tmp.unlink()
    esc = lambda s: __import__("html").escape(s, quote=True)
    for s in secs:
        head = f'<section><h2>{esc(s["title"])}</h2><p>{esc(s.get("caption", ""))}</p>'
        assert page.count(head) == 1, s["title"]
        if "diagram" in s:
            tags = s["diagram"] if isinstance(s["diagram"], list) else [s["diagram"]]
            page = page.replace(head, head + "".join(svg_block(t) for t in tags))
        if "diagram_after" in s:
            page = page.replace(head, head + svg_block(s["diagram_after"]))
    for n, h in tall.items():
        old = f'<div class="plot" id="plot-{n}"></div>'
        assert page.count(old) == 1
        page = page.replace(old, f'<div class="plot" id="plot-{n}" style="height:{h}px"></div>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")

if __name__ == "__main__":
    main()
