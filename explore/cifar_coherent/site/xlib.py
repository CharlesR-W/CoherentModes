"""Shared helpers for experiment generators (site/experiments/<id>.py). See content/experiments/_template.html.

A generator is run in isolated mode, from explore/cifar_coherent/:

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/<id>.py r20s0b

and starts with the explicit path insert below (``-I`` drops the script directory from ``sys.path``):

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/
    import xlib

It writes only these files (``xlib.Out`` enforces the names):

    site/assets/x/<id>.json          chart objects for data-live="lines" / "heatmap" (data-src="assets/x/<id>.json#<key>")
    site/assets/x/<id>.facts.json    flat {name: string}; quoted in the fragment as {{fact:x.<id>.<name>}}
    site/assets/fig/x_<id>_*.png     static figures (matplotlib, ``xlib.mpl_style()``)

The statistics helpers (matched_jaccard, size_ceiling, grad_within_between, nice_range, lines, heat) are copies of
the ones in site/analysis.py (2026-11-19). They are copied, not imported, so that analysis.py and make_figures.py and
their ``-I`` entry points stay untouched. Nothing here enters the construction: these compare already-computed
partitions with each other, with labels (validation only) or with gradients (validation only).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent          # site/
ROOT = SITE.parent                              # explore/cifar_coherent/
DATA = ROOT / "data"
PALETTE = json.loads((SITE / "palette.json").read_text())
ID_RE = re.compile(r"[a-z][a-z0-9-]*")
SITES = ["stem"] + [f"b{i}" for i in range(1, 10)] + ["pool", "logits"]
ARROW = "→"


# ------------------------------------------------------------------------------------------ output
class Out:
    """Collects one experiment's charts, facts and figures and writes them (only) to the allowed paths.

        out = xlib.Out("markov")
        out.chart("lead", xlib.lines(...))            -> assets/x/markov.json#lead
        out.fact("own.real.defect", 0.4213, 2)        -> {{fact:x.markov.own.real.defect}} = "0.42"
        out.fig_path("ck")                            -> site/assets/fig/x_markov_ck.png
        out.write()
    """

    def __init__(self, xid: str):
        if not ID_RE.fullmatch(xid):
            raise SystemExit(f"experiment id '{xid}' must match {ID_RE.pattern}")
        self.id = xid
        self.charts: dict = {}
        self.facts: dict[str, str] = {}

    def chart(self, key: str, obj: dict):
        if key in self.charts:
            raise SystemExit(f"chart '{key}' written twice")
        self.charts[key] = clean(obj)
        return key

    def fact(self, name: str, value, digits: int | None = None, pct: bool = False):
        """Store a fact as a string. Numbers: `digits` decimals (pct: x100 with a % sign); ints: thousands separators.
        None or non-finite numbers are refused: a fact that cannot be computed must not be quoted."""
        if not re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)*", name):
            raise SystemExit(f"fact name '{name}': use lower-case dotted names (a-z, 0-9, _)")
        if name in self.facts:
            raise SystemExit(f"fact '{name}' written twice")
        self.facts[name] = fmt(value, digits, pct) if not isinstance(value, str) else value
        return self.facts[name]

    def fig_path(self, name: str) -> Path:
        if not ID_RE.fullmatch(name.replace("_", "-")):
            raise SystemExit(f"figure name '{name}': lower-case letters, digits, _")
        p = SITE / "assets" / "fig" / f"x_{self.id}_{name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self):
        xd = SITE / "assets" / "x"
        xd.mkdir(parents=True, exist_ok=True)
        (xd / f"{self.id}.json").write_text(json.dumps(self.charts, separators=(",", ":"), allow_nan=False))
        (xd / f"{self.id}.facts.json").write_text(json.dumps(self.facts, indent=1, ensure_ascii=False))
        print(f"wrote assets/x/{self.id}.json ({len(self.charts)} charts), assets/x/{self.id}.facts.json "
              f"({len(self.facts)} facts)")
        for k, v in self.facts.items():
            print(f"  x.{self.id}.{k:40s} {v}")


def fmt(v, digits: int | None = None, pct: bool = False) -> str:
    if isinstance(v, (bool, np.bool_)):
        raise SystemExit("a fact cannot be a bool; write the sentence's words in the fragment")
    if isinstance(v, (int, np.integer)) and digits is None and not pct:
        return f"{int(v):,}"
    v = float(v)
    if not math.isfinite(v):
        raise SystemExit("refusing a non-finite fact value")
    if pct:
        return f"{100 * v:.{digits if digits is not None else 0}f}%"
    s = f"{v:,.{digits if digits is not None else 3}f}"
    return s.lstrip("-") if float(s.replace(",", "")) == 0 else s        # never "-0.00"


def clean(x):
    """JSON-safe copy: numpy -> python, non-finite -> None (drawn as a gap), floats rounded to 5 significant digits."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else float(f"{x:.5g}") + 0.0
    return x


def missing(a, sentinel=-1.0):
    """Array with exported 'undefined' entries (sentinel, e.g. -1 in markov_summary.json, or NaN/None) as NaN."""
    a = np.array(a, dtype=float)
    a[np.isclose(a, sentinel)] = np.nan
    return a


# ------------------------------------------------------------------------------------------ chart objects
def lines(x, series, xlabel, ylabel, xscale="linear", yrange=(0, 1), colors="categorical", **kw):
    """A data-live="lines" object (DESIGN.md, Live mini-figures). series: [{"name", "y", "color"?, "dash"?: true|"dot",
    "width"?, "dots"?: radius, "noline"?: true (markers only), "band"?: [[lo...], [hi...]]}]. Optional kw:
    xticklabels (categorical x), refline {"y", "label"}, band, band_label, band_mode."""
    return dict(x=x, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yrange=list(yrange), series=series,
                colors=colors, **kw)


def heat(z, rows, cols, zlabel, rowlabel, collabel, **kw):
    """A data-live="heatmap" object. kw: zrange (default [0, 1]), slider {"label", "values"}, diverging, fmt."""
    kw.setdefault("zrange", [0, 1])
    return dict(rows=rows, cols=cols, z=z, zlabel=zlabel, rowlabel=rowlabel, collabel=collabel, **kw)


def nice_range(vals, pad=0.08, step=0.01):
    v = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in vals])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0, 1]
    lo, hi = min(v.min(), 0.0), max(v.max(), 0.0)
    span = max(hi - lo, step)
    return [float(np.floor((lo - pad * span) / step) * step), float(np.ceil((hi + pad * span) / step) * step)]


def pair_labels(sites=SITES):
    """Adjacent-pair tick labels: stem->b1, ..., pool->logits."""
    return [f"{a}{ARROW}{b}" for a, b in zip(sites[:-1], sites[1:])]


def step_label(s: int) -> str:
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else f"{s:,}"


# ------------------------------------------------------------------------------------------ statistics (copies)
def _jaccard(A, b, ra, rb):
    """A (P, N) labels, b (N,) labels, -1 = unassigned. Returns J (P, ra, rb) float64."""
    P = A.shape[0]
    va = A >= 0
    mask = va & (b >= 0)[None]
    key = (np.arange(P)[:, None] * (ra * rb) + A * rb + b[None])[mask]
    inter = np.bincount(key, minlength=P * ra * rb).reshape(P, ra, rb)
    ka = (np.arange(P)[:, None] * ra + A)[va]
    sa = np.bincount(ka, minlength=P * ra).reshape(P, ra)
    sb = np.bincount(b[b >= 0], minlength=rb)
    union = sa[:, :, None] + sb[None, None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def _hungarian_mean(J):
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(J, maximize=True)
    return J[rows, cols].sum() / min(J.shape), rows, cols


def matched_jaccard(a, b, ra, rb, perms=None):
    """Matched Jaccard of labelings a, b (App 3 definition). Returns (score, [(i, j, J_ij)], null scores or None).
    Null: a's label vector permuted by each row of `perms` (sizes and unassigned count kept)."""
    J = _jaccard(a[None], b, ra, rb)[0]
    score, rows, cols = _hungarian_mean(J)
    pairs = [(int(i), int(j), float(J[i, j])) for i, j in zip(rows, cols)]
    null = None
    if perms is not None:
        Jn = _jaccard(a[perms], b, ra, rb)
        null = np.array([_hungarian_mean(Jn[p])[0] for p in range(len(perms))])
    return float(score), pairs, null


def size_ceiling(a, b, ra, rb):
    """Largest matched Jaccard any labelings with these set sizes could reach."""
    sa = np.bincount(a[a >= 0], minlength=ra).astype(float)
    sb = np.bincount(b[b >= 0], minlength=rb).astype(float)
    hi = np.maximum(sa[:, None], sb[None])
    C = np.where(hi > 0, np.minimum(sa[:, None], sb[None]) / np.maximum(hi, 1), 0.0)
    return float(_hungarian_mean(C)[0])


def null_summary(null):
    return float(null.mean()), float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))


def grad_within_between(G, l, r, order=None):
    """Pair-weighted mean gradient cosine within / between sets of labeling l (-1 excluded). `order` (P, N): rows
    are uniform random permutations of 0..N-1; restricted to the assigned points each gives a random arrangement that
    receives the observed labels. Returns arrays (1 + P,) within, between: observed first, then the nulls."""
    assigned = l >= 0
    I = np.flatnonzero(assigned)
    la = l[I].astype(np.int64)
    na = len(I)
    a = assigned.astype(np.float32)
    diag = float(np.diagonal(G)[I].sum(dtype=np.float64))
    total = float(a @ (G @ a))
    sq = float((np.bincount(la, minlength=r).astype(np.float64) ** 2).sum())
    wc, bc = sq - na, na * na - sq
    Q = I[None]
    if order is not None:
        Q = np.vstack([Q, order[assigned[order]].reshape(len(order), na)])
    P = Q.shape[0]
    H = np.zeros((len(l), P * r), np.float32)
    H[Q.ravel(), (np.arange(P)[:, None] * r + la[None]).ravel()] = 1
    win = ((H * (G @ H)).sum(0, dtype=np.float64)).reshape(P, r).sum(1)
    within = (win - diag) / wc if wc > 0 else np.full(P, np.nan)
    between = (total - win) / bc if bc > 0 else np.full(P, np.nan)
    return within, between


# ------------------------------------------------------------------------------------------ matplotlib style
def mpl_style():
    """Apply the site's figure style (as analysis.py / make_figures.py) and return (plt, palette).
    Use palette colours only: sets for categorical series, ordinal for ordered series, sequential for magnitudes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f_ in sorted((SITE / "figdata" / "fonts").glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(f_))
    P = PALETTE
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
                         "font.size": 8.5, "axes.edgecolor": P["mute"], "axes.labelcolor": P["fg"], "text.color": P["fg"],
                         "xtick.color": P["mute"], "ytick.color": P["mute"], "xtick.labelcolor": P["fg"],
                         "ytick.labelcolor": P["fg"], "axes.titlesize": 9.5, "axes.titlecolor": P["fg"],
                         "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": P["surface"], "axes.facecolor": P["surface"],
                         "savefig.facecolor": P["surface"], "axes.grid": True, "grid.color": P["line"],
                         "grid.linewidth": 0.6, "lines.linewidth": 1.4, "savefig.dpi": 200})
    return plt, P
