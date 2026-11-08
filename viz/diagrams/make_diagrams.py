"""Explanatory diagrams D1-D7 for THEORY.md (see viz/PLAN.md, "Explanatory diagrams").

Run from the repository root:
    uv run --with numpy python -I viz/diagrams/make_diagrams.py
Writes viz/diagrams/D*.svg. Every number drawn on D2, D3, D4, D5 and D7 is computed
here from small constructed toys (stated in each caption as constructed), and the
identities the figures depict are asserted before anything is drawn.
Palette and type: viz/diagrams/STYLE.md.
"""
import math
import os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette (STYLE.md)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MID = "#f0efec"          # neutral midpoint of the diverging ramp, also the "zero" fill
NEG = "#2a78d6"          # diverging negative pole (blue, categorical slot 1)
POS = "#e34948"          # diverging positive pole (red, categorical slot 8)
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]   # identity: first three slots only
SEQ_STOPS = ["#f0efec", "#898781", "#0b0b0b"]  # unsigned magnitude: neutral ink ramp
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
FS = 18        # body label size (px); nothing below 15
FS_SMALL = 16

# ---------------------------------------------------------------- colour maths (OKLab)
def _srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _lin_to_srgb(c):
    c = min(max(c, 0.0), 1.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

def hex_to_oklab(h):
    r, g, b = (_srgb_to_lin(int(h[i:i + 2], 16) / 255) for i in (1, 3, 5))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (np.cbrt(v) for v in (l, m, s))
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])

def oklab_to_hex(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return "#" + "".join(f"{round(_lin_to_srgb(v) * 255):02x}" for v in (r, g, bb))

def mix(h1, h2, t):
    return oklab_to_hex((1 - t) * hex_to_oklab(h1) + t * hex_to_oklab(h2))

def div_color(v, vmax):
    """Signed value -> blue (negative) / neutral grey (zero) / red (positive)."""
    t = 0.0 if vmax == 0 else max(-1.0, min(1.0, v / vmax))
    return mix(MID, POS if t > 0 else NEG, abs(t))

def seq_color(t):
    """Unsigned magnitude in [0, 1] -> neutral ink ramp (light -> dark)."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return mix(SEQ_STOPS[0], SEQ_STOPS[1], t / 0.5)
    return mix(SEQ_STOPS[1], SEQ_STOPS[2], (t - 0.5) / 0.5)

def num(v, nd=2, sign=True):
    """Format with a typographic minus (and an explicit plus when sign=True)."""
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "−")

# ---------------------------------------------------------------- SVG primitives
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _parse(s):
    """Tiny math markup: a_{ij} subscript, a^{T} superscript (braces optional for one char)."""
    out, buf, i = [], "", 0
    while i < len(s):
        ch = s[i]
        if ch in "_^" and i + 1 < len(s):
            if buf:
                out.append((buf, "n"))
                buf = ""
            if s[i + 1] == "{":
                j = s.index("}", i + 2)
                out.append((s[i + 2:j], "sub" if ch == "_" else "sup"))
                i = j + 1
            else:
                out.append((s[i + 1], "sub" if ch == "_" else "sup"))
                i += 2
            continue
        buf += ch
        i += 1
    if buf:
        out.append((buf, "n"))
    return out

def text(x, y, s, size=FS, anchor="start", fill=INK, weight=None, italic=False, opacity=None):
    attrs = (f'x="{x:.1f}" y="{y:.1f}" font-family=\'{FONT}\' font-size="{size}" '
             f'fill="{fill}" text-anchor="{anchor}"')
    if weight:
        attrs += f' font-weight="{weight}"'
    if italic:
        attrs += ' font-style="italic"'
    if opacity is not None:
        attrs += f' opacity="{opacity}"'
    parts, level = [], 0.0
    for seg, mode in _parse(s):
        target = {"n": 0.0, "sub": 0.3, "sup": -0.42}[mode] * size
        dy = target - level
        level = target
        fs = f' font-size="{size * 0.72:.1f}"' if mode != "n" else ""
        dya = f' dy="{dy:.1f}"' if abs(dy) > 1e-9 else ""
        parts.append(f"<tspan{fs}{dya}>{esc(seg)}</tspan>")
    return f"<text {attrs}>{''.join(parts)}</text>"

def line(x1, y1, x2, y2, stroke=INK2, width=1.5, dash=None, opacity=None, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{width:.2f}" stroke-linecap="{cap}"{d}{o}/>')

def rect(x, y, w, h, fill="none", stroke=None, width=1.0, rx=0, dash=None, opacity=None):
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}"{s}{d}{o}/>')

def circle(cx, cy, r, fill="none", stroke=None, width=1.0, dash=None, opacity=None):
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{s}{d}{o}/>'

def path(d, stroke=INK2, width=1.5, fill="none", dash=None, opacity=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}" '
            f'stroke-linecap="round" stroke-linejoin="round"{da}{o}/>')

def _head(x2, y2, ang, color, size):
    a1, a2 = ang + math.radians(152), ang - math.radians(152)
    p = [(x2, y2), (x2 + size * math.cos(a1), y2 + size * math.sin(a1)),
         (x2 + size * math.cos(a2), y2 + size * math.sin(a2))]
    return f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in p)}" fill="{color}"/>'

def arrow(x1, y1, x2, y2, color=INK2, width=1.8, head=11, dash=None):
    ang = math.atan2(y2 - y1, x2 - x1)
    xe, ye = x2 - 0.8 * head * math.cos(ang), y2 - 0.8 * head * math.sin(ang)
    return line(x1, y1, xe, ye, color, width, dash) + _head(x2, y2, ang, color, head)

def curved_arrow(x1, y1, cx, cy, x2, y2, color=INK2, width=1.8, head=11, dash=None):
    ang = math.atan2(y2 - cy, x2 - cx)
    xe, ye = x2 - 0.8 * head * math.cos(ang), y2 - 0.8 * head * math.sin(ang)
    return (path(f"M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {xe:.1f},{ye:.1f}", color, width, dash=dash)
            + _head(x2, y2, ang, color, head))

def brace_h(x1, x2, y, up=True, color=INK2, depth=9):
    """Horizontal square bracket over (up=True) or under a span."""
    s = -1 if up else 1
    return path(f"M{x1:.1f},{y - s * depth:.1f} L{x1:.1f},{y:.1f} L{x2:.1f},{y:.1f} L{x2:.1f},{y - s * depth:.1f}",
                color, 1.5)

def brace_v(x, y1, y2, right=True, color=INK2, depth=9):
    s = 1 if right else -1
    return path(f"M{x - s * depth:.1f},{y1:.1f} L{x:.1f},{y1:.1f} L{x:.1f},{y2:.1f} L{x - s * depth:.1f},{y2:.1f}",
                color, 1.5)

def half_disc(cx, cy, r, left_fill, right_fill, stroke=INK2):
    return (f'<path d="M{cx:.1f},{cy - r:.1f} A{r},{r} 0 0 0 {cx:.1f},{cy + r:.1f} Z" fill="{left_fill}"/>'
            f'<path d="M{cx:.1f},{cy - r:.1f} A{r},{r} 0 0 1 {cx:.1f},{cy + r:.1f} Z" fill="{right_fill}"/>'
            + circle(cx, cy, r, stroke=stroke, width=1.2) + line(cx, cy - r, cx, cy + r, stroke, 1.2))

def sphere(cx, cy, r):
    """Unit-sphere glyph: outline plus a hairline equator (no shading)."""
    return (circle(cx, cy, r, fill="#f6f5f2", stroke=AXIS, width=1.5)
            + f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{r:.1f}" ry="{r * 0.28:.1f}" fill="none" '
              f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 4"/>')

def diverging_legend(x, y, w, label_neg="−", label_pos="+", title=None):
    els, n = [], 24
    for i in range(n):
        v = -1 + 2 * (i + 0.5) / n
        els.append(rect(x + i * w / n, y, w / n + 0.6, 12, fill=div_color(v, 1)))
    els.append(text(x - 8, y + 11, label_neg, FS_SMALL, "end", INK2))
    els.append(text(x + w + 8, y + 11, label_pos, FS_SMALL, "start", INK2))
    els.append(text(x + w / 2, y + 30, "0", FS_SMALL, "middle", INK2))
    if title:
        els.append(text(x + w / 2, y - 8, title, FS_SMALL, "middle", INK2))
    return "".join(els)

def seq_legend(x, y, w, title=None, lo="0", hi="max"):
    els, n = [], 24
    for i in range(n):
        els.append(rect(x + i * w / n, y, w / n + 0.6, 12, fill=seq_color((i + 0.5) / n)))
    els.append(text(x - 8, y + 11, lo, FS_SMALL, "end", INK2))
    els.append(text(x + w + 8, y + 11, hi, FS_SMALL, "start", INK2))
    if title:
        els.append(text(x + w / 2, y - 8, title, FS_SMALL, "middle", INK2))
    return "".join(els)

def figure(name, width, height, tag, title, theory_ref, caption, body):
    """Frame: title row, body, hairline, two-line caption naming the THEORY section."""
    assert len(caption) == 2
    for c_ in caption:   # rendered length estimate (markup stripped); ~8.9 px per char at 18 px
        plain = c_.replace("_{", "").replace("^{", "").replace("}", "").replace("_", "").replace("^", "")
        assert len(plain) * 8.9 < width - 64, (name, len(plain), c_)
    cap_y = height - 62
    els = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{tag}-title {tag}-desc">',
        f"<title id=\"{tag}-title\">{esc(tag + ' ' + title)}</title>",
        f"<desc id=\"{tag}-desc\">{esc(caption[0] + ' ' + caption[1])}</desc>",
        rect(0, 0, width, height, fill=SURFACE),
        text(32, 44, tag, 24, weight=700),
        text(32 + 16 * len(tag) + 14, 44, title, 24, weight=600),
        text(width - 32, 44, theory_ref, FS, "end", MUTED),
        body,
        line(32, cap_y - 26, width - 32, cap_y - 26, GRID, 1),
        text(32, cap_y, caption[0], FS, fill=INK2),
        text(32, cap_y + 28, caption[1], FS, fill=INK2),
        "</svg>",
    ]
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(els) + "\n")
    print("wrote", name)

# ---------------------------------------------------------------- shared numerics
def sinkhorn(K, tol=1e-13):
    v = np.ones(len(K))
    for _ in range(100000):
        v = np.sqrt(v / (K @ v))
        if np.abs(v * (K @ v) - 1).max() < tol:
            break
    P = v[:, None] * K * v[None, :]
    assert np.allclose(P.sum(1), 1, atol=1e-10) and np.allclose(P, P.T)
    return P

def sphere_P(Phi, beta):
    Ph = Phi / np.linalg.norm(Phi, axis=1, keepdims=True)
    K = np.exp(beta * (Ph @ Ph.T - 1.0))   # shifted kernel: same P after Sinkhorn
    return K, sinkhorn(K)

def eig_sorted(P):
    w, V = np.linalg.eigh(P)
    w, V = w[::-1], V[:, ::-1].copy()
    N = len(P)
    V[:, 0] = np.ones(N) / math.sqrt(N)            # exact constant mode
    for i in range(1, N):                          # sign: largest-|value| site positive
        if V[np.argmax(np.abs(V[:, i])), i] < 0:
            V[:, i] *= -1
    return w, V

def _frame(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return n, e1, np.cross(n, e1)

def dirn(c, ang_deg, az):
    n, e1, e2 = _frame(c)
    a = math.radians(ang_deg)
    return math.cos(a) * n + math.sin(a) * (math.cos(az) * e1 + math.sin(az) * e2)

def rodrigues(v, k, th):
    k = k / np.linalg.norm(k)
    return v * math.cos(th) + np.cross(k, v) * math.sin(th) + k * (k @ v) * (1 - math.cos(th))

def toy16():
    """16 sites in R^3: clusters A, B, C (3 each, related by 120-degree rotations, so their
    two contrasts form a near-degenerate block) and D (4), E (3), 115 degrees away.
    Norms vary so that the normalization of [A2] visibly drops something."""
    n1 = np.array([0.3, 1.0, 0.4]); n1 /= np.linalg.norm(n1)
    n2 = dirn(n1, 115, 0.0)
    cA = dirn(n1, 30, math.radians(90))
    A = [dirn(cA, 3, a) for a in (0.3, 2.4, 4.4)]
    S1 = A + [rodrigues(p, n1, 2 * math.pi / 3) for p in A] + [rodrigues(p, n1, 4 * math.pi / 3) for p in A]
    D = [dirn(dirn(n2, 34, math.radians(90)), 3.5, a) for a in (0, 1.6, 3.2, 4.8)]
    E = [dirn(dirn(n2, 34, math.radians(270)), 3, a) for a in (0.5, 2.6, 4.6)]
    Xhat = np.array(S1 + D + E)
    norms = np.array([1.3, 0.9, 1.1, 1.0, 1.4, 0.8, 1.2, 0.9, 1.1, 1.5, 1.0, 1.2, 0.8, 1.1, 1.3, 0.9])
    H = Xhat * norms[:, None]
    groups = np.array([0] * 3 + [1] * 3 + [2] * 3 + [3] * 4 + [4] * 3)  # display only
    return H, groups

BETA16 = 5.0
H16, GROUPS16 = toy16()
K16, P16 = sphere_P(H16, BETA16)
LAM16, PSI16 = eig_sorted(P16)
N16 = len(H16)
BLOCKS16 = [[1], [2], [3, 4]]       # checked below with the PLAN's block rule

def _check_blocks():
    lam = LAM16
    # pair (3,4): internal gap below 0.1 x the gaps to its neighbours
    g_in = lam[3] - lam[4]
    assert g_in < 0.1 * (lam[2] - lam[3]) and g_in < 0.1 * (lam[4] - lam[5]), lam[:6]
    # singletons 1, 2: their neighbour gaps are not small relative to anything around them
    assert lam[1] - lam[2] > 0.05 and lam[2] - lam[3] > 0.2
_check_blocks()

# ================================================================ D1 sites and views
def d1():
    W, Hh = 1200, 690
    els = []
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    xs = [150 + i * 72 for i in range(6)]
    t_idx = 5                       # the highlighted site x = (doc, t): the last token shown
    rows = {"out": 132, 2: 222, 1: 312, 0: 402}
    y_tok = 478
    els.append(text(48, rows["out"] + 6, "p(·|x)", FS, "start", INK2))
    for L in (2, 1, 0):
        els.append(text(48, rows[L] + 6, f"h^{L}" + (" = h^L" if L == 1 else ""), FS, "start", INK2))
    els.append(text(48, y_tok + 6, "token", FS, "start", INK2))
    xh = xs[t_idx]
    els.append(rect(xh - 28, rows["out"] - 32, 56, y_tok - rows["out"] + 56, fill="#ecebe6", rx=10))
    for i, x in enumerate(xs):
        hi = i == t_idx
        col, wd = (INK, 2.4) if hi else (AXIS, 1.4)
        els.append(line(x, rows[0], x, rows["out"], col, wd))
        for L in (0, 1, 2):
            els.append(circle(x, rows[L], 13, fill="#ffffff" if hi else SURFACE, stroke=col, width=wd))
        els.append(rect(x - 13, rows["out"] - 13, 26, 26, fill="#ffffff" if hi else SURFACE, stroke=col, width=wd, rx=4))
        els.append(text(x, y_tok + 6, tokens[i], FS, "middle", INK if hi else MUTED, weight=700 if hi else None))
    # past window at layer L = 1
    els.append(rect(xs[0] - 24, rows[1] - 24, xs[t_idx - 1] - xs[0] + 48, 48, stroke=INK2, width=1.5, rx=12, dash="5 4"))
    els.append(text((xs[0] + xs[t_idx - 1]) / 2, rows[1] + 46, "past window of x (positions < t)", FS_SMALL, "middle", INK2))
    # the site, under the column
    els.append(text(xh, y_tok + 40, "site x = (doc, t)", FS, "middle", INK, weight=700))
    els.append(text(xh, y_tok + 66, "h^0(x), h^1(x), h^2(x): the same site at every layer [A1]", FS, "middle", INK))
    # view cards
    cx0, cw, ch = 716, 452, 104
    cy = [74, 190, 306, 422]
    def card(y, title, formula, n1, n2):
        tx = cx0 + 112
        return (rect(cx0, y, cw, ch, fill="#ffffff", stroke=GRID, width=1.2, rx=10)
                + text(tx, y + 26, title, FS, "start", INK, weight=700)
                + text(tx, y + 51, formula, FS, "start", INK)
                + text(tx, y + 74, n1, FS_SMALL, "start", INK2)
                + text(tx, y + 94, n2, FS_SMALL, "start", INK2))
    y = cy[0]
    els.append(card(y, "output view", "φ(x) = √p(·|x)", "a point on the positive orthant of the unit",
                    "sphere; the simplex (dashed) is not used"))
    ox, oy, R = cx0 + 22, y + 88, 66
    els.append(line(ox, oy, ox + R + 8, oy, AXIS, 1.2) + line(ox, oy, ox, oy - R - 8, AXIS, 1.2))
    els.append(line(ox + R, oy, ox, oy - R, MUTED, 1.4, dash="4 3"))
    els.append(path(f"M{ox + R},{oy} A{R},{R} 0 0 0 {ox},{oy - R}", INK, 2.2))
    a = math.radians(40)
    els.append(circle(ox + R * math.cos(a), oy - R * math.sin(a), 5.5, fill=INK))
    y = cy[1]
    els.append(card(y, "geometric view, layer L", "φ(x) = h^L(x),  φ̂ = φ / ‖φ‖",
                    "a point on the unit sphere [A2];", "the norm ‖h^L(x)‖ is dropped"))
    gx, gy = cx0 + 56, y + ch / 2
    els.append(sphere(gx, gy, 38) + circle(gx + 16, gy - 18, 5.5, fill=INK))
    y = cy[2]
    els.append(card(y, "action view, layer L", "φ(x) = J^L(x) = ∂h^{L+1}(x) / ∂h^L(x)",
                    "local Jacobian at x (signed entries),", "sketched by random probes, normalized"))
    J = [[0.8, -0.3, 0.1], [0.2, 0.6, -0.5], [-0.1, 0.4, 0.9]]
    for r in range(3):
        for c in range(3):
            els.append(rect(cx0 + 24 + c * 22, y + 20 + r * 22, 20, 20, fill=div_color(J[r][c], 1), rx=2))
    y = cy[3]
    els.append(card(y, "context view, layer L", "φ(x) = past window of ĥ^L at x",
                    "lag weights β_j ≥ 0 [P3]; a product of sphere", "kernels is again a sphere kernel"))
    for k, rr in enumerate((6, 8.5, 11)):
        els.append(circle(cx0 + 26 + k * 28, y + 46, rr, fill="#ffffff", stroke=INK2, width=1.4))
    els.append(text(cx0 + 54, y + 84, "β_3  β_2  β_1", FS_SMALL, "middle", INK2))
    # arrows: output cell -> output view; h^1(x) -> the three layer-L views
    els.append(arrow(xh + 16, rows["out"], cx0 - 8, cy[0] + 58, INK2))
    for k, bend in ((1, -10), (2, 20), (3, 60)):
        ty = cy[k] + 52
        els.append(curved_arrow(xh + 16, rows[1], (xh + cx0) / 2, (rows[1] + ty) / 2 + bend, cx0 - 8, ty, INK2))
    els.append(text(cx0 + cw / 2, 566, "every view → K_β[x, y] = exp(β ⟨φ̂(x), φ̂(y)⟩)", FS, "middle", INK, weight=700))
    figure("D1_sites_views.svg", W, Hh, "D1", "Sites and views", "THEORY §1–2",
           ["A site x is one place where the network has an activation; the same x at layers 0, 1, 2 is the same site [A1].",
            "Each view φ(x) is normalized onto a sphere and compared by a dot product [A2]; the output view is √p, not the simplex."],
           "".join(els))

# ================================================================ D2 view -> operator
def _display_frame(Xhat):
    """Orthographic view along the mean direction; screen axes = top-2 PCA directions."""
    m = Xhat.mean(0); m /= np.linalg.norm(m)
    Y = Xhat - np.outer(Xhat @ m, m)
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    return m, Vt[0], Vt[1]

def d2():
    W, Hh = 1200, 880
    els = []
    Xhat = H16 / np.linalg.norm(H16, axis=1, keepdims=True)
    m, e1, e2 = _display_frame(Xhat)
    # A: sites on the sphere
    cx, cy, R = 150, 262, 100
    els.append(text(cx, 110, "the view φ̂(x)", FS, "middle", INK, weight=700))
    els.append(sphere(cx, cy, R))
    for p in Xhat:
        els.append(circle(cx + R * (p @ e1), cy - R * (p @ e2), 4.5, fill=INK))
    els.append(text(cx, cy + R + 34, "N = 16 sites", FS, "middle", INK2))
    # B: kernel K
    cs, n = 13, N16
    gx, gy = 300, 160
    Kd = K16 / K16.max()
    els.append(text(gx + n * cs / 2 + 30, 110, "K_β[x, y] = exp(β ⟨φ̂(x), φ̂(y)⟩)", FS, "middle", INK, weight=700))
    for i in range(n):
        for j in range(n):
            els.append(rect(gx + j * cs, gy + i * cs, cs - 1, cs - 1, fill=seq_color(Kd[i, j])))
    rs = K16.sum(1)
    for i in range(n):
        els.append(rect(gx + n * cs + 8, gy + i * cs + 2, 44 * rs[i] / rs.max(), cs - 4, fill=INK2, rx=1))
    els.append(text(gx + n * cs + 8, gy - 10, "Σ_y K", FS_SMALL, "start", INK2))
    els.append(text(gx + n * cs + 8, gy + n * cs + 24, "row sums", FS_SMALL, "start", INK2))
    els.append(text(gx + n * cs + 8, gy + n * cs + 44, "unequal", FS_SMALL, "start", INK2))
    els.append(text(gx, gy + n * cs + 24, "sites x, y sorted", FS_SMALL, "start", INK2))
    els.append(text(gx, gy + n * cs + 44, "for display; β = 5", FS_SMALL, "start", INK2))
    # Sinkhorn arrow
    ax0, ax1, ay = gx + n * cs + 70, gx + n * cs + 170, gy + n * cs / 2
    els.append(arrow(ax0, ay, ax1, ay, INK, 2.2, 13))
    els.append(text((ax0 + ax1) / 2, ay - 14, "Sinkhorn", FS, "middle", INK, weight=700))
    els.append(text((ax0 + ax1) / 2, ay + 30, "P = D_v K D_v", FS, "middle", INK))
    # C: operator P
    px = ax1 + 22
    Pd = P16 / P16.max()
    els.append(text(px + n * cs / 2 + 30, 110, "P: symmetric, doubly stochastic", FS, "middle", INK, weight=700))
    for i in range(n):
        for j in range(n):
            els.append(rect(px + j * cs, gy + i * cs, cs - 1, cs - 1, fill=seq_color(Pd[i, j])))
    for i in range(n):
        els.append(rect(px + n * cs + 8, gy + i * cs + 2, 44, cs - 4, fill=INK2, rx=1))
        els.append(rect(px + i * cs + 2, gy + n * cs + 8, cs - 4, 30, fill=INK2, rx=1))
    els.append(text(px + n * cs + 8, gy - 10, "Σ_y P", FS_SMALL, "start", INK2))
    els.append(text(px + n * cs + 8, gy + n * cs + 24, "P1 = 1", FS_SMALL, "start", INK2))
    els.append(text(px - 4, gy + n * cs + 58, "column sums = 1 (P = P^T)", FS_SMALL, "start", INK2))
    # C notes
    nx = px + n * cs + 70
    for k, s in enumerate(["0 ≤ λ_i ≤ 1", "λ_0 = 1 on ψ_0 = 1/√N", "same measure on sites", "for every view [M3]"]):
        els.append(text(nx, gy + 30 + 28 * k, s, FS, "start", INK if k < 2 else INK2))
    els.append(seq_legend(ax0 + 14, ay + 78, 72, "entry / max", "0", "max"))
    # D: eigenfunctions as patterns over the sites
    ey0, ex0, ec = 512, 140, 26
    els.append(text(40, ey0 - 30, "P ψ_i = λ_i ψ_i: each eigenfunction is a pattern over the same 16 sites", FS, "start", INK, weight=700))
    rows = [0, 1, 2, 3, 4, 5]
    vmax = np.abs(PSI16[:, rows]).max()
    for r, i in enumerate(rows):
        y = ey0 + r * 36
        els.append(text(ex0 - 16, y + 19, f"ψ_{i}", FS, "end", INK))
        for x in range(n):
            els.append(rect(ex0 + x * ec, y, ec - 2, 28, fill=div_color(PSI16[x, i], vmax), rx=2))
        els.append(text(ex0 + n * ec + 14, y + 19, f"λ_{i} = {LAM16[i]:.2f}", FS, "start", INK))
    bx = ex0 + n * ec + 128
    lab = {0: "constant", 1: "B_1", 2: "B_2", 5: "past the gap"}
    for i, s in lab.items():
        els.append(text(bx + 16, ey0 + i * 36 + 19, s, FS_SMALL, "start", INK2))
    els.append(brace_v(bx + 6, ey0 + 3 * 36 + 2, ey0 + 4 * 36 + 26, True, INK2, 7))
    els.append(text(bx + 16, ey0 + 3 * 36 + 34, "B_3", FS_SMALL, "start", INK2))
    els.append(diverging_legend(ex0 + 128, ey0 + 6 * 36 + 18, 160, "−", "+", None))
    # spectrum
    sx0, sy0, sh, bw = 836, 718, 190, 17
    els.append(text(sx0 - 20, ey0 - 30, "spectrum of P", FS, "start", INK, weight=700))
    els.append(line(sx0 - 6, sy0, sx0 + n * (bw + 4) + 4, sy0, AXIS, 1.2))
    els.append(line(sx0 - 6, sy0 - sh, sx0 + n * (bw + 4) + 4, sy0 - sh, GRID, 1, dash="3 3"))
    els.append(text(sx0 - 12, sy0 + 5, "0", FS_SMALL, "end", INK2))
    els.append(text(sx0 - 12, sy0 - sh + 5, "1", FS_SMALL, "end", INK2))
    for i in range(n):
        h = sh * LAM16[i]
        els.append(rect(sx0 + i * (bw + 4), sy0 - h, bw, max(h, 1.5), fill=MUTED if i == 0 else INK2, rx=2))
    els.append(text(sx0 + n * (bw + 4) / 2, sy0 + 28, "λ_0 … λ_{15}", FS_SMALL, "middle", INK2))
    gxp = sx0 + 5 * (bw + 4) - 2
    els.append(text(gxp + 6, sy0 - 60, "gap", FS_SMALL, "start", INK2))
    els.append(line(gxp, sy0 - sh * LAM16[4], gxp, sy0 - 6, INK2, 1.2, dash="3 3"))
    for blk, name in zip(BLOCKS16, ("B_1", "B_2", "B_3")):
        x1 = sx0 + blk[0] * (bw + 4); x2 = sx0 + blk[-1] * (bw + 4) + bw
        yb = sy0 - sh * LAM16[blk[0]] - 10
        els.append(brace_h(x1, x2, yb, True, INK2, 6))
        els.append(text((x1 + x2) / 2, yb - 12, name, FS_SMALL, "middle", INK2))
    figure("D2_view_to_operator.svg", W, Hh, "D2", "From a view to an operator", "THEORY §2–3",
           ["A view gives a positive kernel K_β on sites; Sinkhorn scaling makes it a symmetric P whose rows and columns all sum to 1 [M3].",
            "The eigenfunctions ψ_i of P are patterns over the same sites; spectral gaps delimit blocks B. Constructed 16-site toy, β = 5."],
           "".join(els))

# ================================================================ D3 loadings
def d3():
    W, Hh = 1200, 920
    els = []
    Hm, Psi = H16, PSI16
    A = Hm.T @ Psi                      # column i is a_i = H^T psi_i
    assert np.allclose(Psi @ A.T, Hm)   # H = sum_i psi_i a_i^T, exact
    for r in range(N16):
        Hr = Psi[:, :r + 1] @ A[:, :r + 1].T
        assert np.isclose(np.sum((Hm - Hr) ** 2), np.sum(A[:, r + 1:] ** 2))
    i = 1
    # left: psi_1 as bars
    lx, base, sc = 70, 290, 115 / np.abs(Psi[:, i]).max()
    els.append(text(40, 100, "a mode ψ_1 ∈ ℝ^N: a function on sites", FS, "start", INK, weight=700))
    els.append(text(40, 126, "eigenfunction of P from D2, λ_1 = %.2f" % LAM16[i], FS_SMALL, "start", INK2))
    els.append(line(lx - 10, base, lx + N16 * 28 + 4, base, AXIS, 1.2))
    for x in range(N16):
        v = Psi[x, i]
        h = sc * v
        els.append(rect(lx + x * 28, base - max(h, 0), 22, abs(h), fill=div_color(v, np.abs(Psi[:, i]).max()), rx=2))
    els.append(text(lx + N16 * 14, base + 150, "sites x (sorted for display)", FS_SMALL, "middle", INK2))
    els.append(text(lx - 16, base + 5, "0", FS_SMALL, "end", INK2))
    # arrow between the faces
    els.append(arrow(560, 300, 680, 300, INK, 2.2, 13))
    els.append(text(620, 282, "a_1 = (H^L)^T ψ_1", FS, "middle", INK, weight=700))
    els.append(text(620, 330, "summed cross-", FS_SMALL, "middle", INK2))
    els.append(text(620, 350, "moment", FS_SMALL, "middle", INK2))
    # right: activation space, viewed on the plane of a_0 and a_1, tilted 28 degrees
    els.append(text(720, 100, "its loading a_1 ∈ ℝ^d, in activation space", FS, "start", INK, weight=700))
    els.append(text(720, 126, "raw activations h^L(x) (d = 3), coloured by ψ_1(x)", FS_SMALL, "start", INK2))
    u1 = A[:, 0] / np.linalg.norm(A[:, 0])
    u2 = A[:, 1] - (A[:, 1] @ u1) * u1; u2 /= np.linalg.norm(u2)
    nrm = np.cross(u1, u2); tilt = math.radians(28)
    sx_, sy_ = u2, math.cos(tilt) * u1 + math.sin(tilt) * nrm
    ox, oy = 930, 400
    pts_r = max(np.linalg.norm(Hm, axis=1))
    s = 120 / pts_r
    f = 0.5 if max(np.linalg.norm(A[:, :2], axis=0)) * 0.5 <= 2.3 * pts_r else 0.25
    P2 = lambda v: (ox + s * (v @ sx_), oy - s * (v @ sy_))
    for k, lab in enumerate(("e_1", "e_2", "e_3")):
        x2, y2 = P2(np.eye(3)[k] * 0.9)
        els.append(line(ox, oy, x2, y2, AXIS, 1.2, dash="4 3"))
        els.append(text(x2 + (6 if x2 >= ox else -6), y2 + 5, lab, FS_SMALL, "start" if x2 >= ox else "end", MUTED))
    vmax = np.abs(Psi[:, i]).max()
    for x in range(N16):
        px, py = P2(Hm[x])
        els.append(circle(px, py, 7, fill=div_color(Psi[x, i], vmax), stroke="#ffffff", width=2))
    for k, col, lab in ((0, MUTED, "a_0 (constant mode)"), (1, INK, "a_1")):
        x2, y2 = P2(f * A[:, k])
        els.append(arrow(ox, oy, x2, y2, col, 2.6, 14))
        els.append(text(x2 + (10 if x2 >= ox else -10), y2 - 8, lab, FS, "start" if x2 >= ox else "end", col, weight=700 if k else None))
    els.append(circle(ox, oy, 3, fill=INK2))
    frac = "½" if f == 0.5 else "¼"
    els.append(text(720, 160, f"arrows a_0, a_1 drawn at {frac} scale; loadings are", FS_SMALL, "start", INK2))
    els.append(text(720, 180, "coefficients, not orthogonal directions", FS_SMALL, "start", INK2))
    # bottom: exact expansion as outer products
    y0 = 556
    els.append(text(40, y0 - 40, "exact expansion over a complete orthonormal basis (Ψ^TΨ = I, constant mode included)", FS, "start", INK, weight=700))
    rh, cw = 11, 16
    Hv = np.abs(Hm).max()
    def mat(x, y, M, vm):
        out = []
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                out.append(rect(x + c * cw, y + r * rh, cw - 1, rh - 1, fill=div_color(M[r, c], vm)))
        return "".join(out)
    x = 60
    els.append(mat(x, y0 + 24, Hm, Hv))
    els.append(text(x + 1.5 * cw, y0 + 24 + 16 * rh + 26, "H^L", FS, "middle", INK, weight=700))
    x += 3 * cw + 22
    els.append(text(x, y0 + 24 + 8 * rh + 6, "=", 24, "start", INK))
    x += 30
    pv, av = np.abs(Psi).max(), np.abs(A).max()
    for k in range(4):
        els.append(mat(x, y0 + 24, Psi[:, [k]], pv))
        els.append(mat(x + cw + 6, y0, A[:, [k]].T, av))
        els.append(mat(x + cw + 6, y0 + 24, np.outer(Psi[:, k], A[:, k]), Hv))
        els.append(text(x + cw + 6 + 1.5 * cw, y0 + 24 + 16 * rh + 26, f"ψ_{k} a_{k}^T", FS, "middle", INK))
        if k == 0:
            els.append(text(x + cw / 2, y0 + 12, "ψ_0", FS_SMALL, "middle", INK2))
            els.append(text(x + cw + 6 + 3 * cw + 6, y0 + 10, "a_0^T", FS_SMALL, "start", INK2))
        x += cw + 6 + 3 * cw + 16
        els.append(text(x, y0 + 24 + 8 * rh + 6, "+", 24, "start", INK))
        x += 30
    tail = Psi[:, 4:] @ A[:, 4:].T
    els.append(mat(x, y0 + 24, tail, Hv))
    els.append(text(x + 1.5 * cw, y0 + 24 + 16 * rh + 26, "Σ_{i≥4} ψ_i a_i^T", FS, "middle", INK))
    # tail numbers
    tx = x + 3 * cw + 50
    els.append(text(tx, y0 + 30, "‖H^L − Σ_{i≤r} ψ_i a_i^T‖_F^2 = Σ_{i>r} ‖a_i‖^2", FS, "start", INK))
    tot = np.sum(Hm ** 2)
    for k, r in enumerate((0, 2, 4)):
        tl = np.sum(A[:, r + 1:] ** 2)
        els.append(text(tx, y0 + 64 + 26 * k, f"r = {r}:  tail = {tl:.2f}  ({100 * tl / tot:.0f}% of ‖H^L‖_F^2)", FS_SMALL, "start", INK2))
    els.append(text(tx, y0 + 156, "each term: column ψ_i, row a_i^T above,", FS_SMALL, "start", INK2))
    els.append(text(tx, y0 + 176, "their product below", FS_SMALL, "start", INK2))
    els.append(diverging_legend(tx + 30, y0 + 206, 140, "−", "+", None))
    figure("D3_loadings.svg", W, Hh, "D3", "Loadings: the two faces of a mode", "THEORY §1.3, §5",
           ["A mode ψ_i is a function on sites (left) and, through its loading a_i = (H^L)^T ψ_i, a direction in activation space (right) [thm].",
            "Over a complete orthonormal basis, constant mode included, H^L = Σ_i ψ_i a_i^T exactly; dropping modes costs Σ_{i>r} ‖a_i‖^2."],
           "".join(els))

# ================================================================ D4 network toy (shared with D7)
def toy_net():
    """Constructed MLP block on the 16 sites of D2-D3: h^{L+1} = h^L + W_2 ReLU(W_1 h^L + c).
    W_1 rows point at the display clusters (planted), W_2 and a 3-class readout W_U are
    random (seed 279), the loss is mean cross-entropy at these weights. Delta = dL/dz."""
    U, Psi, N = H16, PSI16, N16
    cent = np.array([U[GROUPS16 == g].mean(0) for g in range(5)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    W1 = np.array([cent[0], cent[1], cent[2], cent[3], cent[4], cent[3] - cent[4]])
    c = np.array([-0.55, -0.55, -0.55, -0.5, -0.5, -0.2])
    rng = np.random.default_rng(279)
    W2 = rng.normal(size=(3, 6)) * 0.5
    WU = rng.normal(size=(3, 3))
    y = np.array([0] * 9 + [1] * 4 + [2] * 3)
    Z = U @ W1.T + c
    Nn = np.maximum(Z, 0)
    lg = (U + Nn @ W2.T) @ WU.T
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    Delta = (((p - np.eye(3)[y]) / N) @ WU @ W2) * (Z > 0)
    G = Delta.T @ U                                  # gradient of W_1 (bias handled separately)
    g, b = Delta.T @ Psi, U.T @ Psi                  # columns: g_i, b_i
    assert np.allclose(sum(np.outer(g[:, i], b[:, i]) for i in range(N)), G)
    pre, post = Z.T @ Psi, Nn.T @ Psi                # mode profiles of pre- and post-ReLU units
    assert np.allclose(pre[:, 1:], W1 @ b[:, 1:])    # <z_k, psi_i> = (W_1 b_i)_k for psi_i orthogonal to 1
    return dict(U=U, W1=W1, Z=Z, Delta=Delta, G=G, g=g, b=b, pre=pre, post=post)

NET = toy_net()

def d4():
    W, Hh = 1200, 1010
    els = []
    g, b, pre, post = NET["g"], NET["b"], NET["pre"], NET["post"]
    modes = [1, 2]
    bmax = np.abs(b[:, modes]).max()
    gmax = np.abs(g[:, modes]).max()
    pmax = np.abs(pre[:, modes]).max()
    E = {i: np.abs(np.outer(g[:, i], b[:, i])) for i in modes}
    emax = max(E[i].max() for i in modes)
    uy = [228, 318, 408]
    hy = [168 + 60 * k for k in range(6)]
    for pi, i in enumerate(modes):
        ox = 40 + pi * 590
        X_L, X_Z, X_N, X_O = ox + 60, ox + 246, ox + 318, ox + 500
        blk = "B_1" if i == 1 else "B_2"
        els.append(text(ox, 96, f"mode ψ_{i}", 22, "start", INK, weight=700))
        els.append(text(ox + 120, 96, f"λ_{i} = {LAM16[i]:.2f}, one-mode block {blk}", FS, "start", INK2))
        for xh, lab in ((X_L, "h^L"), (X_Z, "z_k"), (X_N, "neuron"), (X_O, "h^{L+1}")):
            els.append(text(xh, 140, lab, FS_SMALL, "middle", INK2))
        els.append(text((X_L + X_Z) / 2, 140, "W_1", FS_SMALL, "middle", INK, weight=700))
        els.append(text((X_L + X_Z) / 2, 160, "shade = |g_{i,k} b_{i,j}|", FS_SMALL, "middle", INK2))
        els.append(text((X_N + X_O) / 2 + 10, 140, "W_2", FS_SMALL, "middle", INK2))
        # residual path under the block
        els.append(path(f"M{X_L},{uy[-1] + 24} C{X_L},{530} {X_O},{530} {X_O},{uy[-1] + 24}", AXIS, 2))
        els.append(text((X_L + X_O) / 2, 540, "+ h^L (residual)", FS_SMALL, "middle", MUTED))
        # W2 edges, unlit
        for k in range(6):
            for o in range(3):
                els.append(line(X_N + 20, hy[k], X_O - 22, uy[o], GRID, 1.1))
        # W1 edges: base hairline, then shade by |g_k b_j| in increasing order
        order = sorted(((E[i][k, j], k, j) for k in range(6) for j in range(3)))
        for _, k, j in order:
            els.append(line(X_L + 22, uy[j], X_Z - 10, hy[k], GRID, 1.1))
        for e, k, j in order:
            t = e / emax
            if t > 0.04:
                els.append(line(X_L + 22, uy[j], X_Z - 10, hy[k], seq_color(0.2 + 0.8 * t), 0.8 + 6.5 * t))
        # units
        for j in range(3):
            els.append(circle(X_L, uy[j], 22, fill=div_color(b[j, i], bmax), stroke=INK2, width=1.3))
            els.append(text(X_L - 32, uy[j] + 6, f"j={j + 1}", FS_SMALL, "end", MUTED))
            els.append(circle(X_O, uy[j], 22, fill=SURFACE, stroke=AXIS, width=1.3))
        for k in range(6):
            els.append(line(X_Z + 9, hy[k], X_N - 20, hy[k], INK2, 1.2))
            els.append(rect(X_Z - 10, hy[k] - 10, 20, 20, fill=div_color(g[k, i], gmax), stroke=INK2, width=1.2, rx=3))
            els.append(half_disc(X_N, hy[k], 20, div_color(pre[k, i], pmax), div_color(post[k, i], pmax)))
    # vertical separator
    els.append(line(612, 80, 612, 550, GRID, 1))
    # key
    ky = 600
    els.append(text(40, ky, "How to read a panel. Constructed toy: U = H^L of D2–D3, planted W_1, random W_2 and readout.", FS, "start", INK, weight=700))
    rows = [
        ("unit", "unit j at layer L: input loading b_{i,j}   (b_i = U^T ψ_i, U = H^L)"),
        ("square", "pre-activation z_k: error loading g_{i,k}   (g_i = Δ^T ψ_i, Δ = ∂ℒ/∂z)"),
        ("half", "neuron k: left ⟨z_k, ψ_i⟩ = (W_1 b_i)_k, right ⟨ReLU(z_k), ψ_i⟩;"),
        (None, "ReLU moves part of the profile into other modes"),
        ("edge", "W_1 edge (k, j): |g_{i,k} b_{i,j}|, a basis-dependent"),
        (None, "gradient contribution: not a weight magnitude,"),
        (None, "and not an identified mechanism"),
    ]
    for r, (glyph, s_) in enumerate(rows):
        y = ky + 40 + 30 * r
        if glyph == "unit":
            els.append(circle(62, y - 6, 11, fill=div_color(0.7, 1), stroke=INK2, width=1.2))
        elif glyph == "square":
            els.append(rect(52, y - 16, 20, 20, fill=div_color(-0.7, 1), stroke=INK2, width=1.2, rx=3))
        elif glyph == "half":
            els.append(half_disc(62, y - 6, 12, div_color(0.9, 1), div_color(0.4, 1)))
        elif glyph == "edge":
            els.append(line(46, y - 6, 80, y - 6, seq_color(0.8), 5))
        strong = r >= 4
        els.append(text(96, y, s_, FS, "start", INK if (glyph or strong) else INK2, weight=700 if strong else None))
    els.append(diverging_legend(124, ky + 268, 150, "−", "+", "signed; each quantity on its own scale"))
    els.append(seq_legend(420, ky + 268, 150, "edge |g_{i,k} b_{i,j}|, shared scale", "0", "max"))
    # interaction callout
    g1, g2, b1, b2 = g[:, 1], g[:, 2], b[:, 1], b[:, 2]
    gg, bb = g1 @ g2, b1 @ b2
    cg = gg / (np.linalg.norm(g1) * np.linalg.norm(g2))
    cb = bb / (np.linalg.norm(b1) * np.linalg.norm(b2))
    Iv = gg * bb
    assert np.isclose(Iv, np.sum(np.outer(g1, b1) * np.outer(g2, b2)))
    mant, ex = f"{Iv:.1e}".split("e")
    bx, by = 770, ky + 20
    els.append(rect(bx, by, 400, 282, fill="#ffffff", stroke=GRID, width=1.2, rx=10))
    els.append(text(bx + 18, by + 32, "signed interaction of the two pieces", FS, "start", INK, weight=700))
    els.append(text(bx + 18, by + 64, "⟨g_1 b_1^T, g_2 b_2^T⟩_F = ⟨g_1, g_2⟩ ⟨b_1, b_2⟩", FS, "start", INK))
    els.append(text(bx + 18, by + 96, "as cosines: " + num(cg) + " × " + num(cb) + " = " + num(cg * cb), FS, "start", INK))
    els.append(text(bx + 18, by + 126, "value " + num(float(mant), 1) + " × 10^{" + str(int(ex)).replace("-", "−") + "}: " + ("interference" if Iv < 0 else "cooperative"), FS, "start", INK))
    els.append(text(bx + 18, by + 162, "the error loadings oppose, but the input", FS_SMALL, "start", INK2))
    els.append(text(bx + 18, by + 182, "loadings barely overlap: the product is small", FS_SMALL, "start", INK2))
    els.append(text(bx + 18, by + 218, "B_1, B_2 are one-mode blocks, so this is the", FS_SMALL, "start", INK2))
    els.append(text(bx + 18, by + 240, "invariant I_{B₁B₂} = ⟨G_{B₁}, G_{B₂}⟩_F", FS_SMALL, "start", INK2))
    figure("D4_mode_lights_network.svg", W, Hh, "D4", "A mode lights up weights and neurons", "THEORY §7.1–7.2",
           ["Since ∇_{W₁}ℒ = Σ_i g_i b_i^T [thm, A3], a mode shades layer-L units by b_i, pre-activations by g_i, W_1 edges by |g_{i,k} b_{i,j}|.",
            "Edges: a basis-dependent gradient contribution, not a weight or a mechanism; only block sums and interactions are invariant."],
           "".join(els))

# ================================================================ D5 cross-layer coherence
def _blob(theta_deg, n, phi_deg=55, jitter=4.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        th = math.radians(theta_deg + jitter * rng.normal())
        ph = math.radians(phi_deg + jitter * rng.normal())
        out.append([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)])
    return np.array(out)

def d5():
    W, Hh = 1200, 772
    beta, nA = 5.0, 6
    sep = np.vstack([_blob(150, nA, seed=1), _blob(30, nA, seed=2)])
    sep2 = np.vstack([_blob(205, nA, seed=3), _blob(-25, nA, seed=4)])
    one = np.vstack([_blob(90, nA, phi_deg=22, jitter=7, seed=5), _blob(90, nA, phi_deg=22, jitter=7, seed=6)])
    scen = [("coherent", sep, sep2), ("merge", sep, one), ("split", one, sep)]   # split = merge with layers swapped
    c = np.r_[np.ones(nA), -np.ones(nA)] / math.sqrt(2 * nA)
    N = 2 * nA
    C = np.eye(N) - np.ones((N, N)) / N
    els = []
    els.append(text(W / 2, 96, "Q^{L→L′} = P^{L′} P^L = Σ_i σ_i u_i v_i^T;  v_i is a pattern on the sites at L, u_i at L′", FS, "middle", INK, weight=700))
    els.append(text(W / 2, 124, "same 12 sites at both layers [A1]; contrast c = (1_A − 1_B) / √12, ‖c‖ = 1; slow: ‖P c‖ ≈ ‖c‖ (survives diffusion)", FS, "middle", INK2))
    sig = {}
    for col, (name, XL, XLp) in enumerate(scen):
        _, PL = sphere_P(XL, beta)
        _, PLp = sphere_P(XLp, beta)
        Q = PLp @ PL
        s1 = np.linalg.svd(C @ Q @ C, compute_uv=False)[0]
        sig[name] = s1
        nL, nLp, nQ = (np.linalg.norm(M @ c) for M in (PL, PLp, Q))
        nQt = np.linalg.norm(PL @ PLp @ c)
        cx = 40 + col * 380 + 180
        els.append(text(cx, 176, name, 22, "middle", INK, weight=700))
        for side, (X, lab) in enumerate(((XL, "layer L"), (XLp, "layer L′"))):
            sx = cx - 92 + side * 184
            sy, R = 290, 76
            els.append(sphere(sx, sy, R))
            els.append(text(sx, sy + R + 28, lab, FS_SMALL, "middle", INK2))
            for k in [v for pair in zip(range(nA), range(nA, N)) for v in pair]:
                px, py = sx + R * X[k, 0], sy - R * X[k, 1]
                els.append(circle(px, py, 5.5, fill=CAT[0] if k < nA else CAT[1], stroke="#ffffff", width=1.5))
            # direct labels at each blob's centroid (secondary encoding for identity)
            for grp, lab2, colr in ((slice(0, nA), "A", CAT[0]), (slice(nA, N), "B", CAT[1])):
                mx, my = X[grp, 0].mean(), X[grp, 1].mean()
                together = np.linalg.norm(X[:nA].mean(0) - X[nA:].mean(0)) < 0.2
                if together:
                    if lab2 == "A":
                        els.append(text(sx + R * mx, sy - R * my - 16, "A, B", FS_SMALL, "middle", INK))
                else:
                    els.append(text(sx + R * mx, sy - R * my - 14, lab2, FS_SMALL, "middle", INK))
        els.append(arrow(cx - 10, 290, cx + 10, 290, INK2, 1.6, 9))
        y0 = 420
        if name == "coherent":
            lines_ = [("‖P^L c‖ = " + num(nL, 2, False) + ", ‖P^{L′}P^L c‖ = " + num(nQ, 2, False), INK),
                      ("c is slow at both layers:", INK2), ("a set kept together", INK2)]
        elif name == "merge":
            lines_ = [("‖P^L c‖ = " + num(nL, 2, False) + " → ‖P^{L′}P^L c‖ = " + num(nQ, 2, False), INK),
                      ("slow at L, nearly annihilated at L′:", INK2), ("a merge (read off Q)", INK2)]
        else:
            lines_ = [("‖P^L c‖ = " + num(nL, 2, False) + ", ‖P^{L′} c‖ = " + num(nLp, 2, False), INK),
                      ("not slow at L, slow at L′:", INK2), ("a split (read off Q^T)", INK2)]
        for k, (s_, colr) in enumerate(lines_):
            els.append(text(cx, y0 + 24 * k + (4 if k else 0), s_, FS_SMALL if k else FS, "middle", colr))
        els.append(text(cx, y0 + 92, "top centred σ_1 = " + num(s1, 2, False), FS, "middle", INK))
    assert abs(sig["merge"] - sig["split"]) < 1e-9          # Q and Q^T share singular values
    els.append(line(40, 552, W - 40, 552, GRID, 1))
    notes = ["merge and split have the same σ_1 (Q and Q^T share singular values): σ alone gives no direction;",
             "the direction comes from comparing ‖P^L c‖ with ‖P^{L′} c‖. When L = L′, σ_i = λ_i^2, so a small σ is never by",
             "itself evidence of a new distinction. Reading a merge or split off a singular pair is [hyp], checked on extracted sets."]
    for k, s_ in enumerate(notes):
        els.append(text(40, 584 + 26 * k, s_, FS, "start", INK2))
    figure("D5_cross_layer_coherence.svg", W, Hh, "D5", "Cross-layer coherence: coherent, merge, split", "THEORY §4.1",
           ["The same sites at two layers [A1] give Q^{L→L′} = P^{L′}P^L. A contrast slow at L but nearly annihilated at L′: a merge;",
            "one slow only at L′ is a split, read off Q^T = P^L P^{L′}. Constructed toy, β = 5; the groups A, B are for display only."],
           "".join(els))

# ================================================================ D6 carried versus read
def d6():
    W, Hh = 1200, 860
    els = []
    # left: positions x layers
    cols = list(range(-4, 4))                 # offsets from t
    xs = {o: 90 + (o + 4) * 62 for o in cols}
    yL, yL1 = 380, 260
    els.append(text(40, yL1 + 6, "L+1", FS, "start", INK2))
    els.append(text(40, yL + 6, "L", FS, "start", INK2))
    for o in cols:
        x = xs[o]
        els.append(line(x, yL + 90, x, yL1 - 30, GRID, 1.2))
        for y in (yL, yL1):
            hi = o == 0
            els.append(circle(x, y, 14, fill="#ffffff" if hi else SURFACE, stroke=INK if hi else AXIS, width=2.4 if hi else 1.4))
        lab = "t" if o == 0 else (f"t{o:+d}".replace("-", "−"))
        els.append(text(x, yL + 112, lab, FS_SMALL, "middle", INK if o == 0 else MUTED, weight=700 if o == 0 else None))
    els.append(text(xs[0], yL + 140, "site x = (doc, t)", FS, "middle", INK, weight=700))
    # past window
    els.append(rect(xs[-4] - 24, yL - 26, xs[-1] - xs[-4] + 48, 52, stroke=INK2, width=1.6, rx=12, dash="5 4"))
    els.append(text((xs[-4] + xs[-1]) / 2, yL + 52, "past window z_{past}(x)", FS_SMALL, "middle", INK2))
    # carried target: future of the same stream
    els.append(rect(xs[1] - 24, yL - 26, xs[3] - xs[1] + 48, 52, stroke=CAT[0], width=2.6, rx=12))
    els.append(text((xs[1] + xs[3]) / 2, yL + 52, "carried target", FS_SMALL, "middle", INK, weight=700))
    els.append(text((xs[1] + xs[3]) / 2, yL + 72, "ĥ^L at positions > t", FS_SMALL, "middle", INK2))
    # read target: the update at the next layer at x
    els.append(circle(xs[0], yL1, 24, stroke=CAT[1], width=2.6))
    els.append(arrow(xs[0], yL - 16, xs[0], yL1 + 26, INK2, 1.6, 9))
    els.append(text(xs[0] - 4, yL1 - 64, "read target", FS_SMALL, "middle", INK, weight=700))
    els.append(text(xs[0] - 4, yL1 - 44, "Δh^{L+1}(x) = h^{L+1}(x) − h^L(x)", FS_SMALL, "middle", INK2))
    els.append(text(40, 100, "two targets for the same past [P5]", FS, "start", INK, weight=700))
    els.append(text(40, 126, "neither implies the other [thm]", FS_SMALL, "start", INK2))
    # right top: whitened cross-covariance -> SVD
    rx0 = 620
    els.append(text(rx0, 112, "regularized canonical-correlation SVD (a fit) [P6]", FS, "start", INK, weight=700))
    py = [180, 222, 264, 306]
    lx, mx = rx0 + 30, rx0 + 170
    rng = np.random.default_rng(4)
    Wt = rng.uniform(0.15, 1, size=(4, 4))
    for a in range(4):
        for bq in range(4):
            els.append(line(lx, py[a], mx, py[bq], seq_color(0.25 + 0.6 * Wt[a, bq]), 0.8 + 2.4 * Wt[a, bq]))
    for y in py:
        els.append(circle(lx, y, 9, fill="#ffffff", stroke=INK2, width=1.4))
        els.append(circle(mx, y, 9, fill="#ffffff", stroke=INK2, width=1.4))
    els.append(text(lx, 340, "past", FS_SMALL, "middle", INK2))
    els.append(text(mx, 340, "target", FS_SMALL, "middle", INK2))
    els.append(text((lx + mx) / 2, 368, "R_q^{1/2} R_p^{1/2}", FS, "middle", INK))
    els.append(text((lx + mx) / 2, 392, "whitened cross-covariance", FS_SMALL, "middle", INK2))
    els.append(arrow(mx + 30, 243, mx + 110, 243, INK, 2, 12))
    els.append(text(mx + 70, 230, "SVD", FS, "middle", INK, weight=700))
    l2, m2 = mx + 140, mx + 250
    kap = [0.86, 0.55, 0.28, 0.08]
    for a, k_ in enumerate(kap):
        els.append(line(l2, py[a], m2, py[a], seq_color(0.2 + 0.8 * k_), 0.8 + 6 * k_))
        els.append(circle(l2, py[a], 9, fill="#ffffff", stroke=INK2, width=1.4))
        els.append(circle(m2, py[a], 9, fill="#ffffff", stroke=INK2, width=1.4))
        els.append(text(l2 - 16, py[a] + 6, f"s_{a + 1}", FS_SMALL, "end", INK))
        els.append(text((l2 + m2) / 2, py[a] - 12, f"κ_{a + 1}", FS_SMALL, "middle", INK))
    els.append(text((l2 + m2) / 2, 340, "κ_i = σ_i(R_q^{1/2} R_p^{1/2}), 0 ≤ κ_i ≤ 1", FS_SMALL, "middle", INK2))
    els.append(text((l2 + m2) / 2, 362, "past variates: R_p R_q s_i = κ_i^2 s_i,", FS_SMALL, "middle", INK2))
    els.append(text((l2 + m2) / 2, 384, "normalized to unit variance", FS_SMALL, "middle", INK2))
    # bottom right: risk staircase
    sx0, sy0, sw, sh = 700, 724, 420, 220
    els.append(text(640, 448, "risk ladder: idealized Bayes risks (schematic)", FS, "start", INK, weight=700))
    els.append(text(640, 474, "risk ℛ of predicting the read target Δh^{L+1}(x)", FS_SMALL, "start", INK2))
    els.append(line(sx0, sy0, sx0 + sw, sy0, AXIS, 1.2))
    els.append(line(sx0, sy0, sx0, sy0 - sh, AXIS, 1.2))
    els.append(text(sx0 - 10, sy0 - sh + 16, "ℛ", FS_SMALL, "end", INK2))
    els.append(text(sx0 - 10, sy0 + 5, "0", FS_SMALL, "end", INK2))
    vals = [1.0, 0.62, 0.45, 0.37, 0.33, 0.315]
    labs = ["ℛ_0", "ℛ(h^L)", "+s_1", "+s_2", "+s_3", "…"]
    step = sw / len(vals)
    d = []
    for k, v in enumerate(vals):
        x1, x2, y = sx0 + k * step, sx0 + (k + 1) * step, sy0 - sh * 0.9 * v
        d.append(f"{'M' if k == 0 else 'L'}{x1:.1f},{y:.1f} L{x2:.1f},{y:.1f}")
        els.append(text((x1 + x2) / 2, sy0 + 24, labs[k], FS_SMALL, "middle", INK))
    els.append(path(" ".join(d), INK, 2.4))
    floor = sy0 - sh * 0.9 * vals[-1]
    els.append(line(sx0, floor, sx0 + sw, floor, MUTED, 1, dash="3 4"))
    els.append(text(sx0 + sw - 4, floor + 20, "illustrative floor (≥ 0)", FS_SMALL, "end", MUTED))
    els.append(text(sx0 + 1.5 * step + 12, sy0 - sh * 0.9 * 0.82, "from h^L(x) alone", FS_SMALL, "start", INK2))
    els.append(text(sx0 + 2.5 * step + 12, sy0 - sh * 0.9 * 0.55, "added by the context", FS_SMALL, "start", INK2))
    # cautions on the left, under the sequence
    cy0 = 600
    els.append(text(40, cy0, "plotted: idealized Bayes risks; measured: held-out risks", FS, "start", INK, weight=700))
    caut = ["Bayes risks are nested, so they never increase [thm];",
            "fitted held-out risks need not decrease or reach zero,",
            "and are reported as a sequence, not an inequality.",
            "They measure predictive information at this window",
            "and scale, not the layer's architectural use of context."]
    for k, s_ in enumerate(caut):
        els.append(text(40, cy0 + 32 + 26 * k, s_, FS, "start", INK2))
    figure("D6_carried_vs_read.svg", W, Hh, "D6", "Carried versus read", "THEORY §6.3–6.4",
           ["Past window at x; two fitted targets: carried (the same stream's future) and read (the next layer's update at x) [P5, P6].",
            "The ladder is an idealized Bayes-risk schematic: fitted held-out risks need not fall or reach zero; not architectural use."],
           "".join(els))

# ================================================================ D7 gradient as rank-one pieces
def d7():
    W, Hh = 1200, 900
    els = []
    g, b, G = NET["g"], NET["b"], NET["G"]
    U, Delta = NET["U"], NET["Delta"]
    # rotated basis inside block B_3 = {psi_3, psi_4}
    th = math.radians(40)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    PsiR = PSI16[:, [3, 4]] @ R
    gR, bR = Delta.T @ PsiR, U.T @ PsiR
    pieces = [np.outer(g[:, i], b[:, i]) for i in range(N16)]
    GB3 = pieces[3] + pieces[4]
    assert np.allclose(np.outer(gR[:, 0], bR[:, 0]) + np.outer(gR[:, 1], bR[:, 1]), GB3)
    Pi3 = PSI16[:, [3, 4]] @ PSI16[:, [3, 4]].T
    assert np.allclose(Delta.T @ Pi3 @ U, GB3)
    tail = sum(pieces[5:])
    assert np.allclose(pieces[0] + pieces[1] + pieces[2] + GB3 + tail, G)
    vG = max(np.abs(p).max() for p in pieces + [G, tail, np.outer(gR[:, 0], bR[:, 0]), np.outer(gR[:, 1], bR[:, 1])])
    vg = max(np.abs(g[:, :5]).max(), np.abs(gR).max())
    vb = max(np.abs(b[:, :5]).max(), np.abs(bR).max())
    cw, rh = 22, 22
    def mat(x, y, M, vm):
        return "".join(rect(x + c * cw, y + r * rh, cw - 2, rh - 2, fill=div_color(M[r, c], vm), rx=2)
                       for r in range(M.shape[0]) for c in range(M.shape[1]))
    def term(x, y, gv, bv, label):
        out = mat(x, y + 30, gv[:, None], vg) + mat(x + cw + 8, y, bv[None, :], vb)
        out += mat(x + cw + 8, y + 30, np.outer(gv, bv), vG)
        out += text(x + cw + 8 + 1.5 * cw, y + 30 + 6 * rh + 28, label, FS, "middle", INK)
        return out, x + cw + 8 + 3 * cw
    y0 = 168
    x = 50
    els.append(text(50, 100, "the W_1 gradient of D4 (6 × 3): each piece is a column g_i = Δ^T ψ_i times a row b_i^T = (U^T ψ_i)^T", FS, "start", INK))
    els.append(mat(x, y0 + 30, G, vG))
    els.append(text(x + 1.5 * cw, y0 + 30 + 6 * rh + 28, "G", FS, "middle", INK, weight=700))
    els.append(text(x + 1.5 * cw, y0 + 30 + 6 * rh + 52, "∇_{W₁}ℒ", FS_SMALL, "middle", INK2))
    x += 3 * cw + 18
    plus_y = y0 + 30 + 3 * rh + 8
    els.append(text(x, plus_y, "=", 26, "start", INK)); x += 34
    spans = {}
    for name, idx in (("G_0", [0]), ("B_1", [1]), ("B_2", [2]), ("B_3", [3, 4])):
        x_start = x
        for kk, i in enumerate(idx):
            e, x = term(x, y0, g[:, i], b[:, i], f"g_{i} b_{i}^T")
            els.append(e)
            if kk < len(idx) - 1:
                els.append(text(x + 8, plus_y, "+", 24, "start", INK)); x += 32
        spans[name] = (x_start, x)
        els.append(text(x + 10, plus_y, "+", 26, "start", INK)); x += 40
    els.append(mat(x, y0 + 30, tail, vG))
    els.append(text(x + 1.5 * cw, y0 + 30 + 6 * rh + 28, "Σ_{i≥5} g_i b_i^T", FS, "middle", INK))
    spans["rest"] = (x, x + 3 * cw)
    # brackets
    labels = {"G_0": "constant mode", "B_1": "block B_1", "B_2": "block B_2",
              "B_3": "block B_3 (λ_3 ≈ λ_4)", "rest": "complement"}
    for k, (x1, x2) in spans.items():
        els.append(brace_h(x1, x2, y0 - 12, True, INK2, 8))
        anchor = "middle" if k != "B_3" else "start"
        xl = (x1 + x2) / 2 if k != "B_3" else x1
        els.append(text(xl, y0 - 26, labels[k], FS_SMALL, anchor, INK2))
    # rotated basis inside B_3
    x1, x2 = spans["B_3"]
    yr = y0 + 290
    els.append(text(x1, yr - 26, "B_3 in a basis rotated by 40°: new pieces, same sum", FS_SMALL, "start", INK, weight=700))
    xr = x1
    e, xr = term(xr, yr, gR[:, 0], bR[:, 0], "g′_3 b′_3^T"); els.append(e)
    els.append(text(xr + 8, yr + 30 + 3 * rh + 8, "+", 24, "start", INK)); xr += 32
    e, xr = term(xr, yr, gR[:, 1], bR[:, 1], "g′_4 b′_4^T"); els.append(e)
    els.append(text(xr + 10, yr + 30 + 3 * rh + 8, "=", 24, "start", INK)); xr += 40
    els.append(mat(xr, yr + 30, GB3, vG))
    els.append(text(xr + 1.5 * cw, yr + 30 + 6 * rh + 28, "G_{B₃}", FS, "middle", INK))
    els.append(text(xr + 1.5 * cw, yr + 30 + 6 * rh + 50, "= g_3 b_3^T + g_4 b_4^T", FS_SMALL, "middle", INK2))
    # notes on the left of the lower row
    ny = yr + 10
    pressure = {k: np.linalg.norm(v) for k, v in (("G_0", pieces[0]), ("B_1", pieces[1]), ("B_2", pieces[2]), ("B_3", GB3))}
    notes = [("invariant (reported):", INK, 700),
             ("block pressure ‖G_B‖_F, G_B = Δ^T Π_B U", INK2, None),
             ("(Π_B: projector onto the block's modes)", INK2, None),
             ("B_1: " + num(pressure["B_1"], 3, False) + ",  B_2: " + num(pressure["B_2"], 3, False) + ",  B_3: " + num(pressure["B_3"], 3, False), INK2, None),
             ("signed interaction ⟨G_B, G_{B′}⟩_F", INK2, None),
             ("not invariant inside a block:", INK, 700),
             ("individual g_i b_i^T and ‖g_i‖‖b_i‖", INK2, None)]
    for k, (s_, colr, wt) in enumerate(notes):
        els.append(text(50, ny + 28 * k, s_, FS, "start", colr, weight=wt))
    els.append(text(50, ny + 222, "pieces are not Frobenius-orthogonal: ‖G − G_r‖_F is computed, not summed", FS_SMALL, "start", INK2))
    els.append(text(50, ny + 244, "this is the gradient, not the optimizer's update", FS_SMALL, "start", INK2))
    els.append(diverging_legend(80, ny + 276, 150, "−", "+", None))
    els.append(text(270, ny + 288, "g column and b row each on its own scale; products share G's scale", FS_SMALL, "start", INK2))
    figure("D7_gradient_rank_one.svg", W, Hh, "D7", "The gradient as a sum of rank-one pieces", "THEORY §7.2",
           ["For W used once per site, G = Δ^T U = Σ_i g_i b_i^T exactly, constant mode and complement included [thm, A3].",
            "Inside a near-degenerate block the pieces depend on the basis; the block sum G_B = Δ^T Π_B U, its norm and interactions do not."],
           "".join(els))

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    for name, fn in sorted((k, v) for k, v in globals().items() if k[:1] == "d" and k[1:].isdigit()):
        if not only or name in only:
            fn()
