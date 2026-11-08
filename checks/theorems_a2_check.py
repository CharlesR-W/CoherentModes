"""Assert the finite-site stability bounds and counterexamples in draft A2.

Run from the repository root:
    uv run --with numpy python -I checks/theorems_a2_check.py

Only NumPy is required. The examples are deterministic numerical checks, not
proofs of uniform bounds or searches certifying worst-case upper bounds.
"""

import numpy as np

RNG = np.random.default_rng(20261107)


def fro(a):
    return np.linalg.norm(a, "fro")


def op(a):
    return np.linalg.norm(a, 2)


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def scale(k):
    """Damped symmetric scaling; all test kernels converge within this cap."""
    v = 1 / np.sqrt(k.sum(axis=1))
    for _ in range(10000):
        p = v[:, None] * k * v[None, :]
        if np.max(np.abs(p.sum(axis=1) - 1)) < 2e-14:
            return p, v
        v /= np.sqrt(p.sum(axis=1))
    raise AssertionError("normalization failed")


def derivative(p, h):
    """Derivative for the log-kernel tangent h, equation A.23.4."""
    w = -np.linalg.solve(np.eye(len(p)) + p, (p * h).sum(axis=1))
    return p * (h + w[:, None] + w[None, :]), w


def cf(b, n):
    return min(np.exp(4 * b) / n, np.exp(b) / np.sqrt(2))


def ell(b, n):
    return min(np.exp(2 * b), np.sqrt(n / 2))


def cop_psd(b, n):
    return min(np.exp(3 * b) / n, np.exp(b)) + 2 * np.sqrt(
        min(1, np.exp(4 * b) / n)
    ) * min(np.exp(2 * b) / np.sqrt(n), np.exp(b))


def temperature_integral(lo, hi, n):
    """Exact integral of min(exp(2 beta), sqrt(N/2))."""
    cap = np.sqrt(n / 2)
    switch = np.log(cap) / 2
    if hi <= switch:
        return (np.exp(2 * hi) - np.exp(2 * lo)) / 2
    if lo >= switch:
        return cap * (hi - lo)
    return (cap - np.exp(2 * lo)) / 2 + cap * (hi - switch)


ratios = {}


def bound(name, actual, upper):
    assert actual <= upper * (1 + 1e-9) + 2e-12, (name, actual, upper)
    if upper > 0:
        ratios[name] = max(ratios.get(name, 0), float(actual / upper))


# Finite differences, finite-pair normalization, and view/temperature bounds.
for n in [4, 8, 17]:
    for beta in [0.01, 0.2, 1.0, 2.0]:
        for _ in range(8):
            logk = RNG.uniform(-beta, beta, (n, n))
            logk = (logk + logk.T) / 2
            logkp = RNG.uniform(-beta, beta, (n, n))
            logkp = (logkp + logkp.T) / 2
            k, kp = np.exp(logk), np.exp(logkp)
            p, _ = scale(k)
            pp, _ = scale(kp)
            error, h = pp - p, kp - k
            bound("normF", fro(error), cf(beta, n) * fro(h))
            coeff = min(
                (np.exp(3 * beta) + 2 * np.exp(8 * beta)) / n,
                np.sqrt(n) * cf(beta, n),
            )
            bound("normOp", op(error), coeff * op(h))
            bound("logF", fro(error), ell(beta, n) * np.max(abs(logkp - logk)))
            tangent = RNG.normal(size=(n, n))
            tangent = (tangent + tangent.T) / 2
            dp, w = derivative(p, tangent)
            step = 1e-5
            plus, _ = scale(k * np.exp(step * tangent))
            minus, _ = scale(k * np.exp(-step * tangent))
            assert fro((plus - minus) / (2 * step) - dp) < 2e-8
            assert np.max(abs(dp.sum(axis=1))) < 1e-12
            energy = lambda x: np.sum(p * x * x)
            assert abs(energy(tangent) - energy(dp / p)
                       - energy(w[:, None] + w[None, :])) < 1e-10
            # New cancellation-preserving inequality, including its factor 2.
            assert 2 * fro(dp) ** 2 <= energy(dp / p) + 1e-10
            bound("logDerivative", fro(dp), ell(beta, n) * np.max(abs(tangent)))
            z = unit(RNG.normal(size=(n, 5)))
            zp = unit(z + 0.003 * RNG.normal(size=z.shape))
            k = np.exp(beta * z @ z.T)
            kp = np.exp(beta * zp @ zp.T)
            p, _ = scale(k)
            pp, _ = scale(kp)
            eps = np.linalg.norm(z - zp, axis=1).max()
            bound("view", fro(pp - p), 2 * beta * eps * ell(beta, n))
            bound("psdOp", op(pp - p), cop_psd(beta, n) * op(kp - k))
            pp, _ = scale(np.exp((beta + 0.01) * z @ z.T))
            bound("beta", fro(pp - p), temperature_integral(beta, beta + 0.01, n))

# Davis--Kahan (operator constant 1), Wedin, pressure, and alignment.
for n in [5, 9, 15]:
    for _ in range(30):
        u = np.linalg.qr(RNG.normal(size=(n, n)))[0]
        v = np.linalg.qr(RNG.normal(size=(n, n)))[0]
        vals = np.linspace(0.1, 2, n)
        sl = slice(1, n - 1)
        r, gap = n - 2, vals[1] - vals[0]
        a = (u * vals) @ u.T
        e = RNG.normal(size=(n, n))
        e = (e + e.T) / 2
        e *= 0.12 * gap / op(e)
        _, up = np.linalg.eigh(a + e)
        p, pp = u[:, sl] @ u[:, sl].T, up[:, sl] @ up[:, sl].T
        g = gap - op(e)
        bound("DKop", op(p - pp), op(e) / g)
        bound("DKF", fro(p - pp), np.sqrt(2) * min(np.sqrt(r) * op(e), fro(e)) / g)
        a = (u * vals) @ v.T
        e = RNG.normal(size=(n, n))
        e *= 0.12 * gap / op(e)
        uu, _, vvt = np.linalg.svd(a + e)
        uu, vv = uu[:, ::-1], vvt.T[:, ::-1]
        pu, pv = u[:, sl] @ u[:, sl].T, v[:, sl] @ v[:, sl].T
        pup, pvp = uu[:, sl] @ uu[:, sl].T, vv[:, sl] @ vv[:, sl].T
        g = gap - op(e)
        bound("WedinOp", max(op(pu - pup), op(pv - pvp)), (1 + np.sqrt(2)) * op(e) / g)
        bound("WedinF", np.hypot(fro(pu - pup), fro(pv - pvp)),
              2 * min(np.sqrt(r) * op(e), fro(e)) / g)
        de, inp = RNG.normal(size=(n, 3)), RNG.normal(size=(n, 4))
        dep = de + 0.01 * RNG.normal(size=de.shape)
        inpp = inp + 0.01 * RNG.normal(size=inp.shape)
        pressure = abs(fro(dep.T @ pp @ inpp) - fro(de.T @ p @ inp))
        upper = (fro(dep - de) * op(inpp) + op(de) * op(pp - p) * fro(inpp)
                 + op(de) * fro(inpp - inp))
        bound("pressure", pressure, upper)
        x, y = RNG.normal(size=(n, n)), RNG.normal(size=(n, n))
        xp, yp = x + 0.01 * RNG.normal(size=x.shape), y + 0.01 * RNG.normal(size=y.shape)
        align = lambda a, b: np.sum(a * b) / fro(a) / fro(b)
        bound("alignment", abs(align(x, y) - align(xp, yp)),
              2 * fro(x - xp) / fro(x) + 2 * fro(y - yp) / fro(y))

# The paired-block family saturates the exact b-free sqrt(N/2) coefficient.
sharpness = []
for n in [4, 8, 16, 32]:
    b = 8.0
    k = np.full((n, n), np.exp(-b))
    h = np.zeros_like(k)
    for i in range(0, n, 2):
        k[i:i + 2, i:i + 2] = np.exp(b)
        h[i:i + 2, i:i + 2] = [[1, -1], [-1, 1]]
    p, _ = scale(k)
    dp, w = derivative(p, h)
    exact = np.sqrt(2 * n) / (2 + (n - 2) * np.exp(-2 * b))
    assert abs(fro(dp) - exact) < 1e-12
    assert np.max(abs(w)) < 1e-12
    assert 0.99999 < fro(dp) / np.sqrt(n / 2) <= 1
    bound("sharpLogF", fro(dp), ell(b, n))
    sharpness.append((n, float(fro(dp))))

# Explicit stronger operator stress cases than the review's random search.
# These K are PSD: m*11^T + (M-m)*tt^T, but are not sphere kernels.
# H >= 0 is PSD and is a feasible one-sided perturbation inside the entry box.
operator_stress = []
for n, expected in [(4, 38.26592559273047), (6, 56.22736716622519)]:
    b = 3.0
    k = np.full((n, n), np.exp(-b))
    k[2:, 2:] = np.exp(b)
    h = np.diag([1.0, 1.0] + [0.0] * (n - 2))
    p, _ = scale(k)
    dp, _ = derivative(p, h / k)
    value = n * op(dp) / op(h)
    assert np.linalg.eigvalsh(k).min() > -1e-12
    assert abs(value - expected) < 1e-9
    bound("stressOpFroute", op(dp), np.sqrt(n) * cf(b, n) * op(h))
    bound("stressOpPSD", op(dp), cop_psd(b, n) * op(h))
    step = 1e-7
    pp, _ = scale(k + step * h)
    assert fro((pp - p) / step - dp) < 2e-5
    operator_stress.append((n, value))

# Near-bipartite conditioning and cancellation of a pure endpoint gauge.
for n in [4, 8]:
    for b in [1.0, 3.0, 6.0, 8.0]:
        signs = np.array([1] * (n // 2) + [-1] * (n // 2))
        k = np.exp(-b * np.outer(signs, signs))
        p, _ = scale(k)
        smallest = np.linalg.eigvalsh(np.eye(n) + p)[0]
        assert abs(smallest - 2 / (np.exp(2 * b) + 1)) < 1e-14
        dp, _ = derivative(p, signs[:, None] + signs[None, :])
        assert fro(dp) < 1e-12

# At b ~ 0, the 1/N additive Frobenius coefficient is approached.
n, b = 4, 1e-8
k = np.ones((n, n))
p, _ = scale(k)
c = np.eye(n) - np.ones((n, n)) / n
h = c @ np.diag([1.0, -1.0, 2.0, -2.0]) @ c
slope, _ = derivative(p, h)
assert fro(slope) / (cf(b, n) * fro(h)) > 0.999999

# Gauge invariance and approximate diagonal-gauge cancellation (A.29).
k = np.exp(RNG.normal(size=(7, 7)))
k = (k + k.T) / 2
a = np.exp(RNG.normal(size=7))
p, _ = scale(k)
pp, _ = scale(a[:, None] * k * a[None, :])
assert fro(p - pp) < 1e-12
residual = RNG.uniform(-0.02, 0.02, k.shape)
residual = (residual + residual.T) / 2
kp = a[:, None] * (k * np.exp(residual)) * a[None, :]
pp, _ = scale(kp)
b, eta = np.max(abs(np.log(k))), np.max(abs(residual))
bound("gaugeResidual", fro(pp - p), ell(b + eta, len(k)) * eta)

# The diagonal bias decays with N at fixed beta*tau^2; exact formula.
diagonal_bias = []
for n in [4, 100, 10000]:
    q = np.expm1(1.0)
    exact = np.sqrt(n - 1) * q / (n + q)
    asymptotic = q / np.sqrt(n)
    assert exact < asymptotic
    if n == 10000:
        assert abs(exact / asymptotic - 1) < 0.001
    if n <= 100:
        k = np.full((n, n), np.exp(-1.0))
        np.fill_diagonal(k, 1)
        p, _ = scale(k)
        assert abs(fro(p - np.ones((n, n)) / n) - exact) < 1e-12
    diagonal_bias.append((n, float(exact)))

# Hollow sphere row-normalization bias persists across this site-count range.
angular_bias = []
for n, expected in [(4, 0.009834925508581897), (50, 0.00782), (400, 0.00754)]:
    attenuation = np.linspace(0.6, 0.9, n)
    k = np.exp(np.outer(attenuation, attenuation))
    np.fill_diagonal(k, 0)
    p, _ = scale(k)
    clean = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    error = fro(p - clean)
    assert abs(error - expected) < 6e-6
    assert error > 0.0074
    angular_bias.append((n, float(error)))

# Repeated-type family: the cross ratio survives replication and aggregation.
for copies in [1, 5, 25]:
    attenuation = np.repeat([0.9, 0.8, 0.7, 0.6], copies)
    n = len(attenuation)
    k = np.exp(np.outer(attenuation, attenuation))
    np.fill_diagonal(k, 0)
    p, _ = scale(k)
    clean = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    aggregate = p.reshape(4, copies, 4, copies).sum(axis=(1, 3)) / copies
    logratio = np.log(aggregate[0, 1] * aggregate[2, 3]
                      / aggregate[0, 2] / aggregate[1, 3])
    assert abs(logratio - 0.03) < 1e-12
    d = p - clean
    da = d.reshape(4, copies, 4, copies).sum(axis=(1, 3)) / copies
    assert fro(da) <= fro(d) + 1e-12
    assert fro(d) > 0.008

# Exact nonidentifiability inside positive PSD stochastic matrices (A.28).
e0 = np.ones(3) / np.sqrt(3)
f = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
g = np.cross(e0, f)
theta = 0.7
q = np.column_stack([f, g])
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
for delta in [0.099, 0.01, 0.001]:
    a = np.outer(e0, e0) + q @ np.diag([0.2 + delta, 0.2]) @ q.T
    ap = np.outer(e0, e0) + q @ rot @ np.diag([0.2 + delta, 0.2]) @ rot.T @ q.T
    assert np.all(a > 0) and np.all(ap > 0)
    assert abs(op(ap - a) - delta * abs(np.sin(theta))) < 1e-14

print("max observed actual/bound ratios:", {k: round(v, 5) for k, v in ratios.items()})
print("sharp paired-block log-Frobenius slopes:", sharpness)
print("explicit PSD operator stress cases, N*slope:", operator_stress)
print("kept-diagonal bias:", diagonal_bias)
print("hollow normalized-sphere bias:", angular_bias)
print("all A2 checks passed")
