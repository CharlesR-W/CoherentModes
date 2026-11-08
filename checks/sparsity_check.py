"""Assert finite-site sparsity identities and documented SEBA failure modes.

Run from the repository root:
    uv run --with numpy python -I checks/sparsity_check.py

No learned model or population claim: deterministic algebraic/planted checks only.
The unequal-size SEBA experiment uses the same 200 Haar starts at every threshold.
"""
import itertools
import numpy as np
N,r,n=12,4,3
H=np.kron(np.eye(r),np.ones((n,1))/np.sqrt(n))
alpha=.1
P=(1-alpha)*H@H.T+alpha*np.ones((N,N))/N
vals=np.linalg.eigvalsh(P)[::-1]
h=1.; best=None
for k in range(1,N//2+1):
    for A in itertools.combinations(range(N), k):
        mask=np.zeros(N,dtype=bool); mask[list(A)]=True
        c=P[np.ix_(mask,~mask)].sum()/k
        if c<h: h,best=c,A
lo=(1-vals[1])/2; hi=np.sqrt(2*(1-vals[1]))
assert lo-1e-13<=h<=hi+1e-13
rho_planted=max((P[np.ix_(np.arange(N)//n==j,np.arange(N)//n!=j)].sum()/n) for j in range(r))
assert (1-vals[r-1])/2<=rho_planted+1e-13
print(f'Cheeger: lambda1={vals[1]:.12f}, lower={lo:.12f}, h={h:.12f}, upper={hi:.12f}')
print(f'Planted 4-way conductance={rho_planted:.12f}; lower={(1-vals[r-1])/2:.12f}')

def soft(z,mu): return np.sign(z)*np.maximum(np.abs(z)-mu,0.)
def step(V,R,mu):
    S=soft(V@R.T,mu); S/=np.linalg.norm(S,axis=0)
    U,_,Wt=np.linalg.svd(S.T@V); return S,U@Wt
D=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=float)/2
O=np.diag([1,1,1,-1])@D
V=H@O; mu=.1
# Aligned start, no constant-vector perturbation.
S,R=step(V,O,mu)
exact=np.linalg.norm(S-H)
assert exact<1e-13 and np.linalg.norm(R-O)<1e-13
print(f'Indicator-aligned SEBA error={exact:.3e}; threshold={mu}; minimum indicator height={1/np.sqrt(n):.12f}')
Sbad,Rbad=step(V,np.eye(r),mu)
fixed=np.linalg.norm(Sbad-V); rerr=np.linalg.norm(Rbad-np.eye(r))
nearest=min(np.linalg.norm(Sbad-H[:,perm]*sign) for perm in itertools.permutations(range(r)) for sign in itertools.product([-1,1],repeat=r))
assert fixed<1e-13 and rerr<1e-13 and nearest>1
assert np.all(np.std(V,axis=0)>1e-4) # no constant-column perturbation triggered
print(f'Bad-init fixed-point residual={fixed:.3e}; rotation residual={rerr:.3e}; nearest signed indicator error={nearest:.12f}')
rng=np.random.default_rng(20261107)
Vn,_=np.linalg.qr(H+.002*rng.normal(size=H.shape))
U,_,Wt=np.linalg.svd(Vn.T@H); Oa=U@Wt
eta=np.linalg.norm((np.eye(N)-Vn@Vn.T)@H)
Sn,_=step(Vn,Oa.T,mu)
bound=2*np.sqrt(2)*eta/(1-mu*np.sqrt(n)); actual=np.linalg.norm(Sn-H)
assert actual<=bound+1e-13
print(f'Approximate recovery: eta={eta:.12f}; error={actual:.12f}; bound={bound:.12f}')
Q=P@P; A=np.arange(N)<6; a=A.sum(); mass=Q[np.ix_(A,A)].sum()
e=(A-a/N)/np.sqrt(a*(1-a/N)); score=e@Q@e
assert abs(score-np.linalg.svd(Q,compute_uv=False)[1])<1e-13
print(f'Coherent pair: transport={mass/a:.12f}; centered score={score:.12f}; sigma1={np.linalg.svd(Q,compute_uv=False)[1]:.12f}')
ts=np.linspace(.01,100,10000); gap=.9**ts-.5**ts
idx=np.where(gap>.2)[0]
assert len(idx)>0 and np.all(np.diff(idx)==1)
print(f'Gap interval check: sampled t endpoints={ts[idx[0]]:.4f},{ts[idx[-1]]:.4f}')


# The loop-free conversion in S.3 must keep the two vertex measures separate.
def normalized_partition(sizes):
    H = np.zeros((sum(sizes), len(sizes)))
    offset = 0
    for j, size in enumerate(sizes):
        H[offset:offset+size, j] = 1 / np.sqrt(size)
        offset += size
    return H


def check_loop_conversion(P):
    N = len(P)
    L = np.eye(N) - P
    W = P - np.diag(np.diag(P))
    degree = W.sum(axis=1)
    assert np.min(degree) > 0
    assert np.allclose(np.diag(degree) - W, L, atol=1e-13)
    mu = np.linalg.eigvalsh(L)
    Lnorm = L / np.sqrt(np.outer(degree, degree))
    nu = np.linalg.eigvalsh(Lnorm)
    assert np.all(nu >= mu / degree.max() - 1e-11)
    assert np.all(nu <= mu / degree.min() + 1e-11)
    for mask in range(1, (1 << N) - 1):
        A = np.array([bool(mask & (1 << i)) for i in range(N)])
        cut = P[np.ix_(A, ~A)].sum()
        phi_P = cut / A.sum()
        phi_W = cut / degree[A].sum()
        assert degree.min() * phi_W - 1e-13 <= phi_P
        assert phi_P <= degree.max() * phi_W + 1e-13
    return degree.min(), degree.max()


# A nearly isolated vertex makes degree normalization visibly heterogeneous.
rng_loop = np.random.default_rng(918)
W = rng_loop.uniform(.1, 1., size=(8, 8))
W = (W + W.T) / 2
np.fill_diagonal(W, 0.)
W[0, :] *= .001
W[:, 0] *= .001
L = np.diag(W.sum(axis=1)) - W
P_heterogeneous = np.eye(8) - .8 * L / np.linalg.eigvalsh(L)[-1]
assert np.min(P_heterogeneous) >= 0
assert np.allclose(P_heterogeneous.sum(axis=1), 1.)
assert np.linalg.eigvalsh(P_heterogeneous)[0] >= .2 - 1e-13
lo_degree, hi_degree = check_loop_conversion(P_heterogeneous)
print(f'Loop-free conversion: degree min={lo_degree:.9f}, max={hi_degree:.9f}; all directions checked')

# A collapsed pair of thresholded columns must still have a valid orthogonal
# Procrustes update. This is Sol's concrete N=3 counterexample.
Vc = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [.5, -.5], [.5, -.5]])
assert np.allclose(Vc.T @ Vc, np.eye(2))
Sc, Rc = step(Vc, np.eye(2), .55)
assert .55 < 1 / np.sqrt(3)
assert np.allclose(Sc, np.array([[1., 1.], [0., 0.], [0., 0.]]))
assert np.linalg.matrix_rank(Sc.T @ Vc, tol=1e-12) == 1
assert np.allclose(Rc.T @ Rc, np.eye(2))
assert abs(np.trace(Rc.T @ (Sc.T @ Vc)) - np.linalg.svd(Sc.T @ Vc, compute_uv=False).sum()) < 1e-13
print('Collapsed-column Procrustes: rank=1; orthogonal SVD completion is optimal')

# S.7 one-step basin on unequal-sized groups, including signed recovery.
sizes = [20, 5, 3, 2]
Hu = normalized_partition(sizes)
ru = len(sizes)
Nu = len(Hu)
angle = .05
Uclose = np.eye(ru)
Uclose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
Uclose = Uclose @ np.diag([1., -1., 1., -1.])
seed_basin = np.random.default_rng(221)
O_basin, _ = np.linalg.qr(seed_basin.standard_normal((ru, ru)))
Vu = Hu @ O_basin
Rstart = Uclose.T @ O_basin  # O_basin Rstart^T = Uclose.
heights = np.abs(Uclose) / np.sqrt(np.array(sizes))[:, None]
offdiag_height = np.max(heights - np.diag(np.diag(heights)))
diag_height = np.min(np.diag(heights))
mu_basin = .1
assert offdiag_height < mu_basin < diag_height
Sbasin, Rbasin = step(Vu, Rstart, mu_basin)
expected = Hu * np.sign(np.diag(Uclose))
assert np.linalg.norm(Sbasin - expected) < 1e-12
Ssecond, Rsecond = step(Vu, Rbasin, mu_basin)
assert np.linalg.norm(Ssecond - expected) < 1e-12
assert np.linalg.norm(Rsecond - Rbasin) < 1e-12
print(f'Nearby-start basin: off={offdiag_height:.9f} < mu=.1 < diagonal={diag_height:.9f}; error={np.linalg.norm(Sbasin-expected):.3e}')

# Reproduce Opus's generic failure with the published threshold, but pair the
# same starts across thresholds. This is the fixed-orthonormal-input core, not
# a claim about the constant-column perturbation or 5000-step cap in FRS code.
def seba_core(V, mu, max_iterations=20000):
    R = np.eye(V.shape[1])
    converged = False
    previous_objective = np.inf
    for iteration in range(max_iterations):
        S, Rnew = step(V, R, mu)
        objective = .5 * np.linalg.norm(V @ Rnew.T - S)**2 + mu * np.abs(S).sum()
        assert objective <= previous_objective + 1e-11
        previous_objective = objective
        converged = np.linalg.norm(Rnew - R) < 1e-13
        R = Rnew
        if converged:
            break
    return S, R, converged, objective


def indicator_distance(S, H):
    best = np.inf
    for perm in itertools.permutations(range(H.shape[1])):
        target = H[:, perm]
        signs = np.where(np.sum(target * S, axis=0) >= 0, 1., -1.)
        best = min(best, np.linalg.norm(S - target * signs))
    return best


rng_rotations = np.random.default_rng(1)
rotations = []
for _ in range(200):
    Z = rng_rotations.standard_normal((ru, ru))
    O, R = np.linalg.qr(Z)
    rotations.append(O * np.sign(np.diag(R)))
assert .99 / np.sqrt(Nu) < 1 / np.sqrt(max(sizes))
sweep_successes = []
for mu in (.99 / np.sqrt(Nu), .1, .05, .02):
    successes = converged_runs = 0
    failures = []
    for O in rotations:
        V = Hu @ O
        assert np.allclose(V.T @ V, np.eye(ru))
        S, R, converged, objective = seba_core(V, mu)
        success = indicator_distance(S, Hu) < 1e-6
        successes += int(success)
        converged_runs += int(converged)
        if not success:
            failures.append((S, converged, objective))
    sweep_successes.append(successes)
    print(f'Unequal-size SEBA: mu={mu:.9f}, recovered={successes}/200, converged={converged_runs}/200')
    if mu > .15:
        # Assert robust qualitative failures, not a BLAS-version-specific count.
        assert 0 < successes < 100
        deficient = [S for S, conv, _ in failures if conv and np.linalg.matrix_rank(S, tol=1e-8) < ru and np.linalg.norm(S[:sizes[0]]) < 1e-8]
        assert deficient
        S = deficient[0]
        display = S * np.where(S.sum(axis=0) >= 0, 1., -1.)
        display = display / display.max(axis=0)
        assert np.allclose(display.min(axis=0), 0., atol=1e-10)
        assert np.linalg.matrix_rank(display, tol=1e-8) == 3
        objectives = [objective for _, _, objective in failures]
        print(f'  Failure objectives={min(objectives):.6f}..{max(objectives):.6f}; largest group lost; sign-corrected minima all zero')
    else:
        assert successes == 200
assert sweep_successes[1:] == [200, 200, 200]

# The coverage union bound is sufficient, not necessary (Sol counterexample).
coverage_exact_failure = 2 * (.5 ** 6)
coverage_sufficient_N = np.log(2 / .05) / .5
assert coverage_exact_failure <= .05 and 6 < coverage_sufficient_N
print(f'Coverage wording: exact failure at N=6 is {coverage_exact_failure:.5f}, sufficient bound is N>={coverage_sufficient_N:.6f}')
print('all sparsity checks passed')
