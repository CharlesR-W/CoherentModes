"""Assert the population identities and quoted diagnostics in B1-belief.md.

Run from the repository root:
    uv run --with numpy python -I checks/theorems_b1_check.py

Only synthetic validation data are used; no learned model or semantic probe.
"""
import itertools
from fractions import Fraction

import numpy as np


def stationary(P):
    A = np.vstack([P.T - np.eye(len(P)), np.ones(len(P))])
    return np.linalg.lstsq(A, np.r_[np.zeros(len(P)), 1.0], rcond=None)[0]


def root(K):
    e, U = np.linalg.eigh(K)
    return (U * np.sqrt(np.maximum(e, 0))) @ U.T


def invroot(C, rho=0):
    e, U = np.linalg.eigh(C)
    keep = e > 1e-11
    return (U[:, keep] * (e[keep] + rho) ** -0.5) @ U[:, keep].T


def cca(F, G, J, rho=0):
    p, q = J.sum(1), J.sum(0)
    Fc, Gc = F - p @ F, G - q @ G
    Cpp, Cqq = Fc.T @ (p[:, None] * Fc), Gc.T @ (q[:, None] * Gc)
    Cpq = Fc.T @ J @ Gc
    Wp, Wq = invroot(Cpp, rho), invroot(Cqq, rho)
    U, s, Vt = np.linalg.svd(Wp @ Cpq @ Wq)
    return s, Fc @ Wp @ U, Cpq, Cpp


def sphere_features(H, beta=5.0):
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    return root(np.exp(beta * (Hn @ Hn.T - 1)))


def relative_fit(Y, X, p):
    X = np.c_[np.ones(len(X)), X]
    L = np.sqrt(p)[:, None]
    pred = X @ np.linalg.lstsq(L * X, L * Y, rcond=None)[0]
    return np.linalg.norm(L * (Y - pred)) / np.linalg.norm(L * (Y - p @ Y))


np.set_printoptions(precision=10, suppress=True)
# Three-state edge-emitting HMM: T_a = W[:,a] M[a,:].
M = np.array([[1,0,0],[0,1,0],[0,0,1],[.6,.3,.1],[.1,.3,.6]])
W = np.array([[.4,.1,.1,.25,.15],[.1,.4,.15,.1,.25],[.15,.1,.4,.2,.15]])
T, P = W @ M, M @ W
p = stationary(P)
pi = stationary(T)
assert np.allclose(p @ M, pi)
assert np.allclose(W.sum(1), 1)
assert np.linalg.matrix_rank(T) == 3
assert np.linalg.matrix_rank(W) == 3
A = np.array([[1,.2,0],[0,1,.3],[.1,0,1],[.4,.2,.1],[.1,.3,.5]])
assert np.linalg.matrix_rank(A) == 3
H = M @ A.T
F = sphere_features(H)
J = np.diag(p) @ P @ P  # strict <t versus >t, one lag each
s, S, cross, Cpp = cca(F, F, J)
assert np.linalg.matrix_rank(Cpp, tol=1e-10) == 4
assert np.linalg.matrix_rank(cross, tol=1e-10) == 2
assert np.sum(s > 1e-9) == 2
forward_err = relative_fit(S[:, :2], M, p)
backward_err = relative_fit(M, S[:, :2], p)
assert max(forward_err, backward_err) < 2e-14
assert np.allclose(s[:2], [.1120986861, .0754723813], atol=5e-11, rtol=0)
Hankel = np.diag(p) @ P
Hankel_c = Hankel - np.outer(p, p)
assert np.linalg.matrix_rank(Hankel, tol=1e-10) == 3
assert np.linalg.matrix_rank(Hankel_c, tol=1e-10) == 2
print('3-hidden-state / 5-belief exact stationary HMM')
print('stationary belief masses:', p)
print('past covariance rank / cross rank:', np.linalg.matrix_rank(Cpp, tol=1e-10), np.linalg.matrix_rank(cross, tol=1e-10))
print('zero-ridge carried canonical correlations:', s)
print('affine relative errors scores<-belief / belief<-scores:', forward_err, backward_err)
print('token Hankel / centered ranks:', np.linalg.matrix_rank(Hankel, tol=1e-10), np.linalg.matrix_rank(Hankel_c, tol=1e-10))
for rho, expected_err in [(1e-2, .02096), (1e-4, 2.435e-4), (1e-6, 2.439e-6)]:
    sr, Sr, _, _ = cca(F, F, J, rho)
    err = relative_fit(Sr[:, :2], M, p)
    assert np.isclose(err, expected_err, rtol=3e-4, atol=1e-11)
    assert np.sum(sr > 1e-9) == 2
    print('ridge', rho, 'corr', sr[:2], 'affine score error', err)
# Independent documents, one pair per document; samples are NOT exact finite rank.
rng = np.random.default_rng(20261107)
N = 200000
counts = rng.multinomial(N, J.ravel()).reshape(J.shape)
s_emp, _, _, _ = cca(F, F, counts / N, rho=1e-3)
assert np.allclose(s_emp[:4], [.109768, .077067, .003584, .001718], atol=5e-7, rtol=0)
print('iid pairs N=', N, 'ridge=.001 empirical correlations:', s_emp)
# A minimal two-hidden-state HMM with three reachable beliefs.
c = a = d = 1 / 3
M2 = np.array([[1,0],[0,1],[.5,.5]])
P2 = np.array([[c+a,0,d],[0,c+a,d],[a/2,a/2,c+d]])
p2 = stationary(P2)
assert np.allclose(p2, [.25, .25, .5])
assert np.allclose(np.sort(np.linalg.eigvals(P2).real), [1/3, 2/3, 1])
F2 = sphere_features(M2)
s2, _, cross2, _ = cca(F2, F2, np.diag(p2) @ P2 @ P2)
assert np.allclose(s2[:2], [4/9,1/9])
assert np.linalg.matrix_rank(cross2, tol=1e-10) == 2
# Emission likelihood vectors for symbols hold, 0, 1, reset.
W2 = np.array([[c,a,0,d],[c,0,a,d]])
T2_symbols = [c * np.eye(2), a * np.diag([1, 0]), a * np.diag([0, 1]), d * np.ones((2, 2)) / 2]
pi2 = np.array([.5, .5])
token_joint = np.array([[pi2 @ Ta @ Tb @ np.ones(2) for Tb in T2_symbols] for Ta in T2_symbols])
centered = token_joint - np.outer(token_joint.sum(1), token_joint.sum(0))
assert np.linalg.matrix_rank(token_joint, tol=1e-10) == 2
assert np.linalg.matrix_rank(centered, tol=1e-10) == 1
print('2-hidden-state counterexample carried correlations:', s2)
print('counterexample carried / centered token ranks:', np.linalg.matrix_rank(cross2, tol=1e-10), np.linalg.matrix_rank(centered, tol=1e-10))
# Three causal states but only one centered predictive direction.
Q = np.array([[.6,.3,.1],[.2,.3,.5],[.4,.3,.3]])
q = stationary(Q)
assert np.min(Q) > 0 and np.unique(Q, axis=0).shape[0] == 3
assert np.linalg.matrix_rank(Q) == 2
assert np.linalg.matrix_rank(np.diag(q) @ Q - np.outer(q,q), tol=1e-10) == 1
print('three distinct Markov causal states / centered rank: 3 / 1')
# Uniform previous-token-copy read versus carried, V=3.
V = 3
Sigma = np.eye(V)/V - np.ones((V,V))/V**2
assert np.linalg.matrix_rank(Sigma, tol=1e-10) == V-1
print('V=3 previous-token copy carried/read ranks: 0 /', np.linalg.matrix_rank(Sigma, tol=1e-10))


# Reviewer counterexample: affine means do not imply a probability extension.
mixture_beliefs = np.array([[.4, .6], [.6, .4]])
vertex_masses = np.linalg.solve(mixture_beliefs, np.array([0., 1.]))
assert np.allclose(vertex_masses, [3., -2.])
assert np.allclose(mixture_beliefs @ np.array([[.8,.2],[.2,.8]]), [[.44,.56],[.56,.44]])
assert relative_fit(np.eye(2), mixture_beliefs, np.ones(2)/2) < 1e-14

# B.6 is an observed first-order Markov chain, not a history-dependent filter.
reset_matrices = [np.outer(W[:, a], M[a]) for a in range(len(M))]
markov_error = 0.
for a, b, c0 in itertools.product(range(len(M)), repeat=3):
    p_ab = (pi @ reset_matrices[a] @ reset_matrices[b]).sum()
    p_abc = (pi @ reset_matrices[a] @ reset_matrices[b] @ reset_matrices[c0]).sum()
    markov_error = max(markov_error, abs(p_abc/p_ab - P[b, c0]))
assert markov_error < 1e-14
full_l2 = (J - np.outer(p, p)) / np.sqrt(np.outer(p, p))
assert np.allclose(np.linalg.svd(full_l2, compute_uv=False)[:2], s[:2], atol=1e-12)
print('B.6 Markov property and equality to one-hot token CCA verified')

# Enumerate ALL B.11 variants, not just the contrast-covariance rank.
for vocab in [2, 3]:
    eye = np.eye(vocab)
    pairs = list(itertools.product(range(vocab), repeat=2))
    Jcache = np.zeros((len(pairs), len(pairs)))
    for shared, old, future in itertools.product(range(vocab), repeat=3):
        Jcache[pairs.index((shared, old)), pairs.index((future, shared))] += vocab**-3
    pair_views = np.array([np.r_[eye[a], eye[b]]/np.sqrt(2) for a, b in pairs])
    pair_features = sphere_features(pair_views)
    scache, _, _, _ = cca(pair_features, pair_features, Jcache)
    assert np.allclose(scache[:vocab-1], 1, atol=1e-12)
    assert np.sum(scache > 1e-9) == vocab-1
    token_features = sphere_features(eye)
    sind, _, _, _ = cca(token_features, token_features, np.ones((vocab, vocab))/vocab**2)
    sread, _, _, _ = cca(token_features, token_features, np.eye(vocab)/vocab)
    assert np.max(sind) < 1e-12
    assert np.allclose(sread[:vocab-1], 1, atol=1e-12)
    sig = np.eye(vocab)/vocab - np.ones((vocab, vocab))/vocab**2
    linear_corr = np.linalg.svd(invroot(sig) @ sig @ invroot(2*sig), compute_uv=False)
    assert np.allclose(linear_corr[:vocab-1], 1/np.sqrt(2), atol=1e-12)
    retained = [(a, b) for a, b in pairs if a != b]
    Ju = np.zeros((vocab, len(retained)))
    for j, (a, b) in enumerate(retained):
        Ju[a, j] = 1/len(retained)
    updates = np.array([eye[a]-eye[b] for a, b in retained])
    supdate, _, _, _ = cca(token_features, sphere_features(updates), Ju)
    assert np.allclose(supdate[:vocab-1], 1, atol=1e-12)
    assert np.sum(supdate > 1e-9) == vocab-1
    assert np.isclose(1-len(retained)/vocab**2, 1/vocab)
print('B.11 exact cache, carried, read, linear-update and retained-sphere ranks/correlations verified')

# B.13: valid positive joint law, known affine span, and an explicit defect.
rootp = np.sqrt(p)
belief_eval = rootp[:, None] * (M - p @ M)
Ub, sb, _ = np.linalg.svd(belief_eval, full_matrices=False)
belief_projector = Ub[:, :2] @ Ub[:, :2].T
U0, sv0, Vt0 = np.linalg.svd(full_l2)
left_null_projector = np.eye(len(p)) - belief_projector - np.outer(rootp, rootp)
left = left_null_projector[:, np.argmax(np.linalg.norm(left_null_projector, axis=0))]
left /= np.linalg.norm(left)
right_null_projector = np.eye(len(p)) - Vt0[:2].T @ Vt0[:2] - np.outer(rootp, rootp)
right_null = right_null_projector[:, np.argmax(np.linalg.norm(right_null_projector, axis=0))]
right_null /= np.linalg.norm(right_null)
right = (Vt0[0] + right_null)/np.sqrt(2)
T_pert = full_l2 + .005 * np.outer(left, right)
J_pert = J + .005 * np.outer(rootp*left, rootp*right)
assert np.min(J_pert) > 0
assert np.allclose(J_pert.sum(0), p) and np.allclose(J_pert.sum(1), p)
T_aff = belief_projector @ T_pert
T_def = T_pert - T_aff
gamma = np.linalg.svd(T_aff, compute_uv=False)[1]
eps = np.linalg.norm(T_def, 2)
assert np.isclose(eps, .005)
Upert, spert, _ = np.linalg.svd(T_pert)
angle = np.linalg.norm((np.eye(len(p))-belief_projector) @ Upert[:, :2], 2)
assert gamma > 2*eps
assert spert[2] <= eps+1e-13 and spert[1] >= gamma-eps-1e-13
assert angle <= eps/(gamma-eps)+1e-13
assert np.sum(spert > .025) == 2
# Actual ridge-score smoothing bound on the common finite-support L2 space.
weighted_features = rootp[:, None] * (F - p @ F)
function_cov = weighted_features @ weighted_features.T
vals, vecs = np.linalg.eigh(function_cov)
vals = np.maximum(vals, 0)
rho_approx = 1e-5
smooth_root = (vecs * np.sqrt(vals/(vals+rho_approx))) @ vecs.T
T_ridge = smooth_root @ T_pert @ smooth_root
ridge_error = np.linalg.norm(T_ridge-T_pert, 2)
eta = eps+ridge_error
Ur, sr, _ = np.linalg.svd(T_ridge)
zeta = np.linalg.norm((np.eye(len(p))-smooth_root) @ Ur[:, :2], 2)
score_frame, _ = np.linalg.qr(smooth_root @ Ur[:, :2])
score_angle = np.linalg.norm((np.eye(len(p))-belief_projector) @ score_frame, 2)
assert gamma > 2*eta and zeta < 1
assert sr[2] <= eta+1e-13 and sr[1] >= gamma-eta-1e-13
score_bound = min(1, (eta/(gamma-eta)+zeta)/(1-zeta))
assert score_angle <= score_bound+1e-13
print('B.13 zero-ridge and actual ridge-score perturbation bounds verified')

# Mess3: fixed-seed, long-run diagnostic, NOT i.i.d. population CCA or a
# stationary-filter exact sample. The analytic ATM failure proof is B.12.
x_mess, alpha_mess = .05, .85
beta_emission = (1-alpha_mess)/2
base = np.full((3, 3), x_mess)
np.fill_diagonal(base, 1-2*x_mess)
mess_symbols = [base @ np.diag([alpha_mess if j == a else beta_emission for j in range(3)]) for a in range(3)]
assert np.allclose(sum(mess_symbols), base)
assert np.allclose(np.linalg.eigvalsh(base), [.85, .85, 1.])
assert all(np.min(Ta) > 0 and abs(np.linalg.det(Ta)) > 1e-8 for Ta in mess_symbols)
uniform = np.ones(3)/3
for a, Ta in enumerate(mess_symbols):
    out = uniform @ Ta
    out /= out.sum()
    assert np.allclose(out, [alpha_mess if j == a else beta_emission for j in range(3)])
    assert np.linalg.norm(out-uniform) > .5

mess_rng = np.random.default_rng(20261108)
latent = 0
belief = uniform.copy()
rows = []
for t in range(60000):
    probs = np.array([Ta[latent].sum() for Ta in mess_symbols])
    sym = mess_rng.choice(3, p=probs/probs.sum())
    latent = mess_rng.choice(3, p=mess_symbols[sym][latent]/mess_symbols[sym][latent].sum())
    belief = belief @ mess_symbols[sym]
    belief /= belief.sum()
    if t >= 1000 and t % 20 == 0:
        rows.append(belief.copy())
beliefs = np.array(rows)
assert beliefs.shape == (2950, 3)
refs = beliefs[mess_rng.choice(len(beliefs), 200, replace=False)]
refs_norm = refs / np.linalg.norm(refs, axis=1, keepdims=True)

def conditional_sections(lag):
    means = np.zeros((len(beliefs), len(refs)))
    for word in itertools.product(range(3), repeat=lag):
        product = np.eye(3)
        for a in word:
            product = product @ mess_symbols[a]
        unnormalized = beliefs @ product
        word_probs = unnormalized.sum(1)
        directions = unnormalized / np.linalg.norm(unnormalized, axis=1, keepdims=True)
        means += word_probs[:, None] * np.exp(5*(directions @ refs_norm.T-1))
    return means

for lag, label, expected_ratio, expected_residual in [
    (2, 'carried', .0150055709203, .0140056748295),
    (1, 'read-style', .0301274417429, .0402949768392),
]:
    means = conditional_sections(lag)
    centered_means = means-means.mean(0)
    singular = np.linalg.svd(centered_means/np.sqrt(len(beliefs)), compute_uv=False)
    ratios = singular/singular[0]
    residual = relative_fit(means, beliefs, np.ones(len(beliefs))/len(beliefs))
    assert np.isclose(ratios[2], expected_ratio, rtol=1e-7, atol=1e-11)
    assert np.isclose(residual, expected_residual, rtol=1e-7, atol=1e-11)
    print('Mess3', label, 'third/first singular value:', ratios[2], 'relative affine residual:', residual)

token_means = np.array([[np.sum(b @ Ta @ Tb) for Ta, Tb in itertools.product(mess_symbols, repeat=2)] for b in beliefs])
token_residual = relative_fit(token_means, beliefs, np.ones(len(beliefs))/len(beliefs))
assert token_residual < 2e-14
assert np.linalg.matrix_rank(token_means-token_means.mean(0), tol=1e-10) == 2
print('Mess3 two-token indicator affine residual:', token_residual, 'rank: 2')
# B.14: analytic noncompactness bounds, not estimates from a sampled spectrum.
# Fractions avoid treating a coarse simplex-grid cutoff as an exact inequality.
xf, af, rf = Fraction(1, 20), Fraction(17, 20), Fraction(3, 40)
lamf = 1-3*xf

def symbol_probability(z):
    return rf+(af-rf)*(xf+lamf*z)

def own_coordinate(z):
    return af*(xf+lamf*z)/symbol_probability(z)

cross_cutoff = rf*(1-2*xf)/(af*xf+rf*(1-xf))
assert cross_cutoff == Fraction(54, 91)
assert np.isclose(float(cross_cutoff), .5934065934, atol=5e-11, rtol=0)
region_cutoff = Fraction(2, 3)
preimage_cutoff = Fraction(2, 17)
assert cross_cutoff < region_cutoff
assert own_coordinate(preimage_cutoff) == region_cutoff
assert own_coordinate(Fraction(0)) == Fraction(34, 91) > preimage_cutoff
assert own_coordinate(own_coordinate(Fraction(0))) > region_cutoff
assert own_coordinate(region_cutoff) > region_cutoff
read_probability_lower = symbol_probability(preimage_cutoff)
carried_probability_lower = read_probability_lower*symbol_probability(region_cutoff)
assert read_probability_lower == Fraction(153, 800)
assert symbol_probability(region_cutoff) == Fraction(1327, 2400)
assert carried_probability_lower == Fraction(67677, 640000)
read_bound = np.sqrt(float(read_probability_lower))
carried_bound = np.sqrt(float(carried_probability_lower))
assert np.isclose(read_bound, .4373213921, atol=5e-11, rtol=0)
assert np.isclose(carried_bound, .3251850435, atol=5e-11, rtol=0)
assert read_bound > .406 and carried_bound > .289
# Independent matrix-level checks on vertices and a deterministic simplex slice.
for a in range(3):
    for other in range(3):
        if other == a:
            continue
        vertex_out = np.eye(3) @ mess_symbols[other]
        vertex_out /= vertex_out.sum(1, keepdims=True)
        assert np.isclose(vertex_out[:, a].max(), float(cross_cutoff))
z_values = np.linspace(0, 1, 1001)
input_beliefs = np.c_[z_values, (1-z_values)/2, (1-z_values)/2]
first_unnormalized = input_beliefs @ mess_symbols[0]
p_first = first_unnormalized.sum(1)
first = first_unnormalized/p_first[:, None]
second_unnormalized = first_unnormalized @ mess_symbols[0]
p_aa = second_unnormalized.sum(1)
exclusive_preimages = first[:, 0] > float(region_cutoff)
assert np.any(exclusive_preimages)
assert p_first[exclusive_preimages].min() >= float(read_probability_lower)
assert p_aa[exclusive_preimages].min() >= float(carried_probability_lower)
print('Mess3 B.14 rigorous variational lower bounds: read-style', read_bound, 'carried', carried_bound)
print('all B1 checks passed')
