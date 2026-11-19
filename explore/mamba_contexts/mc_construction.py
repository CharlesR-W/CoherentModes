"""Constants, construction fingerprint and import bridge for the token-context study (SPEC sections 2-8).

New module for `explore/mamba_contexts/` (a delta against `../mamba_coherent/`). Importing it puts
`../mamba_coherent` (and through it `../lm_coherent` and `../cifar_coherent`) and `../mamba_coherent/diagnosis`
(the token-class taxonomy, validation only) on sys.path. On Modal the same files sit under /root/mb, /root/lm,
/root/cc and /root/mb/diagnosis.

Bump `implementation_version` for ANY change to a mathematical output: the fingerprint keys every cached unit. It
folds in the v1 Mamba fingerprint (which folds in the Pythia and ResNet ones), so a change to a reused contract
invalidates cached units here too.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_MB = [HERE.parent / 'mamba_coherent', Path('/root/mb')]
MBDIR = next((p for p in _MB if (p / 'mb_construction.py').is_file()), _MB[0])
for p in (MBDIR, MBDIR / 'diagnosis'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mb_construction as MB  # noqa: E402  (bridges lm_coherent and cifar_coherent)
from mb_construction import (LC, attempt_cost, budget, gate_ok, key_matches, probe_identity,  # noqa: E402,F401
                             resnet_site_analysis)
import token_classes as TCL  # noqa: E402  (validation labels only; never in a construction)

# ----------------------------------------------------------------------------------------------- substrate
MODEL, REVISION = MB.MODEL, MB.REVISION
CHECKPOINTS, FINAL, INIT_SEED = MB.CHECKPOINTS, MB.FINAL, MB.INIT_SEED
SEQ, EOT = MB.SEQ, MB.EOT
SOURCES = MB.SOURCES
PROBE_SHA = MB.PROBE_SHA
N_LAYERS = 24
LAYERS = list(range(1, N_LAYERS + 1))
SINGLE_LAYERS = [4, 8, 12, 16, 20, 24]                  # single-cell spectra (point, state)
LAG_LAYERS = [8, 16, 24]                                # lag cells (point, state)
THIRDS = {'early': list(range(1, 9)), 'mid': list(range(9, 17)), 'late': list(range(17, 25))}
LAG_POINT_LAYER = 16                                    # B-lag-point
PARTS = ['c', 'z']

# ----------------------------------------------------------------------------------------------- items
N_SNIPPETS, PER_SNIPPET = 1000, 8
N = N_SNIPPETS * PER_SNIPPET
T_MIN, T_MAX = 16, 112
ITEM_SEED = 20261113                                    # + snippet id (SPEC 2)
ORDER_SEED = 720261113                                  # storage order of the items (SPEC: "a seeded random order")
DEDUP_WINDOW = 16
LAGS = [0, 1, 2, 4, 8, 16]
CONTEXT_BEFORE, CONTEXT_AFTER = 24, 4
C3_POSITIONS = [16, 32, 64]                             # v1's sampled positions that fall in 16..112
C3_TOL = 1e-6

# ----------------------------------------------------------------------------------------------- statistics
K, FLOOR = MB.K, MB.FLOOR                               # 15, 1e-6
K_SENS = [8, 30]
N_PERM = 50                                             # walk shares, eta^2, kappa conditional null
PERM_SEED = 20261113
WITHIN_SEED = 20261114                                  # within-token permutations of the predictions
N_PERM_JACC, JACC_SEED = MB.N_PERM, MB.PERM_SEED        # matched Jaccard null: v1's 200-permutation bank
R = [2, 3, 4, 6, 8, 12, 16, 24, 32]
N_EIG = 33                                              # eigenpairs incl. the constant: mu_1..mu_32
N_ETA = 16
N_ENDS = 12
SET_R = [8, 16]
N_EXAMPLES = 12
N_TOP_TOKENS = 8
KMEANS_N_INIT, KMEANS_SEED = 10, 0
GROUPINGS = ['T', 'Nx', 'U', 'A', 'W', 'Cc', 'Cn']      # SPEC 5 (A is the network's output; W, Cc, Cn validation)
LAG_GROUPINGS = ['T', 'Cc', 'U', 'W', 'Ttau']           # lag cells: lag-0 groupings plus the token at t + tau
FEW_GROUPS = ['W', 'Cc', 'Cn']                          # matched Jaccard
CLASSES = list(TCL.CLASSES)

# ----------------------------------------------------------------------------------------------- controls
SINKHORN_TOL = 2e-12
DUPLICATE_TOL = 1e-9
PN_MIN = 10
P3_SUBSAMPLE, P3_TOL, P3_SEED = 500, 1e-12, 0

# ----------------------------------------------------------------------------------------------- fixture (SPEC 8)
FIX_N, FIX_CELLS, FIX_DIM = 1200, 24, 64
FIX_K = [2, 4, 8, 16]
FIX_SIZES = ['balanced', 'zipf']
FIX_S = [0.5, 0.75, 1.0, 2.0]
PLANTED_S = [1.0, 2.0]
DEV_SEEDS = [0, 1, 2]
CHECK_SEEDS = [100, 101, 102, 103, 104]
BENCH_N, BENCH_CHECK = 8000, dict(k=8, sizes='zipf', s=1.0, seed=100)

# ----------------------------------------------------------------------------------------------- trees
TREES = [dict(id='T1', block='B-point', checkpoint='trained'),
         dict(id='T2', block='whole', checkpoint='trained')]

# --------------------------------------------------------------------------------------- construction
CONSTRUCTION = {
    'implementation_version': 'mamba-contexts-1-2026-11-12-v1',
    'mamba_v1_fingerprint': MB.FINGERPRINT,
    'items': f'N = {N}: {PER_SNIPPET} distinct t in {T_MIN}..{T_MAX} per v1 snippet, sequential draws '
             f'RandomState({ITEM_SEED} + snippet id).randint({T_MIN}, {T_MAX + 1}); repeats, and clashes of the multiset '
             f'of the last {DEDUP_WINDOW} tokens with any earlier item, redrawn; storage order RandomState({ORDER_SEED}).permutation',
    'lags': LAGS,
    'decode': 'v1 decode (HF MambaCache, decode branch, cache_position = max(u, 3)); one pass per lag over all '
              'snippets; at step u snapshot the items with t + tau = u (residuals of every layer, conv_states[..., 1:], '
              'ssm_states); predictions sqrt(softmax(lm_head(backbone output))) at lag 0',
    'views': {'point': 'h^l_t, l = 1..24',
              'state': 'running mean of the standard-pipeline cosine Grams of c^l_t, z^l_t; double-centre; renormalise',
              'whole': 'the same over all 48 constituents',
              'bag': 'mean input embedding of tokens 1..t (float64, sorted summation)',
              'recent': 'mean input embedding of tokens t-15..t (float64, sorted summation)',
              'pred': 'sqrt(p_t) through the standard pipeline'},
    'kernel': f'union kNN k = {K} (sensitivity {K_SENS}), floor {FLOOR}, zero diagonal; symmetric_balance; M = (I+P)/2',
    'block': 'running mean of the cells\' P (never stacked); Mbar = (I + Pbar)/2; block_modes(shared_stationary)',
    'spectra': f'{N_EIG} eigenpairs; SEBA at r in {R}; hard sets seba_labels; partitions: SEBA argmax (ties and '
               f'zero rows to the lowest index) and KMeans(k, n_init={KMEANS_N_INIT}, random_state={KMEANS_SEED}) '
               'on psi_1..psi_{k-1}; empty part = invalid',
    'upsilon': 'Upsilon_hat(k) = mu_k / max_j phi(A_j), phi(A) = cut(A)/min(|A|, N-|A|) on Pbar; larger valid '
               'rounding kept; cap 2 mu_k/mu_{k-1}',
    'shares': f'v1 walk shares; exact chance; {N_PERM}-permutation band (2.5/97.5 %)',
    'kappa': 'sum_{i!=j, different snippet} P_ij BC_ij / sum_{same pairs} P_ij; kappa0 = mean BC over different-'
             f'snippet pairs; within-token null: {N_PERM} permutations of the prediction rows within token-at-t groups',
    'eta2': f'between-group share of psi_k variance, k = 1..{N_ETA}; chance (G-1)/(N-1); {N_PERM}-permutation band; '
            'lambda(Mbar)-weighted mean',
    'retention': 'lag: f^T P_c f / f^T f for centred SEBA r = 8 soft vectors; block axis: v1 block_record (M)',
    'tree': 'lm_tree sweep, m_min, depth < 3, n >= 64; node rebuild by the view\'s own construction, streaming',
    'export': 'int16/32767 eigenvectors with float32 scales; tree psi int8/127',
}
# The fingerprint of the original construction. U1's outputs (decode, Grams, predictions, raw vectors) are unchanged
# by every amendment below, so U1 is keyed on this value and its cached outputs stay valid.
DECODE_FINGERPRINT = hashlib.sha256(json.dumps(CONSTRUCTION, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

# Amendments after a run started (SPEC section 11). Each changes U2/U3 outputs, hence the full fingerprint.
AMENDMENTS = {
    'implementation_version': 'mamba-contexts-1-2026-11-13-v2',
    'reference_duplicates': 'after pilot 1: bag, recent, pred exempt from the duplicate assertion (point, state, '
                            'whole cells incl. lag cells keep it); per reference: min off-diagonal distance and the '
                            'number of pairs below 1e-9; any reference with such a pair is rebuilt under item '
                            'reorderings RandomState(1, 2, 3) (undone), a_8, a_16 and AMI of the r = 8, 16 sets vs the '
                            'original stored; tie_sensitive if any a_r < 0.9 or any AMI < 0.8',
}
FINGERPRINT = hashlib.sha256((DECODE_FINGERPRINT + json.dumps(AMENDMENTS, sort_keys=True)).encode()).hexdigest()
REFERENCES = ['bag', 'recent', 'pred']
TIE_SEEDS = [1, 2, 3]
TIE_MIN_A, TIE_MIN_AMI = 0.9, 0.8


def cached_completion(path, required_outputs=(), require_fingerprint=True, expected_fingerprint=None):
    return LC._resnet_construction.cached_completion(
        path, required_outputs, require_fingerprint,
        FINGERPRINT if expected_fingerprint is None else expected_fingerprint)


def unit_key(probe_sha=None, removed=(), decode=False):
    """Cache key of a unit. `decode=True` for U1 (keyed on DECODE_FINGERPRINT, see above)."""
    return dict(construction_fingerprint=DECODE_FINGERPRINT if decode else FINGERPRINT, probe_sha=probe_sha,
                removed=sorted(int(i) for i in removed))


def cached_unit(path, required_outputs, key):
    row = cached_completion(path, required_outputs, expected_fingerprint=key['construction_fingerprint'])
    return row if key_matches(row, key) else None


# ------------------------------------------------------------------------------------------------ json
def jsonable(o):
    """Recursively convert numpy/torch values for json; non-finite floats become strings ('inf', 'nan') so that a
    record of a failed control can still be written (allow_nan=False everywhere)."""
    import math
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, 'tolist') and not isinstance(o, (str, bytes)):
        return jsonable(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if hasattr(o, 'item') and not isinstance(o, (str, bytes, int, float, bool)):
        return jsonable(o.item())
    return o


def write_json(path, obj, indent=None):
    Path(path).write_text(json.dumps(jsonable(obj), allow_nan=False, indent=indent))


# ------------------------------------------------------------------------------------- fail-closed folds
def worst_of(values):
    """max for a control's error-like numbers that fails closed: any non-finite value (NaN, inf) gives inf. Python's
    max(...) silently drops a NaN that follows a finite value; this never does."""
    import math
    vals = [float(v) for v in values]
    if not vals or any(not math.isfinite(v) for v in vals):
        return math.inf
    return max(vals)


def least_of(values):
    """min for a control's margin-like numbers (larger is safer) that fails closed: any NaN gives NaN, so every
    comparison with it is False; an empty input gives NaN too."""
    import math
    vals = [float(v) for v in values]
    if not vals or any(math.isnan(v) for v in vals):
        return math.nan
    return min(vals)
