"""Constants, construction fingerprint and the import bridge to the ResNet study (SPEC sections 2, 5, 10.1).

Every module of this study imports this first. It puts `explore/cifar_coherent/` (and its `site/`, for the
ResNet matched-Jaccard statistics) on `sys.path`, so the ResNet code is reused by import and never copied or
edited. On Modal the same files are added to the image under /root/cc, which is on PYTHONPATH.

Bump `implementation_version` for ANY change to a mathematical output (as in `cifar_coherent/construction.py`).
The fingerprint also folds in the ResNet construction fingerprint, so a change to the reused Sinkhorn,
spectrum or SEBA contract invalidates cached LM units too.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CANDIDATES = [HERE.parent / 'cifar_coherent', Path('/root/cc')]
CIFAR = next((p for p in _CANDIDATES if (p / 'coherent.py').is_file()), _CANDIDATES[0])
if str(CIFAR) not in sys.path:
    sys.path.insert(0, str(CIFAR))

import construction as _resnet_construction  # noqa: E402  (cifar_coherent/construction.py)

RESNET_FINGERPRINT = _resnet_construction.FINGERPRINT


def resnet_site_analysis():
    """The ResNet `site/analysis.py` (matched Jaccard, size ceiling, null summary), loaded under a unique name."""
    name = 'cifar_site_analysis'
    if name in sys.modules:
        return sys.modules[name]
    path = next(p for p in (CIFAR / 'site' / 'analysis.py', CIFAR / 'analysis.py') if p.is_file())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------- substrate
MODEL = 'EleutherAI/pythia-160m'
STEPS = [0, 512, 2000, 16000, 143000]
FINAL = 143000
LAYERS = [2, 4, 6, 8, 10, 12]
POSITIONS = [4, 8, 16, 32, 64, 128]
SEQ = 128                       # snippet tokens; the input is [EOT] + SEQ tokens
EOT = 0
SOURCES = ['maths', 'code', 'wiki', 'forum', 'poetry']
PER_SOURCE = 200
N = len(SOURCES) * PER_SOURCE
VIEWS = ['point', 'slab']       # bag is stored separately as bag_* arrays
K = 15
FLOOR = 1e-6
N_PERM = 200
PERM_SEED = 20261112            # one permutation bank, reused by every statistic (as in the ResNet site)
N_FOLLOW = 10
SEBA_R_SHOWN = [2, 4]
TOP_MEMBERS = 24
TRAIN_RUNS = [[0, 512, 2000, 16000, 143000], [2000, 16000, 143000]]
COMPARE_LAYER = 8
TO_FINAL_POSITIONS = [16, 128]
TREES = [  # SPEC 7.4: id, kind, layer, position (None = reading block over all positions), step
    dict(id='T1', kind='reading', layer=8, position=None, step=143000),
    dict(id='T2', kind='reading', layer=2, position=None, step=143000),
    dict(id='T3', kind='single', layer=8, position=128, step=143000),
    dict(id='T4', kind='reading', layer=8, position=None, step=2000),
]

# --------------------------------------------------------------------------------------- construction (S5, S7)
CONSTRUCTION = {
    'implementation_version': 'lm-toy-1-2026-11-12-v1',
    'resnet_fingerprint': RESNET_FINGERPRINT,
    'standard_pipeline': 'float64; centre over items; row-normalise; C = U U^T clamped to [-1, 1]; d = 1 - C',
    'views': {
        'point': 'h^l_t (raw residual after block l, forward hook output[0])',
        'slab': 'renormalise(double-centre(mean_{u=1..t} C_u)); C_u per-position standard Grams',
        'bag': 'standard pipeline on (1/t) sum_{u<=t} E[x_u], float64, summed in sorted token-id order',
    },
    'kernel': 'union of each row\'s 15 smallest d (stable argsort: ties by lower index); weight 1; 1e-6 elsewhere; '
              'zero diagonal',
    'operator': 'coherent.symmetric_balance (zero-diagonal log-domain symmetric scaling, row error < 2e-12) -> P; '
                'M = (I + P)/2',
    'block': 'equal-weight mean of cell M; coherent.block_modes(shared_stationary=True); schema_v2.block_record',
    'degenerate': 'assert min off-diagonal d > 1e-9; zero centred norm < 1e-10 * median -> remove item run-wide',
    'shares': 'S = (1/N) sum_g sum_{i!=j in g} P_ij; S0 = sum |g|(|g|-1)/(N(N-1)); X = (N-1)/(|A||B|) sum P_AB; '
              f'{N_PERM}-permutation bands, 2.5/97.5 percentiles',
    'tree': 'rebuild on subset (re-centre, kNN among S, Sinkhorn); sweep psi1..3 on mean zero-diagonal P; '
            'm_min = max(16, ceil(0.05 n)); conductance cut / min side; depth < 3 and n >= 64',
    'export': 'int16/32767 eigenvectors with float32 scales (schema_v2); tree psi int8/127 with float32 scales',
}
FINGERPRINT = hashlib.sha256(json.dumps(CONSTRUCTION, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cached_completion(path, required_outputs=(), require_fingerprint=True, expected_fingerprint=None):
    """The ResNet strict completion cache, keyed on this study's fingerprint by default."""
    return _resnet_construction.cached_completion(
        path, required_outputs, require_fingerprint,
        FINGERPRINT if expected_fingerprint is None else expected_fingerprint)


# ------------------------------------------------------------------------------- run provenance and budget
def probe_identity(probe_bytes):
    """Identity of a probe file: sha256 of its bytes."""
    return hashlib.sha256(probe_bytes).hexdigest()


def unit_key(probe_sha=None, removed=()):
    """What every derived unit's completion record must match to be reused: the construction, the probe and the
    run-wide removal list (SPEC 5)."""
    return dict(construction_fingerprint=FINGERPRINT, probe_sha=probe_sha, removed=sorted(int(i) for i in removed))


def key_matches(row, key):
    return row is not None and all(row.get(k) == v for k, v in key.items())


def cached_unit(path, required_outputs, key):
    """The strict completion cache, additionally keyed on probe identity and removal list."""
    row = cached_completion(path, required_outputs)
    return row if key_matches(row, key) else None


def attempt_cost(attempt):
    """An attempt costs its reconciled actual cost if known, else its full reserved worst case (outcome unknown,
    failed, timed out, or still running)."""
    actual = attempt.get('actual_usd')
    return float(actual) if actual is not None else float(attempt['reserved_usd'])


def budget(attempts, worst_by_unit, complete, extra=0.0):
    """Spending guard. attempts: every dispatched attempt ever recorded (append-only; counted regardless of
    fingerprint). worst_by_unit: {unit: timeout x rate}; complete: units with a matching completion record. The
    reservation is spent + the worst case of every incomplete unit + `extra` (e.g. the coordinator itself).
    Units dispatch in sequential batches, so an in-flight attempt is never also counted as an incomplete unit
    by the guard that dispatched it."""
    spent = sum(attempt_cost(a) for a in attempts)
    reserve = sum(w for u, w in worst_by_unit.items() if u not in complete)
    return dict(spent=spent, reserve=reserve, extra=extra, total=spent + reserve + extra)


def gate_ok(record, key):
    """A full run may start only from a passing pilot-gate record made under the same construction, probe and
    removal list."""
    return bool(record and record.get('pass') and key_matches(record, key))
