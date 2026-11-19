"""Constants, construction fingerprint and import bridge for the Mamba study (SPEC sections 2, 4, 5, 6, 8).

FORK of `../lm_coherent/lm_construction.py`. What changes:
  - the substrate: `state-spaces/mamba-130m-hf` at a pinned commit, checkpoints are LABELS ('init', 'trained')
    instead of integer training steps, layers L4..L24 every 4th of 24, views point/state/stack (+ bag);
  - this study's own CONSTRUCTION dict and FINGERPRINT (it folds in the Pythia fingerprint, which folds in the ResNet
    one, so a change to any reused contract invalidates cached Mamba units);
  - `cached_completion` / `unit_key` / `cached_unit` are re-bound to this fingerprint.
Everything else (the ResNet bridge, the budget ledger arithmetic, the gate predicate, probe identity) is imported
from `lm_construction` unchanged. Importing this module puts `../lm_coherent` and `../cifar_coherent` on sys.path.

Bump `implementation_version` for ANY change to a mathematical output.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_LM = [HERE.parent / 'lm_coherent', Path('/root/lm')]
LM = next((p for p in _LM if (p / 'lm_construction.py').is_file()), _LM[0])
if str(LM) not in sys.path:
    sys.path.insert(0, str(LM))

import lm_construction as LC  # noqa: E402  (puts cifar_coherent on sys.path too)
from lm_construction import (attempt_cost, budget, gate_ok, key_matches, probe_identity,  # noqa: E402,F401
                             resnet_site_analysis)

# ----------------------------------------------------------------------------------------------- substrate
MODEL = 'state-spaces/mamba-130m-hf'
REVISION = '1e76775f628fbf1350fbe4dbb3d971ba64af25a1'      # checked 2026-11-12
PYTHIA_TOKENIZER = ('EleutherAI/pythia-160m', 'step143000')  # the tokenizer the probe was built with
VOCAB = 50280                                               # Mamba config.vocab_size
CHECKPOINTS = ['init', 'trained']
FINAL = 'trained'
INIT_SEED = 0
N_LAYERS = 24                                               # all layers enter the stack view
LAYERS = [4, 8, 12, 16, 20, 24]
POSITIONS = LC.POSITIONS                                    # [4, 8, 16, 32, 64, 128]
SEQ, EOT = LC.SEQ, LC.EOT
SOURCES, PER_SOURCE, N = LC.SOURCES, LC.PER_SOURCE, LC.N
VIEWS = ['point', 'state', 'stack']                         # bag is stored separately as bag_* arrays
PARTS = ['c', 'z']                                          # conv buffer (last 3 inputs), SSM state
K, FLOOR, N_PERM, PERM_SEED = LC.K, LC.FLOOR, LC.N_PERM, LC.PERM_SEED
N_FOLLOW, SEBA_R_SHOWN, TOP_MEMBERS = LC.N_FOLLOW, LC.SEBA_R_SHOWN, LC.TOP_MEMBERS
COMPARE_LAYER = 16
TO_FINAL_POSITIONS = [16, 128]
RESUME_T = 64                                               # C1b hand-off position
C1A_TOL, C1B_TOL = 1e-4, 1e-12
GRAM_ID_TOL = 1e-12                                         # P3 Gram identity
T4_ROOT_TOL = 1e-8                                          # T4 Gram route vs stored stack Gram at the root
TREES = [  # SPEC 6: id, kind, view, layer, position (None = reading block over all positions), checkpoint
    dict(id='T1', kind='reading', view='point', layer=16, position=None, checkpoint='trained'),
    dict(id='T2', kind='reading', view='point', layer=4, position=None, checkpoint='trained'),
    dict(id='T3', kind='single', view='point', layer=16, position=128, checkpoint='trained'),
    dict(id='T4', kind='reading', view='stack', layer=24, position=None, checkpoint='trained'),
]
CROSS_ARCH_PAIRS = [(2 * k, 4 * k) for k in range(1, 7)]   # (Pythia layer, Mamba layer)
PYTHIA_FINAL = LC.FINAL
PROBE_SHA = '461b04503e22d81d150032a9886e4a3939d1ec5aa70855f8f07cbb0803606a65'   # lm_coherent/data/p160t/probe.json

# --------------------------------------------------------------------------------------- construction
CONSTRUCTION = {
    'implementation_version': 'mamba-toy-1-2026-11-12-v1',
    'pythia_fingerprint': LC.FINGERPRINT,
    'model': f'{MODEL}@{REVISION}; HF torch path (no mamba-ssm / causal-conv1d); eval; TF32 off',
    'checkpoints': 'trained = pinned revision; init = torch.manual_seed(0); MambaForCausalLM(config) on CPU',
    'decode': 'recurrent, one token per step, MambaCache, decode branch from an empty cache, '
              'cache_position = max(u, conv_kernel - 1) for sequence index u (EOT is u = 0)',
    'state': 'after reading token t: c = conv_states[..., 1:] (E x 3), z = ssm_states (E x S), every layer',
    'views': {
        'point': 'h^l_t (raw residual, forward hook output of backbone.layers[l-1])',
        'state': 'renormalise(double-centre(mean(C[c^l_t], C[z^l_t]))); C = standard-pipeline cosine Gram',
        'stack': 'renormalise(double-centre(mean_{m<=l, part in c,z} C[part^m_t])) over all 24 layers m <= l',
        'bag': 'as lm_coherent, with backbone.embeddings',
    },
    'tree_rebuild': 'point: vectors re-centred over S; stack (T4): raw uncentred constituent Grams restricted to S, '
                    'double-centred over S, diagonal-normalised (clamped), averaged, renormalised',
    'blocks': 'reading (point, state, stack) and depth (point), trained only',
}
FINGERPRINT = hashlib.sha256(json.dumps(CONSTRUCTION, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cached_completion(path, required_outputs=(), require_fingerprint=True, expected_fingerprint=None):
    return LC._resnet_construction.cached_completion(
        path, required_outputs, require_fingerprint,
        FINGERPRINT if expected_fingerprint is None else expected_fingerprint)


def unit_key(probe_sha=None, removed=()):
    """What every derived unit's completion record must match to be reused: construction, probe, removal list."""
    return dict(construction_fingerprint=FINGERPRINT, probe_sha=probe_sha, removed=sorted(int(i) for i in removed))


def cached_unit(path, required_outputs, key):
    row = cached_completion(path, required_outputs)
    return row if key_matches(row, key) else None


def layer_index(layer):
    return LAYERS.index(layer)


def cache_position(u, conv_kernel=4):
    """HF 4.57.6 `MambaCache.update_conv_state` rolls the buffer left and writes the new input at column
    clamp(cache_position, 0, K-1). Only K-1 puts it in the last slot, so the true index u < K-1 would misplace it.
    Every value >= K-1 is equivalent; we pass the true index wherever the clamp leaves it meaningful."""
    return max(int(u), conv_kernel - 1)
