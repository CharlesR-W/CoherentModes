"""Shared construction provenance and strict completion-cache handling.

Bump implementation_version for ANY change to mathematical outputs. This explicit
contract hash covers rule ordering/parameters, precision, normalization, SEBA and
quantization, not incidental logging/validation changes.
"""
import hashlib
import json
from pathlib import Path

CONSTRUCTION = {
    'implementation_version': 'final-five-2026-11-12-v1',
    'rules': ['union15:edge1:nonedge1e-6', 'symmetric-perpoint-perplexity100',
              'symmetric-perpoint-perplexity300', 'median15NN-inverse-distance',
              'median-nonself15NN-mass0.5'],
    'operator': 'zero-diagonal-symmetric-scaling-then-(I+P)/2',
    'scaling': 'float64-log-domain-10-warmups-Newton-row-error<2e-12',
    'spectrum': 'float64-cache-and-Householder-constant-deflation',
    'seba': 'reference-QR-normalized-threshold-.99/sqrtN-parity-max-reliability-original-basis-rotation',
    'export': 'int16-32767-vectors-float32-scales-and-rotations',
    'probe': '2000-images-stratified-RandomState0-sorted',
}
FINGERPRINT=hashlib.sha256(json.dumps(CONSTRUCTION,sort_keys=True,separators=(',',':')).encode()).hexdigest()
SWEEP_FINGERPRINT=hashlib.sha256((FINGERPRINT+'|sweep:v1:globals2,3,5,8,12,16,20,25,30:mass.05,.1,.2,.5:perplexity15,30,100,300:union15,50:additive-floor1e-6').encode()).hexdigest()


def cached_completion(path, required_outputs=(), require_fingerprint=True, expected_fingerprint=FINGERPRINT):
    path=Path(path)
    if not path.is_file() or any(not Path(p).is_file() for p in required_outputs):
        return None
    row=json.loads(path.read_text())
    if require_fingerprint and row.get('construction_fingerprint')!=expected_fingerprint:
        return None
    return row
