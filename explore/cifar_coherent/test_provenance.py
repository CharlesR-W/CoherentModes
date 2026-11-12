"""Regression: stale same-shape cache records cannot authorize reuse."""
import json
import tempfile
from pathlib import Path
from construction import FINGERPRINT,cached_completion

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);marker=root/'timing_layer_0.json';output=root/'layer_0.npz'
    output.write_bytes(b'placeholder')
    marker.write_text(json.dumps({'construction_fingerprint':'previous-rules','seconds':2}))
    assert cached_completion(marker,[output]) is None
    marker.write_text(json.dumps({'seconds':2}))
    assert cached_completion(marker,[output]) is None
    marker.write_text(json.dumps({'construction_fingerprint':FINGERPRINT,'seconds':2}))
    assert cached_completion(marker,[output])['seconds']==2
    output.unlink()
    assert cached_completion(marker,[output]) is None
print('all cache provenance checks passed')
