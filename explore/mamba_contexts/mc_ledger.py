"""Budget reservation logic of `mc_modal.py`, as pure functions so that `test_mc.py` can check it without Modal.

The rule (review finding 1, 2026-11-12): a unit counts as complete, and drops out of the reservation, only if it
WILL be skipped when dispatched, i.e. its function really reuses a cached result (it is in CACHED_KINDS) and both
conditions the function itself checks hold: a completion record under the current key, and every required output
present. Every other unit that a stage dispatches is reserved at its worst case, which includes the per-attempt
overhead the ledger charges on reconciliation: (timeout + OVERHEAD_S) x rate.
"""
from __future__ import annotations

import mc_construction as K

OVERHEAD_S = 90            # per attempt: container start and image load, not timed inside the function
# Functions that skip on a matching completion record with all required outputs present (mc_modal). U3 and diag
# always rerun, so they are always reserved.
CACHED_KINDS = frozenset({'bench', 'calib', 'U1_trained', 'U1_init', 'U2_trained', 'U2_init'})
LABEL_FREE_KINDS = frozenset({'bench', 'calib', 'diag'})        # keyed on the construction only, not the probe


def worst_case(kind, timeouts, rate):
    return (timeouts[kind] + OVERHEAD_S) * rate(kind)


def complete_units(units, record_of, key, outputs_of, exists):
    """Names of the units that a dispatch would skip. units: [(name, kind, arg)]; record_of(name) -> completion
    record or None; key: the run key (probe-dependent units); outputs_of(name) -> required output paths;
    exists(path) -> bool."""
    done = set()
    for name, kind, _ in units:
        if kind not in CACHED_KINDS:
            continue
        if kind in LABEL_FREE_KINDS:
            want = K.unit_key(None, [])
        elif kind.startswith('U1'):
            want = dict(key, construction_fingerprint=K.DECODE_FINGERPRINT)       # U1: keyed on the decode
        else:
            want = key
        if not K.key_matches(record_of(name), want):
            continue
        if all(exists(p) for p in outputs_of(name)):
            done.add(name)
    return done


def reservation(units, timeouts, rate, complete):
    """{name: worst case} for every unit of `units` that will execute."""
    return {name: worst_case(kind, timeouts, rate) for name, kind, _ in units if name not in complete}


def completion_row(name, seconds, usd, started_at, completed_at, key, extra):
    """A unit's completion record. The ledger's own fields and the cache key win; an `extra` field with the same
    name (e.g. U2's summary also has `seconds`) is kept under `unit_<name>` (pilot 2 failed on such a clash)."""
    row = dict(unit=name, seconds=seconds, estimated_usd=usd, started_at=started_at, completed_at=completed_at, **key)
    for k, v in extra.items():
        row[f'unit_{k}' if k in row else k] = v
    return row
