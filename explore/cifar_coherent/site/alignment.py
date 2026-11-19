"""Site data for the "Alignment" tab (build_site.py ... --alignment <data dir>).

<data dir> is a measurement directory of the spectral-alignment study (explore/spectral_align/SPEC.md): normally
explore/spectral_align/data, or explore/spectral_align/data_fixture (synthetic, from make_fixture.py). Its analysis
(explore/spectral_align/analyze.py) is kept in explore/spectral_align/analysis/<data dir name>/ and rerun only when
the data directory changes (input stamp). analyze.py needs torch (align.py) and dtw-python; when they are missing and
the analysis is stale, the build stops and prints the command that refreshes it.

Writes viewer/site/d/al/ (index.json, summary_r*.json, diag_r*.bin, s/, U/; layout in analyze.py's docstring) and
returns the al.* facts. The probe-image order of the study is matched to the site's sprite (thumbnails in C5); if the
two probe sets differ, thumbnails are switched off and the facts say so.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                    # explore/cifar_coherent
STUDY = ROOT.parent / "spectral_align"
UV_CMD = ("uv run --with numpy --with scipy --with dtw-python --with torch --index https://download.pytorch.org/whl/cpu "
          "--index-strategy unsafe-best-match python explore/spectral_align/analyze.py {data}")


def log(*a):
    print(*a, flush=True)


def resolve_data(arg: str) -> Path:
    p = Path(arg)
    for cand in (p, Path.cwd() / p, STUDY / p, ROOT.parent.parent / p):
        if (cand / "cells.json").exists():
            return cand.resolve()
    sys.exit(f"--alignment {arg}: no cells.json found (tried {p}, {STUDY / p})")


def analysis_for(data: Path) -> Path:
    sys.path.insert(0, str(STUDY))
    import analyze                                    # numpy-only at import time
    out = STUDY / "analysis" / data.name
    st = out / "stamp.json"
    want = analyze.input_stamp(data)
    if st.exists():
        have = json.loads(st.read_text())
        if have.get("stamp") == want and have.get("format") == analyze.FORMAT:
            log(f"alignment: analysis of {data.name} is current ({out})")
            return out
    log(f"alignment: analysis of {data.name} is missing or stale; running analyze.py")
    try:
        import torch  # noqa: F401
        import dtw  # noqa: F401
    except ImportError:
        sys.exit("alignment: analyze.py needs torch and dtw-python. Run, from the repository root:\n  "
                 + UV_CMD.format(data=data))
    analyze.run(data, out)
    return out


def build(b, arg: str) -> dict:
    data = resolve_data(arg)
    src = analysis_for(data)
    dst = b.d / "al"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("facts.json", "stamp.json"))
    index = json.loads((dst / "index.json").read_text())
    facts = json.loads((src / "facts.json").read_text())

    # thumbnails: study probe order -> the site's sprite order (data/<tag>/probe.npz)
    thumb, note = None, "unavailable (no probe in the study data)"
    pr = index.get("probe") or {}
    if pr.get("idx"):
        with np.load(b.src / "probe.npz") as p:
            site_idx = p["idx"].astype(int).tolist()
        pos = {v: i for i, v in enumerate(site_idx)}
        if all(v in pos for v in pr["idx"]):
            thumb = [pos[v] for v in pr["idx"]]
            same = thumb == list(range(len(thumb)))
            note = "same probe images in the same order as the explorer" if same else "same probe images as the explorer, reordered"
            lab = b.labels.astype(int).tolist()
            if any(lab[t] != l for t, l in zip(thumb, pr["labels"])):
                sys.exit("alignment: probe labels disagree with the explorer's labels for the same image indices")
        else:
            note = "the study's probe images differ from the explorer's; thumbnails off"
    index["thumb"] = thumb
    index["data_dir"] = str(data.relative_to(ROOT.parent.parent)) if data.is_relative_to(ROOT.parent.parent) else data.name
    (dst / "index.json").write_text(json.dumps(index, separators=(",", ":"), allow_nan=False))

    tot = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    vec = sum(p.stat().st_size for p in (dst / "U").rglob("*") if p.is_file())
    facts["al.site_mb"] = f"{tot / 1e6:.1f}"
    facts["al.site_vectors_mb"] = f"{vec / 1e6:.1f}"
    facts["al.site_eager_kb"] = f"{sum((dst / n).stat().st_size for n in ('index.json',)) / 1e3:.0f}"
    facts["al.probe_match"] = note
    facts["al.data_dir"] = index["data_dir"]
    fixture = bool(index.get("fixture"))
    facts["al.data_slug"] = "fixture" if fixture else "measured"
    facts["al.data_note"] = ("SYNTHETIC FIXTURE. Every figure on this tab is drawn from data made by make_fixture.py "
                             "(planted, smoothly drifting subspaces). No number or picture here is a measurement."
                             if fixture else f"Measured data: {index['data_dir']}.")
    log(f"alignment: {tot / 1e6:.1f} MB in d/al ({vec / 1e6:.1f} MB eigenvectors, fetched per snapshot); "
        f"{'FIXTURE' if fixture else 'measured'} data from {data}")
    return facts
