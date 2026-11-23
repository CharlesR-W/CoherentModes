"""Appendix D (`#measurements`, content/measurements.html): reference charts.

    uv run --with numpy --with scipy --with matplotlib python -I site/experiments/app_measurements.py r20s0b

Writes (only) site/assets/x/app-measurements.json (empty: every chart on the page is an experiment's `app4` panel,
assets/questions.json, d/decod.json or d/curves.json) and site/assets/x/app-measurements.facts.json
({{fact:x.app-measurements.<name>}}): the counts and protocol numbers the page's captions quote that no experiment
or base fact provides. Reads, all existing exports:
    data/<tag>/meta.json, data/<tag>/decod.json, data/<tag>/markov_summary.json
    data/r20s0c/meta_c.json            (baselines meta: permutations of the across-seed set null)
    site/assets/questions.json         (meta.n_perm: the matched-Jaccard permutation bank)
    site/experiments/training-signal.py (N_PERM_SITES, read with ast, not executed)
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # site/ (python -I drops the script directory)
import xlib  # noqa: E402

XID = "app-measurements"


def const_from(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{path.name}: constant {name} not found")


def main(tag):
    src = xlib.DATA / tag
    out = xlib.Out(XID)
    meta = json.loads((src / "meta.json").read_text())
    out.fact("first_step", int(meta["steps"][0]))
    # the time block b9 over the later half of the checkpoints (q_classmix.named.time_b9_4-7_r4)
    out.fact("late_half_first_step", int(meta["steps"][len(meta["steps"]) // 2]))

    q = json.loads((xlib.SITE / "assets" / "questions.json").read_text())
    out.fact("jaccard_n_perm", int(q["meta"]["n_perm"]))

    out.fact("grad_sites_n_perm", int(const_from(xlib.SITE / "experiments" / "training-signal.py", "N_PERM_SITES")))

    ms = json.loads((src / "markov_summary.json").read_text())
    out.fact("markov_n_triples", len(ms["sites"]) - 2)

    mc = json.loads((xlib.DATA / "r20s0c" / "meta_c.json").read_text())
    out.fact("seed_sets_n_perm", int(mc["n_perm"]))

    d = json.loads((src / "decod.json").read_text())
    p = d["protocol"]
    m_train = re.search(r"([\d,]+) train images, ([\d,]+)/class", p)
    m_test = re.search(r"all ([\d,]+) test images", p)
    m_knn = re.search(r"kNN k=(\d+)", p)
    if not (m_train and m_test and m_knn):
        raise SystemExit("decod.json protocol: unexpected wording")
    n_train, per_class = (int(g.replace(",", "")) for g in m_train.groups())
    n_classes = n_train // per_class
    out.fact("decod.n_train", n_train)
    out.fact("decod.n_test", int(m_test.group(1).replace(",", "")))
    out.fact("decod.n_classes", n_classes)
    out.fact("decod.chance", 1 / n_classes, 0, pct=True)
    out.fact("decod.knn_k", int(m_knn.group(1)))
    out.write()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
