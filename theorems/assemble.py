"""Assemble THEOREMS.md and SPARSITY.md from theorems/FRONTMATTER.md and theorems/drafts/.

Usage: python3 theorems/assemble.py   (from the repo root)
Headings inside each draft are demoted one level so that each cluster sits under
its Part heading.  Drafts are included verbatim otherwise.
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "theorems" / "drafts"

def demote(text: str) -> str:
    out = []
    in_code = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,5} ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)

def body(name: str) -> str:
    t = (D / name).read_text().rstrip() + "\n"
    return demote(t)

front = (ROOT / "theorems" / "FRONTMATTER.md").read_text().rstrip() + "\n"

parts = [front,
"\n---\n\n# Part A. Guarantees that follow from the construction\n\nNo assumption about trained networks enters Part A. Everything here holds for every finite site set, every view, every layer.\n",
body("A1-exactness.md"), "\n---\n", body("A2-stability.md"), "\n---\n", body("A3-sampling.md"),
"\n---\n\n# Part B. What the construction recovers in idealized regimes\n\nEach cluster opens with named idealizations. Whether a layer satisfies them is a measurement (`THEORY.md` Sections 4.3 and 11), never a premise; the theorems say what is recovered when they hold, with what error when they nearly hold, and what fails when they do not.\n",
body("B1-belief.md"), "\n---\n", body("B2-dictionary.md"), "\n---\n", body("B3-stitching.md"),
]
(ROOT / "THEOREMS.md").write_text("".join(parts))

sp = (D / "SPARSITY-draft.md").read_text().rstrip() + "\n"
header = ("# Sparsity as a certified regime\n\n"
"Status: v1.0, 2026-11-07. Companion to `THEORY.md` (v1.5) and `THEOREMS.md`; written for readers who want to know in what sense this construction recovers sparse or discrete structure. Sparsity is never a premise here: `THEORY.md` treats clustering as a measured regime (Sections 4.2–4.3), and this document proves what is recovered, and how its presence is certified, when that regime holds. Drafted by a GPT-6 Astra agent, reviewed independently by GPT-6.1 Sol and Claude Opus 5.5 agents (`reviews/sparsity-review-*.md`), revised and re-approved. Asserted numerical checks: `uv run --with numpy python -I checks/sparsity_check.py`. The caution in `CLAUDE.md` about the vocabulary of `THEOREMS.md` applies here too.\n\n")
sp = sp.split("\n", 1)[1]  # drop the draft's own H1
(ROOT / "SPARSITY.md").write_text(header + sp.lstrip("\n"))
print("assembled", (ROOT/'THEOREMS.md').stat().st_size, (ROOT/'SPARSITY.md').stat().st_size)
