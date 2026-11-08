"""Run every asserted check behind THEOREMS.md and SPARSITY.md.

Usage, from the repo root:
    uv run --with numpy python -I checks/theorems_check.py

Each sub-check is a standalone script in this directory that ends with its own
"all ... checks passed" line.  This runner executes them in fresh isolated
interpreters, requires every one to exit 0 and print its sentinel, and ends
with "all theorem checks passed" only if all of them did.
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = [
    ("theorems_a1_check.py", "all A1 checks passed"),
    ("theorems_a2_check.py", "all A2 checks passed"),
    ("theorems_a3_check.py", "all A3 checks passed"),
    ("theorems_b1_check.py", "all B1 checks passed"),
    ("theorems_b2_check.py", "all B2 checks passed"),
    ("theorems_b3_check.py", "all B3 checks passed"),
    ("sparsity_check.py", "all sparsity checks passed"),
]

failed = []
for name, sentinel in SCRIPTS:
    path = HERE / name
    if not path.exists():
        print(f"[missing] {name}")
        failed.append(name)
        continue
    proc = subprocess.run(
        [sys.executable, "-I", str(path)], capture_output=True, text=True, cwd=HERE.parent
    )
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    ok = proc.returncode == 0 and last == sentinel
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {last or proc.stderr.strip().splitlines()[-1:]}")
    if not ok:
        failed.append(name)
        if proc.stderr:
            print(proc.stderr[-2000:])

if failed:
    print(f"FAILED: {', '.join(failed)}")
    sys.exit(1)
print("all theorem checks passed")
