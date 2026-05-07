"""Add a project-root sys.path bootstrap to verify scripts that lack one.

After moving verify_*.py → verify/*.py, scripts that import directly from
configs/lifecycle (without sys.path.insert) break because Python prepends
the script's *directory* (verify/) to sys.path, not the CWD.

Insert a path-to-parent-of-this-script bootstrap right before the first
project import. Use underscore-aliased imports so we don't clash with any
later `import os`/`import sys`.
"""
import os
import re

TARGETS = [
    "verify/canonical_small.py",
    "verify/ccv_solver_sim_parity.py",
    "verify/chunking.py",
    "verify/mixed_precision.py",
    "verify/mixed_precision_tiny.py",
    "verify/smoke.py",
    "verify/smoke_small.py",
]

BOOTSTRAP = (
    "# --- path bootstrap (verify/ subdir) ---\n"
    "import os as _os, sys as _sys\n"
    "_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n"
    "# --- end path bootstrap ---\n"
)

# Pattern: first occurrence of an import from a project-local module.
PROJECT_IMPORT = re.compile(
    r"^(from\s+(configs|lifecycle|scripts|data)\b)", re.MULTILINE
)

for path in TARGETS:
    with open(path, "r", encoding="utf-8") as fp:
        src = fp.read()
    if "_sys.path.insert" in src or "sys.path.insert" in src:
        print(f"SKIP (already bootstrapped): {path}")
        continue
    m = PROJECT_IMPORT.search(src)
    if not m:
        print(f"SKIP (no project import found): {path}")
        continue
    insert_at = m.start()
    new_src = src[:insert_at] + BOOTSTRAP + "\n" + src[insert_at:]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        fp.write(new_src)
    print(f"PATCHED: {path}")
