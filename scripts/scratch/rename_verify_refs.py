"""One-shot script: update all verify_<name>.{py,ipynb,txt} references in
tracked active files to verify/<name>.{py,ipynb,txt}. Skips docs/archive/.
Run from repo root. Idempotent (string replace; second run is a no-op)."""
import os

RENAMES = [
    ("verify/benchmark_bundle_6666.py", "verify/benchmark_bundle_6666.py"),
    ("verify/ccv_solver_sim_parity.py", "verify/ccv_solver_sim_parity.py"),
    ("verify/mixed_precision_hlo_f32.txt", "verify/mixed_precision_hlo_f32.txt"),
    ("verify/mixed_precision_tiny.py", "verify/mixed_precision_tiny.py"),
    ("verify/ee_residuals.py", "verify/ee_residuals.py"),
    ("verify/invalid_cells.py", "verify/invalid_cells.py"),
    ("verify/canonical_small.py", "verify/canonical_small.py"),
    ("verify/benchmark_bundle.py", "verify/benchmark_bundle.py"),
    ("verify/mixed_precision.py", "verify/mixed_precision.py"),
    ("verify/discretization.ipynb", "verify/discretization.ipynb"),
    ("verify/compare_jax.py", "verify/compare_jax.py"),
    ("verify/smoke_small.py", "verify/smoke_small.py"),
    ("verify/ee_simpath.py", "verify/ee_simpath.py"),
    ("verify/arbitrage.py", "verify/arbitrage.py"),
    ("verify/chunking.py", "verify/chunking.py"),
    ("verify/smoke.py", "verify/smoke.py"),
]

SKIP_DIRS = {".git", "saved_runs", ".cache", "venv", "__pycache__", "node_modules"}
SKIP_PATH_FRAGMENTS = ("docs/archive/", "docs\\archive\\")
EXTS = (".md", ".py", ".sh", ".ipynb", ".txt", ".yml", ".yaml", ".toml")

modified = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        if not f.endswith(EXTS):
            continue
        path = os.path.join(root, f)
        norm = path.replace("\\", "/")
        if any(frag.replace("\\", "/") in norm for frag in SKIP_PATH_FRAGMENTS):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                content = fp.read()
        except (UnicodeDecodeError, OSError):
            continue
        new = content
        for old, new_name in RENAMES:
            new = new.replace(old, new_name)
        if new != content:
            with open(path, "w", encoding="utf-8", newline="") as fp:
                fp.write(new)
            modified.append(path)

for m in modified:
    print(m)
print(f"\nTOTAL: {len(modified)} files modified")
