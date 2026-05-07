"""Verify fp32 gather + fp64 FOC produces alphas within 1e-4 of all-fp64.

Smoke gate (#2 in HANDOFF_MIXED_PRECISION_GATHER §6):
  - Default behaviour preserved (gather_precision='f64' baseline matches
    all-fp64 baseline bit-identical — checked separately by verify_smoke).
  - fp32 gather alphas agree with fp64 within 1e-4 relative tolerance.
  - No NaN/Inf at any tail wealth state.
  - Tail-cell spot check (gate #5) at the edges of the (z, state, wealth)
    cube — relative error stays < 1e-4 even at edges.
"""
import time
import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

# Smoke config — same shape as verify/smoke.py so the gate matches.
disc = DiscretizationConfig(
    n_wealth=20, wealth_min=0.13, wealth_max=200.0,
    n_savings=20,
    state_grid_sizes=(3, 3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=5, n_eps_nodes=3, n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 2, 3),
)
base = dict(BASE_CONFIG)
base.update(start_age=60, retire_age=63, terminal_age=65)
var = build_nominal_system1_var_config_hardcoded()
model = build_model(base, var, verbose=False)
pc = build_precompute(model, disc, verbose=False)

print(f"Built model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}")

# Baseline: all fp64.
sc_f64 = CANONICAL_SOLVER._replace(
    max_iter=100, max_iter_unconstrained=100, gather_precision="f64",
)
t0 = time.time()
C64, S64, B64, _ = run_lifecycle_solver(model, pc, sc_f64, verbose=0)
t_f64 = time.time() - t0
print(f"f64 baseline solve: {t_f64:.1f}s")

# Test: fp32 gather.
sc_f32 = CANONICAL_SOLVER._replace(
    max_iter=100, max_iter_unconstrained=100, gather_precision="f32",
)
t0 = time.time()
C32, S32, B32, _ = run_lifecycle_solver(model, pc, sc_f32, verbose=0)
t_f32 = time.time() - t0
print(f"f32 gather solve:   {t_f32:.1f}s  (speedup vs f64: {t_f64/t_f32:.2f}x)")


def relmax(a, b):
    """Max |a - b| / (|b| + eps), ignoring NaN slots in b (unsolved ages)."""
    a = np.asarray(a)
    b = np.asarray(b)
    valid = np.isfinite(b)
    if not valid.any():
        return 0.0
    diff = np.abs(a[valid] - b[valid]) / (np.abs(b[valid]) + 1e-10)
    return float(diff.max())


rel_C = relmax(C32, C64)
rel_S = relmax(S32, S64)
rel_B = relmax(B32, B64)
print(f"\nMax relative error (f32 vs f64):")
print(f"  C={rel_C:.2e}  S={rel_S:.2e}  B={rel_B:.2e}")

# NaN/Inf checks on the f32 solve (only count solved-age slots).
def count_bad(arr):
    a = np.asarray(arr)
    finite_or_unsolved_nan = np.isfinite(a) | np.isnan(a)
    # Unsolved slots are NaN by design; only flag Inf and *unexpected* NaN.
    # The C64 finite mask is the "solved-cell" mask we compare against.
    return int(np.isinf(a).sum()), int((np.isnan(a) & np.isfinite(C64)).sum() if arr is not C64 else 0)

inf_C, nan_C = count_bad(C32)
inf_S, nan_S = count_bad(S32)
inf_B, nan_B = count_bad(B32)
print(f"\nf32 sanity:")
print(f"  Inf: C={inf_C} S={inf_S} B={inf_B}")
print(f"  Unexpected NaN: C={nan_C} S={nan_S} B={nan_B}")

mask = np.isfinite(C64)
print(f"\nalpha_s f64 range: [{S64[mask].min():.4f}, {S64[mask].max():.4f}]")
print(f"alpha_s f32 range: [{S32[mask].min():.4f}, {S32[mask].max():.4f}]")
print(f"alpha_b f64 range: [{B64[mask].min():.4f}, {B64[mask].max():.4f}]")
print(f"alpha_b f32 range: [{B32[mask].min():.4f}, {B32[mask].max():.4f}]")

# Tail-cell spot check (gate #5): edges of the (z, state, wealth) cube.
# Per the handoff, rel err should stay < 1e-4 even at edges.
print("\nTail-cell rel err (solved ages only):")
for z_label, z_idx in [("z=lo", 0), ("z=hi", -1)]:
    for s_label, s_idx in [("state=lo", 0), ("state=hi", -1)]:
        for w_label, w_idx in [("w=lo", 0), ("w=hi", -1)]:
            c64 = C64[mask.all(axis=(1,2,3)) if False else slice(None), z_idx, s_idx, w_idx]
            c32 = C32[:, z_idx, s_idx, w_idx]
            s64 = S64[:, z_idx, s_idx, w_idx]; s32 = S32[:, z_idx, s_idx, w_idx]
            b64 = B64[:, z_idx, s_idx, w_idx]; b32 = B32[:, z_idx, s_idx, w_idx]
            v = np.isfinite(c64)
            if not v.any():
                continue
            rC = np.max(np.abs(c32[v] - c64[v]) / (np.abs(c64[v]) + 1e-10))
            rS = np.max(np.abs(s32[v] - s64[v]) / (np.abs(s64[v]) + 1e-10))
            rB = np.max(np.abs(b32[v] - b64[v]) / (np.abs(b64[v]) + 1e-10))
            print(f"  {z_label:6s} {s_label:9s} {w_label:5s}  C={rC:.2e}  S={rS:.2e}  B={rB:.2e}")

# Hard gate.
TOL = 1e-4
fail = []
if rel_C > TOL: fail.append(f"C rel_err {rel_C:.2e} > {TOL}")
if rel_S > TOL: fail.append(f"S rel_err {rel_S:.2e} > {TOL}")
if rel_B > TOL: fail.append(f"B rel_err {rel_B:.2e} > {TOL}")
if inf_C + inf_S + inf_B > 0: fail.append(f"Inf count > 0")
if nan_C + nan_S + nan_B > 0: fail.append(f"Unexpected NaN > 0")

print()
if fail:
    for f in fail:
        print(f"  FAIL: {f}")
    raise SystemExit(1)
print("Mixed-precision smoke OK")
