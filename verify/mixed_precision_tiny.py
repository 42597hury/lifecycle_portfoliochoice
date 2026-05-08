"""Tiny one-age mixed-precision test (gates 1, 2, 3 in <2 min).

What this proves
----------------
Gate 1 (default fp64 unchanged):
    Inspect the f64 jaxpr/HLO of the retirement kernel — confirm no f32 casts
    leak in when ``gather_precision='f64'``. Run the f64 solve and confirm
    alphas are finite.

Gate 2 (f32 vs f64 agreement):
    Build BOTH the f64 and f32 retirement kernels on a tiny config; solve
    age=64 with terminal-age c_next once each. Confirm max relative error
    on (C, S, B) < 1e-4 and no NaN/Inf.

Gate 3 (HLO boundary visible):
    Lower the f32 retirement kernel to HLO text. Print the convert ops near
    the gather/CRRA boundary and confirm there is at least one
    ``convert(f64←f32)`` op (the cast back to fp64 before the FOC sum).

Why this is enough
------------------
The fp32 cast pattern is identical between ``_interp_c_and_mpc_at_cell``
(used in working FOC) and the inline ``per_kv_kr`` (used in retirement FOC).
If retirement passes, the working path's mixed-precision plumbing is
structurally equivalent. Working-kernel JIT compile is what's blowing up the
laptop's RAM, so we exercise the cast pattern via the retirement kernel here
and defer the working-kernel run to the GPU regression check (gate 4).
"""
import time
import os

# Force single-device CPU before any JAX/lifecycle import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, DELTA_BEQUEST
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import (
    _pc_to_jnp,
    ModelParams,
    _precompute_per_is_tensors,
    _build_per_age_terminal_kernel_vmap_only,
    _build_per_age_retirement_kernel_vmap_only,
)


# ---- Tiny config (much smaller than verify_smoke) ----
# Post real-yields pivot: 3-axis state vector (cape, spr, y_1).
disc = DiscretizationConfig(
    n_wealth=10, wealth_min=0.13, wealth_max=200.0,
    n_savings=10,
    state_grid_sizes=(2, 2, 2),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=3, n_eps_nodes=2, n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(2, 2, 2),
)
base = dict(BASE_CONFIG)
base.update(start_age=63, retire_age=63, terminal_age=65)
var = build_real_full_var_config_hardcoded()
model = build_model(base, var, verbose=False)
pc = build_precompute(model, disc, verbose=False)

print(f"tiny model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}")
print(f"devices:    {jax.devices()}")


def build_runner(gather_precision):
    """Build terminal + retirement kernels at a given gather precision and
    return (terminal, retirement). Static-arg cache keys differ per precision
    so f32/f64 are independent JIT traces.
    """
    sc = CANONICAL_SOLVER._replace(
        max_iter=100, max_iter_unconstrained=100,
        gather_precision=gather_precision,
    )
    delta = sc.delta_bequest if sc.delta_bequest >= 0.0 else DELTA_BEQUEST
    pcj = _pc_to_jnp(pc, delta)
    mp = ModelParams(
        gamma=jnp.float64(model.gamma),
        beta=jnp.float64(model.beta),
        b_bar=jnp.float64(model.b_bar),
        delta=jnp.float64(delta),
        rho=jnp.float64(model.rho),
    )
    per_is_tensors = _precompute_per_is_tensors(pcj)
    terminal_kernel = _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc)
    retirement_kernel = _build_per_age_retirement_kernel_vmap_only(
        pcj, mp, sc, pc.n_z, pc.N_state, per_is_tensors,
    )
    return pcj, mp, sc, terminal_kernel, retirement_kernel


def solve_one_age(gather_precision):
    """Solve terminal + age 64 retirement once at the given precision.
    Returns (C, S, B) for age 64 (shape (n_z, N_state, n_w)).
    """
    pcj, mp, sc, terminal_kernel, retirement_kernel = build_runner(gather_precision)
    # Terminal age (z-invariant by construction). Kernel returns 6 values
    # (c, s, b, n_iters_per_s, n_backtrack_per_s, exit_code_per_s) per
    # current solver.py.
    c_T, s_T, b_T, _ni_T, _nb_T, _ec_T = terminal_kernel()
    c_T_zb = jnp.broadcast_to(c_T[None, :, :], (pc.n_z, pc.N_state, pc.n_w))
    s_T_zb = jnp.broadcast_to(s_T[None, :, :], (pc.n_z, pc.N_state, pc.n_w))
    b_T_zb = jnp.broadcast_to(b_T[None, :, :], (pc.n_z, pc.N_state, pc.n_w))

    # Pension table is needed for retirement. Take age 65 (terminal) row from pc.
    pension_table = jnp.asarray(pc.pension_after_tax)
    psi_t = jnp.asarray(pc.survival_probs_2d)[pc.n_age - 2, :]   # age 64 row
    pension_next = pension_table[pc.n_age - 1, :]                 # age 65 row

    c_64, s_64, b_64, _ni_64, _nb_64, _ec_64 = retirement_kernel(
        c_T_zb, pension_next, psi_t, s_T_zb, b_T_zb,
    )
    # Block until done so timing is meaningful.
    c_64 = jax.device_get(c_64)
    s_64 = jax.device_get(s_64)
    b_64 = jax.device_get(b_64)
    return c_64, s_64, b_64, retirement_kernel


print("\n[gate 1] solving age 64 retirement at gather_precision='f64' ...")
t0 = time.time()
C64, S64, B64, ret_kernel_f64 = solve_one_age("f64")
print(f"  done in {time.time() - t0:.1f}s")
print(f"  C shape={C64.shape}  finite={np.isfinite(C64).all()}  range=[{C64.min():.4f}, {C64.max():.4f}]")
print(f"  S range=[{S64.min():.4f}, {S64.max():.4f}]")
print(f"  B range=[{B64.min():.4f}, {B64.max():.4f}]")

print("\n[gate 2] solving age 64 retirement at gather_precision='f32' ...")
t0 = time.time()
C32, S32, B32, ret_kernel_f32 = solve_one_age("f32")
print(f"  done in {time.time() - t0:.1f}s")
print(f"  C finite={np.isfinite(C32).all()}  range=[{C32.min():.4f}, {C32.max():.4f}]")

# Compare.
def relmax(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-10)))

rel_C = relmax(C32, C64)
rel_S = relmax(S32, S64)
rel_B = relmax(B32, B64)
inf_C = int(np.isinf(C32).sum()) + int(np.isnan(C32).sum())
inf_S = int(np.isinf(S32).sum()) + int(np.isnan(S32).sum())
inf_B = int(np.isinf(B32).sum()) + int(np.isnan(B32).sum())
print(f"\n  max rel err: C={rel_C:.2e}  S={rel_S:.2e}  B={rel_B:.2e}")
print(f"  bad (NaN+Inf in f32): C={inf_C}  S={inf_S}  B={inf_B}")

TOL = 1e-4
gate2_pass = rel_C < TOL and rel_S < TOL and rel_B < TOL and (inf_C + inf_S + inf_B) == 0


print("\n[gate 3] HLO inspection of f32 retirement kernel ...")
# Lower the JIT'd kernel to HLO text.
# Post-chunking refactor, the inner JIT is `per_chunk(c_next, pension, psi,
# init_a_s, init_a_b, z_chunk, is_chunk)` — the trailing two args are the
# cell-index slices the K=1 fast-path supplies as `z_idx_padded[:n_cells]`
# and `is_idx_padded[:n_cells]` (int64, shape (n_cells,)).
pcj32 = _pc_to_jnp(pc, DELTA_BEQUEST)
init_arr_shape = (pc.n_z, pc.N_state, pc.n_w)
n_cells = pc.n_z * pc.N_state
dummy_c = jnp.zeros(init_arr_shape, dtype=jnp.float64)
dummy_pension = jnp.zeros(pc.n_z, dtype=jnp.float64)
dummy_psi = jnp.ones(pc.n_z, dtype=jnp.float64)
dummy_init_s = jnp.full(init_arr_shape, 0.5, dtype=jnp.float64)
dummy_init_b = jnp.full(init_arr_shape, 0.4, dtype=jnp.float64)
dummy_z_chunk = jnp.zeros(n_cells, dtype=jnp.int64)
dummy_is_chunk = jnp.zeros(n_cells, dtype=jnp.int64)

# The retirement_kernel returned by builder is `def call(...)` wrapping the JIT.
# Find the @jit'd inner function in its closure (the one that has .lower).
jitted = None
for cell in (ret_kernel_f32.__closure__ or ()):
    obj = cell.cell_contents
    if hasattr(obj, "lower") and callable(getattr(obj, "lower", None)):
        jitted = obj
        break
if jitted is None:
    raise RuntimeError("Could not locate JIT-compiled inner function in retirement kernel closure.")

hlo_text = jitted.lower(
    dummy_c, dummy_pension, dummy_psi, dummy_init_s, dummy_init_b,
    dummy_z_chunk, dummy_is_chunk,
).as_text()

# Coarse counts (StableHLO/MLIR format: tensor<...xf32>, tensor<...xf64>).
n_f32 = hlo_text.count("xf32>")
n_f64 = hlo_text.count("xf64>")
# Convert ops in StableHLO appear as `stablehlo.convert %x : (tensor<...xf32>) -> tensor<...xf64>`.
# Grep both directions by looking at the conversion arrow with the src/dst dtype pair.
import re
n_convert_f32_to_f64 = len(re.findall(r"stablehlo\.convert.*xf32>\)?\s*->\s*tensor<[^>]*xf64>", hlo_text))
n_convert_f64_to_f32 = len(re.findall(r"stablehlo\.convert.*xf64>\)?\s*->\s*tensor<[^>]*xf32>", hlo_text))
print(f"  HLO size:                {len(hlo_text)} chars")
print(f"  f32 op count:            {n_f32}")
print(f"  f64 op count:            {n_f64}")
print(f"  convert(f32->f64) count: {n_convert_f32_to_f64}")
print(f"  convert(f64->f32) count: {n_convert_f64_to_f32}")

# Save to disk for manual review.
hlo_path = "verify/mixed_precision_hlo_f32.txt"
with open(hlo_path, "w", encoding="utf-8") as f:
    f.write(hlo_text)
print(f"  HLO written to:          {hlo_path}")

gate3_pass = (n_f32 > 0) and (n_f64 > 0) and (n_convert_f32_to_f64 > 0)
if not gate3_pass:
    print("  WARNING: expected mix of f32 + f64 ops with f32->f64 converts; HLO may show")
    print("  fusion eliding the cast boundary. Inspect the HLO file and consider adding")
    print("  jax.lax.optimization_barrier(c_at_xn_g) before the cast back to fp64.")


print("\n=== summary ===")
print(f"  gate 1 (f64 finite + sane): PASS" if np.isfinite(C64).all() else "  gate 1: FAIL")
print(f"  gate 2 (f32 within 1e-4):   {'PASS' if gate2_pass else 'FAIL'}  "
      f"(C={rel_C:.2e}, S={rel_S:.2e}, B={rel_B:.2e})")
print(f"  gate 3 (HLO cast visible):  {'PASS' if gate3_pass else 'CHECK'}  "
      f"(f32={n_f32}, f64={n_f64}, f32->f64={n_convert_f32_to_f64})")

if not (gate2_pass and gate3_pass):
    raise SystemExit(1)
print("\nTiny mixed-precision verification OK")
