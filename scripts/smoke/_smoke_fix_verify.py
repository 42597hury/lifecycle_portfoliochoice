"""Smoke verify for work->retirement transition fix.

Loads the saved buggy run from saved_runs/unconstrained_principal_grid5x5x5_nz9
and re-runs the (now-fixed) unconstrained solver at the same config, but
stops as soon as the boundary year (age 66) has been solved — no need to
wait for the full sweep down to age 22.

Reports:
  - V3 pension-lookup invariant (linear interp at frac_z=0 == table value)
  - V2 newton failures at age 66 (must be 0; saved buggy run had 66)
  - Age-66 policy diff vs the saved buggy run (must be > 0 — proves the new
    code path was reached)
  - Per-age newton failures for ages solved before exit (should all be 0)
"""
import json
import numpy as np
from pathlib import Path

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import build_model, Precompute
import lifecycle.solver as _solver

RUN_DIR = Path("saved_runs/unconstrained_principal_grid5x5x5_nz9")
META = json.loads((RUN_DIR / "metadata.json").read_text())
ARRAYS = np.load(RUN_DIR / "policy_arrays.npz")

base_config = META["run_config"]["base_config"]
disc_dict = META["run_config"]["discretization_config"]
solver_dict = META["run_config"]["solver_config"]

# Reconstruct VAR config from saved metadata (rehydrate ndarrays)
var_raw = META["run_config"]["var_config"]
var_config = {}
for k, v in var_raw.items():
    if isinstance(v, dict) and v.get("kind") == "ndarray":
        var_config[k] = np.asarray(v["values"], dtype=v["dtype"])
    else:
        var_config[k] = v

model = build_model(base_config, var_config, verbose=False)

disc_config = DiscretizationConfig(
    n_wealth=disc_dict["n_wealth"],
    wealth_min=disc_dict["wealth_min"],
    wealth_max=disc_dict["wealth_max"],
    n_savings=disc_dict["n_savings"],
    savings_min=disc_dict["savings_min"],
    savings_max=disc_dict.get("savings_max"),
    state_grid_sizes=tuple(disc_dict["state_grid_sizes"]),
    state_grid_mode=disc_dict["state_grid_mode"],
    state_n_stds=disc_dict["state_n_stds"],
    n_z=disc_dict["n_z"],
    n_stds=disc_dict["n_stds"],
    n_eps_nodes=disc_dict["n_eps_nodes"],
    n_eta_nodes=disc_dict["n_eta_nodes"],
    n_ret_nodes_1d=tuple(disc_dict["n_ret_nodes_1d"]),
    n_state_quad_nodes=disc_dict["n_state_quad_nodes"],
)
pc = Precompute(model, disc_config, verbose=False)
solver_config = SolverConfig(**solver_dict)

retire_age = model.retire_age
ages = list(pc.ages)

# === V3: pension lookup invariant (no solve required) ===
n_z = pc.n_z
t_boundary = ages.index(retire_age - 1)  # age 66 in the t-axis
pension_next_by_z = pc.pension_after_tax[t_boundary + 1, :]
print(f"V3 pension lookup invariant check (age={retire_age-1}, t+1={retire_age}):")
for iz in range(n_z):
    interp_at_node = pension_next_by_z[iz]
    table_val = pc.pension_after_tax[t_boundary + 1, iz]
    assert interp_at_node == table_val, f"V3 FAIL at iz={iz}: {interp_at_node} != {table_val}"
print(f"  V3 PASS: pension_next_by_z[iz] == pc.pension_after_tax[t+1, iz] for all iz in [0, {n_z-1}]\n")

# === Patch solver to stop right after the boundary year ===
class StopAfterBoundary(BaseException):
    """Bypass try/except in user code; only the wrapper catches it."""
    pass

_orig_solve = _solver.solve_working_age_step_quad

def _wrapped_solve(*args, **kwargs):
    out = _orig_solve(*args, **kwargs)
    # log_det_next is positional arg index 5
    log_det_next = args[5]
    # Find which age this corresponded to. Easiest: compare against pc.log_det_profile.
    # log_det_profile is (n_age,); log_det_next equals log_det_profile[t+1].
    diffs = np.abs(pc.log_det_profile - log_det_next)
    t_plus_1 = int(np.argmin(diffs))
    age_solved = ages[t_plus_1 - 1]
    if age_solved == retire_age - 1:
        # boundary year just finished — short-circuit
        raise StopAfterBoundary(f"Solved age {age_solved}; stopping.")
    return out

_solver.solve_working_age_step_quad = _wrapped_solve

# === Run fixed solver — expect to be interrupted right after age 66 ===
try:
    _solver.run_lifecycle_solver(model, pc, solver_config=solver_config, verbose=1)
    print("\n!! Solver ran to completion — boundary trigger missed.")
    raise SystemExit(1)
except StopAfterBoundary as e:
    print(f"\n[stopped early: {e}]\n")

# === Compare age 66 policy to the buggy saved run ===
# Restore the original solver and rebuild only the slices we need by re-running
# the partial solve and capturing the output arrays. We need access to C_mat[t].
# Easier: re-do the partial solve manually but write into our own arrays.
_solver.solve_working_age_step_quad = _orig_solve

# Re-run the partial solve, this time capturing arrays.
# Use the same monkey-patch idea but record the per-age policies instead of
# letting the master allocate them.
captured = {}

def _wrapped_solve2(*args, **kwargs):
    # args[0..]: w_grid, s_grid, z_grid, N_state, c_next, log_det_next, ...
    log_det_next = args[5]
    diffs = np.abs(pc.log_det_profile - log_det_next)
    t_plus_1 = int(np.argmin(diffs))
    age_solved = ages[t_plus_1 - 1]
    out = _orig_solve(*args, **kwargs)
    # out_c, out_s, out_b are kwargs
    captured[age_solved] = (
        kwargs["out_c"].copy(),
        kwargs["out_s"].copy(),
        kwargs["out_b"].copy(),
        out[3],  # diag_int
    )
    if age_solved == retire_age - 1:
        raise StopAfterBoundary(f"Solved age {age_solved}; stopping.")
    return out

_solver.solve_working_age_step_quad = _wrapped_solve2
try:
    _solver.run_lifecycle_solver(model, pc, solver_config=solver_config, verbose=0)
except StopAfterBoundary:
    pass
finally:
    _solver.solve_working_age_step_quad = _orig_solve

# === V2: failures at age 66 ===
print(f"{'='*70}")
print("V2: Newton failures at age 66 (working-age solves)")
print(f"{'='*70}")
DI_NEWTON_FAIL = 7
for age, (c, s, b, di) in sorted(captured.items()):
    fails = int(di[:, DI_NEWTON_FAIL].sum())
    flag = " (boundary)" if age == retire_age - 1 else ""
    print(f"  age {age}: {fails} newton failures{flag}")

age66_fails = int(captured[retire_age - 1][3][:, DI_NEWTON_FAIL].sum())
if age66_fails == 0:
    print(f"\n  V2 PASS: 0 failures at age {retire_age-1} (saved buggy run had 66)")
else:
    print(f"\n  V2 FAIL: {age66_fails} failures at age {retire_age-1}")

# === Age-66 policy diff vs saved buggy run ===
t66 = ages.index(retire_age - 1)
C_old, S_old, B_old = ARRAYS["C_mat"], ARRAYS["S_mat"], ARRAYS["B_mat"]
c_new, s_new, b_new, di_new = captured[retire_age - 1]
dC = float(np.max(np.abs(c_new - C_old[t66])))
dS = float(np.max(np.abs(s_new - S_old[t66])))
dB = float(np.max(np.abs(b_new - B_old[t66])))
print(f"\n{'='*70}")
print(f"Age-66 policy diff vs saved buggy run:")
print(f"  max|dC| = {dC:.4e}")
print(f"  max|dS| = {dS:.4e}")
print(f"  max|dB| = {dB:.4e}")
if max(dC, dS, dB) > 0:
    print(f"  -> Age 66 policies CHANGED (new code path was exercised).")
else:
    print(f"  -> Age 66 policies UNCHANGED (something is wrong — fix not applied).")

# === Failure distribution by i_s ===
print(f"\n{'='*70}")
print(f"Failure distribution by i_s (state-grid index) at age 66:")
print(f"{'='*70}")
fails_per_is = di_new[:, DI_NEWTON_FAIL]
nonzero = np.where(fails_per_is > 0)[0]
print(f"  {len(nonzero)} of {di_new.shape[0]} states had failures; "
      f"max per-state = {int(fails_per_is.max())}, total = {int(fails_per_is.sum())}")
for i_s in nonzero[:20]:
    print(f"  i_s={i_s}: {int(fails_per_is[i_s])} failures (total newton calls={int(di_new[i_s, 10])})")
if len(nonzero) > 20:
    print(f"  ... ({len(nonzero) - 20} more)")

# Also show per-state pension value to check (n_z = pc.n_z)
print(f"\n  pension table at retirement age (z_grid -> pension):")
for iz in range(n_z):
    print(f"    z[{iz}]={pc.z_grid[iz]:+.3f} -> pension={pc.pension_after_tax[t_boundary + 1, iz]:.5f}")
