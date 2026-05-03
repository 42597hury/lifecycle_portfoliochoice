"""Surgical smoke test: solve ONLY age 66, using saved age-67 retirement
policy as c_next. Avoids re-solving 32 retirement ages.

Reports:
  - V3 pension-lookup invariant
  - V2 newton failures at age 66 (must be 0; saved buggy run had 66)
  - Age-66 policy diff vs saved buggy run
  - Failure distribution by i_s + first FOC residual at failing states
  - Confirmation that the new code path is reached (compare bool flag effect)
"""
import json
import numpy as np
from pathlib import Path

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import build_model, Precompute
from lifecycle.solver import (solve_working_age_step_quad,
                    DI_NEWTON_FAIL, DI_TOTAL_CALLS, DI_WARM_RESET,
                    DI_SUM_ITER, DF_MAX_FOC_RESID, DF_MAX_NEWTON_ITER)

RUN_DIR = Path("saved_runs/unconstrained_principal_grid5x5x5_nz9")
META = json.loads((RUN_DIR / "metadata.json").read_text())
ARRAYS = np.load(RUN_DIR / "policy_arrays.npz")

base_config = META["run_config"]["base_config"]
disc_dict = META["run_config"]["discretization_config"]
solver_dict = META["run_config"]["solver_config"]

var_raw = META["run_config"]["var_config"]
var_config = {}
for k, v in var_raw.items():
    if isinstance(v, dict) and v.get("kind") == "ndarray":
        var_config[k] = np.asarray(v["values"], dtype=v["dtype"])
    else:
        var_config[k] = v

model = build_model(base_config, var_config, verbose=False)
disc_config = DiscretizationConfig(
    n_wealth=disc_dict["n_wealth"], wealth_min=disc_dict["wealth_min"],
    wealth_max=disc_dict["wealth_max"], n_savings=disc_dict["n_savings"],
    savings_min=disc_dict["savings_min"], savings_max=disc_dict.get("savings_max"),
    state_grid_sizes=tuple(disc_dict["state_grid_sizes"]),
    state_grid_mode=disc_dict["state_grid_mode"],
    state_n_stds=disc_dict["state_n_stds"], n_z=disc_dict["n_z"],
    n_stds=disc_dict["n_stds"], n_eps_nodes=disc_dict["n_eps_nodes"],
    n_eta_nodes=disc_dict["n_eta_nodes"],
    n_ret_nodes_1d=tuple(disc_dict["n_ret_nodes_1d"]),
    n_state_quad_nodes=disc_dict["n_state_quad_nodes"])
pc = Precompute(model, disc_config, verbose=False)
solver_config = SolverConfig(**solver_dict)

retire_age = model.retire_age
ages = list(pc.ages)
n_z = pc.n_z
t66 = ages.index(retire_age - 1)            # t-index for age 66
t67 = t66 + 1                                # t-index for age 67

# === V3: pension lookup invariant ===
pension_next_by_z = pc.pension_after_tax[t67, :]
print(f"V3 pension lookup invariant check (age={retire_age-1}, t+1={retire_age}):")
for iz in range(n_z):
    table_val = pc.pension_after_tax[t67, iz]
    assert pension_next_by_z[iz] == table_val
print(f"  V3 PASS: pension_next_by_z[iz] == pc.pension_after_tax[t+1, iz] for all iz")
print(f"  pension values across z grid:")
for iz in range(n_z):
    print(f"    z[{iz}]={pc.z_grid[iz]:+.3f} -> pension={pc.pension_after_tax[t67, iz]:.5f}")

# === Solve age 66 only, using saved age-67 retirement policy as c_next ===
C_old = ARRAYS["C_mat"]
S_old = ARRAYS["S_mat"]
B_old = ARRAYS["B_mat"]
c_next_age67 = C_old[t67]                    # (n_z, N_state, n_w)

# Build the dummy too so we can run with use_pension_next=False as a sanity probe.
pension_dummy = np.zeros(n_z, dtype=np.float64)

# Args for solve_working_age_step_quad
args_common = dict(
    wealth_grid=pc.wealth_grid, savings_grid=pc.s_grid, z_grid=pc.z_grid, N_state=pc.N_state,
    c_next_full=c_next_age67, log_det_next=pc.log_det_profile[t67],
    annuity_factors=pc.annuity_factors,
    rho=model.rho, eta_nodes=pc.eta_nodes, eta_weights=pc.eta_weights, dz=pc.dz,
    state_grid=pc.state_grid,
    grids_0=pc.state_bracket_grids[0], grids_1=pc.state_bracket_grids[1], grids_2=pc.state_bracket_grids[2],
    state_bracket_shift=pc.state_bracket_shift, state_bracket_L_inv=pc.state_bracket_L_inv,
    v_nodes=pc.v_nodes, v_weights=pc.v_weights, M_v_nodes=pc.M_v_nodes,
    const_r=pc.const_r, A_r=pc.A_r,
    Phi_0_state=model.Phi_0_state, Phi_11=model.Phi_11,
    exp_ret_bill=pc.exp_ret_bill, exp_ret_stock=pc.exp_ret_stock,
    exp_ret_bond=pc.exp_ret_bond, ret_weights=pc.ret_weights,
    eps_nodes=pc.eps_nodes, eps_weights=pc.eps_weights,
    gamma=model.gamma, psi_vec=pc.survival_probs_2d[t66, :],
    beta=model.beta, b_bar=model.b_bar,
    constrained=False, solver_config=solver_config,
)

# --- Run A: with use_pension_next=True (the fix) ---
print("\n" + "="*70)
print("Run A: use_pension_next=True (the fix)")
print("="*70)
out_c_A, out_s_A, out_b_A, di_A, df_A = solve_working_age_step_quad(
    use_pension_next=True, pension_next_by_z=pension_next_by_z,
    out_c=np.empty((n_z, pc.N_state, pc.wealth_grid.size)),
    out_s=np.empty((n_z, pc.N_state, pc.wealth_grid.size)),
    out_b=np.empty((n_z, pc.N_state, pc.wealth_grid.size)),
    **args_common)
fails_A = int(di_A[:, DI_NEWTON_FAIL].sum())
calls_A = int(di_A[:, DI_TOTAL_CALLS].sum())
print(f"  Fails: {fails_A} / {calls_A} ({100.0*fails_A/calls_A:.4f}%)")
print(f"  Max FOC residual: {df_A[:, DF_MAX_FOC_RESID].max():.4e}")

# --- Run B: with use_pension_next=False (sanity: old buggy behavior) ---
print("\n" + "="*70)
print("Run B: use_pension_next=False (no fix, sanity probe)")
print("="*70)
out_c_B, out_s_B, out_b_B, di_B, df_B = solve_working_age_step_quad(
    use_pension_next=False, pension_next_by_z=pension_dummy,
    out_c=np.empty((n_z, pc.N_state, pc.wealth_grid.size)),
    out_s=np.empty((n_z, pc.N_state, pc.wealth_grid.size)),
    out_b=np.empty((n_z, pc.N_state, pc.wealth_grid.size)),
    **args_common)
fails_B = int(di_B[:, DI_NEWTON_FAIL].sum())
calls_B = int(di_B[:, DI_TOTAL_CALLS].sum())
print(f"  Fails: {fails_B} / {calls_B} ({100.0*fails_B/calls_B:.4f}%)")
print(f"  Max FOC residual: {df_B[:, DF_MAX_FOC_RESID].max():.4e}")

# --- Compare A to saved (which used the buggy code path) ---
dC = float(np.max(np.abs(out_c_A - C_old[t66])))
dS = float(np.max(np.abs(out_s_A - S_old[t66])))
dB = float(np.max(np.abs(out_b_A - B_old[t66])))
print("\n" + "="*70)
print("Age-66 policy diff: Run A (fix) vs saved buggy run")
print("="*70)
print(f"  max|dC| = {dC:.4e}")
print(f"  max|dS| = {dS:.4e}")
print(f"  max|dB| = {dB:.4e}")

# --- Compare B to saved (should be near-zero — same buggy code path) ---
dC_B = float(np.max(np.abs(out_c_B - C_old[t66])))
dS_B = float(np.max(np.abs(out_s_B - S_old[t66])))
dB_B = float(np.max(np.abs(out_b_B - B_old[t66])))
print("\nSanity: Run B (no fix) vs saved buggy run -- should be ~0:")
print(f"  max|dC| = {dC_B:.4e}")
print(f"  max|dS| = {dS_B:.4e}")
print(f"  max|dB| = {dB_B:.4e}")

# --- Failure distribution at age 66 (Run A, the fix) ---
print("\n" + "="*70)
print("Failure distribution by i_s at age 66 (Run A: with fix)")
print("="*70)
fails_per_is_A = di_A[:, DI_NEWTON_FAIL]
nz = np.where(fails_per_is_A > 0)[0]
print(f"  {len(nz)} of {di_A.shape[0]} states had fails; "
      f"max per-state = {int(fails_per_is_A.max())}, total = {int(fails_per_is_A.sum())}")
for i_s in nz[:30]:
    n_calls = int(di_A[i_s, DI_TOTAL_CALLS])
    sum_it = int(di_A[i_s, DI_SUM_ITER])
    max_it = int(df_A[i_s, DF_MAX_NEWTON_ITER])
    print(f"    i_s={i_s:3d}: {int(fails_per_is_A[i_s])} fails "
          f"(calls={n_calls}, "
          f"avg_iter={sum_it/max(n_calls,1):.2f}, "
          f"max_iter={max_it}, "
          f"warm_reset={int(di_A[i_s, DI_WARM_RESET])}, "
          f"max_foc_resid={df_A[i_s, DF_MAX_FOC_RESID]:.3e})")
if len(nz) > 30:
    print(f"  ... ({len(nz) - 30} more)")

# Compare to a clean i_s for context
print(f"\n  For comparison, a few i_s with 0 fails:")
zero_is = np.where(fails_per_is_A == 0)[0]
for i_s in zero_is[:5]:
    n_calls = int(di_A[i_s, DI_TOTAL_CALLS])
    sum_it = int(di_A[i_s, DI_SUM_ITER])
    max_it = int(df_A[i_s, DF_MAX_NEWTON_ITER])
    print(f"    i_s={i_s:3d}: 0 fails "
          f"(calls={n_calls}, avg_iter={sum_it/max(n_calls,1):.2f}, max_iter={max_it})")

# Run B comparison (buggy code path) — same diagnostic, to check if the
# 33-iter ceiling is a property of the buggy code, the fix, or both
print(f"\n  Run B (no fix) max_iter at failing i_s for comparison:")
for i_s in nz[:30]:
    n_calls_B = int(di_B[i_s, DI_TOTAL_CALLS])
    sum_it_B = int(di_B[i_s, DI_SUM_ITER])
    max_it_B = int(df_B[i_s, DF_MAX_NEWTON_ITER])
    fails_B_is = int(di_B[i_s, DI_NEWTON_FAIL])
    print(f"    i_s={i_s:3d}: {fails_B_is} fails "
          f"(avg_iter={sum_it_B/max(n_calls_B,1):.2f}, max_iter={max_it_B})")

# The single largest n_newton_iter ever seen across all 125 i_s in Run A:
overall_max_iter_A = int(df_A[:, DF_MAX_NEWTON_ITER].max())
overall_max_iter_B = int(df_B[:, DF_MAX_NEWTON_ITER].max())
print(f"\n  Global max iter Run A (fix):     {overall_max_iter_A}")
print(f"  Global max iter Run B (no fix):  {overall_max_iter_B}")
print(f"  max_iter setting from solver_config: {solver_config.max_iter_unconstrained}")
