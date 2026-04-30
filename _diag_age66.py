"""Age-66 root-cause diagnostics. Uses saved age-67 retirement policy as
c_next; only solves age 66.

Experiments:
  E1. Per-(z, s_val) failure map at i_s in {23, 48, 54}
  E2. c_next-at-age-67 inspection at the same i_s
  E3. R_p / w_inv structural probe — check min_wealth_inv floor exposure
  E4. Re-run age 66 with use_line_search=False to localise the bailout
"""
import json
import numpy as np
from pathlib import Path
import dataclasses

from model import DiscretizationConfig, SolverConfig
from precompute import build_model, Precompute
import solver as _solver

# ----- Setup (same as smoke test) -----
RUN_DIR = Path("saved_runs/unconstrained_principal_grid5x5x5_nz9")
META = json.loads((RUN_DIR / "metadata.json").read_text())
ARRAYS = np.load(RUN_DIR / "policy_arrays.npz")

base_config = META["run_config"]["base_config"]
disc_dict = META["run_config"]["discretization_config"]
solver_dict = META["run_config"]["solver_config"]

var_raw = META["run_config"]["var_config"]
var_config = {k: (np.asarray(v["values"], dtype=v["dtype"])
                  if isinstance(v, dict) and v.get("kind") == "ndarray" else v)
              for k, v in var_raw.items()}

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
t66 = ages.index(retire_age - 1)
t67 = t66 + 1
n_z = pc.n_z
N_state = pc.N_state
n_w = pc.wealth_grid.size

C_old = ARRAYS["C_mat"]
S_old = ARRAYS["S_mat"]
B_old = ARRAYS["B_mat"]
c_next_age67 = C_old[t67]                    # (n_z, N_state, n_w)

pension_next_by_z = pc.pension_after_tax[t67, :].astype(np.float64)
pension_dummy = np.zeros(n_z, dtype=np.float64)

PROBLEM_IS = [23, 48, 54]


# ============================================================================
# E2 — c_next at age 67 inspection
# ============================================================================
print("=" * 78)
print("E2: c_next at age 67 (saved retirement policy) at problem i_s")
print("=" * 78)

for i_s in PROBLEM_IS:
    print(f"\n--- i_s={i_s} ---")
    # state index decomposition
    i0, i1, i2 = i_s // 25, (i_s % 25) // 5, i_s % 5
    s_i = pc.state_grid[i_s]
    print(f"  (y_1_idx, spr_idx, cy_idx) = ({i0}, {i1}, {i2})")
    print(f"  s_i = {s_i}  (raw state vector)")
    # base conditional return mean
    base_mu = pc.const_r + pc.A_r @ s_i
    print(f"  base_mu_r [bill, xr, xb] = {base_mu}")
    # c_next at age 67 across z, at smallest and largest wealth
    print(f"  c_next[i_s={i_s}] at (z, w):")
    print(f"    {'z':>2}  {'pen(z)':>8}  {'c[w_min]':>10}  {'c[w_med]':>10}  {'c[w_max]':>10}")
    for iz in range(n_z):
        c_lo = c_next_age67[iz, i_s, 0]
        c_md = c_next_age67[iz, i_s, n_w // 2]
        c_hi = c_next_age67[iz, i_s, -1]
        print(f"    {iz:>2}  {pension_next_by_z[iz]:>8.5f}  "
              f"{c_lo:>10.4e}  {c_md:>10.4e}  {c_hi:>10.4e}")


# ============================================================================
# E3 — R_p range and w_inv vs min_wealth_inv probe
# ============================================================================
print("\n" + "=" * 78)
print("E3: portfolio-return range at problem i_s vs min_wealth_inv floor")
print("=" * 78)

# Compute R_p min/max across (state-quadrature, return-quadrature) nodes for
# a representative (a_s, a_b) — use the warm-start init values.
init_alpha_s = solver_config.init_alpha_s
init_alpha_b = solver_config.init_alpha_b
print(f"  Init (a_s, a_b) = ({init_alpha_s}, {init_alpha_b})  =>  a_bill={1-init_alpha_s-init_alpha_b:.3f}")
print(f"  min_wealth_inv = {solver_config.min_wealth_inv}")

def rp_range(i_s, a_s, a_b):
    s_i = pc.state_grid[i_s]
    base_mu = pc.const_r + pc.A_r @ s_i
    a_bill = 1.0 - a_s - a_b
    Rp_vals = []
    for k_v in range(pc.v_nodes.shape[0]):
        mu_bill_k = base_mu[0] + pc.M_v_nodes[k_v, 0]
        mu_s_k = base_mu[1] + pc.M_v_nodes[k_v, 1]
        mu_b_k = base_mu[2] + pc.M_v_nodes[k_v, 2]
        for k_r in range(pc.ret_weights.shape[0]):
            R_bill = np.exp(mu_bill_k) * pc.exp_ret_bill[k_r]
            R_s = R_bill * np.exp(mu_s_k) * pc.exp_ret_stock[k_r]
            R_b = R_bill * np.exp(mu_b_k) * pc.exp_ret_bond[k_r]
            Rp = a_s * R_s + a_b * R_b + a_bill * R_bill
            Rp_vals.append(Rp)
    return np.asarray(Rp_vals)

s_grid_min = pc.s_grid[0]
print(f"  smallest savings node s_val = {s_grid_min:.3e}")
for i_s in PROBLEM_IS + [0]:                  # plus a clean i_s for context
    Rp = rp_range(i_s, init_alpha_s, init_alpha_b)
    w_inv_min_at_smallest_s = s_grid_min * Rp.min()
    print(f"  i_s={i_s:3d}: R_p in [{Rp.min():.4e}, {Rp.max():.4e}]  "
          f"=>  w_inv at s_val={s_grid_min:.1e}: min={w_inv_min_at_smallest_s:.3e}  "
          f"(floored: {w_inv_min_at_smallest_s < solver_config.min_wealth_inv})")


# ============================================================================
# E1 — Per-(z, s_val) failure map at problem i_s
#       Re-implements the savings-grid loop in Python so we can record per-call
# ============================================================================
print("\n" + "=" * 78)
print("E1: per-(z, s_val) failure map at problem i_s")
print("=" * 78)

# Pull all the args the JIT portfolio solver needs.
solve_one = _solver.solve_portfolio_unconstrained_working_quad

n_savings = pc.s_grid.size
psi_vec = pc.survival_probs_2d[t66, :]
log_det_next = pc.log_det_profile[t67]
N1 = len(pc.state_bracket_grids[1])
N2 = len(pc.state_bracket_grids[2])

def run_per_call_at_i_s(i_s, use_pension=True):
    """Mirror of the (z, s_val) inner loop in _solve_working_age_step_quad_jit.

    Returns (exit_code_grid, niter_grid, foc_resid_grid) of shape (n_z, n_savings).
    """
    s_i = pc.state_grid[i_s]
    base_mu = pc.const_r + pc.A_r @ s_i
    annuity_factor_is = pc.annuity_factors[i_s]

    pen = pension_next_by_z if use_pension else pension_dummy

    exit_grid = np.zeros((n_z, n_savings), dtype=np.int32)
    niter_grid = np.zeros((n_z, n_savings), dtype=np.int32)
    resid_grid = np.zeros((n_z, n_savings), dtype=np.float64)

    sc = solver_config
    for z_i in range(n_z):
        psi = psi_vec[z_i]
        last_a_s = sc.init_alpha_s
        last_a_b = sc.init_alpha_b
        for j, s_val in enumerate(pc.s_grid):
            opt_s, opt_b, euler, exit_code, foc_resid, n_iter = solve_one(
                s_val, z_i, i_s,
                pc.wealth_grid, c_next_age67, log_det_next,
                annuity_factor_is,
                pc.z_grid, model.rho, pc.eta_nodes, pc.eta_weights, pc.dz,
                pc.v_nodes, pc.v_weights, pc.M_v_nodes, base_mu,
                model.Phi_0_state, model.Phi_11, s_i,
                pc.state_bracket_shift, pc.state_bracket_L_inv,
                pc.state_bracket_grids[0], pc.state_bracket_grids[1], pc.state_bracket_grids[2],
                N1, N2,
                pc.exp_ret_bill, pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
                pc.eps_nodes, pc.eps_weights,
                model.gamma, psi, model.beta, model.b_bar,
                use_pension, pen,
                init_s=last_a_s, init_b=last_a_b,
                tol=sc.tol, max_iter=sc.max_iter_unconstrained,
                tiny_savings=sc.tiny_savings,
                singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                step_damp=sc.step_damp_unconstrained, grad_denom_eps=sc.grad_denom_eps,
                min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                prob_skip=sc.prob_skip_threshold,
                use_line_search=sc.use_line_search,
                max_backtrack_iter=sc.max_backtrack_iter,
                line_search_max_step=sc.line_search_max_step)
            exit_grid[z_i, j] = exit_code
            niter_grid[z_i, j] = n_iter
            resid_grid[z_i, j] = foc_resid
            # Mirror the warm-reset-on-FAIL logic
            if exit_code == _solver.EC_NEWTON_FAIL:
                last_a_s = sc.init_alpha_s
                last_a_b = sc.init_alpha_b
            else:
                last_a_s = opt_s
                last_a_b = opt_b
    return exit_grid, niter_grid, resid_grid


for i_s in PROBLEM_IS:
    print(f"\n--- i_s={i_s} ---")
    exit_grid, niter_grid, resid_grid = run_per_call_at_i_s(i_s, use_pension=True)
    fails = (exit_grid == _solver.EC_NEWTON_FAIL)
    n_fails = int(fails.sum())
    print(f"  total fails: {n_fails}")
    if n_fails > 0:
        # break down by z
        fails_per_z = fails.sum(axis=1)
        print(f"  fails per z (z[0]..z[{n_z-1}]): {list(fails_per_z)}")
        # show first 10 (z, s_val) failures with their iter / resid
        zs, js = np.where(fails)
        print(f"  first {min(10, n_fails)} (z, j_s, s_val, niter, resid):")
        for k in range(min(10, n_fails)):
            print(f"    z={zs[k]} j_s={js[k]:3d} s_val={pc.s_grid[js[k]]:.4e} "
                  f"niter={niter_grid[zs[k], js[k]]:3d} "
                  f"resid={resid_grid[zs[k], js[k]]:.3e}")
        # range of failing s_val
        failing_svals = pc.s_grid[js]
        print(f"  failing s_val range: [{failing_svals.min():.3e}, {failing_svals.max():.3e}]")
        print(f"  s_grid range:        [{pc.s_grid.min():.3e}, {pc.s_grid.max():.3e}]")


# ============================================================================
# E4 — Re-run age 66 with use_line_search=False
# ============================================================================
print("\n" + "=" * 78)
print("E4: re-run age 66 with use_line_search=False")
print("=" * 78)

sc_no_ls = solver_config._replace(use_line_search=False)
# step_damp_unconstrained = 0.3, max_iter_unconstrained = 8000
print(f"  use_line_search={sc_no_ls.use_line_search}, "
      f"max_iter_unconstrained={sc_no_ls.max_iter_unconstrained}, "
      f"step_damp={sc_no_ls.step_damp_unconstrained}")

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
    constrained=False,
    use_pension_next=True, pension_next_by_z=pension_next_by_z)

out_c, out_s, out_b, di, df = _solver.solve_working_age_step_quad(
    solver_config=sc_no_ls,
    out_c=np.empty((n_z, N_state, n_w)),
    out_s=np.empty((n_z, N_state, n_w)),
    out_b=np.empty((n_z, N_state, n_w)),
    **args_common)
fails_no_ls = int(di[:, _solver.DI_NEWTON_FAIL].sum())
calls = int(di[:, _solver.DI_TOTAL_CALLS].sum())
max_it_global = int(df[:, _solver.DF_MAX_NEWTON_ITER].max())
sum_it_global = int(di[:, _solver.DI_SUM_ITER].sum())
print(f"  fails: {fails_no_ls} / {calls} ({100.0*fails_no_ls/calls:.4f}%)")
print(f"  global max_iter: {max_it_global}  (max_iter_unconstrained={sc_no_ls.max_iter_unconstrained})")
print(f"  global avg_iter: {sum_it_global/max(calls,1):.2f}")

print(f"\n  Per-i_s breakdown for problem states (no line search):")
for i_s in PROBLEM_IS:
    print(f"    i_s={i_s}: fails={int(di[i_s, _solver.DI_NEWTON_FAIL])}, "
          f"max_iter={int(df[i_s, _solver.DF_MAX_NEWTON_ITER])}, "
          f"sum_iter={int(di[i_s, _solver.DI_SUM_ITER])}, "
          f"max_resid={df[i_s, _solver.DF_MAX_FOC_RESID]:.3e}")
