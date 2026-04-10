"""
Thorough test suite for the continuous-z simulation refactor.

Tests:
  1.  Scalar tax/pension functions match vectorized originals
  2.  log_det_profile and avg_det consistency
  3.  Simulation output structure
  4.  z_idx nearest-neighbor correctness
  5.  z dynamics: no systematic drift
  6.  z variance convergence toward unconditional
  7.  z frozen in retirement
  8.  Income positivity and plausibility
  9.  Policy interpolation sanity
  10. Mixture draw statistics (implied eta moments)
  11. Income-dependent mortality effect
  12. Grid-point income consistency with precomputed table
"""

import sys
import numpy as np
from policy_io import load_policy_bundle
from precompute import Precompute, build_model
from model import (
    DiscretizationConfig, disposable_income_working, compute_pension_after_tax,
)
from simulation import (
    simulate_lifecycle, _scalar_disposable_income, _scalar_pension_after_tax,
)


# ================================================================
# Setup
# ================================================================

def setup():
    C, S, B, _diag, meta = load_policy_bundle(
        "saved_runs/constrained_grid5x5x5_nz11"
    )
    rc = meta["run_config"]
    bc = rc["base_config"]
    vc = rc["var_config"]
    dc_raw = rc["discretization_config"]

    disc = DiscretizationConfig(
        n_wealth=dc_raw["n_wealth"],
        wealth_min=dc_raw["wealth_min"],
        wealth_max=dc_raw["wealth_max"],
        n_savings=dc_raw["n_savings"],
        savings_min=dc_raw["savings_min"],
        state_grid_sizes=tuple(dc_raw["state_grid_sizes"]),
        n_z=dc_raw["n_z"],
        n_eps_nodes=dc_raw["n_eps_nodes"],
        n_ret_nodes_1d=dc_raw["n_ret_nodes_1d"],
    )

    for key in ("Phi", "Omega"):
        vc[key] = np.array(vc[key], dtype=float)
    vc["z_bar"] = np.array(vc["z_bar"]["values"], dtype=float)

    model = build_model(bc, vc, verbose=False)
    pc = Precompute(model, disc, verbose=False)
    return C, S, B, model, pc


# ================================================================
# Test harness
# ================================================================

n_pass = 0
n_fail = 0


def test(name, condition, detail=""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  PASS  {name}")
    else:
        n_fail += 1
        print(f"  FAIL  {name}  -- {detail}")


# ================================================================
# Main
# ================================================================

def main():
    global n_pass, n_fail

    C, S, B, model, pc = setup()

    # Force Numba compilation of scalar helpers
    _ = _scalar_disposable_income(1.0)
    _ = _scalar_pension_after_tax(0.0, 1.0)

    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: Scalar tax function vs vectorized model function")
    print("=" * 70)

    y_test = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0,
                        3.0, 5.0, 10.0, 20.0])
    vec_result = disposable_income_working(y_test)
    scalar_results = np.array([_scalar_disposable_income(y) for y in y_test])
    max_diff = np.max(np.abs(vec_result - scalar_results))
    test("Scalar tax matches vectorized (labor income)", max_diff < 1e-14,
         f"max diff = {max_diff:.2e}")

    z_test = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    vec_pension = compute_pension_after_tax(z_test, pc.avg_det)
    scalar_pension = np.array(
        [_scalar_pension_after_tax(z, pc.avg_det) for z in z_test]
    )
    max_diff_p = np.max(np.abs(vec_pension - scalar_pension))
    test("Scalar tax matches vectorized (pension)", max_diff_p < 1e-14,
         f"max diff = {max_diff_p:.2e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 2: log_det_profile and avg_det consistency")
    print("=" * 70)

    ages = pc.ages
    log_det_check = (model.b0 + model.b1 * ages
                     + model.b2 * ages**2 / 10.0
                     + model.b3 * ages**3 / 100.0)
    test("log_det_profile matches formula",
         np.allclose(pc.log_det_profile, log_det_check, atol=1e-15))

    working_ages = np.arange(model.start_age, model.retire_age)
    log_det_work = (model.b0 + model.b1 * working_ages
                    + model.b2 * working_ages**2 / 10.0
                    + model.b3 * working_ages**3 / 100.0)
    avg_det_check = float(np.mean(np.exp(log_det_work)))
    test("avg_det matches independent computation",
         abs(pc.avg_det - avg_det_check) < 1e-14,
         f"diff = {abs(pc.avg_det - avg_det_check):.2e}")

    # Verify working_income table matches direct computation at grid points
    diffs = []
    for t in [0, 20, 44]:
        for iz in [0, 5, 10]:
            for ie in [0, 4, 9]:
                yg = np.exp(pc.log_det_profile[t]
                            + pc.z_grid[iz] + pc.eps_nodes[ie])
                d = abs(_scalar_disposable_income(yg)
                        - pc.working_income[t, iz, ie])
                diffs.append(d)
    test("Direct income matches table at 27 grid points",
         max(diffs) < 1e-12, f"max diff = {max(diffs):.2e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 3: Simulation output structure")
    print("=" * 70)

    sim = simulate_lifecycle(C, S, B, pc, model, n_simulations=10_000,
                             initial_z="median", seed=42, verbose=False)

    test("Output has z key (float64)",
         "z" in sim and sim["z"].dtype == np.float64)
    test("Output has z_idx key (int32)",
         "z_idx" in sim and sim["z_idx"].dtype == np.int32)
    test("z shape matches (n_sim, n_age)",
         sim["z"].shape == (10_000, pc.n_age))
    test("z_idx shape matches",
         sim["z_idx"].shape == (10_000, pc.n_age))
    test("All z_idx in valid range [0, n_z-1]",
         int(sim["z_idx"].min()) >= 0
         and int(sim["z_idx"].max()) < pc.n_z)

    alive = sim["alive"]
    z_alive = sim["z"][alive]
    test("All z values within grid bounds (alive)",
         float(z_alive.min()) >= pc.z_grid[0] - 1e-10
         and float(z_alive.max()) <= pc.z_grid[-1] + 1e-10,
         f"min={z_alive.min():.6f}, max={z_alive.max():.6f}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 4: z_idx is correct nearest-neighbor of z")
    print("=" * 70)

    for label, t_ck in [("age 52", 30), ("age 66", 44)]:
        mask = alive[:, t_ck]
        z_cont = sim["z"][mask, t_ck]
        z_idx = sim["z_idx"][mask, t_ck]
        true_nn = np.argmin(
            np.abs(z_cont[:, None] - pc.z_grid[None, :]), axis=1
        )
        frac = float(np.mean(z_idx == true_nn))
        test(f"z_idx = nearest grid index at {label}", frac == 1.0,
             f"match = {frac*100:.2f}%")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 5: z dynamics -- no systematic drift")
    print("=" * 70)

    print(f"  {'t':>3}  {'Age':>4}  {'N':>6}  {'Mean_z':>8}  "
          f"{'Std_z':>7}  {'SE_mean':>8}  {'|t-stat|':>8}")
    all_tstat_ok = True
    for t in [5, 15, 25, 35, 44]:
        m = alive[:, t]
        z_v = sim["z"][m, t]
        n = len(z_v)
        se = float(z_v.std() / np.sqrt(n))
        tstat = abs(float(z_v.mean()) / se) if se > 0 else 0.0
        age = model.start_age + t
        print(f"  {t:>3}  {age:>4}  {n:>6}  {z_v.mean():>8.4f}  "
              f"{z_v.std():>7.4f}  {se:>8.5f}  {tstat:>8.2f}")
        if tstat > 3.0:
            all_tstat_ok = False
    test("Mean z not significantly != 0 (|t| < 3 at all ages)",
         all_tstat_ok)

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 6: z variance convergence toward unconditional")
    print("=" * 70)

    rho = model.rho
    p = model.pz
    mu1, s1 = model.mu_eta1, model.sigma_eta1
    mu2_eff = -(p / (1 - p)) * mu1
    s2 = model.sigma_eta2
    mu_eta = p * mu1 + (1 - p) * mu2_eff
    var_eta = (p * (s1**2 + (mu1 - mu_eta)**2)
               + (1 - p) * (s2**2 + (mu2_eff - mu_eta)**2))
    var_z_uncond = var_eta / (1 - rho**2)

    # Var(z_T) = var_eta * (1 - rho^{2T}) / (1 - rho^2)  starting from 0
    T = 44
    var_z_T_theory = var_eta * (1 - rho**(2*T)) / (1 - rho**2)
    std_z_T_theory = np.sqrt(var_z_T_theory)

    m44 = alive[:, T]
    std_z_T_sim = float(sim["z"][m44, T].std())

    print(f"  Unconditional std_z      = {np.sqrt(var_z_uncond):.4f}")
    print(f"  Theory std_z at t={T:>2}     = {std_z_T_theory:.4f}")
    print(f"  Simulated std_z at t={T:>2}  = {std_z_T_sim:.4f}")
    ratio = std_z_T_sim / std_z_T_theory
    test("Simulated std_z within 10% of theory at t=44",
         0.9 < ratio < 1.1, f"ratio = {ratio:.4f}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 7: z frozen in retirement")
    print("=" * 70)

    retire_t = model.retire_age - model.start_age

    mask_r = alive[:, retire_t] & alive[:, retire_t + 1]
    z_r0 = sim["z"][mask_r, retire_t]
    z_r1 = sim["z"][mask_r, retire_t + 1]
    test("z frozen at retirement (age 67 -> 68)",
         np.allclose(z_r0, z_r1, atol=1e-15),
         f"max diff = {np.max(np.abs(z_r0 - z_r1)):.2e}")

    zi_r0 = sim["z_idx"][mask_r, retire_t]
    zi_r1 = sim["z_idx"][mask_r, retire_t + 1]
    test("z_idx frozen at retirement", bool(np.all(zi_r0 == zi_r1)))

    all_frozen = True
    for dt in range(1, 10):
        t2 = retire_t + dt
        if t2 >= pc.n_age:
            break
        m = alive[:, retire_t] & alive[:, t2]
        if m.sum() == 0:
            continue
        if not np.allclose(sim["z"][m, retire_t], sim["z"][m, t2],
                           atol=1e-15):
            all_frozen = False
            break
    test("z frozen across first 10 retirement periods", all_frozen)

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 8: Income positivity and plausibility")
    print("=" * 70)

    income_alive = sim["income"][alive]
    test("All income > 0 for alive agents",
         bool(np.all(income_alive > 0)),
         f"min income = {income_alive.min():.6e}")

    # Spot-check working income in valid range
    t_spot = 20
    agent = 42
    if alive[agent, t_spot]:
        z_val = sim["z"][agent, t_spot]
        inc_sim = sim["income"][agent, t_spot]
        y_min = np.exp(pc.log_det_profile[t_spot] + z_val
                       + pc.eps_nodes.min())
        y_max = np.exp(pc.log_det_profile[t_spot] + z_val
                       + pc.eps_nodes.max())
        inc_min = _scalar_disposable_income(y_min)
        inc_max = _scalar_disposable_income(y_max)
        test("Spot-check: working income in valid range",
             inc_min - 1e-10 <= inc_sim <= inc_max + 1e-10,
             f"income={inc_sim:.6f}, range=[{inc_min:.6f}, {inc_max:.6f}]")

    # Retirement income matches scalar pension
    t_ret = retire_t + 5
    mask_r5 = alive[:, t_ret]
    z_ret = sim["z"][mask_r5, t_ret]
    inc_ret = sim["income"][mask_r5, t_ret]
    pension_exp = np.array(
        [_scalar_pension_after_tax(z, pc.avg_det) for z in z_ret]
    )
    test("Retirement income matches scalar pension formula",
         np.allclose(inc_ret, pension_exp, atol=1e-12),
         f"max diff = {np.max(np.abs(inc_ret - pension_exp)):.2e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 9: Policy interpolation sanity")
    print("=" * 70)

    c_alive = sim["c"][alive]
    x_alive = sim["x"][alive]
    pos_x = x_alive > 0
    test("Consumption > 0 when x > 0",
         bool(np.all(c_alive[pos_x] > 0)),
         f"min c (when x>0) = {c_alive[pos_x].min():.6e}")
    test("Consumption <= x (budget constraint)",
         bool(np.all(c_alive <= x_alive + 1e-10)),
         f"max violation = {(c_alive - x_alive).max():.2e}")

    alpha_s = sim["alpha_s"][alive]
    alpha_b = sim["alpha_b"][alive]
    test("alpha_s in [-tol, 1+tol] (constrained)",
         float(alpha_s.min()) >= -1e-8
         and float(alpha_s.max()) <= 1.0 + 1e-8,
         f"range = [{alpha_s.min():.6e}, {alpha_s.max():.6e}]")
    test("alpha_s + alpha_b <= 1+tol (constrained)",
         float((alpha_s + alpha_b).max()) <= 1.0 + 1e-8,
         f"max sum = {(alpha_s + alpha_b).max():.6e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 10: Mixture draw statistics (implied eta moments)")
    print("=" * 70)

    etas = []
    for t in range(0, retire_t - 1):
        m = alive[:, t] & alive[:, t + 1]
        if m.sum() == 0:
            continue
        z_t = sim["z"][m, t]
        z_t1 = sim["z"][m, t + 1]
        eta_implied = z_t1 - rho * z_t
        etas.append(eta_implied)

    etas_all = np.concatenate(etas)
    eta_mean = float(etas_all.mean())
    eta_var = float(etas_all.var())
    print(f"  Implied eta mean: {eta_mean:.6f} (should be ~0)")
    print(f"  Implied eta var:  {eta_var:.6f} (theory: {var_eta:.6f})")
    print(f"  N observations:   {len(etas_all):,}")

    # Clamping biases these slightly, so generous tolerance
    test("Implied E[eta] near 0", abs(eta_mean) < 0.01,
         f"mean = {eta_mean:.6f}")
    test("Implied Var[eta] within 15% of theory",
         abs(eta_var / var_eta - 1.0) < 0.15,
         f"ratio = {eta_var/var_eta:.4f}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 11: Income-dependent mortality (higher z -> longer life)")
    print("=" * 70)

    t_mid = 25  # age 47
    m_mid = alive[:, t_mid]
    z_mid = sim["z"][m_mid, t_mid]
    death_ages = sim["death_age"][m_mid]

    median_z = float(np.median(z_mid))
    lo_mask = z_mid < median_z
    hi_mask = z_mid >= median_z
    mean_death_lo = float(death_ages[lo_mask].mean())
    mean_death_hi = float(death_ages[hi_mask].mean())
    print(f"  Mean death age (low z):  {mean_death_lo:.1f}")
    print(f"  Mean death age (high z): {mean_death_hi:.1f}")
    test("Higher z -> longer life (income-dependent mortality)",
         mean_death_hi > mean_death_lo,
         f"diff = {mean_death_hi - mean_death_lo:.2f}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 12: Grid-point income consistency")
    print("=" * 70)

    sim_grid = simulate_lifecycle(
        C, S, B, pc, model, n_simulations=1000,
        initial_z=np.full(1000, 5, dtype=np.int32),
        seed=123, verbose=False,
    )

    z_t0 = sim_grid["z"][:, 0]
    test("z starts at exact grid value when init from index",
         bool(np.all(np.abs(z_t0 - pc.z_grid[5]) < 1e-15)),
         f"max deviation = {np.max(np.abs(z_t0 - pc.z_grid[5])):.2e}")

    inc_t0 = sim_grid["income"][:, 0]
    possible_incomes = pc.working_income[0, 5, :]
    matches = sum(
        1 for inc in inc_t0
        if np.any(np.abs(possible_incomes - inc) < 1e-10)
    )
    test("t=0 incomes match precomputed table at grid point z=0",
         matches == len(inc_t0),
         f"{matches}/{len(inc_t0)} matched")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"RESULTS: {n_pass} passed, {n_fail} failed "
          f"out of {n_pass + n_fail}")
    print("=" * 70)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
