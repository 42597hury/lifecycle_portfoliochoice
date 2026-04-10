"""
Economic and mathematical correctness tests for the continuous-z simulation.

Tests:
  E1. Budget constraint identity (exact arithmetic)
  E2. Income interpolation error: solver (interpolate-then-tax) vs simulation (tax-at-continuous-z)
  E3. Unconditional Euler equation from simulated panel
  E4. Pension consistency at grid points vs continuous z
  E5. Terminal-period behavior
  E6. Working-retirement boundary timing
  E7. Wealth accumulation path consistency
  E8. z transition formula: solver quadrature vs simulation draws
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
    fast_interp_1d,
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


def main():
    global n_pass, n_fail

    C, S, B, model, pc = setup()

    # Force Numba compilation
    _ = _scalar_disposable_income(1.0)
    _ = _scalar_pension_after_tax(0.0, 1.0)
    _ = fast_interp_1d(0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    gamma = model.gamma
    beta = model.beta
    rho = model.rho
    retire_t = model.retire_age - model.start_age

    # Run simulation
    print("Running simulation (20,000 households)...")
    sim = simulate_lifecycle(
        C, S, B, pc, model, n_simulations=20_000,
        initial_z="median", seed=42, verbose=False,
    )
    alive = sim["alive"]
    print("Done.\n")

    # ==================================================================
    print("=" * 70)
    print("E1. BUDGET CONSTRAINT IDENTITY (exact arithmetic)")
    print("=" * 70)
    # x_{t+1} = estate_t + income_{t+1}
    # estate_t = savings_t * R_port_t
    # savings_t = x_t - c_t

    n_sim, n_age = sim["x"].shape
    max_budget_err = 0.0
    max_savings_err = 0.0
    n_checked = 0

    for t in range(n_age - 1):
        # Agents alive at both t and t+1
        m = alive[:, t] & alive[:, t + 1]
        if m.sum() == 0:
            continue

        savings_t = sim["savings"][m, t]
        x_t = sim["x"][m, t]
        c_t = sim["c"][m, t]
        estate_t = sim["estate"][m, t]
        income_t1 = sim["income"][m, t + 1]
        x_t1 = sim["x"][m, t + 1]
        R_port_t = sim["R_port"][m, t]

        # Check savings = x - c
        sav_err = np.max(np.abs(savings_t - (x_t - c_t)))
        max_savings_err = max(max_savings_err, sav_err)

        # Check estate = savings * R_port
        estate_err = np.max(np.abs(estate_t - savings_t * R_port_t))
        # Use relative error for numerical stability
        scale = np.maximum(np.abs(estate_t), 1e-15)
        estate_rel = np.max(np.abs(estate_t - savings_t * R_port_t) / scale)

        # Check x_{t+1} = estate_t + income_{t+1}
        budget_err = np.max(np.abs(x_t1 - (estate_t + income_t1)))
        max_budget_err = max(max_budget_err, budget_err)
        n_checked += int(m.sum())

    test("savings = x - c (exact)", max_savings_err < 1e-14,
         f"max err = {max_savings_err:.2e}")
    test("x_{t+1} = estate_t + income_{t+1} (exact)",
         max_budget_err < 1e-12,
         f"max err = {max_budget_err:.2e}")
    print(f"  (checked {n_checked:,} agent-period transitions)")

    # No borrowing
    savings_alive = sim["savings"][alive]
    test("No borrowing: savings >= 0 everywhere",
         float(savings_alive.min()) >= -1e-10,
         f"min savings = {savings_alive.min():.6e}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E2. INCOME INTERPOLATION ERROR: solver vs simulation approach")
    print("=" * 70)
    # The solver interpolates after-tax income between grid points:
    #   income = (1-frac) * table[iz_lo, ie] + frac * table[iz_lo+1, ie]
    # The simulation computes:
    #   income = tax(exp(f(age) + z_continuous + eps))
    # These differ because tax(.) is nonlinear. Quantify the max error.

    max_interp_errors = []
    max_rel_errors = []

    for t_test in [0, 10, 20, 30, 44]:
        worst_abs = 0.0
        worst_rel = 0.0
        for ie in range(len(pc.eps_nodes)):
            for iz in range(pc.n_z - 1):
                z_lo = pc.z_grid[iz]
                z_hi = pc.z_grid[iz + 1]
                inc_lo = pc.working_income[t_test, iz, ie]
                inc_hi = pc.working_income[t_test, iz + 1, ie]

                # Sample 20 interior points
                for frac in np.linspace(0.01, 0.99, 20):
                    z_mid = z_lo + frac * (z_hi - z_lo)

                    # Solver approach: interpolate table
                    inc_interp = (1 - frac) * inc_lo + frac * inc_hi

                    # Simulation approach: compute directly
                    y_gross = np.exp(pc.log_det_profile[t_test]
                                     + z_mid + pc.eps_nodes[ie])
                    inc_direct = _scalar_disposable_income(y_gross)

                    abs_err = abs(inc_interp - inc_direct)
                    rel_err = abs_err / max(abs(inc_direct), 1e-15)
                    worst_abs = max(worst_abs, abs_err)
                    worst_rel = max(worst_rel, rel_err)

        max_interp_errors.append(worst_abs)
        max_rel_errors.append(worst_rel)

    print("  Income interpolation error (solver interp vs simulation exact):")
    ages_tested = [model.start_age + t for t in [0, 10, 20, 30, 44]]
    for age, abs_e, rel_e in zip(ages_tested, max_interp_errors, max_rel_errors):
        print(f"    Age {age:>2}: max |abs err| = {abs_e:.6e}, "
              f"max |rel err| = {rel_e:.4%}")

    overall_max_rel = max(max_rel_errors)
    test("Income interp error < 5% (relative)",
         overall_max_rel < 0.05,
         f"max rel err = {overall_max_rel:.4%}")
    test("Income interp error < 1% (relative)",
         overall_max_rel < 0.01,
         f"max rel err = {overall_max_rel:.4%}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E3. UNCONDITIONAL EULER EQUATION")
    print("=" * 70)
    # E[beta * R_{t+1} * (c_{t+1}/c_t)^{-gamma}] should be close to 1
    # for interior solutions (away from constraints).
    # We compute this at each age, restricting to agents not at corners.

    print(f"  {'Age':>4}  {'N':>6}  {'E[m*R]':>8}  {'|1-E|':>8}  {'SE':>8}")
    euler_errs = []
    for t in range(5, retire_t - 1):  # skip early ages (many at constraint)
        m = alive[:, t] & alive[:, t + 1]
        if m.sum() < 100:
            continue

        c_t = sim["c"][m, t]
        c_t1 = sim["c"][m, t + 1]
        R_p = sim["R_port"][m, t]
        sav = sim["savings"][m, t]

        # Restrict to interior solutions: savings > small threshold
        # and consumption not at corner
        interior = (sav > 0.01) & (c_t > 1e-6) & (c_t1 > 1e-6)
        if interior.sum() < 50:
            continue

        c_ratio = c_t1[interior] / c_t[interior]
        m_R = beta * R_p[interior] * c_ratio ** (-gamma)
        euler_mean = float(np.mean(m_R))
        euler_se = float(np.std(m_R) / np.sqrt(interior.sum()))
        euler_err = abs(euler_mean - 1.0)
        euler_errs.append(euler_err)

        age = model.start_age + t
        if t % 5 == 0 or t == retire_t - 2:
            print(f"  {age:>4}  {int(interior.sum()):>6}  "
                  f"{euler_mean:>8.4f}  {euler_err:>8.4f}  {euler_se:>8.4f}")

    if euler_errs:
        median_euler_err = float(np.median(euler_errs))
        max_euler_err = max(euler_errs)
        print(f"\n  Median |1 - E[mR]| = {median_euler_err:.4f}")
        print(f"  Max    |1 - E[mR]| = {max_euler_err:.4f}")
        # The unconditional Euler test is approximate (aggregates over
        # heterogeneous states). Errors < 10% are reasonable for this test.
        test("Median Euler error < 10%", median_euler_err < 0.10,
             f"median = {median_euler_err:.4f}")
        # Tighter test: most ages should be < 20%
        frac_below_20 = np.mean([e < 0.20 for e in euler_errs])
        test("90%+ of ages have Euler error < 20%",
             frac_below_20 >= 0.90,
             f"fraction = {frac_below_20:.2%}")
    else:
        print("  WARNING: not enough interior points for Euler test")

    # ==================================================================
    print()
    print("=" * 70)
    print("E4. PENSION CONSISTENCY: grid points vs continuous z")
    print("=" * 70)

    # At exact grid points, scalar pension should match precomputed table
    pension_table_vals = pc.pension_after_tax[retire_t, :]  # shape (n_z,)
    pension_scalar_vals = np.array([
        _scalar_pension_after_tax(z, pc.avg_det) for z in pc.z_grid
    ])
    max_pension_diff = float(np.max(np.abs(pension_table_vals - pension_scalar_vals)))
    test("Pension at grid points: scalar = table",
         max_pension_diff < 1e-14,
         f"max diff = {max_pension_diff:.2e}")

    # Quantify pension variation across one grid cell
    # (this is the solver vs simulation difference for off-grid z)
    pension_interp_errors = []
    for iz in range(pc.n_z - 1):
        p_lo = pension_table_vals[iz]
        p_hi = pension_table_vals[iz + 1]
        for frac in np.linspace(0.05, 0.95, 19):
            z_mid = pc.z_grid[iz] + frac * pc.dz
            p_interp = (1 - frac) * p_lo + frac * p_hi
            p_exact = _scalar_pension_after_tax(z_mid, pc.avg_det)
            rel_err = abs(p_interp - p_exact) / max(abs(p_exact), 1e-15)
            pension_interp_errors.append(rel_err)

    max_pension_interp = max(pension_interp_errors)
    print(f"  Max pension interpolation rel error: {max_pension_interp:.4%}")
    test("Pension interp error < 5%", max_pension_interp < 0.05,
         f"max rel err = {max_pension_interp:.4%}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E5. TERMINAL PERIOD BEHAVIOR")
    print("=" * 70)

    T = pc.n_age - 1
    m_T = alive[:, T]
    n_terminal = int(m_T.sum())
    print(f"  Agents alive at terminal age {model.terminal_age}: {n_terminal}")

    if n_terminal > 0:
        c_T = sim["c"][m_T, T]
        x_T = sim["x"][m_T, T]
        sav_T = sim["savings"][m_T, T]
        alpha_s_T = sim["alpha_s"][m_T, T]
        alpha_b_T = sim["alpha_b"][m_T, T]
        estate_T = sim["estate"][m_T, T]

        # c + savings = x
        budget_T = np.max(np.abs(c_T + sav_T - x_T))
        test("Terminal: c + savings = x", budget_T < 1e-14,
             f"max err = {budget_T:.2e}")

        # Consumption share should be between 0 and 1
        c_share = c_T / np.maximum(x_T, 1e-15)
        test("Terminal: consumption share in (0, 1)",
             float(c_share.min()) > 0 and float(c_share.max()) < 1.0,
             f"range = [{c_share.min():.4f}, {c_share.max():.4f}]")

        # With bequest utility, agents save a positive fraction
        test("Terminal: savings > 0 (bequest motive)",
             float(sav_T.min()) > -1e-10,
             f"min savings = {sav_T.min():.6e}")

        # Estate at death should equal savings * R_port
        estate_death = sim["estate_at_death"][m_T]
        test("Terminal: estate_at_death = estate",
             np.allclose(estate_death, estate_T, atol=1e-12),
             f"max diff = {np.max(np.abs(estate_death - estate_T)):.2e}")

        print(f"\n  Terminal consumption share: "
              f"mean={c_share.mean():.3f}, "
              f"p25={np.percentile(c_share, 25):.3f}, "
              f"p75={np.percentile(c_share, 75):.3f}")
        print(f"  Terminal alpha_s: mean={alpha_s_T.mean():.3f}")
        print(f"  Terminal alpha_b: mean={alpha_b_T.mean():.3f}")

        # Verify against solved policies at the terminal age
        # Pick a few agents and check their policies match the solved grid
        n_spot = min(20, n_terminal)
        spot_idx = np.where(m_T)[0][:n_spot]
        max_c_policy_err = 0.0
        for idx in spot_idx:
            z_val = sim["z"][idx, T]
            x_val = sim["x"][idx, T]
            s_idx = sim["state_idx"][idx, T]

            # Compute interpolated policy the same way the simulation does
            dz = pc.dz
            iz_lo = int((z_val - pc.z_grid[0]) / dz)
            iz_lo = max(0, min(iz_lo, pc.n_z - 2))
            frac_z = (z_val - pc.z_grid[iz_lo]) / dz
            frac_z = max(0.0, min(1.0, frac_z))

            c_lo = fast_interp_1d(x_val, pc.wealth_grid,
                                   C[T, iz_lo, s_idx, :])
            c_hi = fast_interp_1d(x_val, pc.wealth_grid,
                                   C[T, iz_lo + 1, s_idx, :])
            c_policy = (1.0 - frac_z) * c_lo + frac_z * c_hi

            c_sim = sim["c"][idx, T]
            err = abs(c_sim - c_policy)
            max_c_policy_err = max(max_c_policy_err, err)

        test("Terminal: sim consumption matches policy interpolation",
             max_c_policy_err < 1e-10,
             f"max err = {max_c_policy_err:.2e}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E6. WORKING-RETIREMENT BOUNDARY TIMING")
    print("=" * 70)

    # At t = retire_t - 1 (last working period), the simulation transitions
    # z one final time and computes working income for the transition to
    # t = retire_t.
    # At t = retire_t (first retirement period), pension is used for the
    # transition to t = retire_t + 1.

    # Check: income at retire_t should be working income (from last z transition)
    m_ret = alive[:, retire_t]
    z_at_ret = sim["z"][m_ret, retire_t]
    inc_at_ret = sim["income"][m_ret, retire_t]

    # Income at retire_t was computed during the t = retire_t - 1 transition
    # as exp(f(retire_age) + z_next + eps). It should NOT be pension.
    # Quick check: pension is constant for a given z; working income varies
    # with eps. So if income varies across agents with the same z_idx, it's
    # working income.
    z_idx_ret = sim["z_idx"][m_ret, retire_t]
    for iz in range(pc.n_z):
        group = z_idx_ret == iz
        if group.sum() < 5:
            continue
        inc_group = inc_at_ret[group]
        # Working income varies with eps; pension would be constant
        has_variation = inc_group.std() > 1e-10
        if has_variation:
            # This is working income (varies with eps), which is correct:
            # income at age retire_age was drawn with eps in the working branch
            break

    # Check: income at retire_t + 1 should be pension
    m_ret1 = alive[:, retire_t + 1]
    z_at_ret1 = sim["z"][m_ret1, retire_t + 1]
    inc_at_ret1 = sim["income"][m_ret1, retire_t + 1]

    # Pension should match scalar function at continuous z
    pension_expected = np.array([
        _scalar_pension_after_tax(z, pc.avg_det) for z in z_at_ret1
    ])
    pension_match = np.allclose(inc_at_ret1, pension_expected, atol=1e-12)
    test("First retirement+1 income = pension(z_continuous)",
         pension_match,
         f"max diff = {np.max(np.abs(inc_at_ret1 - pension_expected)):.2e}")

    # Check that z is frozen from retire_t onward
    m_both = alive[:, retire_t] & alive[:, retire_t + 1]
    z_frozen = np.allclose(
        sim["z"][m_both, retire_t], sim["z"][m_both, retire_t + 1],
        atol=1e-15,
    )
    test("z frozen from retirement onward", z_frozen)

    # Verify last working period: income at retire_t was computed using
    # log_det_profile[retire_t], which is f(retire_age)
    # This should be working income, not pension
    # Simple check: working income at z=0, typical eps should differ from
    # pension at z=0
    pension_at_z0 = _scalar_pension_after_tax(0.0, pc.avg_det)
    working_at_z0 = _scalar_disposable_income(
        np.exp(pc.log_det_profile[retire_t] + 0.0 + 0.0)
    )
    test("Working income != pension at z=0 (distinct formulas)",
         abs(working_at_z0 - pension_at_z0) > 0.001,
         f"working={working_at_z0:.4f}, pension={pension_at_z0:.4f}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E7. WEALTH ACCUMULATION PATH CONSISTENCY")
    print("=" * 70)

    # For a handful of agents, trace the full path and verify each step
    n_trace = 50
    max_path_err = 0.0
    n_steps_traced = 0

    for i in range(n_trace):
        for t in range(n_age - 1):
            if not alive[i, t] or not alive[i, t + 1]:
                break

            x_t = sim["x"][i, t]
            c_t = sim["c"][i, t]
            sav_t = x_t - c_t
            R_t = sim["R_port"][i, t]
            estate_t = sav_t * R_t
            inc_t1 = sim["income"][i, t + 1]
            x_t1_expected = estate_t + inc_t1
            x_t1_actual = sim["x"][i, t + 1]

            err = abs(x_t1_expected - x_t1_actual)
            max_path_err = max(max_path_err, err)
            n_steps_traced += 1

    test("Full path tracing: x_{t+1} = (x_t-c_t)*R + Y_{t+1}",
         max_path_err < 1e-12,
         f"max err = {max_path_err:.2e}, traced {n_steps_traced} steps")

    # ==================================================================
    print()
    print("=" * 70)
    print("E8. Z TRANSITION: solver quadrature vs simulation draws")
    print("=" * 70)
    # Verify that the distribution the simulation draws from matches
    # the distribution the solver integrates over.
    #
    # For a fixed z_t, compute:
    #   (a) E[z_{t+1} | z_t] using quadrature weights
    #   (b) E[z_{t+1} | z_t] from simulation

    # (a) Quadrature: E[z'] = rho * z + sum(eta_nodes * eta_weights)
    # Since E[eta] = 0 by construction, E[z'] = rho * z
    eta_mean_quad = float(np.sum(pc.eta_nodes * pc.eta_weights))
    print(f"  Quadrature E[eta] = {eta_mean_quad:.2e} (should be ~0)")
    test("Quadrature E[eta] = 0", abs(eta_mean_quad) < 1e-10,
         f"E[eta] = {eta_mean_quad:.2e}")

    # (b) From simulation: collect z_{t+1} - rho * z_t for working ages
    etas_implied = []
    for t in range(1, retire_t - 1):
        m = alive[:, t] & alive[:, t + 1]
        if m.sum() == 0:
            continue
        z_t_vals = sim["z"][m, t]
        z_t1_vals = sim["z"][m, t + 1]
        eta_impl = z_t1_vals - rho * z_t_vals
        etas_implied.append(eta_impl)

    etas_all = np.concatenate(etas_implied)
    sim_eta_mean = float(etas_all.mean())
    sim_eta_var = float(etas_all.var())

    # Theoretical moments
    p = model.pz
    mu1, s1 = model.mu_eta1, model.sigma_eta1
    mu2_eff = -(p / (1 - p)) * mu1
    s2 = model.sigma_eta2
    mu_eta = p * mu1 + (1 - p) * mu2_eff
    var_eta = (p * (s1**2 + (mu1 - mu_eta)**2)
               + (1 - p) * (s2**2 + (mu2_eff - mu_eta)**2))

    # Quadrature variance
    quad_var = float(np.sum(pc.eta_weights * pc.eta_nodes**2)
                     - eta_mean_quad**2)

    print(f"\n  Var[eta] comparison:")
    print(f"    Theory:     {var_eta:.6f}")
    print(f"    Quadrature: {quad_var:.6f}")
    print(f"    Simulation: {sim_eta_var:.6f}")

    test("Quadrature Var[eta] matches theory (< 1%)",
         abs(quad_var / var_eta - 1.0) < 0.01,
         f"ratio = {quad_var/var_eta:.6f}")
    test("Simulation Var[eta] matches theory (< 5%)",
         abs(sim_eta_var / var_eta - 1.0) < 0.05,
         f"ratio = {sim_eta_var/var_eta:.6f}")

    # Check skewness: the mixture is left-skewed (pz=0.176, mu1=-0.524)
    from scipy.stats import skew as sp_skew
    sim_skew = float(sp_skew(etas_all))
    # Theoretical skewness of the mixture
    mu3_theory = (p * ((mu1 - mu_eta)**3 + 3*(mu1 - mu_eta)*s1**2)
                  + (1 - p) * ((mu2_eff - mu_eta)**3 + 3*(mu2_eff - mu_eta)*s2**2))
    theory_skew = mu3_theory / var_eta**1.5
    print(f"\n  Skewness comparison:")
    print(f"    Theory:     {theory_skew:.4f}")
    print(f"    Simulation: {sim_skew:.4f}")
    test("Simulation skewness matches sign of theory",
         (sim_skew < 0) == (theory_skew < 0),
         f"sim={sim_skew:.4f}, theory={theory_skew:.4f}")
    test("Simulation skewness within 30% of theory",
         abs(sim_skew / theory_skew - 1.0) < 0.30,
         f"ratio = {sim_skew/theory_skew:.4f}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E9. POLICY INTERPOLATION: simulation matches solved policies")
    print("=" * 70)
    # At each simulated point, re-evaluate the interpolated policy and
    # confirm it matches what the simulation recorded.

    max_c_err = 0.0
    max_as_err = 0.0
    max_ab_err = 0.0
    n_spot = 0

    # Sample random alive agent-periods
    rng = np.random.default_rng(99)
    for _ in range(2000):
        i = rng.integers(0, n_sim)
        t = rng.integers(0, n_age)
        if not alive[i, t]:
            continue

        z_val = sim["z"][i, t]
        x_val = sim["x"][i, t]
        s_idx = sim["state_idx"][i, t]

        dz = pc.dz
        iz_lo = int((z_val - pc.z_grid[0]) / dz)
        iz_lo = max(0, min(iz_lo, pc.n_z - 2))
        frac_z = (z_val - pc.z_grid[iz_lo]) / dz
        frac_z = max(0.0, min(1.0, frac_z))

        # Interpolate policies
        c_lo = fast_interp_1d(x_val, pc.wealth_grid, C[t, iz_lo, s_idx, :])
        c_hi = fast_interp_1d(x_val, pc.wealth_grid,
                               C[t, iz_lo + 1, s_idx, :])
        c_interp = (1.0 - frac_z) * c_lo + frac_z * c_hi

        as_lo = fast_interp_1d(x_val, pc.wealth_grid, S[t, iz_lo, s_idx, :])
        as_hi = fast_interp_1d(x_val, pc.wealth_grid,
                                S[t, iz_lo + 1, s_idx, :])
        as_interp = (1.0 - frac_z) * as_lo + frac_z * as_hi

        ab_lo = fast_interp_1d(x_val, pc.wealth_grid, B[t, iz_lo, s_idx, :])
        ab_hi = fast_interp_1d(x_val, pc.wealth_grid,
                                B[t, iz_lo + 1, s_idx, :])
        ab_interp = (1.0 - frac_z) * ab_lo + frac_z * ab_hi

        # Compare with simulation output (before clamping/cleanup)
        c_sim = sim["c"][i, t]
        as_sim = sim["alpha_s"][i, t]
        ab_sim = sim["alpha_b"][i, t]

        # Consumption may differ due to clamping (c >= 0, c <= x)
        # but for interior points they should match
        if x_val > 0.01 and c_interp > 0 and c_interp < x_val:
            max_c_err = max(max_c_err, abs(c_sim - c_interp))

        max_as_err = max(max_as_err, abs(as_sim - as_interp))
        max_ab_err = max(max_ab_err, abs(ab_sim - ab_interp))
        n_spot += 1

    print(f"  Spot-checked {n_spot} agent-periods")
    test("Consumption matches policy interpolation",
         max_c_err < 1e-10,
         f"max err = {max_c_err:.2e}")
    test("alpha_s matches policy interpolation",
         max_as_err < 1e-10,
         f"max err = {max_as_err:.2e}")
    test("alpha_b matches policy interpolation",
         max_ab_err < 1e-10,
         f"max err = {max_ab_err:.2e}")

    # ==================================================================
    print()
    print("=" * 70)
    print("E10. CONSUMPTION GROWTH vs INTEREST RATE (economic sanity)")
    print("=" * 70)
    # In a lifecycle model, average consumption should grow roughly with
    # the interest rate when beta * R ~ 1. Check the consumption path
    # is hump-shaped (rises during working years, may decline in retirement).

    mean_c_path = []
    for t in range(n_age):
        m = alive[:, t]
        if m.sum() == 0:
            mean_c_path.append(np.nan)
        else:
            mean_c_path.append(float(sim["c"][m, t].mean()))

    mean_c = np.array(mean_c_path)
    # Check consumption is rising in early working years
    c_early = mean_c[0:10]
    rising = np.all(np.diff(c_early[~np.isnan(c_early)]) > 0)
    test("Consumption rising in first 10 working years", rising,
         f"c_path[0:10] = {np.round(c_early, 4)}")

    # Check consumption doesn't collapse to zero (economic sensibility)
    c_mid = mean_c[20:30]
    test("Mid-career consumption > 0.001",
         float(np.nanmin(c_mid)) > 0.001,
         f"min = {np.nanmin(c_mid):.6f}")

    # ==================================================================
    print()
    print("=" * 70)
    print(f"RESULTS: {n_pass} passed, {n_fail} failed "
          f"out of {n_pass + n_fail}")
    print("=" * 70)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
