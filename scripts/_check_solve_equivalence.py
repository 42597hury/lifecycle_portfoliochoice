"""End-to-end solve equivalence check.

Runs a small partial solve that exercises both retirement-quad and
working-quad code paths. Compares the resulting consumption / savings
policies between the post-hoist solver and a synthesized pre-hoist
copy. The pre-hoist copy is built by string-reverting the hoist edits
in `lifecycle/solver.py` into a separate module file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = PROJECT_ROOT / "lifecycle" / "solver.py"
TMP_REF_PATH = PROJECT_ROOT / "lifecycle" / "_solver_pre_hoist_for_test.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_pre_hoist_source(text: str) -> str:
    marker = "    # --- Hoist (k_eta, i_e)-only quantities out of the (k_v, k_r) loop ---"
    end_marker = "    for k_v in range(n_state_quad):"
    i = text.index(marker)
    j = text.index(end_marker, i)
    text = text[:i] + text[j:]

    inner_old = (
        "            # Alive contribution: quadrature over persistent and transitory innovations\n"
        "            for k_eta in range(n_eta):\n"
        "                w_eta = eta_weights[k_eta]\n"
        "                if w_eta < prob_skip:\n"
        "                    continue\n"
        "\n"
        "                # Lookup hoisted (k_eta)-only quantities\n"
        "                iz_lo = eta_iz_lo[k_eta]\n"
        "                frac_z = eta_frac_z[k_eta]\n"
        "                use_cubic = eta_use_cubic[k_eta] != 0\n"
        "                h00 = eta_h00[k_eta]\n"
        "                h10 = eta_h10[k_eta]\n"
        "                h01 = eta_h01[k_eta]\n"
        "                h11 = eta_h11[k_eta]\n"
        "\n"
        "                p_out_base = p_state_ret * w_eta\n"
        "\n"
        "                for i_e in range(n_eps):\n"
        "                    weight = p_out_base * eps_weights[i_e]\n"
        "\n"
        "                    income_next = income_table[k_eta, i_e]\n"
        "                    x_next = w_inv + income_next\n"
        "\n"
        "                    iw, frac_w, inv_dw = find_bracket(x_next, wealth_grid)\n"
        "\n"
        "                    # Trilinear blend of z-wealth interpolated values at 8 corners\n"
        "                    c000, mpc000 = _interp_z_wealth_pre(c_next_full, j000, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c001, mpc001 = _interp_z_wealth_pre(c_next_full, j001, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c010, mpc010 = _interp_z_wealth_pre(c_next_full, j010, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c011, mpc011 = _interp_z_wealth_pre(c_next_full, j011, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c100, mpc100 = _interp_z_wealth_pre(c_next_full, j100, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c101, mpc101 = _interp_z_wealth_pre(c_next_full, j101, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c110, mpc110 = _interp_z_wealth_pre(c_next_full, j110, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)\n"
        "                    c111, mpc111 = _interp_z_wealth_pre(c_next_full, j111, iz_lo, frac_z, h00, h10, h01, h11, iw, frac_w, inv_dw, use_cubic, min_consumption)"
    )

    inner_new = (
        "            # Alive contribution: quadrature over persistent and transitory innovations\n"
        "            for k_eta in range(n_eta):\n"
        "                w_eta = eta_weights[k_eta]\n"
        "                if w_eta < prob_skip:\n"
        "                    continue\n"
        "\n"
        "                z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]\n"
        "\n"
        "                iz_lo = int((z_next - z_grid[0]) / dz)\n"
        "                iz_lo = max(0, min(iz_lo, n_z - 2))\n"
        "                frac_z = (z_next - z_grid[iz_lo]) / dz\n"
        "                frac_z = max(0.0, min(1.0, frac_z))\n"
        "\n"
        "                use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)\n"
        "\n"
        "                p_out_base = p_state_ret * w_eta\n"
        "                det_z_eta = base_det_z * exp_eta[k_eta]\n"
        "\n"
        "                if use_pension_next:\n"
        "                    income_next_const = (1.0 - frac_z) * pension_next_by_z[iz_lo] + frac_z * pension_next_by_z[iz_lo + 1]\n"
        "                else:\n"
        "                    income_next_const = 0.0\n"
        "\n"
        "                for i_e in range(n_eps):\n"
        "                    weight = p_out_base * eps_weights[i_e]\n"
        "\n"
        "                    if use_pension_next:\n"
        "                        income_next = income_next_const\n"
        "                    else:\n"
        "                        y_gross_next = det_z_eta * exp_eps[i_e]\n"
        "                        income_next = scalar_disposable_income(y_gross_next)\n"
        "                    x_next = w_inv + income_next\n"
        "\n"
        "                    iw, frac_w, inv_dw = find_bracket(x_next, wealth_grid)\n"
        "\n"
        "                    c000, mpc000 = _interp_z_wealth(c_next_full, j000, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c001, mpc001 = _interp_z_wealth(c_next_full, j001, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c010, mpc010 = _interp_z_wealth(c_next_full, j010, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c011, mpc011 = _interp_z_wealth(c_next_full, j011, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c100, mpc100 = _interp_z_wealth(c_next_full, j100, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c101, mpc101 = _interp_z_wealth(c_next_full, j101, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c110, mpc110 = _interp_z_wealth(c_next_full, j110, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)\n"
        "                    c111, mpc111 = _interp_z_wealth(c_next_full, j111, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)"
    )

    if inner_old not in text:
        raise RuntimeError("could not locate inner-loop hoist block to revert")
    return text.replace(inner_old, inner_new, 1)


def _build_small_problem():
    from lifecycle.model import DiscretizationConfig
    from lifecycle.precompute import Precompute, build_model
    from lifecycle.var import build_nominal_system1_var_config

    base_cfg = {
        "beta": 0.96, "gamma": 5.0, "b_bar": 10,
        "start_age": 22, "retire_age": 67, "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991, "pz": 0.176,
        "mu_eta1": -0.524, "sigma_eta1": 0.113,
        "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524),
        "sigma_eta2": 0.046, "pe": 0.044,
        "mu_eps1": 0.134, "sigma_eps1": 0.762,
        "mu_eps2": 0.0, "sigma_eps2": 0.055,
        "constrained": True,
    }
    var_config, _, _ = build_nominal_system1_var_config(
        csv_path=str(PROJECT_ROOT / "data" / "var_dataset.csv")
    )
    model = build_model(base_cfg, var_config, verbose=False)
    disc = DiscretizationConfig(
        n_wealth=12, n_savings=12,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=2.0,
        n_z=3, n_stds=2.0,
        n_eps_nodes=2, n_eta_nodes=2,
        n_ret_nodes_1d=2, n_state_quad_nodes=2,
    )
    pc = Precompute(model, disc, verbose=False)
    return model, pc


def _solve_with(solver_module, model, pc):
    from lifecycle.model import SolveControl, SolverConfig
    sc = SolveControl(
        youngest_age_to_solve=64,  # exercises both retirement and working-quad
        checkpoint_path=None,
        checkpoint_every_n_ages=1,
        save_on_interrupt=False,
        return_partial_on_interrupt=False,
    )
    C, S, B, diag = solver_module.run_lifecycle_solver(
        model, pc,
        solver_config=SolverConfig(),
        solve_control=sc,
        verbose=0,
    )
    return C, S, B, diag


def main():
    cur = SOLVER_PATH.read_text(encoding="utf-8")
    pre = _make_pre_hoist_source(cur)
    TMP_REF_PATH.write_text(pre, encoding="utf-8")
    try:
        # Load both modules
        from lifecycle import solver as mod_post

        spec = importlib.util.spec_from_file_location(
            "lifecycle._solver_pre_hoist_for_test", TMP_REF_PATH
        )
        mod_pre = importlib.util.module_from_spec(spec)
        sys.modules["lifecycle._solver_pre_hoist_for_test"] = mod_pre
        spec.loader.exec_module(mod_pre)

        # Reuse the same model/pc to keep inputs identical
        model, pc = _build_small_problem()

        print("Solving with POST-hoist solver...")
        C_post, S_post, B_post, _ = _solve_with(mod_post, model, pc)
        print("Solving with PRE-hoist solver...")
        C_pre, S_pre, B_pre, _ = _solve_with(mod_pre, model, pc)

        # Compare on the solved slice (last several ages from age 64)
        mask = np.isfinite(C_pre) & np.isfinite(C_post)
        n_solved = int(mask.sum())
        print(f"\nSolved cells (per matrix): {n_solved}")

        for name, A, B in [("C", C_pre, C_post), ("S", S_pre, S_post), ("B", B_pre, B_post)]:
            d = np.abs(A[mask] - B[mask])
            denom = np.maximum(np.abs(A[mask]), 1e-12)
            rel = d / denom
            print(f"  {name}: max abs diff = {d.max():.3e},  max rel diff = {rel.max():.3e}")

        # Tight tolerance: machine epsilon scaled
        worst_rel = max(
            (np.abs(C_pre[mask] - C_post[mask]) / np.maximum(np.abs(C_pre[mask]), 1e-12)).max(),
            (np.abs(S_pre[mask] - S_post[mask]) / np.maximum(np.abs(S_pre[mask]), 1e-12)).max(),
        )
        if worst_rel < 1e-12:
            print("\nOK: end-to-end solve matches to <1e-12 relative")
        elif worst_rel < 1e-8:
            print(f"\nOK: end-to-end solve matches to {worst_rel:.1e} relative (within fastmath tolerance)")
        else:
            print(f"\nFAIL: relative diff {worst_rel:.3e} exceeds tolerance")
            sys.exit(1)
    finally:
        try:
            TMP_REF_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
