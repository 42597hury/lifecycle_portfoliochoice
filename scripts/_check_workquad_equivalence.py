"""End-to-end equivalence check for the hoisting refactor in
compute_foc_jac_working_quad.

Procedure:
  1. Save a reference copy of the *pre-hoist* solver (built by reverting
     just the hoist edits in-place).
  2. Load both the pre-hoist module and the current (post-hoist) module
     side-by-side under different module names.
  3. Call compute_foc_jac_working_quad with identical synthetic inputs
     across many random seeds.
  4. Assert that all 6 outputs (foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum)
     match to bit identity.

This script is self-contained: it patches solver.py, JIT-compiles the
pre-hoist version, then restores the file before exiting.
"""
import importlib
import importlib.util
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = PROJECT_ROOT / "lifecycle" / "solver.py"
BACKUP_PATH = PROJECT_ROOT / "lifecycle" / "solver_pre_hoist_backup.py"
TMP_REF_PATH = PROJECT_ROOT / "lifecycle" / "_solver_pre_hoist_for_test.py"


def _make_pre_hoist_source(text: str) -> str:
    """Revert the hoist edits in `text` so the function matches the
    pre-edit dirty solver (which itself is bit-equivalent to HEAD logic
    in the inner loop)."""
    # Excise the precomputation block. It begins with the comment line
    # and ends just before the `for k_v` loop. After excision we leave
    # the original `base_det_z = ...` followed by a blank line and the
    # `for k_v` loop, matching the pre-hoist layout.
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
        raise RuntimeError("could not locate hoist inner loop block to revert")
    text = text.replace(inner_old, inner_new, 1)
    return text


def main():
    cur_text = SOLVER_PATH.read_text(encoding="utf-8")
    pre_text = _make_pre_hoist_source(cur_text)
    TMP_REF_PATH.write_text(pre_text, encoding="utf-8")
    try:
        # Import pre-hoist as a separate module name
        spec = importlib.util.spec_from_file_location(
            "lifecycle._solver_pre_hoist_for_test", TMP_REF_PATH
        )
        mod_pre = importlib.util.module_from_spec(spec)
        sys.modules["lifecycle._solver_pre_hoist_for_test"] = mod_pre
        spec.loader.exec_module(mod_pre)

        # And the current (post-hoist) version normally
        from lifecycle import solver as mod_post

        # Build synthetic but realistic inputs
        rng = np.random.default_rng(0)
        n_z, n_w = 7, 21
        n_state_quad = 3
        n_ret_quad = 3
        n_eta = 3
        n_eps = 3
        N0, N1, N2 = 4, 4, 4
        wealth_grid = np.linspace(0.5, 200.0, n_w)
        c_next_full = np.maximum(
            rng.standard_normal((n_z, N0 * N1 * N2, n_w)) * 5.0 + 30.0, 0.5
        )
        z_grid = np.linspace(-1.0, 1.0, n_z)
        dz = z_grid[1] - z_grid[0]
        eta_nodes = np.linspace(-0.4, 0.4, n_eta)
        eta_weights = np.ones(n_eta) / n_eta
        eps_nodes = np.linspace(-0.3, 0.3, n_eps)
        eps_weights = np.ones(n_eps) / n_eps
        v_nodes = rng.standard_normal((n_state_quad, 3)) * 0.05
        v_weights = np.ones(n_state_quad) / n_state_quad
        M_v_nodes = rng.standard_normal((n_state_quad, 3)) * 0.02
        Phi_0_state = np.array([0.05, 0.0, 0.02])
        Phi_11 = np.eye(3) * 0.95
        state_grid_i = np.array([0.1, -0.1, 0.05])
        state_bracket_shift = np.zeros(3)
        state_bracket_L_inv = np.eye(3)
        grids_0 = np.linspace(-1.0, 1.0, N0)
        grids_1 = np.linspace(-1.0, 1.0, N1)
        grids_2 = np.linspace(-1.0, 1.0, N2)
        ret_nodes = rng.standard_normal((n_ret_quad, 3)) * 0.1
        exp_ret_bill = np.exp(ret_nodes[:, 0])
        exp_ret_stock = np.exp(ret_nodes[:, 1])
        exp_ret_bond = np.exp(ret_nodes[:, 2])
        ret_weights = np.ones(n_ret_quad) / n_ret_quad
        base_mu_r_i = np.array([0.0, 0.04, 0.02])
        pension_next_by_z = 8.0 + 0.5 * z_grid

        common_kwargs = dict(
            wealth_grid=wealth_grid,
            c_next_full=c_next_full,
            log_det_next=0.05,
            annuity_factor_is=15.0,
            z_grid=z_grid,
            rho=0.97,
            eta_nodes=eta_nodes,
            eta_weights=eta_weights,
            dz=dz,
            v_nodes=v_nodes,
            v_weights=v_weights,
            M_v_nodes=M_v_nodes,
            base_mu_r_i=base_mu_r_i,
            Phi_0_state=Phi_0_state,
            Phi_11=Phi_11,
            state_grid_i=state_grid_i,
            state_bracket_shift=state_bracket_shift,
            state_bracket_L_inv=state_bracket_L_inv,
            grids_0=grids_0,
            grids_1=grids_1,
            grids_2=grids_2,
            N1=N1,
            N2=N2,
            exp_ret_bill=exp_ret_bill,
            exp_ret_stock=exp_ret_stock,
            exp_ret_bond=exp_ret_bond,
            ret_weights=ret_weights,
            eps_nodes=eps_nodes,
            eps_weights=eps_weights,
            gamma=3.0,
            psi=0.98,
            beta=0.96,
            b_bar=2.0,
        )

        max_diff_per_field = [0.0] * 6
        n_trials = 60

        for trial in range(n_trials):
            alpha_s = float(rng.uniform(0.0, 0.7))
            alpha_b = float(rng.uniform(0.0, 1.0 - alpha_s))
            s_val = float(rng.uniform(1.0, 100.0))
            z_idx = int(rng.integers(0, n_z))
            i_s = int(rng.integers(0, N0 * N1 * N2))
            use_pension = bool(rng.random() < 0.5)

            args = dict(common_kwargs)
            args.update(
                alpha_s=alpha_s,
                alpha_b=alpha_b,
                s_val=s_val,
                z_idx=z_idx,
                i_s=i_s,
                use_pension_next=use_pension,
                pension_next_by_z=pension_next_by_z,
            )

            out_pre = mod_pre.compute_foc_jac_working_quad(**args)
            out_post = mod_post.compute_foc_jac_working_quad(**args)

            for k in range(6):
                d = abs(out_pre[k] - out_post[k])
                if d > max_diff_per_field[k]:
                    max_diff_per_field[k] = d

        names = ("foc_s", "foc_b", "J_ss", "J_bb", "J_sb", "euler_sum")
        print(f"trials = {n_trials}")
        for nm, d in zip(names, max_diff_per_field):
            print(f"  max |pre - post|({nm}) = {d:.3e}")
        worst = max(max_diff_per_field)
        if worst == 0.0:
            print("OK: bit-identical across all trials")
        else:
            print(f"FAIL: nonzero diff (max {worst:.3e})")
            sys.exit(1)
    finally:
        try:
            TMP_REF_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
