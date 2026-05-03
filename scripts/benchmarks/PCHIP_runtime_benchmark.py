"""
Standalone runtime benchmark for the live true-PCHIP z interpolator.

This script keeps `solver.py` untouched. It:

1. Loads the saved retirement-only bundle.
2. Builds an in-memory copy of `solver.py` with `_interp_z_wealth`
   replaced by the previous Catmull-Rom cubic.
3. Warms both solvers on a tiny reduced problem to exclude JIT compile time.
4. Times a short working-age block just below retirement using the saved
   age-67 retirement policy as continuation input.

Why not benchmark a retirement-only partial solve?
Retirement FOCs never call `_interp_z_wealth`, so a retirement-only solve would
not measure Catmull-Rom versus PCHIP at all. The first working age (age 66 in
the current model) is the smallest live solve that exercises the interpolation
change of interest.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import Precompute, build_model
import solver as solver_live


DEFAULT_RUN_DIR = Path("saved_runs/unconstrained_principal_grid9x9x9_nz9_retirement_only")
DEFAULT_MODEL_REF_RUN_DIR = Path("saved_runs/unconstrained_principal_grid5x5x5_nz9")


def _decode_var_config(raw_var_config):
    out = {}
    for key, value in raw_var_config.items():
        if isinstance(value, dict) and value.get("kind") == "ndarray":
            out[key] = np.asarray(value["values"], dtype=value["dtype"])
        else:
            out[key] = value
    return out


def _load_bundle(run_dir: Path, model_ref_run_dir: Path | None = None):
    meta = json.loads((run_dir / "metadata.json").read_text())
    arrays = np.load(run_dir / "policy_arrays.npz")

    run_cfg = meta.get("run_config")
    if run_cfg is not None:
        base_config = run_cfg["base_config"]
        disc_cfg_raw = run_cfg["discretization_config"]
        solver_cfg_raw = run_cfg["solver_config"]
        var_config = _decode_var_config(run_cfg["var_config"])
    else:
        if model_ref_run_dir is None:
            raise ValueError(
                "Bundle metadata has no run_config; pass a reference run dir "
                "with full model settings."
            )
        ref_meta = json.loads((model_ref_run_dir / "metadata.json").read_text())
        ref_run_cfg = ref_meta["run_config"]
        base_config = ref_run_cfg["base_config"]
        var_config = _decode_var_config(ref_run_cfg["var_config"])

        diag_path = run_dir / "diagnostics.pkl"
        import pickle
        diag = pickle.loads(diag_path.read_bytes())
        disc_cfg_obj = diag["disc_config"]
        solver_cfg_obj = diag["solver_config"]

        disc_cfg_raw = {
            "n_wealth": disc_cfg_obj.n_wealth,
            "wealth_min": disc_cfg_obj.wealth_min,
            "wealth_max": disc_cfg_obj.wealth_max,
            "n_savings": disc_cfg_obj.n_savings,
            "savings_min": disc_cfg_obj.savings_min,
            "savings_max": disc_cfg_obj.savings_max,
            "state_grid_sizes": tuple(disc_cfg_obj.state_grid_sizes),
            "state_grid_mode": disc_cfg_obj.state_grid_mode,
            "state_n_stds": disc_cfg_obj.state_n_stds,
            "n_z": disc_cfg_obj.n_z,
            "n_stds": disc_cfg_obj.n_stds,
            "n_eps_nodes": disc_cfg_obj.n_eps_nodes,
            "n_eta_nodes": disc_cfg_obj.n_eta_nodes,
            "n_ret_nodes_1d": tuple(disc_cfg_obj.n_ret_nodes_1d),
            "n_state_quad_nodes": disc_cfg_obj.n_state_quad_nodes,
        }
        solver_cfg_raw = solver_cfg_obj._asdict()

    disc_defaults = DiscretizationConfig()._asdict()
    disc_defaults.update(disc_cfg_raw)
    if isinstance(disc_defaults["state_grid_sizes"], list):
        disc_defaults["state_grid_sizes"] = tuple(disc_defaults["state_grid_sizes"])
    if isinstance(disc_defaults["n_ret_nodes_1d"], list):
        disc_defaults["n_ret_nodes_1d"] = tuple(disc_defaults["n_ret_nodes_1d"])

    solver_defaults = SolverConfig()._asdict()
    solver_defaults.update(solver_cfg_raw)

    model = build_model(base_config, var_config, verbose=False)
    disc_config = DiscretizationConfig(**disc_defaults)
    pc = Precompute(model, disc_config, verbose=False)
    solver_config = SolverConfig(**solver_defaults)
    return meta, arrays, model, pc, solver_config


def _solver_source_with_catmull_rom() -> str:
    src_path = Path("solver.py")
    src = src_path.read_text(encoding="utf-8")

    start_token = "@njit(fastmath=True)\ndef _interp_z_wealth("
    end_token = (
        "\n\n# =============================================================================\n"
        "# FOC AND JACOBIAN -- RETIREMENT (QUADRATURE)\n"
        "# =============================================================================\n"
    )
    start_idx = src.index(start_token)
    end_idx = src.index(end_token)

    replacement = """
@njit(fastmath=True)
def _interp_z_wealth(c_next_full, j_s, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_c):
    \"\"\"Catmull-Rom z interpolation with linear fallback on boundary intervals.

    This reproduces the pre-PCHIP working-age interpolation used as the
    runtime baseline for the benchmark. `frac_w` is the affine wealth
    coordinate on the selected segment and may lie outside [0, 1] for
    linear extrapolation.
    \"\"\"
    if use_cubic:
        c_zm1 = (1.0 - frac_w) * c_next_full[iz_lo - 1, j_s, iw] + frac_w * c_next_full[iz_lo - 1, j_s, iw + 1]
        c_z0  = (1.0 - frac_w) * c_next_full[iz_lo,     j_s, iw] + frac_w * c_next_full[iz_lo,     j_s, iw + 1]
        c_z1  = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]
        c_z2  = (1.0 - frac_w) * c_next_full[iz_lo + 2, j_s, iw] + frac_w * c_next_full[iz_lo + 2, j_s, iw + 1]
        f = frac_z
        f2 = f * f
        f3 = f2 * f
        c_val = (
            c_z0
            + 0.5 * f * (-c_zm1 + c_z1)
            + 0.5 * f2 * (2.0 * c_zm1 - 5.0 * c_z0 + 4.0 * c_z1 - c_z2)
            + 0.5 * f3 * (-c_zm1 + 3.0 * c_z0 - 3.0 * c_z1 + c_z2)
        )
        c_val = max(c_val, min_c)

        mpc_zm1 = (c_next_full[iz_lo - 1, j_s, iw + 1] - c_next_full[iz_lo - 1, j_s, iw]) * inv_dw
        mpc_z0  = (c_next_full[iz_lo,     j_s, iw + 1] - c_next_full[iz_lo,     j_s, iw]) * inv_dw
        mpc_z1  = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw
        mpc_z2  = (c_next_full[iz_lo + 2, j_s, iw + 1] - c_next_full[iz_lo + 2, j_s, iw]) * inv_dw
        mpc_val = (
            mpc_z0
            + 0.5 * f * (-mpc_zm1 + mpc_z1)
            + 0.5 * f2 * (2.0 * mpc_zm1 - 5.0 * mpc_z0 + 4.0 * mpc_z1 - mpc_z2)
            + 0.5 * f3 * (-mpc_zm1 + 3.0 * mpc_z0 - 3.0 * mpc_z1 + mpc_z2)
        )
        mpc_val = max(0.0, min(1.0, mpc_val))
    else:
        c_lo = (1.0 - frac_w) * c_next_full[iz_lo,     j_s, iw] + frac_w * c_next_full[iz_lo,     j_s, iw + 1]
        c_hi = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]
        c_val = (1.0 - frac_z) * c_lo + frac_z * c_hi
        c_val = max(c_val, min_c)

        mpc_lo = (c_next_full[iz_lo,     j_s, iw + 1] - c_next_full[iz_lo,     j_s, iw]) * inv_dw
        mpc_hi = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw
        mpc_val = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi
        mpc_val = max(0.0, min(1.0, mpc_val))

    return c_val, mpc_val
"""
    return src[:start_idx] + replacement + src[end_idx:]


def _build_catmull_solver_module():
    module = types.ModuleType("solver_catmull_runtime")
    module.__file__ = str(Path("solver.py").resolve())
    patched = _solver_source_with_catmull_rom()
    exec(compile(patched, "solver_catmull_runtime.py", "exec"), module.__dict__)
    return module


def _age_to_index(pc: Precompute, age: int) -> int:
    loc = np.where(pc.ages == age)[0]
    if loc.size != 1:
        raise ValueError(f"Age {age} not found in pc.ages")
    return int(loc[0])


def _benchmark_args_for_age(model, pc, solver_config, c_next_full, age: int):
    t = _age_to_index(pc, age)
    t_next = t + 1
    use_pension_next = age == model.retire_age - 1
    pension_next_by_z = (
        pc.pension_after_tax[t_next, :].astype(np.float64)
        if use_pension_next
        else np.zeros(pc.n_z, dtype=np.float64)
    )
    return dict(
        wealth_grid=pc.wealth_grid,
        savings_grid=pc.s_grid,
        z_grid=pc.z_grid,
        N_state=pc.N_state,
        c_next_full=c_next_full,
        log_det_next=pc.log_det_profile[t_next],
        annuity_factors=pc.annuity_factors,
        rho=model.rho,
        eta_nodes=pc.eta_nodes,
        eta_weights=pc.eta_weights,
        dz=pc.dz,
        state_grid=pc.state_grid,
        grids_0=pc.state_bracket_grids[0],
        grids_1=pc.state_bracket_grids[1],
        grids_2=pc.state_bracket_grids[2],
        state_bracket_shift=pc.state_bracket_shift,
        state_bracket_L_inv=pc.state_bracket_L_inv,
        v_nodes=pc.v_nodes,
        v_weights=pc.v_weights,
        M_v_nodes=pc.M_v_nodes,
        const_r=pc.const_r,
        A_r=pc.A_r,
        Phi_0_state=model.Phi_0_state,
        Phi_11=model.Phi_11,
        exp_ret_bill=pc.exp_ret_bill,
        exp_ret_stock=pc.exp_ret_stock,
        exp_ret_bond=pc.exp_ret_bond,
        ret_weights=pc.ret_weights,
        eps_nodes=pc.eps_nodes,
        eps_weights=pc.eps_weights,
        gamma=model.gamma,
        psi_vec=pc.survival_probs_2d[t, :],
        beta=model.beta,
        b_bar=model.b_bar,
        use_pension_next=use_pension_next,
        pension_next_by_z=pension_next_by_z,
        constrained=model.constrained,
        solver_config=solver_config,
        out_c=np.empty((pc.n_z, pc.N_state, pc.wealth_grid.size)),
        out_s=np.empty((pc.n_z, pc.N_state, pc.wealth_grid.size)),
        out_b=np.empty((pc.n_z, pc.N_state, pc.wealth_grid.size)),
    )


def _warmup_args(model, pc, solver_config, c_next_full, age: int):
    t = _age_to_index(pc, age)
    t_next = t + 1
    n_z_small = min(4, pc.n_z)
    n_w_small = min(4, pc.wealth_grid.size)
    pension_next_small = (
        pc.pension_after_tax[t_next, :n_z_small].astype(np.float64)
        if age == model.retire_age - 1
        else np.zeros(n_z_small, dtype=np.float64)
    )
    return dict(
        wealth_grid=pc.wealth_grid[:n_w_small],
        savings_grid=pc.s_grid[:1],
        z_grid=pc.z_grid[:n_z_small],
        N_state=pc.N_state,
        c_next_full=c_next_full[:n_z_small, :, :n_w_small],
        log_det_next=pc.log_det_profile[t_next],
        annuity_factors=pc.annuity_factors,
        rho=model.rho,
        eta_nodes=pc.eta_nodes[:1],
        eta_weights=np.array([1.0], dtype=np.float64),
        dz=pc.dz,
        state_grid=pc.state_grid,
        grids_0=pc.state_bracket_grids[0],
        grids_1=pc.state_bracket_grids[1],
        grids_2=pc.state_bracket_grids[2],
        state_bracket_shift=pc.state_bracket_shift,
        state_bracket_L_inv=pc.state_bracket_L_inv,
        v_nodes=pc.v_nodes[:1],
        v_weights=np.array([1.0], dtype=np.float64),
        M_v_nodes=pc.M_v_nodes[:1, :],
        const_r=pc.const_r,
        A_r=pc.A_r,
        Phi_0_state=model.Phi_0_state,
        Phi_11=model.Phi_11,
        exp_ret_bill=pc.exp_ret_bill[:1],
        exp_ret_stock=pc.exp_ret_stock[:1],
        exp_ret_bond=pc.exp_ret_bond[:1],
        ret_weights=np.array([1.0], dtype=np.float64),
        eps_nodes=pc.eps_nodes[:1],
        eps_weights=np.array([1.0], dtype=np.float64),
        gamma=model.gamma,
        psi_vec=pc.survival_probs_2d[t, :n_z_small],
        beta=model.beta,
        b_bar=model.b_bar,
        use_pension_next=age == model.retire_age - 1,
        pension_next_by_z=pension_next_small,
        constrained=model.constrained,
        solver_config=solver_config,
        out_c=np.empty((n_z_small, pc.N_state, n_w_small)),
        out_s=np.empty((n_z_small, pc.N_state, n_w_small)),
        out_b=np.empty((n_z_small, pc.N_state, n_w_small)),
    )


def _aggregate_diag(di, df):
    total_calls = int(di[:, solver_live.DI_TOTAL_CALLS].sum())
    total_fail = int(di[:, solver_live.DI_NEWTON_FAIL].sum())
    total_iter = int(di[:, solver_live.DI_SUM_ITER].sum())
    max_iter = int(df[:, solver_live.DF_MAX_NEWTON_ITER].max())
    max_foc = float(df[:, solver_live.DF_MAX_FOC_RESID].max())
    return {
        "total_calls": total_calls,
        "total_fail": total_fail,
        "avg_iter": total_iter / max(total_calls, 1),
        "max_iter": max_iter,
        "max_foc": max_foc,
    }


def _solve_working_block(mod, model, pc, solver_config, c_next_start, ages_to_solve):
    c_next = c_next_start
    policies = {}
    diagnostics = {}
    per_age_time = {}
    t0 = time.perf_counter()
    for age in ages_to_solve:
        kwargs = _benchmark_args_for_age(model, pc, solver_config, c_next, age)
        age_t0 = time.perf_counter()
        out_c, out_s, out_b, di, df = mod.solve_working_age_step_quad(**kwargs)
        per_age_time[age] = time.perf_counter() - age_t0
        policies[age] = (out_c.copy(), out_s.copy(), out_b.copy())
        diagnostics[age] = _aggregate_diag(di, df)
        c_next = out_c
    total = time.perf_counter() - t0
    return {"total_sec": total, "per_age_sec": per_age_time, "policies": policies, "diag": diagnostics}


def _print_diag(label, result):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  Total wall time: {result['total_sec']:.2f}s")
    for age, secs in result["per_age_sec"].items():
        diag = result["diag"][age]
        print(
            f"  Age {age}: {secs:.2f}s | calls={diag['total_calls']:,} "
            f"| fail={diag['total_fail']:,} | avg_iter={diag['avg_iter']:.3f} "
            f"| max_iter={diag['max_iter']} | max_foc={diag['max_foc']:.2e}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Catmull-Rom vs PCHIP working-age runtime.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-ref-run-dir", type=Path, default=DEFAULT_MODEL_REF_RUN_DIR,
                        help="Reference run with full run_config metadata. Used only if run-dir metadata lacks it.")
    parser.add_argument("--start-age", type=int, default=None,
                        help="Oldest working age to solve. Default: retire_age - 1.")
    parser.add_argument("--n-work-ages", type=int, default=1,
                        help="Number of working ages to solve backward from start-age.")
    args = parser.parse_args()

    meta, arrays, model, pc, solver_config = _load_bundle(args.run_dir, args.model_ref_run_dir)
    start_age = model.retire_age - 1 if args.start_age is None else int(args.start_age)
    if start_age >= model.retire_age:
        raise ValueError("start-age must be a working age below retire_age.")
    if start_age < model.start_age:
        raise ValueError("start-age must be within model ages.")

    ages_to_solve = []
    for age in range(start_age, start_age - args.n_work_ages, -1):
        if age < model.start_age:
            break
        ages_to_solve.append(age)
    if not ages_to_solve:
        raise ValueError("No working ages selected for benchmark.")

    t_next0 = _age_to_index(pc, ages_to_solve[0]) + 1
    c_next_start = arrays["C_mat"][t_next0]

    print("=" * 78)
    print("PCHIP RUNTIME BENCHMARK")
    print("=" * 78)
    print(f"Bundle: {args.run_dir}")
    print(f"Model ages: {model.start_age}-{model.terminal_age} | retire_age={model.retire_age}")
    print(f"State grid: {pc.N_state} | n_z={pc.n_z} | n_w={pc.wealth_grid.size} | n_s={pc.s_grid.size}")
    print(f"Benchmark ages: {ages_to_solve}")
    print("Note: retirement-only solves do not hit `_interp_z_wealth`; this benchmark uses the")
    print("first working-age block fed by the saved retirement continuation policy instead.")

    print("\nBuilding in-memory Catmull-Rom baseline module...")
    solver_catmull = _build_catmull_solver_module()

    print("Warming current true-PCHIP solver...")
    solver_live.solve_working_age_step_quad(**_warmup_args(model, pc, solver_config, c_next_start, ages_to_solve[0]))
    print("Warming Catmull-Rom baseline...")
    solver_catmull.solve_working_age_step_quad(**_warmup_args(model, pc, solver_config, c_next_start, ages_to_solve[0]))

    print("\nRunning timed Catmull-Rom baseline solve...")
    result_cat = _solve_working_block(solver_catmull, model, pc, solver_config, c_next_start, ages_to_solve)
    print("Running timed true-PCHIP solve...")
    result_pchip = _solve_working_block(solver_live, model, pc, solver_config, c_next_start, ages_to_solve)

    _print_diag("Catmull-Rom Baseline", result_cat)
    _print_diag("True-PCHIP Live Solver", result_pchip)

    print("\nRuntime Comparison")
    print("------------------")
    ratio = result_pchip["total_sec"] / max(result_cat["total_sec"], 1e-30)
    delta = result_pchip["total_sec"] - result_cat["total_sec"]
    print(f"  Catmull-Rom total: {result_cat['total_sec']:.2f}s")
    print(f"  True-PCHIP total:  {result_pchip['total_sec']:.2f}s")
    print(f"  Slowdown ratio:    {ratio:.3f}x")
    print(f"  Extra wall time:   {delta:.2f}s")

    print("\nPolicy Differences")
    print("------------------")
    for age in ages_to_solve:
        c_cat, s_cat, b_cat = result_cat["policies"][age]
        c_p, s_p, b_p = result_pchip["policies"][age]
        dc = float(np.max(np.abs(c_p - c_cat)))
        ds = float(np.max(np.abs(s_p - s_cat)))
        db = float(np.max(np.abs(b_p - b_cat)))
        print(f"  Age {age}: max|dC|={dc:.4e}  max|dS|={ds:.4e}  max|dB|={db:.4e}")


if __name__ == "__main__":
    main()
