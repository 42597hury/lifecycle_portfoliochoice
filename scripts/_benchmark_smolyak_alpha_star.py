"""Optimal-alpha benchmark for Smolyak-Hermite candidates.

Investigation only. Does NOT modify lifecycle/, configs/_canonical.py, or
any production module.

The previous benchmark (_benchmark_smolyak_foc.py) showed Smolyak's
INTEGRAND-LEVEL relative error reaches ~70% at the leverage cap. But the
production question is not "does Q(α) match" — it's "does the FOC ROOT
α* match". Levels can be off by 70% while derivatives still agree well
enough for the root to land in the same place.

Test design:
  At each stress state, find α* = (α_s*, α_b*) that solves the CCV FOC
  under each quadrature rule:
    F_s(α; rule) = E_rule[ exp((1-γ) r_p^CCV) * V_next * ∂r_p/∂α_s ] = 0
    F_b(α; rule) = E_rule[ exp((1-γ) r_p^CCV) * V_next * ∂r_p/∂α_b ] = 0

  with ∂r_p^CCV/∂α_s = x_s + sigma2_xr*(0.5 - α_s) - α_b*sigma_xrxb
       ∂r_p^CCV/∂α_b = x_b + sigma2_xb*(0.5 - α_b) - α_s*sigma_xrxb

  Box-clipped to |α| ≤ 6 (canonical leverage cap). The CCV FOC is concave
  in α (Hessian -Sigma is NSD) so any interior critical point is the
  unique max; otherwise the optimum is at a cap.

Pass criterion (production-relevant):
  At every stress cell, the Smolyak rule's α* matches baseline's α* to
  within 1e-3 in each component (or both rules agree the optimum is at
  the same cap face).
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from scipy.optimize import root, minimize

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from scripts._investigate_smolyak_returnstate import SmolyakRule, _gh_rule
from scripts._benchmark_smolyak_foc import (
    smolyak_z_table, baseline_z_table, trilinear_interp_vectorised,
    synthetic_V_grid,
)


# --------------------------------------------------------------------------- #
# Vectorised FOC residual (returns 2-vector at given alpha)                    #
# --------------------------------------------------------------------------- #
def _foc_residual(alpha: np.ndarray, *, z_arr, w_arr,
                  s_cell, gamma, W,
                  L_state, L_ret, M, Phi_0_state, Phi_11,
                  base_mu_r, sigma2_xr, sigma2_xb, sigma_xrxb,
                  state_bracket_shift, state_bracket_L_inv,
                  state_bracket_grids, V_grid_flat, N_vec):
    a_s, a_b = float(alpha[0]), float(alpha[1])
    z_v = z_arr[:, :3]
    z_r = z_arr[:, 3:]
    v = z_v @ L_state.T
    r_resid = z_r @ L_ret.T
    s_next = (Phi_0_state + Phi_11 @ s_cell)[None, :] + v
    log_R = base_mu_r[None, :] + v @ M.T + r_resid
    log_R_bill = log_R[:, 0]
    x_s = log_R[:, 1]
    x_b = log_R[:, 2]
    r_p = (log_R_bill + a_s * x_s + a_b * x_b
           + 0.5 * (a_s * sigma2_xr + a_b * sigma2_xb)
           - 0.5 * (a_s * a_s * sigma2_xr
                    + 2.0 * a_s * a_b * sigma_xrxb
                    + a_b * a_b * sigma2_xb))
    # ∂r_p/∂α_s and ∂r_p/∂α_b
    dr_da_s = x_s + sigma2_xr * (0.5 - a_s) - a_b * sigma_xrxb
    dr_da_b = x_b + sigma2_xb * (0.5 - a_b) - a_s * sigma_xrxb
    # Integrand kernel: exp((1-γ) r_p) * V_next.
    u_next = (s_next - state_bracket_shift[None, :]) @ state_bracket_L_inv.T
    V_next = trilinear_interp_vectorised(u_next, state_bracket_grids,
                                          V_grid_flat, N_vec)
    kernel = np.exp((1.0 - gamma) * r_p) * V_next * (W ** (1.0 - gamma))
    F_s = float(np.dot(w_arr, kernel * dr_da_s))
    F_b = float(np.dot(w_arr, kernel * dr_da_b))
    return np.array([F_s, F_b])


def find_alpha_star(z_arr, w_arr, *, kw, cap=6.0,
                    init_seeds=None, xtol=1e-9):
    """Find (α_s*, α_b*) solving F = 0 (interior) or hitting a cap (boundary).

    Strategy:
      1. Try several seeds with Powell hybrid root-find (interior branch).
         The CCV FOC is concave -> any interior critical point is THE max.
      2. If no interior root inside [-cap, cap]^2 is found, fall back to a
         box-constrained minimization of ||F||^2 to identify the cap face.

    Returns (alpha_star, status_str, residual_norm).
    """
    if init_seeds is None:
        init_seeds = [
            np.array([0.85, 0.44]),       # canonical solver init
            np.array([2.5, 1.0]),
            np.array([5.0, -2.0]),
            np.array([0.0, 0.0]),
            np.array([-2.0, 3.0]),
        ]

    def residual(a):
        return _foc_residual(a, z_arr=z_arr, w_arr=w_arr, **kw)

    best_alpha = None
    best_resnorm = np.inf
    best_status = "no_solution"
    for x0 in init_seeds:
        try:
            res = root(residual, x0, method="hybr",
                       options={"xtol": xtol, "maxfev": 400})
            if not res.success:
                continue
            a = res.x
            rn = float(np.linalg.norm(res.fun))
            in_box = (-cap - 1e-9 <= a[0] <= cap + 1e-9 and
                      -cap - 1e-9 <= a[1] <= cap + 1e-9)
            if in_box and rn < best_resnorm:
                best_resnorm = rn
                best_alpha = a.copy()
                best_status = "interior"
        except Exception:
            continue

    if best_alpha is not None and best_resnorm < 1e-5:
        return best_alpha, best_status, best_resnorm

    # Fallback: box-constrained minimization of ||F||^2 to capture cap-bound case
    def obj(a):
        F = residual(a)
        return float(F @ F)

    cap_seeds = [
        np.array([cap, cap]), np.array([cap, -cap]),
        np.array([-cap, cap]), np.array([-cap, -cap]),
        np.array([cap, 0.0]), np.array([-cap, 0.0]),
        np.array([0.0, cap]), np.array([0.0, -cap]),
        np.array([1.0, 1.0]),
    ]
    best2_alpha = None; best2_obj = np.inf
    for x0 in cap_seeds:
        try:
            r2 = minimize(obj, x0, method="L-BFGS-B",
                          bounds=[(-cap, cap), (-cap, cap)],
                          options={"ftol": 1e-12, "gtol": 1e-10})
            if r2.success and r2.fun < best2_obj:
                best2_obj = float(r2.fun)
                best2_alpha = r2.x.copy()
        except Exception:
            continue
    if best2_alpha is None:
        return np.array([np.nan, np.nan]), "FAIL", np.inf
    F = residual(best2_alpha)
    return best2_alpha, "boundary_or_residual", float(np.linalg.norm(F))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_cells", type=int, default=8)
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--W", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    np.set_printoptions(precision=4, suppress=True, linewidth=130)

    print("=" * 84)
    print("ALPHA-STAR BENCHMARK: Smolyak FOC root vs baseline FOC root")
    print("=" * 84)

    var_cfg, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)
    pc = Precompute(model, CANONICAL_DISC, verbose=False)
    cap = float(CANONICAL_SOLVER.alpha_max)

    M = np.asarray(model.M, dtype=float)
    Sigma_ss = 0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T)
    Sigma_r_cond = 0.5 * (np.asarray(model.Sigma_r_cond)
                            + np.asarray(model.Sigma_r_cond).T)
    L_state = np.linalg.cholesky(Sigma_ss)
    L_ret   = np.linalg.cholesky(Sigma_r_cond)
    Phi_0_state = np.asarray(model.Phi_0_state, dtype=float)
    Phi_11      = np.asarray(model.Phi_11, dtype=float)
    sigma2_xr  = float(pc.sigma2_xr)
    sigma2_xb  = float(pc.sigma2_xb)
    sigma_xrxb = float(pc.sigma_xrxb)
    state_bracket_shift = np.asarray(pc.state_bracket_shift, dtype=float)
    state_bracket_L_inv = np.asarray(pc.state_bracket_L_inv, dtype=float)
    state_bracket_grids = [np.asarray(g, dtype=float) for g in pc.state_bracket_grids]
    N_vec = tuple(int(len(g)) for g in state_bracket_grids)

    V_grid_flat = synthetic_V_grid(pc.state_grid, gamma=args.gamma, W=args.W)

    # --- Build z-tables for the three rules
    K_v_can = (3, 4, 4); K_r_can = (3, 5, 5); K_max = K_v_can + K_r_can
    L_target = 4.0
    w_canonical = tuple(L_target / (K_max[k] - 1) for k in range(6))
    z_max_per_axis = np.array([float(np.max(np.abs(_gh_rule(K)[0]))) for K in K_max])
    anchors = np.array(list(itertools.product(*[[-1.0, +1.0]] * 6)),
                        dtype=float) * z_max_per_axis[None, :]

    rule_L4 = SmolyakRule.build(4.0, w_canonical, K_max)
    rule_L5 = SmolyakRule.build(5.0, w_canonical, K_max)

    z_base, w_base = baseline_z_table(K_v_can, K_r_can)
    z_L4,   w_L4   = smolyak_z_table(rule_L4, anchors)
    z_L5,   w_L5   = smolyak_z_table(rule_L5, anchors)

    rules = [
        ("baseline",         z_base, w_base),
        ("L=4 + 64 anchors", z_L4,   w_L4),
        ("L=5 + 64 anchors", z_L5,   w_L5),
    ]
    print(f"Rules: baseline N={len(w_base)}, L=4+anch N={len(w_L4)}, L=5+anch N={len(w_L5)}")
    print(f"Cap |alpha| <= {cap}, gamma = {args.gamma}, W = {args.W}\n")

    # --- Pick stress cells: representative spread of state-grid corners and body
    rng = np.random.default_rng(args.seed)
    n_total = pc.state_grid.shape[0]
    proj = pc.state_grid @ np.array([0.1, 1.0, 1.0])
    sorted_idx = np.argsort(proj)
    chosen = [int(sorted_idx[-1]),                         # max bond stress
              int(sorted_idx[0]),                          # min bond stress
              int(sorted_idx[len(sorted_idx)//2])]         # median
    rest = np.setdiff1d(np.arange(n_total), chosen, assume_unique=False)
    n_extra = max(0, args.n_cells - len(chosen))
    chosen += list(map(int, rng.choice(rest, size=n_extra, replace=False)))
    chosen = chosen[:args.n_cells]

    # --- Run alpha-star benchmark per cell
    print("=" * 84)
    print(f"Per-cell alpha* under each rule  (residual_norm in parens)")
    print("=" * 84)
    print(f"{'i_s':>4} {'state (cy,spr,y_1)':<28} "
          f"{'baseline alpha*':>26} {'L=4+anch alpha*':>26} {'L=5+anch alpha*':>26}    "
          f"{'|d_a| L=4':>11} {'|d_a| L=5':>11}")
    print("-" * 156)

    summary_d4 = []
    summary_d5 = []
    cap_disagreement = 0
    cap_match_count = 0
    interior_count = 0
    for i_s in chosen:
        s_cell = pc.state_grid[i_s]
        base_mu_r = pc.const_r + pc.A_r @ s_cell
        kw = dict(s_cell=s_cell, gamma=args.gamma, W=args.W,
                  L_state=L_state, L_ret=L_ret, M=M,
                  Phi_0_state=Phi_0_state, Phi_11=Phi_11,
                  base_mu_r=base_mu_r,
                  sigma2_xr=sigma2_xr, sigma2_xb=sigma2_xb, sigma_xrxb=sigma_xrxb,
                  state_bracket_shift=state_bracket_shift,
                  state_bracket_L_inv=state_bracket_L_inv,
                  state_bracket_grids=state_bracket_grids,
                  V_grid_flat=V_grid_flat, N_vec=N_vec)
        results = []
        for name, z_arr, w_arr in rules:
            t0 = time.time()
            a_star, status, rn = find_alpha_star(z_arr, w_arr, kw=kw, cap=cap)
            dt = time.time() - t0
            results.append((name, a_star, status, rn, dt))

        _, a_base, _, rn_base, _ = next(r for r in results if r[0] == "baseline")
        a_L4 = results[1][1]; rn_L4 = results[1][3]
        a_L5 = results[2][1]; rn_L5 = results[2][3]
        d4 = float(np.linalg.norm(a_L4 - a_base))
        d5 = float(np.linalg.norm(a_L5 - a_base))
        summary_d4.append(d4)
        summary_d5.append(d5)

        # Detect cap-bound agreement
        def at_cap(a):
            return (abs(abs(a[0]) - cap) < 1e-3 or abs(abs(a[1]) - cap) < 1e-3)
        if at_cap(a_base):
            if at_cap(a_L4) and at_cap(a_L5):
                cap_match_count += 1
            else:
                cap_disagreement += 1
        else:
            interior_count += 1

        s_str = f"({s_cell[0]:+.3f}, {s_cell[1]:+.3f}, {s_cell[2]:+.3f})"
        a_base_str = f"({a_base[0]:+6.3f},{a_base[1]:+6.3f}) ({rn_base:.0e})"
        a_L4_str   = f"({a_L4[0]:+6.3f},{a_L4[1]:+6.3f}) ({rn_L4:.0e})"
        a_L5_str   = f"({a_L5[0]:+6.3f},{a_L5[1]:+6.3f}) ({rn_L5:.0e})"
        print(f"{i_s:>4} {s_str:<28} {a_base_str:>26} {a_L4_str:>26} {a_L5_str:>26}    "
              f"{d4:>11.3e} {d5:>11.3e}")

    print()
    print("=" * 84)
    print("Summary")
    print("=" * 84)
    summary_d4 = np.array(summary_d4)
    summary_d5 = np.array(summary_d5)
    print(f"Stress cells scanned: {len(summary_d4)}  ({interior_count} interior baseline-α*, "
          f"{cap_match_count} cap-bound matched, {cap_disagreement} cap disagreed)")
    print()
    print(f"{'rule':<22} {'mean |d_a|':>14} {'max |d_a|':>14} "
          f"{'pass (1e-3)':>14} {'pass (1e-2)':>14}")
    print("-" * 84)
    for name, ds in [("L=4 + 64 anchors", summary_d4), ("L=5 + 64 anchors", summary_d5)]:
        n_pass_3 = int(np.sum(ds < 1e-3))
        n_pass_2 = int(np.sum(ds < 1e-2))
        print(f"{name:<22} {ds.mean():>14.3e} {ds.max():>14.3e} "
              f"{n_pass_3:>5d}/{len(ds):>5d} {n_pass_2:>5d}/{len(ds):>5d}")
    print()
    print("Verdict:")
    print("  pass(1e-3) = production-tight match (|Δα*| < 0.001 component-wise)")
    print("  pass(1e-2) = acceptable for value-function approximation purposes")
    print("  If max |Δα*| < 1e-2 across all stress cells, Smolyak is viable as a")
    print("  speedup despite the 70% integrand-level error at cap-bound corners,")
    print("  because the FOC root happens to land in the same place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
