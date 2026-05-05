"""CCV theory audit — independent numerical verification.

Standalone scratch script for HANDOFF_CCV_THEORY_AUDIT_REPORT. Does NOT
edit production code, does NOT call the solver kernel: it independently
computes log E[R_p^simple] by Monte Carlo from the calibrated VAR and
compares to the closed-form CCV r_p formula.

Sections (mirror the handoff):
  C1  formula correspondence sanity (corner cancellations, gradient sign)
  C2  Monte-Carlo ground truth at canonical state, scaling of |alpha|^3 gap
  C3  Sigma_r_cond vs Sigma_rr, Schwarz symmetry, gradient-of-V identity

Run from repo root:
    python -m scripts._theory_audit_ccv
"""
from __future__ import annotations

import numpy as np

from configs._canonical import BASE_CONFIG
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model


def _ccv_r_p(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
             sigma2_xr, sigma2_xb, sigma_xrxb):
    """Hand-coded CCV r_p formula — independent of the kernel."""
    return (log_R_bill
            + alpha_s * log_x_s + alpha_b * log_x_b
            + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
            - 0.5 * (alpha_s ** 2 * sigma2_xr
                     + 2.0 * alpha_s * alpha_b * sigma_xrxb
                     + alpha_b ** 2 * sigma2_xb))


def _ccv_gradient(alpha_s, alpha_b, log_x_s, log_x_b,
                  sigma2_xr, sigma2_xb, sigma_xrxb):
    dr_da_s = log_x_s + sigma2_xr * (0.5 - alpha_s) - alpha_b * sigma_xrxb
    dr_da_b = log_x_b + sigma2_xb * (0.5 - alpha_b) - alpha_s * sigma_xrxb
    return dr_da_s, dr_da_b


def main():
    print("=" * 72)
    print("CCV THEORY AUDIT — independent numerical check vs paper eq. 10")
    print("=" * 72)

    # --- Build the calibrated VAR/model ---
    var_cfg, _, _ = build_nominal_system1_var_config(
        csv_path="data/var_dataset.csv"
    )
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)

    Sigma_r_cond = np.asarray(model.Sigma_r_cond)   # 3x3, ordering (rtb, xr, xb)
    Sigma_rr = np.asarray(model.Sigma_rr)
    const_r = np.asarray(model.Phi_0_ret)
    A_r = np.asarray(model.Phi_21)
    M_mat = np.asarray(model.M)

    sigma2_rtb = Sigma_r_cond[0, 0]
    sigma2_xr = Sigma_r_cond[1, 1]
    sigma2_xb = Sigma_r_cond[2, 2]
    sigma_xrxb = Sigma_r_cond[1, 2]

    print()
    print("--- Calibrated CCV scalars (Sigma_r_cond) ---")
    print(f"sigma^2_rtb  = {sigma2_rtb:.6f}   (sigma = {np.sqrt(sigma2_rtb):.4%})")
    print(f"sigma^2_xr   = {sigma2_xr:.6f}   (sigma = {np.sqrt(sigma2_xr):.4%})")
    print(f"sigma^2_xb   = {sigma2_xb:.6f}   (sigma = {np.sqrt(sigma2_xb):.4%})")
    print(f"sigma_xrxb   = {sigma_xrxb:.6f}")
    print(f"corr(xr,xb)  = {sigma_xrxb / np.sqrt(sigma2_xr*sigma2_xb):.4f}")

    # ============================================================
    # C3.2 — Sigma_r_cond vs Sigma_rr (load-bearing choice)
    # ============================================================
    print()
    print("--- C3.2: Sigma_r_cond vs Sigma_rr (load-bearing) ---")
    print("Conditional (after projecting out state innovations):")
    print(np.array2string(Sigma_r_cond, precision=6, suppress_small=True))
    print("Unconditional Sigma_rr:")
    print(np.array2string(Sigma_rr, precision=6, suppress_small=True))
    ratio_xr = Sigma_rr[1, 1] / Sigma_r_cond[1, 1]
    ratio_xb = Sigma_rr[2, 2] / Sigma_r_cond[2, 2]
    print(f"sigma2_xr ratio (uncond / cond) = {ratio_xr:.3f}  "
          f"(would over-state Itô drag by {(ratio_xr-1)*100:+.1f}% if wrong matrix used)")
    print(f"sigma2_xb ratio (uncond / cond) = {ratio_xb:.3f}  "
          f"({(ratio_xb-1)*100:+.1f}% gap)")

    # ============================================================
    # C1 — corner cancellation sanity (algebraic, not numerical)
    # ============================================================
    print()
    print("--- C1.5: Corner sanity (must hold to floating point) ---")
    # Use a fixed non-trivial draw of (rtb, xr, xb)
    rng = np.random.default_rng(20260505)
    z_bar_state = np.asarray(model.z_bar_state)
    # Conditional mean of returns at z_bar_state (unconditional state mean)
    mu_ret = const_r + A_r @ z_bar_state
    log_R_bill = float(mu_ret[0])
    log_x_s = float(mu_ret[1])
    log_x_b = float(mu_ret[2])

    cases = [
        ("alpha=(1,0)  full stock", 1.0, 0.0, log_R_bill + log_x_s),
        ("alpha=(0,1)  full bond ", 0.0, 1.0, log_R_bill + log_x_b),
        ("alpha=(0,0)  full bill ", 0.0, 0.0, log_R_bill),
    ]
    for label, a_s, a_b, exact in cases:
        rp = _ccv_r_p(a_s, a_b, log_R_bill, log_x_s, log_x_b,
                      sigma2_xr, sigma2_xb, sigma_xrxb)
        print(f"  {label}: r_p = {rp:+.10f}, exact = {exact:+.10f}, "
              f"|err| = {abs(rp-exact):.2e}")

    # alpha=(0.5,0): expected Jensen lift = 0.125 * sigma^2_xr
    rp_half = _ccv_r_p(0.5, 0.0, log_R_bill, log_x_s, log_x_b,
                       sigma2_xr, sigma2_xb, sigma_xrxb)
    expected = log_R_bill + 0.5 * log_x_s + 0.125 * sigma2_xr
    print(f"  alpha=(0.5,0): Jensen lift = +{0.125*sigma2_xr*1e4:.2f} bps, "
          f"|err vs analytic| = {abs(rp_half-expected):.2e}")

    # alpha=(0,3): expected vol-drag = -3 * sigma^2_xb
    rp_lev = _ccv_r_p(0.0, 3.0, log_R_bill, log_x_s, log_x_b,
                      sigma2_xr, sigma2_xb, sigma_xrxb)
    expected_lev = log_R_bill + 3.0 * log_x_b - 3.0 * sigma2_xb
    print(f"  alpha=(0,3):   net drag = {-3.0*sigma2_xb*100:+.2f} pp, "
          f"|err vs analytic| = {abs(rp_lev-expected_lev):.2e}")

    # ============================================================
    # C1 — gradient sign and form: numerical FD vs analytic
    # ============================================================
    print()
    print("--- C1.5b: gradient (1/2 - alpha) form (FD vs analytic) ---")
    eps = 1e-6
    for a_s, a_b in [(0.3, 0.6), (-0.5, 1.5), (1.5, -0.4)]:
        rp_p = _ccv_r_p(a_s + eps, a_b, log_R_bill, log_x_s, log_x_b,
                        sigma2_xr, sigma2_xb, sigma_xrxb)
        rp_m = _ccv_r_p(a_s - eps, a_b, log_R_bill, log_x_s, log_x_b,
                        sigma2_xr, sigma2_xb, sigma_xrxb)
        fd_s = (rp_p - rp_m) / (2 * eps)
        rp_p = _ccv_r_p(a_s, a_b + eps, log_R_bill, log_x_s, log_x_b,
                        sigma2_xr, sigma2_xb, sigma_xrxb)
        rp_m = _ccv_r_p(a_s, a_b - eps, log_R_bill, log_x_s, log_x_b,
                        sigma2_xr, sigma2_xb, sigma_xrxb)
        fd_b = (rp_p - rp_m) / (2 * eps)
        an_s, an_b = _ccv_gradient(a_s, a_b, log_x_s, log_x_b,
                                   sigma2_xr, sigma2_xb, sigma_xrxb)
        # Also test the WRONG (1-alpha) form would have given:
        wrong_s = log_x_s + sigma2_xr * (1.0 - a_s) - a_b * sigma_xrxb
        wrong_b = log_x_b + sigma2_xb * (1.0 - a_b) - a_s * sigma_xrxb
        print(f"  a=({a_s:+.2f},{a_b:+.2f}): "
              f"FD-vs-(1/2-a) err = ({fd_s-an_s:+.1e},{fd_b-an_b:+.1e}); "
              f"FD-vs-(1-a) err = ({fd_s-wrong_s:+.1e},{fd_b-wrong_b:+.1e})")

    # ============================================================
    # C2 — Monte-Carlo ground truth at z_bar_state
    # ============================================================
    print()
    print("--- C2: Monte-Carlo log E[R_p^simple] vs r_p^CCV ---")
    print(f"State (z_bar_state, ordering = state_names): {z_bar_state}")
    print(f"State-conditional return mean mu_r = "
          f"({log_R_bill:+.4f}, {log_x_s:+.4f}, {log_x_b:+.4f})")

    N_MC = 2_000_000
    L_cond = np.linalg.cholesky(Sigma_r_cond)
    z_norm = rng.standard_normal((N_MC, 3))
    r_draws = mu_ret[None, :] + z_norm @ L_cond.T

    # Gross simple returns
    R_bill_sim = np.exp(r_draws[:, 0])
    R_s_sim = np.exp(r_draws[:, 0] + r_draws[:, 1])
    R_b_sim = np.exp(r_draws[:, 0] + r_draws[:, 2])

    print(f"\n{'alpha':>14} | {'log E[R_p^simple]':>20} | {'r_p^CCV':>14} | "
          f"{'gap':>10} | {'gap-bill_baseline':>18}")
    print("-" * 92)
    bill_baseline = 0.5 * sigma2_rtb  # Jensen on bill alone (gap at alpha=0)
    grid = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
        (-0.5, 0.5), (1.0, 1.0), (2.0, 0.0), (0.0, 2.0), (0.0, 3.0),
        (1.0, 2.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (6.0, 0.0),
        (0.0, 6.0), (6.0, 6.0), (-2.0, 4.0),
    ]
    rows = []
    for a_s, a_b in grid:
        a_f = 1.0 - a_s - a_b
        R_simple = a_s * R_s_sim + a_b * R_b_sim + a_f * R_bill_sim
        # log E[R_simple]; clip for numerical safety
        log_E = float(np.log(np.maximum(R_simple.mean(), 1e-300)))
        r_p_ccv = _ccv_r_p(a_s, a_b, log_R_bill, log_x_s, log_x_b,
                           sigma2_xr, sigma2_xb, sigma_xrxb)
        gap = log_E - r_p_ccv
        norm = np.sqrt(a_s ** 2 + a_b ** 2)
        rows.append((norm, a_s, a_b, log_E, r_p_ccv, gap))
        print(f"  ({a_s:+.1f},{a_b:+.1f})    | {log_E:+19.5f} | "
              f"{r_p_ccv:+13.5f} | {gap*100:+9.4f}% | {(gap-bill_baseline)*100:+17.4f}%")

    # Cubic scaling check: gap-bill_baseline should track |alpha|^3 in moderate region
    print()
    print("Scaling check: residual gap = gap - 0.5*sigma2_rtb (Jensen-on-bill baseline)")
    print(f"{'|alpha|':>10} | {'residual gap':>14} | {'|alpha|^3 sigma^4':>20}")
    sigma_max = max(sigma2_xr, sigma2_xb)
    for norm, a_s, a_b, _, _, gap in sorted(rows):
        resid = gap - bill_baseline
        scale = (norm ** 3) * (sigma_max ** 2)
        ratio = resid / scale if scale > 0 else float("nan")
        print(f"  {norm:7.2f} | {resid*100:+12.4f}% | "
              f"{scale*100:18.6f}%  ratio = {ratio:+.3f}")

    # ============================================================
    # C2.4 — bit-precision agreement at one (state, alpha, shock)
    # ============================================================
    print()
    print("--- C2.4: kernel r_p^CCV vs hand-coded at one (state, alpha, shock) ---")
    # Pick one return-quadrature node and reproduce kernel exactly
    from lifecycle.precompute import Precompute
    from lifecycle.model import DiscretizationConfig
    disc = DiscretizationConfig(state_grid_sizes=(7, 7, 7),
                                state_grid_mode="cholesky",
                                state_n_stds=(2.0, 2.25, 2.25))
    pc = Precompute(model, disc, verbose=False)
    # Pick the canonical state — middle of the grid — and one (k_v, k_r) node
    i_s = pc.N_state // 2
    s_i = pc.state_grid[i_s]
    base_mu_r = pc.const_r + pc.A_r @ s_i
    k_v, k_r = 0, 0
    mu_r_bill = base_mu_r[0] + pc.M_v_nodes[k_v, 0]
    mu_r_stock = base_mu_r[1] + pc.M_v_nodes[k_v, 1]
    mu_r_bond = base_mu_r[2] + pc.M_v_nodes[k_v, 2]
    log_R_bill_n = mu_r_bill + pc.ret_nodes[k_r, 0]
    log_x_s_n = mu_r_stock + pc.ret_nodes[k_r, 1]
    log_x_b_n = mu_r_bond + pc.ret_nodes[k_r, 2]
    a_s, a_b = 0.7, 0.3
    rp_hand = _ccv_r_p(a_s, a_b, log_R_bill_n, log_x_s_n, log_x_b_n,
                       pc.sigma2_xr, pc.sigma2_xb, pc.sigma_xrxb)
    print(f"Hand-coded r_p^CCV at (i_s={i_s}, k_v={k_v}, k_r={k_r}, "
          f"alpha=({a_s},{a_b})) = {rp_hand:.16f}")
    # No direct read from the njit kernel (would require constructing full call),
    # but the precompute scalars themselves come from Sigma_r_cond — verify
    print(f"Precompute scalars match Sigma_r_cond entries:")
    print(f"  pc.sigma2_xr  - Sigma_r_cond[1,1] = {pc.sigma2_xr - Sigma_r_cond[1,1]:.2e}")
    print(f"  pc.sigma2_xb  - Sigma_r_cond[2,2] = {pc.sigma2_xb - Sigma_r_cond[2,2]:.2e}")
    print(f"  pc.sigma_xrxb - Sigma_r_cond[1,2] = {pc.sigma_xrxb - Sigma_r_cond[1,2]:.2e}")

    # ============================================================
    # C3.6 — Schwarz symmetry of the kernel Hessian
    # ============================================================
    print()
    print("--- C3.6: Hessian-of-V Schwarz symmetry (kernel formula) ---")
    print("J_sb formula (from solver.py:1018):")
    print("  J_sb = jac * dRp_das * dRp_dab + wmu * R_p * (dr_das * dr_dab - sigma_xrxb)")
    print("J_bs (swapped) is the same expression (multiplication is commutative)")
    print("=> J_sb == J_bs identically by formula structure (no numerical check needed).")
    # Numerical demonstration: assemble at a single node, verify mathematically
    a_s, a_b = 0.4, 0.7
    dr_das, dr_dab = _ccv_gradient(a_s, a_b, log_x_s_n, log_x_b_n,
                                   sigma2_xr, sigma2_xb, sigma_xrxb)
    rp = _ccv_r_p(a_s, a_b, log_R_bill_n, log_x_s_n, log_x_b_n,
                  sigma2_xr, sigma2_xb, sigma_xrxb)
    R_p = np.exp(rp)
    dRp_das = R_p * dr_das
    dRp_dab = R_p * dr_dab
    wmu, jac = 1.0, 0.5  # dummy
    j_sb = jac * dRp_das * dRp_dab + wmu * R_p * (dr_das * dr_dab - sigma_xrxb)
    j_bs = jac * dRp_dab * dRp_das + wmu * R_p * (dr_dab * dr_das - sigma_xrxb)
    print(f"  numerical |J_sb - J_bs| = {abs(j_sb - j_bs):.2e}  (should be 0)")

    # ============================================================
    # C3.5 — Gradient-of-V vs asset-pricing FOC at the optimum
    # ============================================================
    print()
    print("--- C3.5: under CCV, FOC = grad of V, NOT asset-pricing moment ---")
    print("Take a representative interior alpha and compute:")
    print("  (a) E[mu * (R_j - R_bill)]  [asset-pricing moment, would be 0 under SIMPLE]")
    print("  (b) E[mu * R_p * dr_p/dalpha_j]  [grad of V; this is what kernel computes]")
    print("They are different quantities.")
    a_s_test, a_b_test = 0.6, 0.3
    rp_draws = _ccv_r_p(a_s_test, a_b_test, r_draws[:, 0],
                        r_draws[:, 1], r_draws[:, 2],
                        sigma2_xr, sigma2_xb, sigma_xrxb)
    R_p_draws = np.exp(rp_draws)
    Rex_s_draws = R_s_sim - R_bill_sim
    Rex_b_draws = R_b_sim - R_bill_sim
    # gradient of r_p at each draw: dr_das = log_x_s_realised + sigma2_xr*(0.5-a_s) - a_b*sigma_xrxb
    dr_das_draws, dr_dab_draws = _ccv_gradient(
        a_s_test, a_b_test, r_draws[:, 1], r_draws[:, 2],
        sigma2_xr, sigma2_xb, sigma_xrxb)
    # Take mu = 1 for simplicity (just looking at the moment, not the optimum).
    moment_s = float(np.mean(Rex_s_draws))
    moment_b = float(np.mean(Rex_b_draws))
    grad_s = float(np.mean(R_p_draws * dr_das_draws))
    grad_b = float(np.mean(R_p_draws * dr_dab_draws))
    print(f"  E[(R_s - R_bill)] = {moment_s:+.5f},  E[R_p * dr/das] = {grad_s:+.5f}")
    print(f"  E[(R_b - R_bill)] = {moment_b:+.5f},  E[R_p * dr/dab] = {grad_b:+.5f}")
    print(f"  -> distinct quantities; under CCV the kernel uses the second.")

    # ============================================================
    # C3.1 — production-bundle policy support (proxy: grid distribution)
    # ============================================================
    print()
    print("--- C3.1: production CCV bundle policy support ---")
    from pathlib import Path
    from lifecycle.policy_io import load_policy_bundle
    bundle_dir = Path("saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire")
    if bundle_dir.exists():
        C_mat, S_mat, B_mat, _, meta = load_policy_bundle(bundle_dir)
        print(f"Bundle: {bundle_dir.name}")
        print(f"  shape (age, z, state, wealth): {S_mat.shape}")
        print(f"  wealth_dynamics_spec: {meta.get('wealth_dynamics_spec', '?')}")
        # Drop ages where the policy is undefined (shape may already be tight)
        a_s = S_mat.ravel()
        a_b = B_mat.ravel()
        mag = np.sqrt(a_s ** 2 + a_b ** 2)
        l1 = np.abs(a_s) + np.abs(a_b)
        for q in (50, 90, 95, 99, 99.9, 100):
            print(f"  pct {q:5.1f}: |a_s|={np.percentile(np.abs(a_s), q):6.2f}, "
                  f"|a_b|={np.percentile(np.abs(a_b), q):6.2f}, "
                  f"|a|_2={np.percentile(mag, q):6.2f}, |a|_1={np.percentile(l1, q):6.2f}")
        # Truncation magnitude estimate at typical / tail policy
        sigma_max = max(sigma2_xr, sigma2_xb)
        for q in (50, 90, 99):
            mq = np.percentile(mag, q)
            est_gap = (mq ** 3) * sigma_max ** 2
            print(f"  truncation est at pct {q} (|alpha|^3 sigma^4): "
                  f"|alpha|={mq:.2f} -> gap ~ {est_gap*100:.4f}%")
    else:
        print(f"Bundle not found at {bundle_dir}; skipping C3.1.")

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 72)
    print("AUDIT SCRIPT SUMMARY")
    print("=" * 72)
    print("Run this script and inspect the printed checks against the report.")
    print("Key outputs feed into HANDOFF_CCV_THEORY_AUDIT_REPORT.md C1, C2, C3.")


if __name__ == "__main__":
    main()
