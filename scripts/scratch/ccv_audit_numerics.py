"""CCV audit numerical checks (throwaway).

Builds the canonical model, reports the key VAR/CCV matrices, and runs
three smoke checks against CCV w8566:
  1. Print Sigma_rr, Sigma_r_cond, M, mean returns, sigma vector.
     Compare with the prior CCV audit (HANDOFF_CCV_THEORY_AUDIT_REPORT.md
     §C3.2) and CCV Table 2.
  2. Compute the CCV myopic Markowitz alpha at the unconditional state
     mean for gamma in {2, 5, 10}.
  3. Verify the analytical CCV r_p formula at alpha=0.5 (Jensen lift &
     Itô drag) against a Monte-Carlo log E[arithmetic R_p].

Saved under scripts/scratch/ — not committed.
"""
from __future__ import annotations

import numpy as np

# Project imports must come from JAX-rewrite layer.
from configs._canonical import BASE_CONFIG
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute


def _line(s):
    print(s, flush=True)


def main():
    # Small but legitimate Precompute build; we only need matrices, not policies.
    disc = DiscretizationConfig(
        n_wealth=20, wealth_min=0.13, wealth_max=200.0,
        n_savings=20,
        state_grid_sizes=(3, 3, 3, 3),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_z=3,
        n_eps_nodes=3, n_eta_nodes=3,
        n_ret_nodes_1d=(3, 3),
        n_state_quad_nodes=(2, 3, 2, 3),
    )

    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)

    _line("=" * 78)
    _line("CCV audit numerics — canonical model (hardcoded VAR fallback)")
    _line("=" * 78)

    state_names = list(model.state_names)
    ret_names = list(model.ret_names)
    _line(f"state_names = {state_names}  (n_state={model.n_state})")
    _line(f"ret_names   = {ret_names}    (n_ret={model.n_ret})")
    _line(f"rtb_index_in_state = {model.rtb_index_in_state}")
    xr_pos = ret_names.index("xr")
    xb_pos = ret_names.index("xb")
    _line(f"xr_pos={xr_pos}  xb_pos={xb_pos}")

    _line("")
    _line("--- 1. Unconditional and conditional return covariances ---")
    Sigma_rr = np.asarray(model.Sigma_rr, dtype=float)
    Sigma_r_cond = np.asarray(model.Sigma_r_cond, dtype=float)
    M = np.asarray(model.M, dtype=float)
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=float)
    Sigma_rs = np.asarray(model.Sigma_rs, dtype=float)
    Phi_0_state = np.asarray(model.Phi_0_state, dtype=float)
    Phi_11 = np.asarray(model.Phi_11, dtype=float)
    Phi_0_ret = np.asarray(model.Phi_0_ret, dtype=float)
    Phi_21 = np.asarray(model.Phi_21, dtype=float)

    _line(f"Sigma_rr ({ret_names}):")
    _line(np.array2string(Sigma_rr, precision=6, suppress_small=False))
    _line(f"Sigma_r_cond ({ret_names}):")
    _line(np.array2string(Sigma_r_cond, precision=6, suppress_small=False))
    _line(f"diag(Sigma_rr)   = {np.diag(Sigma_rr)}")
    _line(f"diag(Sigma_cond) = {np.diag(Sigma_r_cond)}")
    _line(f"Annual std (Sigma_rr): {np.sqrt(np.diag(Sigma_rr))}")
    _line(f"Annual std (Sigma_r_cond): {np.sqrt(np.diag(Sigma_r_cond))}")
    _line(f"ratio Sigma_rr/Sigma_cond: {np.diag(Sigma_rr) / np.diag(Sigma_r_cond)}")

    _line("")
    _line("--- 2. M = Sigma_rs @ Sigma_ss^{-1} ---")
    _line(f"M shape: {M.shape}  (rows = {ret_names}, cols = {state_names})")
    _line(np.array2string(M, precision=4, suppress_small=False))

    _line("")
    _line("--- 3. Unconditional moments (stationary state) ---")
    # z_bar_state = (I - Phi_11)^{-1} @ Phi_0_state
    I = np.eye(len(Phi_0_state))
    z_bar_state = np.linalg.solve(I - Phi_11, Phi_0_state)
    z_bar_ret = Phi_0_ret + Phi_21 @ z_bar_state
    _line(f"z_bar_state (from solve): {z_bar_state}")
    _line(f"model.z_bar_state:        {np.asarray(model.z_bar_state)}")
    _line(f"z_bar_ret (model.z_bar_ret): {np.asarray(model.z_bar_ret)}")
    _line(f"Implied state mean: {dict(zip(state_names, z_bar_state))}")
    _line(f"Implied ret mean:   {dict(zip(ret_names, z_bar_ret))}")
    _line("(In Option A 1920-2011 calibration: y_1=4.40%, spr=1.26pp, dp=-3.26, rtb=1.64%, xr=5.18%, xb=1.07%)")

    _line("")
    _line("--- 4. CCV vol scalars actually consumed by the kernel ---")
    _line(f"pc.sigma2_xr  = {pc.sigma2_xr}")
    _line(f"pc.sigma2_xb  = {pc.sigma2_xb}")
    _line(f"pc.sigma_xrxb = {pc.sigma_xrxb}")
    _line("")
    _line("Compare to:")
    _line(f"Sigma_rr[xr,xr]    = {Sigma_rr[xr_pos, xr_pos]}")
    _line(f"Sigma_r_cond[xr,xr]= {Sigma_r_cond[xr_pos, xr_pos]}")
    _line(f"Sigma_rr[xb,xb]    = {Sigma_rr[xb_pos, xb_pos]}")
    _line(f"Sigma_r_cond[xb,xb]= {Sigma_r_cond[xb_pos, xb_pos]}")
    _line(f"Sigma_rr[xr,xb]    = {Sigma_rr[xr_pos, xb_pos]}")
    _line(f"Sigma_r_cond[xr,xb]= {Sigma_r_cond[xr_pos, xb_pos]}")
    _line("")
    _line("=> Production code now uses Sigma_rr (commit f23ac83, May 2026).")
    _line("   Earlier audit (HANDOFF_CCV_THEORY_AUDIT_REPORT.md C3.2) used Sigma_r_cond.")

    _line("")
    _line("--- 5. Markowitz myopic alpha at unconditional state ---")
    # CCV myopic limit:   alpha* = (1/gamma) * Sigma_rr_excess^{-1} @ mu_excess_logreturn
    # where mu is the *log* excess return (not the gross excess), and we use
    # the conditional return variance (the relevant variance for one-period
    # decisions in CCV).
    # In our setup the excess returns ARE (xr, xb), so Sigma_rr_excess = Sigma_rr[ret_idx_xr,xb].
    # Source for σ matrix is the open question — see audit report.
    # We compute under both Sigma_rr and Sigma_r_cond for the report.
    excess_idx = np.array([xr_pos, xb_pos])
    mu_excess = z_bar_ret[excess_idx]                       # E[xr], E[xb]
    Sigma_rr_excess = Sigma_rr[np.ix_(excess_idx, excess_idx)]
    Sigma_cond_excess = Sigma_r_cond[np.ix_(excess_idx, excess_idx)]
    # Apply Jensen lift to convert log-excess mean to "expected log excess
    # plus half-variance" — this is the CCV myopic formula:
    # alpha* = Sigma^{-1} (mu_e + 0.5 sigma^2) / gamma
    # Some derivations omit the +0.5σ² (they work in arithmetic excess); use
    # the version actually appearing in CCV w8566 eq.27 (myopic):
    # alpha* = (1/gamma) Sigma_xx^{-1} (mu_x + 0.5 sigma_xx_diag)
    diag_rr = np.diag(Sigma_rr_excess)
    diag_cond = np.diag(Sigma_cond_excess)

    _line(f"E[xr], E[xb] = {mu_excess}")
    _line(f"Sigma_rr_excess:")
    _line(np.array2string(Sigma_rr_excess, precision=6))
    _line(f"Sigma_cond_excess:")
    _line(np.array2string(Sigma_cond_excess, precision=6))

    for gamma in [2.0, 5.0, 10.0]:
        for label, Sig, diag_v in [("Sigma_rr", Sigma_rr_excess, diag_rr),
                                    ("Sigma_cond", Sigma_cond_excess, diag_cond)]:
            mu_lifted = mu_excess + 0.5 * diag_v
            alpha_star = (1.0 / gamma) * np.linalg.solve(Sig, mu_lifted)
            _line(f"gamma={gamma:.1f}  Σ={label:11s}: "
                  f"alpha_xr={alpha_star[0]:+.3f}  alpha_xb={alpha_star[1]:+.3f}  "
                  f"|α|₂={np.linalg.norm(alpha_star):.3f}")

    _line("")
    _line("--- 6. Variance ratio (Sigma_rr / Sigma_r_cond) by return ---")
    for k, name in enumerate(ret_names):
        v_uncond = Sigma_rr[k, k]
        v_cond = Sigma_r_cond[k, k]
        explained = 1.0 - v_cond / v_uncond if v_uncond > 0 else 0.0
        _line(f"{name}: var_uncond={v_uncond:.4e}, var_cond={v_cond:.4e}, ratio={v_uncond/v_cond:.2f}, R²(state)={explained:.3f}")

    _line("")
    _line("--- 7. Monte-Carlo vs analytical r_p at alpha=(0.5, 0) ---")
    rng = np.random.default_rng(20260506)
    N = 2_000_000
    s_bar = z_bar_state
    mu_r_ret = Phi_0_ret + Phi_21 @ s_bar      # E[r_ret | s_bar] including (xr, xb) in our 2-element ret block
    _line(f"E[xr | s_bar] = {mu_r_ret[xr_pos]:.6f}")
    _line(f"E[xb | s_bar] = {mu_r_ret[xb_pos]:.6f}")

    # Need rtb mean. rtb is now in the state block. Realised log_R_bill_t+1
    # is read from state_t+1 at rtb_idx, so we need the conditional mean of
    # rtb conditional on s_t = s_bar. Under CCV constrained estimation:
    rtb_idx_in_state = model.rtb_index_in_state
    state_next_mean = Phi_0_state + Phi_11 @ s_bar       # E[s_t+1 | s_t = s_bar]
    log_R_bill_mean = state_next_mean[rtb_idx_in_state]
    _line(f"E[rtb_t+1 | s_t = s_bar] = {log_R_bill_mean:.6f}")
    # Variance of rtb_{t+1} given s_t = s_bar is Sigma_ss[rtb_idx, rtb_idx]
    sigma2_rtb = float(Sigma_ss[rtb_idx_in_state, rtb_idx_in_state])
    _line(f"Var(rtb_t+1 | s_bar) = {sigma2_rtb:.4e}")

    # MC: draw v_s ~ N(0, Sigma_ss); rtb_t+1 = state_next_mean[rtb_idx] + v_s[rtb_idx].
    # Conditional on s_t=s_bar: r_excess ~ N(mu_r_ret, Sigma_rr_excess); but
    # in our VAR Sigma_r,s != 0, so we should draw the full joint innovation
    # (v_s, e_r). Actually conditional on s_t the distribution of (v_s, e_r)
    # is the joint covariance Sigma (state | ret) which equals model.Sigma_ss
    # for the state and for returns Sigma_rr | v_s = Sigma_r_cond plus
    # M @ v_s. We follow the simulator's convention: draw v_s ~ N(0,Sigma_ss)
    # then e_r ~ N(0, Sigma_r_cond), independent.
    L_ss = np.linalg.cholesky(Sigma_ss)
    L_rcond = np.linalg.cholesky(Sigma_r_cond)
    z_s = rng.standard_normal((N, len(s_bar)))
    z_r = rng.standard_normal((N, model.n_ret))
    v_s = z_s @ L_ss.T
    e_r = z_r @ L_rcond.T
    # Realised rtb_t+1
    rtb_real = log_R_bill_mean + v_s[:, rtb_idx_in_state]
    # Realised excess returns: mu_r_ret + M @ v_s + e_r
    r_full = mu_r_ret[None, :] + v_s @ M.T + e_r
    xr_real = r_full[:, xr_pos]
    xb_real = r_full[:, xb_pos]

    R_bill = np.exp(rtb_real)
    R_stock = np.exp(rtb_real + xr_real)
    R_bond = np.exp(rtb_real + xb_real)

    test_alphas = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (1.0, 1.0)]
    sigma2_xr = float(Sigma_rr[xr_pos, xr_pos])
    sigma2_xb = float(Sigma_rr[xb_pos, xb_pos])
    sigma_xrxb = float(Sigma_rr[xr_pos, xb_pos])
    sigma2_xr_cond = float(Sigma_r_cond[xr_pos, xr_pos])
    sigma2_xb_cond = float(Sigma_r_cond[xb_pos, xb_pos])
    sigma_xrxb_cond = float(Sigma_r_cond[xr_pos, xb_pos])

    _line(f"sigma2_xr     (Sigma_rr)    = {sigma2_xr:.6e}")
    _line(f"sigma2_xr     (Sigma_cond)  = {sigma2_xr_cond:.6e}")
    _line(f"sigma2_xb     (Sigma_rr)    = {sigma2_xb:.6e}")
    _line(f"sigma2_xb     (Sigma_cond)  = {sigma2_xb_cond:.6e}")

    _line("")
    _line(f"{'(α_xr,α_xb)':>12s} | {'r_p(CCV-Σrr)':>14s}  {'r_p(CCV-Σcond)':>14s}  "
          f"{'log E[Rp]':>14s}  {'gap_Σrr':>9s}  {'gap_Σcond':>11s}")

    for a_s, a_b in test_alphas:
        a_bill = 1.0 - a_s - a_b
        # Arithmetic
        Rp = a_s * R_stock + a_b * R_bond + a_bill * R_bill
        log_E_Rp = np.log(Rp.mean())

        # CCV via Sigma_rr (production, post f23ac83)
        rp_rr = (rtb_real
                 + a_s * xr_real + a_b * xb_real
                 + 0.5 * (a_s * sigma2_xr + a_b * sigma2_xb)
                 - 0.5 * (a_s * a_s * sigma2_xr + 2.0 * a_s * a_b * sigma_xrxb
                          + a_b * a_b * sigma2_xb))
        # Realised log return per draw — we want the *deterministic* CCV
        # forecast at α (the formula); for that the constants are the σ²
        # constants and the realised xr,xb,rtb feed in. log E[exp(rp_rr)]
        # is what the solver sees. We compare log E[exp(rp_rr)] to log E[Rp].
        log_E_exp_rp_rr = np.log(np.exp(rp_rr).mean())

        rp_cond = (rtb_real
                   + a_s * xr_real + a_b * xb_real
                   + 0.5 * (a_s * sigma2_xr_cond + a_b * sigma2_xb_cond)
                   - 0.5 * (a_s * a_s * sigma2_xr_cond + 2.0 * a_s * a_b * sigma_xrxb_cond
                            + a_b * a_b * sigma2_xb_cond))
        log_E_exp_rp_cond = np.log(np.exp(rp_cond).mean())

        _line(f"({a_s:+.2f},{a_b:+.2f}) | {log_E_exp_rp_rr:+.6f}  {log_E_exp_rp_cond:+.6f}  "
              f"{log_E_Rp:+.6f}  {log_E_Rp-log_E_exp_rp_rr:+.4e}  "
              f"{log_E_Rp-log_E_exp_rp_cond:+.4e}")

    _line("")
    _line("--- 8. Jensen lift / Itô drag at alpha=(0.5,0) — pure formula check ---")
    a_s = 0.5
    jensen_rr = 0.5 * a_s * sigma2_xr
    ito_rr = 0.5 * a_s * a_s * sigma2_xr
    jensen_cond = 0.5 * a_s * sigma2_xr_cond
    ito_cond = 0.5 * a_s * a_s * sigma2_xr_cond
    _line(f"Σrr:   Jensen lift = +{jensen_rr*1e4:.2f} bps,  Itô drag = -{ito_rr*1e4:.2f} bps,  net = {(jensen_rr-ito_rr)*1e4:+.2f} bps")
    _line(f"Σcond: Jensen lift = +{jensen_cond*1e4:.2f} bps, Itô drag = -{ito_cond*1e4:.2f} bps, net = {(jensen_cond-ito_cond)*1e4:+.2f} bps")

    # Predicted log E[R_p] - r_p(CCV) under the two specs at α = (0.5, 0).
    # In CCV's own derivation, E[exp(r_p^CCV)] should equal log E[arithmetic R_p]
    # *exactly* (in the zero-mean case before non-Gaussian truncation), and the
    # gap should be the leading-order σ³ truncation only.

    _line("")
    _line("=" * 78)
    _line("END")


if __name__ == "__main__":
    main()
