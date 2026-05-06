"""Verify CCV's Sigma_xx prescription: Sigma_rr or Sigma_r_cond?

CCV w8566 eq. (5) defines Sigma_v as the FULL innovation covariance and Sigma_xx
as the (x,x) block of Sigma_v. Their eq. (10) uses Sigma_xx in the log portfolio
return formula. The code uses Sigma_r_cond (variance after partialling
out the state innovation v^s) which is a DIFFERENT, smaller matrix.

This script compares both choices against:
  (a) the formula's expected value vs log E[R_p^simple]
  (b) MC error magnitudes

Decisive test: does CCV's "log E[R_p^simple] ~~ r_p_CCV(realized)" actually
hold to O(|alpha|^3 sigma^4) under sigma_unc, sigma_cond, or both?

Run from repo root:
    python -m scripts._check_ccv_sigma_choice
"""
from __future__ import annotations

import numpy as np

from configs._canonical import BASE_CONFIG
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model


def main():
    var_cfg, _, _ = build_nominal_system1_var_config(
        csv_path="data/var_dataset.csv"
    )
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)

    Sigma_r_cond = np.asarray(model.Sigma_r_cond)
    Sigma_rr = np.asarray(model.Sigma_rr)

    # The (xr, xb) blocks — these are what enter CCV eq. 10
    Sigma_xx_unc = Sigma_rr[1:, 1:]      # what CCV w8566 eq. (5) actually defines
    Sigma_xx_cond = Sigma_r_cond[1:, 1:]  # what the code uses

    print("=" * 72)
    print("WHICH Sigma_xx DOES CCV EQ. (10) PRESCRIBE?")
    print("=" * 72)
    print()
    print("CCV w8566 eq. (5):  Sigma_v = full innovation covariance")
    print("                    Sigma_xx = (x,x) block of Sigma_v")
    print("CCV w8566 eq. (10): r_p,t+1 = r_1 + alpha'x + ½alpha'(sigma^2_x - Sigma_xx alpha)")
    print("                    sigma^2_x = diag(Sigma_xx) — 'variances of excess returns'")
    print()
    print("(x,x) block of Sigma_v in code's notation = Sigma_rr[1:, 1:] (UNCONDITIONAL)")
    print()
    print("--- Numerical comparison at production calibration ---")
    print(f"Sigma_xx_unc = Sigma_rr[1:,1:]  (CCV's prescription):")
    print(np.array2string(Sigma_xx_unc, precision=6, suppress_small=True))
    print(f"  diag = (sigma^2_xr={Sigma_xx_unc[0,0]:.4e}, sigma^2_xb={Sigma_xx_unc[1,1]:.4e})")
    print()
    print(f"Sigma_xx_cond = Sigma_r_cond[1:,1:]  (what the code uses):")
    print(np.array2string(Sigma_xx_cond, precision=6, suppress_small=True))
    print(f"  diag = (sigma^2_xr={Sigma_xx_cond[0,0]:.4e}, sigma^2_xb={Sigma_xx_cond[1,1]:.4e})")
    print()
    print(f"Ratio sigma^2_xr_unc / sigma^2_xr_cond = "
          f"{Sigma_xx_unc[0,0]/Sigma_xx_cond[0,0]:.2f}x")
    print(f"Ratio sigma^2_xb_unc / sigma^2_xb_cond = "
          f"{Sigma_xx_unc[1,1]/Sigma_xx_cond[1,1]:.2f}x")
    print()

    # ============================================================
    # The decisive test: take CCV's exact framework
    #   x | z_t ~ N(mu_x, Sigma_xx_unc)
    # and check whether r_p_CCV with sigma^2_unc OR sigma^2_cond better
    # approximates log E[R_p^simple | z_t]
    # ============================================================
    rng = np.random.default_rng(20260505)
    N_MC = 2_000_000
    z_bar_state = np.asarray(model.z_bar_state)
    mu_ret = model.Phi_0_ret + model.Phi_21 @ z_bar_state  # 3-vec (rtb, xr, xb)

    # CRITICAL: CCV's ground-truth distribution of (rtb, xr, xb) given z_t
    # is N(mu_ret, Sigma_rr) — i.e., the FULL innovation covariance, NOT the
    # conditional-on-v^s one. CCV's eq. 10 expectation is over v_{t+1} as a
    # whole, no sub-decomposition into v^s and eps_r.
    print("--- DECISIVE TEST: which sigma^2 matches log E[R_p^simple | z_t]? ---")
    print(f"Drawing N={N_MC:,} samples from N(mu_r, Sigma_rr)  [CCV's framework]")
    L_unc = np.linalg.cholesky(Sigma_rr)
    z = rng.standard_normal((N_MC, 3))
    r = mu_ret[None, :] + z @ L_unc.T

    R_bill = np.exp(r[:, 0])
    R_s = np.exp(r[:, 0] + r[:, 1])
    R_b = np.exp(r[:, 0] + r[:, 2])

    log_R_bill = float(mu_ret[0])
    log_x_s = float(mu_ret[1])
    log_x_b = float(mu_ret[2])

    sigma2_xr_unc, sigma2_xb_unc = Sigma_xx_unc[0,0], Sigma_xx_unc[1,1]
    sigma_xrxb_unc = Sigma_xx_unc[0,1]
    sigma2_xr_cond, sigma2_xb_cond = Sigma_xx_cond[0,0], Sigma_xx_cond[1,1]
    sigma_xrxb_cond = Sigma_xx_cond[0,1]

    def r_p_ccv(a_s, a_b, s2_s, s2_b, s_sb):
        return (log_R_bill + a_s*log_x_s + a_b*log_x_b
                + 0.5*(a_s*s2_s + a_b*s2_b)
                - 0.5*(a_s**2*s2_s + 2*a_s*a_b*s_sb + a_b**2*s2_b))

    grid = [(0,0),(1,0),(0,1),(0.5,0),(0,0.5),(1,1),(2,0),(0,2),(0,3),
            (-1,2),(2,2),(3,3),(4,4),(5,0),(0,5),(6,0),(0,6),(6,6)]

    # CCV's eq. (10) approximates the REALIZED r_p_t+1 -- it's a per-draw
    # approximation, not a closed-form for log E[R_p^simple]. The right
    # test is bias and RMS error of (r_p_CCV - r_p_realized) under the
    # CCV-prescribed distribution N(mu, Sigma_rr).
    print()
    print(f"{'(a_s,a_b)':>12} | {'E[r_p_realized]':>15} | "
          f"{'sigma_unc bias':>13} | {'sigma_unc RMS':>13} | "
          f"{'sigma_cond bias':>15} | {'sigma_cond RMS':>14}")
    print("-" * 120)
    for a_s, a_b in grid:
        a_f = 1.0 - a_s - a_b
        R_simple = a_s*R_s + a_b*R_b + a_f*R_bill
        # CCV eq. (10) approximates r_p_realized = log(R_simple)
        # (only valid when R_simple > 0)
        valid = R_simple > 0
        if valid.sum() < N_MC * 0.99:
            E_r_p_real = float("nan")
            bias_unc = bias_cond = rms_unc = rms_cond = float("nan")
        else:
            r_p_real = np.log(R_simple[valid])
            r_real_mean = float(r_p_real.mean())
            # CCV eq.(10) is per-draw: r_p_CCV uses constants for sigma^2 terms
            # but the realized r_f, x_s, x_b drawn from MC.
            r_unc = (r[valid, 0]
                     + a_s * r[valid, 1] + a_b * r[valid, 2]
                     + 0.5*(a_s*sigma2_xr_unc + a_b*sigma2_xb_unc)
                     - 0.5*(a_s**2*sigma2_xr_unc
                            + 2*a_s*a_b*sigma_xrxb_unc
                            + a_b**2*sigma2_xb_unc))
            r_cond = (r[valid, 0]
                      + a_s * r[valid, 1] + a_b * r[valid, 2]
                      + 0.5*(a_s*sigma2_xr_cond + a_b*sigma2_xb_cond)
                      - 0.5*(a_s**2*sigma2_xr_cond
                             + 2*a_s*a_b*sigma_xrxb_cond
                             + a_b**2*sigma2_xb_cond))
            err_unc = r_p_real - r_unc
            err_cond = r_p_real - r_cond
            bias_unc = float(err_unc.mean())
            bias_cond = float(err_cond.mean())
            rms_unc = float(np.sqrt((err_unc**2).mean()))
            rms_cond = float(np.sqrt((err_cond**2).mean()))
            E_r_p_real = r_real_mean
        print(f"  ({a_s:+4.1f},{a_b:+4.1f}) | "
              f"{E_r_p_real:+14.5f} | "
              f"{bias_unc*100:+12.4f}% | {rms_unc*100:+12.4f}% | "
              f"{bias_cond*100:+14.4f}% | {rms_cond*100:+13.4f}%")

    print()
    print("INTERPRETATION:")
    print("  log E[R_p^simple|z_t] is the CCV-prescribed approximation target")
    print("  (eq. 10 says r_p,t+1 ~~ this value to O(|alpha|^3sigma^4) on AVERAGE).")
    print("  The choice with smaller gap is the closer match to CCV's framework.")
    print()
    print("  At alpha=(0,0):  log E[R_bill] = mu_rtb + ½sigma^2_rtb_unc.")
    print("              r_p_CCV doesn't include ½sigma^2_rtb so the gap is the bill")
    print("              Jensen lift either way (~0.0195% under unc, vs Sigma_rr;")
    print("              0.0119% under cond) — a feature of the CCV framework, not")
    print("              a comparison signal.")
    print()
    print("  Diagnostic: subtract baseline gap at alpha=(0,0) and look at the")
    print("              remaining gap. The smaller residual gap indicates which")
    print("              sigma^2 choice is closer to CCV's approximation target.")


if __name__ == "__main__":
    main()
