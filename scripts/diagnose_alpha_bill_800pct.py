"""Direct verification of the 'Newton converged to local min' hypothesis
at the alpha_b = -8.5 production policy.

For the broken benchmark config, at the mean state cell, evaluates the
terminal-age bequest FOC across a sweep of (alpha_s, alpha_b). For each
candidate critical point, computes the analytic Hessian (J_ss, J_bb, J_sb)
and reports whether it is a local MAX (J_ss<0, det>0), local MIN
(J_ss>0, det>0), or SADDLE (det<0).

Also computes the actual EU = E[u_bequest(s*R_p)] across the sweep so we
can see directly whether the broken policy sits at a critical point that
is NOT the global max.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute


# Reproduce the production benchmark config exactly
disc = CANONICAL_DISC._replace(
    wealth_min=0.05,
    state_grid_sizes=(6, 6, 6, 6),
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_stds=2.25,
    n_eta_nodes=3,
    n_state_quad_nodes=(3, 3, 3, 5),
    state_lobatto_Z=(None, None, None, 2.93),
    n_ret_nodes_1d=(3, 3),
    ret_lobatto_Z=None,
)
solver_cfg = CANONICAL_SOLVER._replace(
    delta_bequest=0.0,                # the suspect setting
    wealth_dynamics_spec="ccv_log",
    max_iter=100,
    gather_precision="f32",
)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=False)

print(f"State quadrature: {pc.n_state_quad} joint nodes "
      f"({disc.n_state_quad_nodes}, lobatto Z={disc.state_lobatto_Z})")
print(f"Ret quadrature  : {pc.n_ret_quad} joint nodes")

# Mean state cell
from scipy.linalg import solve_discrete_lyapunov
mu_state = np.linalg.solve(np.eye(4) - np.array(model.Phi_11),
                            np.array(model.Phi_0_state))
state_grid = np.array(pc.state_grid)
i_s = int(np.argmin(np.linalg.norm(state_grid - mu_state[None, :], axis=1)))
print(f"Mean cell i_s = {i_s},  state = {state_grid[i_s]}")

# Build the per-i_s log-return tensors
v_nodes = np.array(pc.v_nodes)
v_weights = np.array(pc.v_weights)
ret_nodes = np.array(pc.ret_nodes)
ret_weights = np.array(pc.ret_weights)
M_v_nodes = np.array(pc.M_v_nodes)
const_r = np.array(model.Phi_0_ret)
A_r = np.array(model.Phi_21)
Phi_0_state = np.array(model.Phi_0_state)
Phi_11 = np.array(model.Phi_11)
rtb_idx = int(model.rtb_index_in_state)
xr_pos = list(model.ret_names).index("xr")
xb_pos = list(model.ret_names).index("xb")

state_i = state_grid[i_s]
base_mu_r = const_r + A_r @ state_i
mu_r_per = base_mu_r[None, :] + M_v_nodes
s_next_4d = Phi_0_state[None, :] + state_i @ Phi_11.T + v_nodes
log_R_bill_kv = s_next_4d[:, rtb_idx]
n_sq = v_nodes.shape[0]
n_rq = ret_nodes.shape[0]
log_R_bill = np.broadcast_to(log_R_bill_kv[:, None], (n_sq, n_rq))
log_x_s = mu_r_per[:, xr_pos:xr_pos+1] + ret_nodes[None, :, xr_pos]
log_x_b = mu_r_per[:, xb_pos:xb_pos+1] + ret_nodes[None, :, xb_pos]
weight_kv_kr = v_weights[:, None] * ret_weights[None, :]

sigma2_xr = float(np.array(model.Sigma_rr)[xr_pos, xr_pos])
sigma2_xb = float(np.array(model.Sigma_rr)[xb_pos, xb_pos])
sigma_xrxb = float(np.array(model.Sigma_rr)[xr_pos, xb_pos])
gamma = float(model.gamma)
b_bar = float(model.b_bar)

# Annuity factor at this cell (terminal-age bequest setup)
A_is = float(np.array(pc.annuity_factors)[i_s])
print(f"A_is at this cell = {A_is:.4f}")

# Choose savings level matching the probe cell (W=6.20, c~0.94, s~5.26)
s_val = 5.26
delta = 0.0   # the suspect setting
print(f"s_val = {s_val},  delta_bequest = {delta},  gamma = {gamma},  b_bar = {b_bar}")
print()


def _ccv_log_return_and_grad(a_s, a_b, log_Rb, log_xs, log_xb,
                              s2xr, s2xb, sxrxb):
    r_p = (log_Rb + a_s * log_xs + a_b * log_xb
           + 0.5 * (a_s * s2xr + a_b * s2xb)
           - 0.5 * (a_s*a_s*s2xr + 2*a_s*a_b*sxrxb + a_b*a_b*s2xb))
    R_p = np.exp(r_p)
    dr_da_s = log_xs + s2xr * (0.5 - a_s) - a_b * sxrxb
    dr_da_b = log_xb + s2xb * (0.5 - a_b) - a_s * sxrxb
    return R_p, dr_da_s, dr_da_b


def terminal_foc_jac(a_s, a_b, s_val, A_is, delta):
    """Reproduces solver.py:terminal_foc_jac_ccv exactly."""
    R_p, dr_das, dr_dab = _ccv_log_return_and_grad(
        a_s, a_b, log_R_bill, log_x_s, log_x_b,
        sigma2_xr, sigma2_xb, sigma_xrxb,
    )
    sR_p = s_val * R_p
    # bequest_mu_and_mup
    C_bar = sR_p / A_is + delta
    mu = b_bar * C_bar ** (-gamma) / A_is
    mup = -gamma * mu / (A_is * C_bar)

    dRp_das = R_p * dr_das
    dRp_dab = R_p * dr_dab
    wmu = weight_kv_kr * mu
    wmup = weight_kv_kr * mup

    foc_s = float(np.sum(wmu * dRp_das))
    foc_b = float(np.sum(wmu * dRp_dab))
    V_dot = float(np.sum(wmu * R_p))

    jac_lin = wmup * s_val
    extra_ss = wmu * R_p * (dr_das*dr_das - sigma2_xr)
    extra_bb = wmu * R_p * (dr_dab*dr_dab - sigma2_xb)
    extra_sb = wmu * R_p * (dr_das*dr_dab - sigma_xrxb)
    J_ss = float(np.sum(jac_lin * dRp_das*dRp_das + extra_ss))
    J_bb = float(np.sum(jac_lin * dRp_dab*dRp_dab + extra_bb))
    J_sb = float(np.sum(jac_lin * dRp_das*dRp_dab + extra_sb))
    return foc_s, foc_b, J_ss, J_bb, J_sb, V_dot


def expected_bequest_utility(a_s, a_b, s_val, A_is, delta):
    """E[u_bq(s*R_p)] under the quadrature."""
    R_p, _, _ = _ccv_log_return_and_grad(
        a_s, a_b, log_R_bill, log_x_s, log_x_b,
        sigma2_xr, sigma2_xb, sigma_xrxb,
    )
    sR_p = s_val * R_p
    C_bar = sR_p / A_is + delta
    u_bq = b_bar * C_bar ** (1.0 - gamma) / (1.0 - gamma)
    return float(np.sum(weight_kv_kr * u_bq))


# ============================================================================
# 1. Direct check at production-reported broken policy
# ============================================================================
print("=" * 90)
print("1. AT THE PRODUCTION POLICY (alpha_s=0.014, alpha_b=-8.054):")
print("=" * 90)
for label, a_s, a_b in [
    ("Production-reported broken policy", 0.014, -8.054),
    ("Markowitz answer",                  0.385, 0.582),
    ("Cold init",                         0.85,  0.44),
]:
    fs, fb, Jss, Jbb, Jsb, Vd = terminal_foc_jac(a_s, a_b, s_val, A_is, delta)
    det = Jss * Jbb - Jsb * Jsb
    Eu = expected_bequest_utility(a_s, a_b, s_val, A_is, delta)
    err = np.sqrt(fs*fs + fb*fb)
    print(f"\n  {label}: alpha = ({a_s:+.3f}, {a_b:+.3f})")
    print(f"    foc_s        = {fs:+.4e}    foc_b = {fb:+.4e}    ||foc|| = {err:.4e}")
    print(f"    J_ss         = {Jss:+.4e}    J_bb  = {Jbb:+.4e}    J_sb = {Jsb:+.4e}")
    print(f"    det(J)       = {det:+.4e}")
    if det > 0:
        if Jss < 0:
            kind = "LOCAL MAX (negative-definite Hessian)"
        else:
            kind = "LOCAL MIN (positive-definite Hessian)"
    else:
        kind = "SADDLE POINT (indefinite Hessian)"
    print(f"    classification = {kind}")
    print(f"    E[u_bequest] = {Eu:+.4e}  (more positive = better)")

# ============================================================================
# 2. Sweep alpha_b at fixed alpha_s = 0.014. Look for zero crossings of foc_b.
# ============================================================================
print()
print("=" * 90)
print("2. SWEEP alpha_b at fixed alpha_s = 0.014, delta = 0")
print("=" * 90)
print(f"  {'alpha_b':>8}  {'foc_b':>14}  {'J_bb':>14}  {'EU':>14}")
ab_grid = np.linspace(-12.0, 5.0, 180)
foc_b_arr = np.zeros_like(ab_grid)
J_bb_arr = np.zeros_like(ab_grid)
EU_arr = np.zeros_like(ab_grid)
for i, ab in enumerate(ab_grid):
    fs, fb, Jss, Jbb, Jsb, Vd = terminal_foc_jac(0.014, ab, s_val, A_is, delta)
    foc_b_arr[i] = fb
    J_bb_arr[i] = Jbb
    EU_arr[i] = expected_bequest_utility(0.014, ab, s_val, A_is, delta)

sign = np.sign(foc_b_arr)
crossings = []
for i in range(len(ab_grid) - 1):
    if sign[i] * sign[i+1] < 0:
        ab_z = ab_grid[i] - foc_b_arr[i] * (ab_grid[i+1] - ab_grid[i]) / (foc_b_arr[i+1] - foc_b_arr[i])
        Jbb_z = 0.5 * (J_bb_arr[i] + J_bb_arr[i+1])
        EU_z = 0.5 * (EU_arr[i] + EU_arr[i+1])
        crossings.append((ab_z, Jbb_z, EU_z))

for i in range(0, len(ab_grid), 12):
    print(f"  {ab_grid[i]:+8.3f}  {foc_b_arr[i]:+14.3e}  {J_bb_arr[i]:+14.3e}  {EU_arr[i]:+14.4e}")

print(f"\nZero crossings of foc_b at delta=0, alpha_s=0.014:")
for ab_z, Jbb_z, EU_z in crossings:
    sign_J = "MAX" if Jbb_z < 0 else "MIN"
    print(f"  alpha_b = {ab_z:+.4f}   J_bb ~ {Jbb_z:+.3e}   ({sign_J})   EU = {EU_z:+.4e}")

i_best_eu = int(np.argmax(EU_arr))
print(f"\nMax EU on this slice: alpha_b = {ab_grid[i_best_eu]:+.3f}, EU = {EU_arr[i_best_eu]:+.4e}")

# ============================================================================
# 3. Simulated Newton from cold init
# ============================================================================
print()
print("=" * 90)
print(f"3. SIMULATED NEWTON: cold init (0.85, 0.44), s_val={s_val}, delta={delta}")
print("   (Same backtracking line search as production)")
print("=" * 90)

a_s, a_b = 0.85, 0.44
fs, fb, Jss, Jbb, Jsb, Vd = terminal_foc_jac(a_s, a_b, s_val, A_is, delta)
err = np.sqrt(fs*fs + fb*fb)
scale = max(abs(Vd), 1e-30)   # matches solver.py:1158
tol = 1e-7
max_iter = 30
line_search_max_step = 2.0
print(f"  iter=0  alpha=({a_s:+.4f}, {a_b:+.4f})  err={err:.3e}  scale={scale:.3e}  tol*scale={tol*scale:.3e}")

for k in range(1, max_iter + 1):
    det = Jss * Jbb - Jsb * Jsb
    if abs(det) < 1e-15:
        print(f"  iter={k}  singular Jacobian, falling back to gradient step")
        break
    inv_d = 1.0 / det
    step_s = -(Jbb * fs - Jsb * fb) * inv_d
    step_b = -(-Jsb * fs + Jss * fb) * inv_d
    slen = np.sqrt(step_s*step_s + step_b*step_b)
    cap = min(1.0, line_search_max_step / max(slen, 1e-30))
    step_s *= cap
    step_b *= cap
    alpha_lr = 1.0
    a_s_t, a_b_t = a_s, a_b
    fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, Vd_t = fs, fb, Jss, Jbb, Jsb, Vd
    err_t = err
    for bt in range(11):
        a_s_try = a_s + alpha_lr * step_s
        a_b_try = a_b + alpha_lr * step_b
        fs_try, fb_try, Jss_try, Jbb_try, Jsb_try, Vd_try = terminal_foc_jac(a_s_try, a_b_try, s_val, A_is, delta)
        err_try = np.sqrt(fs_try*fs_try + fb_try*fb_try)
        if err_try < err:
            a_s_t, a_b_t = a_s_try, a_b_try
            fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, Vd_t = fs_try, fb_try, Jss_try, Jbb_try, Jsb_try, Vd_try
            err_t = err_try
            break
        alpha_lr *= 0.5
    a_s, a_b = a_s_t, a_b_t
    fs, fb, Jss, Jbb, Jsb, Vd = fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, Vd_t
    err = err_t
    print(f"  iter={k:>2}  alpha=({a_s:+.4f}, {a_b:+.4f})  err={err:.3e}  Jbb={Jbb:+.3e}  bt={bt}  step_b={step_b:+.4f}")
    if err < tol * scale:
        det = Jss * Jbb - Jsb * Jsb
        kind = "MAX" if Jss < 0 and det > 0 else ("MIN" if Jss > 0 and det > 0 else "SADDLE")
        print(f"  -> CONVERGED at iter {k}.  Hessian classification: {kind}")
        break

# ============================================================================
# 4. Repeat 1D sweep with delta = 0.005 (canonical default)
# ============================================================================
print()
print("=" * 90)
print("4. SAME 1D SWEEP WITH delta_bequest = 0.005 (canonical default)")
print("=" * 90)
delta_fix = 0.005
foc_b_arr2 = np.zeros_like(ab_grid)
EU_arr2 = np.zeros_like(ab_grid)
J_bb_arr2 = np.zeros_like(ab_grid)
for i, ab in enumerate(ab_grid):
    fs, fb, Jss, Jbb, Jsb, Vd = terminal_foc_jac(0.014, ab, s_val, A_is, delta_fix)
    foc_b_arr2[i] = fb
    J_bb_arr2[i] = Jbb
    EU_arr2[i] = expected_bequest_utility(0.014, ab, s_val, A_is, delta_fix)
sign2 = np.sign(foc_b_arr2)
crossings2 = []
for i in range(len(ab_grid) - 1):
    if sign2[i] * sign2[i+1] < 0:
        ab_z = ab_grid[i] - foc_b_arr2[i] * (ab_grid[i+1] - ab_grid[i]) / (foc_b_arr2[i+1] - foc_b_arr2[i])
        Jbb_z = 0.5 * (J_bb_arr2[i] + J_bb_arr2[i+1])
        EU_z = 0.5 * (EU_arr2[i] + EU_arr2[i+1])
        crossings2.append((ab_z, Jbb_z, EU_z))
print(f"Zero crossings of foc_b at delta=0.005, alpha_s=0.014:")
for ab_z, Jbb_z, EU_z in crossings2:
    sign_J = "MAX" if Jbb_z < 0 else "MIN"
    print(f"  alpha_b = {ab_z:+.4f}   J_bb ~ {Jbb_z:+.3e}   ({sign_J})   EU = {EU_z:+.4e}")

print()
print("=" * 90)
print("CONCLUSION")
print("=" * 90)
print(f"  foc_b zero crossings at delta=0    : {len(crossings)}")
print(f"  foc_b zero crossings at delta=0.005: {len(crossings2)}")
print()
print(f"  At each crossing, J_bb sign tells us MAX (J_bb<0) or MIN (J_bb>0):")
print(f"    delta=0:    {[('MAX' if c[1] < 0 else 'MIN') + f' at a_b={c[0]:+.3f}' for c in crossings]}")
print(f"    delta=.005: {[('MAX' if c[1] < 0 else 'MIN') + f' at a_b={c[0]:+.3f}' for c in crossings2]}")
print()
print("  If delta=0 has multiple crossings or any MIN, the singularity creates")
print("  spurious critical points (hypothesis confirmed).")
print("  If both have only one MAX, the hypothesis is WRONG and the bug is elsewhere.")
