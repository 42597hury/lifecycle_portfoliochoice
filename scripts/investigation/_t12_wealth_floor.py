"""T12: wealth-floor sensitivity at saved policy.

For each cell, evaluate FOC at saved (alpha_s, alpha_b) under different
wealth-floor values. The saved policy is fixed; only the c_next interp
clamping at low x changes. Tests H9 (wealth floor as implicit boundary softener).
"""
import json
import sys
import numpy as np

sys.path.insert(0, '.')

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)

vc = meta['run_config']['var_config']
bc = meta['run_config']['base_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from lifecycle.discretization import (
    build_state_grid, get_state_quadrature, get_return_quadrature,
    discretize_income_ar1_mixture, get_eta_quadrature_mixture,
    get_eps_quadrature_corrected,
)
from lifecycle.model import disposable_income_working

Phi_11 = Phi_full[np.ix_([0, 1, 2], [0, 1, 2])]
Phi_21 = Phi_full[np.ix_([3, 4, 5], [0, 1, 2])]
Sigma_ss = Omega[np.ix_([0, 1, 2], [0, 1, 2])]
Sigma_rr = Omega[np.ix_([3, 4, 5], [3, 4, 5])]
Sigma_rs = Omega[np.ix_([3, 4, 5], [0, 1, 2])]
M = Sigma_rs @ np.linalg.inv(Sigma_ss)
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[:3]
Phi_0_ret = z_bar[3:] - Phi_21 @ z_bar[:3]


class MM:
    pass


mm = MM()
mm.n_state = 3
mm.n_ret = 3
mm.Phi_0_state = Phi_0_state
mm.Phi_0_ret = Phi_0_ret
mm.Phi_11 = Phi_11
mm.Phi_21 = Phi_21
mm.Sigma_ss = Sigma_ss
mm.Sigma_rr = Sigma_rr
mm.Sigma_rs = Sigma_rs
mm.M = M
mm.Sigma_r_cond = Sigma_rr - M @ Sigma_rs.T
for k in ['rho', 'pz', 'mu_eta1', 'sigma_eta1', 'mu_eta2', 'sigma_eta2',
          'pe', 'mu_eps1', 'sigma_eps1', 'mu_eps2', 'sigma_eps2']:
    setattr(mm, k, bc[k])

eta_nodes, eta_weights = get_eta_quadrature_mixture(mm, n_nodes=3)
eps_nodes, eps_weights = get_eps_quadrature_corrected(mm, n_nodes=3)
v_nodes, v_weights = get_state_quadrature(mm, n_nodes=2)
ret_nodes, ret_weights = get_return_quadrature(mm, n_nodes=(3, 5, 3))

info = build_state_grid(N_vec=[5, 5, 5], mu_intercept=Phi_0_state,
                       Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='cholesky')
state_grid = info['state_grid']
state_indices = info['state_indices']

state_lookup_3d = np.full((5, 5, 5), -1, dtype=np.int64)
for m in range(125):
    state_lookup_3d[state_indices[m, 0], state_indices[m, 1], state_indices[m, 2]] = m

n_stds = 2.0
N_state_pts = 5
g0 = -n_stds
dx = 2.0 * n_stds / (N_state_pts - 1)

z_grid, _ = discretize_income_ar1_mixture(
    rho=bc['rho'], p=bc['pz'], mu1=bc['mu_eta1'], sigma1=bc['sigma_eta1'],
    mu2=bc['mu_eta2'], sigma2=bc['sigma_eta2'], N=9, n_stds=3.0,
)
n_z = len(z_grid)
dz = z_grid[1] - z_grid[0]

ages = np.arange(22, 100)
log_det = bc['b0'] + bc['b1']*ages + bc['b2']*ages**2/10 + bc['b3']*ages**3/100

wealth_grid = np.geomspace(0.0001, 200.0, 150)
n_w = len(wealth_grid)

d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S_pol = d['S_mat']
B_pol = d['B_mat']
C_full = d['C_mat']

state_bracket_shift = info['bracket_shift']
state_bracket_L_inv = info['bracket_L_inv']


def interp_c_batch_floor(t_next, z_arr, s_arr, x_arr, x_floor):
    """Interp C with x clamped to >= x_floor. Tests H9: bigger floor = more MU bounded."""
    N = z_arr.shape[0]
    coord = (s_arr - state_bracket_shift) @ state_bracket_L_inv.T
    raw_i = (coord - g0) / dx
    i_state = np.clip(np.floor(raw_i).astype(np.int64), 0, N_state_pts - 2)
    f_state = np.clip(raw_i - i_state, 0.0, 1.0)
    f0, f1, f2 = f_state[:, 0], f_state[:, 1], f_state[:, 2]

    iz_lo = np.clip(np.searchsorted(z_grid, z_arr) - 1, 0, n_z - 2)
    frac_z = np.clip((z_arr - z_grid[iz_lo]) / dz, 0.0, 1.0)

    x_eff = np.maximum(x_arr, x_floor)
    xc = np.clip(x_eff, wealth_grid[0], wealth_grid[-1])
    iw = np.clip(np.searchsorted(wealth_grid, xc) - 1, 0, n_w - 2)
    Wlo = wealth_grid[iw]
    Whi = wealth_grid[iw + 1]
    fw = np.clip((xc - Wlo) / (Whi - Wlo), 0.0, 1.0)

    out = np.zeros(N, dtype=np.float64)
    C_t = C_full[t_next]
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                flat = state_lookup_3d[i_state[:, 0] + di, i_state[:, 1] + dj, i_state[:, 2] + dk]
                c00 = C_t[iz_lo, flat, iw]
                c01 = C_t[iz_lo, flat, iw + 1]
                c10 = C_t[iz_lo + 1, flat, iw]
                c11 = C_t[iz_lo + 1, flat, iw + 1]
                c_lo = (1 - frac_z) * c00 + frac_z * c10
                c_hi = (1 - frac_z) * c01 + frac_z * c11
                c_xy = (1 - fw) * c_lo + fw * c_hi
                w = ((1 - f0) if di == 0 else f0) * \
                    ((1 - f1) if dj == 0 else f1) * \
                    ((1 - f2) if dk == 0 else f2)
                out += w * c_xy
    return np.maximum(out, 1e-10)


def compute_foc_with_floor(alpha_s, alpha_b, t, iz, i_s, iw, x_floor, gamma=3.0):
    sv = state_grid[i_s]
    base_mu_r = Phi_0_ret + Phi_21 @ sv
    z_curr = z_grid[iz]
    log_det_next = log_det[t + 1]
    W_curr = wealth_grid[iw]
    c_curr = C_full[t, iz, i_s, iw]
    s_val = W_curr - c_curr
    a_bill = 1.0 - alpha_s - alpha_b

    n_kv = v_nodes.shape[0]
    n_kr = ret_nodes.shape[0]
    n_ke = eta_nodes.shape[0]
    n_je = eps_nodes.shape[0]

    Mv = v_nodes @ M.T
    s_next_kv = (Phi_0_state + Phi_11 @ sv)[None, :] + v_nodes
    r_log = base_mu_r[None, None, :] + Mv[:, None, :] + ret_nodes[None, :, :]
    R_bill = np.exp(r_log[..., 0])
    R_s = np.exp(r_log[..., 0] + r_log[..., 1])
    R_b = np.exp(r_log[..., 0] + r_log[..., 2])
    R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
    excess_s = R_s - R_bill
    excess_b = R_b - R_bill

    z_next_ke = mm.rho * z_curr + eta_nodes
    log_y_next = log_det_next + z_next_ke[:, None] + eps_nodes[None, :]
    income_next_kj = disposable_income_working(np.exp(log_y_next).ravel()).reshape(n_ke, n_je)

    x_next = s_val * R_p[:, :, None, None] + income_next_kj[None, None, :, :]
    x_next = np.maximum(x_next, 1e-10)

    z_arr = np.broadcast_to(z_next_ke[None, None, :, None], (n_kv, n_kr, n_ke, n_je)).reshape(-1)
    s_arr = np.broadcast_to(s_next_kv[:, None, None, None, :], (n_kv, n_kr, n_ke, n_je, 3)).reshape(-1, 3)
    x_arr = x_next.reshape(-1)
    c_next = interp_c_batch_floor(t + 1, z_arr, s_arr, x_arr, x_floor).reshape(n_kv, n_kr, n_ke, n_je)
    mu = c_next ** (-gamma)

    w_full = v_weights[:, None, None, None] * ret_weights[None, :, None, None] * \
             eta_weights[None, None, :, None] * eps_weights[None, None, None, :]
    foc_s = (mu * excess_s[:, :, None, None] * w_full).sum()
    foc_b = (mu * excess_b[:, :, None, None] * w_full).sum()
    return foc_s, foc_b


def find_optimum_floor(t, iz, i_s, iw, x_floor, gamma=3.0, max_iter=40, tol=1e-7):
    alpha_s = S_pol[t, iz, i_s, iw]
    alpha_b = B_pol[t, iz, i_s, iw]
    h = 0.0005
    for it in range(max_iter):
        fs, fb = compute_foc_with_floor(alpha_s, alpha_b, t, iz, i_s, iw, x_floor, gamma)
        F = np.array([fs, fb])
        if np.linalg.norm(F) < tol:
            return alpha_s, alpha_b, it, np.linalg.norm(F)
        fs_pls, fb_pls = compute_foc_with_floor(alpha_s + h, alpha_b, t, iz, i_s, iw, x_floor, gamma)
        fs_plb, fb_plb = compute_foc_with_floor(alpha_s, alpha_b + h, t, iz, i_s, iw, x_floor, gamma)
        J = np.array([[(fs_pls - fs)/h, (fs_plb - fs)/h],
                      [(fb_pls - fb)/h, (fb_plb - fb)/h]])
        try:
            d_step = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            return alpha_s, alpha_b, it, np.linalg.norm(F)
        step = -d_step
        nrm = np.linalg.norm(step)
        if nrm > 5.0:
            step = step * (5.0 / nrm)
        alpha_s += step[0]
        alpha_b += step[1]
    fs, fb = compute_foc_with_floor(alpha_s, alpha_b, t, iz, i_s, iw, x_floor, gamma)
    return alpha_s, alpha_b, max_iter, np.linalg.norm([fs, fb])


CELLS = [
    ('median',           0, 4, 62, 149),
    ('extreme_bond_pos', 0, 6, 20, 128),
    ('extreme_stock',    0, 6, 4, 113),
    ('extreme_bond_2',   0, 6, 45, 128),
]

print("=" * 95)
print("T12 — WEALTH FLOOR SENSITIVITY (production K_state=2, K_ret=(3,5,3))")
print("=" * 95)
print(f"  {'cell':<22}  {'x_floor':>8}  {'FOC_s':>14}  {'FOC_b':>14}  "
      f"{'opt_s':>8}  {'opt_b':>8}  {'iter':>4}  {'resid':>9}")
for name, t, iz, i_s, iw in CELLS:
    a_save_s = S_pol[t, iz, i_s, iw]
    a_save_b = B_pol[t, iz, i_s, iw]
    print(f"  {name} (saved=({a_save_s:+.3f}, {a_save_b:+.3f})):")
    for x_floor in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        fs, fb = compute_foc_with_floor(a_save_s, a_save_b, t, iz, i_s, iw, x_floor)
        opt_s, opt_b, n_it, resid = find_optimum_floor(t, iz, i_s, iw, x_floor)
        print(f"  {'':<22}  {x_floor:>8.0e}  {fs:>+14.4e}  {fb:>+14.4e}  "
              f"{opt_s:>+8.3f}  {opt_b:>+8.3f}  {n_it:>4}  {resid:>9.2e}")
