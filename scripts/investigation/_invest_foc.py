"""Investigation: verify that saved plateau policy satisfies the working-age FOC."""
import numpy as np
import sys, json
sys.path.insert(0, '.')

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)
vc = meta['run_config']['var_config']
bc = meta['run_config']['base_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from discretization import (
    build_state_grid, get_state_quadrature, get_return_quadrature,
    discretize_income_ar1_mixture, get_eta_quadrature_mixture,
    get_eps_quadrature_corrected,
)
from model import disposable_income_working

Phi_11 = Phi_full[np.ix_([0,1,2],[0,1,2])]
Phi_21 = Phi_full[np.ix_([3,4,5],[0,1,2])]
Sigma_ss = Omega[np.ix_([0,1,2],[0,1,2])]
Sigma_rr = Omega[np.ix_([3,4,5],[3,4,5])]
Sigma_rs = Omega[np.ix_([3,4,5],[0,1,2])]
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
mm.rho = bc['rho']
mm.pz = bc['pz']
mm.mu_eta1 = bc['mu_eta1']
mm.sigma_eta1 = bc['sigma_eta1']
mm.mu_eta2 = bc['mu_eta2']
mm.sigma_eta2 = bc['sigma_eta2']
mm.pe = bc['pe']
mm.mu_eps1 = bc['mu_eps1']
mm.sigma_eps1 = bc['sigma_eps1']
mm.mu_eps2 = bc['mu_eps2']
mm.sigma_eps2 = bc['sigma_eps2']

v_nodes, v_weights = get_state_quadrature(mm, n_nodes=2)
ret_nodes, ret_weights = get_return_quadrature(mm, n_nodes=(3, 5, 3))
eta_nodes, eta_weights = get_eta_quadrature_mixture(mm, n_nodes=3)
eps_nodes, eps_weights = get_eps_quadrature_corrected(mm, n_nodes=3)

info = build_state_grid(N_vec=[5,5,5], mu_intercept=Phi_0_state,
                         Phi=Phi_11, Sigma_innov=Sigma_ss, n_stds=2.0, mode='principal')
state_grid = info['state_grid']
state_indices = info['state_indices']
state_bracket_grids = info['state_bracket_grids']
state_bracket_shift = info['bracket_shift']
state_bracket_L_inv = info['bracket_L_inv']

z_grid, _ = discretize_income_ar1_mixture(
    rho=bc['rho'], p=bc['pz'], mu1=bc['mu_eta1'], sigma1=bc['sigma_eta1'],
    mu2=bc['mu_eta2'], sigma2=bc['sigma_eta2'], N=9, n_stds=3.0,
)
dz = z_grid[1] - z_grid[0]

ages = np.arange(22, 100)
log_det = bc['b0'] + bc['b1']*ages + bc['b2']*ages**2/10 + bc['b3']*ages**3/100

wealth_grid = np.geomspace(0.0001, 200.0, 150)
d = np.load('saved_runs/unconstrained_principal_grid5x5x5_nz9/policy_arrays.npz')
S = d['S_mat']
B = d['B_mat']
C = d['C_mat']

# Build state_lookup: principal-coord index (i,j,k) -> flat index
state_lookup = {}
for m in range(125):
    key = (int(state_indices[m, 0]), int(state_indices[m, 1]), int(state_indices[m, 2]))
    state_lookup[key] = m


def bracket_state(s_next):
    coord = state_bracket_L_inv @ (s_next - state_bracket_shift)
    out = []
    for d in range(3):
        g = state_bracket_grids[d]
        n = len(g)
        x = coord[d]
        if x <= g[0]:
            out.append((0, 0.0))
            continue
        if x >= g[-1]:
            out.append((n - 2, 1.0))
            continue
        idx = int(np.searchsorted(g, x) - 1)
        idx = max(0, min(idx, n - 2))
        f = (x - g[idx]) / (g[idx + 1] - g[idx])
        f = max(0.0, min(f, 1.0))
        out.append((idx, f))
    return out


def interp_c(t_next, z_next, s_next, x_next):
    bracket = bracket_state(s_next)
    (i0, f0), (i1, f1), (i2, f2) = bracket
    iz_lo = int(np.searchsorted(z_grid, z_next) - 1)
    iz_lo = max(0, min(iz_lo, len(z_grid) - 2))
    frac_z = (z_next - z_grid[iz_lo]) / dz
    frac_z = max(0.0, min(frac_z, 1.0))
    iw = int(np.searchsorted(wealth_grid, x_next) - 1)
    iw = max(0, min(iw, len(wealth_grid) - 2))
    fw = (x_next - wealth_grid[iw]) / (wealth_grid[iw + 1] - wealth_grid[iw])
    fw = max(0.0, min(fw, 1.0))

    val = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                key = (i0 + di, i1 + dj, i2 + dk)
                flat = state_lookup[key]
                c_lo = (1 - frac_z) * C[t_next, iz_lo, flat, iw] + frac_z * C[t_next, iz_lo + 1, flat, iw]
                c_hi = (1 - frac_z) * C[t_next, iz_lo, flat, iw + 1] + frac_z * C[t_next, iz_lo + 1, flat, iw + 1]
                c_xy = (1 - fw) * c_lo + fw * c_hi
                wts = ((1 - f0) if di == 0 else f0) * ((1 - f1) if dj == 0 else f1) * ((1 - f2) if dk == 0 else f2)
                val += wts * c_xy
    return max(val, 1e-10)


def working_foc(alpha_s, alpha_b, t, iz, i_s, iw, beta=0.96, gamma=3.0, psi=0.999):
    sv = state_grid[i_s]
    base_mu_r = Phi_0_ret + Phi_21 @ sv
    z_curr = z_grid[iz]
    log_det_next = log_det[t + 1]
    W_curr = wealth_grid[iw]
    c_curr = C[t, iz, i_s, iw]
    s_val = W_curr - c_curr
    a_bill = 1.0 - alpha_s - alpha_b

    foc_s = 0.0
    foc_b = 0.0

    for kv in range(v_nodes.shape[0]):
        mu_r_k = base_mu_r + M @ v_nodes[kv]
        s_next = Phi_0_state + Phi_11 @ sv + v_nodes[kv]

        for kr in range(ret_nodes.shape[0]):
            r_log = mu_r_k + ret_nodes[kr]
            R_bill = np.exp(r_log[0])
            R_s = np.exp(r_log[0] + r_log[1])
            R_b = np.exp(r_log[0] + r_log[2])
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            for ke in range(eta_nodes.shape[0]):
                z_next = mm.rho * z_curr + eta_nodes[ke]

                for je in range(eps_nodes.shape[0]):
                    y_gross = np.exp(log_det_next + z_next + eps_nodes[je])
                    income_next = float(disposable_income_working(np.array([y_gross]))[0])

                    x_next = max(s_val * R_p + income_next, 1e-10)
                    c_next = interp_c(t + 1, z_next, s_next, x_next)
                    mu = c_next ** (-gamma)
                    w = v_weights[kv] * ret_weights[kr] * eta_weights[ke] * eps_weights[je]
                    foc_s += w * mu * (R_s - R_bill)
                    foc_b += w * mu * (R_b - R_bill)
    return foc_s, foc_b


# Cell of interest
t = 0  # age 22
iz = 4
i_s = 62
iw = 149  # W = 200

s_pol = float(S[t, iz, i_s, iw])
b_pol = float(B[t, iz, i_s, iw])
c_pol = float(C[t, iz, i_s, iw])
W = float(wealth_grid[iw])

print(f"Cell: age=22, iz={iz}, state={i_s}, W={W}")
print(f"  saved: alpha_s={s_pol:.4f}, alpha_b={b_pol:.4f}, alpha_bill={1-s_pol-b_pol:.4f}")
print(f"  c={c_pol:.4f}, c/W={c_pol/W:.6f}, savings={W-c_pol:.4f}")

# Verify FOC at saved
print()
print("FOC at saved policy:")
foc_s, foc_b = working_foc(s_pol, b_pol, t, iz, i_s, iw)
print(f"  FOC_s = {foc_s:+.6e}")
print(f"  FOC_b = {foc_b:+.6e}")

# Perturb alpha_s
print()
print("Perturbations of alpha_s (alpha_b fixed at saved):")
print("  alpha_s     FOC_s        FOC_b")
for delta in [-1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0]:
    fs, fb = working_foc(s_pol + delta, b_pol, t, iz, i_s, iw)
    print(f"  {s_pol+delta:7.3f}  {fs:+.4e}  {fb:+.4e}")

# Perturb alpha_b
print()
print("Perturbations of alpha_b (alpha_s fixed at saved):")
print("  alpha_b     FOC_s        FOC_b")
for delta in [-1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0]:
    fs, fb = working_foc(s_pol, b_pol + delta, t, iz, i_s, iw)
    print(f"  {b_pol+delta:7.3f}  {fs:+.4e}  {fb:+.4e}")
