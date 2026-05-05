"""Compute the sigma-coverage and per-step spacing for state_grid=(9,9,9), n_stds=2.0."""
import json
import numpy as np
import sys
sys.path.insert(0, '.')

with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as f:
    meta = json.load(f)
vc = meta['run_config']['var_config']
z_bar = np.array(vc['z_bar']['values'])
Phi_full = np.array(vc['Phi']['values'])
Omega = np.array(vc['Omega']['values'])

from lifecycle.discretization import build_state_grid, stationary_covariance
from scipy.special import roots_hermite

# Honor the saved bundle's state ordering (may be legacy (0,1,2)=y_1/spr/cy or
# default (2,1,0)=cy/spr/y_1 after the 2026-04-30 reorder).
import json as _json
with open('saved_runs/unconstrained_principal_grid5x5x5_nz9/metadata.json') as _f:
    _vc = _json.load(_f)['run_config']['var_config']
_state_idx = list(_vc.get('state_indices', [0, 1, 2]))
_var_names = ['y_1', 'spr', 'cy', 'rtb', 'xr', 'xb']
names = [_var_names[i] for i in _state_idx]

Phi_11 = Phi_full[np.ix_(_state_idx, _state_idx)]
Sigma_ss = Omega[np.ix_(_state_idx, _state_idx)]
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar[_state_idx]
sigma_innov = np.sqrt(np.diag(Sigma_ss))
mu_s = z_bar[_state_idx]
Sigma_z = stationary_covariance(Phi_11, Sigma_ss)
sigma_stat = np.sqrt(np.diag(Sigma_z))

# Build (9, 9, 9) grid, n_stds=2
info_999 = build_state_grid(N_vec=[9,9,9], mu_intercept=Phi_0_state,
                            Phi=Phi_11, Sigma_innov=Sigma_ss,
                            n_stds=2.0, mode='cholesky')
state_grid_999 = info_999['state_grid']

# Build (5, 5, 5) grid for reference
info_555 = build_state_grid(N_vec=[5,5,5], mu_intercept=Phi_0_state,
                            Phi=Phi_11, Sigma_innov=Sigma_ss,
                            n_stds=2.0, mode='cholesky')
state_grid_555 = info_555['state_grid']

print("=" * 90)
print("STATE GRID (9,9,9) at n_stds=2.0 — physical extents and sigma coverage")
print("=" * 90)
print(f"  {'var':<5}  {'min phys':>10}  {'max phys':>10}  "
      f"{'+/- in stat sig':>17}  {'+/- in innov sig':>18}  {'#unique grid pts':>17}")
for d in range(3):
    smin, smax = state_grid_999[:, d].min(), state_grid_999[:, d].max()
    dmax = max(smax - mu_s[d], mu_s[d] - smin)
    n_unique = len(np.unique(np.round(state_grid_999[:, d], 6)))
    print(f"  {names[d]:<5}  {smin:>+10.5f}  {smax:>+10.5f}  "
          f"+/- {dmax/sigma_stat[d]:>5.2f}            "
          f"+/- {dmax/sigma_innov[d]:>5.2f}             {n_unique:>5}")

print()
print("=" * 90)
print("PER-STEP SPACING (in innov sigma, by physical axis)")
print("=" * 90)
print("  Note: cholesky axes mix all 3 physical vars, so per-physical-axis spacing")
print("  varies depending on which cholesky-coord step you make.")
print()
for d in range(3):
    vals = np.unique(np.round(state_grid_999[:, d], 6))
    diffs = np.diff(vals)
    print(f"  {names[d]:<5}: {len(vals)} unique values across grid")
    if len(diffs) > 0:
        print(f"           per-step (innov sigma): "
              f"min={diffs.min()/sigma_innov[d]:.2f}, "
              f"median={np.median(diffs)/sigma_innov[d]:.2f}, "
              f"max={diffs.max()/sigma_innov[d]:.2f}")
        print(f"           per-step (stat sigma) : "
              f"min={diffs.min()/sigma_stat[d]:.2f}, "
              f"median={np.median(diffs)/sigma_stat[d]:.2f}, "
              f"max={diffs.max()/sigma_stat[d]:.2f}")
print()

print("=" * 90)
print("STATE QUADRATURE n_state_quad_nodes=3 — innovation-space sampling")
print("=" * 90)
nodes_1d, weights_1d = roots_hermite(3)
weights_1d = weights_1d / np.sqrt(np.pi)
nodes_1d = nodes_1d * np.sqrt(2.0)
print(f"  1D Gauss-Hermite nodes (in standardized N(0,1) units): {nodes_1d}")
print(f"  1D weights (sum to 1): {weights_1d}")
print(f"  Total joint nodes for v^s: {len(nodes_1d)**3} (= 3^3)")
print()
print("  Innovation v^s ~ N(0, Sigma_ss). After Cholesky transform, the joint nodes")
print("  span the v^s distribution at +/-{:.3f} sigma along each cholesky innovation axis.".format(nodes_1d.max()))
print()
print("  Compared to K=2 (production, prior bundle): nodes at +/- 1.0 sigma standardized.")
print("  K=3 reaches +/- 1.225 sigma — modestly wider but symmetric weighting includes")
print("  a center node at 0 (weight 0.667), so effective tail probability is small.")

print()
print("=" * 90)
print("COMPARISON: (5,5,5) vs (9,9,9) — same n_stds, same outer extent")
print("=" * 90)
print(f"  {'var':<5}  {'5x5x5 phys range':>22}  {'9x9x9 phys range':>22}")
for d in range(3):
    r5 = (state_grid_555[:, d].min(), state_grid_555[:, d].max())
    r9 = (state_grid_999[:, d].min(), state_grid_999[:, d].max())
    print(f"  {names[d]:<5}  [{r5[0]:>+8.5f}, {r5[1]:>+8.5f}]  [{r9[0]:>+8.5f}, {r9[1]:>+8.5f}]")
print()
print("  (Outer extents are identical by construction; (9,9,9) only refines the interior.)")

# Density ratio
print()
print("=" * 90)
print("RETURN QUADRATURE n_ret_nodes_1d=(3, 7, 5) — bond-axis improvement")
print("=" * 90)
print("  Indexing: n_ret_nodes_1d[0]=K_rtb=3, [1]=K_xr=7 (stock), [2]=K_xb=5 (bond).")
print("  Compared to production (3, 5, 3): K_xr 5->7, K_xb 3->5.")
print("  T11 finding: at corner state i_s=20, opt_b dropped 91 -> 60 -> 54 -> 47 as K_xb went 3->5->7->9.")
print("  So K_xb=5 is a partial mitigation; expect corner-state α_b roughly 30-50% smaller than production.")
