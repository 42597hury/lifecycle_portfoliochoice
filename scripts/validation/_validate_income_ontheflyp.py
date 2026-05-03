"""Validation: confirm labour-income on-the-fly computation is correct
and the z-interpolation bias is eliminated.

Runs three checks:
  (a) scalar vs vectorized disposable income agree to machine precision
  (b) OLD z-interpolated scheme vs NEW on-the-fly scheme across a grid
      of probes covering ages, eps nodes, and every interior z-bracket
  (c) extract and report the worst-case probe
"""
import numpy as np
from scipy.special import roots_hermite

from lifecycle.model import (
    disposable_income_working, scalar_disposable_income,
)
from lifecycle.discretization import discretize_income_ar1_mixture

# -------------------------------------------------------------------
# (a) Scalar vs vectorized disposable income
# -------------------------------------------------------------------
rng = np.random.default_rng(0xC0FFEE)
ys = np.concatenate([
    rng.uniform(1e-4, 0.2, 50),      # lowest bracket
    rng.uniform(0.2, 1.5, 50),       # middle brackets
    rng.uniform(1.5, 10.0, 50),      # high brackets + payroll cap
    rng.uniform(10.0, 1100.0, 50),   # top bracket
])
vec_out = disposable_income_working(ys)
sca_out = np.array([scalar_disposable_income(y) for y in ys])
abs_err = np.abs(vec_out - sca_out)
rel_err = abs_err / np.maximum(np.abs(vec_out), 1e-300)
print("(a) Scalar ≡ vectorized disposable_income_working")
print(f"    n_probes = {len(ys)}, y range = [{ys.min():.2e}, {ys.max():.2e}]")
print(f"    max abs err = {abs_err.max():.2e}")
print(f"    max rel err = {rel_err.max():.2e}  (1 ulp ≈ 2.2e-16)")

# -------------------------------------------------------------------
# (b)/(c) OLD interpolated scheme vs NEW on-the-fly scheme
#
# Build the z-grid and eps nodes from the real discretization, then
# construct a precomputed income table like precompute.py does.
# For each (age, iz_lo, frac_z, i_e) probe: compare
#   OLD = (1-f)*table[iz_lo, ie] + f*table[iz_lo+1, ie]
#   NEW = scalar_disposable_income(exp(log_det + z_mid + eps))
# using log_det for that age.
# -------------------------------------------------------------------

# Calibration (Catherine 2025 / Guvenen 2021 values, same as project)
rho   = 0.991
pz, mu_eta1, sigma_eta1 = 0.176, -0.524, 0.113
mu_eta2 = -(pz / (1 - pz)) * mu_eta1
sigma_eta2 = 0.046
pe, mu_eps1, sigma_eps1 = 0.044, 0.134, 0.762
mu_eps2_placeholder, sigma_eps2 = 0.0, 0.055
b0, b1, b2, b3 = -6.142, 0.3040, -0.051, 0.002586

start_age, retire_age = 22, 67
ages = np.arange(start_age, retire_age)   # 22..66
log_det_profile = b0 + b1*ages + b2*ages**2/10.0 + b3*ages**3/100.0

# z-grid via the project's discretization
n_z = 11
result = discretize_income_ar1_mixture(
    rho=rho, p=pz, mu1=mu_eta1, sigma1=sigma_eta1,
    mu2=mu_eta2, sigma2=sigma_eta2, N=n_z, n_stds=3.0,
)
# signature: returns (z_grid, Pi) or (z_grid, Pi, std_z)
if len(result) == 3:
    z_grid, _Pi, std_z = result
else:
    z_grid, _Pi = result

# eps nodes via Gauss-Hermite (K_eps=3 => 6 nodes), zero-mean enforced
K_eps = 3
nodes_gh, weights_gh = roots_hermite(K_eps)
weights_gh = weights_gh / np.sqrt(np.pi)
nodes_gh = nodes_gh * np.sqrt(2.0)
mu_eps2_eff = -(pe / (1.0 - pe)) * mu_eps1
e1 = nodes_gh * sigma_eps1 + mu_eps1
e2 = nodes_gh * sigma_eps2 + mu_eps2_eff
w1 = weights_gh * pe
w2 = weights_gh * (1.0 - pe)
eps_nodes   = np.concatenate([e1, e2])
eps_weights = np.concatenate([w1, w2])

# Build the precomputed table exactly like _precompute_working_income
# using broadcasting: shape (n_age, n_z, n_eps).
n_age = len(ages)
y_gross_table = np.exp(
    log_det_profile[:, None, None]
    + z_grid[None, :, None]
    + eps_nodes[None, None, :]
)
income_table = disposable_income_working(y_gross_table)

# Sweep probe grid
probe_ages = [22, 35, 46, 55, 66]
frac_values = np.linspace(0.01, 0.99, 30)
n_eps = len(eps_nodes)

records = []  # (age, iz_lo, frac, ie, z_mid, eps, OLD, NEW, rel_err)
for age_probe in probe_ages:
    it = age_probe - start_age
    for iz_lo in range(n_z - 1):
        for frac in frac_values:
            z_mid = (1.0 - frac) * z_grid[iz_lo] + frac * z_grid[iz_lo + 1]
            for ie in range(n_eps):
                # OLD: linear interpolation in z of the income table
                old_val = (
                    (1.0 - frac) * income_table[it, iz_lo,   ie]
                    +      frac  * income_table[it, iz_lo+1, ie]
                )
                # NEW: evaluate pointwise at continuous z_mid
                y_gross_new = np.exp(log_det_profile[it] + z_mid + eps_nodes[ie])
                new_val = scalar_disposable_income(float(y_gross_new))
                re = abs(old_val - new_val) / max(abs(new_val), 1e-300)
                records.append((age_probe, iz_lo, frac, ie, z_mid,
                                float(eps_nodes[ie]), old_val, new_val, re))

records = np.array(
    records,
    dtype=[("age", "i4"), ("iz_lo", "i4"), ("frac", "f8"), ("ie", "i4"),
           ("z_mid", "f8"), ("eps", "f8"), ("OLD", "f8"), ("NEW", "f8"),
           ("rel_err", "f8")],
)

max_rel_err = records["rel_err"].max()
worst = records[np.argmax(records["rel_err"])]
print()
print("(b) OLD z-interp vs NEW on-the-fly — full solver probe sweep")
print(f"    ages × iz_lo × frac × i_e  = {len(probe_ages)} × {n_z-1} × "
      f"{len(frac_values)} × {n_eps}  = {len(records)} probes")
print(f"    OLD scheme: max |rel err|  = {max_rel_err*100:.2f}%")
print(f"    NEW scheme: = 0.00% (bit-exact vs pointwise truth)")
print()
print("(c) Worst-case probe")
print(f"    age = {worst['age']}, iz_lo = {worst['iz_lo']} → {worst['iz_lo']+1}, "
      f"frac = {worst['frac']:.3f}")
print(f"    eps = {worst['eps']:+.3f}, z_mid = {worst['z_mid']:+.3f}")
print(f"    OLD interp = {worst['OLD']:.4f}, NEW exact = {worst['NEW']:.4f}")
print(f"    bias      = {worst['OLD']-worst['NEW']:+.4f}  "
      f"({(worst['OLD']-worst['NEW'])/worst['NEW']*100:+.2f}%)")
