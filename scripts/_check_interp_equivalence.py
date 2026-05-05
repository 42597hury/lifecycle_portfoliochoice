"""Verify that _interp_z_wealth_pre returns bit-identical results to
_interp_z_wealth when the Hermite z-basis is supplied consistently."""
import numpy as np
from lifecycle.solver import _interp_z_wealth, _interp_z_wealth_pre

rng = np.random.default_rng(42)

# Build a synthetic c_next_full of shape (n_z, n_state, n_w)
n_z, n_state, n_w = 11, 5, 21
c_next_full = rng.standard_normal((n_z, n_state, n_w)) ** 2 + 1.0
inv_dw = 1.0 / 1.0
min_c = 1e-10

n_trials = 5000
max_diff_c = 0.0
max_diff_mpc = 0.0

for trial in range(n_trials):
    j_s = rng.integers(0, n_state)
    iz_lo = rng.integers(0, n_z - 1)
    frac_z = float(rng.uniform(0.0, 1.0))
    iw = rng.integers(0, n_w - 1)
    frac_w = float(rng.uniform(-0.2, 1.2))  # may extrapolate
    use_cubic = bool((iz_lo >= 1) and (iz_lo + 2 < n_z) and (rng.random() > 0.3))

    f2 = frac_z * frac_z
    f3 = f2 * frac_z
    h00 = 2.0 * f3 - 3.0 * f2 + 1.0
    h10 = f3 - 2.0 * f2 + frac_z
    h01 = -2.0 * f3 + 3.0 * f2
    h11 = f3 - f2

    c_old, m_old = _interp_z_wealth(
        c_next_full, int(j_s), int(iz_lo), frac_z, int(iw), frac_w, inv_dw,
        n_z, use_cubic, min_c
    )
    c_new, m_new = _interp_z_wealth_pre(
        c_next_full, int(j_s), int(iz_lo), frac_z, h00, h10, h01, h11,
        int(iw), frac_w, inv_dw, use_cubic, min_c
    )

    max_diff_c = max(max_diff_c, abs(c_old - c_new))
    max_diff_mpc = max(max_diff_mpc, abs(m_old - m_new))

print(f"trials={n_trials}")
print(f"max |c_old - c_new|   = {max_diff_c:.3e}")
print(f"max |mpc_old - mpc_new| = {max_diff_mpc:.3e}")
assert max_diff_c == 0.0, "c values differ -- not bit identical"
assert max_diff_mpc == 0.0, "mpc values differ -- not bit identical"
print("OK: bit identical")
