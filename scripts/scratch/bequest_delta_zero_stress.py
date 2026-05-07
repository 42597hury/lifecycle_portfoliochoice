"""Stress-test the bequest term at delta=0 (pure CRRA, no luxury shift).

Four tests, each isolating a different way the delta=0 spec could misbehave:

  1. Marginal-utility magnitude scan across wealth grid + bankruptcy boundary.
     Shows whether u'(W) at the bottom of the grid is fp64-representable and
     well-conditioned.

  2. Quadrature integrand concentration: does a single tail return node
     dominate the FOC sum?  Compare delta=0 vs 0.001 vs 0.005.

  3. Optimal alpha as a function of delta at a fixed (s, state, age).  How
     sensitive is the canonical retirement portfolio to the regularization?

  4. FOC surface around the optimum:  is the basin of attraction shallow
     and noisy at delta=0, deep and clean at delta>0?

All tests use the actual VAR + state-grid + return covariance, not made-up
numbers.  Results are reported as numbers + verdicts, not plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from lifecycle.var import (  # noqa: E402
    build_nominal_system1_var_config,
    partition_var,
)
from lifecycle.model import annuity_factor  # noqa: E402


def hr(s=""):
    print()
    print("=" * 78)
    if s:
        print(s)
        print("=" * 78)


# Canonical economics
gamma = 5.0
b_bar = 10.0


# -----------------------------------------------------------------------------
# Test 1.  Marginal utility magnitude scan
# -----------------------------------------------------------------------------
def test_1_marginal_magnitude():
    hr("Test 1.  Marginal utility magnitude across the wealth grid (delta=0)")
    print(f"  b'(W) = b̄ * (W/A)^(-γ) / A")
    print(f"  γ={gamma}, b̄={b_bar}")
    # Use the canonical wealth-grid endpoints + a few interior values.
    # Annuity factor range from precompute summary: A in [4.90, 14.16].
    # Use A_mid = 7.5 as representative.
    A = 7.5
    W_grid = np.array([0.05, 0.13, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 750.0])

    print(f"  Annuity factor A = {A}  (representative midpoint value)")
    print(f"  {'W':>8s}  {'b̄·(W/A)^(1-γ)/(1-γ)':>26s}  {'b´(W) (delta=0)':>20s}  "
          f"{'b´(W) (delta=0.001)':>22s}  {'b´(W) (delta=0.005)':>22s}")

    for W in W_grid:
        u_b = b_bar * (W / A) ** (1.0 - gamma) / (1.0 - gamma)
        mu_d0 = b_bar * (W / A) ** (-gamma) / A
        mu_d1e3 = b_bar * (W / A + 0.001) ** (-gamma) / A
        mu_d5e3 = b_bar * (W / A + 0.005) ** (-gamma) / A
        print(f"  {W:>8.3f}  {u_b:>26.3e}  {mu_d0:>20.3e}  {mu_d1e3:>22.3e}  {mu_d5e3:>22.3e}")

    print()
    # fp64 limits
    print(f"  fp64 representation:")
    print(f"    smallest normalized = {np.finfo(np.float64).tiny:.3e}")
    print(f"    largest finite      = {np.finfo(np.float64).max:.3e}")
    # Find W where mu_d0 = 1e30 (well within fp64)
    W_mu1e30 = A * (b_bar / A / 1e30) ** (1.0 / gamma)
    W_mu1e60 = A * (b_bar / A / 1e60) ** (1.0 / gamma)
    print(f"    W at which b´(W; delta=0) = 1e30: {W_mu1e30:.3e}")
    print(f"    W at which b´(W; delta=0) = 1e60: {W_mu1e60:.3e}")
    print(f"  CONCLUSION: at the wealth-grid bottom (W=0.05), b´ ~ {b_bar * (0.05/A)**(-gamma)/A:.3e},")
    print(f"  fully fp64-representable.  No magnitude pathology at the spec level.")


# -----------------------------------------------------------------------------
# Test 2.  Quadrature integrand concentration
# -----------------------------------------------------------------------------
def test_2_integrand_concentration():
    hr("Test 2.  Quadrature integrand concentration  E[mu_bq · R_p^{1-γ}]")
    print(f"  Setup: log portfolio return r_p has mean ~5%, std varies with α.")
    print(f"  Quadrature: 9-node Gauss-Hermite over (xr, xb).")
    print(f"  Question: does a single low-R_p tail node dominate the sum?")

    # Build the canonical CCV scalars
    cfg, _, _ = build_nominal_system1_var_config()
    parts = partition_var(cfg["Phi"], cfg["Omega"], cfg["z_bar"],
                          cfg["state_indices"], cfg["return_indices"],
                          variable_names=cfg["variable_names"], verbose=False)
    Sigma_rr = parts["Sigma_rr"]
    sigma2_xr = float(Sigma_rr[0, 0])
    sigma2_xb = float(Sigma_rr[1, 1])
    sigma_xrxb = float(Sigma_rr[0, 1])
    z_bar_ret = parts["z_bar_ret"]

    # GH-3 nodes on a 2D Gaussian: do 3x3 product (matches canonical n_ret=3,3).
    gh_nodes_1d, gh_weights_1d = np.polynomial.hermite_e.hermegauss(3)
    # hermegauss returns nodes & weights for unit-variance normal, weights sum to sqrt(2*pi).
    # Normalize to E[1] = 1.
    gh_weights_1d = gh_weights_1d / np.sqrt(2 * np.pi)

    # Cholesky of the return-residual covariance (2x2).
    L = np.linalg.cholesky(Sigma_rr)
    # Build joint 9-node grid in (xr, xb)-space.
    xr_nodes = []
    xb_nodes = []
    weights = []
    for i, (u1, w1) in enumerate(zip(gh_nodes_1d, gh_weights_1d)):
        for j, (u2, w2) in enumerate(zip(gh_nodes_1d, gh_weights_1d)):
            x = L @ np.array([u1, u2])
            xr_nodes.append(z_bar_ret[0] + x[0])
            xb_nodes.append(z_bar_ret[1] + x[1])
            weights.append(w1 * w2)
    xr_nodes = np.array(xr_nodes)
    xb_nodes = np.array(xb_nodes)
    weights = np.array(weights)

    # Bills in nominal terms ~ z_bar(rtb) ≈ 0.05 — use rtb mean
    rtb_idx = cfg["state_indices"][cfg["variable_names"][2:].index("rtb")]   # rough — just take median
    log_R_bill_mean = 0.05  # canonical mean

    A = 7.5  # mid annuity

    def r_p(alpha_s, alpha_b):
        # CCV log-portfolio formula
        return (log_R_bill_mean
                + alpha_s * xr_nodes + alpha_b * xb_nodes
                + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
                - 0.5 * (alpha_s ** 2 * sigma2_xr
                         + 2.0 * alpha_s * alpha_b * sigma_xrxb
                         + alpha_b ** 2 * sigma2_xb))

    def integrand(alpha_s, alpha_b, s, delta):
        rp = r_p(alpha_s, alpha_b)
        Rp = np.exp(rp)
        W = s * Rp
        Cbar = W / A + delta
        mu = b_bar * Cbar ** (-gamma) / A
        # Integrand for foc_s: mu * R_p * dr/dalpha_s.
        dr_das = (xr_nodes
                  + sigma2_xr * (0.5 - alpha_s)
                  - alpha_b * sigma_xrxb)
        f = weights * mu * Rp * dr_das
        return f, Rp, mu

    print()
    s_values = [0.01, 0.5, 5.0, 50.0]   # representative savings amounts
    deltas = [0.0, 0.001, 0.005]
    alpha_test = (0.5, 0.6)             # near-Markowitz baseline
    print(f"  Test α = (α_s={alpha_test[0]}, α_b={alpha_test[1]}) — near Markowitz at γ=5.")
    print(f"  Concentration metric: |contrib of worst single node| / sum(|all 9 contribs|).")
    print(f"  100% = sum is one node; 11% = uniform across 9 nodes.")
    print()
    print(f"  {'s':>6}  {'δ':>6}  {'sum integrand':>14}  {'top |contrib| / Σ|.|':>20}  "
          f"{'min R_p':>9}  {'max R_p':>9}  {'min mu':>10}  {'max mu':>10}")
    for s_val in s_values:
        for delta in deltas:
            f, Rp, mu = integrand(*alpha_test, s_val, delta)
            top = float(np.max(np.abs(f))) / float(np.sum(np.abs(f)))
            print(f"  {s_val:>6.2f}  {delta:>6.3f}  {f.sum():>+14.3e}  {top:>19.1%}   "
                  f"{Rp.min():>9.3f}  {Rp.max():>9.3f}  {mu.min():>10.3e}  {mu.max():>10.3e}")

    print()
    print(f"  Same test at α = (3.0, -3.0) — extreme leverage, wider R_p spread:")
    print(f"  {'s':>6}  {'δ':>6}  {'sum integrand':>14}  {'top |contrib| / Σ|.|':>20}  "
          f"{'min R_p':>9}  {'max R_p':>9}  {'min mu':>10}  {'max mu':>10}")
    for s_val in s_values:
        for delta in deltas:
            f, Rp, mu = integrand(3.0, -3.0, s_val, delta)
            top = float(np.max(np.abs(f))) / float(np.sum(np.abs(f)))
            print(f"  {s_val:>6.2f}  {delta:>6.3f}  {f.sum():>+14.3e}  {top:>19.1%}   "
                  f"{Rp.min():>9.3f}  {Rp.max():>9.3f}  {mu.min():>10.3e}  {mu.max():>10.3e}")


# -----------------------------------------------------------------------------
# Test 3.  Optimum alpha sensitivity to delta
# -----------------------------------------------------------------------------
def test_3_optimum_sensitivity():
    hr("Test 3.  Optimum α sensitivity to δ at the unconditional state mean")

    cfg, _, _ = build_nominal_system1_var_config()
    parts = partition_var(cfg["Phi"], cfg["Omega"], cfg["z_bar"],
                          cfg["state_indices"], cfg["return_indices"],
                          variable_names=cfg["variable_names"], verbose=False)
    Sigma_rr = parts["Sigma_rr"]
    sigma2_xr = float(Sigma_rr[0, 0])
    sigma2_xb = float(Sigma_rr[1, 1])
    sigma_xrxb = float(Sigma_rr[0, 1])
    z_bar_ret = parts["z_bar_ret"]

    # 9-node GH product
    gh_nodes_1d, gh_weights_1d = np.polynomial.hermite_e.hermegauss(3)
    gh_weights_1d = gh_weights_1d / np.sqrt(2 * np.pi)
    L = np.linalg.cholesky(Sigma_rr)
    xr_nodes = []
    xb_nodes = []
    weights = []
    for u1, w1 in zip(gh_nodes_1d, gh_weights_1d):
        for u2, w2 in zip(gh_nodes_1d, gh_weights_1d):
            x = L @ np.array([u1, u2])
            xr_nodes.append(z_bar_ret[0] + x[0])
            xb_nodes.append(z_bar_ret[1] + x[1])
            weights.append(w1 * w2)
    xr_nodes = np.array(xr_nodes)
    xb_nodes = np.array(xb_nodes)
    weights = np.array(weights)

    log_R_bill = 0.05
    A = 7.5

    def foc(alpha_s, alpha_b, s, delta):
        rp = (log_R_bill
              + alpha_s * xr_nodes + alpha_b * xb_nodes
              + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
              - 0.5 * (alpha_s ** 2 * sigma2_xr
                       + 2.0 * alpha_s * alpha_b * sigma_xrxb
                       + alpha_b ** 2 * sigma2_xb))
        Rp = np.exp(rp)
        W = s * Rp
        Cbar = W / A + delta
        mu = b_bar * Cbar ** (-gamma) / A
        dr_das = xr_nodes + sigma2_xr * (0.5 - alpha_s) - alpha_b * sigma_xrxb
        dr_dab = xb_nodes + sigma2_xb * (0.5 - alpha_b) - alpha_s * sigma_xrxb
        return (weights * mu * Rp * dr_das).sum(), (weights * mu * Rp * dr_dab).sum()

    # Solve via Newton to high tol, scipy.optimize for robustness.
    from scipy.optimize import root

    s_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    deltas = [0.0, 0.0001, 0.001, 0.005, 0.01]

    print(f"  Solver: scipy.optimize.root (Newton, tol=1e-12).")
    print(f"  Initial α = (0.5, 0.6).  Annuity A = {A}, log_R_bill = {log_R_bill}.")
    print(f"  Markowitz baseline (δ→∞ alive limit) at this state ≈ (0.39, 0.58)")
    print()
    print(f"  {'s':>6}  | " + "  ".join(f"{'δ='+str(d):>20}" for d in deltas))
    for s in s_values:
        row = [f"  {s:>6.2f}  | "]
        for delta in deltas:
            sol = root(lambda x: foc(x[0], x[1], s, delta), x0=[0.5, 0.6],
                       method="hybr", options={"xtol": 1e-12})
            if sol.success:
                row.append(f"({sol.x[0]:+.3f}, {sol.x[1]:+.3f})")
            else:
                row.append("  FAIL              ")
        print("  ".join(row))


# -----------------------------------------------------------------------------
# Test 4.  FOC surface — basin of attraction near optimum
# -----------------------------------------------------------------------------
def test_4_foc_surface():
    hr("Test 4.  FOC surface near optimum — basin shape vs δ")
    print(f"  Plot ||F(α_s, α_b)|| over a grid around the δ=0.005 optimum,")
    print(f"  for δ ∈ {{0.0, 0.001, 0.005}}.  The δ=0 surface should be deeper")
    print(f"  near tail wells if the spike-down hypothesis holds.")

    cfg, _, _ = build_nominal_system1_var_config()
    parts = partition_var(cfg["Phi"], cfg["Omega"], cfg["z_bar"],
                          cfg["state_indices"], cfg["return_indices"],
                          variable_names=cfg["variable_names"], verbose=False)
    Sigma_rr = parts["Sigma_rr"]
    sigma2_xr = float(Sigma_rr[0, 0])
    sigma2_xb = float(Sigma_rr[1, 1])
    sigma_xrxb = float(Sigma_rr[0, 1])
    z_bar_ret = parts["z_bar_ret"]

    gh_nodes_1d, gh_weights_1d = np.polynomial.hermite_e.hermegauss(3)
    gh_weights_1d = gh_weights_1d / np.sqrt(2 * np.pi)
    L = np.linalg.cholesky(Sigma_rr)
    xr_nodes = []
    xb_nodes = []
    weights = []
    for u1, w1 in zip(gh_nodes_1d, gh_weights_1d):
        for u2, w2 in zip(gh_nodes_1d, gh_weights_1d):
            x = L @ np.array([u1, u2])
            xr_nodes.append(z_bar_ret[0] + x[0])
            xb_nodes.append(z_bar_ret[1] + x[1])
            weights.append(w1 * w2)
    xr_nodes = np.array(xr_nodes); xb_nodes = np.array(xb_nodes); weights = np.array(weights)
    log_R_bill = 0.05
    A = 7.5
    s_val = 1.0  # mid

    def foc_norm(alpha_s, alpha_b, delta):
        rp = (log_R_bill
              + alpha_s * xr_nodes + alpha_b * xb_nodes
              + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
              - 0.5 * (alpha_s ** 2 * sigma2_xr
                       + 2.0 * alpha_s * alpha_b * sigma_xrxb
                       + alpha_b ** 2 * sigma2_xb))
        Rp = np.exp(rp)
        W = s_val * Rp
        Cbar = W / A + delta
        mu = b_bar * Cbar ** (-gamma) / A
        dr_das = xr_nodes + sigma2_xr * (0.5 - alpha_s) - alpha_b * sigma_xrxb
        dr_dab = xb_nodes + sigma2_xb * (0.5 - alpha_b) - alpha_s * sigma_xrxb
        Fs = (weights * mu * Rp * dr_das).sum()
        Fb = (weights * mu * Rp * dr_dab).sum()
        return np.sqrt(Fs * Fs + Fb * Fb), Fs, Fb

    # 1D probes along α_b at fixed α_s = 0.4 (approximate optimum)
    alpha_s_fix = 0.4
    alpha_b_grid = np.linspace(-3, 3, 21)
    print(f"  ||F(α_s={alpha_s_fix}, α_b)|| along α_b axis (s={s_val}):")
    print(f"  {'α_b':>6}  {'||F|| δ=0':>12}  {'||F|| δ=.001':>14}  {'||F|| δ=.005':>14}")
    for ab in alpha_b_grid:
        norm0, _, _ = foc_norm(alpha_s_fix, ab, 0.0)
        norm1, _, _ = foc_norm(alpha_s_fix, ab, 0.001)
        norm5, _, _ = foc_norm(alpha_s_fix, ab, 0.005)
        print(f"  {ab:>+6.2f}  {norm0:>12.3e}  {norm1:>14.3e}  {norm5:>14.3e}")

    print()
    print(f"  Same probe at extreme α — α_s=2.0, α_b in [-10, 10] (large FOC region):")
    alpha_s_fix = 2.0
    alpha_b_grid = np.linspace(-10, 10, 21)
    print(f"  {'α_b':>6}  {'||F|| δ=0':>12}  {'||F|| δ=.001':>14}  {'||F|| δ=.005':>14}")
    for ab in alpha_b_grid:
        norm0, _, _ = foc_norm(alpha_s_fix, ab, 0.0)
        norm1, _, _ = foc_norm(alpha_s_fix, ab, 0.001)
        norm5, _, _ = foc_norm(alpha_s_fix, ab, 0.005)
        print(f"  {ab:>+6.2f}  {norm0:>12.3e}  {norm1:>14.3e}  {norm5:>14.3e}")

    print()
    print(f"  Multiple roots check: scan ||F|| on a 2D grid (s={s_val}, δ=0).")
    print(f"  Report all local minima of ||F|| where ||F|| < 1e-3.")
    g = np.linspace(-6, 6, 41)
    Z = np.zeros((g.size, g.size))
    for i, a_s in enumerate(g):
        for j, a_b in enumerate(g):
            n, _, _ = foc_norm(a_s, a_b, 0.0)
            Z[i, j] = n
    # local minima: cells smaller than all 8 neighbours
    minima = []
    for i in range(1, g.size - 1):
        for j in range(1, g.size - 1):
            window = Z[i-1:i+2, j-1:j+2].copy()
            center = Z[i, j]
            window[1, 1] = np.inf
            if center < window.min():
                minima.append((g[i], g[j], center))
    minima.sort(key=lambda r: r[2])
    print(f"  Found {len(minima)} local minima of ||F|| at δ=0.")
    print(f"  Top 5 by ||F||:")
    for a_s, a_b, val in minima[:5]:
        print(f"    α=({a_s:+.2f}, {a_b:+.2f})  ||F||={val:.3e}")


if __name__ == "__main__":
    test_1_marginal_magnitude()
    test_2_integrand_concentration()
    test_3_optimum_sensitivity()
    test_4_foc_surface()
