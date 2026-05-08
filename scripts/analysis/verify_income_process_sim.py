"""
Verify the simulator's income process matches the theoretical AR(1) mixture.

Theory (from BASE_CONFIG):
    log z_{t+1} = rho * z_t + eta_{t+1}
    eta is a 2-component normal mixture, centered to E[eta]=0:
       w.p. pz       eta ~ N(mu_eta1, sigma_eta1^2)
       w.p. 1-pz     eta ~ N(mu_eta2_eff, sigma_eta2^2)   where mu_eta2_eff = -(pz/(1-pz))*mu_eta1
    eps is similarly a 2-component centered mixture (transitory shock).

Verification (10,000 households, 78 periods):
    1. eta innovation mean / std / skew / kurt against analytical mixture moments.
    2. z autocorrelation against rho.
    3. Stationary cross-section variance against var_eta / (1 - rho^2) AND
       against the realized-at-age-65 variance.
    4. eps innovation moments against analytical.
    5. Mixture component shares: count how often eta lands within ~3σ of each
       component vs the theoretical pz / (1-pz).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG  # noqa: E402
from lifecycle.policy_io import load_policy_bundle  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.simulation import simulate_lifecycle  # noqa: E402
from verify._diag_helpers import build_bundle_var_config  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid  # noqa: E402


def _list_to_tuple(v):
    if isinstance(v, list):
        return tuple(_list_to_tuple(x) for x in v)
    return v


def _rehydrate_disc(d):
    tuple_fields = {"state_grid_sizes", "n_state_quad_nodes", "n_ret_nodes_1d",
                    "state_n_stds", "ret_lobatto_Z", "state_lobatto_Z"}
    valid = set(DiscretizationConfig._fields)
    return DiscretizationConfig(
        **{k: (_list_to_tuple(v) if k in tuple_fields else v)
           for k, v in d.items() if k in valid}
    )


def analytical_mixture_moments(p1, mu1, sigma1, mu2, sigma2):
    """Raw moments of a 2-component normal mixture, plus skew/kurt."""
    p2 = 1.0 - p1
    m1 = p1 * mu1 + p2 * mu2
    m2 = p1 * (sigma1**2 + mu1**2) + p2 * (sigma2**2 + mu2**2)
    var = m2 - m1**2
    # Standardized 3rd and 4th central moments
    # E[(X-m)^3] = sum p_i * (3*sigma_i^2*(mu_i-m) + (mu_i-m)^3)
    c3 = (p1 * (3*sigma1**2*(mu1-m1) + (mu1-m1)**3)
          + p2 * (3*sigma2**2*(mu2-m1) + (mu2-m1)**3))
    # E[(X-m)^4] = sum p_i * (3*sigma_i^4 + 6*sigma_i^2*(mu_i-m)^2 + (mu_i-m)^4)
    c4 = (p1 * (3*sigma1**4 + 6*sigma1**2*(mu1-m1)**2 + (mu1-m1)**4)
          + p2 * (3*sigma2**4 + 6*sigma2**2*(mu2-m1)**2 + (mu2-m1)**4))
    skew = c3 / var**1.5 if var > 0 else float("nan")
    kurt_excess = c4 / var**2 - 3.0 if var > 0 else float("nan")
    return {"mean": m1, "var": var, "std": np.sqrt(var), "skew": skew,
            "kurt_excess": kurt_excess, "c3": c3, "c4": c4}


def empirical_moments(arr):
    arr = arr.ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return {"mean": float("nan"), "std": float("nan"), "skew": float("nan"),
                "kurt_excess": float("nan"), "n": int(arr.size)}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "skew": float(stats.skew(arr)),
        "kurt_excess": float(stats.kurtosis(arr)),  # excess kurtosis
        "n": int(arr.size),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=REPO / "saved_runs" / "ablations"
                                  / "system_1_grid7_nz70_calib1")
    parser.add_argument("--n-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading bundle {args.bundle.name}...", flush=True)
    C, S, B, _, metadata = load_policy_bundle(args.bundle)
    rc = metadata["run_config"]
    base_config = rc["base_config"]
    disc = _rehydrate_disc(rc["discretization_config"])
    disc = disc_config_with_bundle_wealth_grid(disc, args.bundle, metadata)
    var_config = build_bundle_var_config(metadata, args.bundle)
    model = build_model(base_config, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)

    print(f"Simulating {args.n_simulations} households for {pc.n_age} periods...",
          flush=True)
    sim = simulate_lifecycle(
        C, S, B, pc, model,
        n_simulations=args.n_simulations,
        initial_wealth=0.1,
        initial_z="stationary",
        initial_state="median",
        seed=args.seed,
        verbose=False,
    )
    z = np.asarray(sim["z"])           # (n_sim, n_age)
    income = np.asarray(sim["income"]) # (n_sim, n_age) — after-tax labor income
    alive = np.asarray(sim["alive"])
    print(f"  z panel shape: {z.shape}", flush=True)

    # --- Reconstruct eta innovations from z trajectory ---
    # z_{t+1} = rho * z_t + eta_{t+1}  =>  eta_{t+1} = z_{t+1} - rho * z_t
    # We can compute eta for every transition where the household is alive
    # at both t and t+1.
    rho = base_config["rho"]
    pz = base_config["pz"]
    mu_eta1 = base_config["mu_eta1"]
    sigma_eta1 = base_config["sigma_eta1"]
    sigma_eta2 = base_config["sigma_eta2"]
    mu_eta2_eff = -(pz / (1.0 - pz)) * mu_eta1     # solver enforces E[eta]=0

    # eta_{t+1} = z_{t+1} - rho * z_t, only on working-age transitions
    # (after retirement z is frozen since pension is z-tied but no income shocks).
    retire_age = base_config["retire_age"]   # 67
    start_age = base_config["start_age"]     # 22
    retire_idx = retire_age - start_age      # 45
    # Compute eta only for working-age transitions where both periods are alive.
    eta_panel = z[:, 1:retire_idx+1] - rho * z[:, :retire_idx]
    alive_mask = alive[:, 1:retire_idx+1] & alive[:, :retire_idx]
    eta_flat = eta_panel[alive_mask]

    print("\n=== eta (persistent income innovation) ===")
    eta_theory = analytical_mixture_moments(pz, mu_eta1, sigma_eta1,
                                              mu_eta2_eff, sigma_eta2)
    eta_emp = empirical_moments(eta_flat)
    print(f"{'moment':>16} {'theory':>14} {'sim':>14} {'rel diff':>10}")
    for k in ("mean", "std", "skew", "kurt_excess"):
        rel = ((eta_emp[k] - eta_theory[k]) / abs(eta_theory[k])
               if abs(eta_theory[k]) > 1e-9 else float("nan"))
        print(f"{k:>16} {eta_theory[k]:>14.6f} {eta_emp[k]:>14.6f} {rel:>10.3%}")
    print(f"  N = {eta_emp['n']:,} eta innovations across all working-age "
          f"transitions.")

    # --- z autocorrelation per-period (working-age) ---
    print("\n=== z first-order autocorrelation per period (working-age) ===")
    print(f"  Theoretical AR(1) coefficient = rho = {rho:.6f}")
    autos = []
    for t in range(0, retire_idx):
        m = alive[:, t] & alive[:, t+1]
        if m.sum() < 100: continue
        z_t = z[m, t]; z_t1 = z[m, t+1]
        if z_t.std() > 1e-9 and z_t1.std() > 1e-9:
            ac = float(np.corrcoef(z_t, z_t1)[0, 1])
            autos.append((t + start_age, ac))
    autos = np.array(autos)
    print(f"  Empirical AR(1) coef per period (mean across working-age): {autos[:,1].mean():.6f}")
    print(f"  Per-period range: [{autos[:,1].min():.4f}, {autos[:,1].max():.4f}]")
    print(f"  Per-period std: {autos[:,1].std():.4f}")

    # --- Stationary variance check ---
    print("\n=== z stationary variance ===")
    var_eta_theory = eta_theory["var"]
    var_z_stationary = var_eta_theory / (1.0 - rho**2)
    print(f"  Theoretical stationary var(z) = var_eta / (1 - rho^2) = "
          f"{var_eta_theory:.4f} / {1-rho**2:.6f} = {var_z_stationary:.4f}")
    # Realized var(z) at each age (after burn-in via stationary init)
    z_var_age = np.array([z[alive[:, t], t].var(ddof=1) if alive[:, t].any() else 0
                           for t in range(retire_idx)])
    print(f"  Realized var(z) at age 22 (init): {z_var_age[0]:.4f} "
          f"(should ~= {var_z_stationary:.4f} if init drew from stationary)")
    print(f"  Realized var(z) at age 65 (working-age max): {z_var_age[-1]:.4f}")
    print(f"  Realized std(z) at age 65: {np.sqrt(z_var_age[-1]):.4f} "
          f"(theoretical stationary std = {np.sqrt(var_z_stationary):.4f})")

    # --- eps innovation moments ---
    print("\n=== eps (transitory income shock) ===")
    pe = base_config["pe"]
    mu_eps1 = base_config["mu_eps1"]
    sigma_eps1 = base_config["sigma_eps1"]
    sigma_eps2 = base_config["sigma_eps2"]
    mu_eps2_eff = -(pe / (1.0 - pe)) * mu_eps1
    eps_theory = analytical_mixture_moments(pe, mu_eps1, sigma_eps1,
                                              mu_eps2_eff, sigma_eps2)
    # Reconstruct eps from log income: log y_t = b0 + b1*age + ... + z_t + eps_t
    # The simulator stores after-tax income; need pre-tax.
    # log_det_per_age is in pc.log_det_profile.
    log_det = np.asarray(pc.log_det_profile)
    log_y_pretax = np.log(np.maximum(income, 1e-12))
    # Working ages only; income is the after-tax path. Pre-tax log y = log y / (1 - tau).
    # Simpler: just check log(income) - rho*z_{t-1} - log_det residual relative to
    # eps + log(1-tau). The eps mean is offset by log(1-tau) but std/skew/kurt invariant.
    # Use: residual_t = log y_t - log_det[t] - z_t. Then resid = log(1-tau) + eps_t.
    # tau is fixed (10.6% payroll ~= 0.106); the std/skew/kurt of resid match eps's.
    resid = (log_y_pretax[:, :retire_idx]
              - log_det[:retire_idx][None, :]
              - z[:, :retire_idx])
    resid_alive = resid[alive[:, :retire_idx]]
    # Strip the constant offset before comparing higher moments.
    eps_emp = empirical_moments(resid_alive - resid_alive.mean())
    # std/skew/kurt are invariant to constant shift; mean is set to 0 by definition.
    print(f"{'moment':>16} {'theory':>14} {'sim (centered)':>16} {'rel diff':>10}")
    print(f"{'std':>16} {eps_theory['std']:>14.6f} {eps_emp['std']:>16.6f} "
          f"{(eps_emp['std']-eps_theory['std'])/eps_theory['std']:>9.3%}")
    print(f"{'skew':>16} {eps_theory['skew']:>14.6f} {eps_emp['skew']:>16.6f} "
          f"{(eps_emp['skew']-eps_theory['skew'])/abs(eps_theory['skew']):>9.3%}")
    print(f"{'kurt_excess':>16} {eps_theory['kurt_excess']:>14.6f} "
          f"{eps_emp['kurt_excess']:>16.6f} "
          f"{(eps_emp['kurt_excess']-eps_theory['kurt_excess'])/abs(eps_theory['kurt_excess']):>9.3%}")

    # --- Mixture component frequency for eta ---
    # A draw from the mixture is a draw from comp1 with prob pz, comp2 otherwise.
    # We can't recover the latent indicator perfectly, but we can compute the
    # POSTERIOR P(comp1 | eta) and check the empirical mean equals pz.
    print("\n=== eta mixture component balance ===")
    from scipy.stats import norm
    p1_post = (pz * norm.pdf(eta_flat, loc=mu_eta1, scale=sigma_eta1) /
                (pz * norm.pdf(eta_flat, loc=mu_eta1, scale=sigma_eta1)
                 + (1-pz) * norm.pdf(eta_flat, loc=mu_eta2_eff, scale=sigma_eta2)))
    print(f"  Theoretical P(comp1) = pz = {pz:.4f}")
    print(f"  Empirical mean P(comp1 | eta_observed) = {p1_post.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
