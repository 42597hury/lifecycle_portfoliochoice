"""Verify solver and simulator agree on R_p at every realisation.

Background: HANDOFF_SIMULATOR_CCV_FIX. The simulator previously hardwired the
wealth law to arithmetic returns while the solver used CCV log-wealth dynamics.
This test pins the parity invariant: at any (alpha_s, alpha_b, log_R_bill,
log_x_s, log_x_b, sigma2_xr, sigma2_xb, sigma_xrxb), the simulator's R_p and
solver._ccv_log_return_and_grad must agree to machine precision.
"""
import numpy as np
import jax.numpy as jnp

from lifecycle.solver import _ccv_log_return_and_grad


def simulator_R_p(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                  sigma2_xr, sigma2_xb, sigma_xrxb):
    """Mirror of the simulator's CCV log-portfolio block in lifecycle.simulation."""
    log_R_port = (
        log_R_bill
        + alpha_s * log_x_s + alpha_b * log_x_b
        + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
        - 0.5 * (alpha_s * alpha_s * sigma2_xr
                 + 2.0 * alpha_s * alpha_b * sigma_xrxb
                 + alpha_b * alpha_b * sigma2_xb)
    )
    return jnp.exp(log_R_port)


def main():
    rng = np.random.default_rng(0)
    n_trials = 1000
    max_abs_err = 0.0
    for _ in range(n_trials):
        a_s = rng.uniform(-2.0, 3.0)
        a_b = rng.uniform(-2.0, 3.0)
        rb = rng.normal(0.01, 0.005)
        xs = rng.normal(0.05, 0.18)
        xb = rng.normal(0.02, 0.06)
        s2_s = rng.uniform(0.01, 0.05)
        s2_b = rng.uniform(0.001, 0.01)
        sxs = rng.uniform(-0.005, 0.005)

        R_solver, _, _ = _ccv_log_return_and_grad(
            a_s, a_b, rb, xs, xb, s2_s, s2_b, sxs
        )
        R_sim = simulator_R_p(a_s, a_b, rb, xs, xb, s2_s, s2_b, sxs)
        err = abs(float(R_solver) - float(R_sim))
        max_abs_err = max(max_abs_err, err)
        assert err < 1e-12, (
            f"solver/simulator R_p disagree at alpha=({a_s:.3f},{a_b:.3f}): "
            f"solver={float(R_solver):.10e} sim={float(R_sim):.10e} err={err:.2e}"
        )

    print(f"PASS: {n_trials}/{n_trials} random realisations agree to 1e-12 "
          f"(max abs err {max_abs_err:.2e})")


if __name__ == "__main__":
    main()
