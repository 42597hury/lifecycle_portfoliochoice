"""CCV-specific arbitrage diagnostic.

Tests the hypothesis: at any state on the production grid, does there
exist alpha such that

    eta(alpha; n) = r_p^CCV(alpha; n) - r_bill_n
                  = alpha . x^(n) + 0.5 alpha . diag(Sigma)
                    - 0.5 alpha' Sigma alpha
                  > 0   for ALL joint quadrature nodes n ?

This is a CCV-version of the "discrete-cloud arbitrage" check. Pre-flight
T-Q1 tested the simple-return cloud; this tests the log-return cloud
*after* CCV's Itô vol-drag is applied. eta is concave-quadratic in alpha
(Hessian = -Sigma, NSD), so max_alpha min_n eta is a concave maximisation.

Numerical method: for each state i_s, build the joint cloud x^(n) =
realised log-excess return at node n. Maximise min_n eta(alpha) over
alpha in [-CAP, +CAP]^2 by SLSQP with the standard reformulation

    max  t   s.t.  eta^(n)(alpha) >= t  for each n,  |alpha_j| <= CAP

If the optimiser finds (alpha*, t*) with t* > 0, the discrete CCV cloud
admits a guaranteed-positive log-excess portfolio — a real CCV arbitrage.

Run:
    python -m scripts._check_ccv_arbitrage [--cap 6.0] [--n_states 343]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize

from configs._canonical import BASE_CONFIG, CANONICAL_DISC
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute


def _eta_kernel(alpha_s, alpha_b, x_s, x_b, sigma2_xr, sigma2_xb, sigma_xrxb):
    """eta = r_p^CCV - r_bill at one node (vectorised over arrays of x_s, x_b)."""
    return (alpha_s * x_s + alpha_b * x_b
            + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
            - 0.5 * (alpha_s * alpha_s * sigma2_xr
                     + 2.0 * alpha_s * alpha_b * sigma_xrxb
                     + alpha_b * alpha_b * sigma2_xb))


def check_state_arbitrage(x_s_nodes, x_b_nodes,
                          sigma2_xr, sigma2_xb, sigma_xrxb,
                          cap=6.0):
    """Return (max_t, alpha_star) where max_t = max_alpha min_n eta^(n).

    If max_t > 0 -> CCV arbitrage at this state.
    """
    # Variables: x = (alpha_s, alpha_b, t). Maximise t subject to
    # eta^(n)(alpha) - t >= 0 for all n, |alpha_j| <= cap.
    n_nodes = len(x_s_nodes)

    def neg_t(x):
        return -x[2]

    def neg_t_grad(x):
        g = np.zeros(3)
        g[2] = -1.0
        return g

    # Constraints: g^(n)(x) = eta^(n)(alpha) - t >= 0.
    def make_con(i):
        xs_i = float(x_s_nodes[i])
        xb_i = float(x_b_nodes[i])

        def fn(x):
            a_s, a_b, t = x
            return _eta_kernel(a_s, a_b, xs_i, xb_i,
                               sigma2_xr, sigma2_xb, sigma_xrxb) - t

        def jac(x):
            a_s, a_b, _ = x
            g = np.empty(3)
            g[0] = xs_i + sigma2_xr * (0.5 - a_s) - a_b * sigma_xrxb
            g[1] = xb_i + sigma2_xb * (0.5 - a_b) - a_s * sigma_xrxb
            g[2] = -1.0
            return g

        return {"type": "ineq", "fun": fn, "jac": jac}

    cons = [make_con(i) for i in range(n_nodes)]
    bounds = [(-cap, cap), (-cap, cap), (None, None)]

    # Try several seeds: zero, the per-node-mean Markowitz portfolio, and
    # random perturbations
    seeds = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([-1.0, -1.0, 0.0]),
        np.array([cap * 0.5, cap * 0.5, 0.0]),
    ]
    best_t = -np.inf
    best_alpha = np.zeros(2)
    for x0 in seeds:
        try:
            res = minimize(neg_t, x0, jac=neg_t_grad, constraints=cons,
                           bounds=bounds, method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-10})
            if res.success and -res.fun > best_t:
                best_t = -res.fun
                best_alpha = res.x[:2].copy()
        except Exception:
            pass

    return best_t, best_alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, default=6.0,
                        help="leverage cap for arbitrage search (default: 6.0)")
    parser.add_argument("--max_states", type=int, default=343,
                        help="cap on number of states to scan (default: all)")
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="t threshold above which a state is flagged "
                             "(default: 1e-5)")
    args = parser.parse_args()

    print("=" * 72)
    print("CCV-SPECIFIC ARBITRAGE CHECK")
    print("=" * 72)
    print("Tests: does there exist alpha such that")
    print("  r_p^CCV(alpha; node) - r_bill(node) > 0  for ALL joint quad nodes?")
    print("If yes at any state, CCV cloud admits a guaranteed-positive log-excess.")
    print()

    var_cfg, _, _ = build_nominal_system1_var_config(
        csv_path="data/var_dataset.csv"
    )
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)
    pc = Precompute(model, CANONICAL_DISC, verbose=False)

    sigma2_xr = float(pc.sigma2_xr)
    sigma2_xb = float(pc.sigma2_xb)
    sigma_xrxb = float(pc.sigma_xrxb)

    print(f"Sigma_r_cond[1,1]={sigma2_xr:.5e},  [2,2]={sigma2_xb:.5e},  "
          f"[1,2]={sigma_xrxb:.5e}")
    print(f"State grid: {pc.state_grid.shape}, "
          f"v_nodes: {pc.v_nodes.shape}, "
          f"ret_nodes: {pc.ret_nodes.shape}")
    print(f"Joint quadrature cloud per state: "
          f"{pc.v_nodes.shape[0] * pc.ret_nodes.shape[0]} nodes")
    print(f"Leverage cap: ±{args.cap:.1f}")
    print()

    n_state_quad = pc.v_nodes.shape[0]
    n_ret_quad = pc.ret_nodes.shape[0]
    n_joint = n_state_quad * n_ret_quad

    # Flatten the (k_v, k_r) cloud into n_joint nodes
    arb_states = []
    state_max_ts = np.empty(pc.N_state)
    state_max_alphas = np.empty((pc.N_state, 2))

    n_to_scan = min(pc.N_state, args.max_states)
    t0 = time.time()
    print(f"Scanning {n_to_scan} states ...")
    for i_s in range(n_to_scan):
        s_i = pc.state_grid[i_s]
        # Conditional return mean at this state
        base_mu_r = pc.const_r + pc.A_r @ s_i
        # Build per-(k_v, k_r) realised log-excess returns x_s, x_b
        # mu_r_stock = base_mu_r[1] + M_v_nodes[k_v, 1]; log_x_s adds ret_nodes[k_r, 1]
        # x_s = log_x_s_node = mu_xr realised at this (k_v, k_r)
        # We don't need log_R_bill since eta cancels it.
        x_s_grid = (base_mu_r[1] + pc.M_v_nodes[:, 1])[:, None] \
                   + pc.ret_nodes[:, 1][None, :]
        x_b_grid = (base_mu_r[2] + pc.M_v_nodes[:, 2])[:, None] \
                   + pc.ret_nodes[:, 2][None, :]
        x_s = x_s_grid.ravel()
        x_b = x_b_grid.ravel()
        max_t, alpha_star = check_state_arbitrage(
            x_s, x_b, sigma2_xr, sigma2_xb, sigma_xrxb, cap=args.cap
        )
        state_max_ts[i_s] = max_t
        state_max_alphas[i_s] = alpha_star
        if max_t > args.threshold:
            arb_states.append((i_s, max_t, alpha_star, s_i.copy()))

    elapsed = time.time() - t0
    print(f"Scanned {n_to_scan} states in {elapsed:.1f}s "
          f"({elapsed/max(n_to_scan,1)*1000:.0f}ms/state)")
    print()
    print(f"max over states of max_t  =  {state_max_ts[:n_to_scan].max():+.6e}")
    print(f"min over states of max_t  =  {state_max_ts[:n_to_scan].min():+.6e}")
    print(f"states with max_t > {args.threshold:.0e}: {len(arb_states)}")
    print()
    if arb_states:
        print("--- CCV ARBITRAGE STATES (sorted by max_t) ---")
        arb_states.sort(key=lambda r: -r[1])
        for i_s, t_star, alpha, s_vec in arb_states[:20]:
            print(f"  i_s={i_s:3d}  max_t={t_star:+.5e}  "
                  f"alpha=({alpha[0]:+.3f}, {alpha[1]:+.3f})  "
                  f"state={tuple(round(v, 4) for v in s_vec)}")
        if len(arb_states) > 20:
            print(f"  ... and {len(arb_states)-20} more")
    else:
        print("--- NO CCV ARBITRAGE detected at any state ---")
        print("Hypothesis-supportive: 85% Newton rate is non-convexity, not arbitrage.")

    # Distribution of max_t (negative is good — means even the best alpha")
    # gives a min-node eta that's still negative)
    pcts = (50, 90, 95, 99, 100)
    print()
    print("Distribution of max_t over states (large = closer to arbitrage):")
    for q in pcts:
        v = np.percentile(state_max_ts[:n_to_scan], q)
        print(f"  pct {q:5.1f}: {v:+.5e}")


if __name__ == "__main__":
    main()
