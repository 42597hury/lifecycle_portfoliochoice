"""Arbitrage check for Smolyak-Hermite return-state quadrature candidates.

Investigation only. Does NOT modify lifecycle/, configs/_canonical.py, or
any production module. Run standalone:

    python scripts/_check_smolyak_arbitrage.py [--n_states 60] [--cap 6.0]

Question: do the candidate Smolyak rules from
`_investigate_smolyak_returnstate.py` admit a CCV-cloud arbitrage that
the canonical 3,600-node tensor product does not?

Method: for each Smolyak rule, enumerate the *unique* joint node positions
in z-space (signed weights ignored — arbitrage only depends on the support
of the cloud, not the weights). Map z -> (v, r) via Cholesky. At each
canonical state s, build (x_s, x_b) realised log-excess returns and run
the same `check_state_arbitrage` routine used by the production CCV
arbitrage diagnostic ([scripts/_check_ccv_arbitrage.py]). Flag if max_t > 0.

Smolyak's negative weights are NOT a direct arbitrage risk — the test
depends only on whether the cloud's worst-node bond/stock returns are
beatable by some portfolio. The risk is that Smolyak takes a sparser
union of GH-level slices and may miss the "no-arbitrage anchor" nodes
that thicken the canonical cloud's tail.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from scripts._check_ccv_arbitrage import check_state_arbitrage
from scripts._investigate_smolyak_returnstate import (
    SmolyakRule,
    _gh_rule,
)


def smolyak_unique_z_nodes(rule: SmolyakRule) -> np.ndarray:
    """Enumerate unique z-positions in any tensor-subterm with c_j != 0.

    Returns ndarray (n_unique, d).  Signed weights are NOT propagated;
    only the support matters for arbitrage.
    """
    seen: dict[tuple[float, ...], None] = {}
    for j, _c in rule.coefs.items():
        grids = [_gh_rule(jk)[0] for jk in j]
        for idx in itertools.product(*[range(jk) for jk in j]):
            z = tuple(round(float(grids[k][idx[k]]), 12) for k in range(rule.d))
            seen[z] = None
    return np.array([list(z) for z in seen.keys()], dtype=float)


def _build_state_clouds_smolyak(
    z_unique: np.ndarray,
    L_state: np.ndarray,
    L_ret: np.ndarray,
    M: np.ndarray,
    base_mu_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Realised (x_s, x_b) cloud at one state for a given Smolyak z-set.

    Vectorised over the n_unique nodes:  z -> v = L_state @ z_v
                                           -> r = L_ret @ z_r
                                           -> x_s = base_mu_r[1] + (M v)[1] + r[1]
                                           -> x_b = base_mu_r[2] + (M v)[2] + r[2]
    """
    z_v = z_unique[:, :3]                  # (N, 3)
    z_r = z_unique[:, 3:]                  # (N, 3)
    v_nodes = z_v @ L_state.T              # (N, 3)
    r_nodes = z_r @ L_ret.T                # (N, 3)
    Mv = v_nodes @ M.T                     # (N, 3) = M v
    x_s = base_mu_r[1] + Mv[:, 1] + r_nodes[:, 1]
    x_b = base_mu_r[2] + Mv[:, 2] + r_nodes[:, 2]
    return x_s, x_b


def _build_state_clouds_baseline(
    v_nodes: np.ndarray, ret_nodes: np.ndarray, M_v_nodes: np.ndarray,
    base_mu_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the baseline cloud as in scripts/_check_ccv_arbitrage.py:175-180."""
    x_s_grid = (base_mu_r[1] + M_v_nodes[:, 1])[:, None] + ret_nodes[:, 1][None, :]
    x_b_grid = (base_mu_r[2] + M_v_nodes[:, 2])[:, None] + ret_nodes[:, 2][None, :]
    return x_s_grid.ravel(), x_b_grid.ravel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, default=6.0,
                        help="leverage cap for arbitrage search (default: 6.0)")
    parser.add_argument("--n_states", type=int, default=60,
                        help="number of canonical states to scan (default: 60). "
                             "Stratified sample weighted toward extreme y_1+spr")
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="t threshold above which a state is flagged (default: 1e-5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for state subsample")
    args = parser.parse_args()

    print("=" * 78)
    print("SMOLYAK ARBITRAGE CHECK (vs canonical 3,600-node tensor baseline)")
    print("=" * 78)

    # --- Load canonical model & precompute (same setup as _check_ccv_arbitrage)
    var_cfg, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)
    pc = Precompute(model, CANONICAL_DISC, verbose=False)

    sigma2_xr  = float(pc.sigma2_xr)
    sigma2_xb  = float(pc.sigma2_xb)
    sigma_xrxb = float(pc.sigma_xrxb)
    M = np.asarray(model.M, dtype=float)
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=float)
    Sigma_r_cond = np.asarray(model.Sigma_r_cond, dtype=float)
    L_state = np.linalg.cholesky(0.5 * (Sigma_ss + Sigma_ss.T))
    L_ret   = np.linalg.cholesky(0.5 * (Sigma_r_cond + Sigma_r_cond.T))

    print(f"M (return-from-state-innov):\n{M}")
    print(f"L_state diag = {np.diag(L_state)}")
    print(f"L_ret   diag = {np.diag(L_ret)}\n")
    print(f"Canonical cloud: {pc.v_nodes.shape[0]} state-quad x "
          f"{pc.ret_nodes.shape[0]} return-quad = "
          f"{pc.v_nodes.shape[0] * pc.ret_nodes.shape[0]} joint nodes")
    print(f"State grid: {pc.state_grid.shape}, leverage cap |alpha| <= {args.cap}\n")

    # --- Build candidate Smolyak rules (must match _investigate_smolyak_returnstate)
    K_max = (3, 4, 4, 3, 5, 5)
    L_target = 4.0
    w_canonical = tuple(L_target / (K_max[k] - 1) for k in range(6))
    w_aggressive = (2.0, 1.5, 1.5, 2.0, 1.0, 1.0)

    # Keep only the recommended survivors (L=4/L=5 + anchors). The unsafe
    # candidates were ruled out at the 40-state scan; rerunning them on
    # full 729 wastes wall-clock.
    candidates = [
        ("Smolyak L=4 anisotropic",       SmolyakRule.build(4.0, w_canonical, K_max)),
        ("Smolyak L=5 anisotropic",       SmolyakRule.build(5.0, w_canonical, K_max)),
    ]

    # Build the 2^6 = 64 binary corner anchors at canonical max-K tail z values.
    # These are zero-weight quadrature nodes — they do NOT affect integration,
    # but they enrich the cloud support so the discrete return cloud has joint
    # tails consistent with the no-arbitrage shape of the continuous Gaussian.
    z_max_per_axis = np.array([
        float(np.max(np.abs(_gh_rule(K)[0]))) for K in K_max
    ])
    corner_anchors_z = np.array(list(itertools.product(*[[-1.0, +1.0]] * 6)),
                                 dtype=float) * z_max_per_axis[None, :]
    print(f"Per-axis max-K tail z = {z_max_per_axis}")
    print(f"#corner anchors        = {corner_anchors_z.shape[0]}\n")

    # --- Pick stress states: prefer corners of the state grid where (cy, spr, y_1)
    #     are at the high-loading-on-bond extremes. We sample a stratified mix:
    #     half pure-grid corners + half random body states.
    rng = np.random.default_rng(args.seed)
    n_total = pc.state_grid.shape[0]
    n_states_scan = min(args.n_states, n_total)

    # Score each state by projecting onto bond-return loading direction.
    # x_b realised = ... + M[2,:] @ v + L_ret[2,:] @ z_r.  States that maximise
    # |M[2,:] @ s| are most stressed. (Plus the |y_1+spr| corners.)
    scores = np.abs(pc.state_grid @ np.array([0.1, 1.0, 1.0]))    # spr+y_1 dominant
    # Take top half as stress, bottom half random.
    n_stress = n_states_scan // 2
    n_random = n_states_scan - n_stress
    stress_idx = np.argsort(-scores)[:n_stress]
    rest = np.setdiff1d(np.arange(n_total), stress_idx, assume_unique=False)
    random_idx = rng.choice(rest, size=n_random, replace=False)
    state_indices = np.concatenate([stress_idx, random_idx])

    print(f"Scanning {n_states_scan} states  ({n_stress} bond-stress corners + "
          f"{n_random} random body states)\n")

    # --- Pre-print node-count summary
    print("=" * 78)
    print("Smolyak rule support sizes (unique z-positions)")
    print("=" * 78)
    print(f"{'rule':<28} {'#tensor':>9} {'#signed':>9} {'#unique z-nodes':>17}")
    print("-" * 78)
    rule_supports: dict[str, np.ndarray] = {}
    for name, rule in candidates:
        z_unique = smolyak_unique_z_nodes(rule)
        rule_supports[name] = z_unique
        print(f"{name:<28} {rule.n_terms:>9d} {rule.n_signed_nodes:>9d} {z_unique.shape[0]:>17d}")
    # Also build "+ 64 corner anchors" variants of the two leading candidates.
    augmented_specs = [
        ("Smolyak L=4 + 64 anchors", "Smolyak L=4 anisotropic"),
        ("Smolyak L=5 + 64 anchors", "Smolyak L=5 anisotropic"),
    ]
    for new_name, base_name in augmented_specs:
        z_aug = np.vstack([rule_supports[base_name], corner_anchors_z])
        # Dedupe (a Smolyak node may rarely coincide with a corner)
        z_aug = np.unique(np.round(z_aug, 12), axis=0)
        rule_supports[new_name] = z_aug
        print(f"{new_name:<28} {'-':>9} {'-':>9} {z_aug.shape[0]:>17d}")
    print(f"{'baseline (full tensor)':<28} {1:>9d} {3600:>9d} {3600:>17d}\n")

    # --- Per-state arbitrage scan: baseline + each candidate
    print("=" * 78)
    print(f"Arbitrage search (max_t = max_alpha min_node eta;  > 0 = arbitrage)")
    print(f"Threshold for flag: max_t > {args.threshold:.0e}")
    print("=" * 78)
    rule_max_t: dict[str, np.ndarray] = {}
    rule_n_arb: dict[str, int] = {}
    rule_argmax_state: dict[str, int] = {}

    rule_specs: list[tuple[str, object]] = [("baseline (full tensor)", None)]
    rule_specs += [(n, r) for n, r in candidates]
    rule_specs += [(n, "augmented") for n, _ in augmented_specs]

    for name, rule in rule_specs:
        max_ts = np.empty(n_states_scan)
        max_alphas = np.empty((n_states_scan, 2))
        t0 = time.time()
        for k, i_s in enumerate(state_indices):
            s_i = pc.state_grid[i_s]
            base_mu_r = pc.const_r + pc.A_r @ s_i
            if rule is None:
                x_s, x_b = _build_state_clouds_baseline(
                    pc.v_nodes, pc.ret_nodes, pc.M_v_nodes, base_mu_r
                )
            else:
                z_unique = rule_supports[name]
                x_s, x_b = _build_state_clouds_smolyak(
                    z_unique, L_state, L_ret, M, base_mu_r
                )
            # Both branches above produced (x_s, x_b)
            max_t, alpha_star = check_state_arbitrage(
                x_s, x_b, sigma2_xr, sigma2_xb, sigma_xrxb, cap=args.cap
            )
            max_ts[k] = max_t
            max_alphas[k] = alpha_star
        elapsed = time.time() - t0
        rule_max_t[name] = max_ts
        rule_n_arb[name] = int(np.sum(max_ts > args.threshold))
        rule_argmax_state[name] = int(state_indices[int(np.argmax(max_ts))])

        n_unique = (3600 if rule is None else rule_supports[name].shape[0])
        print(f"{name:<28}  n_unique={n_unique:>4d}  "
              f"elapsed={elapsed:5.1f}s  "
              f"max_t={max_ts.max():+.3e}  "
              f"states_above_threshold={rule_n_arb[name]:>2d}/{n_states_scan}  "
              f"argmax_state_idx={rule_argmax_state[name]}")

    # --- Verdict table
    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)
    baseline_max = float(rule_max_t['baseline (full tensor)'].max())
    print(f"baseline max_t over scanned states = {baseline_max:+.3e}")
    print(f"baseline arbitrage states          = {rule_n_arb['baseline (full tensor)']}")
    print()
    print(f"{'rule':<28} {'max_t':>13} {'#flagged':>10} {'safe?':>8}")
    print("-" * 78)
    all_named = [n for n, _ in candidates] + [n for n, _ in augmented_specs]
    for name in all_named:
        mt = float(rule_max_t[name].max())
        n_flag = rule_n_arb[name]
        safe = (mt <= max(baseline_max, args.threshold) + 1e-8)
        flag = "YES" if safe else "NO -- new arbitrage"
        print(f"{name:<28}  {mt:>+13.3e} {n_flag:>10d}   {flag}")
    print()
    print("Notes:")
    print(" * 'safe' means: max_t and #flagged states do not exceed baseline.")
    print(" * Sparser support -> more likely to admit spurious arbitrage at a given state.")
    print(" * The minimisation finds the worst-case alpha; result is concave in alpha so")
    print("   the SLSQP optimum is global modulo numerical tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
