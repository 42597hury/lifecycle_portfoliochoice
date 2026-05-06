"""Lobatto-Smolyak benchmark: prescribed-tail tail-mass within the Smolyak rule.

Investigation only. Does NOT modify lifecycle/, configs/_canonical.py, or
any production module.

Hypothesis from the prior anchored-Smolyak investigation:
  Anchored Smolyak (zero-weight corner anchors on top of Smolyak L=5)
  closes the spurious-arbitrage failure but NOT the integration-error
  failure at high γ × |α|, because zero-weight anchors enrich support
  without carrying probability mass at the tail corners. A Smolyak rule
  built from per-level Lobatto univariate rules (prescribed tail nodes
  with NON-ZERO Gauss-style weights) should close both failures.

Test design:
  Per-axis univariate rule schedule (at Smolyak level i):
    - GH axis:       K = i nodes, standard Gauss-Hermite on N(0,1)
    - Lobatto axis:  i=1 -> K=1 (single 0 node)
                     i>=2 -> K=2i-1 in {3, 5, 7}, Lobatto with tails ±Z
  Lobatto axes carry tail nodes at every level >=2 with non-zero weights.
  When Smolyak combines low-level and high-level multi-indices, the
  joint-corner support is automatically populated.

Per-axis assignment for canonical 6-D quadrature:
    Axis            kind     Z       Why
    z_v0  cy        gh        -      M[xb,cy] = -0.005 (weak channel)
    z_v1  spr       lobatto   2.93   M[xb,spr] = -8.51 (dominant)
    z_v2  y_1       lobatto   2.93   M[xb,y_1] = -8.72 (dominant)
    z_r0  rtb       gh        -      small residual variance
    z_r1  xr        lobatto   2.86   matches GH-5 max tail
    z_r2  xb        lobatto   2.86   matches GH-5 max tail

Comparison candidates:
    (1) baseline full tensor (3,600 nodes, max_t=6.6e-14 from prior 729 scan)
    (2) Smolyak L=5 + 64 anchors (713 nodes, max |Δα*| = 0.088)
    (3) Lobatto-Smolyak L=4 (this script)
    (4) Lobatto-Smolyak L=5 (this script)
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from scipy.optimize import root, minimize

from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from lifecycle.quadrature_with_tails import gauss_hermite_prescribed_tails
from scripts._investigate_smolyak_returnstate import _gh_rule
from scripts._benchmark_smolyak_foc import (
    baseline_z_table, trilinear_interp_vectorised, synthetic_V_grid,
)
from scripts._check_ccv_arbitrage import check_state_arbitrage
from scripts._benchmark_smolyak_alpha_star import find_alpha_star


# --------------------------------------------------------------------------- #
# Hybrid axis rule + Lobatto-Smolyak combination                              #
# --------------------------------------------------------------------------- #
def _axis_rule(level: int, kind: str, Z: float | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Univariate rule at Smolyak level i, dispatched by axis kind."""
    if level < 1:
        raise ValueError(f"level >= 1 required, got {level}")
    if kind == "gh":
        if level == 1:
            return np.zeros(1), np.ones(1)
        return _gh_rule(level)
    if kind == "lobatto":
        if Z is None:
            raise ValueError("Lobatto axis requires Z")
        if level == 1:
            return np.zeros(1), np.ones(1)
        K = 2 * level - 1
        if K not in (3, 5, 7):
            raise ValueError(f"Lobatto level {level} -> K={K} not in {{3,5,7}}")
        return gauss_hermite_prescribed_tails(K, float(Z))
    raise ValueError(f"unknown axis kind: {kind!r}")


def lobatto_smolyak_admissible(L: float, w: tuple[float, ...],
                               level_max: tuple[int, ...]
                               ) -> list[tuple[int, ...]]:
    d = len(w)
    out: list[tuple[int, ...]] = []
    def recurse(prefix, k, slack):
        if k == d:
            out.append(tuple(prefix))
            return
        for i_k in range(1, level_max[k] + 1):
            cost = w[k] * (i_k - 1)
            if cost > slack + 1e-12:
                break
            recurse(prefix + [i_k], k + 1, slack - cost)
    recurse([], 0, L)
    return out


def lobatto_smolyak_z_table(L: float, w: tuple[float, ...],
                            level_max: tuple[int, ...],
                            kinds: tuple[str, ...], Zs: tuple[float | None, ...]
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Build (z_arr, w_arr) for a Lobatto-Smolyak rule via combination technique."""
    A = lobatto_smolyak_admissible(L, w, level_max)
    A_set = set(A)
    d = len(w)
    # Compute c_j combination coefficients
    j_set: set[tuple[int, ...]] = set()
    for i in A:
        for q in itertools.product((0, 1), repeat=d):
            j = tuple(i[k] - q[k] for k in range(d))
            if all(jk >= 1 for jk in j):
                j_set.add(j)
    coefs: dict[tuple[int, ...], int] = {}
    for j in j_set:
        c = 0
        for q in itertools.product((0, 1), repeat=d):
            i = tuple(j[k] + q[k] for k in range(d))
            if i in A_set:
                c += (-1) ** sum(q)
        if c != 0:
            coefs[j] = c

    # Enumerate signed nodes, dedupe by z position
    accum: dict[tuple[float, ...], float] = {}
    for j, c in coefs.items():
        grids = [_axis_rule(jk, kinds[k], Zs[k]) for k, jk in enumerate(j)]
        zs = [g[0] for g in grids]
        ws = [g[1] for g in grids]
        n_per = [len(g[0]) for g in grids]
        for idx in itertools.product(*[range(n) for n in n_per]):
            z = tuple(round(float(zs[k][idx[k]]), 12) for k in range(d))
            wprod = float(np.prod([ws[k][idx[k]] for k in range(d)]))
            accum[z] = accum.get(z, 0.0) + c * wprod

    z_arr = np.array([list(z) for z in accum.keys()], dtype=float)
    w_arr = np.array(list(accum.values()), dtype=float)
    return z_arr, w_arr


# --------------------------------------------------------------------------- #
# Cloud builder for arbitrage check (matches scripts/_check_smolyak_arbitrage)  #
# --------------------------------------------------------------------------- #
def _build_state_cloud(z_unique, L_state, L_ret, M, base_mu_r):
    z_v = z_unique[:, :3]; z_r = z_unique[:, 3:]
    v = z_v @ L_state.T; r = z_r @ L_ret.T
    Mv = v @ M.T
    x_s = base_mu_r[1] + Mv[:, 1] + r[:, 1]
    x_b = base_mu_r[2] + Mv[:, 2] + r[:, 2]
    return x_s, x_b


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_arb_states", type=int, default=80,
                   help="states to scan for arbitrage (default 80; use 729 for full)")
    p.add_argument("--n_alpha_cells", type=int, default=8,
                   help="cells for the alpha-star root benchmark")
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--W", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    np.set_printoptions(precision=4, suppress=True, linewidth=130)

    print("=" * 84)
    print("LOBATTO-SMOLYAK BENCHMARK")
    print("=" * 84)

    var_cfg, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)
    pc = Precompute(model, CANONICAL_DISC, verbose=False)
    cap = float(CANONICAL_SOLVER.alpha_max)

    M = np.asarray(model.M, dtype=float)
    Sigma_ss = 0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T)
    Sigma_r_cond = 0.5 * (np.asarray(model.Sigma_r_cond)
                            + np.asarray(model.Sigma_r_cond).T)
    L_state = np.linalg.cholesky(Sigma_ss)
    L_ret   = np.linalg.cholesky(Sigma_r_cond)
    Phi_0_state = np.asarray(model.Phi_0_state, dtype=float)
    Phi_11      = np.asarray(model.Phi_11, dtype=float)
    sigma2_xr  = float(pc.sigma2_xr)
    sigma2_xb  = float(pc.sigma2_xb)
    sigma_xrxb = float(pc.sigma_xrxb)
    state_bracket_shift = np.asarray(pc.state_bracket_shift, dtype=float)
    state_bracket_L_inv = np.asarray(pc.state_bracket_L_inv, dtype=float)
    state_bracket_grids = [np.asarray(g, dtype=float) for g in pc.state_bracket_grids]
    N_vec = tuple(int(len(g)) for g in state_bracket_grids)
    V_grid_flat = synthetic_V_grid(pc.state_grid, gamma=args.gamma, W=args.W)

    # --- Per-axis assignment
    kinds = ("gh", "lobatto", "lobatto", "gh", "lobatto", "lobatto")
    Zs = (None, 2.93, 2.93, None, 2.86, 2.86)
    level_max = (3, 3, 3, 3, 3, 3)   # Lobatto level 3 -> K=5; GH level 3 -> K=3

    print("Per-axis schedule:")
    for k, (kind, Z, lm) in enumerate(zip(kinds, Zs, level_max)):
        K_at_lm = (lm if kind == "gh" else (1 if lm == 1 else 2*lm - 1))
        print(f"  axis {k}: kind={kind:<7}  Z={Z}  level_max={lm}  -> K_max={K_at_lm}")
    print()

    # --- Build candidate rules: vary Smolyak level budget L
    rule_specs = [
        ("Lobatto-Smolyak L=2", 2.0),
        ("Lobatto-Smolyak L=3", 3.0),
        ("Lobatto-Smolyak L=4", 4.0),
        ("Lobatto-Smolyak L=5", 5.0),
    ]
    w_iso = (1.0,) * 6      # Isotropic in level space; K-anisotropy is in the schedule itself

    rule_tables = {}
    for name, L_budget in rule_specs:
        z_arr, w_arr = lobatto_smolyak_z_table(L_budget, w_iso, level_max, kinds, Zs)
        rule_tables[name] = (z_arr, w_arr)
        print(f"{name:<24} unique z-nodes = {z_arr.shape[0]:>4d}   "
              f"sum w = {w_arr.sum():+.10e}")
    print()

    # Reference baseline tensor
    K_v_can = (3, 4, 4); K_r_can = (3, 5, 5)
    z_base, w_base = baseline_z_table(K_v_can, K_r_can)
    rule_tables["baseline"] = (z_base, w_base)
    print(f"baseline (full tensor): N={len(w_base)}  sum w = {w_base.sum():+.10e}\n")

    # ---------------- Arbitrage scan ---------------------------------------
    print("=" * 84)
    print(f"Arbitrage scan ({args.n_arb_states} stress states; max_t > 0 = arbitrage)")
    print("=" * 84)
    rng = np.random.default_rng(args.seed)
    n_total = pc.state_grid.shape[0]
    n_arb = min(args.n_arb_states, n_total)
    proj = pc.state_grid @ np.array([0.1, 1.0, 1.0])
    sorted_idx = np.argsort(-np.abs(proj))
    n_stress = n_arb // 2
    stress_idx = sorted_idx[:n_stress]
    rest = np.setdiff1d(np.arange(n_total), stress_idx, assume_unique=False)
    rand_idx = rng.choice(rest, size=n_arb - n_stress, replace=False)
    arb_state_indices = np.concatenate([stress_idx, rand_idx])

    arb_results = {}
    for name, (z_arr, w_arr) in rule_tables.items():
        if name == "baseline":
            continue   # known: max_t = 6.6e-14 from prior 729-state scan
        max_ts = np.empty(n_arb)
        t0 = time.time()
        for k, i_s in enumerate(arb_state_indices):
            s_cell = pc.state_grid[i_s]
            base_mu_r = pc.const_r + pc.A_r @ s_cell
            x_s, x_b = _build_state_cloud(z_arr, L_state, L_ret, M, base_mu_r)
            mt, _ = check_state_arbitrage(x_s, x_b, sigma2_xr, sigma2_xb,
                                            sigma_xrxb, cap=cap)
            max_ts[k] = mt
        elapsed = time.time() - t0
        arb_results[name] = (float(max_ts.max()), int(np.sum(max_ts > 1e-5)), elapsed)
        print(f"{name:<24} N={z_arr.shape[0]:>4d}  elapsed={elapsed:5.1f}s  "
              f"max_t={float(max_ts.max()):+.3e}  flagged={int(np.sum(max_ts > 1e-5)):>3d}/{n_arb}")
    print()

    # ---------------- alpha-star root benchmark ----------------------------
    print("=" * 84)
    print(f"alpha-star benchmark ({args.n_alpha_cells} stress cells)")
    print("=" * 84)
    proj_full = pc.state_grid @ np.array([0.1, 1.0, 1.0])
    sorted_full = np.argsort(proj_full)
    chosen = [int(sorted_full[-1]), int(sorted_full[0]),
              int(sorted_full[len(sorted_full)//2])]
    rest_full = np.setdiff1d(np.arange(n_total), chosen, assume_unique=False)
    extra = max(0, args.n_alpha_cells - len(chosen))
    chosen += list(map(int, rng.choice(rest_full, size=extra, replace=False)))
    chosen = chosen[:args.n_alpha_cells]

    # Run baseline once per cell, then each Lobatto-Smolyak rule
    print(f"{'i_s':>4} {'state (cy,spr,y_1)':<26} {'rule':<24} "
          f"{'alpha*':>22} {'|d_a| vs base':>14}")
    print("-" * 130)

    summary = {name: [] for name, _ in rule_specs}
    for i_s in chosen:
        s_cell = pc.state_grid[i_s]
        base_mu_r = pc.const_r + pc.A_r @ s_cell
        kw = dict(s_cell=s_cell, gamma=args.gamma, W=args.W,
                  L_state=L_state, L_ret=L_ret, M=M,
                  Phi_0_state=Phi_0_state, Phi_11=Phi_11,
                  base_mu_r=base_mu_r,
                  sigma2_xr=sigma2_xr, sigma2_xb=sigma2_xb, sigma_xrxb=sigma_xrxb,
                  state_bracket_shift=state_bracket_shift,
                  state_bracket_L_inv=state_bracket_L_inv,
                  state_bracket_grids=state_bracket_grids,
                  V_grid_flat=V_grid_flat, N_vec=N_vec)
        # baseline
        a_base, _, _ = find_alpha_star(z_base, w_base, kw=kw, cap=cap)
        s_str = f"({s_cell[0]:+.2f}, {s_cell[1]:+.2f}, {s_cell[2]:+.2f})"
        print(f"{i_s:>4} {s_str:<26} {'baseline':<24} "
              f"({a_base[0]:+.3f},{a_base[1]:+.3f})".ljust(22) + f" {'-':>14}")
        for name, _ in rule_specs:
            z_arr, w_arr = rule_tables[name]
            a_rule, _, _ = find_alpha_star(z_arr, w_arr, kw=kw, cap=cap)
            d = float(np.linalg.norm(a_rule - a_base))
            summary[name].append(d)
            a_str = f"({a_rule[0]:+.3f},{a_rule[1]:+.3f})"
            print(f"{'':>4} {'':<26} {name:<24} "
                  f"{a_str:>22} {d:>14.3e}")
        print()

    # ---------------- Summary ---------------------------------------------
    print("=" * 84)
    print("Summary")
    print("=" * 84)
    print(f"{'rule':<24} {'#nodes':>8} {'arb max_t':>14} "
          f"{'mean |d_a|':>14} {'max |d_a|':>14} {'pass(1e-3)':>12}")
    print("-" * 84)
    for name, _ in rule_specs:
        ds = np.array(summary[name])
        n_nodes = rule_tables[name][0].shape[0]
        arb_max, n_flag, _ = arb_results[name]
        n_pass3 = int(np.sum(ds < 1e-3))
        print(f"{name:<24} {n_nodes:>8d} {arb_max:>+14.3e} "
              f"{ds.mean():>14.3e} {ds.max():>14.3e} {n_pass3:>5d}/{len(ds):>5d}")
    print()
    # Reference rows
    print("Reference (from prior investigations):")
    print(f"  baseline (full tensor)       3600   +6.6e-14         0.000e+00      0.000e+00    8/8")
    print(f"  Smolyak L=5 + 64 anchors      713   +1.1e-13         1.6e-02        8.8e-02    4/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
