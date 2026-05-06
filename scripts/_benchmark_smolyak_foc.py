"""End-to-end FOC integrand benchmark for Smolyak-Hermite candidates.

Investigation only. Does NOT modify lifecycle/, configs/_canonical.py, or
any production module. Run standalone:

    python -m scripts._benchmark_smolyak_foc [--n_cells 6]

Compares per-cell FOC integrand evaluations between:
  - canonical 3,600-node tensor product   (baseline, "ground truth")
  - Smolyak L=4 anisotropic + 64 corner anchors  (305 nodes)
  - Smolyak L=5 anisotropic + 64 corner anchors  (713 nodes)

The integrand is the production CCV FOC kernel
  f(z) = u'(W_next) * V_next * exp(r_p^CCV)
with v = L_state z_v, r = L_ret z_r, r_p^CCV from the Itô-corrected
log-portfolio formula at [precompute.py:231-237], and V_next obtained by
trilinearly interpolating a synthetic-but-realistic V_grid on the
canonical state grid (in u-space, matching solver.py:884-895).

The benchmark stresses two integration challenges that the prior MGF
test did NOT exercise:
  1. The CRRA u'(W_next) = W_next^{-gamma} amplifies tail mass.
  2. Trilinear V interpolation has C^0 kinks across cell boundaries —
     polynomial-exactness guarantees do NOT cover these.

Synthetic V_grid: V(s) = -W^{1-gamma}/(1-gamma) * exp(rho . s) where
rho = (0.3, 1.5, 1.2) over (cy, spr, y_1). Magnitude of V varies ~3x
across the canonical 9x9x9 state grid -- representative of post-retirement
production V curvature.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_DISC
from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from scripts._investigate_smolyak_returnstate import SmolyakRule, _gh_rule


# --------------------------------------------------------------------------- #
# Quadrature -> unique-(z, weight) tables                                     #
# --------------------------------------------------------------------------- #
def smolyak_z_table(rule: SmolyakRule, anchors_z: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Collapse Smolyak's signed tensor terms (+ optional anchors) into a
    unique-z table.  At each unique z, weight = sum of c_j * w_tensor over
    all (j, c_j) tensor terms that hit z.  Anchors not already in the
    table are inserted with weight 0 (support enrichment, no integration
    contribution)."""
    accum: dict[tuple[float, ...], float] = {}
    for j, c in rule.coefs.items():
        grids = [_gh_rule(jk) for jk in j]
        zs = [g[0] for g in grids]
        ws = [g[1] for g in grids]
        for idx in itertools.product(*[range(jk) for jk in j]):
            z = tuple(round(float(zs[k][idx[k]]), 12) for k in range(rule.d))
            wprod = float(np.prod([ws[k][idx[k]] for k in range(rule.d)]))
            accum[z] = accum.get(z, 0.0) + c * wprod
    if anchors_z is not None:
        for row in anchors_z:
            z = tuple(round(float(x), 12) for x in row)
            accum.setdefault(z, 0.0)
    z_arr = np.array([list(z) for z in accum.keys()], dtype=float)
    w_arr = np.array(list(accum.values()), dtype=float)
    return z_arr, w_arr


def baseline_z_table(K_v: tuple[int, ...], K_r: tuple[int, ...]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Canonical 6-D tensor product Gauss-Hermite rule in z-space."""
    grids_v = [_gh_rule(K) for K in K_v]
    grids_r = [_gh_rule(K) for K in K_r]
    zs  = [g[0] for g in grids_v] + [g[0] for g in grids_r]
    wts = [g[1] for g in grids_v] + [g[1] for g in grids_r]
    n_per = [len(z) for z in zs]
    n_total = int(np.prod(n_per))
    d = 6
    z_arr = np.empty((n_total, d), dtype=float)
    w_arr = np.empty(n_total, dtype=float)
    for k, idx in enumerate(itertools.product(*[range(n) for n in n_per])):
        for axis in range(d):
            z_arr[k, axis] = zs[axis][idx[axis]]
        w_arr[k] = float(np.prod([wts[axis][idx[axis]] for axis in range(d)]))
    return z_arr, w_arr


# --------------------------------------------------------------------------- #
# Trilinear V interpolation in u-space (matches solver.py:884-895)            #
# --------------------------------------------------------------------------- #
def trilinear_interp_vectorised(u_arr: np.ndarray, axes_grids: list[np.ndarray],
                                V_grid_flat: np.ndarray, N_vec: tuple[int, ...]
                                ) -> np.ndarray:
    """u_arr (N, 3) -> V_next (N,).  Boundary clamping matches bracket_state_3d."""
    N_pts = u_arr.shape[0]
    los = np.zeros((N_pts, 3), dtype=int)
    fs  = np.zeros((N_pts, 3), dtype=float)
    his = np.zeros((N_pts, 3), dtype=int)
    for d in range(3):
        g = axes_grids[d]
        if N_vec[d] == 1:
            continue
        x = u_arr[:, d]
        idx = np.searchsorted(g, x) - 1
        idx = np.clip(idx, 0, N_vec[d] - 2)
        los[:, d] = idx
        his[:, d] = idx + 1
        denom = (g[idx + 1] - g[idx])
        frac = (x - g[idx]) / denom
        frac = np.where(x <= g[0], 0.0, frac)
        frac = np.where(x >= g[-1], 1.0, frac)
        fs[:, d] = frac
    N0, N1, N2 = N_vec
    def flat(i0, i1, i2): return i0 * N1 * N2 + i1 * N2 + i2
    f0, f1, f2 = fs[:, 0], fs[:, 1], fs[:, 2]
    l0, l1, l2 = los[:, 0], los[:, 1], los[:, 2]
    h0, h1, h2 = his[:, 0], his[:, 1], his[:, 2]
    return ((1-f0)*(1-f1)*(1-f2) * V_grid_flat[flat(l0, l1, l2)]
          + (1-f0)*(1-f1)*f2     * V_grid_flat[flat(l0, l1, h2)]
          + (1-f0)*f1*(1-f2)     * V_grid_flat[flat(l0, h1, l2)]
          + (1-f0)*f1*f2         * V_grid_flat[flat(l0, h1, h2)]
          + f0*(1-f1)*(1-f2)     * V_grid_flat[flat(h0, l1, l2)]
          + f0*(1-f1)*f2         * V_grid_flat[flat(h0, l1, h2)]
          + f0*f1*(1-f2)         * V_grid_flat[flat(h0, h1, l2)]
          + f0*f1*f2             * V_grid_flat[flat(h0, h1, h2)])


# --------------------------------------------------------------------------- #
# Vectorised CCV FOC integrand                                                #
# --------------------------------------------------------------------------- #
def evaluate_foc(z_arr: np.ndarray, w_arr: np.ndarray, *,
                 s_cell, alpha_s, alpha_b, gamma, W,
                 L_state, L_ret, M, Phi_0_state, Phi_11,
                 base_mu_r, sigma2_xr, sigma2_xb, sigma_xrxb,
                 state_bracket_shift, state_bracket_L_inv,
                 state_bracket_grids, V_grid_flat, N_vec) -> float:
    z_v = z_arr[:, :3]
    z_r = z_arr[:, 3:]
    v = z_v @ L_state.T
    r_resid = z_r @ L_ret.T
    s_next = (Phi_0_state + Phi_11 @ s_cell)[None, :] + v
    log_R = base_mu_r[None, :] + v @ M.T + r_resid
    log_R_bill = log_R[:, 0]
    x_s = log_R[:, 1]
    x_b = log_R[:, 2]
    r_p = (log_R_bill + alpha_s * x_s + alpha_b * x_b
           + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
           - 0.5 * (alpha_s**2 * sigma2_xr
                    + 2.0 * alpha_s * alpha_b * sigma_xrxb
                    + alpha_b**2 * sigma2_xb))
    W_next = W * np.exp(r_p)
    u_marg = W_next ** (-gamma)
    u_next = (s_next - state_bracket_shift[None, :]) @ state_bracket_L_inv.T
    V_next = trilinear_interp_vectorised(u_next, state_bracket_grids,
                                          V_grid_flat, N_vec)
    f_vals = u_marg * V_next * np.exp(r_p)
    return float(np.dot(w_arr, f_vals))


def synthetic_V_grid(state_grid: np.ndarray, gamma: float = 5.0, W: float = 1.0
                     ) -> np.ndarray:
    """V(s) = -W^{1-gamma}/(1-gamma) * exp(rho . s).

    State ordering (cy, spr, y_1).  rho = (0.3, 1.5, 1.2) gives ~3x
    variation across the canonical 9x9x9 grid (range ~ 1 to 3, mean ~ 1.7).
    """
    rho = np.array([0.3, 1.5, 1.2])
    base = -(W ** (1 - gamma)) / (1 - gamma)
    return base * np.exp(state_grid @ rho)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_cells", type=int, default=6,
                   help="number of stress state cells to test (default: 6)")
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--W", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 80)
    print("END-TO-END FOC INTEGRAND BENCHMARK: Smolyak vs canonical tensor")
    print("=" * 80)

    var_cfg, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
    model = build_model(BASE_CONFIG, var_cfg, verbose=False)
    pc = Precompute(model, CANONICAL_DISC, verbose=False)

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
    print(f"V_grid range = [{V_grid_flat.min():+.4e}, {V_grid_flat.max():+.4e}]"
          f"   mean={V_grid_flat.mean():+.4e}   ratio max/min = "
          f"{abs(V_grid_flat.max()/V_grid_flat.min()):.2f}\n")

    # --- Build z-tables for the three rules
    K_v_can = (3, 4, 4); K_r_can = (3, 5, 5); K_max = K_v_can + K_r_can
    L_target = 4.0
    w_canonical = tuple(L_target / (K_max[k] - 1) for k in range(6))
    z_max_per_axis = np.array([float(np.max(np.abs(_gh_rule(K)[0]))) for K in K_max])
    anchors = np.array(list(itertools.product(*[[-1.0, +1.0]] * 6)),
                        dtype=float) * z_max_per_axis[None, :]

    rule_L4 = SmolyakRule.build(4.0, w_canonical, K_max)
    rule_L5 = SmolyakRule.build(5.0, w_canonical, K_max)

    z_base, w_base = baseline_z_table(K_v_can, K_r_can)
    z_L4,   w_L4   = smolyak_z_table(rule_L4, anchors)
    z_L5,   w_L5   = smolyak_z_table(rule_L5, anchors)

    # Sanity: weights should sum to 1 (probability measure).
    print(f"baseline:   N={len(w_base):>4d}   sum w={w_base.sum():+.10e}")
    print(f"L=4 + anch: N={len(w_L4):>4d}   sum w={w_L4.sum():+.10e}")
    print(f"L=5 + anch: N={len(w_L5):>4d}   sum w={w_L5.sum():+.10e}\n")

    rules = [
        ("baseline (3,600 tensor)", z_base, w_base),
        ("Smolyak L=4 + 64 anch",   z_L4,   w_L4),
        ("Smolyak L=5 + 64 anch",   z_L5,   w_L5),
    ]

    # --- Pick stress cells
    rng = np.random.default_rng(args.seed)
    n_total = pc.state_grid.shape[0]

    # Hand-pick a few corners with extreme bond-loading + a few random body cells.
    proj = pc.state_grid @ np.array([0.1, 1.0, 1.0])      # spr+y_1 dominant
    sorted_idx = np.argsort(proj)
    chosen_idx = []
    chosen_idx += [int(sorted_idx[-1])]                    # max-bond-stress
    chosen_idx += [int(sorted_idx[0])]                     # min-bond-stress
    chosen_idx += [int(sorted_idx[len(sorted_idx)//2])]    # median
    n_extra = max(0, args.n_cells - len(chosen_idx))
    if n_extra:
        rest = np.setdiff1d(np.arange(n_total), chosen_idx, assume_unique=False)
        chosen_idx += list(map(int, rng.choice(rest, size=n_extra, replace=False)))
    chosen_idx = chosen_idx[:args.n_cells]

    # --- Stress alphas
    alphas = [
        (0.0, 0.0),
        (1.0, 1.0),
        (3.0, 2.0),
        (5.0, -3.0),
        (-3.0, 5.0),
        (6.0, 6.0),       # cap-bound, both long
        (6.0, -6.0),      # cap-bound, opposite signs
    ]

    # --- Run the benchmark
    print("=" * 80)
    print(f"FOC integrand Q[u'(W_next) * V_next * exp(r_p^CCV)] at gamma={args.gamma}, W={args.W}")
    print("=" * 80)
    print(f"State cells (i_s, s_vector):")
    for k, i_s in enumerate(chosen_idx):
        print(f"  cell {k}:  i_s={i_s:3d}   s={tuple(round(float(v), 4) for v in pc.state_grid[i_s])}")
    print()

    overall_max_relerr = {name: 0.0 for name, _, _ in rules if name != "baseline (3,600 tensor)"}

    for cell_k, i_s in enumerate(chosen_idx):
        s_cell = pc.state_grid[i_s]
        base_mu_r = pc.const_r + pc.A_r @ s_cell
        print(f"Cell {cell_k}  (i_s={i_s})")
        header = f"{'alpha=(a_s,a_b)':<18}" + "  ".join(
            f"{name:>26}" for name, _, _ in rules) + "    " + \
            "  ".join(f"{'rel.err vs baseline':>22}" for _ in rules[1:])
        print(header)
        for a_s, a_b in alphas:
            row_vals = []
            for name, z_arr, w_arr in rules:
                t0 = time.time()
                Q = evaluate_foc(z_arr, w_arr,
                                 s_cell=s_cell, alpha_s=a_s, alpha_b=a_b,
                                 gamma=args.gamma, W=args.W,
                                 L_state=L_state, L_ret=L_ret, M=M,
                                 Phi_0_state=Phi_0_state, Phi_11=Phi_11,
                                 base_mu_r=base_mu_r,
                                 sigma2_xr=sigma2_xr, sigma2_xb=sigma2_xb,
                                 sigma_xrxb=sigma_xrxb,
                                 state_bracket_shift=state_bracket_shift,
                                 state_bracket_L_inv=state_bracket_L_inv,
                                 state_bracket_grids=state_bracket_grids,
                                 V_grid_flat=V_grid_flat, N_vec=N_vec)
                row_vals.append(Q)
            base_Q = row_vals[0]
            relerr = [(Q - base_Q) / abs(base_Q) for Q in row_vals[1:]]
            cells = [f"({a_s:+.0f},{a_b:+.0f})".ljust(18)]
            for Q in row_vals:
                cells.append(f"{Q:>+26.6e}")
            for re in relerr:
                cells.append(f"{re:>+22.3e}")
            print("  ".join(cells))
            for (name, _, _), re in zip(rules[1:], relerr):
                if abs(re) > overall_max_relerr[name]:
                    overall_max_relerr[name] = abs(re)
        print()

    print("=" * 80)
    print("Summary: max |relative error| over all (cell, alpha) combinations")
    print("=" * 80)
    for name in overall_max_relerr:
        print(f"  {name:<32}  max |relerr| = {overall_max_relerr[name]:.3e}")
    print()
    print("Interpretation:")
    print("  Baseline is 'ground truth' for this comparison. Relative errors are")
    print("  the discrepancy a Smolyak swap would introduce in the FOC integrand")
    print("  *holding the integrand fixed* — i.e. they isolate quadrature error,")
    print("  not solver convergence error. At interior alpha (~1-3 range) we want")
    print("  relerr < 1e-3 for a clean swap; at cap-bound (+/-6) corners the FOC")
    print("  is anyway dominated by the cap, so 1e-2 is acceptable there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
