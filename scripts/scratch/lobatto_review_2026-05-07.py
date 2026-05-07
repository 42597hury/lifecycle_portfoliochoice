"""lobatto_review_2026-05-07.py — independent re-run of the Lobatto investigation.

Re-runs the §3 evidence table from
`docs/handoff/HANDOFF_LOBATTO_QUADRATURE_INVESTIGATION.md` but with
substantially more rigour:

  * Multi-cell stress (Task 1): not just the centroid — also corner cells
    and tail cells.
  * Truth bias check (Task 3): compare K=7 truth to K=9 truth.
  * Per-axis sensitivity scan (Task 5): bump each axis individually.
  * Cheap-Lobatto-without-K-bump variant (Task 6): same node count as
    current canonical, Lobatto-on-K=3.

FOC integrand exactly mirrors `_eval_integrand` in
`scripts/scratch/smolyak_feasibility_jax.py`.

Run from project root:
    python -m scripts.scratch.lobatto_review_2026-05-07
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np
from scipy.special import roots_hermite

sys.path.insert(0, ".")

from configs._canonical_jax import CANONICAL_DISC, BASE_CONFIG
from lifedpcle.var import build_nominal_system1_var_config_hardcoded
from lifedpcle.precompute import build_model, build_precompute
from lifedpcle.quadrature_with_tails import gauss_hermite_prescribed_tails


# ---------------------------------------------------------------------------
# 1D rules
# ---------------------------------------------------------------------------

def gh_1d(K: int):
    if K == 1:
        return np.zeros(1), np.ones(1)
    z, w = roots_hermite(int(K))
    return z * np.sqrt(2.0), w / np.sqrt(np.pi)


def lobatto_1d(K: int, Z: float):
    return gauss_hermite_prescribed_tails(int(K), float(Z))


def tensor_mixed_rule(K_per_axis, Z_per_axis):
    grids = []
    for K, Z in zip(K_per_axis, Z_per_axis):
        if Z is None:
            grids.append(gh_1d(K))
        else:
            grids.append(lobatto_1d(K, Z))
    z_grids, w_grids = zip(*grids)
    meshes_z = np.meshgrid(*z_grids, indexing="ij")
    meshes_w = np.meshgrid(*w_grids, indexing="ij")
    nodes = np.stack([m.ravel() for m in meshes_z], axis=1)
    weights = np.prod(np.stack(meshes_w, axis=0), axis=0).ravel()
    return nodes, weights


def tensor_gh_rule(K_per_axis):
    return tensor_mixed_rule(K_per_axis, (None,) * len(K_per_axis))


# ---------------------------------------------------------------------------
# Build model + precompute
# ---------------------------------------------------------------------------

def build():
    disc = CANONICAL_DISC._replace(ret_lobatto_Z=None, state_lobatto_Z=None)
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(BASE_CONFIG, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    return model, pc, disc


# ---------------------------------------------------------------------------
# FOC-style integrand (exact match to smolyak_feasibility_jax)
# ---------------------------------------------------------------------------

def make_integrand(model, s_t, v_next_kind="smooth"):
    """Returns a function joint_z -> integrand value at chosen state cell s_t.

    v_next_kind:
      'smooth':  V_next = 0.1 * exp(-0.3*dp_next - 0.1*y_1_next)  [original]
      'curved':  V_next has Gaussian curvature centred at the body cell, so
                 its z-shape *depends* on s_t. Specifically:
                 V_next(s_next) = 0.1 * exp(-0.5 * sum_d ((s_next[d] - mu_d) / w_d)^2)
                 with mu_d = 0 (in z-space, after subtracting state_grid_mu) and
                 w_d ~ 1.5 * sigma_z. This makes the integrand's curvature in z
                 actually depend on where s_t lives in the grid.
    """
    L_s = np.linalg.cholesky(0.5 * (model.Sigma_ss + model.Sigma_ss.T))
    L_r = np.linalg.cholesky(0.5 * (model.Sigma_r_cond + model.Sigma_r_cond.T))
    Phi_0 = np.asarray(model.Phi_0_state, dtype=np.float64)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
    rtb_idx = int(model.rtb_index_in_state)
    xr_pos = list(model.ret_names).index("xr")
    xb_pos = list(model.ret_names).index("xb")
    A_r = np.asarray(model.Phi_21, dtype=np.float64)
    const_r = np.asarray(model.Phi_0_ret, dtype=np.float64)
    M = np.asarray(model.M, dtype=np.float64)
    base_mu_r = const_r + A_r @ s_t
    sigma2_xr = float(model.Sigma_rr[xr_pos, xr_pos])
    sigma2_xb = float(model.Sigma_rr[xb_pos, xb_pos])
    sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])
    gamma = float(getattr(model, "gamma", 5.0))

    def integrand(joint_z, alpha_s, alpha_b, s_val=10.0, w_floor=1e-3):
        v_nodes = joint_z[:, :model.n_state] @ L_s.T
        r_nodes = joint_z[:, model.n_state:] @ L_r.T
        s_next = Phi_0 + Phi_11 @ s_t + v_nodes
        log_R_bill = s_next[:, rtb_idx]
        Mv = v_nodes @ M.T
        log_x_s = base_mu_r[xr_pos] + Mv[:, xr_pos] + r_nodes[:, xr_pos]
        log_x_b = base_mu_r[xb_pos] + Mv[:, xb_pos] + r_nodes[:, xb_pos]
        r_p = (log_R_bill + alpha_s * log_x_s + alpha_b * log_x_b
               + 0.5 * (alpha_s - alpha_s ** 2) * sigma2_xr
               + 0.5 * (alpha_b - alpha_b ** 2) * sigma2_xb
               - alpha_s * alpha_b * sigma_xrxb)
        R_p = np.exp(r_p)
        W = np.maximum(s_val * R_p, w_floor)
        if v_next_kind == "smooth":
            V = np.exp(-0.3 * s_next[:, 0] - 0.1 * s_next[:, 3]) * 0.1
        elif v_next_kind == "curved":
            # Curvature centred at the global state-grid centroid (mu_s).
            # Width ~ 1.5*sigma_z so V varies meaningfully across the grid;
            # this introduces s_t-dependent shape in z.
            mu = np.array([-2.99, 0.02, 0.009, 0.048])    # approximate state_grid_mu_s
            w  = 1.5 * np.array([0.531, 0.0162, 0.0289, 0.0358])
            r2 = ((s_next - mu) / w) ** 2
            V = 0.1 * np.exp(-0.5 * r2.sum(axis=1))
        elif v_next_kind == "kinked":
            # Kink near the work-retire boundary in dp: V is V0 * f(dp) where
            # f has a kink at dp = -3 mimicking the constraint switch.
            kink_x = -3.0
            kink_strength = 5.0   # large -> sharper kink, more curvature
            V = 0.1 * np.exp(-0.05 * s_next[:, 3]) * (
                1.0 + 0.5 * np.tanh(kink_strength * (s_next[:, 0] - kink_x))
            )
        else:
            raise ValueError(v_next_kind)
        u_prime = W ** (-gamma)
        return u_prime * V * R_p

    return integrand


# ---------------------------------------------------------------------------
# Cell selection: pick representative stress cells
# ---------------------------------------------------------------------------

def pick_stress_cells(pc):
    """Return list of (label, i_s, s_t) tuples covering body, tails, corners.

    State ordering: (dp, spr, rtb, y_1).
    """
    g = np.asarray(pc.state_grid)
    N_state = g.shape[0]
    sigma_z = np.asarray(pc.state_grid_sigma_z)
    mu_s = np.asarray(pc.state_grid_mu_s)
    # In bracket coords b = L_inv @ (s - shift), each axis has the same per-axis stds
    # as the bracket grid (dp: ±2.0, ...). Find cells in the original s-coord by
    # measuring (s - mu_s) / sigma_z which captures axis-aligned z-distance.
    # That doesn't fully capture dp/y_1 cross-correlation, but is good enough for picking.
    z_dist = (g - mu_s[None, :]) / sigma_z[None, :]
    cells = []
    cells.append(("body centroid",          int(N_state // 2)))
    cells.append(("deep -dp tail #1",       int(np.argmin(g[:, 0]))))
    cells.append(("deep +dp tail #2",       int(np.argmax(g[:, 0]))))
    cells.append(("deep -y_1 tail",         int(np.argmin(g[:, 3]))))
    cells.append(("deep +y_1 tail",         int(np.argmax(g[:, 3]))))
    cells.append(("body off-centroid",      int(np.argmin(np.abs(z_dist[:, 0] - 0.0) +
                                                          np.abs(z_dist[:, 1] - 0.5) +
                                                          np.abs(z_dist[:, 2] - 0.0) +
                                                          np.abs(z_dist[:, 3] - 0.5)))))
    # Worst-case |state coords| corner: maximise sum of |z_dist| across axes
    corner_score = np.abs(z_dist).sum(axis=1)
    cells.append(("worst corner |z|",       int(np.argmax(corner_score))))
    cells.append(("bond stress: low spr × high y_1",
                  int(np.argmin(z_dist[:, 1]) // 7 * 7 +  # not quite right but ok
                      0)))  # let's do proper
    # bond stress: low spr × high y_1
    score = z_dist[:, 1] - z_dist[:, 3]
    cells[-1] = ("bond stress -spr +y_1", int(np.argmin(score)))
    out = []
    for label, i_s in cells:
        out.append((label, i_s, np.asarray(g[i_s], dtype=np.float64)))
    return out


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

def make_rule_set():
    """All rules under review. Returns dict: label -> (joint_z, weights)."""
    rules = {}
    # Baselines
    rules["GH (3,3,3,3,4,4) curr canonical"] = tensor_gh_rule((3, 3, 3, 3, 4, 4))
    # Recommended
    rules["Lob (5,3,3,5,3,5) dp=2.93,y_1=2.93,xb=2.86 RECOMMENDED"] = tensor_mixed_rule(
        (5, 3, 3, 5, 3, 5), (2.93, None, None, 2.93, None, 2.86)
    )
    # Equal-K-cost GH (sanity vs Lobatto) — shows what Lobatto buys above pure GH refinement
    rules["GH (5,3,3,5,3,5)"] = tensor_gh_rule((5, 3, 3, 5, 3, 5))
    # Brute force GH 5^6
    rules["GH (5,5,5,5,5,5)"] = tensor_gh_rule((5, 5, 5, 5, 5, 5))
    # Cheap-Lobatto-without-K-bump (Task 6)
    rules["Lob (3,3,3,3,3,5) Z=2.93 on dp,spr,y_1 + xb=2.86 CHEAP"] = tensor_mixed_rule(
        (3, 3, 3, 3, 3, 5), (2.93, 2.93, None, 2.93, None, 2.86)
    )
    # Even cheaper: just K=(3,3,3,3) state with Lobatto on dp + y_1, K=(3,5) ret
    rules["Lob (3,3,3,3,3,5) Z=2.93 dp,y_1 + xb=2.86"] = tensor_mixed_rule(
        (3, 3, 3, 3, 3, 5), (2.93, None, None, 2.93, None, 2.86)
    )
    # Smaller still: keep K=(3,3,3,3,4,4) but Lobatto on dp and y_1 only
    rules["Lob (3,3,3,3,4,4) Z=2.93 dp,y_1"] = tensor_mixed_rule(
        (3, 3, 3, 3, 4, 4), (2.93, None, None, 2.93, None, None)
    )
    # Lobatto on dp only at K=3
    rules["Lob (3,3,3,3,4,4) Z=2.93 dp only"] = tensor_mixed_rule(
        (3, 3, 3, 3, 4, 4), (2.93, None, None, None, None, None)
    )
    # Per-axis bumps (Task 5)
    for d, name in enumerate(["dp", "spr", "rtb", "y_1"]):
        K = [3, 3, 3, 3, 4, 4]
        K[d] = 5
        rules[f"GH bump {name} only -> {tuple(K)}"] = tensor_gh_rule(tuple(K))
    rules["GH bump xr only -> (3,3,3,3,5,4)"] = tensor_gh_rule((3, 3, 3, 3, 5, 4))
    rules["GH bump xb only -> (3,3,3,3,4,5)"] = tensor_gh_rule((3, 3, 3, 3, 4, 5))
    return rules


# ---------------------------------------------------------------------------
# Truth comparators
# ---------------------------------------------------------------------------

ALPHAS = [
    ("at zero",      0.0,  0.0),
    ("body",         0.5,  0.5),
    ("stress mid",   1.5,  1.0),
    ("stress high",  3.0,  2.0),
    ("near cap",     5.0, -3.0),
    ("near cap +/+", 6.0,  6.0),
]


def truth_value(integrand, K=7):
    z, w = tensor_gh_rule((K,) * 6)
    out = {}
    for label_a, a_s, a_b in ALPHAS:
        out[(label_a, a_s, a_b)] = float(w @ integrand(z, a_s, a_b))
    return out


def evaluate_rules(integrand, rules, truth):
    rows = []
    for label, (z, w) in rules.items():
        for (label_a, a_s, a_b), tval in truth.items():
            val = float(w @ integrand(z, a_s, a_b))
            relerr = abs(val - tval) / max(abs(tval), 1e-30)
            rows.append((label, label_a, a_s, a_b, len(w), val, tval, relerr))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_pct(x):
    if x < 1e-12:
        return "<1e-12"
    if x < 1e-6:
        return f"{x:.1e}"
    if x < 1e-2:
        return f"{x:.1e}"
    return f"{x*100:.1f}%"


def main():
    print("Building model...")
    model, pc, disc = build()
    print(f"  N_state={pc.N_state}  state_grid_sizes={pc.state_grid_sizes}")
    print(f"  state_n_stds={disc.state_n_stds}  ret_names={model.ret_names}")
    g = np.asarray(pc.state_grid)

    # ----------------------------------------------------------------------
    # Pick cells
    # ----------------------------------------------------------------------
    cells = pick_stress_cells(pc)
    sigma_z = np.asarray(pc.state_grid_sigma_z)
    mu_s = np.asarray(pc.state_grid_mu_s)
    print()
    print("Stress cells:")
    print(f"  {'label':<28}  {'i_s':>5}  {'s_t (dp,spr,rtb,y_1)':<48}  z_dist")
    for label, i_s, s_t in cells:
        zd = (s_t - mu_s) / sigma_z
        print(f"  {label:<28}  {i_s:>5}  ({s_t[0]:+.3f}, {s_t[1]:+.3f}, {s_t[2]:+.3f}, {s_t[3]:+.3f})  ({zd[0]:+.2f}, {zd[1]:+.2f}, {zd[2]:+.2f}, {zd[3]:+.2f})")

    # ----------------------------------------------------------------------
    # Rules
    # ----------------------------------------------------------------------
    rules = make_rule_set()
    print()
    print(f"Rules: {len(rules)} variants")
    for label, (_z, w) in rules.items():
        print(f"  {label:<60} N={len(w):>5}")

    # ----------------------------------------------------------------------
    # IMPORTANT methodology note:
    # The original investigation's V_next is V = 0.1 * exp(-0.3*dp_next - 0.1*y_1_next).
    # That's *additively-separable in s_t and v* and *exponential* in s_next, so the
    # ratio (val/truth) is a multiplicative constant in s_t and the relerr is
    # *identical* across cells. To produce a meaningful multi-cell test, we run
    # the same comparison under three V_next variants, including ones whose curvature
    # in z depends on s_t.
    # ----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("TASK 1 + 5 + 6: Multi-cell FOC integrand stress (truth = GH 7^6 = 117,649 nodes)")
    print("=" * 100)
    print("  v_next_kind='smooth' is the original handoff-test V; relerr is s_t-INVARIANT.")
    print("  Adding 'curved' (Gaussian-shaped) and 'kinked' (constraint-mimicking) variants.")
    cell_results = {}  # (kind, cell_label) -> list of rows
    for kind in ("smooth", "curved", "kinked"):
        for cell_label, i_s, s_t in cells:
            integrand = make_integrand(model, s_t, v_next_kind=kind)
            truth = truth_value(integrand, K=7)
            rows = evaluate_rules(integrand, rules, truth)
            cell_results[(kind, cell_label)] = rows

    # Compact table per alpha across cells, with the recommended rule highlighted.
    # Worst-case relerr per (kind, rule) across cells x alphas.
    worst_per_rule = {}
    for (kind, cell_label), rows in cell_results.items():
        for r in rows:
            label, label_a, a_s, a_b, n, val, tval, relerr = r
            worst_per_rule.setdefault((kind, label), {}).setdefault(cell_label, {})[(label_a, a_s, a_b)] = relerr

    alpha_keys = [("at zero", 0.0, 0.0), ("body", 0.5, 0.5),
                  ("stress mid", 1.5, 1.0), ("stress high", 3.0, 2.0),
                  ("near cap", 5.0, -3.0), ("near cap +/+", 6.0, 6.0)]
    # Print compact relerr table per (kind, cell)
    for kind in ("smooth", "curved", "kinked"):
        for cell_label in [c[0] for c in cells]:
            print()
            print(f"--- V={kind}  Cell: {cell_label} ---")
            print(f"  {'rule':<60}  N      a=0       body       (1.5,1)   (3,2)     (5,-3)    (6,6)")
            for label, (_z, w) in rules.items():
                entries = worst_per_rule[(kind, label)][cell_label]
                cells_str = "  ".join(f"{fmt_pct(entries[k]):>8}" for k in alpha_keys)
                print(f"  {label:<60}  {len(w):>5}  {cells_str}")

    # ----------------------------------------------------------------------
    # Task 1 worst-cell summary
    # ----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("TASK 1 SUMMARY: worst-cell relerr per (V_kind, rule), across all (cell x alpha != 0)")
    print("=" * 100)
    for kind in ("smooth", "curved", "kinked"):
        print()
        print(f"-- V_kind = {kind} --")
        print(f"  {'rule':<60}  N      worst_relerr  worst_cell                worst_alpha")
        for label, (_z, w) in rules.items():
            worst = (-1, None, None)
            for cell_label, alpha_dict in worst_per_rule[(kind, label)].items():
                for k, relerr in alpha_dict.items():
                    if k[0] == "at zero":
                        continue
                    if relerr > worst[0]:
                        worst = (relerr, cell_label, k)
            print(f"  {label:<60}  {len(w):>5}  {fmt_pct(worst[0]):>10}    {worst[1]:<24}  {worst[2]}")

    # ----------------------------------------------------------------------
    # Task 3: truth bias check at K=9 vs K=7 truth
    # ----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("TASK 3: Truth bias check (K=9 vs K=7 GH tensor 'truth') under each V_next variant")
    print("=" * 100)
    K9_cells = [cells[0], cells[6]]   # body centroid and worst corner
    print(f"  Building K=9 truth (9^6 = 531,441 nodes). May take a few seconds...")
    z9, w9 = tensor_gh_rule((9, 9, 9, 9, 9, 9))
    z7, w7 = tensor_gh_rule((7, 7, 7, 7, 7, 7))
    print(f"  Loaded K=9 ({len(w9)} nodes) and K=7 ({len(w7)} nodes)")
    print()
    print(f"  {'V_kind':<8}  {'cell':<24}  alpha             K=7 truth        K=9 truth        |dT|/|K9|")
    for kind in ("smooth", "curved", "kinked"):
        for cell_label, i_s, s_t in K9_cells:
            integrand = make_integrand(model, s_t, v_next_kind=kind)
            for label_a, a_s, a_b in ALPHAS:
                v7 = float(w7 @ integrand(z7, a_s, a_b))
                v9 = float(w9 @ integrand(z9, a_s, a_b))
                d = abs(v7 - v9) / max(abs(v9), 1e-30)
                print(f"  {kind:<8}  {cell_label:<24}  {label_a:<14}  {v7:+.6e}   {v9:+.6e}   {fmt_pct(d):>10}")

    # ----------------------------------------------------------------------
    # Task 2: alpha* root benchmark
    # FOC condition for CCV log dynamics: the optimal (alpha_s*, alpha_b*) solves
    #   E[u'(W) * V * R_p * log_x_s] = 0   and same for log_x_b
    # We use a fixed-point on a simplified FOC:
    #   F_s(alpha) = sum_n w_n * u'(W_n) * V_n * R_p_n * log_x_s_n
    #   F_b(alpha) = sum_n w_n * u'(W_n) * V_n * R_p_n * log_x_b_n
    # with the "truth" being K=9 GH. Compute alpha* under each rule and report
    # |Delta alpha*| vs truth.
    # ----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("TASK 2: alpha* root benchmark (per cell x rule, truth = K=9 GH)")
    print("=" * 100)
    from scipy.optimize import root

    def make_foc(model, s_t, v_next_kind):
        L_s = np.linalg.cholesky(0.5 * (model.Sigma_ss + model.Sigma_ss.T))
        L_r = np.linalg.cholesky(0.5 * (model.Sigma_r_cond + model.Sigma_r_cond.T))
        Phi_0 = np.asarray(model.Phi_0_state, dtype=np.float64)
        Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
        rtb_idx = int(model.rtb_index_in_state)
        xr_pos = list(model.ret_names).index("xr")
        xb_pos = list(model.ret_names).index("xb")
        A_r = np.asarray(model.Phi_21, dtype=np.float64)
        const_r = np.asarray(model.Phi_0_ret, dtype=np.float64)
        M = np.asarray(model.M, dtype=np.float64)
        base_mu_r = const_r + A_r @ s_t
        sigma2_xr = float(model.Sigma_rr[xr_pos, xr_pos])
        sigma2_xb = float(model.Sigma_rr[xb_pos, xb_pos])
        sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])
        gamma = float(getattr(model, "gamma", 5.0))

        def precompute(joint_z):
            v_nodes = joint_z[:, :model.n_state] @ L_s.T
            r_nodes = joint_z[:, model.n_state:] @ L_r.T
            s_next = Phi_0 + Phi_11 @ s_t + v_nodes
            log_R_bill = s_next[:, rtb_idx]
            Mv = v_nodes @ M.T
            log_x_s = base_mu_r[xr_pos] + Mv[:, xr_pos] + r_nodes[:, xr_pos]
            log_x_b = base_mu_r[xb_pos] + Mv[:, xb_pos] + r_nodes[:, xb_pos]
            if v_next_kind == "smooth":
                V = np.exp(-0.3 * s_next[:, 0] - 0.1 * s_next[:, 3]) * 0.1
            elif v_next_kind == "curved":
                mu = np.array([-2.99, 0.02, 0.009, 0.048])
                wd = 1.5 * np.array([0.531, 0.0162, 0.0289, 0.0358])
                r2 = ((s_next - mu) / wd) ** 2
                V = 0.1 * np.exp(-0.5 * r2.sum(axis=1))
            elif v_next_kind == "kinked":
                V = 0.1 * np.exp(-0.05 * s_next[:, 3]) * (
                    1.0 + 0.5 * np.tanh(5.0 * (s_next[:, 0] - (-3.0)))
                )
            else:
                raise ValueError(v_next_kind)
            return log_R_bill, log_x_s, log_x_b, V

        def foc(alpha, weights, log_R_bill, log_x_s, log_x_b, V, s_val=10.0, w_floor=1e-3):
            a_s, a_b = alpha
            r_p = (log_R_bill + a_s * log_x_s + a_b * log_x_b
                   + 0.5 * (a_s - a_s ** 2) * sigma2_xr
                   + 0.5 * (a_b - a_b ** 2) * sigma2_xb
                   - a_s * a_b * sigma_xrxb)
            R_p = np.exp(r_p)
            W = np.maximum(s_val * R_p, w_floor)
            u_prime = W ** (-gamma)
            kernel = weights * u_prime * V * R_p
            return np.array([(kernel * log_x_s).sum(),
                             (kernel * log_x_b).sum()])

        return precompute, foc

    print(f"  {'V_kind':<8}  {'cell':<24}  {'rule':<55}  alpha_truth         alpha_rule        |Delta|max")
    foc_cells = cells   # all 8 stress cells
    foc_kinds = ("smooth", "curved", "kinked")
    foc_alpha_max = {}    # (kind, rule) -> max abs delta
    foc_alpha_mean = {}   # (kind, rule) -> mean abs delta
    for kind in foc_kinds:
        for cell_label, i_s, s_t in foc_cells:
            precomp, foc = make_foc(model, s_t, kind)
            # truth alpha at K=9
            data_truth = precomp(z9)
            sol_truth = root(lambda a: foc(a, w9, *data_truth), x0=np.array([0.5, 0.5]),
                             method='hybr', tol=1e-12)
            if not sol_truth.success:
                # Try different init
                sol_truth = root(lambda a: foc(a, w9, *data_truth), x0=np.array([0.0, 0.0]),
                                 method='hybr', tol=1e-12)
            alpha_truth = sol_truth.x if sol_truth.success else np.array([np.nan, np.nan])
            for label, (z, w) in rules.items():
                data = precomp(z)
                sol = root(lambda a: foc(a, w, *data), x0=alpha_truth if not np.isnan(alpha_truth).any() else np.array([0.5, 0.5]),
                           method='hybr', tol=1e-10)
                if not sol.success:
                    sol = root(lambda a: foc(a, w, *data), x0=np.array([0.0, 0.0]),
                               method='hybr', tol=1e-10)
                alpha_rule = sol.x if sol.success else np.array([np.nan, np.nan])
                if np.isnan(alpha_rule).any() or np.isnan(alpha_truth).any():
                    delta = np.array([np.nan, np.nan])
                else:
                    delta = alpha_rule - alpha_truth
                key = (kind, label)
                if key not in foc_alpha_max:
                    foc_alpha_max[key] = []
                foc_alpha_max[key].append(np.max(np.abs(delta)) if not np.isnan(delta).any() else np.nan)
                # Print all cells (no filter) so we can see where pathological deltas come from
                print(f"  {kind:<8}  {cell_label:<22}  {label:<55}  ({alpha_truth[0]:+.4f},{alpha_truth[1]:+.4f})  ({alpha_rule[0]:+.4f},{alpha_rule[1]:+.4f})  {np.max(np.abs(delta)):.4f}")

    print()
    print("=" * 100)
    print("TASK 2 SUMMARY: max |Delta alpha*| across all 8 cells, per (V_kind, rule)")
    print("=" * 100)
    for kind in foc_kinds:
        print()
        print(f"-- V_kind = {kind} --")
        print(f"  {'rule':<60}  N      max|Delta a*|  mean|Delta a*|  PASS?")
        for label, (_z, w) in rules.items():
            arr = np.array(foc_alpha_max[(kind, label)])
            arr_clean = arr[~np.isnan(arr)]
            if len(arr_clean) == 0:
                print(f"  {label:<60}  {len(w):>5}  N/A (all NaN)")
                continue
            mx = float(np.nanmax(arr))
            mn = float(np.nanmean(arr))
            verdict = "PASS" if (mx < 0.05 and mn < 0.01) else "FAIL"
            print(f"  {label:<60}  {len(w):>5}    {mx:>9.4f}    {mn:>9.4f}    {verdict}")


if __name__ == "__main__":
    main()
